"""Standalone VBench evaluator for accuracy-mode harness runs.

Reads the artefacts an accuracy-mode run already produces (the ``artefacts/``
sub-directory with one ``{sample_index}.mp4`` per sample plus a
``prompts.json`` mapping filename -> prompt), stages those videos under
the ``{prompt}-{index}.mp4`` filename layout VBench's *vbench_standard*
mode requires, launches the upstream VBench evaluator under
``torch.distributed.run``, and emits:

1. ``<run_dir>/accuracy.txt`` — the MLPerf submission artefact, formatted
   to match the v6.0 NVIDIA / Cisco / GigaComputing submissions so it both
   passes the upstream submission checker
   (``tools/submission/submission_checker/constants.py:1427``,
   ``checks/accuracy_check.py:88-167``) and stays human-readable. Contains
   the canonical ``'vbench_score': XX.XXXX`` line on the 0-100 scale (target
   ``>= 69.7752``) plus a ``hash=<sha256>`` line over
   ``mlperf_log_accuracy.json``.
2. ``<run_dir>/vbench/vbench_summary.json`` — a structured sidecar for
   tooling/CI, with per-dimension means, paths, the sha256, and
   reproducibility metadata.
3. ``<run_dir>/vbench/`` — VBench's own ``results_*_eval_results.json``
   files plus the subprocess stdout/stderr, kept verbatim so ``--parse-only``
   can re-derive the summary cheaply.

The staging step exists because VBench's *custom_input* mode (filename ->
prompt sidecar JSON) refuses six of its dimensions outright -- including
two of the MLPerf six (``scene``, ``appearance_style``); see
``vbench/__init__.py:check_dimension_requires_extra_info``. The reference
dimensions need the structured ``auxiliary_info`` metadata bundled in
VBench's built-in ``VBench_full_info.json``, which is only consulted in
*vbench_standard* mode. That mode iterates its built-in 946-prompt set
and, for each prompt, looks up to 5 videos named ``{prompt}-0.mp4``
through ``{prompt}-4.mp4`` in ``--videos_path``
(``VBench.build_full_info_json``). We therefore stage each artefact
under ``<prompt>-<iteration>.mp4`` per the official MLPerf reference's
``run_inference.py`` layout, with ``iteration`` counted *per prompt*
(0, 1, ... as the same prompt reappears in ``prompts.json``) so the
suffix is always in ``[0, 4]`` regardless of the harness's global
on-disk index. Keeping the harness's on-disk artefacts numeric
(``<index>.mp4``) is still the right design: it sidesteps the
filesystem-safety pitfalls of long / slash-bearing prompts, and only
the evaluator -- which knows the active prompt set is safe -- pays the
cost of the rename.

The module deliberately avoids importing torch / transformers / VBench at
module-load time so ``wan-harness print-config`` keeps working when only the
inference venv is available.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

_log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DIMENSIONS",
    "REFERENCE_ACCURACY",
    "ACCURACY_THRESHOLD_99",
    "MAX_ACCURACY_LOG_SIZE",
    "DimensionScore",
    "VBenchResult",
    "discover_inputs",
    "stage_videos_for_vbench",
    "build_command",
    "parse_results",
    "sha256_of",
    "render_accuracy_txt",
    "write_accuracy_txt",
    "write_summary",
    "run_evaluation",
]


# ----------------------------------------------------------------------
# Constants. Numbers are baked into the MLPerf submission checker; do
# not change without bumping the benchmark version.
# ----------------------------------------------------------------------


DEFAULT_DIMENSIONS: tuple[str, ...] = (
    "subject_consistency",
    "dynamic_degree",
    "motion_smoothness",
    "appearance_style",
    "scene",
    "background_consistency",
)
"""The 6 VBench dimensions averaged for the MLPerf wan-2.2-t2v-a14b score.

Matches ``run_evaluation.py`` in ``mlcommons/inference/text_to_video/
wan-2.2-t2v-a14b/`` and the wan-2.2 task force decision documented in
https://mlcommons.org/2026/03/texttovideo-inference/ .
"""

REFERENCE_ACCURACY: float = 70.48
"""BF16 reference VBench score for wan-2.2-t2v-a14b (0-100 scale)."""

ACCURACY_THRESHOLD_99: float = round(REFERENCE_ACCURACY * 0.99, 4)
"""99% of reference, the official submission threshold (69.7752)."""

MAX_ACCURACY_LOG_SIZE: int = 10 * 1024
"""``mlperf_log_accuracy.json`` ceiling per
``tools/submission/submission_checker/constants.py:MAX_ACCURACY_LOG_SIZE``.
Submitters must run ``truncate_accuracy_log.py`` to fit under this."""


# Regex patterns the upstream submission checker uses against accuracy.txt.
# We assert against these in tests so any format drift in
# render_accuracy_txt() is caught locally.
_VBENCH_SCORE_RE = re.compile(r".*'vbench_score':\s([\d.]+).*")
_HASH_RE = re.compile(r"^hash=([\w\d]+)$")


# ----------------------------------------------------------------------
# Data model.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionScore:
    """Mean and per-video values for one VBench dimension."""

    name: str
    mean: float
    per_video: tuple[float, ...] = field(default=())

    @property
    def n(self) -> int:
        return len(self.per_video)


@dataclass(frozen=True)
class VBenchResult:
    """Parsed VBench evaluation output, suitable for both submission
    formatting (``write_accuracy_txt``) and tooling (``write_summary``).
    """

    dimensions: tuple[DimensionScore, ...]
    overall_mean: float          # mean over dimension means, on the 0-1 scale.
    vbench_score: float          # overall_mean * 100, the submission metric.
    videos_path: Path
    prompts_path: Path
    results_file: Path
    nproc_per_node: int
    reference: float = REFERENCE_ACCURACY
    threshold_99: float = ACCURACY_THRESHOLD_99

    @property
    def pass_99(self) -> bool:
        return self.vbench_score >= self.threshold_99


# ----------------------------------------------------------------------
# Input discovery.
# ----------------------------------------------------------------------


def discover_inputs(run_dir: Path) -> tuple[Path, Path]:
    """Locate ``artefacts/`` and ``artefacts/prompts.json`` under ``run_dir``.

    ``run_dir`` is normally an accuracy-mode directory like
    ``runs/wan22/<exp>/Offline/accuracy/``. Raises ``FileNotFoundError``
    with a precise message if either artefact is missing, and a
    ``ValueError`` if ``run_dir`` looks like a performance run
    (``mlperf_log_accuracy.json`` is conspicuously small or absent) so the
    user gets pointed at the right directory.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist or is not a directory: {run_dir}")

    artefacts_dir = run_dir / "artefacts"
    prompts_path = artefacts_dir / "prompts.json"
    if not artefacts_dir.is_dir():
        raise FileNotFoundError(
            f"no artefacts/ under {run_dir}. VBench needs the videos written "
            f"by accuracy mode; point --videos-dir explicitly if this run "
            f"used a different layout."
        )
    if not prompts_path.is_file():
        raise FileNotFoundError(
            f"no {prompts_path}. The harness writes this in "
            f"ArtefactWriter.finalize(); did this run finish?"
        )

    mp4s = sorted(p for p in artefacts_dir.glob("*.mp4"))
    if not mp4s:
        # Common cause: dry-run / mock backend that wrote .bin frames.
        raise FileNotFoundError(
            f"{artefacts_dir} has no .mp4 files. VBench evaluates videos; "
            f"re-run with a backend that encodes mp4 (the mock backend does "
            f"not by default)."
        )

    return artefacts_dir, prompts_path


# ----------------------------------------------------------------------
# Staging: rename to the {prompt}-{index}.mp4 layout VBench's
# vbench_standard mode parses with utils.get_prompt_from_filename.
# ----------------------------------------------------------------------


def stage_videos_for_vbench(
    artefacts_dir: Path,
    prompts_json: Path,
    staging_dir: Path,
    *,
    use_symlinks: bool = True,
) -> Path:
    """Build a videos-dir suitable for VBench's vbench_standard mode.

    Reads ``prompts_json`` (filename -> prompt) and creates one entry per
    artefact under ``staging_dir`` named ``{prompt}-{iteration}.mp4``,
    where ``iteration`` is the 0-based occurrence count *of that exact
    prompt* in ``prompts_json``. The layout matches the upstream MLPerf
    reference's ``run_inference.py`` (``f"{prompt}-{iteration}.mp4"``)
    and VBench's ``vbench_standard`` mode, which looks up to 5 indexed
    videos per prompt (``-0.mp4`` through ``-4.mp4``) -- see
    ``vbench/__init__.py:VBench.build_full_info_json``. Using the
    harness's global on-disk index here would produce suffixes like
    ``-37`` that VBench never looks for, leaving the dimension's video
    list empty and triggering a divide-by-zero downstream.

    ``use_symlinks=True`` is the default and cheap; falls back to a hard
    copy if symlinks are not supported on the filesystem (e.g. some
    container bind-mounts onto host filesystems with restricted perms).

    Idempotent: stale entries are removed before the new links are
    created.
    """
    artefacts_dir = Path(artefacts_dir)
    prompts_json = Path(prompts_json)
    staging_dir = Path(staging_dir)

    with prompts_json.open("r", encoding="utf-8") as fh:
        prompt_map = json.load(fh)
    if not isinstance(prompt_map, dict) or not prompt_map:
        raise ValueError(
            f"{prompts_json} is empty or not a JSON object. "
            f"ArtefactWriter.finalize() should have populated it."
        )

    staging_dir.mkdir(parents=True, exist_ok=True)

    # Wipe any previously-staged entries so renames after a prompt change
    # don't leave dangling files in the staging dir.
    for stale in staging_dir.glob("*.mp4"):
        try:
            stale.unlink()
        except OSError:
            _log.warning("vbench-stage: could not remove stale %s", stale)

    n_linked = 0
    iteration_by_prompt: dict[str, int] = {}
    for filename, prompt in prompt_map.items():
        src = artefacts_dir / filename
        if not src.is_file():
            _log.warning("vbench-stage: %s missing in %s (skipping)", filename, artefacts_dir)
            continue
        iteration = iteration_by_prompt.get(prompt, 0)
        iteration_by_prompt[prompt] = iteration + 1
        if iteration >= 5:
            # VBench's vbench_standard mode looks at -0.mp4 .. -4.mp4 only;
            # anything beyond that is silently ignored. Warn the operator
            # so they don't wonder why the 6th+ videos didn't influence
            # the score.
            _log.warning(
                "vbench-stage: %s already has 5 iterations staged; "
                "VBench standard mode only looks at iterations 0-4 "
                "(skipping additional copy from %s)",
                prompt, filename,
            )
            continue
        target_name = f"{prompt}-{iteration}.mp4"
        # Defensive: filesystems cap individual filenames at NAME_MAX
        # (typically 255 bytes). Refuse rather than truncate -- the
        # standard 248-prompt MLPerf set fits comfortably (longest
        # observed: 208 bytes including the suffix), so an overflow is
        # most likely a custom prompt set bug we want surfaced.
        if len(target_name.encode("utf-8")) > 255:
            raise ValueError(
                f"vbench-stage: target name {target_name!r} would exceed "
                f"NAME_MAX (255 bytes). Did the prompt set bypass the "
                f"248-prompt MLPerf list?"
            )
        target = staging_dir / target_name
        if use_symlinks:
            try:
                # Absolute path so the symlink survives later moves of staging_dir.
                target.symlink_to(src.resolve())
            except OSError as exc:
                _log.warning("vbench-stage: symlink %s failed (%s); copying", target, exc)
                shutil.copy2(src, target)
        else:
            shutil.copy2(src, target)
        n_linked += 1

    _log.info("vbench-stage: %d videos staged under %s", n_linked, staging_dir)
    return staging_dir


# ----------------------------------------------------------------------
# Command construction. Mirrors the upstream
# text_to_video/wan-2.2-t2v-a14b/run_evaluation.py launcher.
# ----------------------------------------------------------------------


def _resolve_vbench_evaluate_py(vbench_dir: Path | None) -> Path:
    """Find ``VBench/evaluate.py`` so we can pass it to torch.distributed.run."""
    if vbench_dir is None:
        env_dir = os.environ.get("WAN_VBENCH_DIR")
        if env_dir:
            vbench_dir = Path(env_dir)
        else:
            # Fall back to the upstream layout used by mlcommons/inference.
            vbench_dir = Path(__file__).resolve().parents[2] / "submodules" / "VBench"
    evaluate_py = vbench_dir / "evaluate.py"
    if not evaluate_py.is_file():
        raise FileNotFoundError(
            f"VBench/evaluate.py not found at {evaluate_py}. Set --vbench-dir "
            f"or WAN_VBENCH_DIR to the directory containing evaluate.py."
        )
    return evaluate_py


def build_command(
    *,
    videos_path: Path,
    output_dir: Path,
    dimensions: Sequence[str] = DEFAULT_DIMENSIONS,
    nproc_per_node: int = 1,
    vbench_dir: Path | None = None,
    use_with_vbench: bool = True,
    python_executable: str | None = None,
) -> list[str]:
    """Build the argv VBench is launched with.

    ``videos_path`` is the staging dir produced by
    :func:`stage_videos_for_vbench` (files named ``{prompt}-{index}.mp4``).
    ``use_with_vbench`` prefixes the command with ``with-vbench`` (the wrapper
    we install in the Docker image). Outside Docker, callers should pass
    ``use_with_vbench=False`` so the current Python is used directly.

    Flag set matches the upstream MLPerf reference's ``run_evaluation.py``:
    no ``--mode`` (defaults to ``vbench_standard``), no ``--prompt_file``
    (prompts are parsed from filenames via
    ``vbench.utils.get_prompt_from_filename``). The reference dimensions
    -- ``scene``, ``appearance_style``, etc. -- need the structured
    ``auxiliary_info`` from VBench's built-in ``VBench_full_info.json``,
    which is only consulted in vbench_standard mode.

    ``nproc_per_node`` defaults to 1 -- see :func:`run_evaluation` for the
    rationale. The MLPerf submission path is not on the latency-critical
    path, so we trade ~4 minutes of wall-clock for operational simplicity.
    """
    if not dimensions:
        raise ValueError("dimensions must be non-empty")
    if nproc_per_node < 1:
        raise ValueError(f"nproc_per_node must be >= 1, got {nproc_per_node}")

    evaluate_py = _resolve_vbench_evaluate_py(vbench_dir)
    python = python_executable or sys.executable or "python3"

    cmd: list[str] = []
    if use_with_vbench:
        # The Dockerfile installs this wrapper at /usr/local/bin/with-vbench.
        # When it is present, it activates the VBench venv and execs the rest;
        # the venv's python is then the one torch.distributed.run finds.
        cmd.append("with-vbench")
        cmd.append("python")
    else:
        cmd.append(python)

    cmd += [
        "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        str(evaluate_py),
        f"--videos_path={videos_path}",
        f"--output_path={output_dir}",
        "--load_ckpt_from_local=True",
        "--dimension",
        *dimensions,
    ]
    return cmd


# ----------------------------------------------------------------------
# Result parsing.
# ----------------------------------------------------------------------


def _find_latest_results_file(output_dir: Path) -> Path:
    """Pick the newest ``results_*_eval_results.json`` in ``output_dir``.

    VBench emits a fresh timestamped file each run; the most recent one
    is the one we want.
    """
    candidates = sorted(output_dir.glob("results_*_eval_results.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no results_*_eval_results.json under {output_dir}. Either the "
            f"VBench subprocess never ran to completion or --output-dir "
            f"points somewhere unexpected."
        )
    # Sorted lexicographically, the YYYY-MM-DD-HH:MM:SS prefix gives us
    # chronological order; no need to stat each one.
    return candidates[-1]


def parse_results(
    output_dir: Path,
    *,
    videos_path: Path,
    prompts_path: Path,
    nproc_per_node: int,
) -> VBenchResult:
    """Parse VBench's results JSON and assemble a :class:`VBenchResult`.

    The on-disk shape is ``{<dimension>: [<avg_score>, [<per_video_dict>, ...]]}``,
    matching upstream ``run_evaluation.py:parse_results``.
    """
    results_file = _find_latest_results_file(Path(output_dir))
    with results_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            f"{results_file} is empty or not a JSON object; VBench likely "
            f"failed before emitting scores. Inspect the subprocess logs."
        )

    dims: list[DimensionScore] = []
    means: list[float] = []
    for name in sorted(payload):
        entry = payload[name]
        # Each dimension stores [mean_score, [per_video_records...]].
        if not isinstance(entry, list) or not entry:
            _log.warning("%s: skipping dimension %s (unexpected shape)", results_file, name)
            continue
        mean = entry[0]
        if not isinstance(mean, (int, float)):
            _log.warning("%s: skipping dimension %s (non-numeric mean)", results_file, name)
            continue
        per_video: tuple[float, ...] = ()
        if len(entry) > 1 and isinstance(entry[1], list):
            extracted: list[float] = []
            for record in entry[1]:
                if isinstance(record, dict):
                    # VBench's record dicts vary by dimension; pull whichever
                    # numeric field is present.
                    for key in ("video_results", "score", name):
                        if key in record and isinstance(record[key], (int, float)):
                            extracted.append(float(record[key]))
                            break
                elif isinstance(record, (int, float)):
                    extracted.append(float(record))
            per_video = tuple(extracted)
        dims.append(DimensionScore(name=name, mean=float(mean), per_video=per_video))
        means.append(float(mean))

    if not means:
        raise ValueError(f"{results_file} produced no numeric dimension means.")

    overall_mean = sum(means) / len(means)
    return VBenchResult(
        dimensions=tuple(dims),
        overall_mean=overall_mean,
        vbench_score=round(overall_mean * 100, 4),
        videos_path=Path(videos_path),
        prompts_path=Path(prompts_path),
        results_file=results_file,
        nproc_per_node=nproc_per_node,
    )


# ----------------------------------------------------------------------
# Hashing. Byte-equivalent to
# tools/submission/truncate_accuracy_log.py:get_hash() so submitters can
# re-verify with the upstream tool.
# ----------------------------------------------------------------------


def sha256_of(path: Path, *, chunk_size: int = 4096) -> str:
    """Stream the contents of ``path`` through sha256 in 4 KiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Output rendering.
# ----------------------------------------------------------------------


_ACC_TXT_BAR = "=" * 60
_ACC_TXT_THIN = "-" * 60


def render_accuracy_txt(result: VBenchResult, *, acc_json_sha256: str) -> str:
    """Format the diagnostic ``accuracy.txt`` body.

    Shape mirrors v6.0 NVIDIA / Cisco submissions, which gives us:

    * the ``'vbench_score': XX.XXXX`` line the submission checker greps for
      (``r".*'vbench_score':\\s([\\d.]+).*"``), with the score on the 0-100
      scale and target ``>= 69.7752``;
    * the ``hash=<sha256>`` line the checker requires
      (``r"^hash=([\\w\\d]+)$"``) over ``mlperf_log_accuracy.json``;
    * per-dimension scores and an overall average, for humans.
    """
    lines: list[str] = []
    lines.append(_ACC_TXT_BAR)
    lines.append("VBench Evaluation Results")
    lines.append(_ACC_TXT_BAR)
    lines.append("")
    lines.append("Dimension Scores:")
    lines.append(_ACC_TXT_THIN)
    for dim in sorted(result.dimensions, key=lambda d: d.name):
        lines.append(f"  {dim.name:30s}: {dim.mean:6.4f}")
    # The canonical score line. Submission checker regex sources from this.
    lines.append(f"'vbench_score': {result.vbench_score:.4f}")
    lines.append(_ACC_TXT_THIN)
    pct = result.overall_mean * 100
    lines.append(f"  {'Overall Average':30s}: {result.overall_mean:6.4f} ({pct:.2f}%)")
    lines.append("")
    lines.append(f"Threshold: {result.threshold_99:.4f}%")
    lines.append(f"Pass: {'Yes' if result.pass_99 else 'No'}")
    lines.append(_ACC_TXT_BAR)
    lines.append(f"Detailed results: {result.results_file}")
    lines.append(_ACC_TXT_BAR)
    lines.append("")
    lines.append(f"hash={acc_json_sha256}")
    return "\n".join(lines) + "\n"


def write_accuracy_txt(
    result: VBenchResult,
    dest: Path,
    *,
    acc_json_path: Path,
) -> Path:
    """Write the submission ``accuracy.txt`` to ``dest``.

    The hash is computed over ``acc_json_path`` as it is on disk *now*;
    submitters typically run ``truncate_accuracy_log.py`` first so this
    hash matches the truncated file that ships in the submission.

    A warning (not an error) is logged when ``acc_json_path`` is over the
    ``MAX_ACCURACY_LOG_SIZE`` limit, so cases where the truncation step
    was forgotten are noisy but not fatal.
    """
    dest = Path(dest)
    acc_json_path = Path(acc_json_path)
    if not acc_json_path.is_file():
        raise FileNotFoundError(
            f"{acc_json_path} not found; cannot compute hash for accuracy.txt. "
            f"Pass --accuracy-json to point at the right file."
        )
    size = acc_json_path.stat().st_size
    if size > MAX_ACCURACY_LOG_SIZE:
        _log.warning(
            "%s is %d bytes (> MAX_ACCURACY_LOG_SIZE=%d); run "
            "tools/submission/truncate_accuracy_log.py before submitting so "
            "the hash in accuracy.txt matches the truncated file.",
            acc_json_path, size, MAX_ACCURACY_LOG_SIZE,
        )
    digest = sha256_of(acc_json_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_accuracy_txt(result, acc_json_sha256=digest), encoding="utf-8")
    _log.info("vbench: wrote %s (vbench_score=%.4f pass=%s)",
              dest, result.vbench_score, result.pass_99)
    return dest


def write_summary(
    result: VBenchResult,
    dest: Path,
    *,
    run_dir: Path,
    accuracy_txt: Path,
    accuracy_json: Path,
    accuracy_json_sha256: str,
    extra: dict | None = None,
) -> Path:
    """Write the structured ``vbench_summary.json`` sidecar.

    ``extra`` is merged into the top-level object so callers can stash
    additional metadata (vbench commit, evaluator version, ...) without
    plumbing more parameters through.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    acc_json = Path(accuracy_json)
    acc_size = acc_json.stat().st_size if acc_json.is_file() else None
    payload: dict = {
        "run_dir": str(run_dir),
        "videos_path": str(result.videos_path),
        "prompts_path": str(result.prompts_path),
        "results_file": str(result.results_file),
        "accuracy_txt": str(accuracy_txt),
        "accuracy_json": str(accuracy_json),
        "accuracy_json_sha256": accuracy_json_sha256,
        "accuracy_json_bytes": acc_size,
        "accuracy_json_truncated": (
            acc_size is not None and acc_size <= MAX_ACCURACY_LOG_SIZE
        ),
        "dimensions": [
            {"name": d.name, "mean": d.mean, "n": d.n}
            for d in sorted(result.dimensions, key=lambda d: d.name)
        ],
        "overall_mean": result.overall_mean,
        "vbench_score": result.vbench_score,
        "reference": result.reference,
        "threshold_99": result.threshold_99,
        "pass_99": result.pass_99,
        "nproc_per_node": result.nproc_per_node,
        "evaluated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    if extra:
        payload.update(extra)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


# ----------------------------------------------------------------------
# Top-level orchestrator.
# ----------------------------------------------------------------------


def _stream_subprocess(cmd: Sequence[str], *, stdout_log: Path, stderr_log: Path) -> int:
    """Run ``cmd`` while teeing stdout/stderr through ``logging`` and to disk.

    VBench is chatty; we want both the log file (for after-the-fact
    debugging) and live logging output (so a user watching the run can
    see progress).
    """
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("w", encoding="utf-8") as out_fh, \
         stderr_log.open("w", encoding="utf-8") as err_fh:
        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Drain both pipes in parallel via select() to avoid deadlocks on
        # processes that fill stderr faster than stdout.
        import select
        assert proc.stdout is not None and proc.stderr is not None
        streams = {
            proc.stdout.fileno(): (proc.stdout, out_fh, _log.info),
            proc.stderr.fileno(): (proc.stderr, err_fh, _log.warning),
        }
        while streams:
            ready, _, _ = select.select(list(streams), [], [], 0.5)
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            for fd in ready:
                stream, sink, log_fn = streams[fd]
                line = stream.readline()
                if not line:
                    del streams[fd]
                    continue
                sink.write(line)
                sink.flush()
                log_fn("vbench: %s", line.rstrip())
        proc.wait()
        return proc.returncode


def run_evaluation(
    run_dir: Path,
    *,
    videos_dir: Path | None = None,
    prompts_json: Path | None = None,
    output_dir: Path | None = None,
    accuracy_txt: Path | None = None,
    accuracy_json: Path | None = None,
    dimensions: Sequence[str] = DEFAULT_DIMENSIONS,
    nproc_per_node: int = 1,
    vbench_dir: Path | None = None,
    use_with_vbench: bool = True,
    python_executable: str | None = None,
    parse_only: Path | None = None,
    dry_run: bool = False,
) -> VBenchResult:
    """Orchestrate one VBench evaluation against an accuracy-mode run.

    All ``Path`` arguments are resolved to absolute paths so the emitted
    summary / accuracy.txt are reproducible regardless of cwd.

    ``nproc_per_node`` defaults to 1. Upstream's ``run_evaluation.py``
    uses 8 to match a typical 8-GPU node, but submission scoring is not
    on a latency-critical path -- the 248-prompt MLPerf set finishes in
    ~4 min single-rank on a single MI355X -- and single-rank avoids two
    real failure modes we hit in 8-rank mode:

    * the upstream ``dynamic_degree`` distributed bug
      (Vchitect/VBench#141), and
    * a one-time DINO checkpoint-download race in the per-dimension
      preludes, which can leave the cache half-populated if a rank loses.

    Override with ``nproc_per_node>1`` once the checkpoint cache is warm
    and the VBench venv's ROCm runtime matches the host's.
    """
    run_dir = Path(run_dir).resolve()

    if videos_dir is None or prompts_json is None:
        discovered_videos, discovered_prompts = discover_inputs(run_dir)
        videos_dir = videos_dir or discovered_videos
        prompts_json = prompts_json or discovered_prompts

    videos_dir = Path(videos_dir).resolve()
    prompts_json = Path(prompts_json).resolve()

    output_dir = Path(output_dir).resolve() if output_dir else (run_dir / "vbench").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_txt = Path(accuracy_txt).resolve() if accuracy_txt else (run_dir / "accuracy.txt").resolve()
    accuracy_json = Path(accuracy_json).resolve() if accuracy_json else (run_dir / "mlperf_log_accuracy.json").resolve()

    # vbench_standard mode parses prompts from filenames -- stage the
    # numeric-index artefacts under {prompt}-{index}.mp4 so the upstream
    # filename parser does the right thing. We point --videos_path at
    # the staging dir, not at the original artefacts/.
    staging_dir = output_dir / "videos_staged"
    if parse_only is None:
        stage_videos_for_vbench(videos_dir, prompts_json, staging_dir)

    cmd = build_command(
        videos_path=staging_dir,
        output_dir=output_dir,
        dimensions=dimensions,
        nproc_per_node=nproc_per_node,
        vbench_dir=vbench_dir,
        use_with_vbench=use_with_vbench and shutil.which("with-vbench") is not None,
        python_executable=python_executable,
    )

    if dry_run:
        _log.info("vbench (dry-run): %s", " ".join(cmd))
        # Construct a placeholder result so downstream callers can format
        # something sensible. We return the empty result rather than
        # raising so --dry-run is composable with --parse-only.
        if parse_only is None:
            raise SystemExit(0)

    if parse_only is None and not dry_run:
        _log.info("vbench: running %s", " ".join(cmd))
        rc = _stream_subprocess(
            cmd,
            stdout_log=output_dir / "stdout.log",
            stderr_log=output_dir / "stderr.log",
        )
        if rc != 0:
            raise SystemExit(
                f"VBench subprocess exited with status {rc}; see "
                f"{output_dir / 'stderr.log'}"
            )

    parse_target = Path(parse_only).resolve() if parse_only else output_dir
    # Report videos_path as the staging dir (where VBench actually read
    # from) so the summary's paths are reproducible. For --parse-only we
    # still report the staging dir since that's what generated the
    # results being parsed.
    result = parse_results(
        parse_target,
        videos_path=staging_dir,
        prompts_path=prompts_json,
        nproc_per_node=nproc_per_node,
    )

    # accuracy.txt + sha256 first so write_summary can stamp the hash in.
    write_accuracy_txt(result, accuracy_txt, acc_json_path=accuracy_json)
    digest = sha256_of(accuracy_json)
    write_summary(
        result,
        output_dir / "vbench_summary.json",
        run_dir=run_dir,
        accuracy_txt=accuracy_txt,
        accuracy_json=accuracy_json,
        accuracy_json_sha256=digest,
        extra={"command": cmd},
    )

    # Echo the human-readable box to stdout for immediate feedback.
    sys.stdout.write(render_accuracy_txt(result, acc_json_sha256=digest))
    sys.stdout.flush()
    return result
