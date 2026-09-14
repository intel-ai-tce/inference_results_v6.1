"""Build an MLPerf Inference submission tree from a ``run_all`` experiment.

Maps the experiment layout produced by ``scripts/run_all.sh`` into the
directory structure expected by the upstream submission checker
(``performance/run_1``, ``accuracy/``, ``TEST04/performance/run_1``,
``user.conf``, ``measurements.json``, ``README.md`` per scenario).

When a source snapshot is requested (default), ``validate_git_state`` runs
first so a HEAD / dirty-tree mismatch fails fast before copying or
truncating accuracy logs. For each scenario the tool then:

1. Copies LoadGen logs and the submission-checker audit video subset
   (10 fixed sample indices under ``accuracy/videos/``).
2. Truncates ``Accuracy/mlperf_log_accuracy.json`` to the MLPerf 10 KiB
   limit (same algorithm as ``tools/submission/truncate_accuracy_log.py``).
3. Re-emits ``Accuracy/accuracy.txt`` from existing VBench results so the
   ``hash=`` line matches the truncated log on disk.

Additionally installs:

* ``closed/<submitter>/src/<benchmark>/`` — git archive at experiment SHA.
* ``measurements.json`` and per-scenario ``README.md`` under each result dir.

The source experiment directory is never modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tools.fetch_data import DEFAULT_COMMIT as FETCH_DATA_COMMIT  # noqa: E402
from tools.package_code import (  # noqa: E402
    load_experiment_manifest,
    package_code_snapshot,
    validate_git_state,
)
from wan_harness.vbench import (  # noqa: E402
    MAX_ACCURACY_LOG_SIZE,
    parse_results,
    sha256_of,
    write_accuracy_txt,
)

_log = logging.getLogger("prepare_submission")

DEFAULT_BENCHMARK = "wan-2.2-t2v-a14b"
DEFAULT_DIVISION = "closed"
DEFAULT_SYSTEM = "8xMI355X_2xEPYC_9575F"
DEFAULT_SYSTEM_DESC = _REPO_ROOT / "systems" / f"{DEFAULT_SYSTEM}.json"
DEFAULT_MEASUREMENTS = _REPO_ROOT / "configs" / "measurements.json"
DEFAULT_README_TEMPLATE = _REPO_ROOT / "templates" / "submission_scenario_README.md"
SCENARIOS = ("Offline", "SingleStream")
VIEWABLE_SIZE = 4096

# Upstream submission checker audit subset for wan-2.2-t2v-a14b v6.0:
# tools/submission/submission_checker/constants.py REQUIRED_ACC_BENCHMARK
WAN_AUDIT_VIDEO_INDICES: dict[str, tuple[str, ...]] = {
    "v6.0": (
        "130",
        "106",
        "84",
        "59",
        "12",
        "31",
        "86",
        "122",
        "233",
        "96",
    ),
    "v6.1": (
        "130",
        "106",
        "84",
        "59",
        "12",
        "123",
        "43",
        "22",
        "238",
        "32",
    ),
}

PERF_LOGS = ("mlperf_log_summary.txt", "mlperf_log_detail.txt")
ACC_LOGS = PERF_LOGS + ("mlperf_log_accuracy.json",)


@dataclass(frozen=True)
class PrepareConfig:
    experiment_root: Path
    version: str
    output_root: Path
    division: str
    submitter: str
    system: str
    benchmark: str
    user_conf: Path
    system_desc: Path
    measurements_template: Path
    readme_template: Path
    repo_root: Path
    skip_compliance: bool
    skip_vbench_refresh: bool
    skip_code: bool
    skip_measurements: bool
    skip_readme: bool
    dry_run: bool


def truncate_accuracy_log(path: Path) -> tuple[int, int]:
    """Truncate *path* in place using the upstream MLPerf algorithm.

    Returns ``(size_before, size_after)``.
    """
    size_before = path.stat().st_size
    if size_before <= MAX_ACCURACY_LOG_SIZE:
        return size_before, size_before
    if size_before < VIEWABLE_SIZE:
        return size_before, size_before

    with path.open("r", encoding="utf-8", errors="replace") as src:
        start = src.read(VIEWABLE_SIZE)
        src.seek(size_before - VIEWABLE_SIZE, 0)
        end = src.read(VIEWABLE_SIZE)
    with path.open("w", encoding="utf-8") as dst:
        dst.write(start)
        dst.write("\n\n...\n\n")
        dst.write(end)
    size_after = path.stat().st_size
    return size_before, size_after


def install_system_desc(cfg: PrepareConfig) -> dict:
    """Copy the system description JSON into the submission tree."""
    source = cfg.system_desc.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"system description not found: {source}")

    dest_dir = cfg.output_root / cfg.division / cfg.submitter / "systems"
    dest = dest_dir / f"{cfg.system}.json"

    if cfg.dry_run:
        _log.info("would copy %s -> %s", source, dest)
        return {"source": str(source), "destination": str(dest)}

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["submitter"] = cfg.submitter
    payload["division"] = cfg.division

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=4, sort_keys=True) + "\n", encoding="utf-8")
    _log.info("wrote %s", dest)
    return {
        "source": str(source),
        "destination": str(dest),
        "submitter": cfg.submitter,
        "division": cfg.division,
    }


def install_measurements(cfg: PrepareConfig, scenario: str) -> dict:
    """Copy ``measurements.json`` into a scenario results directory."""
    source = cfg.measurements_template.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"measurements template not found: {source}")

    dest = _scenario_root(cfg, scenario) / "measurements.json"
    if cfg.dry_run:
        _log.info("[%s] would copy %s -> %s", scenario, source, dest)
        return {"source": str(source), "destination": str(dest)}

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    _log.info("[%s] wrote %s", scenario, dest)
    return {"source": str(source), "destination": str(dest)}


def render_scenario_readme(
    cfg: PrepareConfig, scenario: str, _manifest: dict
) -> str:
    """Render per-scenario README (provenance is in ``src/REPRODUCIBILITY.json``)."""
    template = cfg.readme_template.read_text(encoding="utf-8")
    mapping = {
        "scenario": scenario,
        "benchmark": cfg.benchmark,
        "system": cfg.system,
        "submitter": cfg.submitter,
        "submitter_lower": cfg.submitter.lower(),
        "division": cfg.division,
    }
    return Template(template).safe_substitute(mapping)


def install_scenario_readme(cfg: PrepareConfig, scenario: str, manifest: dict) -> dict:
    """Write per-scenario ``README.md`` under the results tree."""
    dest = _scenario_root(cfg, scenario) / "README.md"
    if cfg.dry_run:
        _log.info("[%s] would write %s", scenario, dest)
        return {"destination": str(dest)}

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_scenario_readme(cfg, scenario, manifest), encoding="utf-8")
    _log.info("[%s] wrote %s", scenario, dest)
    return {"destination": str(dest)}


def _scenario_root(cfg: PrepareConfig, scenario: str) -> Path:
    return (
        cfg.output_root
        / cfg.division
        / cfg.submitter
        / "results"
        / cfg.system
        / cfg.benchmark
        / scenario
    )


def _require_files(src_dir: Path, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if not (src_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{label}: missing {', '.join(missing)} under {src_dir}"
        )


def _copy_logs(src_dir: Path, dst_dir: Path, names: tuple[str, ...], *, dry_run: bool) -> None:
    if dry_run:
        for name in names:
            _log.info("would copy %s -> %s", src_dir / name, dst_dir / name)
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(src_dir / name, dst_dir / name)


def _load_artefact_prompts(artefacts_dir: Path) -> dict[str, str]:
    prompts_path = artefacts_dir / "prompts.json"
    if not prompts_path.is_file():
        raise FileNotFoundError(
            f"no {prompts_path}; accuracy mode must emit artefacts/prompts.json"
        )
    payload = json.loads(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{prompts_path} must contain a JSON object")
    return {str(key): str(value) for key, value in payload.items()}


def _write_audit_captions(
    dest: Path,
    indices: tuple[str, ...],
    prompts: dict[str, str],
) -> None:
    lines = []
    for idx in indices:
        prompt = prompts.get(f"{idx}.mp4", "")
        lines.append(f"{idx}: {prompt}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def install_audit_videos(
    cfg: PrepareConfig,
    scenario: str,
    *,
    experiment_accuracy_dir: Path,
    submission_accuracy_dir: Path,
) -> dict:
    """Copy the checker-required audit ``.mp4`` subset into ``accuracy/videos/``."""
    artefacts = experiment_accuracy_dir / "artefacts"
    if not artefacts.is_dir():
        raise FileNotFoundError(
            f"{scenario} accuracy: missing {artefacts} "
            f"(run accuracy mode to produce per-sample .mp4 artefacts)"
        )

    videos_dst = submission_accuracy_dir / "videos"

    try:
        indices = WAN_AUDIT_VIDEO_INDICES[cfg.version]
    except KeyError:
        raise ValueError(f"invalid MLPerf inference version: {cfg.version}")

    if cfg.dry_run:
        for idx in indices:
            _log.info(
                "[%s] would copy %s -> %s",
                scenario,
                artefacts / f"{idx}.mp4",
                videos_dst / f"{idx}.mp4",
            )
        _log.info("[%s] would write %s", scenario, videos_dst / "captions.txt")
        return {
            "indices": list(indices),
            "destination": str(videos_dst),
        }

    missing = [
        f"{idx}.mp4"
        for idx in indices
        if not (artefacts / f"{idx}.mp4").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{scenario} accuracy artefacts missing audit videos: "
            f"{', '.join(missing)} under {artefacts}"
        )

    prompts = _load_artefact_prompts(artefacts)
    videos_dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for idx in indices:
        name = f"{idx}.mp4"
        shutil.copy2(artefacts / name, videos_dst / name)
        copied.append(name)

    captions = videos_dst / "captions.txt"
    _write_audit_captions(captions, indices, prompts)
    _log.info("[%s] wrote %d audit videos under %s", scenario, len(copied), videos_dst)
    return {
        "indices": list(indices),
        "destination": str(videos_dst),
        "copied": copied,
        "captions": str(captions),
    }


def refresh_accuracy_txt(
    *,
    experiment_accuracy_dir: Path,
    submission_accuracy_dir: Path,
    dry_run: bool,
) -> None:
    """Re-emit ``accuracy.txt`` from existing VBench results + truncated log."""
    vbench_dir = experiment_accuracy_dir / "vbench"
    if not vbench_dir.is_dir():
        raise FileNotFoundError(
            f"no {vbench_dir}; run VBench first "
            f"(./scripts/run_vbench.sh or run_all without --skip-vbench)"
        )

    acc_json = submission_accuracy_dir / "mlperf_log_accuracy.json"
    acc_txt = submission_accuracy_dir / "accuracy.txt"
    prompts_json = experiment_accuracy_dir / "artefacts" / "prompts.json"
    staging_dir = vbench_dir / "videos_staged"
    videos_path = staging_dir if staging_dir.is_dir() else experiment_accuracy_dir / "artefacts"

    if dry_run:
        _log.info(
            "would refresh %s from %s (hash over %s)",
            acc_txt,
            vbench_dir,
            acc_json,
        )
        return

    result = parse_results(
        vbench_dir,
        videos_path=videos_path,
        prompts_path=prompts_json,
        nproc_per_node=1,
    )
    write_accuracy_txt(result, acc_txt, acc_json_path=acc_json)


def prepare_scenario(cfg: PrepareConfig, scenario: str) -> dict:
    """Copy, truncate, and refresh one scenario. Returns a summary dict."""
    exp = cfg.experiment_root / scenario
    perf_src = exp / "performance" / "run_1"
    acc_src = exp / "accuracy"
    dest = _scenario_root(cfg, scenario)

    _require_files(perf_src, PERF_LOGS, f"{scenario} performance")
    _require_files(acc_src, ACC_LOGS, f"{scenario} accuracy")

    perf_dst = dest / "performance" / "run_1"
    acc_dst = dest / "accuracy"

    _log.info("[%s] performance %s -> %s", scenario, perf_src, perf_dst)
    _copy_logs(perf_src, perf_dst, PERF_LOGS, dry_run=cfg.dry_run)

    _log.info("[%s] accuracy %s -> %s", scenario, acc_src, acc_dst)
    _copy_logs(acc_src, acc_dst, ACC_LOGS, dry_run=cfg.dry_run)

    audit_videos = install_audit_videos(
        cfg,
        scenario,
        experiment_accuracy_dir=acc_src,
        submission_accuracy_dir=acc_dst,
    )

    acc_json = acc_dst / "mlperf_log_accuracy.json"
    size_before = acc_json.stat().st_size if acc_json.is_file() and not cfg.dry_run else None
    size_after = size_before

    if cfg.dry_run:
        _log.info("[%s] would truncate %s if > %d bytes", scenario, acc_json, MAX_ACCURACY_LOG_SIZE)
    else:
        size_before, size_after = truncate_accuracy_log(acc_json)
        if size_before != size_after:
            _log.info(
                "[%s] truncated mlperf_log_accuracy.json %d -> %d bytes",
                scenario,
                size_before,
                size_after,
            )

    if not cfg.skip_vbench_refresh:
        if cfg.dry_run:
            refresh_accuracy_txt(
                experiment_accuracy_dir=acc_src,
                submission_accuracy_dir=acc_dst,
                dry_run=True,
            )
        else:
            # Seed accuracy.txt from the experiment run when present; parse-only
            # overwrites it with a hash matching the truncated submission log.
            exp_acc_txt = acc_src / "accuracy.txt"
            if exp_acc_txt.is_file():
                shutil.copy2(exp_acc_txt, acc_dst / "accuracy.txt")
            refresh_accuracy_txt(
                experiment_accuracy_dir=acc_src,
                submission_accuracy_dir=acc_dst,
                dry_run=False,
            )
            digest = sha256_of(acc_json)
            txt = (acc_dst / "accuracy.txt").read_text(encoding="utf-8")
            if f"hash={digest}" not in txt:
                raise RuntimeError(
                    f"{scenario}: accuracy.txt hash does not match truncated "
                    f"mlperf_log_accuracy.json"
                )

    compliance: dict | None = None
    if not cfg.skip_compliance:
        test04_src = exp / "TEST04"
        verify_src = exp / "compliance" / "TEST04" / "verify_performance.txt"
        _require_files(test04_src, PERF_LOGS, f"{scenario} TEST04")
        if not verify_src.is_file():
            raise FileNotFoundError(f"{scenario} TEST04: missing {verify_src}")

        test04_dst = dest / "TEST04" / "performance" / "run_1"
        verify_dst = dest / "TEST04" / "verify_performance.txt"
        _log.info("[%s] TEST04 %s -> %s", scenario, test04_src, test04_dst)
        _copy_logs(test04_src, test04_dst, PERF_LOGS, dry_run=cfg.dry_run)
        if cfg.dry_run:
            _log.info("[%s] would copy %s -> %s", scenario, verify_src, verify_dst)
        else:
            verify_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(verify_src, verify_dst)
        compliance = {
            "source": str(test04_src),
            "verify_performance_txt": str(verify_dst),
        }

    user_conf_dst = dest / "user.conf"
    if cfg.dry_run:
        _log.info("[%s] would copy %s -> %s", scenario, cfg.user_conf, user_conf_dst)
    else:
        user_conf_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg.user_conf, user_conf_dst)

    return {
        "scenario": scenario,
        "destination": str(dest),
        "performance_source": str(perf_src),
        "accuracy_source": str(acc_src),
        "audit_videos": audit_videos,
        "accuracy_log_bytes_before": size_before,
        "accuracy_log_bytes_after": size_after,
        "accuracy_log_truncated": (
            size_before is not None
            and size_after is not None
            and size_after <= MAX_ACCURACY_LOG_SIZE
        ),
        "compliance": compliance,
        "user_conf": str(user_conf_dst),
    }


def write_manifest(
    cfg: PrepareConfig,
    scenario_summaries: list[dict],
    *,
    system_summary: dict | None = None,
    code_summary: dict | None = None,
    experiment_manifest: dict | None = None,
) -> Path:
    manifest = {
        "prepared_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
        "experiment_root": str(cfg.experiment_root.resolve()),
        "output_root": str(cfg.output_root.resolve()),
        "division": cfg.division,
        "submitter": cfg.submitter,
        "system": cfg.system,
        "benchmark": cfg.benchmark,
        "user_conf_source": str(cfg.user_conf.resolve()),
        "system_desc_source": str(cfg.system_desc.resolve()),
        "system_desc_destination": str(
            cfg.output_root / cfg.division / cfg.submitter / "systems" / f"{cfg.system}.json"
        ),
        "measurements_source": str(cfg.measurements_template.resolve()),
        "readme_template": str(cfg.readme_template.resolve()),
        "skip_compliance": cfg.skip_compliance,
        "skip_vbench_refresh": cfg.skip_vbench_refresh,
        "skip_code": cfg.skip_code,
        "skip_measurements": cfg.skip_measurements,
        "skip_readme": cfg.skip_readme,
        "scenarios": scenario_summaries,
    }
    if system_summary is not None:
        manifest["system_desc"] = system_summary
    if code_summary is not None:
        manifest["code"] = code_summary
    if experiment_manifest is not None:
        manifest["experiment_manifest"] = experiment_manifest
    dest = cfg.output_root / "PREPARE_MANIFEST.json"
    if cfg.dry_run:
        _log.info("would write %s", dest)
        return dest
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log.info("wrote %s", dest)
    return dest


def prepare_submission(cfg: PrepareConfig) -> Path:
    experiment_root = cfg.experiment_root.resolve()
    if not experiment_root.is_dir():
        raise FileNotFoundError(f"experiment root not found: {experiment_root}")
    if not cfg.user_conf.is_file():
        raise FileNotFoundError(f"user.conf not found: {cfg.user_conf}")
    if not cfg.system_desc.is_file():
        raise FileNotFoundError(f"system description not found: {cfg.system_desc}")

    experiment_manifest = load_experiment_manifest(experiment_root)

    results_root = (
        cfg.output_root
        / cfg.division
        / cfg.submitter
        / "results"
        / cfg.system
        / cfg.benchmark
    )
    if results_root.exists() and any(results_root.iterdir()) and not cfg.dry_run:
        raise FileExistsError(
            f"refusing to overwrite non-empty submission tree: {results_root}"
        )

    code_root = cfg.output_root / cfg.division / cfg.submitter / "src" / cfg.benchmark
    if (
        not cfg.skip_code
        and code_root.exists()
        and any(code_root.iterdir())
        and not cfg.dry_run
    ):
        raise FileExistsError(f"refusing to overwrite non-empty code tree: {code_root}")

    if not cfg.skip_code:
        validate_git_state(cfg.repo_root, experiment_manifest)

    summaries = [prepare_scenario(cfg, scenario) for scenario in SCENARIOS]

    measurements_summaries: list[dict] = []
    readme_summaries: list[dict] = []
    for scenario in SCENARIOS:
        if not cfg.skip_measurements:
            measurements_summaries.append(install_measurements(cfg, scenario))
        if not cfg.skip_readme:
            readme_summaries.append(install_scenario_readme(cfg, scenario, experiment_manifest))

    for summary in summaries:
        scenario = summary["scenario"]
        for ms in measurements_summaries:
            if ms.get("destination", "").endswith(f"/{scenario}/measurements.json"):
                summary["measurements"] = ms
        for rs in readme_summaries:
            if rs.get("destination", "").endswith(f"/{scenario}/README.md"):
                summary["readme"] = rs

    system_summary = install_system_desc(cfg)

    code_summary: dict | None = None
    if not cfg.skip_code:
        code_summary = package_code_snapshot(
            repo_root=cfg.repo_root,
            experiment_root=experiment_root,
            output_root=cfg.output_root,
            division=cfg.division,
            submitter=cfg.submitter,
            benchmark=cfg.benchmark,
            fetch_data_commit=FETCH_DATA_COMMIT,
            dry_run=cfg.dry_run,
        )

    return write_manifest(
        cfg,
        summaries,
        system_summary=system_summary,
        code_summary=code_summary,
        experiment_manifest=experiment_manifest,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.prepare_submission",
        description=(
            "Copy a run_all experiment into an MLPerf Inference submission tree, "
            "truncate accuracy logs, refresh accuracy.txt hashes, and install "
            "source/measurements/README artefacts."
        ),
    )
    p.add_argument(
        "experiment_root",
        type=Path,
        help="Experiment directory from run_all (contains Offline/, SingleStream/).",
    )
    p.add_argument(
        "--version",
        default="v6.1",
        help="MLPerf Inference version (default: v6.1).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Submission root (creates <output>/<division>/<submitter>/results/...).",
    )
    p.add_argument("--submitter", required=True, help="Submitting organization name.")
    p.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help=f"System description id / results directory name (default: {DEFAULT_SYSTEM}).",
    )
    p.add_argument(
        "--system-desc",
        type=Path,
        default=DEFAULT_SYSTEM_DESC,
        help=f"Path to system description JSON (default: {DEFAULT_SYSTEM_DESC}).",
    )
    p.add_argument(
        "--division",
        default=DEFAULT_DIVISION,
        choices=("closed", "open", "network"),
    )
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument(
        "--user-conf",
        type=Path,
        default=_REPO_ROOT / "configs" / "user.conf",
        help="user.conf copied into each scenario directory.",
    )
    p.add_argument(
        "--measurements",
        type=Path,
        default=DEFAULT_MEASUREMENTS,
        help=f"measurements.json template (default: {DEFAULT_MEASUREMENTS}).",
    )
    p.add_argument(
        "--readme-template",
        type=Path,
        default=DEFAULT_README_TEMPLATE,
        help=f"Per-scenario README template (default: {DEFAULT_README_TEMPLATE}).",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Git repository root for code archival (default: repo root).",
    )
    p.add_argument(
        "--skip-compliance",
        action="store_true",
        help="Omit TEST04 directories (use when compliance was not run).",
    )
    p.add_argument(
        "--skip-vbench-refresh",
        action="store_true",
        help="Copy accuracy.txt from the experiment without --parse-only refresh.",
    )
    p.add_argument(
        "--skip-code",
        action="store_true",
        help="Omit closed/<submitter>/src/<benchmark>/ source snapshot.",
    )
    p.add_argument(
        "--skip-measurements",
        action="store_true",
        help="Omit measurements.json per scenario.",
    )
    p.add_argument(
        "--skip-readme",
        action="store_true",
        help="Omit per-scenario README.md files.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = PrepareConfig(
        experiment_root=args.experiment_root,
        version=args.version,
        output_root=args.output,
        division=args.division,
        submitter=args.submitter,
        system=args.system,
        benchmark=args.benchmark,
        user_conf=args.user_conf,
        system_desc=args.system_desc,
        measurements_template=args.measurements,
        readme_template=args.readme_template,
        repo_root=args.repo_root,
        skip_compliance=args.skip_compliance,
        skip_vbench_refresh=args.skip_vbench_refresh,
        skip_code=args.skip_code,
        skip_measurements=args.skip_measurements,
        skip_readme=args.skip_readme,
        dry_run=args.dry_run,
    )

    try:
        manifest = prepare_submission(cfg)
    except (FileNotFoundError, FileExistsError, RuntimeError) as exc:
        _log.error("%s", exc)
        return 1

    if cfg.dry_run:
        _log.info("dry-run complete (no files written)")
    else:
        _log.info("submission tree ready under %s", cfg.output_root)
        _log.info("manifest: %s", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
