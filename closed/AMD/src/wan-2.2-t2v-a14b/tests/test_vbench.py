"""Tests for :mod:`wan_harness.vbench`.

The VBench subprocess itself (a full ``torch.distributed.run evaluate.py``
launch that pulls multi-GB of checkpoints) is intentionally not exercised
here; that's covered by a follow-up smoke script. These tests pin the
pure-function behaviour:

* input discovery happy + sad paths;
* command construction (golden argv for the 6-dim default and a custom set);
* result parsing against a synthesised VBench output;
* sha256 byte-equivalence with ``hashlib.sha256``;
* ``accuracy.txt`` rendering, **including regex assertions against the exact
  patterns the upstream submission checker uses**, so any future format
  drift fails this test instead of a real submission run.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from wan_harness.vbench import (
    ACCURACY_THRESHOLD_99,
    DEFAULT_DIMENSIONS,
    MAX_ACCURACY_LOG_SIZE,
    REFERENCE_ACCURACY,
    DimensionScore,
    VBenchResult,
    build_command,
    discover_inputs,
    parse_results,
    render_accuracy_txt,
    sha256_of,
    stage_videos_for_vbench,
    write_accuracy_txt,
    write_summary,
)


# ----------------------------------------------------------------------
# Submission checker regexes. Sourced verbatim from
# tools/submission/submission_checker/constants.py:1427 and
# checks/accuracy_check.py:116 in mlcommons/inference.
# ----------------------------------------------------------------------


SUBMISSION_VBENCH_SCORE_RE = re.compile(r".*'vbench_score':\s([\d.]+).*")
SUBMISSION_HASH_RE = re.compile(r"^hash=([\w\d]+)$")


# ----------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------


def _make_accuracy_run(root: Path, *, mp4_count: int = 3) -> Path:
    """Create an accuracy-mode run dir mock: artefacts/, prompts.json,
    mlperf_log_accuracy.json. Returns the run_dir path.
    """
    run = root / "accuracy"
    artefacts = run / "artefacts"
    artefacts.mkdir(parents=True)
    prompts = {}
    for i in range(mp4_count):
        name = f"{i}.mp4"
        (artefacts / name).write_bytes(b"\x00\x00\x00\x18ftypmp42")
        prompts[name] = f"prompt {i}"
    (artefacts / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
    (run / "mlperf_log_accuracy.json").write_bytes(b'[{"qsl_idx": 0}]')
    return run


def _make_vbench_results(
    out_dir: Path,
    *,
    timestamp: str = "2026-06-05-08:30:00",
    scores: dict[str, float] | None = None,
) -> Path:
    """Write a VBench-shaped ``results_<ts>_eval_results.json`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if scores is None:
        scores = {dim: 0.7048 for dim in DEFAULT_DIMENSIONS}
    payload = {
        name: [
            mean,
            [{"video_path": f"{i}.mp4", "video_results": mean} for i in range(3)],
        ]
        for name, mean in scores.items()
    }
    path = out_dir / f"results_{timestamp}_eval_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_result(vbench_score: float, results_file: Path | None = None) -> VBenchResult:
    """Build a :class:`VBenchResult` with the given vbench_score on the 0-100 scale."""
    overall = vbench_score / 100
    dims = tuple(
        DimensionScore(name=name, mean=overall, per_video=(overall, overall))
        for name in DEFAULT_DIMENSIONS
    )
    return VBenchResult(
        dimensions=dims,
        overall_mean=overall,
        vbench_score=round(vbench_score, 4),
        videos_path=Path("/fake/videos"),
        prompts_path=Path("/fake/prompts.json"),
        results_file=results_file or Path("/fake/results.json"),
        nproc_per_node=8,
    )


# ----------------------------------------------------------------------
# Constants.
# ----------------------------------------------------------------------


def test_default_dimensions_match_upstream_reference() -> None:
    """The MLPerf wan-2.2-t2v-a14b task force averaged exactly 6 dimensions."""
    assert DEFAULT_DIMENSIONS == (
        "subject_consistency",
        "dynamic_degree",
        "motion_smoothness",
        "appearance_style",
        "scene",
        "background_consistency",
    )


def test_threshold_is_99pct_of_reference() -> None:
    """The submission checker uses 70.48 * 0.99 = 69.7752 as the gate."""
    assert REFERENCE_ACCURACY == 70.48
    assert ACCURACY_THRESHOLD_99 == round(70.48 * 0.99, 4)
    assert ACCURACY_THRESHOLD_99 == pytest.approx(69.7752)


# ----------------------------------------------------------------------
# discover_inputs.
# ----------------------------------------------------------------------


def test_discover_inputs_happy(tmp_path: Path) -> None:
    run = _make_accuracy_run(tmp_path)
    videos, prompts = discover_inputs(run)
    assert videos == run / "artefacts"
    assert prompts == run / "artefacts" / "prompts.json"


def test_discover_inputs_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_inputs(tmp_path / "nonexistent")


def test_discover_inputs_missing_artefacts(tmp_path: Path) -> None:
    (tmp_path / "accuracy").mkdir()
    with pytest.raises(FileNotFoundError, match="no artefacts/"):
        discover_inputs(tmp_path / "accuracy")


def test_discover_inputs_missing_prompts_json(tmp_path: Path) -> None:
    run = tmp_path / "accuracy"
    (run / "artefacts").mkdir(parents=True)
    (run / "artefacts" / "0.mp4").write_bytes(b"\x00")
    with pytest.raises(FileNotFoundError, match="prompts.json"):
        discover_inputs(run)


def test_discover_inputs_no_mp4s(tmp_path: Path) -> None:
    """Mock-backend dry runs write .bin frames; reject those clearly."""
    run = tmp_path / "accuracy"
    (run / "artefacts").mkdir(parents=True)
    (run / "artefacts" / "0.bin").write_bytes(b"\x00")
    (run / "artefacts" / "prompts.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="no .mp4"):
        discover_inputs(run)


# ----------------------------------------------------------------------
# build_command.
# ----------------------------------------------------------------------


def _fake_vbench_dir(tmp_path: Path) -> Path:
    """Create a minimal VBench dir with an evaluate.py stub."""
    d = tmp_path / "VBench"
    d.mkdir()
    (d / "evaluate.py").write_text("# stub\n")
    return d


def test_build_command_defaults_use_with_vbench(tmp_path: Path) -> None:
    vd = _fake_vbench_dir(tmp_path)
    cmd = build_command(
        videos_path=Path("/v"),
        output_dir=Path("/o"),
        vbench_dir=vd,
        nproc_per_node=8,
        use_with_vbench=True,
    )
    # with-vbench wrapper -> python -m torch.distributed.run -> evaluate.py
    assert cmd[0] == "with-vbench"
    assert cmd[1] == "python"
    assert cmd[2:4] == ["-m", "torch.distributed.run"]
    assert cmd[4] == "--nproc_per_node=8"
    assert cmd[5] == str(vd / "evaluate.py")
    # vbench_standard mode is the default; --mode and --prompt_file must
    # be absent (they would cause the upstream evaluate.py to refuse
    # scene/appearance_style/etc.). See module docstring for rationale.
    assert "--mode=custom_input" not in cmd
    assert not any(arg.startswith("--prompt_file") for arg in cmd)
    assert "--load_ckpt_from_local=True" in cmd
    assert "--videos_path=/v" in cmd
    dim_idx = cmd.index("--dimension")
    # Default 6 dimensions in the order our module exposes them.
    assert cmd[dim_idx + 1 : dim_idx + 1 + 6] == list(DEFAULT_DIMENSIONS)


def test_build_command_no_with_vbench_uses_sys_executable(tmp_path: Path) -> None:
    """Outside Docker we drive the same evaluate.py from the active Python."""
    import sys

    vd = _fake_vbench_dir(tmp_path)
    cmd = build_command(
        videos_path=Path("/v"),
        output_dir=Path("/o"),
        vbench_dir=vd,
        nproc_per_node=4,
        use_with_vbench=False,
    )
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "torch.distributed.run"]
    assert cmd[3] == "--nproc_per_node=4"


def test_build_command_custom_dimensions(tmp_path: Path) -> None:
    vd = _fake_vbench_dir(tmp_path)
    cmd = build_command(
        videos_path=Path("/v"),
        output_dir=Path("/o"),
        vbench_dir=vd,
        dimensions=("dynamic_degree", "scene"),
        use_with_vbench=False,
    )
    dim_idx = cmd.index("--dimension")
    assert cmd[dim_idx + 1 :] == ["dynamic_degree", "scene"]


def test_build_command_rejects_empty_dimensions(tmp_path: Path) -> None:
    vd = _fake_vbench_dir(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        build_command(
            videos_path=Path("/v"),
            output_dir=Path("/o"),
            vbench_dir=vd,
            dimensions=(),
            use_with_vbench=False,
        )


def test_build_command_rejects_bad_nproc(tmp_path: Path) -> None:
    vd = _fake_vbench_dir(tmp_path)
    with pytest.raises(ValueError, match="nproc_per_node"):
        build_command(
            videos_path=Path("/v"),
            output_dir=Path("/o"),
            vbench_dir=vd,
            nproc_per_node=0,
            use_with_vbench=False,
        )


def test_build_command_missing_vbench_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="evaluate.py"):
        build_command(
            videos_path=Path("/v"),
            output_dir=Path("/o"),
            vbench_dir=tmp_path / "does-not-exist",
            use_with_vbench=False,
        )


# ----------------------------------------------------------------------
# stage_videos_for_vbench.
# ----------------------------------------------------------------------


def test_stage_videos_renames_to_prompt_index_form(tmp_path: Path) -> None:
    run = _make_accuracy_run(tmp_path, mp4_count=3)
    artefacts = run / "artefacts"
    staging = tmp_path / "staged"
    stage_videos_for_vbench(artefacts, artefacts / "prompts.json", staging)

    # Each prompt in the mock run is unique, so VBench's per-prompt
    # iteration counter is always 0 -- the suffix is `-0.mp4` for every
    # staged entry, not the harness's global on-disk index.
    staged = sorted(staging.glob("*.mp4"))
    assert len(staged) == 3
    expected = {f"prompt {i}-0.mp4" for i in range(3)}
    assert {p.name for p in staged} == expected
    # Symlinks point at the canonical numeric source.
    for s in staged:
        assert s.is_symlink()
        assert s.resolve().parent == artefacts


def test_stage_videos_iteration_counter_increments_per_repeated_prompt(
    tmp_path: Path,
) -> None:
    """When prompts.json repeats the same prompt (5-iteration MLPerf
    layout), the staged suffix increments 0..4 so VBench's standard mode
    finds all of them. The 6th+ copy is dropped with a warning."""
    run = tmp_path / "run"
    artefacts = run / "artefacts"
    artefacts.mkdir(parents=True)
    # 6 videos all pointing at the same prompt -- VBench standard mode
    # only looks at -0.mp4 .. -4.mp4, so the 6th must be skipped.
    prompts: dict[str, str] = {}
    for i in range(6):
        name = f"{i}.mp4"
        (artefacts / name).write_bytes(b"\x00\x00\x00\x18ftypmp42")
        prompts[name] = "repeated prompt"
    (artefacts / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")

    staging = tmp_path / "staged"
    stage_videos_for_vbench(artefacts, artefacts / "prompts.json", staging)
    staged = sorted(p.name for p in staging.glob("*.mp4"))
    assert staged == [f"repeated prompt-{i}.mp4" for i in range(5)]


def test_stage_videos_is_idempotent(tmp_path: Path) -> None:
    """Re-staging clears stale entries so a renamed prompt set doesn't
    leave dangling files behind."""
    run = _make_accuracy_run(tmp_path, mp4_count=2)
    artefacts = run / "artefacts"
    staging = tmp_path / "staged"
    stage_videos_for_vbench(artefacts, artefacts / "prompts.json", staging)

    # Bake a stale entry into the staging dir to simulate a prior run.
    stale = staging / "old-prompt-99.mp4"
    stale.symlink_to(artefacts / "0.mp4")
    assert stale.exists()

    stage_videos_for_vbench(artefacts, artefacts / "prompts.json", staging)
    # The stale name is gone; only the 2 current entries remain.
    assert not stale.exists()
    assert sorted(p.name for p in staging.glob("*.mp4")) == [
        "prompt 0-0.mp4",
        "prompt 1-0.mp4",
    ]


def test_stage_videos_warns_on_missing_source(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """prompts.json may list samples whose .mp4 never got written (rare
    but possible after a partial accuracy run); skip + log instead of
    failing the whole evaluation."""
    run = _make_accuracy_run(tmp_path, mp4_count=2)
    artefacts = run / "artefacts"
    # Reference a third file that does not exist on disk.
    prompts = json.loads((artefacts / "prompts.json").read_text())
    prompts["99.mp4"] = "prompt 99"
    (artefacts / "prompts.json").write_text(json.dumps(prompts))

    staging = tmp_path / "staged"
    with caplog.at_level("WARNING", logger="wan_harness.vbench"):
        stage_videos_for_vbench(artefacts, artefacts / "prompts.json", staging)
    assert any("99.mp4 missing" in r.message for r in caplog.records)
    # The two valid entries still get staged.
    assert len(list(staging.glob("*.mp4"))) == 2


def test_stage_videos_rejects_prompts_overflowing_namemax(tmp_path: Path) -> None:
    run = _make_accuracy_run(tmp_path, mp4_count=1)
    artefacts = run / "artefacts"
    long_prompt = "x" * 260
    (artefacts / "prompts.json").write_text(json.dumps({"0.mp4": long_prompt}))
    with pytest.raises(ValueError, match="NAME_MAX"):
        stage_videos_for_vbench(
            artefacts, artefacts / "prompts.json", tmp_path / "staged"
        )


# ----------------------------------------------------------------------
# parse_results.
# ----------------------------------------------------------------------


def test_parse_results_computes_overall_mean(tmp_path: Path) -> None:
    _make_vbench_results(
        tmp_path,
        scores={
            "subject_consistency": 0.9,
            "dynamic_degree": 0.8,
            "motion_smoothness": 0.7,
            "appearance_style": 0.6,
            "scene": 0.5,
            "background_consistency": 0.4,
        },
    )
    result = parse_results(
        tmp_path,
        videos_path=Path("/v"),
        prompts_path=Path("/p.json"),
        nproc_per_node=8,
    )
    assert len(result.dimensions) == 6
    expected_mean = (0.9 + 0.8 + 0.7 + 0.6 + 0.5 + 0.4) / 6
    assert result.overall_mean == pytest.approx(expected_mean)
    assert result.vbench_score == pytest.approx(round(expected_mean * 100, 4))


def test_parse_results_picks_latest_by_timestamp(tmp_path: Path) -> None:
    _make_vbench_results(
        tmp_path,
        timestamp="2026-06-05-08:00:00",
        scores={"subject_consistency": 0.10},
    )
    _make_vbench_results(
        tmp_path,
        timestamp="2026-06-05-08:30:00",
        scores={"subject_consistency": 0.99},
    )
    result = parse_results(
        tmp_path,
        videos_path=Path("/v"),
        prompts_path=Path("/p.json"),
        nproc_per_node=1,
    )
    assert result.results_file.name == "results_2026-06-05-08:30:00_eval_results.json"
    # The 0.99 score from the latest file, not 0.10 from the older one.
    assert result.overall_mean == pytest.approx(0.99)


def test_parse_results_pass_99_boundary() -> None:
    """The pass/fail flips around 69.7752."""
    above = _make_result(69.7752 + 1e-3)
    assert above.pass_99 is True

    below = _make_result(69.7752 - 1e-3)
    assert below.pass_99 is False


def test_parse_results_empty_payload_raises(tmp_path: Path) -> None:
    (tmp_path / "results_2026-06-05-00:00:00_eval_results.json").write_text("{}")
    with pytest.raises(ValueError, match="empty"):
        parse_results(
            tmp_path,
            videos_path=Path("/v"),
            prompts_path=Path("/p.json"),
            nproc_per_node=1,
        )


def test_parse_results_no_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="results_"):
        parse_results(
            tmp_path,
            videos_path=Path("/v"),
            prompts_path=Path("/p.json"),
            nproc_per_node=1,
        )


# ----------------------------------------------------------------------
# sha256_of.
# ----------------------------------------------------------------------


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    """Truncate_accuracy_log.py:get_hash() reads 4096-byte chunks; our
    streaming implementation must produce the same digest as the obvious
    single-shot hashlib call.
    """
    body = b"x" * (4096 * 3 + 17)  # exercise multiple chunks + remainder
    f = tmp_path / "blob.bin"
    f.write_bytes(body)
    assert sha256_of(f) == hashlib.sha256(body).hexdigest()


def test_sha256_of_known_value(tmp_path: Path) -> None:
    f = tmp_path / "hello.bin"
    f.write_bytes(b"hello")
    assert sha256_of(f) == hashlib.sha256(b"hello").hexdigest()


# ----------------------------------------------------------------------
# render_accuracy_txt / write_accuracy_txt — submission contract.
# ----------------------------------------------------------------------


def test_accuracy_txt_passes_submission_checker_regexes(tmp_path: Path) -> None:
    """The two regexes the upstream submission checker uses against
    accuracy.txt MUST match every file we emit. This pins us to the
    exact format the checker expects, today and going forward.
    """
    result = _make_result(vbench_score=70.16)
    txt = render_accuracy_txt(result, acc_json_sha256="deadbeef" * 8)

    # 'vbench_score' line, parsed by constants.py:1427.
    score_match = None
    hash_match = None
    for line in txt.splitlines():
        if score_match is None:
            score_match = SUBMISSION_VBENCH_SCORE_RE.match(line)
        if hash_match is None:
            hash_match = SUBMISSION_HASH_RE.match(line)
    assert score_match is not None, f"no 'vbench_score' line in:\n{txt}"
    assert float(score_match.group(1)) == pytest.approx(70.16, abs=1e-4)
    assert hash_match is not None, f"no hash= line in:\n{txt}"
    assert hash_match.group(1) == "deadbeef" * 8


def test_accuracy_txt_pass_yes_when_above_threshold() -> None:
    result = _make_result(vbench_score=70.16)
    txt = render_accuracy_txt(result, acc_json_sha256="x" * 64)
    assert "Pass: Yes" in txt


def test_accuracy_txt_pass_no_when_below_threshold() -> None:
    result = _make_result(vbench_score=69.0)
    txt = render_accuracy_txt(result, acc_json_sha256="x" * 64)
    assert "Pass: No" in txt


def test_accuracy_txt_shape_matches_v6_reference() -> None:
    """Spot-check against the v6.0 NVIDIA / Cisco accuracy.txt shape."""
    result = _make_result(vbench_score=70.16)
    txt = render_accuracy_txt(result, acc_json_sha256="abc123")
    assert "VBench Evaluation Results" in txt
    assert "Dimension Scores:" in txt
    assert "Overall Average" in txt
    assert "Threshold:" in txt
    assert "Detailed results:" in txt
    # Each of the 6 reference dimensions appears once.
    for dim in DEFAULT_DIMENSIONS:
        assert txt.count(dim) >= 1


def test_write_accuracy_txt_writes_file_and_hashes_acc_json(tmp_path: Path) -> None:
    result = _make_result(vbench_score=70.16)
    acc_json = tmp_path / "mlperf_log_accuracy.json"
    body = b'[{"qsl_idx": 0}]'
    acc_json.write_bytes(body)
    dest = tmp_path / "accuracy.txt"
    write_accuracy_txt(result, dest, acc_json_path=acc_json)
    txt = dest.read_text()
    expected = hashlib.sha256(body).hexdigest()
    assert f"hash={expected}" in txt


def test_write_accuracy_txt_warns_on_oversize_acc_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Submitters who forgot to truncate get a warning, not an error."""
    result = _make_result(vbench_score=70.16)
    acc_json = tmp_path / "mlperf_log_accuracy.json"
    acc_json.write_bytes(b"x" * (MAX_ACCURACY_LOG_SIZE + 1))
    dest = tmp_path / "accuracy.txt"
    with caplog.at_level("WARNING", logger="wan_harness.vbench"):
        write_accuracy_txt(result, dest, acc_json_path=acc_json)
    assert any("MAX_ACCURACY_LOG_SIZE" in r.message for r in caplog.records)
    assert dest.exists()


def test_write_accuracy_txt_missing_acc_json_raises(tmp_path: Path) -> None:
    result = _make_result(vbench_score=70.16)
    with pytest.raises(FileNotFoundError, match="cannot compute hash"):
        write_accuracy_txt(
            result,
            tmp_path / "accuracy.txt",
            acc_json_path=tmp_path / "missing.json",
        )


# ----------------------------------------------------------------------
# write_summary.
# ----------------------------------------------------------------------


def test_write_summary_round_trip(tmp_path: Path) -> None:
    result = _make_result(vbench_score=70.0, results_file=tmp_path / "rf.json")
    acc_json = tmp_path / "mlperf_log_accuracy.json"
    acc_json.write_bytes(b"{}")
    dest = tmp_path / "vbench_summary.json"
    write_summary(
        result,
        dest,
        run_dir=tmp_path,
        accuracy_txt=tmp_path / "accuracy.txt",
        accuracy_json=acc_json,
        accuracy_json_sha256="d" * 64,
        extra={"extra_key": "extra_val"},
    )
    payload = json.loads(dest.read_text())
    assert payload["vbench_score"] == pytest.approx(70.0)
    assert payload["overall_mean"] == pytest.approx(0.70)
    assert payload["pass_99"] is True
    assert payload["accuracy_json_sha256"] == "d" * 64
    # accuracy_json is 2 bytes (b"{}"), well under MAX_ACCURACY_LOG_SIZE.
    assert payload["accuracy_json_truncated"] is True
    assert payload["accuracy_json_bytes"] == 2
    # Dimensions echoed back sorted by name.
    names = [d["name"] for d in payload["dimensions"]]
    assert names == sorted(names)
    assert payload["extra_key"] == "extra_val"
