"""Tests for the SUT against the MockBackend, without LoadGen installed.

These exercise the most important loadgen-side invariants of the harness
that the v6.0 implementation got wrong:

  * ``issue_queries`` calls ``QuerySamplesComplete`` per-sample, in the order
    the backend yields them.
  * Response buffers stay alive across the issue/flush boundary.
  * The artefact writer is invoked exactly once per yielded sample.
  * Each ``QuerySample.id`` is matched to the correct ``QuerySample.index``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wan_harness.artefacts import ArtefactWriter
from wan_harness.backends.mock import MockBackend
from wan_harness.config import HarnessConfig
from wan_harness.data.prompts import synthetic_prompts
from wan_harness.dispatcher import SingleProcessDispatcher
from wan_harness.qsl import WanQSL
from wan_harness.response import RecordingResponseWriter
from wan_harness.sut import QuerySampleLike, WanSUT


def _build_harness(
    *,
    height: int = 8,
    width: int = 16,
    num_frames: int = 2,
    mock_delay_ms: int = 0,
    accuracy_dir: Path | None = None,
):
    cfg = HarnessConfig(
        height=height,
        width=width,
        num_frames=num_frames,
        mock_delay_ms=mock_delay_ms,
    )
    backend = MockBackend(cfg)
    backend.setup()
    dispatcher = SingleProcessDispatcher(backend)
    qsl = WanQSL(synthetic_prompts(8))
    writer = RecordingResponseWriter()
    artefact_writer = ArtefactWriter(accuracy_dir) if accuracy_dir else None
    sut = WanSUT(
        dispatcher=dispatcher,
        qsl=qsl,
        response_writer=writer,
        artefact_writer=artefact_writer,
    )
    return sut, backend, qsl, writer, artefact_writer


def test_offline_batch_completes_all() -> None:
    sut, backend, _, writer, _ = _build_harness()
    samples = [
        QuerySampleLike(index=i, id=1000 + i) for i in range(5)
    ]
    sut.issue_queries(samples)
    sut.flush_queries()
    backend.teardown()

    assert sut.issued_count == 5
    assert sut.completed_count == 5
    assert len(writer.responses) == 5

    # Order: MockBackend yields in input order.
    received_ids = [qid for qid, _ in writer.responses]
    assert received_ids == [1000, 1001, 1002, 1003, 1004]


def test_payload_size_matches_frames() -> None:
    H, W, F = 16, 32, 3
    sut, backend, _, writer, _ = _build_harness(height=H, width=W, num_frames=F)
    samples = [QuerySampleLike(index=0, id=42)]
    sut.issue_queries(samples)
    sut.flush_queries()
    backend.teardown()

    assert len(writer.responses) == 1
    _, payload = writer.responses[0]
    assert len(payload) == F * H * W * 3


def test_single_stream_one_at_a_time() -> None:
    sut, backend, _, writer, _ = _build_harness()
    for i in range(3):
        sut.issue_queries([QuerySampleLike(index=i, id=500 + i)])
        sut.flush_queries()
    backend.teardown()

    assert [qid for qid, _ in writer.responses] == [500, 501, 502]


def test_artefact_writer_called_per_sample(tmp_path: Path) -> None:
    out = tmp_path / "artefacts"
    sut, backend, qsl, _, artefact_writer = _build_harness(accuracy_dir=out)
    samples = [QuerySampleLike(index=i, id=900 + i) for i in range(4)]
    sut.issue_queries(samples)
    sut.flush_queries()
    backend.teardown()

    assert artefact_writer is not None
    written = sorted(out.iterdir())
    bins = [p for p in written if p.suffix == ".bin"]
    assert len(bins) == 4
    # Filenames are bare sample indices, no zero-padding: 0.bin..3.bin.
    assert sorted(p.name for p in bins) == ["0.bin", "1.bin", "2.bin", "3.bin"]
    index_lines = (out / "artefacts.jsonl").read_text().strip().splitlines()
    assert len(index_lines) == 4

    # Every artefacts.jsonl record carries the prompt the SUT was asked for.
    import json
    expected_prompts = qsl.get_prompts([0, 1, 2, 3])
    records = [json.loads(line) for line in index_lines]
    by_idx = {r["sample_index"]: r for r in records}
    for i, expected in enumerate(expected_prompts):
        assert by_idx[i]["prompt"] == expected


def test_artefact_writer_finalize_emits_vbench_prompts_json(tmp_path: Path) -> None:
    """The custom_input prompt sidecar must materialise on finalize() with
    one entry per written artefact, in the shape VBench expects."""
    out = tmp_path / "artefacts"
    sut, backend, qsl, _, artefact_writer = _build_harness(accuracy_dir=out)
    samples = [QuerySampleLike(index=i, id=700 + i) for i in range(3)]
    sut.issue_queries(samples)
    sut.flush_queries()
    backend.teardown()

    assert artefact_writer is not None
    out_path = artefact_writer.finalize()
    assert out_path == out / "prompts.json"

    import json
    mapping = json.loads(out_path.read_text())
    assert isinstance(mapping, dict)
    assert set(mapping.keys()) == {"0.bin", "1.bin", "2.bin"}
    expected_prompts = qsl.get_prompts([0, 1, 2])
    for i, expected in enumerate(expected_prompts):
        assert mapping[f"{i}.bin"] == expected


def test_artefact_writer_finalize_is_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "artefacts"
    sut, backend, _, _, artefact_writer = _build_harness(accuracy_dir=out)
    sut.issue_queries([QuerySampleLike(index=0, id=1)])
    sut.flush_queries()
    backend.teardown()

    assert artefact_writer is not None
    p1 = artefact_writer.finalize()
    p2 = artefact_writer.finalize()
    assert p1 == p2
    assert p1.exists()


def test_backend_returns_unknown_index_raises() -> None:
    """The SUT must reject completions for sample indices it did not issue."""
    sut, backend, _, _, _ = _build_harness()
    samples = [QuerySampleLike(index=0, id=10)]

    # Drive the dispatcher manually to inject a bad index.
    from wan_harness.backends.base import GeneratedVideo

    class BadDispatcher:
        info = sut._dispatcher.info  # type: ignore[attr-defined]

        def generate(self, prompts, indices):
            yield GeneratedVideo(
                sample_index=99,  # not in `indices`
                frames_bytes=b"x",
                frame_count=1,
                height=1,
                width=1,
            )

        def is_response_owner(self):
            return True

        def flush(self):
            return

        def shutdown(self):
            return

    sut._dispatcher = BadDispatcher()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        sut.issue_queries(samples)
    backend.teardown()


def test_offline_duplicate_indices_complete_each_qid_once() -> None:
    """LoadGen Offline replicates QSL indices to satisfy ``samples_per_query``
    when it exceeds the QSL size. The SUT must call ``QuerySamplesComplete``
    once per *distinct* ``query_id`` – never twice on the same id, never
    skipping ids – even though the same ``sample_index`` appears multiple
    times in the batch.

    Regression test for the v6.1 baseline hang where the SUT collapsed
    duplicate indices to a single ``qid`` via ``setdefault``, then completed
    that ``qid`` once per occurrence and silently dropped the rest. LoadGen
    logged hundreds of ``error_runtime: "Attempted to complete a sample
    twice."`` and stalled in its post-test phase.
    """
    sut, backend, _, writer, _ = _build_harness()
    # Two distinct QSL indices, each issued multiple times with distinct qids
    # in interleaved order – mirrors what LoadGen does with replacement.
    samples = [
        QuerySampleLike(index=0, id=100),
        QuerySampleLike(index=1, id=200),
        QuerySampleLike(index=0, id=101),
        QuerySampleLike(index=1, id=201),
        QuerySampleLike(index=0, id=102),
    ]
    sut.issue_queries(samples)
    sut.flush_queries()
    backend.teardown()

    assert sut.issued_count == 5
    assert sut.completed_count == 5
    assert len(writer.responses) == 5

    # Every issued query_id is completed exactly once.
    completed_ids = [qid for qid, _ in writer.responses]
    assert sorted(completed_ids) == [100, 101, 102, 200, 201]
    assert len(set(completed_ids)) == 5

    # FIFO contract: occurrences of a given sample_index must be paired with
    # query_ids in the order they were issued. The MockBackend yields in
    # input order, so completions for index=0 should come back as 100, 101,
    # 102 (and 200, 201 for index=1) – in exactly that interleaving.
    assert completed_ids == [100, 200, 101, 201, 102]


def test_offline_duplicate_indices_artefact_writer_called_per_yield(
    tmp_path: Path,
) -> None:
    """When indices repeat, the artefact writer is called once per yielded
    completion (not once per *unique* index) and each call carries the
    prompt that was paired with that occurrence.

    The on-disk filename is ``{index}.bin``, so repeated indices overwrite
    the same file with byte-identical content (deterministic for the Mock
    backend with the ``zeros`` payload). The ``artefacts.jsonl`` index
    therefore grows by one line per yield.
    """
    out = tmp_path / "artefacts"
    sut, backend, qsl, _, artefact_writer = _build_harness(accuracy_dir=out)
    samples = [
        QuerySampleLike(index=0, id=10),
        QuerySampleLike(index=1, id=11),
        QuerySampleLike(index=0, id=12),
    ]
    sut.issue_queries(samples)
    sut.flush_queries()
    backend.teardown()

    assert sut.completed_count == 3
    assert artefact_writer is not None

    import json
    index_lines = (out / "artefacts.jsonl").read_text().strip().splitlines()
    assert len(index_lines) == 3
    records = [json.loads(line) for line in index_lines]
    # Every record must carry the prompt the QSL holds for that index.
    expected = qsl.get_prompts([0, 1, 0])
    for rec, prompt in zip(records, expected):
        assert rec["prompt"] == prompt


def test_offline_overproduction_raises() -> None:
    """If a buggy dispatcher yields more completions than the SUT issued
    for a sample_index (i.e. the per-index FIFO is empty), the SUT must
    fail loudly instead of silently double-completing some other ``qid``.
    """
    sut, backend, _, _, _ = _build_harness()

    from wan_harness.backends.base import GeneratedVideo

    class OverproducingDispatcher:
        info = sut._dispatcher.info  # type: ignore[attr-defined]

        def generate(self, prompts, indices):
            # Yield twice for index 0 even though the SUT only issued it once.
            for _ in range(2):
                yield GeneratedVideo(
                    sample_index=0,
                    frames_bytes=b"x",
                    frame_count=1,
                    height=1,
                    width=1,
                )

        def is_response_owner(self):
            return True

        def flush(self):
            return

        def shutdown(self):
            return

    sut._dispatcher = OverproducingDispatcher()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="no pending query_id"):
        sut.issue_queries([QuerySampleLike(index=0, id=42)])
    backend.teardown()


def test_mock_backend_noise_payload_deterministic() -> None:
    cfg = HarnessConfig(height=8, width=8, num_frames=1, mock_payload="noise")
    a = MockBackend(cfg); a.setup()
    b = MockBackend(cfg); b.setup()
    payload_a = list(a.generate(["hello"], [0]))[0].frames_bytes
    payload_b = list(b.generate(["hello"], [0]))[0].frames_bytes
    assert payload_a == payload_b
    payload_c = list(b.generate(["world"], [0]))[0].frames_bytes
    assert payload_a != payload_c
    a.teardown(); b.teardown()
