"""Round-trip tests for the dispatcher wire protocol.

The message dataclasses must be picklable (they cross processes) and
their default values must survive ``copy``. Network-level tests with
``torch.distributed`` are deferred to the in-container manual smoke
script – we don't spawn worker processes in pytest.
"""

from __future__ import annotations

import copy
import pickle

import pytest

from wan_harness.wire import (
    Result,
    ResultMeta,
    Shutdown,
    WorkUnit,
    WorkerFailure,
    meta_to_result,
    result_to_meta,
)


def test_workunit_pickle_roundtrip() -> None:
    unit = WorkUnit(
        sample_index=42,
        prompt="A serene mountain at sunrise",
        input_args={
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "seed": 42,
            "guidance_scale": 4.0,
            "guidance_scale_2": 3.0,
            "negative_prompt": "blurry",
        },
    )
    blob = pickle.dumps(unit, protocol=pickle.HIGHEST_PROTOCOL)
    back: WorkUnit = pickle.loads(blob)
    assert back == unit
    assert back.input_args["height"] == 720


def test_result_pickle_roundtrip() -> None:
    payload = bytes(range(256))
    res = Result(
        sample_index=7,
        frames_bytes=payload,
        frame_count=2,
        height=8,
        width=8,
        mp4_bytes=None,
    )
    back = pickle.loads(pickle.dumps(res))
    assert back == res
    assert back.frames_bytes == payload


def test_result_meta_roundtrip() -> None:
    payload = bytes([0, 1, 2, 3] * 1024)
    mp4 = b"mp4-bytes"
    res = Result(
        sample_index=11,
        frames_bytes=payload,
        frame_count=4,
        height=16,
        width=16,
        mp4_bytes=mp4,
    )
    meta = result_to_meta(res)
    back = meta_to_result(meta, frames_bytes=payload, mp4_bytes=mp4)
    assert back == res
    assert isinstance(meta, ResultMeta)


def test_result_meta_roundtrip_no_mp4() -> None:
    payload = b"\xff" * 4096
    res = Result(
        sample_index=0,
        frames_bytes=payload,
        frame_count=1,
        height=8,
        width=8,
        mp4_bytes=None,
    )
    meta = result_to_meta(res)
    back = meta_to_result(meta, frames_bytes=payload, mp4_bytes=None)
    assert back == res


def test_bytes_to_uint8_tensor_avoids_frombuffer_warning() -> None:
    import warnings

    torch = pytest.importorskip("torch")
    from wan_harness.wire import _bytes_to_uint8_tensor, _uint8_tensor_to_bytes

    blob = bytes(range(256)) * 1024
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        tensor = _bytes_to_uint8_tensor(blob)
    assert tensor.dtype == torch.uint8
    assert _uint8_tensor_to_bytes(tensor) == blob


def test_shutdown_is_singleton_picklable() -> None:
    msg = Shutdown()
    assert pickle.loads(pickle.dumps(msg)) == msg


def test_worker_failure_pickle_roundtrip() -> None:
    """Both DP dispatchers wrap worker exceptions in WorkerFailure and ship
    it over the wire (gather bucket for wave, point-to-point send for
    async). The dataclass must survive pickle without losing fields."""
    fail = WorkerFailure(
        rank=3,
        sample_index=42,
        error_repr="RuntimeError('boom')",
    )
    back = pickle.loads(pickle.dumps(fail))
    assert back == fail
    assert back.rank == 3
    assert back.sample_index == 42
    assert "boom" in back.error_repr


def test_workunit_copy_is_independent() -> None:
    """A future change to a copied unit's input_args must not leak back."""
    unit = WorkUnit(sample_index=1, prompt="p", input_args={"k": 1})
    clone = copy.deepcopy(unit)
    clone.input_args["k"] = 99
    assert unit.input_args["k"] == 1
    assert clone.input_args["k"] == 99


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_index", 0),
        ("sample_index", 247),
        ("prompt", "a" * 4096),  # long prompt
    ],
)
def test_workunit_field_edges(field: str, value) -> None:
    kwargs = {"sample_index": 0, "prompt": "p", "input_args": {}}
    kwargs[field] = value
    unit = WorkUnit(**kwargs)
    back = pickle.loads(pickle.dumps(unit))
    assert getattr(back, field) == value
