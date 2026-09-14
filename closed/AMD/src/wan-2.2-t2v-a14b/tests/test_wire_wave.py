"""Pure-Python tests for the wave-dispatcher wire helpers.

The encode/decode pair operates on plain ``list[int]`` payloads so the
edge logic stays testable without ``torch`` or ``torch.distributed``.
The tensor wrappers and the broadcast/gather helpers are exercised by
the Gloo multi-process integration tests in ``tests/test_wave_dispatcher_dist.py``.
"""

from __future__ import annotations

import pytest

from wan_harness.wire import (
    WAVE_CMD_EXIT,
    WAVE_CMD_RUN,
    WAVE_INACTIVE_INDEX,
    decode_wave_command_payload,
    encode_wave_command_payload,
)


# ----------------------------------------------------------------------
# Constants.
# ----------------------------------------------------------------------


def test_command_codes_are_distinct() -> None:
    assert WAVE_CMD_RUN != WAVE_CMD_EXIT


def test_inactive_index_is_negative() -> None:
    # Real QSL indices are non-negative; warmup uses negatives but those
    # never enter the wave payload (warmup goes through the same path
    # but with synthetic indices < 0 -- if we ever overlap, the EXIT cmd
    # disambiguates first, and active slots are explicitly sized by ``n``
    # rather than by sniffing for the sentinel).
    assert WAVE_INACTIVE_INDEX < 0


# ----------------------------------------------------------------------
# encode_wave_command_payload.
# ----------------------------------------------------------------------


def test_encode_full_wave() -> None:
    payload = encode_wave_command_payload(WAVE_CMD_RUN, [10, 11, 12, 13], world_size=4)
    assert payload == [WAVE_CMD_RUN, 4, 10, 11, 12, 13]


def test_encode_short_wave_pads_trailing_slots() -> None:
    payload = encode_wave_command_payload(WAVE_CMD_RUN, [10, 11], world_size=4)
    assert payload == [WAVE_CMD_RUN, 2, 10, 11, WAVE_INACTIVE_INDEX, WAVE_INACTIVE_INDEX]


def test_encode_empty_wave_is_all_padding() -> None:
    payload = encode_wave_command_payload(WAVE_CMD_EXIT, [], world_size=4)
    assert payload[0] == WAVE_CMD_EXIT
    assert payload[1] == 0
    # All slots are inactive.
    assert payload[2:] == [WAVE_INACTIVE_INDEX] * 4


def test_encode_world_size_one() -> None:
    payload = encode_wave_command_payload(WAVE_CMD_RUN, [42], world_size=1)
    assert payload == [WAVE_CMD_RUN, 1, 42]


def test_encode_rejects_oversized_wave() -> None:
    with pytest.raises(ValueError):
        encode_wave_command_payload(WAVE_CMD_RUN, [1, 2, 3, 4, 5], world_size=4)


def test_encode_rejects_zero_world_size() -> None:
    with pytest.raises(ValueError):
        encode_wave_command_payload(WAVE_CMD_RUN, [], world_size=0)


def test_encode_payload_length_is_fixed() -> None:
    """The recv side allocates ``2 + world_size`` ints unconditionally,
    so the encoder must always produce that length."""
    for n in range(0, 5):
        payload = encode_wave_command_payload(
            WAVE_CMD_RUN, list(range(n)), world_size=4
        )
        assert len(payload) == 2 + 4


# ----------------------------------------------------------------------
# decode_wave_command_payload.
# ----------------------------------------------------------------------


def test_decode_full_wave() -> None:
    cmd, indices = decode_wave_command_payload([WAVE_CMD_RUN, 4, 10, 11, 12, 13])
    assert cmd == WAVE_CMD_RUN
    assert indices == [10, 11, 12, 13]


def test_decode_short_wave_strips_padding() -> None:
    raw = [WAVE_CMD_RUN, 2, 10, 11, WAVE_INACTIVE_INDEX, WAVE_INACTIVE_INDEX]
    cmd, indices = decode_wave_command_payload(raw)
    assert cmd == WAVE_CMD_RUN
    # Only the first ``n`` slots are exposed.
    assert indices == [10, 11]


def test_decode_exit_has_no_indices() -> None:
    raw = encode_wave_command_payload(WAVE_CMD_EXIT, [], world_size=8)
    cmd, indices = decode_wave_command_payload(raw)
    assert cmd == WAVE_CMD_EXIT
    assert indices == []


def test_decode_rejects_truncated_payload() -> None:
    with pytest.raises(ValueError):
        decode_wave_command_payload([WAVE_CMD_RUN])


def test_decode_rejects_negative_n() -> None:
    with pytest.raises(ValueError):
        decode_wave_command_payload([WAVE_CMD_RUN, -1, 0, 0])


def test_decode_rejects_oversized_n_for_payload_length() -> None:
    with pytest.raises(ValueError):
        decode_wave_command_payload([WAVE_CMD_RUN, 5, 0, 0])


# ----------------------------------------------------------------------
# Round-trip.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "indices,world_size",
    [
        ([], 1),
        ([0], 1),
        ([0, 1, 2, 3, 4, 5, 6, 7], 8),
        ([0, 1, 2], 8),  # short last wave
        ([99], 8),
        (list(range(10, 18)), 8),
    ],
)
def test_roundtrip_preserves_indices(indices: list[int], world_size: int) -> None:
    payload = encode_wave_command_payload(WAVE_CMD_RUN, indices, world_size)
    cmd, decoded = decode_wave_command_payload(payload)
    assert cmd == WAVE_CMD_RUN
    assert decoded == indices


def test_exit_roundtrip_ignores_indices() -> None:
    # Encoder accepts any indices on EXIT but ``n`` is what the decoder uses;
    # an EXIT wave is canonically empty.
    payload = encode_wave_command_payload(WAVE_CMD_EXIT, [], world_size=4)
    cmd, decoded = decode_wave_command_payload(payload)
    assert cmd == WAVE_CMD_EXIT
    assert decoded == []
