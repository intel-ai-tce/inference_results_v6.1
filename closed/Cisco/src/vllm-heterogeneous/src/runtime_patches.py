from __future__ import annotations

_ALLOWED = {
    "gc_gen2",
    "allreduce_interval",
    "decode_step_pacing",
    "dp_engine_id",
    "engine_core_env",
    "rocm_unified_kv_scale_override",
    "nhd_mode",
    "shuffle_kv",
    "shuffle_to_nhd",
    "kv_scale_override",
    "kv_scale_override_nvidia",
}


def prepare_runtime_for_standalone(patch_names, do_aiter=False):
    unknown = set(patch_names or ()) - _ALLOWED
    if unknown:
        raise RuntimeError(f"Unsupported runtime patch(es): {sorted(unknown)}")
