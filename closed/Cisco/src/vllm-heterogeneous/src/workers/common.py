"""Shared utilities for ZMQ prefill and decode workers."""

import json
import os


MLPERF_PREFIX_CACHING_RULE = (
    "https://github.com/mlcommons<submission-root>_policies/blob/master/"
    "inference_rules.adoc#L894-L904"
)

try:
    import msgpack as _msgpack
    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False


def nixl_kv_connector_extra_config():
    """Build kv_connector_extra_config for NixlConnector.

    Precedence: NIXL_BACKENDS (comma-separated) > NIXL_USE_UCCL > default UCX.
    """
    extra = {"enforce_handshake_compat": False}
    raw = os.environ.get("NIXL_BACKENDS", "").strip()
    if raw:
        names = [x.strip() for x in raw.split(",") if x.strip()]
        if names:
            extra["backends"] = names
            return extra
    if os.environ.get("NIXL_USE_UCCL", "false").lower() == "true":
        extra["backends"] = ["UCCL"]
    return extra


def pack(obj):
    """Serialize a Python object to bytes (msgpack preferred, JSON fallback)."""
    if _HAS_MSGPACK:
        return _msgpack.packb(obj, use_bin_type=True)
    return json.dumps(obj, separators=(",", ":")).encode()


def unpack(data):
    """Deserialize bytes to a Python object (auto-detect msgpack vs JSON)."""
    if isinstance(data, (bytes, bytearray)) and data and data[0] not in (0x7B, 0x5B):
        if _HAS_MSGPACK:
            return _msgpack.unpackb(data, raw=False)
    return json.loads(data)


def load_yaml_config(path):
    """Load a worker YAML config and return (engine, network, full) dicts."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("engine", {}), cfg.get("network", {}), cfg


def build_capture_sizes(range_spec):
    """Expand a cudagraph_capture_range spec like [[4096,0,-8], 4, 2, 1]."""
    sizes = []
    for item in range_spec:
        if isinstance(item, list):
            sizes.extend(range(*item))
        else:
            sizes.append(item)
    return sizes


def cfg(engine, key, env_var, default, typ=str):
    """Resolve a config value: YAML engine dict > env var > default."""
    if key in engine:
        return typ(engine[key])
    return typ(os.environ.get(env_var, default))


def optional_bool_cfg(engine, key, env_var):
    """Resolve an optional boolean while preserving an unset value as None."""
    value = engine[key] if key in engine else os.environ.get(env_var, "")
    if value is None or str(value).strip().lower() in ("", "none", "null"):
        return None
    normalized = str(value).strip().lower()
    if normalized not in ("0", "1", "false", "true"):
        raise ValueError(f"Invalid boolean for {key}: {value!r}")
    return normalized in ("1", "true")


def serialization_name():
    """Return the active serialization backend name."""
    return "msgpack" if _HAS_MSGPACK else "json"


def require_prefix_caching_disabled(enabled):
    """Reject cross-query KV reuse, which MLPerf explicitly disallows."""
    if enabled:
        raise ValueError(
            "Automatic prefix caching is disallowed for MLPerf: every input "
            "query must be computed in its entirety. See "
            f"{MLPERF_PREFIX_CACHING_RULE}"
        )
