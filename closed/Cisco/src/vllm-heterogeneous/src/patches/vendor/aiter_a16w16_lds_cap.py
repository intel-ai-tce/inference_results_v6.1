
"""Clamp aiter gfx950 A16W16 GEMM autotune configs to the MI350X LDS budget.

The aiter package ships several ``gfx950-*A16W16*.json`` tuning tables under
``aiter/ops/triton/configs/gemm/``. A number of the M-bucket entries pick
block tiles that exceed the gfx950 / MI350X 160 KB LDS budget when the
operand-tile double buffer is materialized, e.g.

    BM=64, BN=128, BK=256, num_stages=2  -> 196608 B  (A=32 KB, B=64 KB) * 2
    BM=128,BN=128, BK=256, num_stages=2  -> 262144 B  (A=64 KB, B=64 KB) * 2
    BM=256,BN=128, BK=128, num_stages=2  -> 196608 B  (A=64 KB, B=32 KB) * 2

vLLM hits these during cudagraph capture for GPT-OSS-120B on TP=8: the
unquantized router/lm_head projections go through ``rocm_unquantized_gemm``
-> ``aiter.ops.triton.gemm_a16w16.gemm_a16w16`` which calls ``_get_config``
which returns one of these oversize configs, and the Triton driver
aborts kernel load with::

    triton.runtime.errors.OutOfResources:
        out of resource: shared memory, Required: 196608, Hardware limit: 163840

This patch walks every shipped ``gfx950-*A16W16*.json`` (excluding the
mixed AFP4xA16 fused variants, whose LDS budget is dominated by the
FP4 weight tile and is *not* the simple 2-byte calculation here) and, for
every entry whose LDS budget exceeds the gfx950 limit, shrinks
``num_stages`` from 2 down to 1. With num_stages=1 the worst-case entry
(BM=128, BN=128, BK=256) lands at 131072 B (< 163840 B) which fits.

The shape of the kernel and the choice of block tile are otherwise
unchanged, so the autotune key still matches and the compiled kernel
remains correct -- we only lose K-loop pipelining. On gfx950 a single
LDS-resident operand tile is already the steady state for many of the
existing well-tuned aiter configs, so the throughput cost is minor and
strictly preferable to an unconditional crash.

The patch is idempotent: a marker key ``_mi350x_lds_cap`` is written into
each rewritten JSON, and the original file is preserved as
``<name>.orig_lds_cap``.

Apply by importing this module or running it as a script::

    python3 src/patches/vendor/aiter_a16w16_lds_cap.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from glob import glob
from pathlib import Path
from typing import Tuple


LDS_LIMIT_BYTES = 163840






ELEM_BYTES = 2

MARKER_KEY = "_mi350x_lds_cap"
MARKER_VALUE = "v1"

CONFIG_GLOB = "gfx950-*A16W16*.json"



EXCLUDE_SUBSTRINGS = ("AFP4WFP4-A16W16",)


def _find_config_dir() -> Path | None:
    """Locate aiter's gfx950 GEMM tuning config dir on disk."""
    try:
        import aiter  
        import aiter.ops.triton  
    except Exception:
        return None

    import aiter.ops.triton as _ops_triton

    triton_pkg_dir = Path(_ops_triton.__file__).parent
    cfg_dir = triton_pkg_dir / "configs" / "gemm"
    if not cfg_dir.is_dir():
        return None
    return cfg_dir


def _lds_bytes(cfg: dict) -> int:
    bm = cfg["BLOCK_SIZE_M"]
    bn = cfg["BLOCK_SIZE_N"]
    bk = cfg["BLOCK_SIZE_K"]
    ns = max(int(cfg.get("num_stages", 1)), 1)
    return (bm * bk + bk * bn) * ELEM_BYTES * ns


def _clamp_entry(cfg: dict) -> Tuple[bool, int, int]:
    """Reduce LDS usage in place. Returns (changed, before, after)."""
    if not isinstance(cfg, dict):
        return False, 0, 0
    if "BLOCK_SIZE_M" not in cfg or "BLOCK_SIZE_N" not in cfg or "BLOCK_SIZE_K" not in cfg:
        return False, 0, 0

    before = _lds_bytes(cfg)
    if before <= LDS_LIMIT_BYTES:
        return False, before, before

    
    if int(cfg.get("num_stages", 1)) > 1:
        cfg["num_stages"] = 1

    after = _lds_bytes(cfg)
    while after > LDS_LIMIT_BYTES and cfg["BLOCK_SIZE_K"] > 16:
        cfg["BLOCK_SIZE_K"] = cfg["BLOCK_SIZE_K"] // 2
        after = _lds_bytes(cfg)

    return True, before, after


def _patch_file(path: Path) -> bool:
    try:
        with path.open() as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"  skip {path.name}: cannot parse ({exc})", file=sys.stderr)
        return False

    if not isinstance(data, dict):
        return False
    if data.get(MARKER_KEY) == MARKER_VALUE:
        return False

    changes: list[str] = []
    for key, cfg in data.items():
        if key.startswith("_"):
            continue
        changed, before, after = _clamp_entry(cfg)
        if changed:
            changes.append(f"{key}: {before}B -> {after}B")

    if not changes:
        data[MARKER_KEY] = MARKER_VALUE
        backup = path.with_suffix(path.suffix + ".orig_lds_cap")
        if not backup.exists():
            shutil.copy2(path, backup)
        tmp = path.with_suffix(path.suffix + ".lds_cap_tmp")
        with tmp.open("w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
        return False

    data[MARKER_KEY] = MARKER_VALUE

    backup = path.with_suffix(path.suffix + ".orig_lds_cap")
    if not backup.exists():
        shutil.copy2(path, backup)

    tmp = path.with_suffix(path.suffix + ".lds_cap_tmp")
    with tmp.open("w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)

    print(f"  patched {path.name}: " + "; ".join(changes))
    return True


def apply() -> bool:
    cfg_dir = _find_config_dir()
    if cfg_dir is None:
        print(
            "WARNING: aiter triton GEMM config directory not found; "
            "skipping MI350X A16W16 LDS cap.",
            file=sys.stderr,
        )
        return False

    patched_any = False
    paths = sorted(Path(p) for p in glob(str(cfg_dir / CONFIG_GLOB)))
    for path in paths:
        if any(s in path.name for s in EXCLUDE_SUBSTRINGS):
            continue
        if _patch_file(path):
            patched_any = True

    if patched_any:
        print(f"Applied MI350X A16W16 LDS cap to configs under {cfg_dir}")
    return patched_any


if __name__ == "__main__":
    apply()
