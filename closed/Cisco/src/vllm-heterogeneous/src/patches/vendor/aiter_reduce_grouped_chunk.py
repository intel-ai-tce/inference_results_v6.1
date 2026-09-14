
"""Chunk aiter ``reduce_grouped`` launches along ``num_groups`` for gfx950.

HIP rejects Triton kernel launches whose grid-Y dimension exceeds 65535 with::

    Triton Error [HIP]:  Code: 1, Messsage: invalid argument

The aiter ``reduce_grouped`` wrappers in ``moe_op_gemm_a8w4`` and
``moe_op_gemm_a8w8`` launch ``_reduce_grouped`` with::

    grid = (cdiv(N, BLOCK_N), num_groups)

where ``num_groups`` is either ``scatter_indx.shape[0]`` (output tokens after
MoE routing) or ``x.shape[-2]`` when ``indx is None``. Under vLLM internal
data parallelism (``data_parallel_size > 1``), batch coordination can pad token
counts so ``num_groups`` crosses the HIP limit even when
``max_num_batched_tokens`` is below 65536.

This patch rewrites ``reduce_grouped`` in both MoE GEMM modules to split large
``num_groups`` into multiple launches of at most 32768 groups. When ``indx``
is provided the index/output tensors are sliced per chunk but the full ``x``
tensor is retained (indices still point into the shared routed-token pool).
When ``indx is None`` both ``x`` and ``out`` are sliced along the group
dimension.

Apply by importing this module or running it as a script::

    python3 src/patches/vendor/aiter_reduce_grouped_chunk.py
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

PATCH_MARKER = "# [MI350X reduce_grouped grid_y chunk v1]"

TARGET_MODULES = (
    "aiter.ops.triton.moe_op_gemm_a8w4",
    "aiter.ops.triton.moe_op_gemm_a8w8",
)

ANCHOR = """\
    _reduce_grouped[(num_blocks, num_groups)](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),  #
        out,
        out.stride(0),
        out.stride(1),  #
        indx,  #
        x.shape[0],
        x.shape[-1],  #
        apply_swiglu,
        alpha,
        limit,
        reduction_n,
        BLOCK_N=BLOCK_N,
        EVEN_N=(x.shape[-1] % BLOCK_N == 0),
        K=K,  #
        num_warps=2,  #
    )
    return out"""

PATCH_BLOCK = """\
    # [MI350X reduce_grouped grid_y chunk v1]
    # HIP grid-Y must stay <= 65535 on gfx950. DP-padded MoE batches can
    # exceed that even when max_num_batched_tokens does not, so chunk along
    # num_groups instead of lowering mnbt and losing prefill throughput.
    _MAX_REDUCE_GROUPED_GRID_Y = 32768
    if num_groups <= _MAX_REDUCE_GROUPED_GRID_Y:
        _group_slices = ((0, num_groups),)
    else:
        _group_slices = tuple(
            (_g0, min(_g0 + _MAX_REDUCE_GROUPED_GRID_Y, num_groups))
            for _g0 in range(0, num_groups, _MAX_REDUCE_GROUPED_GRID_Y)
        )
    for _g0, _g1 in _group_slices:
        _ng = _g1 - _g0
        if indx is None:
            if x.ndim >= 3:
                _x_launch = x[..., _g0:_g1, :]
            else:
                _x_launch = x[_g0:_g1]
            _out_launch = out[_g0:_g1]
            _indx_launch = None
        else:
            _x_launch = x
            _out_launch = out[_g0:_g1]
            _indx_launch = indx[_g0:_g1]
        _reduce_grouped[(num_blocks, _ng)](
            _x_launch,
            _x_launch.stride(0),
            _x_launch.stride(1),
            _x_launch.stride(2),
            _out_launch,
            _out_launch.stride(0),
            _out_launch.stride(1),
            _indx_launch,
            _x_launch.shape[0],
            _x_launch.shape[-1],
            apply_swiglu,
            alpha,
            limit,
            reduction_n,
            BLOCK_N=BLOCK_N,
            EVEN_N=(x.shape[-1] % BLOCK_N == 0),
            K=K,
            num_warps=2,
        )
    return out"""


def _resolve_target(module_name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ModuleNotFoundError, ImportError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin)


def _patch_file(target: Path) -> bool:
    text = target.read_text()
    if PATCH_MARKER in text:
        return False

    if ANCHOR not in text:
        print(
            f"WARNING: reduce_grouped launch anchor not found in {target}; "
            "aiter API may have changed. Skipping.",
            file=sys.stderr,
        )
        return False

    backup = target.with_suffix(target.suffix + ".orig_reduce_chunk")
    if not backup.exists():
        shutil.copy2(target, backup)

    patched = text.replace(ANCHOR, PATCH_BLOCK, 1)
    tmp = target.with_suffix(target.suffix + ".reduce_chunk_tmp")
    tmp.write_text(patched)
    os.replace(tmp, target)

    pycache = target.parent / "__pycache__"
    if pycache.is_dir():
        for f in pycache.glob(target.stem + ".*"):
            try:
                f.unlink()
            except OSError:
                pass

    print(f"Applied reduce_grouped grid_y chunk patch to {target}")
    return True


def apply() -> bool:
    patched_any = False
    for module_name in TARGET_MODULES:
        target = _resolve_target(module_name)
        if target is None:
            print(
                f"WARNING: {module_name} not installed; skipping chunk patch.",
                file=sys.stderr,
            )
            continue
        if _patch_file(target):
            patched_any = True

    if patched_any:
        print("Applied MI350X reduce_grouped grid_y chunk patch.")
    return patched_any


if __name__ == "__main__":
    apply()
