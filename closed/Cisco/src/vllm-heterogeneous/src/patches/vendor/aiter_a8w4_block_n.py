
"""Apply MI350X-safe MoE a8w4 block_n cap to the installed aiter package.

This is an extension of the patch AMD ships with its MLPerf v6.0
GPT-OSS-120B submission:

    closed/AMD/setup/gpt-oss-120b/patches/gpt-oss-120b_aiter.patch
    pinned to aiter@6af8b6874

The stock ``aiter.ops.triton.moe_op_gemm_a8w4.get_kernel_config`` picks
``block_n=512`` for any ``block_m>32`` routing tile. On GPT-OSS-120B the
resulting Triton kernel asks for ~200 KB of shared memory, which exceeds
the MI350X / gfx950 160 KB LDS budget and aborts ``profile_run`` with
``triton.runtime.errors.OutOfResources: Required: 204800``.

AMD's upstream patch shrinks ``block_n`` (halving it) only while the
launch grid is below 4096 tiles. That covers small-grid cases but does
NOT cover the profile_run shape (m = max_num_batched_tokens * top_k =
65536*4 routed tokens, n = 5760) where ``grid_m * grid_n`` is already
~24k and the loop never executes. The result is that the AMD patch alone
is insufficient on our installed aiter and the profile run still
overflows LDS.

Our additional cap closes that hole: when ``block_m >= 64`` we
unconditionally hold ``block_n <= 128``. With block_k=256, num_stages=2,
this gives an operand-tile LDS budget of:

    (block_m*block_k + block_k*block_n) * num_stages
        = (128*256 + 256*128) * 2
        = 131072 bytes < 163840 byte LDS limit (gfx950)

For ``block_m`` in {16, 32} the AMD upstream behaviour is preserved
verbatim, so small-batch decode shapes are unaffected. The trade-off is
slightly more grid_n iterations for very-large prefill batches; this is
the price of running on a non-AMD-prebuilt image.

Sources:
    https://github.com/mlcommons<submission-root>_results_v6.0/blob/main/closed/AMD/setup/gpt-oss-120b/patches/gpt-oss-120b_aiter.patch
    https://github.com/ROCm/aiter/blob/6af8b6874/aiter/ops/triton/moe_op_gemm_a8w4.py

The patch is idempotent: the marker comment prevents reapplication, and a
``.orig_amd_a8w4`` backup is created on first run.

Apply by importing this module or running it as a script:

    python3 src/patches/vendor/aiter_a8w4_block_n.py
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

PATCH_MARKER = "# [MLPerf v6.0 AMD GPT-OSS AITER patch + MI350X LDS cap v2]"

PATCH_BLOCK = """
        # [MLPerf v6.0 AMD GPT-OSS AITER patch + MI350X LDS cap v2]
        #
        # Upstream AMD shrink loop (gpt-oss-120b_aiter.patch @ aiter@6af8b6874):
        # only halves block_n when the launch grid would otherwise be smaller
        # than 4096 tiles. Covers small-batch cases.
        grid_m = routing_data.n_blocks(m, block_m)
        grid_n = triton.cdiv(n, block_n)
        grid = grid_m * grid_n * split_k
        while block_n > 256 and grid < 4096:
            block_n = block_n // 2
            grid_n = triton.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k
        # Additional MI350X / gfx950 LDS safety cap (NOT in AMD's upstream
        # patch). Required because the AMD shrink loop above does not trigger
        # for large-grid shapes such as profile_run with
        # max_num_batched_tokens=65536, where block_m=128, block_n=512,
        # num_stages=2 needs ~200 KB of shared memory and exceeds the 160 KB
        # LDS budget on gfx950.
        #
        # When we shrink block_n to 128 we also need to drop num_warps from 8
        # to 4 to match what AMD's get_kernel_config uses elsewhere whenever
        # block_n == 128 (see block_m==16 and block_m==32/n<=1024 branches).
        # Keeping num_warps=8 with a 128x128 output tile aborts kernel launch
        # on HIP with "Triton Error [HIP]:  Code: 1, Messsage: invalid
        # argument" because the per-warp mfma layout becomes invalid.
        if block_m >= 64 and block_n > 128:
            block_n = 128
            num_warps = 4
"""



ANCHOR = (
    "    else:\n"
    "        block_n = 512\n"
    "        num_warps = 8\n"
)


def _resolve_target() -> Path | None:
    """Locate the installed aiter moe_op_gemm_a8w4 module on disk."""
    try:
        spec = importlib.util.find_spec("aiter.ops.triton.moe_op_gemm_a8w4")
    except (ModuleNotFoundError, ImportError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin)





LEGACY_PATCH_MARKERS = (
    "# [MLPerf v6.0 AMD GPT-OSS AITER patch]",
    "# [MLPerf v6.0 AMD GPT-OSS AITER patch + MI350X LDS cap]",
)


def apply() -> bool:
    """Apply the patch in place. Returns True if a change was made."""
    target = _resolve_target()
    if target is None:
        print(
            "WARNING: aiter.ops.triton.moe_op_gemm_a8w4 not installed; "
            "skipping AMD GPT-OSS AITER patch.",
            file=sys.stderr,
        )
        return False

    text = target.read_text()
    if PATCH_MARKER in text:
        return False

    backup = target.with_suffix(target.suffix + ".orig_amd_a8w4")

    
    
    
    has_legacy = any(marker in text for marker in LEGACY_PATCH_MARKERS)
    if has_legacy:
        if backup.exists():
            text = backup.read_text()
            target.write_text(text)
            print(
                f"Reverted previous AITER patch revision in {target} before "
                "re-applying."
            )
        else:
            print(
                f"WARNING: previous AITER patch present in {target} but no "
                ".orig_amd_a8w4 backup found; refusing to re-patch to avoid "
                "stacking edits.",
                file=sys.stderr,
            )
            return False

    if ANCHOR not in text:
        print(
            f"WARNING: anchor not found in {target}; aiter API may have "
            "changed. Skipping AITER patch.",
            file=sys.stderr,
        )
        return False

    if not backup.exists():
        shutil.copy2(target, backup)

    patched = text.replace(ANCHOR, ANCHOR + PATCH_BLOCK, 1)
    tmp = target.with_suffix(target.suffix + ".amd_a8w4_tmp")
    tmp.write_text(patched)
    os.replace(tmp, target)

    
    pycache = target.parent / "__pycache__"
    if pycache.is_dir():
        for f in pycache.glob(target.stem + ".*"):
            try:
                f.unlink()
            except OSError:
                pass

    print(
        f"Applied AITER moe_a8w4 block_n patch (AMD upstream + MI350X LDS "
        f"cap) to {target}"
    )
    return True


if __name__ == "__main__":
    apply()
