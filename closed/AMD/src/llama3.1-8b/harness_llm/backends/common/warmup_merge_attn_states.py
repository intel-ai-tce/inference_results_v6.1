"""Warm-up/preload for vLLM's ``merge_attn_states_kernel`` (chunked-prefill
attention-state merge).

Why this exists
---------------
``vllm/v1/attention/ops/triton_merge_attn_states.py`` declares
``prefill_tokens_with_context: tl.constexpr`` on the @triton.jit kernel. In the
ROCm aiter FlashAttention backend (``rocm_aiter_fa``) the two call sites pass no
explicit value, so it defaults to ``num_tokens`` = the number of prefill query
tokens in that scheduler step. Because it is a ``tl.constexpr``, Triton compiles
a *separate* kernel specialization for every distinct value.

In Offline that value takes ~1.5k distinct values (measured: densely spanning
2..~2420, i.e. essentially every integer up to the max prompt length), because
the whole 16.7k-sample shard is submitted at once and chunked-prefill steps vary
in size. Those specializations otherwise JIT lazily *during the timed run* --
invisible to vLLM's jit_monitor (it dedups warnings by kernel name) -- stalling
whichever engine hits an uncompiled shape and producing a rotating-straggler GPU
that gates the whole run (Offline time = slowest engine). Server/Interactive are
low-concurrency streams that hit only a handful of values, so they never showed
it.

We CANNOT de-constexpr the kernel: lowering the arg to a runtime value hits a
ROCm/Triton codegen path that computes wrong attention merges (ROUGE collapses).
So instead we keep the exact (numerically-correct) kernel and simply precompile
every specialization into the on-disk Triton cache up front, on the engine's own
device, before LoadGen timing starts. The dummy tensors only drive compilation;
their contents are irrelevant, so this is data-independent (MLPerf-legal).

Enabled by default. Disable with HARNESS_WARMUP_MERGE_ATTN_KERNELS=0 (YAML
env_config). Sweep upper bound defaults to the engine's max_model_len and can be
overridden with HARNESS_MERGE_ATTN_MAX_TOKENS. Compile parallelism is set with
HARNESS_MERGE_ATTN_THREADS (default 8; independent compiles, atomic cache
writes). Safe no-op if the kernel/module is unavailable.
"""

import json
import os

import harness_llm.common.logging as logger

log = logger.get_logger(__name__)

ENABLE_FLAG = "HARNESS_WARMUP_MERGE_ATTN_KERNELS"
MAX_TOKENS_FLAG = "HARNESS_MERGE_ATTN_MAX_TOKENS"
THREADS_FLAG = "HARNESS_MERGE_ATTN_THREADS"


def _read_head_size(model_path: str):
    """Return the attention head dimension from the model config."""
    with open(os.path.join(model_path, "config.json"), "r") as f:
        cfg = json.load(f)
    text = cfg.get("text_config", cfg)
    head_dim = text.get("head_dim")
    if head_dim:
        return int(head_dim)
    return int(text["hidden_size"]) // int(text["num_attention_heads"])


def warmup_merge_attn_states_kernels(
    max_num_tokens: int,
    model_path: str,
    device_label: str = "",
):
    """Compile ``merge_attn_states_kernel`` for every prefill token-count
    specialization (1..max_num_tokens), for both OUTPUT_LSE variants, on cuda:0
    (the engine's physical GPU) so the on-disk Triton cache is warm before timing.
    """
    if os.environ.get(ENABLE_FLAG, "1") == "0":
        return

    if not model_path:
        log.warning(f"[merge-warmup{device_label}] no model path; skipping")
        return

    try:
        cap = int(os.environ.get(MAX_TOKENS_FLAG, max_num_tokens) or max_num_tokens)
    except (TypeError, ValueError):
        cap = int(max_num_tokens or 0)
    if cap <= 0:
        log.warning(f"[merge-warmup{device_label}] non-positive token cap ({cap}); skipping")
        return

    try:
        import torch
        from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states
    except Exception as e:  # noqa: BLE001
        log.warning(f"[merge-warmup{device_label}] kernel not importable ({e}); skipping")
        return

    if not torch.cuda.is_available():
        log.warning(f"[merge-warmup{device_label}] CUDA not available; skipping")
        return

    try:
        head_size = _read_head_size(model_path)
    except Exception as e:  # noqa: BLE001
        log.warning(f"[merge-warmup{device_label}] could not read head_size ({e}); skipping")
        return

    try:
        threads = int(os.environ.get(THREADS_FLAG, "8") or "8")
    except (TypeError, ValueError):
        threads = 8
    threads = max(1, threads)

    dev = torch.device("cuda:0")
    dt = torch.bfloat16
    nh = 1  # grid dim only; NOT part of the kernel specialization

    def _compile_one(v: int, with_lse: bool):
        # Contents are irrelevant -- we only need Triton to compile the
        # (HEAD_SIZE, OUTPUT_LSE, prefill_tokens_with_context=v, USE_FP8=False)
        # specialization. prefill_tokens_with_context defaults to num_tokens (v).
        out = torch.empty(v, nh, head_size, dtype=dt, device=dev)
        prefix_output = torch.zeros(v, nh, head_size, dtype=dt, device=dev)
        suffix_output = torch.zeros(v, nh, head_size, dtype=dt, device=dev)
        prefix_lse = torch.zeros(nh, v, dtype=torch.float32, device=dev)
        suffix_lse = torch.zeros(nh, v, dtype=torch.float32, device=dev)
        output_lse = torch.empty(nh, v, dtype=torch.float32, device=dev) if with_lse else None
        merge_attn_states(
            out, prefix_output, prefix_lse, suffix_output, suffix_lse,
            output_lse=output_lse,
        )

    # Both variants used by rocm_aiter_fa: the in-loop chunk merge writes the LSE
    # (OUTPUT_LSE=True), the final merge does not (OUTPUT_LSE=False).
    combos = (False, True)
    total = cap * len(combos)
    log.info(
        f"[merge-warmup{device_label}] precompiling merge_attn_states for "
        f"num_tokens=1..{cap} x {len(combos)} lse-variants ({total} kernels, "
        f"head_size={head_size}, threads={threads})"
    )

    tasks = [(v, combo) for combo in combos for v in range(1, cap + 1)]
    failures = [0]

    def _run(task):
        v, combo = task
        try:
            _compile_one(v, combo)
        except Exception as e:  # noqa: BLE001
            failures[0] += 1
            if failures[0] <= 5:
                log.warning(f"[merge-warmup{device_label}] v={v} lse={combo} failed: {e}")

    if threads == 1:
        for t in tasks:
            _run(t)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(_run, tasks))

    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
    log.info(
        f"[merge-warmup{device_label}] done "
        f"({total - failures[0]}/{total} compiled, {failures[0]} failures)"
    )


if __name__ == "__main__":
    # Run as an isolated subprocess so the CUDA context created for compilation is
    # fully released (process exit) before the engine allocates VRAM -- the engine
    # runs at gpu_memory_utilization ~0.97 and cannot tolerate a residual parent
    # footprint. The compiled kernels persist in TRITON_CACHE_DIR (inherited env).
    #   argv: <max_num_tokens> <model_path> [device_label]
    import sys

    _cap = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    _model = sys.argv[2] if len(sys.argv) > 2 else ""
    _label = sys.argv[3] if len(sys.argv) > 3 else ""
    warmup_merge_attn_states_kernels(_cap, _model, device_label=_label)
