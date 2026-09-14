from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
try:
    from vllm.inputs import TokenInputs
    _make_token_input = lambda token_ids: TokenInputs(type="token", prompt_token_ids=token_ids)
except ImportError:
    try:
        from vllm.inputs import TokensInput
        _make_token_input = lambda token_ids: TokensInput(prompt_token_ids=token_ids)
    except ImportError:
        _make_token_input = lambda token_ids: {"prompt_token_ids": token_ids}

import logging
import multiprocessing as mp
from multiprocessing import connection as conn
import os, gc, asyncio, time

import harness_llm.common.numa_helpers as nh
from harness_llm.common.rpd_trace_utils import rpd_trace_range, rpd_trace_range_non_timed, ENABLE_TRACING_RPD
import harness_llm.backends.common.constants as constants
import harness_llm.backends.vllm.vllm_utils as utils

try:
    from ray.util.queue import Queue as RayQueue
except ImportError:
    RayQueue = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

# flags to notify the processes.

HARNESS_GC_LIMIT = int(os.getenv('HARNESS_GC_LIMIT', 0))


def _torch_profile_settings():
    """Read the vLLM torch-profiler window config from the environment.

    Enabled by ENABLE_TORCH_PROFILE=1 in run.sh, which exports
    VLLM_TORCH_PROFILER_DIR (+ optional VLLM_PROFILE_DEVICES / _DELAY_SEC /
    _DURATION_SEC). Returns (dir, device_set, delay_sec, duration_sec).
    """
    profile_dir = os.getenv("VLLM_TORCH_PROFILER_DIR", "")
    devs = os.getenv("VLLM_PROFILE_DEVICES", "0")
    try:
        dev_set = {int(x) for x in devs.replace(",", " ").split()}
    except ValueError:
        dev_set = {0}
    delay = float(os.getenv("VLLM_PROFILE_DELAY_SEC", "30"))
    duration = float(os.getenv("VLLM_PROFILE_DURATION_SEC", "60"))
    # When >0 (async offline only), the capture is anchored to the decode tail:
    # it starts once this fraction of the shard's samples have finished, instead
    # of the wall-clock delay. e.g. 0.8 -> start after 80% of samples completed.
    start_frac = float(os.getenv("VLLM_PROFILE_START_FRAC", "0"))
    return profile_dir, dev_set, delay, duration, start_frac

def _build_profiler_config():
    """Build a vLLM ProfilerConfig when torch profiling is enabled.

    In vLLM 0.22.0 the torch profiler is created from the engine's
    profiler_config (not the VLLM_TORCH_PROFILER_DIR env var alone), and
    start_profile() raises if no profiler_config is attached. Returns None when
    profiling is disabled.
    """
    profile_dir = os.getenv("VLLM_TORCH_PROFILER_DIR", "")
    if not profile_dir:
        return None
    try:
        from vllm.config import ProfilerConfig
        return ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=profile_dir,
            torch_profiler_with_stack=False,
            torch_profiler_record_shapes=False,
            torch_profiler_use_gzip=True,
        )
    except Exception as e:
        log.error(f"Could not build ProfilerConfig, disabling profiling: {e}")
        return None


def _run_fused_mxfp4_warmup(llm_config: dict, device_label: str = ""):
    """Warm-up-coverage guard: precompile the fused MXFP4 Triton kernels for every
    decode cudagraph size (+ a power-of-two prefill-shape sweep) into this device's
    on-disk Triton cache BEFORE the engine starts, so kernels don't JIT on the timed
    hot path. Fixes the rotating-idle-GPU (0<->100% util) sawtooth + throughput loss
    on cold containers (fresh Triton cache). The compiled kernels are content-hashed
    on disk, so the EngineCore subprocess reuses them. Runs while VRAM is still free
    (before engine allocation) so it can't OOM against gpu_memory_utilization.
    Default ON; disable with HARNESS_WARMUP_FUSED_MXFP4_KERNELS=0 in YAML env_config.
    Safe no-op on images without the fused aiter kernels.
    """
    try:
        from harness_llm.backends.common.warmup_fused_mxfp4 import warmup_fused_mxfp4_kernels
    except Exception as e:  # noqa: BLE001
        log.warning(f"[warmup-guard{device_label}] import failed, skipping: {e}")
        return
    cc = llm_config.get("compilation_config", {}) or {}
    if isinstance(cc, dict):
        sizes = list(cc.get("cudagraph_capture_sizes", []) or [])
        if not sizes and "cudagraph_capture_range" in cc:
            for item in cc.get("cudagraph_capture_range", []):
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    sizes.extend(range(*item))
                else:
                    sizes.append(item)
    else:  # already a CompilationConfig object (server/interactive path)
        sizes = list(getattr(cc, "cudagraph_capture_sizes", []) or [])
    # Extended prefill-shape coverage: prefill runs eager (not cudagraph-captured),
    # so its fused-kernel variants (keyed by token count M) otherwise JIT at runtime.
    # Warm power-of-two M up to max_num_batched_tokens (heuristics bucket by ~pow2).
    mnbt = int(llm_config.get("max_num_batched_tokens", 0) or 0)
    m = 256
    while m < mnbt:
        sizes.append(m)
        m *= 2
    if mnbt:
        sizes.append(mnbt)
    model_path = llm_config.get("model", "")
    tp = int(llm_config.get("tensor_parallel_size", 1) or 1)
    warmup_fused_mxfp4_kernels(sizes, model_path, tp_size=tp, device_label=device_label)


def _run_merge_attn_states_warmup(llm_config: dict, device_label: str = ""):
    """Precompile every ``merge_attn_states_kernel`` specialization (the chunked-
    prefill attention-state merge) into this device's Triton cache BEFORE the
    engine starts. The kernel takes ``prefill_tokens_with_context`` as a
    tl.constexpr, so a distinct kernel is compiled per prefill token-count; in
    Offline that is ~1.5k values that otherwise JIT on the timed hot path and
    cause a rotating-straggler GPU. See warmup_merge_attn_states for details.
    Default ON; disable with HARNESS_WARMUP_MERGE_ATTN_KERNELS=0 in env_config.
    """
    cap = int(llm_config.get("max_model_len", 0) or 0)
    model_path = llm_config.get("model", "")
    if cap <= 0 or not model_path:
        return
    # Run in an isolated subprocess: the compile CUDA context must be released
    # (process exit) before the engine allocates VRAM at gpu_memory_utilization
    # ~0.97, otherwise the residual parent footprint OOMs EngineCore startup.
    # Compiled kernels persist in the inherited per-device TRITON_CACHE_DIR.
    import subprocess
    import sys
    try:
        subprocess.run(
            [sys.executable, "-m",
             "harness_llm.backends.common.warmup_merge_attn_states",
             str(cap), model_path, device_label],
            env=os.environ.copy(),
            check=False,
            timeout=int(os.getenv("HARNESS_MERGE_ATTN_TIMEOUT_S", "1200")),
        )
    except Exception as e:  # noqa: BLE001
        log.warning(f"[merge-warmup{device_label}] subprocess failed, skipping: {e}")


def create_engine(llm_config: dict, async_engine=False):
    llm_config = utils.validate_and_correct(utils.populate_compile_config(llm_config))
    log.info(f"{llm_config=}")
    profiler_config = _build_profiler_config()
    if async_engine:
        # AsyncEngineArgs does not accept `swap_space` (it is an LLM/EngineArgs
        # field only), so drop it here as the async server path already does.
        async_engine_config = {k: v for k, v in llm_config.items() if k != "swap_space"}
        if profiler_config is not None:
            async_engine_config["profiler_config"] = profiler_config
            log.info("ProfilerConfig attached (torch) to async engine")
        engine_args = AsyncEngineArgs(**async_engine_config)
        return AsyncLLMEngine.from_engine_args(
            engine_args = engine_args,
            start_engine_loop = True
        )
    if profiler_config is not None:
        llm_config = {**llm_config, "profiler_config": profiler_config}
        log.info("ProfilerConfig attached (torch) to LLM engine")
    return LLM(**llm_config)

@rpd_trace_range("SUT:Worker")
def _run_vllm(llm, prompt_token_ids, sampling_params):
    return llm.generate(
        prompts = [_make_token_input(token_ids) for token_ids in prompt_token_ids],
        sampling_params=sampling_params,
        use_tqdm=False if os.getenv("HARNESS_DISABLE_VLLM_LOGS", "0") == "1" else True,
    )

@rpd_trace_range_non_timed("SUT:Worker")
def initialize_engine_and_generate(
    device_ids: tuple[int, ...],
    qdata_in: conn.Connection,
    qdata_out: conn.Connection,
    qstatus_out: mp.Queue,
    llm_config: dict,
    sampling_params_config: dict = {"temperature": 0.0, "max_tokens": 1024},
    engine_version: str = "sync",
    benchmark: str = "",
):
    """
    Initialize the llm engine and generate the responses.
    """
    use_async_engine = (engine_version == "async")

    for id in device_ids:
        nh.set_affinity_by_device(int(id))

    # Initialize the vllm engine.
    # All cache dirs are set per-device to avoid file lock contention across the 8 GPU processes.
    dev = str(device_ids[0])
    home = os.path.expanduser("~")
    os.environ["VLLM_CACHE_ROOT"] = utils.generate_vllm_cache_dir(dev)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = utils.generate_torch_inductor_cache_dir(dev)
    os.environ["TRITON_CACHE_DIR"] = f"{home}/.triton/cache_{dev}"
    os.environ["MIOPEN_USER_DB_PATH"] = f"{home}/.config/miopen_{dev}"
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = f"{home}/.config/miopen_{dev}"
    os.environ["PYTORCH_KERNEL_CACHE_PATH"] = f"{home}/.cache/torch/kernels_{dev}"
    if os.getenv("HARNESS_WARMUP_FUSED_MXFP4_KERNELS", "1") != "0":
        _run_fused_mxfp4_warmup(llm_config, device_label=f"[dev{dev}]")
    if os.getenv("HARNESS_WARMUP_MERGE_ATTN_KERNELS", "1") != "0":
        _run_merge_attn_states_warmup(llm_config, device_label=f"[dev{dev}]")
    llm = create_engine(llm_config = llm_config, async_engine = use_async_engine)

    qstatus_out.put(constants.HarnessStates.LLM_MODEL_LOAD_DONE)

    generate(device_ids, qdata_in, qdata_out, use_async_engine, llm, sampling_params_config)

@rpd_trace_range("SUT:Worker")
def generate(
        device_ids,
        data_in,
        data_output,
        use_async_engine,
        llm,
        sampling_params_config,
):
    if isinstance(data_in, conn.Connection):
        input_fn = data_in.recv
    elif RayQueue is not None and isinstance(data_in, RayQueue):
        input_fn = data_in.get
    else:
        raise AttributeError(
            "The data_in object must be a known type (conn.Connection, RayQueue)"
        )

    event_loop = asyncio.new_event_loop() if use_async_engine else None

    # Torch profiler window (ENABLE_TORCH_PROFILE=1 -> VLLM_TORCH_PROFILER_DIR).
    # Offline submits each engine's whole shard as a single blocking batch. For
    # the async engine the capture window runs as an asyncio task *on the same
    # event loop* as generation (see _generate_batch_with_profile): it waits for
    # the decode tail (VLLM_PROFILE_START_FRAC) or a fixed delay, calls
    # start_profile, sleeps for the duration, then stop_profile. Driving start/
    # stop from the loop thread (never a background OS thread) keeps roctracer's
    # start/stop on the same thread, and a finally-clause guarantees stop_profile
    # runs while the engine core is still alive -- so the trace is always flushed
    # and the engine never hangs on teardown. vLLM exports the trace to
    # VLLM_TORCH_PROFILER_DIR on stop. The sync LLM path is not windowed.
    prof_dir, prof_dev_set, prof_delay, prof_dur, prof_start_frac = _torch_profile_settings()
    prof_enabled = bool(prof_dir) and int(device_ids[0]) in prof_dev_set
    prof_total = [0]                     # number of samples in the real batch
    prof_completed = [0]                 # samples finished so far (async path)

    def _on_sample_done():
        prof_completed[0] += 1

    # The GC is going to be called after certain number of steps
    sample_count = 0
    is_gc_limit_specified = HARNESS_GC_LIMIT > 0
    if is_gc_limit_specified:
        gc.collect()
        gc.disable()

    if ENABLE_TRACING_RPD:
        llm.start_profile()
    if prof_enabled:
        if use_async_engine:
            log.info(
                f"Torch profiler ENABLED on device {device_ids[0]}: "
                f"start_frac={prof_start_frac} delay={prof_delay}s "
                f"duration={prof_dur}s dir={prof_dir}"
            )
        else:
            log.warning(
                "Torch profiler windowing is only supported on the async engine; "
                "disabling for the sync LLM path."
            )
            prof_enabled = False
    # Generates the completions for the input prompt tokens.
    while True:
        try:
            item = input_fn()
            if item is None:
                log.info(f"LLM is stopping")
                if use_async_engine:
                    _shutdown_async_engine(llm, event_loop)
                    del llm
                data_output.put(constants.HarnessStates.LLM_GENERATION_DONE)
                break

            start, end, prompt_token_ids, stop_token_ids = item
            is_real_batch = end is not None
            profile_this_batch = prof_enabled and is_real_batch
            if profile_this_batch:
                # First real (non-warmup) batch -> arm the profiler window.
                prof_total[0] = len(prompt_token_ids)
            sample_count += len(prompt_token_ids)
            if is_gc_limit_specified and sample_count >= HARNESS_GC_LIMIT:
                gc.collect()
                sample_count = 0

            sampling_params_list = []
            if stop_token_ids:
                for stop_seq_ids in stop_token_ids:
                    sampling_param = SamplingParams(**sampling_params_config)
                    sampling_param.stop_seq_ids = tuple(stop_seq_ids)
                    sampling_params_list.append(sampling_param)

            # Per-GPU timing: brackets one whole Offline shard (real batch) so the
            # per-device wall time / throughput can be compared across GPUs to tell
            # a data-imbalance straggler apart from a hardware/system-slow GPU.
            gen_start_t = time.time() if is_real_batch else None

            pred_output_tokens = None
            if use_async_engine:
                profile = None
                if profile_this_batch:
                    profile = {
                        "start_frac": prof_start_frac,
                        "delay": prof_delay,
                        "duration": prof_dur,
                        "total": prof_total,
                        "completed": prof_completed,
                        "dir": prof_dir,
                    }
                pred_output_tokens = _run_async_vllm(
                    engine=llm,
                    event_loop=event_loop,
                    sampling_params=sampling_params_list if sampling_params_list else SamplingParams(**sampling_params_config),
                    start=start,
                    prompt_token_ids=prompt_token_ids,
                    on_done=(_on_sample_done if profile_this_batch else None),
                    profile=profile,
                )
            else:
                pred_output_tokens = _run_vllm(llm, prompt_token_ids, sampling_params_list if sampling_params_list else SamplingParams(**sampling_params_config))

            if ENABLE_TRACING_RPD:
                llm.stop_profile()
            log.info(f"VLLM finished")

            processed_output = [
                output.outputs[0].token_ids for output in pred_output_tokens
            ]
            log.info(f"output tokens collected")

            data_output.put((start, end, processed_output))
            log.info(f"Processed output | start, end = {start}, {end}")

            if gen_start_t is not None:
                elapsed = time.time() - gen_start_t
                n_samples = len(prompt_token_ids)
                in_tokens = sum(len(p) for p in prompt_token_ids)
                out_tokens = sum(len(o) for o in processed_output)
                rate = n_samples / elapsed if elapsed > 0 else 0.0
                log.info(
                    f"[LB] device {device_ids[0]} shard done: "
                    f"samples={n_samples} in_tokens={in_tokens} out_tokens={out_tokens} "
                    f"wall={elapsed:.2f}s throughput={rate:.1f} samples/s"
                )
        except:
            logging.exception("Exception running vLLM")
            break
    log.info(f"vLLM engine thread finished for {device_ids=}")


def _shutdown_async_engine(engine, event_loop):
    """Gracefully tear down the v1 AsyncLLM engine and its event loop.

    Unlike the server path, the offline path drives the asyncio loop on the main
    thread, so its background tasks (AsyncLLM `output_handler`, the engine-core
    `process_outputs_socket` task) must be cancelled and awaited before the loop
    is closed. Otherwise they are finalized at interpreter exit on a closed loop,
    producing the "Task was destroyed but it is pending" / "Event loop is closed"
    errors.
    """
    # Stop the engine core / background output handler if the engine exposes it.
    try:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        log.exception("Error during async engine shutdown")

    if event_loop is None or event_loop.is_closed():
        return

    # Cancel and drain any tasks still pending on the loop, then close it.
    try:
        pending = asyncio.all_tasks(loop=event_loop)
        for task in pending:
            task.cancel()
        if pending:
            event_loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        event_loop.run_until_complete(event_loop.shutdown_asyncgens())
    except Exception:
        log.exception("Error draining async event loop during shutdown")
    finally:
        event_loop.close()


def _run_async_vllm(engine, event_loop, sampling_params, start, prompt_token_ids, on_done=None, profile=None):
    if profile is not None:
        coro = _generate_batch_with_profile(
            engine, sampling_params, start, prompt_token_ids, on_done, profile
        )
    else:
        coro = _async_generate_batch(engine, sampling_params, start, prompt_token_ids, on_done)
    return event_loop.run_until_complete(coro)


async def _profile_window(engine, profile, state):
    """Open the torch-profiler capture window on the generation event loop.

    Runs as a task alongside _async_generate_batch, so start_profile/stop_profile
    execute on the loop thread (satisfying roctracer's same-thread requirement).
    """
    start_frac = profile["start_frac"]
    delay = profile["delay"]
    duration = profile["duration"]
    total = profile["total"]
    completed = profile["completed"]
    try:
        if start_frac > 0 and total[0] > 0:
            target = start_frac * total[0]
            log.info(
                f"Torch profiler waiting for decode tail: {start_frac:.2f} of "
                f"{total[0]} samples (={target:.0f}) to complete"
            )
            while completed[0] < target:
                await asyncio.sleep(0.5)
        elif delay > 0:
            await asyncio.sleep(delay)
        log.info(
            f"Starting torch profiler capture window "
            f"(completed={completed[0]}/{total[0]})"
        )
        await engine.start_profile()
        state["started"] = True
        await asyncio.sleep(duration)
        await engine.stop_profile()
        state["stopped"] = True
        log.info(f"Torch profiler trace written under {profile['dir']}")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.error(f"Torch profiler window failed: {e}")


async def _generate_batch_with_profile(engine, sampling_params, start, prompt_token_ids, on_done, profile):
    state = {"started": False, "stopped": False}
    prof_task = asyncio.create_task(_profile_window(engine, profile, state))
    try:
        return await _async_generate_batch(engine, sampling_params, start, prompt_token_ids, on_done)
    finally:
        # Generation finished. Guarantee the profiler is stopped *now*, while the
        # engine core is still alive to service the stop RPC and flush the trace.
        if state["started"] and not state["stopped"]:
            try:
                await engine.stop_profile()
                state["stopped"] = True
                log.info(
                    f"Torch profiler trace written under {profile['dir']} "
                    "(stopped at generation end)"
                )
            except Exception as e:
                log.error(f"Torch profiler stop at generation end failed: {e}")
        prof_task.cancel()
        try:
            await prof_task
        except asyncio.CancelledError:
            pass


async def _async_generate_batch(engine, sampling_params, start, prompt_token_ids, on_done=None):
    tasks = []
    for i in range(len(prompt_token_ids)):
        tasks.append(_async_generate(engine,
                                     sampling_params,
                                     str(start + i),
                                     prompt_token_ids[i],
                                     on_done)
        )
    return await asyncio.gather(*tasks)


async def _async_generate(engine, sampling_params, sample_id, prompt_token_ids, on_done=None):
    results_generator = engine.generate(
        {"prompt_token_ids": prompt_token_ids},
        sampling_params,
        sample_id
    )
    final_request_output = None
    async for request_output in results_generator:
        final_request_output = request_output
    if on_done is not None:
        on_done()
    return final_request_output
