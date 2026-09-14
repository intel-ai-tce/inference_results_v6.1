import nv_mlpinf.common.constants as C
import nv_mlpinf.llmlib.fields as llm_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.harness as harness_fields

import os
import nv_mlpinf.common.paths as paths

base = {
    llm_fields.llm_gen_config_path: 'src/nv_mlpinf/benchmarks/deepseek_r1/generation_config.json',
    harness_fields.tensor_path: paths.PREPROCESSED_DATA_DIR / 'deepseek-r1/',
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 26382,
    llm_fields.warmup_iterations: 0,
    llm_fields.use_token_latencies: True,
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    harness_fields.max_concurrency: 5120,
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    harness_fields.enable_metrics: False,

    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 4608,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.95,  # Progressively lower by 0.1/0.05 if you hit OOM errors.
        'enable_chunked_context': False,
        'max_concurrency': 10240,
        'cuda_graph_batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024],
        'cuda_graph_padding_enabled': True,
        'moe_backend': 'CUTEDSL',
        "adp_balancing_enable": True,
        # Setting lower bound and upper bound for iters to wait.
        "adp_balancing_batching_wait_iters": 3,
        "adp_balancing_timeout_iters": 9,
        "stream_interval": 20,
    },
    harness_fields.use_graphs: True,

    # Tune me! If you hit an OOM, decrease the batch size.
    model_fields.gpu_batch_size: {
        'deepseek-r1': 512,
    },
    # B300 has 1100W TGP vs GB300's 1400W, so ~0.79x performance per GPU
    # Same 8 GPUs, so expect similar throughput to GB300x8
    loadgen_fields.server_target_qps: 15,

    # Only supported on Hopper and Blackwell GPUs. On other GPUs, this will not do anything.
    harness_fields.vboost_slider: 1,

    llm_fields.server_instance_size: 8,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base,
}
