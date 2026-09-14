import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.llmlib.fields as llm_fields

# Used in single-node benchmark with docker environment only

ifb_config = {
    llm_fields.trtllm_yml_override: paths.PROJECT_BASE_DIR / 'configs/llama2_70b/HPE_PROLIANT_DL380A_RTX_PRO_6000_PCIE_96GBx8/TRTLLM/Offline/trtllm-serve-ifb-1gpu.yaml',
    llm_fields.env_yml_override: paths.PROJECT_BASE_DIR / 'configs/llama2_70b/HPE_PROLIANT_DL380A_RTX_PRO_6000_PCIE_96GBx8/TRTLLM/Offline/trtllm-serve-ifb-1gpu-env.yaml',
    model_fields.precision: 'fp4',
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): ifb_config,
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): ifb_config,
}

# base = {
#     llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
#     harness_fields.tensor_path: 'build/preprocessed_data/llama2-70b/',

#     harness_fields.use_graphs: False,
#     llm_fields.use_token_latencies: True,
#     llm_fields.trtllm_build_flags: {
#         'max_beam_width': 1,
#         'kv_cache_type': 'paged',
#         'remove_input_padding': 'enable',
#         'multiple_profiles': 'enable',
#         'use_fused_mlp': 'enable',
#         'context_fmha': 'enable',
#         'max_num_tokens': 2048,
#         'max_input_len': 1024,
#         'max_seq_len': 2048,
#         'use_fp8_context_fmha': 'enable',
#         'use_paged_context_fmha': 'enable',
#         'tokens_per_block': 32,
#         'gemm_swiglu_plugin': 'disable',
#         'gpus_per_node': 8
#     },
#     llm_fields.trtllm_runtime_flags: {
#         'exclude_input_from_output': True,
#         'use_inflight_batching': True,
#         'max_num_tokens': 2048,
#         'batch_scheduler_policy': 'max_util',
#         'context_chunking_policy': 'first_come_first_served',
#         'kvcache_free_gpu_mem_frac': 0.90,
#         'enable_chunked_context': True,
#     },

#     llm_fields.trtllm_checkpoint_flags: {
#         'kv_cache_dtype': 'fp8',
#     },
#     model_fields.precision: 'fp4',
#     model_fields.input_dtype: 'int32',

#     model_fields.gpu_batch_size: {
#         'llama2-70b': 2048,
#     },
#     loadgen_fields.offline_expected_qps: 100,
#     loadgen_fields.min_duration: 1200000,

#     llm_fields.tensor_parallelism: 1,
#     llm_fields.pipeline_parallelism: 1,
    
#     harness_fields.vboost_slider: 1,
# }

# EXPORTS = {
#     C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base,
#     C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): base,
# }