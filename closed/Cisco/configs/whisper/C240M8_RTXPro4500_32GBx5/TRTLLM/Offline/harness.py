import nv_mlpinf.common.constants as C
import nv_mlpinf.common.paths as paths
import nv_mlpinf.fields.harness as harness_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.llmlib.fields as llm_fields
import os
from importlib import import_module
from nvmitten.constants import Precision

whisper_fields = import_module("nv_mlpinf.benchmarks.whisper.fields")

# GPU0/GPU1 are local to NUMA 0; GPU2/GPU3 are local to NUMA 1.
# Give each in-process Whisper GPU worker a disjoint set of local physical
# cores and their SMT siblings.
os.environ.setdefault(
    "WHISPER_GPU_CPU_AFFINITY",
    "0=0-42,172-214;1=43-85,215-257;"
    "2=86-128,258-300;3=129-171,301-343",
)

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: 130,
        model_fields.input_dtype: Precision.FP32,
        llm_fields.llm_gen_config_path: "src/nv_mlpinf/benchmarks/whisper/generation_config.json",
        loadgen_fields.offline_expected_qps: 265.3,
        model_fields.precision: Precision.FP16,
        harness_fields.tensor_path: paths.PREPROCESSED_DATA_DIR / "whisper-large-v3/",
        llm_fields.tensor_parallelism: 1,
        llm_fields.pipeline_parallelism: 1,

        whisper_fields.whisper_encoder_build_flags: {
            "max_beam_width": 1,
            "max_batch_size": 130,
            "kv_cache_type": "paged",
            "remove_input_padding": "enable",
            "moe_plugin": "disable",
            "gemm_plugin": "disable",
            "max_input_len": 3000,
            "max_seq_len": 3000,
            "bert_attention_plugin": "float16",
        },
        whisper_fields.whisper_decoder_build_flags: {
            "max_beam_width": 1,
            "max_batch_size": 130,
            "kv_cache_type": "paged",
            "remove_input_padding": "enable",
            "moe_plugin": "disable",
            "max_input_len": 14,
            "max_seq_len": 174,
            "max_encoder_input_len": 3000,
            "gpt_attention_plugin": "float16",
            "gemm_plugin": "float16",
        },
        llm_fields.trtllm_runtime_flags: {
            "exclude_input_from_output": True,
            "use_inflight_batching": False,
            "max_num_tokens": 3000,
            "enable_chunked_context": False,
        },
        harness_fields.use_graphs: False,
        llm_fields.use_token_latencies: False,
        harness_fields.vboost_slider: 1,
    }
}
