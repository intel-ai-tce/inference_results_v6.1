import nv_mlpinf.common.constants as C
import nv_mlpinf.llmlib.fields as llm_fields
import nv_mlpinf.fields.models as model_fields
import nv_mlpinf.fields.loadgen as loadgen_fields
import nv_mlpinf.fields.harness as harness_fields

import os
import nv_mlpinf.common.paths as paths

num_servers = int(os.environ.get('DP_MULTIPLICITY', '18'))
total_gpus = int(os.environ.get('TOTAL_GPUS', '72'))
scale = total_gpus // 4  # relative to single node (4 GPUs)

base = {
    llm_fields.llm_gen_config_path: 'src/nv_mlpinf/benchmarks/deepseek_r1/generation_config.json',
    harness_fields.tensor_path: paths.PREPROCESSED_DATA_DIR / 'deepseek-r1/',

    loadgen_fields.min_duration: 2_000_000,
    loadgen_fields.min_query_count: 20 * 4388 * scale,
    llm_fields.warmup_iterations: 0,
    llm_fields.use_token_latencies: True,

    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',

    loadgen_fields.offline_expected_qps: 10 * scale,
    harness_fields.max_concurrency: 10240,
}


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base,
}
