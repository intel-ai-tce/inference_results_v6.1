import logging
import multiprocessing as mp
import os
from pathlib import Path

import mlperf_loadgen as lg

from config import HarnessCfg
from sut.base import SUT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SCENARIO_MAP = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
    "interactive": lg.TestScenario.Server,
}


V61_LOADGEN_SEEDS = {
    "qsl_rng_seed": 2085463073848966840,
    "sample_index_rng_seed": 2747215439041700203,
    "schedule_rng_seed": 16159082839903944936,
}


def _seed_assignments(config_path):
    assignments = {}
    with open(config_path, encoding="utf-8") as config_file:
        for line_number, raw_line in enumerate(config_file, 1):
            line = raw_line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, raw_value = (part.strip() for part in line.split("=", 1))
            seed_name = next(
                (name for name in V61_LOADGEN_SEEDS
                 if key.endswith(f".{name}")),
                None,
            )
            if seed_name is None:
                continue
            try:
                value = int(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {seed_name} in {config_path}:{line_number}: "
                    f"{raw_value!r}") from exc
            assignments.setdefault(seed_name, []).append(
                (line_number, value))
    return assignments


def _validate_seed_file(config_path, require_all):
    assignments = _seed_assignments(config_path)
    for seed_name, expected in V61_LOADGEN_SEEDS.items():
        values = assignments.get(seed_name, [])
        if require_all and not values:
            raise ValueError(
                f"MLPerf v6.1 {seed_name} is missing from {config_path}")
        stale = [(line, value) for line, value in values if value != expected]
        if stale:
            details = ", ".join(
                f"line {line}={value}" for line, value in stale)
            raise ValueError(
                f"Stale MLPerf seed in {config_path}: {seed_name} "
                f"must be {expected}; found {details}")


def _validate_effective_seeds(settings):
    for seed_name, expected in V61_LOADGEN_SEEDS.items():
        actual = getattr(settings, seed_name)
        if actual != expected:
            raise ValueError(
                f"Effective LoadGen {seed_name}={actual}; "
                f"MLPerf v6.1 requires {expected}")


def _write_effective_mlperf_conf(mlperf_conf_path, user_conf_path,
                                 output_log_dir):
    """Persist one LoadGen config, avoiding the multiple-config invalidation."""
    output_dir = Path(output_log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_conf_path = output_dir / "effective_mlperf.conf"
    official_conf = Path(mlperf_conf_path).read_text(encoding="utf-8").rstrip()
    user_conf = Path(user_conf_path).read_text(encoding="utf-8").rstrip()
    effective_conf_path.write_text(
        "# Generated from the pinned MLPerf v6.1 config and harness user config.\n"
        f"# official_source={mlperf_conf_path}\n"
        f"# user_source={user_conf_path}\n\n"
        f"{official_conf}\n\n{user_conf}\n",
        encoding="utf-8",
    )
    return str(effective_conf_path)


def _validate_speculative_scenario(conf):
    required = getattr(conf, "speculative_decoding_scenario", None)
    if required and conf.scenario.lower() != str(required).lower():
        raise ValueError(
            "Speculative decoding profile is restricted to "
            f"{required}; requested {conf.scenario}")


def _count_samples(dataset_path):
    """Auto-detect the number of samples in a dataset directory or file."""
    import numpy as np
    if os.path.isdir(dataset_path):
        for fname in sorted(os.listdir(dataset_path)):
            if fname.startswith("input_lens") and fname.endswith(".npy"):
                lengths = np.load(os.path.join(dataset_path, fname))
                return len(lengths)
            if fname.startswith("input_ids_padded") and fname.endswith(".npy"):
                ids = np.load(os.path.join(dataset_path, fname))
                return ids.shape[0]
    elif os.path.isfile(dataset_path):
        import pandas as pd
        if dataset_path.endswith(".parquet"):
            return len(pd.read_parquet(dataset_path))
        if dataset_path.endswith(".json"):
            return len(pd.read_json(dataset_path))
        return len(pd.read_pickle(dataset_path))
    log.warning("Cannot auto-detect sample count for %s, using 999999",
                dataset_path)
    return 999999


def get_test_settings(benchmark, scenario, test_mode, harness_config,
                      sample_count=None):
    settings = lg.TestSettings()
    settings.scenario = SCENARIO_MAP[scenario.lower()]

    if not os.path.isfile(harness_config.user_conf_path):
        raise FileNotFoundError(
            f"user_conf not found: {harness_config.user_conf_path}")

    
    
    if scenario.lower() == "interactive":
        conf_benchmark = f"{benchmark}-interactive"
        conf_scenario = "Server"
    else:
        conf_benchmark = benchmark.lower()
        conf_scenario = scenario.capitalize()

    repo_mlperf_conf = (
        Path(__file__).resolve().parent.parent
        / "mlperf_inference"
        / "mlperf.conf"
    )
    mlperf_conf_path = str(
        getattr(harness_config, "mlperf_conf_path", None)
        or repo_mlperf_conf
    )
    if not os.path.isfile(mlperf_conf_path):
        raise FileNotFoundError(
            f"official mlperf.conf not found: {mlperf_conf_path}")
    _validate_seed_file(mlperf_conf_path, require_all=True)
    _validate_seed_file(harness_config.user_conf_path, require_all=False)

    effective_conf_path = _write_effective_mlperf_conf(
        mlperf_conf_path,
        harness_config.user_conf_path,
        harness_config.output_log_dir,
    )
    settings.FromConfig(effective_conf_path, conf_benchmark, conf_scenario)

    
    
    
    
    for seed_name, expected in V61_LOADGEN_SEEDS.items():
        setattr(settings, seed_name, expected)
    _validate_effective_seeds(settings)
    log.info(
        "Applied and verified MLPerf v6.1 LoadGen seeds through TestSettings; "
        "official source=%s, user source=%s",
        mlperf_conf_path,
        harness_config.user_conf_path)

    audit_conf_path = os.path.join(os.getcwd(), "audit.config")
    if os.path.isfile(audit_conf_path):
        
        
        
        log.info("Detected MLPerf audit config: %s", audit_conf_path)

    if harness_config.target_qps > 0:
        settings.offline_expected_qps = harness_config.target_qps
        settings.server_target_qps = harness_config.target_qps
        log.warning("Overriding QPS: %s", harness_config.target_qps)

    if test_mode.lower() == "accuracy":
        acc_path = getattr(harness_config, "accuracy_dataset_path", "MISSING")
        if acc_path and acc_path != "MISSING":
            harness_config.dataset_path = acc_path
            acc_count = getattr(harness_config, "accuracy_sample_count", -1)
            if acc_count > 0:
                harness_config.total_sample_count = acc_count
            else:
                harness_config.total_sample_count = _count_samples(acc_path)
            log.info("Accuracy mode: dataset=%s  samples=%d",
                     acc_path, harness_config.total_sample_count)

    if sample_count is None:
        sample_count = harness_config.get(
            "sample_count", harness_config.total_sample_count)
    if harness_config.total_sample_count != sample_count:
        settings.min_query_count = harness_config.total_sample_count
        settings.max_query_count = harness_config.total_sample_count

    if benchmark.lower().startswith("deepseek-r1"):
        settings.use_token_latencies = True
        settings.server_coalesce_queries = True
        if test_mode.lower() == "accuracy":
            settings.min_query_count = harness_config.total_sample_count
            settings.max_query_count = harness_config.total_sample_count

    if harness_config.duration_sec != -1:
        time_ms = harness_config.duration_sec * 1000
        settings.min_duration_ms = time_ms
        settings.max_duration_ms = time_ms

    if test_mode.lower() == "accuracy":
        settings.mode = lg.TestMode.AccuracyOnly
    elif test_mode.lower() == "performance":
        settings.mode = lg.TestMode.PerformanceOnly
    else:
        raise ValueError(f"Unknown test_mode: {test_mode}")

    return settings


def get_log_settings(harness_config):
    os.makedirs(harness_config.output_log_dir, exist_ok=True)
    log_output = lg.LogOutputSettings()
    log_output.outdir = harness_config.output_log_dir
    log_output.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output
    log_settings.enable_trace = harness_config.enable_log_trace
    return log_settings


def get_sut(scenario, backend, conf):
    """Dispatch to the SUT for the active backend.

    Only two backends are supported now -- both fan out across multiple
    independent vLLM engines spawned by ``start_server.sh``:

      - ``standalone``: full prefill + decode on each engine (1 SUT, N engines)
      - ``pd``:         prefill-decode disaggregation (1 SUT, N prefill +
                        M decode engines, KV transferred over NIXL)

    Legacy single-engine backends (``pd_zmq``, ``standalone_zmq``, the old
    HTTP ``pd``) live under ``src/sut/*_zmq.py`` and
    ``src/sut/pd_http_legacy.py`` for reference only and are not wired
    into this dispatch.
    """
    sampling_config = conf["sampling_params"]
    if backend == "standalone":
        from sut.standalone import StandaloneMPServerSUT
        return StandaloneMPServerSUT(
            config=conf, sampling_config=sampling_config)
    if backend == "pd":
        from sut.pd import PDServerSUT
        return PDServerSUT(config=conf, sampling_config=sampling_config)
    if backend == "pd_zmq":
        
        
        
        from sut.pd_zmq import PDZmqOfflineSUT, PDZmqServerSUT
        if scenario.lower() == "offline":
            return PDZmqOfflineSUT(config=conf, sampling_config=sampling_config)
        return PDZmqServerSUT(config=conf, sampling_config=sampling_config)
    raise ValueError(
        f"Backend {backend!r} is deprecated. Use 'standalone' or 'pd'. "
        "Legacy backends (pd_zmq, standalone_zmq, http pd) live under "
        "src/sut/*_zmq.py / src/sut/pd_http_legacy.py for reference only.")


def set_envs(env_config):
    for env, val in env_config.items():
        
        
        
        if val is not None and not isinstance(val, (str, int, float, bool)):
            continue
        
        
        
        
        if val is None or val == "":
            os.environ.pop(env, None)
        else:
            os.environ[env] = str(val)


def run(conf):
    _validate_speculative_scenario(conf)
    test_settings = get_test_settings(
        benchmark=conf.benchmark_name,
        scenario=conf.scenario,
        test_mode=conf.test_mode,
        harness_config=conf.harness_config,
        sample_count=getattr(conf, "sample_count", None),
    )
    log_settings = get_log_settings(harness_config=conf.harness_config)
    set_envs(conf["env_config"])

    sut = get_sut(
        scenario=conf.scenario,
        backend=conf.backend,
        conf=conf,
    )

    sut.start()
    lg_sut = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    conf.print_config()
    audit_conf_path = os.path.join(os.getcwd(), "audit.config")
    lg.StartTestWithLogSettings(
        lg_sut, sut.qsl, test_settings, log_settings, audit_conf_path)
    log.info("Benchmark complete")
    sut.stop()
    lg.DestroySUT(lg_sut)
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    cfg = HarnessCfg().create_from_cli()
    run(cfg)
