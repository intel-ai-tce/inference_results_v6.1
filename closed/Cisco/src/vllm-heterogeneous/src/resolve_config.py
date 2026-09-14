
"""Resolve all configuration for start_server.sh.

Reads hardware and model YAML configs once, resolves cross-vendor
detection, hardware env vars, engine parameters, and runtime patches.
Prints shell-ready variable assignments to stdout for eval.

Usage (from start_server.sh):
    _cfg=$(python3 src/resolve_config.py \
        --role prefill --hardware h200 --model llama2_70b \
        [--remote-hardware mi350x] [--gpus 0,1,2,3] [--nixl-port 5601])
    eval "$_cfg"
"""

import argparse
import fcntl
import os
import re
import socket
import struct
import sys
from pathlib import Path

import yaml

from config_helpers import select_model_profile


_ENV_PATTERN = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}")


def load_yaml(path: Path):
    text = path.read_text()
    missing = sorted({name for name in _ENV_PATTERN.findall(text) if not os.environ.get(name)})
    if missing:
        raise RuntimeError(
            f"{path} requires non-empty environment variable(s): {', '.join(missing)}"
        )
    return yaml.safe_load(
        _ENV_PATTERN.sub(lambda match: os.environ[match.group(1)], text)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        required=True,
        choices=["prefill", "decode", "standalone"],
    )
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", default="")
    parser.add_argument(
        "--scenario",
        default="",
        dest="scenario",
        help="active MLPerf scenario; selects the <role>.<hw>.<scenario> and "
        "vllm_env_config.<hw>.<scenario> sub-blocks (offline/server/interactive)",
    )
    parser.add_argument("--remote-hardware", default="", dest="remote_hardware")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--nixl-port", default="", dest="nixl_port")
    parser.add_argument("--mgmt-iface", default="", dest="mgmt_iface")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    hw_path = root / "config" / "hardware" / f"{args.hardware}.yaml"
    model_path = root / "config" / "model" / f"{args.model}.yaml"

    for p in (hw_path, model_path):
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(1)

    try:
        hw = load_yaml(hw_path)
        model = load_yaml(model_path)
        profile = args.profile
        candidate_profiles = model.get("profiles") or {}
        if not profile and isinstance(candidate_profiles, dict):
            if args.role == "standalone":
                standalone_profile = f"standalone_{args.hardware}"
                if standalone_profile in candidate_profiles:
                    profile = standalone_profile
            elif "pd" in candidate_profiles:
                profile = "pd"
        model = select_model_profile(model, profile or None, args.scenario or None)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    
    
    
    
    
    
    
    
    
    
    
    
    shell_vars: "dict[str, str]" = {}
    env_exports: "dict[str, str]" = {}
    env_unsets: "list[str]" = []

    def var(name, value):
        shell_vars[name] = value

    def export_var(name, value):
        
        if name in env_unsets:
            env_unsets.remove(name)
        env_exports[name] = value

    def unset_var(name):
        
        
        env_exports.pop(name, None)
        if name not in env_unsets:
            env_unsets.append(name)

    
    
    
    local_platform = hw.get("platform", "nvidia")
    remote_platform = ""
    apply_patches = "false"
    hw_patches = ""
    kv_scale_source = ""

    cross_vendor = "false"

    if args.remote_hardware:
        rhw_path = root / "config" / "hardware" / f"{args.remote_hardware}.yaml"
        if not rhw_path.exists():
            print(f"ERROR: {rhw_path} not found", file=sys.stderr)
            sys.exit(1)
        try:
            remote_hw = load_yaml(rhw_path)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        remote_platform = remote_hw.get("platform", "nvidia")

        if local_platform != remote_platform:
            cross_vendor = "true"

            if args.role == "decode":
                apply_patches = "true"
                patches = hw.get("cross_platform_decode_patches", [])
                hw_patches = " ".join(patches)

                base = model.get("model_base", "")
                pfx = (
                    remote_hw.get("model_format", {})
                    .get("prefill", {})
                    .get("suffix", "fp8_dynamic")
                )
                kv_scale_source = f"{base}/{pfx}" if base else ""

            elif args.role == "prefill":
                prefill_patches = hw.get(
                    "cross_platform_prefill_patches", []
                )
                if prefill_patches:
                    apply_patches = "true"
                    hw_patches = " ".join(prefill_patches)

    var("APPLY_PATCHES", apply_patches)
    var("CROSS_VENDOR", cross_vendor)
    var("LOCAL_PLATFORM", local_platform)
    var("REMOTE_PLATFORM", remote_platform)
    var("HW_PATCHES", hw_patches)
    var("KV_SCALE_SOURCE", kv_scale_source)

    
    
    
    for section in ("nccl", "nccl_extra", "uccl", "ucx", "vllm", "rocm",
                     "triton", "fp4_gemm", "pytorch", "cuda"):
        for k, v in hw.get(section, {}).items():
            export_var(k, v)

    net = hw.get("network", {})
    nixl_port = args.nixl_port or str(net.get("nixl_side_channel_port", 5600))
    export_var("VLLM_NIXL_SIDE_CHANNEL_PORT", nixl_port)

    gpu = hw.get("gpu", {})
    dev_env = gpu.get("device_env", "CUDA_VISIBLE_DEVICES")
    devices = args.gpus or str(gpu.get("visible_devices", "0,1,2,3,4,5,6,7"))
    export_var(dev_env, devices)
    var("VISIBLE_DEVICES", devices)

    
    
    
    iface = args.mgmt_iface or os.environ.get("MGMT_INTERFACE", "")
    if not iface:
        print("ERROR: Set MGMT_INTERFACE in config/deployment.env or pass --mgmt-iface.", file=sys.stderr)
        sys.exit(1)
    var("MGMT_IFACE", iface)
    export_var("GLOO_SOCKET_IFNAME", iface)
    export_var("NCCL_SOCKET_IFNAME", iface)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(
            fcntl.ioctl(
                sock.fileno(),
                0x8915,  
                struct.pack("256s", iface.encode()[:15]),
            )[20:24]
        )
    except Exception:
        print(
            f"ERROR: Could not detect IP for interface '{iface}'. "
            "Set MGMT_INTERFACE to an interface with an IPv4 address.",
            file=sys.stderr,
        )
        sys.exit(1)

    var("MGMT_IP", ip)
    export_var("VLLM_NIXL_SIDE_CHANNEL_HOST", ip)

    
    
    
    role = args.role
    if role == "standalone":
        
        
        
        section_name = "standalone"
    elif role == "prefill":
        
        
        section_name = "prefiller"
    else:
        
        section_name = "decoder"
    s = model.get(section_name, {})
    hw_key = args.hardware
    hw_overrides = s.get(hw_key, {}) if isinstance(s.get(hw_key), dict) else {}
    
    
    
    
    
    
    scenario = (args.scenario or "").lower()
    if scenario and isinstance(hw_overrides.get(scenario), dict):
        scn_over = hw_overrides[scenario]
        hw_overrides = {
            k: v for k, v in hw_overrides.items() if not isinstance(v, dict)
        }
        hw_overrides.update(scn_over)
    if role == "standalone":
        fmt_key = "prefill"
    else:
        fmt_key = role
    fmt = hw.get("model_format", {}).get(fmt_key, {})

    def p(key, default):
        """Lookup priority: hw_overrides > role section > hardware model_format > default."""
        return hw_overrides.get(key, s.get(key, fmt.get(key, default)))

    
    base = model.get("model_base", "")
    suffix = hw_overrides.get(
        "suffix", s.get("suffix", fmt.get("suffix", "fp8_dynamic"))
    )
    model_path_val = f"{base}/{suffix}" if base and suffix else (base or "")
    var("MODEL_PATH", model_path_val)

    quant = hw_overrides.get(
        "quantization", s.get("quantization", fmt.get("quantization"))
    )
    var("QUANTIZATION", quant if quant else "")

    var("TP_SIZE", p("tp_size", 1))
    var("DP_SIZE", p("dp_size", 8))
    var("MAX_MODEL_LEN", p("max_model_len", 2048))
    var("MAX_NUM_SEQS", p("max_num_seqs", 1024))
    var("MAX_BATCHED_TOKENS", p("max_num_batched_tokens", 32768))
    var("GPU_MEM_UTIL", p("gpu_memory_utilization", 0.92))
    var("BLOCK_SIZE", p("block_size", 16))
    var("KV_CACHE_DTYPE", p("kv_cache_dtype", "fp8"))
    ec_raw = model.get("vllm_engine_config", {})
    ec_hw = (
        ec_raw.get(hw_key, {}) if isinstance(ec_raw.get(hw_key), dict) else {}
    )
    var(
        "DTYPE",
        hw_overrides.get(
            "dtype",
            s.get("dtype", ec_hw.get("dtype", ec_raw.get("dtype", "bfloat16"))),
        ),
    )
    var("MODEL_SEED", p("seed", 0))
    var("CALCULATE_KV_SCALES", p("calculate_kv_scales", False))
    
    
    
    
    kv_scale_source_override = p("kv_scale_source", None)
    if kv_scale_source_override is not None:
        var("KV_SCALE_SOURCE", kv_scale_source_override)
    var("KV_SCALE_OUTPUT", p("kv_scale_output", "/tmp/fp8_kv_scales.json"))
    var("ENABLE_PREFIX_CACHING", p("enable_prefix_caching", False))
    var("ENABLE_CHUNKED_PREFILL", p("enable_chunked_prefill", True))
    var("ENFORCE_EAGER", p("enforce_eager", False))
    var("DISABLE_SLIDING_WINDOW", p("disable_sliding_window", False))
    disable_hybrid_kv_cache_manager = p("disable_hybrid_kv_cache_manager", None)
    if disable_hybrid_kv_cache_manager in (None, ""):
        unset_var("DISABLE_HYBRID_KV_CACHE_MANAGER")
    else:
        var("DISABLE_HYBRID_KV_CACHE_MANAGER", disable_hybrid_kv_cache_manager)
    var("TRUST_REMOTE_CODE", p("trust_remote_code", False))
    var("ENABLE_EXPERT_PARALLEL", p("enable_expert_parallel", False))
    var("ENABLE_DBO", p("enable_dbo", False))
    var("DBO_DECODE_TOKEN_THRESHOLD", p("dbo_decode_token_threshold", 256))
    var("ENABLE_EPLB", p("enable_eplb", False))

    var("ALL2ALL_BACKEND", p("all2all_backend", ""))
    var("LINEAR_BACKEND", p("linear_backend", "auto"))
    var("ATTENTION_BACKEND", p("attention_backend", ""))
    var("SPECULATIVE_METHOD", p("speculative_method", ""))
    var("SPECULATIVE_MODEL", p("speculative_model", ""))
    var("SPECULATIVE_MODEL_REFERENCE", p("speculative_model_reference", ""))
    var("SPECULATIVE_NUM_TOKENS", p("speculative_num_tokens", 0))
    var("SPECULATIVE_EAGLE_TOPK", p("speculative_eagle_topk", 0))
    var("SPECULATIVE_DRAFT_SAMPLE_METHOD", p("speculative_draft_sample_method", ""))
    var("MOE_BACKEND", p("moe_backend", ec_hw.get("moe_backend", ec_raw.get("moe_backend", ""))))

    asched = hw_overrides.get("async_scheduling", s.get("async_scheduling", True))
    var("ASYNC_SCHEDULING", asched)
    var(
        "DISABLE_NCCL_FOR_DP_SYNCHRONIZATION",
        p("disable_nccl_for_dp_synchronization", False),
    )

    cg_mode = hw_overrides.get(
        "cudagraph_mode",
        s.get(
            "cudagraph_mode",
            ec_hw.get(
                "cudagraph_mode",
                ec_raw.get("cudagraph_mode", "FULL_DECODE_ONLY"),
            ),
        ),
    )
    var("CUDAGRAPH_MODE", cg_mode)

    compilation_mode = hw_overrides.get(
        "compilation_mode",
        s.get(
            "compilation_mode",
            ec_hw.get("compilation_mode", ec_raw.get("compilation_mode", "")),
        ),
    )
    if compilation_mode != "" and compilation_mode is not None:
        var("COMPILATION_MODE", compilation_mode)

    compile_sizes_val = hw_overrides.get(
        "compile_sizes", s.get("compile_sizes", ""))
    if compile_sizes_val:
        import json
        var("COMPILE_SIZES", json.dumps(
            compile_sizes_val if isinstance(compile_sizes_val, list)
            else compile_sizes_val))

    capture_range_val = hw_overrides.get(
        "cudagraph_capture_range", s.get("cudagraph_capture_range", ""))
    if capture_range_val:
        import json
        var("CUDAGRAPH_CAPTURE_RANGE", json.dumps(
            capture_range_val if isinstance(capture_range_val, list)
            else capture_range_val))

    var("DISABLE_CUSTOM_ALL_REDUCE", p("disable_custom_all_reduce", False))

    moe_replica = p("moe_dp_replica_mode", None)
    if moe_replica is not None:
        export_var("VLLM_MOE_DP_REPLICA_MODE", "1" if moe_replica else "0")

    
    
    
    
    
    
    sc = model.get("vllm_sampling_config", {})

    def s_(role_key, sc_key, default):
        return hw_overrides.get(
            role_key, s.get(role_key, sc.get(sc_key, default)))

    var("DECODE_MAX_TOKENS", s_("decode_max_tokens", "max_tokens", 1024))
    var("DECODE_MIN_TOKENS", s_("decode_min_tokens", "min_tokens", 0))
    var("DECODE_TEMPERATURE", s_("decode_temperature", "temperature", 0.0))
    var("DECODE_TOP_K", s_("decode_top_k", "top_k", 1))
    var("DECODE_TOP_P", s_("decode_top_p", "top_p", 0.001))
    var("USE_GENERATION_STOP_TOKEN_IDS", p("use_generation_stop_token_ids", True))

    
    pd_raw = model.get("pd_config", {})
    pd_hw = (
        pd_raw.get(hw_key, {}) if isinstance(pd_raw.get(hw_key), dict) else {}
    )
    var("MODEL_NAME", pd_hw.get("model_name", pd_raw.get("model_name", "")))

    decode_addr = pd_hw.get("decode_addr", pd_raw.get("decode_addr", ""))
    var("DECODE_FORWARD_ADDRS", decode_addr)
    var(
        "DECODE_FORWARD_PORT",
        pd_hw.get(
            "zmq_decode_pull_port", pd_raw.get("zmq_decode_pull_port", 5557)
        ),
    )

    
    servers_raw = model.get("servers", {})
    prefill_servers = servers_raw.get("prefill", []) or []
    if isinstance(prefill_servers, str):
        prefill_servers = prefill_servers.split()
    prefill_ip = (
        str(prefill_servers[0]).split(":")[0] if prefill_servers else "")
    var("PREFILL_IP", prefill_ip)
    var("PREFILL_ENGINE_COUNT", len(prefill_servers))

    
    
    
    
    
    
    
    
    
    
    
    
    if role == "standalone":
        from config_helpers import resolve_standalone_endpoints
        try:
            own_eps = resolve_standalone_endpoints(
                model, hardware=args.hardware, scenario=args.scenario
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        own_eps = servers_raw.get(role, []) or []
        if isinstance(own_eps, str):
            own_eps = own_eps.split()
    var("MP_ENDPOINTS", " ".join(str(e) for e in own_eps))

    peer_eps: list = []
    if role == "prefill":
        peer_eps = servers_raw.get("decode", []) or []
        if isinstance(peer_eps, str):
            peer_eps = peer_eps.split()
    var("MP_PEER_ENDPOINTS", " ".join(str(e) for e in peer_eps))

    port = hw.get("server", {}).get("port", 8100)
    var("PORT", port)
    kv_role = hw.get("server", {}).get("kv_role", "kv_both")
    var("KV_ROLE", kv_role)

    
    
    
    env_raw = model.get("vllm_env_config", {})
    env_hw = (
        env_raw.get(hw_key, {}) if isinstance(env_raw.get(hw_key), dict) else {}
    )
    
    
    
    env_scn = env_hw.get(scenario) if scenario else None
    env_hw_eff = {k: v for k, v in env_hw.items() if not isinstance(v, dict)}
    if isinstance(env_scn, dict):
        env_hw_eff.update(
            {k: v for k, v in env_scn.items() if not isinstance(v, dict)}
        )
    for k, v in env_raw.items():
        if isinstance(v, dict):
            continue
        v = env_hw_eff.get(k, v)
        if v == "" or v is None:
            unset_var(k)
        else:
            export_var(k, v)
    for k, v in env_hw_eff.items():
        if k in env_raw:
            continue
        if v == "" or v is None:
            unset_var(k)
        else:
            export_var(k, v)

    
    
    
    runtime_patches = model.get("runtime_patches", [])
    if isinstance(runtime_patches, str):
        runtime_patches = runtime_patches.split()
    runtime_patches = list(runtime_patches or [])

    runtime_patches_by_hw = model.get("runtime_patches_by_hardware", {})
    if isinstance(runtime_patches_by_hw, dict):
        hw_runtime_patches = runtime_patches_by_hw.get(hw_key, [])
        if isinstance(hw_runtime_patches, str):
            hw_runtime_patches = hw_runtime_patches.split()
        runtime_patches.extend(hw_runtime_patches or [])
    var("RUNTIME_PATCHES", " ".join(runtime_patches))

    
    
    var("IS_MOE", "1" if model.get("moe") else "0")

    
    
    
    
    
    collision = set(shell_vars) & set(env_exports)
    if collision:
        print(
            f"ERROR: variable(s) emitted as both shell var and export: "
            f"{sorted(collision)}",
            file=sys.stderr,
        )
        sys.exit(1)

    lines: "list[str]" = []
    for name, value in shell_vars.items():
        lines.append(f'{name}="{value}"')
    for name, value in env_exports.items():
        lines.append(f'export {name}="{value}"')
    for name in env_unsets:
        lines.append(f"unset {name} 2>/dev/null || true")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
