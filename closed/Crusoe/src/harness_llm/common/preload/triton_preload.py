import json
import triton
import hashlib
import glob
import os
import itertools

from triton.runtime.driver import driver

from aiter.ops.triton.quant import _dynamic_mxfp4_quant_kernel
from aiter.utility.fp4_utils import _moe_mxfp4_sort_kernel
from aiter.ops.triton.gemm_afp4wfp4 import _gemm_afp4_wfp4_kernel
from aiter.ops.triton.fused_qk_concat import _qk_rope_cat_kernel
from aiter.ops.triton.batched_gemm_afp4wfp4_pre_quant import _batched_gemm_afp4_wfp4_pre_quant_kernel


DEFAULT_KERNEL_RECORDING_FILE = "/lab-mlperf-inference/code/triton_kernel_recording_{}.txt"
DEFAULT_KERNEL_PRELOAD_FILE = "/lab-mlperf-inference/code/triton_kernel_preload.txt"
GEMM_KERNEL_FILE = "/lab-mlperf-inference/code/harness_llm/common/preload/gemm_afp4_wfp4_kernel.txt"


def hash(str):
    return hashlib.sha256(str.encode('utf-8')).hexdigest()


class SchedulerTritonPreloadMixin:

    def start_recording(self):
        self.recording_enabled = bool(int(os.getenv("TRITON_ENABLE_RECORD_KERNELS", 0)))

        if not self.recording_enabled:
            return

        self.kernel_recordings = []
        self.recording_file = os.getenv("TRITON_KERNEL_RECORDING_FILE",
                                        DEFAULT_KERNEL_RECORDING_FILE)
        if "{}" in self.recording_file:
            self.recording_file = self.recording_file.format(os.getpid())

        with open(self.recording_file, "w") as f:
            f.write("")

        def cache_hook(*args, **kwargs):
            kernel_descriptor = kwargs["compile"]["specialization_data"]
            device = kwargs["compile"]["device"]
            self.kernel_recordings.append({
                "device": device,
                "hash": hash(kernel_descriptor),
                "kernel_descriptor": kernel_descriptor
                })

        triton.knobs.runtime.jit_cache_hook = cache_hook


    def create_preload_descriptor(self, kernel_recording):
        return {
            "device": kernel_recording["device"], 
            "hash": kernel_recording["hash"],
            "kernel_descriptor": json.loads(kernel_recording["kernel_descriptor"])
        }


    def save_recording(self):
        if not self.recording_enabled:
            return

        with open(self.recording_file, "a") as f:
            for kernel_recording in self.kernel_recordings:
                preload_descriptor = self.create_preload_descriptor(kernel_recording)
                f.write(json.dumps(preload_descriptor))
                f.write("\n")
        self.kernel_recordings.clear()


def mock_get_current_device(device_index):
    return lambda: device_index


def preload_kernel(kernel, kernel_descriptor, devices):
    saved_get_current_device = driver.active.get_current_device

    for device in devices:
        driver.active.get_current_device = mock_get_current_device(device)
        kernel.preload(kernel_descriptor)

    driver.active.get_current_device = saved_get_current_device


def preload_kernels(from_main=False):
    if not from_main and not bool(int(os.getenv("TRITON_ENABLE_PRELOAD_KERNELS", 0))):
        print(f"Triton kernel preload skipped\n", flush=True)
        return

    preload_gemm_kernel()
    preload_moe_mxfp4()
    preload_qk_rope_kernel()
    preload_gemm_batched_kernel()
    preload_dynamic_mxfp4_quant_kernel()

def merge_recordings(file: str = DEFAULT_KERNEL_RECORDING_FILE):
    if not file:
        file = DEFAULT_KERNEL_RECORDING_FILE

    if "{}" not in file:
        print("Input file does not contain placeholder")
        return

    recording_files = file.replace("{}", "*")
    base_dir = os.path.dirname(recording_files)
    filename_pattern = os.path.basename(recording_files)

    search_pattern = os.path.join(base_dir, filename_pattern)
    recording_files = glob.glob(search_pattern)

    if not recording_files:
        print(f"No recording files found matching pattern: {search_pattern}")
        return

    merged_file = os.path.join(base_dir, "triton_kernel_recording.txt")
    with open(merged_file, "w") as outfile:
        for input_file in recording_files:
            try:
                with open(input_file, "r") as infile:
                    for line in infile:
                        line = line.strip()
                        if line:  # Skip empty lines
                            outfile.write(line)
                            outfile.write("\n")
            except FileNotFoundError:
                print(f"  Error: File not found: {input_file}")
            except Exception as e:
                print(f"  Error processing {input_file}: {e}")

    for input_file in recording_files:
        os.remove(input_file)

preloaded_gemm_num = 0

def preload_gemm_kernels(kernel_json, kernel):
    kernel_descriptor = json.dumps(kernel_json["kernel_descriptor"])
    devices = range(8)
    global preloaded_gemm_num 
    preloaded_gemm_num += 1
    if kernel:
        preload_kernel(kernel, kernel_descriptor, devices)
    if preloaded_gemm_num % 1000 == 0:
        print(f"{preloaded_gemm_num} gemm kernels loaded\n", flush=True)

def preload_gemm_kernel():
    kernel = _gemm_afp4_wfp4_kernel.fn
    try:
        with open(GEMM_KERNEL_FILE, 'r') as file:
            for line in file:
                kernel_record = {"kernel_descriptor": {}}
                kernel_line = json.loads(line)
                kernel_record["kernel_descriptor"] = kernel_line
                preload_gemm_kernels(kernel_record, kernel)
    except FileNotFoundError:
        print(f"Error: The file '{GEMM_KERNEL_FILE}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

preloaded_mxfp4_num = 0

def replace_moe_mxfp4_keys(kernel_json):
    position = 0
    for num in kernel_json["kernel_descriptor"]["constant_vals"]:
        kernel_json["kernel_descriptor"]["key"] = kernel_json["kernel_descriptor"]["key"].replace(f"('constexpr', {position})", f"('constexpr', {num})")
        position += 1

def preload_moe_mxfp4_kernels(kernel_json):
    kernel_descriptor = json.dumps(kernel_json["kernel_descriptor"])
    devices = range(8)
    global preloaded_mxfp4_num
    preloaded_mxfp4_num += 1
    kernel = _moe_mxfp4_sort_kernel
    if kernel:
        preload_kernel(kernel, kernel_descriptor, devices)
    if preloaded_mxfp4_num % 1000 == 0:
        print(f"{preloaded_mxfp4_num} moe triton kernels loaded\n", flush=True)

def process_moe_mxfp4_kernels(num, kernel_record_224, kernel_record_8, orig_keys):
    kernel_record_224["kernel_descriptor"]["constant_vals"] = [224, 1, 1792, 64, 16, 1, num, num, 224, 16, 4, 1]
    kernel_record_8["kernel_descriptor"]["constant_vals"] = [8, 1, 64, 64, 16, 1, num, num*8, 8, 16, 4, 8]
    replace_moe_mxfp4_keys(kernel_record_224)
    replace_moe_mxfp4_keys(kernel_record_8)
    preload_moe_mxfp4_kernels(kernel_record_224)
    preload_moe_mxfp4_kernels(kernel_record_8)
    kernel_record_224["kernel_descriptor"]["key"] = orig_keys
    kernel_record_8["kernel_descriptor"]["key"] = orig_keys


def preload_moe_mxfp4():
    kernel_str = """{"kernel_descriptor": {"name": "aiter.utility.fp4_utils._moe_mxfp4_sort_kernel", "signature": {"blockscale_e8m0_ptr": "*u8", "sorted_ids_ptr": "*i32", "num_valid_ids_ptr": "*i32", "blockscale_e8m0_sorted_ptr": "*u32", "stride_blockscale_e8m0_m": "constexpr", "stride_blockscale_e8m0_n": "constexpr", "stride_o3": "constexpr", "stride_o2": "constexpr", "stride_o1": "constexpr", "stride_o0": "constexpr", "token_num": "constexpr", "M_i": "constexpr", "N_i": "constexpr", "BLOCK_SIZE_M": "constexpr", "BLOCK_SIZE_N": "constexpr", "TOPK": "constexpr"}, "constant_keys": [[4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]], "constant_vals": [224, 1, 1792, 64, 16, 1, 16384, 16384, 224, 16, 4, 1], "attrs_keys": [[0], [1], [2], [3]], "attrs_vals": [[["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]]], "options": {"num_warps": 4, "waves_per_eu": 1, "num_stages": 2, "num_ctas": 1, "extern_libs": [["ocml", "/opt/venv/lib/python3.10/site-packages/triton/backends/amd/lib/ocml.bc"], ["ockl", "/opt/venv/lib/python3.10/site-packages/triton/backends/amd/lib/ockl.bc"]], "cluster_dims": [1, 1, 1], "debug": false, "sanitize_overflow": true, "arch": "gfx950", "supported_fp8_dtypes": ["fp8e4b8", "fp8e4nv", "fp8e5", "fp8e5b16"], "deprecated_fp8_dot_operand_dtypes": ["fp8e4b8", "fp8e5b16"], "default_dot_input_precision": "ieee", "allowed_dot_input_precisions": ["ieee"], "enable_fp_fusion": true, "launch_cooperative_grid": false, "matrix_instr_nonkdim": 0, "kpack": 1, "allow_flush_denorm": false, "max_num_imprecise_acc_default": 0, "backend_name": "hip", "instrumentation_mode": "", "schedule_hint": "none", "warp_size": 64}, "key": "[('*u8', 'DS'), ('*i32', 'DS'), ('*i32', 'DS'), ('*u32', 'DS'), ('constexpr', 0), ('constexpr', 1), ('constexpr', 2), ('constexpr', 3), ('constexpr', 4), ('constexpr', 5), ('constexpr', 6), ('constexpr', 7), ('constexpr', 8), ('constexpr', 9), ('constexpr', 10), ('constexpr', 11)]{'debug': False}"}}"""
    kernel_record_224 = json.loads(kernel_str)
    kernel_record_8 = json.loads(kernel_str)
    orig_keys = kernel_record_224["kernel_descriptor"]["key"]

    for i in range(0, 16385):
        process_moe_mxfp4_kernels(i, kernel_record_224, kernel_record_8, orig_keys)

    print(f"Preloaded moe_mxfp4 kernels: {preloaded_mxfp4_num}", flush=True)

def preload_qk_rope_kernel():
    kernel_str = """{"kernel_descriptor": {"name": "aiter.ops.triton._triton_kernels.fused_qk_concat._qk_rope_cat_kernel", "signature": {"q_nope_ptr": "*bf16", "q_pe_ptr": "*bf16", "k_nope_ptr": "*bf16", "k_pe_ptr": "*bf16", "pos_ptr": "*i64", "cos_ptr": "*bf16", "sin_ptr": "*bf16", "q_out_ptr": "*bf16", "k_out_ptr": "*bf16", "q_nope_stride_b": "i32", "q_nope_stride_h": "i32", "q_nope_stride_d": "constexpr", "q_pe_stride_b": "i32", "q_pe_stride_h": "i32", "q_pe_stride_d": "constexpr", "k_nope_stride_b": "i32", "k_nope_stride_h": "i32", "k_nope_stride_d": "constexpr", "k_pe_stride_b": "i32", "k_pe_stride_h": "i32", "k_pe_stride_d": "constexpr", "pos_stride_b": "constexpr", "cos_stride_b": "i32", "cos_stride_d": "constexpr", "q_out_stride_b": "i32", "q_out_stride_h": "i32", "q_out_stride_d": "constexpr", "k_out_stride_b": "i32", "k_out_stride_h": "i32", "k_out_stride_d": "constexpr", "QH_PER_KH": "constexpr", "REUSE_FREQS_FRONT_PART": "constexpr", "IS_NEOX": "constexpr", "BLOCK_D_nope": "constexpr", "BLOCK_D_pe": "constexpr", "BLOCK_D_HALF_pe": "constexpr"}, "constant_keys": [[11], [14], [17], [20], [21], [23], [26], [29], [30], [31], [32], [33], [34], [35]], "constant_vals": [1, 1, 1, 1, 1, 1, 1, 1, 128, true, false, 512, 64, 32], "attrs_keys": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [12], [13], [15], [16], [18], [19], [22], [24], [25], [27], [28]], "attrs_vals": [[["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]]], "options": {"num_warps": 4, "waves_per_eu": 1, "num_stages": 2, "num_ctas": 1, "extern_libs": [["ocml", "/opt/venv/lib/python3.10/site-packages/triton/backends/amd/lib/ocml.bc"], ["ockl", "/opt/venv/lib/python3.10/site-packages/triton/backends/amd/lib/ockl.bc"]], "cluster_dims": [1, 1, 1], "debug": false, "sanitize_overflow": true, "arch": "gfx950", "supported_fp8_dtypes": ["fp8e4b8", "fp8e4nv", "fp8e5", "fp8e5b16"], "deprecated_fp8_dot_operand_dtypes": ["fp8e4b8", "fp8e5b16"], "default_dot_input_precision": "ieee", "allowed_dot_input_precisions": ["ieee"], "enable_fp_fusion": true, "launch_cooperative_grid": false, "matrix_instr_nonkdim": 0, "kpack": 1, "allow_flush_denorm": false, "max_num_imprecise_acc_default": 0, "backend_name": "hip", "instrumentation_mode": "", "schedule_hint": "none", "warp_size": 64}, "key": "[('*bf16', 'DS'), ('*bf16', 'DS'), ('*bf16', 'DS'), ('*bf16', 'DS'), ('*i64', 'DS'), ('*bf16', 'DS'), ('*bf16', 'DS'), ('*bf16', 'DS'), ('*bf16', 'DS'), ('i32', 'D'), ('i32', 'D'), ('constexpr', 1), ('i32', 'D'), ('i32', 'D'), ('constexpr', 1), ('i32', 'D'), ('i32', 'D'), ('constexpr', 1), ('i32', 'D'), ('i32', 'D'), ('constexpr', 1), ('constexpr', 1), ('i32', 'D'), ('constexpr', 1), ('i32', 'D'), ('i32', 'D'), ('constexpr', 1), ('i32', 'D'), ('i32', 'D'), ('constexpr', 1), ('constexpr', 128), ('constexpr', True), ('constexpr', False), ('constexpr', 512), ('constexpr', 64), ('constexpr', 32)]{'debug': False}"}}"""
    kernel_record = json.loads(kernel_str)
    kernel = _qk_rope_cat_kernel
    devices = range(8)
    kernel_descriptor = json.dumps(kernel_record["kernel_descriptor"])
    if kernel:
        preload_kernel(kernel, kernel_descriptor, devices)
    print("Preloaded qk_rope_cat kernel", flush=True)


preloaded_batched_gemm_num = 0

def preload_gemm_batched_kernels(kernel_json, kernel):
    kernel_descriptor = json.dumps(kernel_json["kernel_descriptor"])
    devices = range(8)
    global preloaded_batched_gemm_num
    preloaded_batched_gemm_num += 1
    if kernel:
        preload_kernel(kernel, kernel_descriptor, devices)
    if preloaded_batched_gemm_num % 1000 == 0:
        print(f"{preloaded_batched_gemm_num} gemm batched triton kernels loaded\n", flush=True)

def preload_gemm_batched_kernel():
    kernel_str = """{"kernel_descriptor": {"name": "aiter.ops.triton._triton_kernels.batched_gemm_afp4wfp4_pre_quant._batched_gemm_afp4_wfp4_pre_quant_kernel", "signature": {"a_ptr": "*bf16", "b_ptr": "*u8", "c_ptr": "*bf16", "b_scales_ptr": "*u8", "M": "i32", "N": "i32", "K": "i32", "stride_ab": "i32", "stride_am": "i32", "stride_ak": "constexpr", "stride_bb": "i32", "stride_bk": "constexpr", "stride_bn": "i32", "stride_cb": "i32", "stride_ck": "i32", "stride_cm": "i32", "stride_cn": "constexpr", "stride_bsb": "i32", "stride_bsn": "i32", "stride_bsk": "constexpr", "BLOCK_SIZE_M": "constexpr", "BLOCK_SIZE_N": "constexpr", "BLOCK_SIZE_K": "constexpr", "GROUP_SIZE_M": "constexpr", "NUM_KSPLIT": "constexpr", "SPLITK_BLOCK_SIZE": "constexpr", "EVEN_K": "constexpr", "GRID_MN": "constexpr", "cache_modifier": "constexpr"}, "constant_keys": [[9], [11], [16], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28]], "constant_vals": [1, 1, 1, 1, 32, 256, 128, 1, 1, 128, true, 92, ".cg"], "attrs_keys": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [10], [12], [13], [14], [15], [17], [18], [28]], "attrs_vals": [[["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [], []], "options": {"num_warps": 8, "waves_per_eu": 0, "num_stages": 1, "num_ctas": 1, "extern_libs": [["ocml", "/opt/venv/lib/python3.10/site-packages/triton/backends/amd/lib/ocml.bc"], ["ockl", "/opt/venv/lib/python3.10/site-packages/triton/backends/amd/lib/ockl.bc"]], "cluster_dims": [1, 1, 1], "debug": false, "sanitize_overflow": true, "arch": "gfx950", "supported_fp8_dtypes": ["fp8e4b8", "fp8e4nv", "fp8e5", "fp8e5b16"], "deprecated_fp8_dot_operand_dtypes": ["fp8e4b8", "fp8e5b16"], "default_dot_input_precision": "ieee", "allowed_dot_input_precisions": ["ieee"], "enable_fp_fusion": true, "launch_cooperative_grid": false, "matrix_instr_nonkdim": 16, "kpack": 1, "allow_flush_denorm": false, "max_num_imprecise_acc_default": 0, "backend_name": "hip", "instrumentation_mode": "", "schedule_hint": "none", "warp_size": 64}, "key": "[('*bf16', 'DS'), ('*u8', 'DS'), ('*bf16', 'DS'), ('*u8', 'DS'), ('i32', ''), ('i32', 'D'), ('i32', 'D'), ('i32', 'D'), ('i32', 'D'), ('constexpr', 1), ('i32', 'D'), ('constexpr', 1), ('i32', 'D'), ('i32', 'D'), ('i32', 'D'), ('i32', 'D'), ('constexpr', 1), ('i32', 'D'), ('i32', ''), ('constexpr', 1), ('constexpr', 32), ('constexpr', 256), ('constexpr', 128), ('constexpr', 1), ('constexpr', 1), ('constexpr', 128), ('constexpr', True), ('constexpr', 92), ('constexpr', '.cg')]{'num_warps': 8, 'num_stages': 1, 'waves_per_eu': 0, 'matrix_instr_nonkdim': 16, 'debug': False}"}}"""
    kernel_record = json.loads(kernel_str)
    kernel = _batched_gemm_afp4_wfp4_pre_quant_kernel.fn

    block_size_m_values = [32, 4, 16, 8]
    block_size_n_values = [256, 128, 64]
    block_size_k_values = [128, 512]
    num_warps_values = [4, 8]
    grid_mn_values = list(range(4, 129))

    for block_m, block_n, block_k, num_warps in itertools.product(
        block_size_m_values, block_size_n_values, block_size_k_values, num_warps_values
    ):
        for grid_mn in grid_mn_values:
            kernel_record["kernel_descriptor"]["constant_vals"] = [
                1, 1, 1, 1,
                block_m,
                block_n,
                block_k,
                1, 1,
                block_k,
                True,
                grid_mn,
                ".cg"
            ]

            key_parts = [
                ('*bf16', 'DS'), ('*u8', 'DS'), ('*bf16', 'DS'), ('*u8', 'DS'),
                ('i32', ''), ('i32', 'D'), ('i32', 'D'), ('i32', 'D'), ('i32', 'D'),
                ('constexpr', 1), ('i32', 'D'), ('constexpr', 1), ('i32', 'D'),
                ('i32', 'D'), ('i32', 'D'), ('i32', 'D'), ('constexpr', 1),
                ('i32', 'D'), ('i32', ''),
                ('constexpr', 1),
                ('constexpr', block_m), ('constexpr', block_n), ('constexpr', block_k),
                ('constexpr', 1), ('constexpr', 1), ('constexpr', block_k),
                ('constexpr', True), ('constexpr', grid_mn), ('constexpr', '.cg')
            ]
            options_part = {
                'num_warps': num_warps,
                'num_stages': kernel_record["kernel_descriptor"]["options"]["num_stages"],
                'waves_per_eu': kernel_record["kernel_descriptor"]["options"]["waves_per_eu"],
                'matrix_instr_nonkdim': 16,
                'debug': False
            }

            if block_k == 512:
                kernel_record["kernel_descriptor"]["attrs_vals"][-2] = list(kernel_record["kernel_descriptor"]["attrs_vals"][-3])
                key_parts[18] = ('i32', 'D')
            else:
                kernel_record["kernel_descriptor"]["attrs_vals"][-2] = []

            key_str = str(key_parts) + str(options_part)
            kernel_record["kernel_descriptor"]["key"] = key_str
            kernel_record["kernel_descriptor"]["attrs_vals"][4] = []
            preload_gemm_batched_kernels(kernel_record, kernel)

            kernel_record["kernel_descriptor"]["attrs_vals"][4] = kernel_record["kernel_descriptor"]["attrs_vals"][5]
            key_parts[4] = ('i32', 'D')
            key_str = str(key_parts) + str(options_part)
            kernel_record["kernel_descriptor"]["key"] = key_str
            preload_gemm_batched_kernels(kernel_record, kernel)


dynamic_mxfp4_quant_num = 0

def preload_dynamic_mxfp4_quant_kernels(kernel_json, kernel):
    kernel_descriptor = json.dumps(kernel_json["kernel_descriptor"])
    devices = range(8)
    global dynamic_mxfp4_quant_num 
    dynamic_mxfp4_quant_num += 1
    if kernel:
        preload_kernel(kernel, kernel_descriptor, devices)
    if dynamic_mxfp4_quant_num % 1000 == 0:
        print(f"{dynamic_mxfp4_quant_num} gemm kernels loaded\n", flush=True)

def generate_dynamic_mxfp4_quant_attrs_vals(extra_tt):
    base_attrs = [
        [["tt.divisibility", 16], ["tt.pointer_range", 32]],
        [["tt.divisibility", 16], ["tt.pointer_range", 32]],
        [["tt.divisibility", 16], ["tt.pointer_range", 32]],
        [["tt.divisibility", 16]],
        [["tt.divisibility", 16]]
    ]
    
    if extra_tt:
        base_attrs.append([["tt.divisibility", 16]])
        base_attrs.append([["tt.divisibility", 16]])
    else:
        base_attrs.append([])
        base_attrs.append([])
    
    base_attrs.extend([
        [["tt.divisibility", 16]],
    ])
    
    return base_attrs

def generate_dynamic_mxfp4_quant_key_string(block_m, block_n, block_k, block_v, bool_val, extra_tt):
    bool_str = 'D' if extra_tt else ''
    
    key_parts = [
        "('*bf16', 'DS')", "('*u8', 'DS')", "('*u8', 'DS')", "('i32', 'D')", "('constexpr', 1)", "('i32', 'D')", 
        "('constexpr', 1)", "('constexpr', 1)", f"('i32', '{bool_str}')", f"('i32', '{bool_str}')", "('i32', 'D')", 
        f"('constexpr', {block_m})", f"('constexpr', {block_n})", f"('constexpr', {block_k})", f"('constexpr', {block_v})", 
        "('constexpr', 32)", f"('constexpr', {bool_val})", "('constexpr', 0)"
    ]
    
    options_dict = {
        'num_warps': 4, 'waves_per_eu': 0, 'num_stages': 1, 'debug': False
    }
    
    return f"[{', '.join(key_parts)}]{options_dict}"

def preload_dynamic_mxfp4_quant_kernel():
    kernel_str = """{"kernel_descriptor": {"name": "aiter.ops.triton._triton_kernels.quant._dynamic_mxfp4_quant_kernel", "signature": {"x_ptr": "*bf16", "x_fp4_ptr": "*u8", "bs_ptr": "*u8", "stride_x_m_in": "i32", "stride_x_n_in": "constexpr", "stride_x_fp4_m_in": "i32", "stride_x_fp4_n_in": "constexpr", "stride_bs_m_in": "constexpr", "stride_bs_n_in": "i32", "M": "i32", "N": "i32", "BLOCK_SIZE_M": "constexpr", "BLOCK_SIZE_N": "constexpr", "NUM_ITER": "constexpr", "NUM_STAGES": "constexpr", "MXFP4_QUANT_BLOCK_SIZE": "constexpr", "EVEN_M_N": "constexpr", "SCALING_MODE": "constexpr"}, "constant_keys": [[4], [6], [7], [11], [12], [13], [14], [15], [16], [17]], "constant_vals": [1, 1, 1, 32, 128, 4, 2, 32, false, 0], "attrs_keys": [[0], [1], [2], [3], [5], [8], [9], [10]], "attrs_vals": [[["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16], ["tt.pointer_range", 32]], [["tt.divisibility", 16]], [["tt.divisibility", 16]], [], [], [["tt.divisibility", 16]]], "options": {"num_warps": 4, "waves_per_eu": 0, "num_stages": 1, "num_ctas": 1, "extern_libs": [["ocml", "/opt/venv/lib/python3.10/site-packages/triton/backends/amd/lib/ocml.bc"], ["ockl", "/opt/venv/lib/python3.10/site-packages/triton/backends/amd/lib/ockl.bc"]], "cluster_dims": [1, 1, 1], "debug": false, "sanitize_overflow": true, "arch": "gfx950", "supported_fp8_dtypes": ["fp8e4b8", "fp8e4nv", "fp8e5", "fp8e5b16"], "deprecated_fp8_dot_operand_dtypes": ["fp8e4b8", "fp8e5b16"], "default_dot_input_precision": "ieee", "allowed_dot_input_precisions": ["ieee"], "enable_fp_fusion": true, "launch_cooperative_grid": false, "matrix_instr_nonkdim": 0, "kpack": 1, "allow_flush_denorm": false, "max_num_imprecise_acc_default": 0, "backend_name": "hip", "instrumentation_mode": "", "schedule_hint": "none", "warp_size": 64}, "key": "[('*bf16', 'DS'), ('*u8', 'DS'), ('*u8', 'DS'), ('i32', 'D'), ('constexpr', 1), ('i32', 'D'), ('constexpr', 1), ('constexpr', 1), ('i32', ''), ('i32', ''), ('i32', 'D'), ('constexpr', 32), ('constexpr', 128), ('constexpr', 4), ('constexpr', 2), ('constexpr', 32), ('constexpr', False), ('constexpr', 0)]{'num_warps': 4, 'waves_per_eu': 0, 'num_stages': 1, 'debug': False}"}}"""
    kernel_record = json.loads(kernel_str)
    kernel = _dynamic_mxfp4_quant_kernel.fn

    block_configs = [
        (32, 128, 4, 2),
        (8, 256, 1, 1),
    ]

    block_bool = [True, False]
    
    for (block_m, block_n, block_k, block_v), bool_val, extra_tt in itertools.product(
        block_configs, block_bool, block_bool
    ):
        kernel_record["kernel_descriptor"]["constant_vals"] = [
            1, 1, 1,
            block_m, block_n, block_k, block_v,
            32, bool_val, 0
        ]
        
        kernel_record["kernel_descriptor"]["attrs_vals"] = generate_dynamic_mxfp4_quant_attrs_vals(
            extra_tt
        )
        
        kernel_record["kernel_descriptor"]["key"] = generate_dynamic_mxfp4_quant_key_string(
            block_m, block_n, block_k, block_v, bool_val, extra_tt
        )
        
        preload_dynamic_mxfp4_quant_kernels(kernel_record, kernel)

if __name__ == "__main__":
    preload_kernels(from_main=True)
