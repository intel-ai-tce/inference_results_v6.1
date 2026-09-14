import torch
import time
import os
import math
from tqdm import tqdm
from aiter import gemm_a4w4
from aiter.utility.fp4_utils import dynamic_mxfp4_quant as dynamic_mxfp4_quant_asm
from aiter.ops.triton.activation import act_mul_and_mxfp4_quant
from aiter.ops.triton.mha import flash_attn_varlen_func
from vllm.model_executor.layers.layernorm import dispatch_cuda_rmsnorm_func
import vllm.envs as envs
import harness_llm.common.logging as logger

log = logger.get_logger(__name__)

# tqdm - increase time between progress updates
loginterval_sec = {'mininterval': 30.0, 'maxinterval': 60.0}

USE_TRITON_FLASH_ATTN = envs.VLLM_USE_TRITON_FLASH_ATTN and envs.VLLM_ROCM_USE_AITER
USE_AITER_RMSNORM = envs.VLLM_ROCM_USE_AITER_RMSNORM and envs.VLLM_ROCM_USE_AITER
USE_AITER_TRITON_SILU_MUL = (os.environ.get("VLLM_USE_AITER_TRITON_SILU_MUL", "0") == "1")
USE_AITER_TRITON_FP4_GEMM_ASM = (os.environ.get("VLLM_TRITON_FP4_GEMM_USE_ASM", "0") == "1")

def cache_kernels_for_device(params):
    cuda_device_index, config, tensor_parallel_size, real_device = params
    start_time = time.time()
    with torch.cuda.device(device=cuda_device_index):
        device = torch.device(f'cuda')

        # Cache attention kernels
        if USE_TRITON_FLASH_ATTN:
            cache_attention_kernels(device, config, tensor_parallel_size, real_device)
        else:
            log.warning(f"Skipping caching attention kernels for device {real_device} as USE_TRITON_FLASH_ATTN is not enabled")

        # Cache rmsnorm kernels
        if USE_AITER_RMSNORM:
            cache_rmsnorm_kernels(device, config, real_device)
        else:
            log.warning(f"Skipping caching rmsnorm kernels for device {real_device} as USE_AITER_RMSNORM is not enabled")

        # Cache activation kernels
        if USE_AITER_TRITON_SILU_MUL:
            cache_activation_kernels(device, config, tensor_parallel_size, real_device)
        else:
            log.warning(f"Skipping caching activation kernels for device {real_device} as USE_AITER_TRITON_SILU_MUL is not enabled")

        if USE_AITER_TRITON_FP4_GEMM_ASM:
            # Cache dynamic_mxfp4_quant_asm gemms
            cache_dynamic_mxfp4_quant_asm_gemms(device, config, tensor_parallel_size, real_device)

            # TODO: this gemm crashes when loading on device which is not the first in the TP group, we should figure it out
            # Cache gemm_a4w4_asm gemms
            if cuda_device_index == 0:
                cache_gemm_a4w4_asm_gemms(device, config, tensor_parallel_size, real_device)
        else:
            log.warning(f"Skipping caching dynamic_mxfp4_quant_asm and gemm_a4w4_asm kernels for device {real_device} as USE_AITER_TRITON_FP4_GEMM_ASM is not enabled")

    execution_time = time.time() - start_time
    log.info(f"Preload for device {real_device} took: {execution_time} seconds")

class ModelConfig:
    def __init__(self):
        self.input_start: int = 1024
        self.input_end: int = 22017
        self.q_shape: tuple[int, ...] = (128, 128)
        self.k_shape: tuple[int, ...] = (8, 128)
        self.v_shape: tuple[int, ...] = (8, 128)
        self.rmsnorm_shape: int = 16384
        self.act_shape: int = 106496
        self.mxfp4_quant_asm_shape: int = 16384
        self.gemm_a4w4_asm_layer_shapes: tuple[tuple[int, ...], ...] = (
            (16384, 26624 , 1664),
            (16384, 8192, 512),
            (106496, 8192, 512),
            (18432, 8912, 512)
        )
        self.batch_size_start: int = 1
        self.batch_size_end: int = 8

    def get_q_shape(self, first_dim, tp_size):
        return (first_dim, self.q_shape[0] // tp_size, self.q_shape[1])

    def get_k_shape(self, first_dim, tp_size):
        return (first_dim, self.k_shape[0] // tp_size, self.k_shape[1])

    def get_v_shape(self, first_dim, tp_size):
        return (first_dim, self.v_shape[0] // tp_size, self.v_shape[1])

    def get_rmsnorm_shape(self, first_dim):
        return (first_dim, self.rmsnorm_shape)

    def get_act_shape(self, first_dim, tp_size):
        return (first_dim, self.act_shape // tp_size)

    def get_mxfp4_quant_asm_shape(self, first_dim, tp_size=1):
        return (first_dim, self.mxfp4_quant_asm_shape // tp_size)

    def get_gemm_a4w4_asm_shapes(self, tp_size):
        s = self.gemm_a4w4_asm_layer_shapes
        return [
            (s[0][0], s[0][1] // tp_size, s[0][2] // tp_size),
            (s[1][0], s[1][1] // tp_size, s[1][2] // tp_size),
            (s[2][0] // tp_size, s[2][1], s[2][2]),
            (s[3][0] // tp_size, s[3][1], s[3][2]),
        ]

class Llama31_405bModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.input_start: int = 1024
        self.input_end: int = 22017
        self.q_shape: tuple[int, ...] = (128, 128)
        self.k_shape: tuple[int, ...] = (8, 128)
        self.v_shape: tuple[int, ...] = (8, 128)
        self.rmsnorm_shape: int = 16384
        self.act_shape: int = 106496
        self.mxfp4_quant_asm_shape: int = 16384
        self.gemm_a4w4_asm_layer_shapes: tuple[tuple[int, ...], ...] = (
            (16384, 26624 , 1664),
            (16384, 8192, 512),
            (106496, 8192, 512),
            (18432, 8912, 512)
        )
        self.batch_size_start: int = 1
        self.batch_size_end: int = 4

class Llama2_70bModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.input_start: int = 128
        self.input_end: int = 36864
        self.q_shape: tuple[int, ...] = (64, 128)
        self.k_shape: tuple[int, ...] = (8, 128)
        self.v_shape: tuple[int, ...] = (8, 128)
        self.rmsnorm_shape: int = 8192
        self.act_shape: int = 57344
        self.mxfp4_quant_asm_shape: int = 8192
        self.gemm_a4w4_asm_layer_shapes: tuple[tuple[int, ...], ...] = (
            (57344, 4096, 256),
            (10240, 4096, 256),
            (8192, 4096, 256),
            (8192, 14336, 896)
        )

def get_config(benchmark: str):
    if benchmark == "llama3_1-405b":
        return Llama31_405bModelConfig()
    elif benchmark == "llama2-70b":
        return Llama2_70bModelConfig()
    elif benchmark == "llama2-70b-interactive":
        return Llama2_70bModelConfig()
    else:
        raise Exception(f"No configuration found for model name: {benchmark}")

def preload_aiter_kernel_modules(device_ids: tuple[int, ...], llm_config, benchmark):
    if os.environ.get("HARNESS_PRELOAD_AITER_KERNELS", "0") != "1":
        log.info("Skipping preloading aiter modules")
        return
    log.info("Preloading aiter kernel modules")
    tensor_parallel_size = llm_config['tensor_parallel_size']
    config = get_config(benchmark=benchmark)
    assert len(device_ids) == tensor_parallel_size
    params_list = [(cuda_device_index, config, tensor_parallel_size, device_ids[cuda_device_index]) for cuda_device_index in range(0, tensor_parallel_size)]
    for params in params_list:
        cache_kernels_for_device(params)

    time.sleep(10)
    log.info("Preloading aiter modules done")

def cache_attention_kernels(device, config, tensor_parallel_size, real_device):
    input_sizes = range(config.input_start, config.input_end, 32)
    query_seq_start_loc = torch.ones(2, dtype=torch.int32, device=device)
    key_seq_start_loc = torch.ones(2, dtype=torch.int32, device=device)
    scale=0.08838834764831845
    padding = 32
    i = 0
    for bs in range(config.batch_size_start, config.batch_size_end):
        for size in tqdm(input_sizes, **loginterval_sec, desc=f"[device:{real_device}] Caching Attention Kernels for BS: {bs}", unit="kernel"):
            size = (size + padding - 1) // padding * padding
            query = torch.ones(config.get_q_shape(size, tensor_parallel_size), dtype=torch.bfloat16, device=device)
            key = torch.ones(config.get_k_shape(size, tensor_parallel_size), dtype=torch.bfloat16, device=device)
            value = torch.ones(config.get_v_shape(size, tensor_parallel_size), dtype=torch.bfloat16, device=device)
            if bs == 1:
                seq_lens = [size]
            elif size >= 7008: # TODO: make this a parameter
                seq_len_est = math.ceil((size / bs) / padding) * padding
                seq_lens = [seq_len_est - padding, seq_len_est, seq_len_est + padding]
            else:
                continue
            for max_seq_len in seq_lens:
                flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=query_seq_start_loc,
                    cu_seqlens_k=key_seq_start_loc,
                    max_seqlen_q=max_seq_len,
                    max_seqlen_k=max_seq_len,
                    dropout_p=0.0,
                    softmax_scale=scale,
                    causal=True,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=False,
                    return_lse=False,
                    return_attn_probs=False,
                    block_table=None,
                )

            del query
            del key
            del value
            i += 1
            if (i % 100) == 0:
                time.sleep(1)


def cache_rmsnorm_kernels(device, config, real_device):
    input_sizes = range(config.input_start, config.input_end)
    weight = torch.ones(config.get_rmsnorm_shape(1)[1], dtype=torch.bfloat16, device=device)
    for M in tqdm(input_sizes, **loginterval_sec, desc=f"[device:{real_device}] Caching RMSNorm Kernels", unit="kernel"):
        shape = config.get_rmsnorm_shape(M)
        x = torch.ones(shape, dtype=torch.bfloat16, device=device)
        residual = torch.ones(shape, dtype=torch.bfloat16, device=device)
        rms_norm_func_resid = dispatch_cuda_rmsnorm_func(True)
        rms_norm_func_resid(x, residual, weight, 1e-05)
        rms_norm_func = dispatch_cuda_rmsnorm_func(False)
        rms_norm_func(x, weight, 1e-05)

        del x
        del residual
    del weight

def cache_activation_kernels(device, config, tensor_parallel_size, real_device):
    input_sizes = range(config.input_start, config.input_end)
    for M in tqdm(input_sizes, **loginterval_sec, desc=f"[device:{real_device}] Caching Activation Kernels", unit="kernel"):
        x = torch.ones(config.get_act_shape(M, tensor_parallel_size), dtype=torch.bfloat16, device=device)
        act_mul_and_mxfp4_quant(x, "silu", shuffle=True)
        del x

def cache_dynamic_mxfp4_quant_asm_gemms(device, config, tensor_parallel_size, real_device):
    input_sizes = range(config.input_start, config.input_end, 32)
    for M in tqdm(input_sizes, **loginterval_sec, desc=f"[device:{real_device}] Caching dynamic_mxfp4_quant Kernels", unit="kernel"):
        x = torch.ones(config.get_mxfp4_quant_asm_shape(M), dtype=torch.bfloat16, device=device)
        dynamic_mxfp4_quant_asm(x, shuffle=(M >= 32))
        if tensor_parallel_size > 1:
            del x
            x = torch.ones(config.get_mxfp4_quant_asm_shape(M, tensor_parallel_size), dtype=torch.bfloat16, device=device)
            dynamic_mxfp4_quant_asm(x, shuffle=(M >= 32))
        del x

def cache_gemm_a4w4_asm_gemms(device, config, tensor_parallel_size, real_device):
    input_sizes = range(config.input_start, config.input_end, 32)
    layer_shapes = config.get_gemm_a4w4_asm_shapes(tensor_parallel_size)
    for input_size in tqdm(input_sizes, **loginterval_sec, desc=f"[device:{real_device}] Caching gemm_a4w4_asm Kernels", unit="kernel"):
        for layer_w_x, layer_w_y, layer_ws_y in layer_shapes:
            x_q = torch.ones((input_size, layer_w_y), dtype=torch.uint8, device=device)
            x_s = torch.ones((input_size, layer_ws_y), dtype=torch.uint8, device=device)
            layer_weight = torch.ones((layer_w_x, layer_w_y), dtype=torch.uint8, device=device)
            layer_weight_scale = torch.ones((layer_w_x, layer_ws_y), dtype=torch.uint8, device=device)
            y = torch.zeros((input_size, layer_w_x), dtype=torch.bfloat16, device=device)
            gemm_a4w4(x_q, layer_weight, x_s, layer_weight_scale, y, y, bpreshuffle=True)
            del x_q
            del x_s
            del layer_weight
            del layer_weight_scale
            del y
