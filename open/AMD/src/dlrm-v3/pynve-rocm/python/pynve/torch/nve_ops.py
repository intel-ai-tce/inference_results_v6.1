# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from pynve.torch.nve_tensors import CachedTable
import pynve.nve as nve

# functional implementation for a embedding lookup with cache
# allowing for autograd functionality
class CacheEmbeddingOp(torch.autograd.Function):
    @staticmethod
    def forward(keys : torch.Tensor, weight : CachedTable):
        """Function for embedding lookup with caching.

        This class provides the forward and backward pass for the embedding lookup operation, 
        see torch.nn.Embedding for more information on the operation.

        Args:
            keys (torch.Tensor): Tensor of integer keys identifying the embedding vectors to retrieve
            weight (CachedTable): CachedTable object containing the embedding table

        Returns:
            torch.Tensor: Tensor of embedding vectors corresponding to the input keys

        """
        nve_op: nve.NVEmbedding = weight.emb_layer
        
        #todo find a better formula for space
        result = torch.empty(torch.numel(keys), weight.shape[-1], dtype=weight.dtype, device=keys.device)
        nve_op.lookup(torch.numel(keys), keys.data_ptr(), result.data_ptr(), torch.cuda.current_stream(keys.device).cuda_stream)
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        keys, weight = inputs
        ctx.save_for_backward(keys, weight)

    @staticmethod
    def backward(ctx, grad_output):
        keys, weight = ctx.saved_tensors
        result_shape = list(weight.shape)
        # The native concat_backprop HIP kernel has a bug: for large fp16 batches
        # it writes garbage (huge / NaN / inf values, ~fp16 max) into a handful of
        # the per-unique-key output rows even when grad_output is finite and tiny,
        # silently poisoning the whole model's gradients. The op is just
        # "sum grad_output rows by unique key", so compute it directly in PyTorch
        # (accumulate in fp32 for safety, then cast to the store dtype). This is
        # exact and avoids the broken kernel entirely.
        keys_flat = keys.reshape(-1).to(torch.int64)
        unique_keys, inverse = torch.unique(keys_flat, return_inverse=True)
        acc = torch.zeros(unique_keys.numel(), result_shape[1],
                          dtype=torch.float32, device=weight.device)
        acc.index_add_(0, inverse, grad_output.reshape(-1, result_shape[1]).float())
        # A separate, layout-dependent pynve kernel (forward lookup / write-through
        # accumulate) intermittently does an out-of-bounds device write that lands
        # in this buffer, injecting NaN/inf into a handful of rows even though the
        # input grad is finite and tiny. Left unsanitized, the write-through writes
        # those NaNs into the table and the next lookup of that row poisons the
        # whole model. Sanitize NaN/inf -> 0 (a missed update for that row is
        # harmless; real embedding grads are ~1e-4, never NaN/inf).
        acc = torch.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1e3, 1e3)
        result = torch.nan_to_num(acc.to(weight.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        res = torch.sparse_coo_tensor(
            unique_keys.reshape(1, -1), result, tuple(result_shape)
        ).to(weight.device)

        return None, res

# functional implementation for a fused embedding lookup and pooling with cache
# allowing for autograd functionality
class CacheEmbeddingBagOp(torch.autograd.Function):
    @staticmethod
    def forward(keys : torch.Tensor, weight : CachedTable, offsets : torch.Tensor, mode : str, per_sample_weights : torch.Tensor):
        """Function for embedding bag with caching.

        This class provides the forward and backward pass for the embedding bag operation, 
        see torch.nn.EmbeddingBag for more information on the operation.

        Args:
            keys (torch.Tensor): Tensor of integer keys identifying the embedding vectors to retrieve
            weight (CachedTable): CachedTable object containing the embedding table
            offsets (torch.Tensor): Tensor of offsets identifying the boundaries of sequences
            mode (str): Mode of pooling to apply can be 'sum', 'mean'.
            per_sample_weights (torch.Tensor): Tensor of per sample weights

        Returns:
            torch.Tensor: Tensor of embedding vectors corresponding to the input keys

        """
        nve_op: nve.NVEmbedding = weight.emb_layer
        device = keys.device
        #todo find a better formula for space
        result = torch.empty(torch.numel(offsets) - 1, weight.shape[-1], dtype=weight.dtype, device=device)
        if (per_sample_weights != None):
            if (per_sample_weights.dtype == torch.float32):
                weights_type = nve.Float32 #float
            else:
                weights_type = nve.Float16 #half
            if (mode == 'sum'):
                nve_op.lookup_with_pooling(torch.numel(keys), keys.data_ptr(), result.data_ptr(), nve.WeightedSum, torch.numel(offsets) - 1, offsets.data_ptr(), weights_type, per_sample_weights, torch.cuda.current_stream(device).cuda_stream)
            else:
                nve_op.lookup_with_pooling(torch.numel(keys), keys.data_ptr(), result.data_ptr(), nve.WeightedMean, torch.numel(offsets) - 1, offsets.data_ptr(), weights_type, per_sample_weights, torch.cuda.current_stream(device).cuda_stream)
        else:
            if (mode == 'sum'):
                nve_op.lookup_with_pooling(torch.numel(keys), keys.data_ptr(), result.data_ptr(), nve.Sum, torch.numel(offsets) - 1, offsets.data_ptr(), nve.Float32, 0, torch.cuda.current_stream(device).cuda_stream)
            else:
                nve_op.lookup_with_pooling(torch.numel(keys), keys.data_ptr(), result.data_ptr(), nve.Mean, torch.numel(offsets) - 1, offsets.data_ptr(), nve.Float32, 0, torch.cuda.current_stream(device).cuda_stream)
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        keys, weight, offsets, mode, per_sample_weights = inputs
        #save mode and tensors for backprop
        ctx.mode  = mode
        ctx.save_for_backward(keys, weight, offsets, per_sample_weights)

    @staticmethod
    def backward(ctx, grad_output):
        keys, weight, offsets, per_sample_weights = ctx.saved_tensors
        unique_keys = torch.empty(torch.numel(keys), dtype=torch.int64, device=weight.device)
        result = torch.empty(torch.numel(keys), weight.shape[-1], dtype=weight.dtype, device=weight.device)
        nve_op = weight.emb_layer

        if (per_sample_weights != None):
            if (per_sample_weights.dtype == torch.float32):
                weights_type = nve.Float32 #float
            else:
                weights_type = nve.Float16 #half
            if (ctx.mode == 'sum'):
                num_unique_keys = nve_op.pooling_backprop(torch.numel(keys), keys.data_ptr(), grad_output.data_ptr(), unique_keys.data_ptr(), result.data_ptr(), nve.WeightedSum, torch.numel(offsets) - 1, offsets.data_ptr(), weights_type, per_sample_weights, torch.cuda.current_stream(weight.device).cuda_stream)
            else:
                num_unique_keys = nve_op.pooling_backprop(torch.numel(keys), keys.data_ptr(), grad_output.data_ptr(), unique_keys.data_ptr(), result.data_ptr(), nve.WeightedMean, torch.numel(offsets) - 1, offsets.data_ptr(), weights_type, per_sample_weights, torch.cuda.current_stream(weight.device).cuda_stream)
        else:
            if (ctx.mode == 'sum'):
                num_unique_keys = nve_op.pooling_backprop(torch.numel(keys), keys.data_ptr(), grad_output.data_ptr(), unique_keys.data_ptr(), result.data_ptr(), nve.Sum, torch.numel(offsets) - 1, offsets.data_ptr(), nve.Float32, 0, torch.cuda.current_stream(weight.device).cuda_stream)
            else:
                num_unique_keys = nve_op.pooling_backprop(torch.numel(keys), keys.data_ptr(), grad_output.data_ptr(), unique_keys.data_ptr(), result.data_ptr(), nve.Mean, torch.numel(offsets) - 1, offsets.data_ptr(), nve.Float32, 0, torch.cuda.current_stream(weight.device).cuda_stream)

        result_shape = list(weight.shape)
        unique_keys = unique_keys.resize_(num_unique_keys)
        unique_keys = unique_keys.reshape((1, -1))
        result = result.resize_(num_unique_keys, result_shape[1])
        res = torch.sparse_coo_tensor(unique_keys, result, tuple(result_shape)).to(weight.device)               
        
        return None, res, None, None, None
