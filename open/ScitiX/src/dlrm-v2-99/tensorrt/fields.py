# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from nvmitten.configurator import Field

import pathlib

from code.common.constants import Precision, InputFormats


__doc__ = """DLRMv2 fields"""


sample_partition_path = Field(
    "sample_partition_path",
    description="Path to sample partition file in npy format.",
    from_string=pathlib.Path)

embedding_weights_on_gpu_part = Field(
    "embedding_weights_on_gpu_part",
    description=("Percentage of the embedding weights to store on GPU. Lower this if "
                 "your GPU has lower VRAM. Default: 1.0 (100% of embedding weights on GPU)"),
    from_string=float)

qsl_numa_override = Field(
    "qsl_numa_override",
    description="Designate NUMA node(s) each QSL maps to; this overrides numa config for QSL")

num_staging_threads = Field(
    "num_staging_threads",
    description="Number of staging threads in DLRMv2 BatchMaker",
    from_string=int)

num_staging_batches = Field(
    "num_staging_batches",
    description="Number of staging batches in DLRMv2 BatchMaker",
    from_string=int)

max_pairs_per_staging_thread = Field(
    "max_pairs_per_staging_thread",
    description="Maximum pairs to copy in one BatchMaker staging thread",
    from_string=int)

gpu_num_bundles = Field(
    "gpu_num_bundles",
    description="Number of event+buffer bundles per GPU for DLRMv2 (default: 2)",
    from_string=int)

check_contiguity = Field(
    "check_contiguity",
    description="Check if inputs are already contiguous in QSL to avoid copying",
    from_string=bool)

embeddings_path = Field(
    "embeddings_path",
    description="Path to DLRMv2 embeddings in npy format.",
    from_string=pathlib.Path)

bot_mlp_precision = Field(
    'bot_mlp_precision',
    description='Precision to run bot-mlp in',
    from_string=Precision.get_match)

embeddings_precision = Field(
    'embeddings_precision',
    description='Precision to run embeddings in',
    from_string=Precision.get_match)

interaction_op_precision = Field(
    'interaction_op_precision',
    description='Precision to run interaction-op in',
    from_string=Precision.get_match)

top_mlp_precision = Field(
    'top_mlp_precision',
    description='Precision to run top-mlp in',
    from_string=Precision.get_match)

final_linear_precision = Field(
    'final_linear_precision',
    description='Precision to run final-linear in',
    from_string=Precision.get_match)
