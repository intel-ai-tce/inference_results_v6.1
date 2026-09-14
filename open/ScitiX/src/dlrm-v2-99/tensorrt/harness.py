# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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

"""DLRMv2 TensorRT Harness Module.

This module provides the harness implementation for running DLRMv2 (Deep Learning Recommendation Model v2)
benchmarks using TensorRT. It handles the configuration, execution, and result processing for the DLRMv2 model.
"""

from code.common.constants import Precision
from code.common.systems.system_list import DETECTED_SYSTEM
from code.ops.harness import ExecutableHarness, LWISFlags
from code.ops.generate_engines import EngineBuilderOp
from code.ops.loadgen import LoadgenConfFilesOp
from nvmitten.configurator import bind, autoconfigure
from nvmitten.pipeline.resource import Resource, GETableSource, MD5Checksum
from pathlib import Path
from typing import Optional

import dataclasses as dcls
import logging
import numpy as np
import os
import re

from .criteo import CriteoDay23Dataset, convert_sample_partition_to_npy
from . import fields as dlrm_fields
import code.fields.models as model_fields
import code.fields.harness as harness_fields


DLRM_V2_SCRATCH_PATH = Path("/home/mlperf_inf_dlrmv2")


@autoconfigure
@bind(dlrm_fields.sample_partition_path)
@bind(dlrm_fields.num_staging_threads)
@bind(dlrm_fields.num_staging_batches)
@bind(dlrm_fields.max_pairs_per_staging_thread)
@bind(dlrm_fields.gpu_num_bundles)
@bind(dlrm_fields.check_contiguity)
@bind(dlrm_fields.qsl_numa_override)
@dcls.dataclass(frozen=True)
class DLRMv2Flags(LWISFlags):
    """Configuration flags for DLRMv2 harness execution.

    This class defines the configuration parameters specific to DLRMv2 benchmark execution,
    including paths, thread counts, and other runtime parameters.

    Attributes:
        sample_partition_path (Path): Path to the sample partition file.
        num_staging_threads (int): Number of threads used for staging data.
        num_staging_batches (int): Number of batches to stage.
        max_pairs_per_staging_thread (int): Maximum number of pairs per staging thread.
        gpu_num_bundles (int): Number of GPU bundles to use.
        check_contiguity (bool): Whether to check for contiguity in data.
        qsl_numa_override (Optional[str]): Optional NUMA override for QSL.
    """
    sample_partition_path: Path = DLRM_V2_SCRATCH_PATH / "criteo/day23/sample_partition.npy"
    num_staging_threads: int = 8
    num_staging_batches: int = 4
    max_pairs_per_staging_thread: int = 0
    gpu_num_bundles: int = 2
    check_contiguity: bool = False
    qsl_numa_override: Optional[str] = None


@autoconfigure
@bind(model_fields.input_dtype)
class DLRMv2Harness(ExecutableHarness):
    """DLRMv2 benchmark harness implementation.

    This class implements the harness for running DLRMv2 benchmarks using TensorRT.
    It handles the execution of the benchmark, including data loading, model inference,
    and result processing.
    """

    @classmethod
    def immediate_dependencies(cls):
        """Get the immediate dependencies required for this harness.

        Returns:
            set: Set of dependency classes (EngineBuilderOp, LoadgenConfFilesOp).
        """
        return {EngineBuilderOp, LoadgenConfFilesOp}

    @classmethod
    def output_keys(self):
        """Get the output keys produced by this harness.

        Returns:
            list: List of output keys ["log_dir", "result_metadata"].
        """
        return ["log_dir", "result_metadata"]

    def __init__(self, input_dtype: Precision = Precision.FP16):
        """Initialize the DLRMv2 harness.

        Args:
            input_dtype (Precision): The precision to use for input data (default: FP16).
        """
        super().__init__(executable_fpath="./build/bin/harness_dlrm_v2")

        self.input_dtype = input_dtype
        self.conf = DLRMv2Flags()
        self.generate_required_files()

    def build_flags(self, user_conf, engine_index):
        """Build the configuration flags for the harness execution.

        Args:
            user_conf: User configuration.
            engine_index: Engine index information.

        Returns:
            dict: Dictionary of configuration flags.
        """
        flags = super().build_flags(user_conf, engine_index)
        flags.update(self.conf.asdict())

        # DLRMv2 harness does not support GBS / multi-engine
        if "gpu_engine_batch_size" in flags:
            flags.pop("gpu_engine_batch_size")

        if "devices" in flags:
            flags.pop("devices")

        if "coalesced_tensor" in flags:
            flags.pop("coalesced_tensor")

        flags["scenario"] = engine_index.wl.scenario.valstr
        flags["model"] = engine_index.wl.benchmark.valstr
        return flags

    def run(self, scratch_space, dependency_outputs):
        """Execute the benchmark and process results.

        Args:
            scratch_space: Scratch space for temporary files.
            dependency_outputs: Outputs from dependencies.

        Returns:
            dict: Dictionary containing benchmark results and metadata.
        """
        d = super().run(scratch_space, dependency_outputs)
        if self.test_mode == "PerformanceOnly":
            partitions = np.load(self.conf.sample_partition_path)
            mu = np.mean(partitions[1:] - partitions[:-1])
            d["result_metadata"]["dlrm_partition_mean_size"] = mu

            k = d["result_metadata"]["scenario_key"]
            d["result_metadata"]["dlrm_pairs_per_second"] = d["result_metadata"][k] * mu
        return d

    def generate_required_files(self):
        """Generate required input files for the benchmark.

        This method ensures all necessary input files exist, including:
        - Dense input files in the specified precision
        - Sparse input concatenated files
        - Sample partition files

        If any files are missing, they will be generated using the CriteoDay23Dataset.
        """
        if "is_soc" in DETECTED_SYSTEM.extras["tags"]:
            logging.warning("SoC does not support DLRMv2! Bypass DLRMv2 on SoC systems...")
            return

        tensor_paths = self.conf.tensor_path.split(',')
        assert len(tensor_paths) == 2, "DLRMv2 requires 2 input tensor files"

        dense_input_filepath = Path(tensor_paths[0])
        sparse_input_filepath = Path(tensor_paths[1])

        # load dataset to generate missing files only if needed, takes a long time
        if (not dense_input_filepath.exists()) or (not sparse_input_filepath.exists()):
            ds = CriteoDay23Dataset()

        # generate lower precision dense input files if needed
        if not dense_input_filepath.exists():
            # Create the file
            logging.info(f"Dense input file does not exist for precision: {self.input_dtype}. Generating...")
            ds.dump_dense_input(self.input_dtype.valstr.lower())
            logging.info(f"Generated dense input file for precision: {self.input_dtype}.")
        else:
            logging.info(f"Found dense input file for precision: {self.input_dtype}.")

        # generate sparse input concat file if needed
        if not sparse_input_filepath.exists():
            # Create the file
            logging.info("Coalesced sparse input file does not exist. Generating...")
            ds.dump_concatenated_sparse_input()
            logging.info("Generated coalesced sparse inputs.")
        else:
            logging.info("Found coalesced sparse input file.")

        # Get sample_partition file
        txt_path = self.conf.sample_partition_path.with_suffix(".txt"),
        if not self.conf.sample_partition_path.exists():
            sample_partition = Resource(
                txt_path,
                source_url=GETableSource("https://zenodo.org/record/3941795/files/dlrm_trace_of_aggregated_samples.txt"),
                checksum=MD5Checksum("3db90209564316f2506c99cc994ad0b2"))
            sample_partitition.create()
            convert_sample_partition_to_npy(txt_path)
        else:
            logging.info("Found sample partition file.")
