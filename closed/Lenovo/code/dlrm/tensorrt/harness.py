# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os
import re
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from code.common import logging, dict_get, run_command, args_to_string
from code.common import BENCHMARKS, SCENARIOS
from code.common.harness import BaseBenchmarkHarness
import code.common.arguments as common_args


class DLRMHarness(BaseBenchmarkHarness):
    """DLRM benchmark harness."""

    def __init__(self, args, name=""):
        super().__init__(args, name)
        custom_args = [
            "gpu_copy_streams",
            "complete_threads",
            "sample_partition_path",
            "warmup_duration",
            "gpu_inference_streams",
            "num_staging_threads",
            "num_staging_batches",
            "max_pairs_per_staging_thread",
            "gpu_num_bundles",
            "check_contiguity",
            "start_from_device",
            "use_jemalloc",
        ]
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS + custom_args

    def _get_harness_executable(self):
        return "./build/bin/harness_dlrm"

    def _build_custom_flags(self, flag_dict):
        # Handle use_jemalloc
        self.use_jemalloc = dict_get(flag_dict, "use_jemalloc", False)
        flag_dict['use_jemalloc'] = None
        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario + " --model " + self.name
        return argstr

    def _handle_harness_result(self, result):
        """Parse result from harness and return it."""

        partitions = np.load(os.path.expandvars(self.args["sample_partition_path"]))
        partition_mean_size = np.mean(partitions[1:] - partitions[:-1])
        # Attempt to calculate pairs per second metric
        nums = re.findall(r"[-+]?\d+\.?\d*e?[-+]?\d*", result)
        if len(nums) == 1:
            print("User-item pairs per second: {:.3f}".format(float(nums[0]) * partition_mean_size))

        return result
