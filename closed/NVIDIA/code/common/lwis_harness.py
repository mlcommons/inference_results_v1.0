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

import re
import os
import sys
sys.path.insert(0, os.getcwd())

from code.common import BENCHMARKS, SCENARIOS, logging, args_to_string
from code.common.harness import BaseBenchmarkHarness
import code.common.arguments as common_args

response_postprocess_map = {
    BENCHMARKS.SSDResNet34: "coco",
    BENCHMARKS.SSDMobileNet: "coco"
}


class LWISHarness(BaseBenchmarkHarness):

    def __init__(self, args, name=""):
        super().__init__(args, name=name)

        self.use_jemalloc = (self.scenario == SCENARIOS.Server)
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.LWIS_ARGS + common_args.SHARED_ARGS

    def _get_harness_executable(self):
        return "./build/bin/harness_default"

    def _build_custom_flags(self, flag_dict):
        if self.has_dla:
            flag_dict["dla_engines"] = self.dla_engine

        if self.has_gpu and self.has_dla:
            pass
        elif self.has_gpu:
            flag_dict["max_dlas"] = 0
        elif self.has_dla:
            flag_dict["max_dlas"] = 1
        else:
            raise ValueError("Cannot specify --no_gpu and --gpu_only at the same time")

        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario + " --model " + self.name

        if self.name in response_postprocess_map:
            argstr += " --response_postprocess " + response_postprocess_map[self.name]

        return argstr
