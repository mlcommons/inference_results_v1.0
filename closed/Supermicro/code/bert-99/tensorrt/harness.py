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
import sys
sys.path.insert(0, os.getcwd())
from code.common import logging, dict_get, run_command, args_to_string
from code.common import BENCHMARKS, SCENARIOS
from code.common.harness import BaseBenchmarkHarness
import code.common.arguments as common_args
import pycuda
import pycuda.autoinit


class BertHarness(BaseBenchmarkHarness):
    """BERT harness."""

    def __init__(self, args, name=""):
        # TODO file check will not work for current way to pass multiple engines.
        #super().__init__(args, name)
        self.enable_interleaved = False
        self.is_int8 = args['precision'] == 'int8'
        if self.is_int8 and 'enable_interleaved' in args:
            self.enable_interleaved = args['enable_interleaved']
        super(BertHarness, self).__init__(args, name, skip_file_checks=True)
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS +\
            ["gpu_inference_streams", "gpu_copy_streams", "gpu_batch_size", "graphs_max_seqlen", "soft_drop", "server_num_issue_query_threads", "devices", "graph_specs"]

    def _get_harness_executable(self):
        """Return path to BERT harness binary."""
        return "./build/bin/harness_bert"

    def _build_custom_flags(self, flag_dict):
        return args_to_string(flag_dict) + " --scenario " + self.scenario + " --model " + self.name

    def _get_engine_fpath(self, device_type, batch_size):
        """Return file path of engine."""
        num_profiles = 1
        if 'gpu_inference_streams' in self.args:
            # use gpu_inference_streams to determine the number of duplicated profiles
            # in the engine when not using lwis mode
            num_profiles = self.args['gpu_inference_streams']

        seq_len = 384  # default sequence length
        engine_name = "{:}/{:}-{:}-{:}-{:}_S_{:}_B_{:}_P_{:}_vs{:}.{:}.plan".format(
            self.engine_dir, self.name, self.scenario,
            device_type, self.precision, seq_len, batch_size, num_profiles, '_il' if self.enable_interleaved else '', self.config_ver)

        return engine_name
