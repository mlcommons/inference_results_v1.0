#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

# Nasty hack to get BenchmarkBuilder to come peacefully (so we don't need to import trt)
# We occupy the "tensorrt" namespace with an opaque, no-op module.
from unittest.mock import MagicMock
sys.modules['tensorrt'] = MagicMock()

from code.common import logging, dict_get, BENCHMARKS
from code.common.builder import BenchmarkBuilder
from code.common.harness import BaseBenchmarkHarness


class ResNet50Builder(BenchmarkBuilder):
    def __init__(self, config):
        pass

    def initialize(self):
        pass

    def build_engines(self):
        # We don't have any engines to build, so No-Op
        pass


class ResNet50Harness(BaseBenchmarkHarness):
    def __init__(self, args):

        assert args['precision'] == 'int4'
        # Fill up some reqired argments to not have to change the over-arching BaseBenchmark:
        # and that are too "boilerplate-y" to add to the config.json
        args['input_dtype'] = "fp32"
        args['performance_sample_count'] = 1024

        super(ResNet50Harness, self).__init__(args, name=BENCHMARKS.ResNet50, skip_file_checks=True)
        self.autotuningFile = args['autotuningFile']
        self.batch_size = args['gpu_batch_size']

        #self.system_id = args['system_id']

    def get_system_name(self):
        return super().get_system_name(add_trt=False)

    def _get_harness_executable(self):
        return "./code/resnet50/int4/int4_offline"

    def _get_engine_fpath(self, device_type, batch_size):
        # We don't run on TRT, but we do have some breadcrumbs used during inference:
        return "./code/resnet50/int4/model/model"

    def _build_custom_flags(self, flag_dict):
        # We have some interesting renaming:
        # Format, src:dest
        def rename_config_dict(d, src_dest_remappings):
            for src_name, dest_name in src_dest_remappings.items():
                if dest_name:
                    d[dest_name] = d[src_name]
                del d[src_name]
        rename_dict = {
            "logfile_outdir": "lgls_logfile_outdir",
            "logfile_prefix": "lgls_logfile_prefix",
            "performance_sample_count": None,
            "gpu_engines": None,
            "verbose": None,
            "offline_expected_qps": None,
            "performance_sample_count_override": None,
        }
        rename_config_dict(flag_dict, rename_dict)
        # Now we add other flags we want
        flag_dict['batch_size'] = self.batch_size
        if 'test_mode' in self.args:
            flag_dict['test-mode'] = self.args['test_mode']
        # A bit hacky: The harness has some hardcoded values that expects a known working dir
        # Set that working dir, and make all other paths absolute ()
        flag_dict['workingPath'] = os.path.abspath("./code/resnet50/int4/")
        flag_dict['tensorPath'] = os.path.abspath("build/preprocessed_data/imagenet/ResNet50_int4")
        flag_dict['mapPath'] = os.path.abspath("data_maps/imagenet/val_map.txt")
        flag_dict['autotuningFile'] = os.path.abspath(self.autotuningFile)
        flag_dict['mlperf_conf_path'] = os.path.abspath(flag_dict['mlperf_conf_path'])
        flag_dict['user_conf_path'] = os.path.abspath(flag_dict['user_conf_path'])

        # We also can't use the default behavior of "--arg=argval", because the harness wants "--arg argval"
        # Really, we should just overhaul the harness to use gflags, but that's a heavy hammer when it aready works
        args = []
        for arg, val in flag_dict.items():
            args.append(f"--{arg} {val}")

        return " ".join(args)
