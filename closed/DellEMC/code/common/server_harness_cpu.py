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
import math
sys.path.insert(0, os.getcwd())
from code.common import logging, dict_get, run_command, args_to_string
from code.common.harness import BaseBenchmarkHarness, benchmark_qsl_size_map
from code.common.submission import TRITON_VERSION

from functools import reduce
from code.common import logging, run_command, args_to_string, BENCHMARKS
import code.common.arguments as common_args

import numpy as np

TRITON_TF_CONFIG = """name: "{config_name}"
platform: "tensorflow_graphdef"
max_batch_size: {max_batch_size}
{io_info}
{dynamic_batching}
"""

TRITON_TF_DYNAMIC_BATCHING_FORMAT = """

dynamic_batching {{
    preferred_batch_size: {preferred_batch_size}
    max_queue_delay_microseconds: {max_queue_delay_usec}
    default_queue_policy {{
        timeout_action: DELAY
        default_timeout_microseconds: {request_timeout_usec}
    }}
}}
"""
TRITON_OV_CONFIG = """name: "{config_name}"
backend: "openvino"
max_batch_size: {max_batch_size}
{parameters}
{io_info}

instance_group {{
    count: {instance_group_count}
    kind: KIND_CPU
}}
{dynamic_batching}
"""

TRITON_OV_PARAMETERS = """
parameters: {{
    key: "{key}"
    value: {{
        string_value : "{value}"
    }}
}}
"""
# OV doesn't use preferred batch size?
TRITON_OV_DYNAMIC_BATCHING_FORMAT = """

dynamic_batching {{
    max_queue_delay_microseconds: {max_queue_delay_usec}
    default_queue_policy {{
        timeout_action: DELAY
        default_timeout_microseconds: {request_timeout_usec}
    }}
}}
"""

class TritonHarnessCPU(BaseBenchmarkHarness):

    def __init__(self, args, name=""):
        self.enable_interleaved = False
        self.is_int8 = args['precision'] == 'int8'
        # !is_tf signifies an OV model
        self.is_tf = args["config_ver"] == "tensorflow"
        super().__init__(args, name=name, skip_file_checks=True)
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS
        self.model_store_path = os.path.abspath("./build/model_repo")
        self.model_binaries = ["model.graphdef"] if self.is_tf else ["model.xml", "model.bin", "model.mapping"]
        self.abs_path = "/work/build/models/" + args["model_name"]
        self.model_name = args["model_name"]
        self.model_version = "1"
        self.map_path = args["map_path"] if "map_path" in args else None
        self.test_mode = args["test_mode"] if "test_mode" in args else None
        self.coalesced = args["coalesced_tensor"] if "coalesced_tensor" in args else None
        self.tensor_path = args["tensor_path"]

    def _get_harness_executable(self):
        return "./build/bin/harness_multi_triton_cpu" if self.is_tf else "./build/bin/harness_triton_cpu"

    def get_system_name(self):
        return super().get_system_name() + "_Triton" + TRITON_VERSION

    def build_default_flags(self):
        flag_dict = {}
        flag_dict["verbose"] = self.verbose

        # Generate flags for logfile names.
        log_dir = self.get_full_log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        flag_dict["logfile_outdir"] = log_dir
        flag_dict["logfile_prefix"] = "mlperf_log_"

        # Handle performance sample count
        perf_sample_count = dict_get(self.args, "performance_sample_count", None)
        if perf_sample_count is not None:
            flag_dict["performance_sample_count"] = perf_sample_count
        elif benchmark_qsl_size_map[self.name] > 0:
            flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self.name]
        else:
            flag_dict["performance_sample_count"] = self.args["gpu_batch_size"]

        # Handle custom arguments
        for arg in self.flag_builder_custom_args:
            val = dict_get(self.args, arg, None)
            if val is not None:
                flag_dict[arg] = val

        return flag_dict

    def _build_custom_flags(self, flag_dict):
        # Triton does not use gpu_engines flag
        flag_dict["gpu_engines"] = None

        # Force performance sample count
        flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self.name]

        flag_dict["model_store_path"] = self.model_store_path
        flag_dict["model_name"] = self.model_name
        flag_dict["model_version"] = self.model_version
        flag_dict["buffer_manager_thread_count"] = self.args.get("buffer_manager_thread_count", 0)
        flag_dict["pinned_input"] = True

        # TF-CPU multi-server run:
        if self.is_tf:
            flag_dict["num_instances"] = self.args.get("num_instances", 1)

        # Inform the server to use different QSL
        flag_dict["use_dlrm_qsl"] = (self.name == BENCHMARKS.DLRM)

        # Specify harness-specific flags here
        flag_dict["tensor_path"] = self.tensor_path
        if self.test_mode:
            flag_dict["test_mode"] = self.test_mode
        if self.map_path:
            flag_dict["map_path"] = self.map_path
        if self.coalesced:
            flag_dict["coalesced_tensor"] = self.coalesced

        self.setup_triton_model_repo()

        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario + " --model " + self.name

        # Assign proper callback function here
        if self.name == BENCHMARKS.ResNet50:
            argstr += " --response_postprocess tfrn50" if self.is_tf else " --response_postprocess ovrn50"
        elif self.name in [BENCHMARKS.SSDMobileNet, BENCHMARKS.SSDResNet34]:
            argstr += " --response_postprocess ovcoco"

        return argstr

    def _handle_harness_result(self, result):
        if self.name == BENCHMARKS.DLRM:
            partitions = np.load(os.path.expandvars(self.args["sample_partition_path"]))
            partition_mean_size = np.mean(partitions[1:] - partitions[:-1])

            # Attempt to calculate pairs per second metric
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", result)
            if len(nums) == 1:
                print("User-item pairs per second: {:.3f}".format(float(nums[0]) * partition_mean_size))

        return result

    def setup_triton_model_repo(self):
        # Create model dir where model is loaded from
        model_dir = os.path.join(self.model_store_path, self.model_name, self.model_version)
        os.makedirs(model_dir, exist_ok=True)

        # Create a sym link to model binaries
        for binary in self.model_binaries:
            dst = os.path.join(model_dir, binary)
            if os.path.exists(dst):
                os.remove(dst)
            bin_path = os.path.join(self.abs_path, self.model_version, binary)
            os.symlink(bin_path, dst)

        # Generate configs - common config args
        config = {}
        config["config_name"] = self.model_name
        config["max_batch_size"] = self.args["batch_size"]

        # TF config
        if self.is_tf:
            config["io_info"] = """input [
            {
                name: "input_tensor"
                data_type: TYPE_FP32
                format: FORMAT_NHWC
                dims: [ 224, 224, 3 ]
            }
            ]
            output [
            {
                name: "ArgMax"
                data_type: TYPE_INT32
                dims: [ 1 ]
            }
            ]"""
            config["preferred_batch_size"] = self.args.get("batch_size", 1) if self.args.get("preferred_batch_size") is None else self.args.get("preferred_batch_size")
            config["max_queue_delay_usec"] = self.args.get("max_queue_delay_usec", 1000000)
            config["request_timeout_usec"] = self.args.get("request_timeout_usec", 1000000000)
            config["dynamic_batching"] = TRITON_TF_DYNAMIC_BATCHING_FORMAT.format(**config)
            config_file_path = os.path.join(self.model_store_path, self.model_name, "config.pbtxt")
            with open(config_file_path, 'w') as f:
                f.write(TRITON_TF_CONFIG.format(**config))
        # OV config
        else:
            # Common OV args
            config["parameters"] = ""
            config["instance_group_count"] = self.args.get("num_instances", 1)
            parameters = self.args.get("ov_parameters")

            for p in parameters:
                parameter = {}
                parameter["key"] = p
                parameter["value"] = parameters[p]
                config["parameters"] += TRITON_OV_PARAMETERS.format(**parameter)

            config["max_queue_delay_usec"] = self.args.get("max_queue_delay_usec", 1000000)
            config["request_timeout_usec"] = self.args.get("request_timeout_usec", 1000000000)
            config["dynamic_batching"] = TRITON_OV_DYNAMIC_BATCHING_FORMAT.format(**config)

            # Populate I/O information based on the model
            if self.model_name == "resnet50_int8_openvino":
                config["io_info"] = """input [
                {
                    name: "input_tensor"
                    data_type: TYPE_FP32
                    format: FORMAT_NCHW
                    dims: [ 3, 224, 224 ]
                }
                ]
                output [
                {
                    name: "softmax_tensor"
                    data_type: TYPE_FP32
                    dims: [ 1001 ]
                }
                ]"""
            elif self.model_name == "ssd-resnet34_int8_openvino":
                config["io_info"] = """input [
                {
                    name: "image"
                    data_type: TYPE_FP32
                    format: FORMAT_NCHW
                    dims: [ 3, 1200, 1200 ]
                }
                ]
                output [
                {
                    name: "output"
                    data_type: TYPE_FP32
                    dims: [ 200, 7 ]
                }
                ]"""
            elif self.model_name == "3dunet_int8_openvino":
                config["io_info"] = """input [
                {
                    name: "input"
                    data_type: TYPE_FP32
                    dims: [ 1, 4, 224, 224,160 ]
                #    reshape { shape: [ 1, 3, 160, 224, 224 ] }
                }
                ]
                output [
                {
                    name: "output/add_"
                    data_type: TYPE_FP32
                    dims: [1,4,224,224,160]
                }
                ]"""
            elif self.model_name == "bert_int8_openvino":
                config["io_info"] = """input [
                {
                    name: "input_ids"
                    data_type: TYPE_INT32
                    dims: [ 1, 384 ]
                },
                {
                    name: "attention_mask"
                    data_type: TYPE_INT32
                    dims: [ 1, 384 ]
                },
                {
                    name: "token_type_ids"
                    data_type: TYPE_INT32
                    dims: [ 1, 384 ]
                }
                ]
                output [
                {
                    name: "6703"
                    data_type: TYPE_FP32
                    dims: [1,384,2]
                }
                ]"""

            # Write config.pbtxt
            config_file_path = os.path.join(self.model_store_path, self.model_name, "config.pbtxt")
            with open(config_file_path, 'w') as f:
                f.write(TRITON_OV_CONFIG.format(**config))
