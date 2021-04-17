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
from code.common.harness import BaseBenchmarkHarness, plugin_map, benchmark_qsl_size_map
from code.common.runner import EngineRunner
from code.common.submission import TRITON_VERSION

from functools import reduce
from code.common import logging, run_command, args_to_string, BENCHMARKS
import code.common.arguments as common_args

import numpy as np

TRITON_CONFIG_FILE_FORMAT = """name: "{config_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch_size}

{io_tensors}
{batch_inputs}
{batch_outputs}

instance_group {{
    name: "{instance_group_name}"
    count: {instance_group_count}
    kind: KIND_GPU
    profile: [{optimization_profiles}]
}}

{dynamic_batching}

optimization {{
    cuda {{
        graphs: {cuda_graph}
        {graph_spec}
        {stream_spec}
    }}
    
    output_pinned_memory {{
        enable : {output_pinned}
    }}

    gather_kernel_buffer_threshold : {gather_kernel_buffer_threshold}

}}


default_model_filename: "{model_filename}"
version_policy {{ all {{ }}}}
"""

TRITON_TENSOR_CONFIG_FORMAT = """

{io_type} {{
    name: "{name}"
    data_type: {dtype}
    dims: {dims}
    {additional_config}
}}
"""

TRITON_BATCH_INPUT_CONFIG_FORMAT = """

batch_input [{{
    kind : {kind}
    target_name: "{name}"
    data_type: {dtype}
    source_input: "{source_name}"
}}]
"""

TRITON_BATCH_OUTPUT_CONFIG_FORMAT = """

batch_output [{{
    kind : {kind}
    target_name: "{name}"
    source_input: "{source_name}"
}}]
"""

TRITON_GRAPH_SPEC_FORMAT = """

{graph_spec_name} {{
    batch_size: {batch_size}
    {graph_spec_inputs}
    {graph_lower_bound}
}}
"""

TRITON_GRAPH_SPEC_INPUT_FORMAT = """

input {{
    key: "{name}"
    value: {{ dim: {dims} }}
}}
"""

TRITON_DYNAMIC_BATCHING_FORMAT = """

dynamic_batching {{
    preferred_batch_size: {preferred_batch_size}
    max_queue_delay_microseconds: {max_queue_delay_usec}
    default_queue_policy {{
        timeout_action: DELAY
        default_timeout_microseconds: {request_timeout_usec}
    }}
}}
"""


class TritonHarness(BaseBenchmarkHarness):

    def __init__(self, args, name=""):
        self.is_int8 = args.get('precision', "fp32") == 'int8'
        self.enable_interleaved = False
        if self.is_int8:
            self.enable_interleaved = args.get('enable_interleaved', False)

        super().__init__(args, name=name)

        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS + ["coalesced_tensor", "sample_partition_path", "start_from_device"]
        # FIXME Adding deque_timeout_us from LWIS for Multi-MIG Server, to easily share configs
        self.flag_builder_custom_args.append("deque_timeout_us")
        self.model_store_path = os.path.abspath("./build/model_repo")
        self.model_name = "model"
        self.model_version = "1"

    def _get_harness_executable(self):
        num_mig_match = re.search("MIG_(\d+)x", self.system_id)
        if num_mig_match and int(num_mig_match.group(1)) > 1:
            return "./build/bin/harness_triton_mig"
        return "./build/bin/harness_triton"

    def _get_engine_fpath(self, device_type, batch_size):
        batch_sizes = [batch_size]  # start with the standard batch_size argument
        if 'batch_sizes' in self.args:
            batch_sizes = self.args.get('batch_sizes', [1])
        if self.name != BENCHMARKS.BERT:
            return super()._get_engine_fpath(device_type, batch_size)

        seq_lens = [384]  # default sequence length

        sstr = '_'.join([str(x) for x in seq_lens])
        bstr = '_'.join([str(x) for x in batch_sizes])

        # use gpu_inference_streams to determine the number of duplicated profiles
        # in the engine when not using lwis mode
        num_profiles = self.args.get('gpu_inference_streams', 1)

        engine_name = ','.join("{:}/{:}-{:}-{:}-{:}_S_{:}_B_{:}_P_{:}_vs{:}.{:}.plan".format(
            self.engine_dir, self.name, self.scenario,
            device_type, self.precision, sstr, bstr, num_profiles, '_il' if self.enable_interleaved else '', self.config_ver) for S_ in seq_lens)

        return engine_name

    def _append_config_ver_name(self, system_name):
        system_name += "_Triton" + TRITON_VERSION
        return super()._append_config_ver_name(system_name)

    def _build_custom_flags(self, flag_dict):
        # Triton does not use gpu_engines flag
        flag_dict["gpu_engines"] = None

        # Force performance sample count
        flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self.name]

        # Server harness binary assumes GPU and uses --batch_size instead of --gpu_batch_size
        flag_dict["batch_size"] = flag_dict["gpu_batch_size"]
        flag_dict["gpu_batch_size"] = None

        engine_info = self.get_engine_info()
        flag_dict["model_store_path"] = self.model_store_path
        flag_dict["model_name"] = self.model_name
        flag_dict["model_version"] = self.model_version
        flag_dict["buffer_manager_thread_count"] = self.args.get("buffer_manager_thread_count", 0)
        flag_dict["pinned_input"] = True if flag_dict["buffer_manager_thread_count"] == 0 else False
        flag_dict["batch_triton_requests"] = self.args.get("batch_triton_requests", False)

        # Inform the server to use different QSL
        flag_dict["use_dlrm_qsl"] = (self.name == BENCHMARKS.DLRM)

        # Set up Triton model repo
        self.setup_triton_model_repo(engine_info)

        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario + " --model " + self.name

        if self.name in [BENCHMARKS.SSDMobileNet, BENCHMARKS.SSDResNet34]:
            argstr += " --response_postprocess coco"

        return argstr

    def _handle_harness_result(self, result):
        if self.name == BENCHMARKS.DLRM:
            partitions = np.load(os.path.expandvars(self.args.get("sample_partition_path", "")))
            partition_mean_size = np.mean(partitions[1:] - partitions[:-1])

            # Attempt to calculate pairs per second metric
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", result)
            if len(nums) == 1:
                print("User-item pairs per second: {:.3f}".format(float(nums[0]) * partition_mean_size))

        return result

    def get_engine_info(self):
        if self.verbose:
            logging.info("Loading engine to get engine info")

        def extract_dtype(s):
            if "INT8" in s:
                return "TYPE_INT8"
            elif "FP32" in s:
                return "TYPE_FP32"
            elif "INT32" in s:
                return "TYPE_INT32"
            elif "FP16" in s:
                return "TYPE_FP16"
            else:
                raise ValueError("Data type must be INT8 or FP32 or INT32, got {:}".format(s))

        format_rgx = re.compile(r"\(k[A-Z]+[0-9]*\)")

        # EngineRunner is the convention to load engines
        plugins = None
        if self.name in plugin_map:
            plugins = plugin_map[self.name]
            for plugin in plugins:
                self.check_file_exists(plugin)
        runner = EngineRunner(self.gpu_engine, verbose=self.verbose, plugins=plugins)
        inputs = []
        outputs = []
        # FIXME exploit the use of optimization profile if needed
        num_profiles = runner.engine.num_optimization_profiles
        num_bindings_per_profile = runner.engine.num_bindings // num_profiles
        has_dynamic_shape = False
        for idx in range(num_bindings_per_profile):
            tensor = {}
            tensor["name"] = runner.engine.get_binding_name(idx)
            binding_shape = runner.engine.get_binding_shape(idx)
            if -1 in binding_shape:
                tensor["dims"] = binding_shape[1:]
                has_dynamic_shape = True
            else:
                tensor["dims"] = binding_shape
            tensor["format"] = runner.engine.get_binding_format_desc(idx)
            tensor["dtype"] = extract_dtype(tensor["format"])
            match = format_rgx.search(tensor["format"])
            if match is None:
                raise ValueError("Invalid input format: {:}".format(tensor["format"]))
            tensor["dformat"] = match.group(0).strip("()")
            if runner.engine.binding_is_input(idx):
                inputs.append(tensor)
            else:
                outputs.append(tensor)

        is_static = not has_dynamic_shape and not runner.engine.has_implicit_batch_dimension

        # Clean up runner
        del runner

        return (inputs, outputs, [0], is_static)

    def setup_triton_model_repo(self, engine_info):
        model_dir = os.path.join(self.model_store_path, self.model_name, self.model_version)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Create a sym link to the model repo that points to the engine file
        engine_file_name = os.path.basename(self.gpu_engine)
        dst = os.path.join(model_dir, engine_file_name)
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(self.gpu_engine), dst)

        # Write Triton Config file
        config = {}
        config["config_name"] = self.model_name

        config["cuda_graph"] = "true" if self.args.get("use_graphs", False) else "false"
        config["graph_spec"] = ""

        config["io_tensors"] = ""
        config["batch_inputs"] = ""
        config["batch_outputs"] = ""

        config["output_pinned"] = "false" if self.scenario == "SingleStream" else "true"
        config["gather_kernel_buffer_threshold"] = self.args.get("gather_kernel_buffer_threshold", 0)

        # special handling for BERT
        if self.name == BENCHMARKS.BERT:
            self.set_bert_cuda_graph_specs(config)
            for tensor in engine_info[0]:
                if "ids" in tensor["name"]:
                    tensor_config = {}
                    tensor_config["io_type"] = "input"
                    tensor_config["name"] = tensor["name"]
                    tensor_config["dtype"] = tensor["dtype"]
                    tensor_config["dims"] = [-1]
                    tensor_config["additional_config"] = "allow_ragged_batch: true"
                    config["io_tensors"] += TRITON_TENSOR_CONFIG_FORMAT.format(**tensor_config)
                else:
                    tensor_config = {}
                    tensor_config["name"] = tensor["name"]
                    tensor_config["dtype"] = tensor["dtype"]
                    tensor_config["source_name"] = "input_ids"
                    if "max" in tensor["name"]:
                        tensor_config["kind"] = "BATCH_MAX_ELEMENT_COUNT_AS_SHAPE"
                    else:
                        tensor_config["kind"] = "BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO"
                    config["batch_inputs"] += TRITON_BATCH_INPUT_CONFIG_FORMAT.format(**tensor_config)
            for tensor in engine_info[1]:
                tensor_config = {}
                tensor_config["io_type"] = "output"
                tensor_config["name"] = tensor["name"]
                tensor_config["dtype"] = tensor["dtype"]
                # Add additional -1 dims for scattering output
                tensor_config["dims"] = [-1, ] + list(tensor["dims"])
                tensor_config["additional_config"] = ""
                config["io_tensors"] += TRITON_TENSOR_CONFIG_FORMAT.format(**tensor_config)

                batch_output_config = {}
                batch_output_config["name"] = tensor["name"]
                batch_output_config["kind"] = "BATCH_SCATTER_WITH_INPUT_SHAPE"
                batch_output_config["source_name"] = "input_ids"
                config["batch_outputs"] += TRITON_BATCH_OUTPUT_CONFIG_FORMAT.format(**batch_output_config)

        else:
            for tensor in engine_info[0]:
                tensor_config = {}
                tensor_config["io_type"] = "input"
                tensor_config["name"] = tensor["name"]
                tensor_config["dtype"] = tensor["dtype"]
                tensor_config["dims"] = list(tensor["dims"])
                tensor_config["additional_config"] = ""
                config["io_tensors"] += TRITON_TENSOR_CONFIG_FORMAT.format(**tensor_config)
            for tensor in engine_info[1]:
                tensor_config = {}
                tensor_config["io_type"] = "output"
                tensor_config["name"] = tensor["name"]
                tensor_config["dtype"] = tensor["dtype"]
                tensor_config["dims"] = list(tensor["dims"])
                tensor_config["additional_config"] = ""
                config["io_tensors"] += TRITON_TENSOR_CONFIG_FORMAT.format(**tensor_config)

        config["instance_group_name"] = self.model_name
        config["instance_group_count"] = self.args.get("instance_group_count", 1)
        config["optimization_profiles"] = ",".join(['"' + str(x) + '"' for x in engine_info[2]])

        config["preferred_batch_size"] = self.args.get("gpu_batch_size", 1) if self.args.get("preferred_batch_size") is None else self.args.get("preferred_batch_size")
        config["max_queue_delay_usec"] = self.args.get("max_queue_delay_usec", 1000000)
        config["request_timeout_usec"] = self.args.get("request_timeout_usec", 1000000000)

        config["model_filename"] = engine_file_name

        is_static = engine_info[3]
        config["max_batch_size"] = 0 if is_static else self.args.get("gpu_batch_size", 1)
        config["dynamic_batching"] = "" if is_static else TRITON_DYNAMIC_BATCHING_FORMAT.format(**config)
        # By default triton has one copy stream to copy the input and one inference stream that also copies out the output
        # In the cases where the ninja harness uses more than one copy stream, it is beneficial to use a separate stream
        # for output copy in triton
        config["stream_spec"] = "output_copy_stream: true" if self.args.get("gpu_copy_streams", 1) > 1 else ""
        config_file_path = os.path.join(self.model_store_path, self.model_name, "config.pbtxt")
        with open(config_file_path, 'w') as f:
            f.write(TRITON_CONFIG_FILE_FORMAT.format(**config))

    def set_bert_cuda_graph_specs(self, config):
        max_seq_len = 384
        start = self.args.get("gpu_batch_size", 1)
        end = max_seq_len * self.args.get("gpu_batch_size", 1)
        # FIXME 400 is hard-coded, should calculate based on device memory
        step = max(math.floor((end - start) / (400 - 2)), 1)
        graph_lower_bound = {
            "graph_spec_name": "graph_lower_bound",
            "batch_size": 1,
            "graph_spec_inputs": "",
            "graph_lower_bound": ""
        }
        inputs = [{"name": "input_ids", "dims": [1]},
                  {"name": "segment_ids", "dims": [1]},
                  {"name": "cu_seqlens", "dims": [2]},
                  {"name": "max_seqlen", "dims": [1]}]
        for io in inputs:
            graph_lower_bound["graph_spec_inputs"] += TRITON_GRAPH_SPEC_INPUT_FORMAT.format(**io)

        for total_seq_len in range(start, end, step):
            graph_spec = {}
            graph_spec["graph_spec_name"] = "graph_spec"
            graph_spec["batch_size"] = self.args.get("gpu_batch_size", 1)
            graph_spec["graph_spec_inputs"] = ""
            graph_spec["graph_lower_bound"] = TRITON_GRAPH_SPEC_FORMAT.format(**graph_lower_bound)
            inputs = [{"name": "input_ids", "dims": [total_seq_len]},
                      {"name": "segment_ids", "dims": [total_seq_len]},
                      {"name": "cu_seqlens", "dims": [self.args.get("gpu_batch_size", 1) + 1]},
                      {"name": "max_seqlen", "dims": [max_seq_len]}]
            for io in inputs:
                graph_spec["graph_spec_inputs"] += TRITON_GRAPH_SPEC_INPUT_FORMAT.format(**io)
            config["graph_spec"] += TRITON_GRAPH_SPEC_FORMAT.format(**graph_spec)
