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


class RNNTHarness(BaseBenchmarkHarness):

    required_engine_files = [
        "encoder.plan",
        "decoder.plan",
        "fc1_a.plan",
        "fc1_b.plan",
        "igather.plan",
        "isel.plan",
        "joint_backend.plan",
        "joint_fc2_bias_ckpt.fp32.dat",
        "joint_fc2_weight_ckpt.fp32.dat",
        "joint_fc2_bias_ckpt.fp16.dat",
        "joint_fc2_weight_ckpt.fp16.dat",
    ]

    def __init__(self, args, name=""):
        super().__init__(args, name)
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + ["audio_batch_size",
                                                                    "audio_buffer_num_lines",
                                                                    "audio_fp16_input",
                                                                    "dali_batches_issue_ahead",
                                                                    "dali_pipeline_depth",
                                                                    "devices",
                                                                    "disable_encoder_plugin",
                                                                    "gpu_batch_size",
                                                                    "max_seq_length",
                                                                    "nobatch_sorting",
                                                                    "noenable_audio_processing",
                                                                    "nopipelined_execution",
                                                                    "nouse_copy_kernel",
                                                                    "num_warmups",
                                                                    "server_num_issue_query_threads",
                                                                    "use_graphs"]

    def _get_harness_executable(self):
        return "./build/bin/harness_rnnt"

    # Currently, RNNTHarness is using non-standard directory structure and filenames to store its engine files. Since
    # BaseBenchmarkHarness calls this function at the end of __init__, we have to put our custom directory structure setup
    # here instead of in __init__.
    def enumerate_engines(self):
        for fname in RNNTHarness.required_engine_files:
            self.check_file_exists(os.path.join(self.engine_dir, fname))

    def _build_custom_flags(self, flag_dict):
        # Rename gpu_batch_size to batch_size
        batch_size = dict_get(self.args, "gpu_batch_size", default=None)
        flag_dict["batch_size"] = batch_size
        flag_dict["gpu_batch_size"] = None

        # Rename use_graphs to cuda_graph
        use_graphs = dict_get(self.args, "use_graphs", default=False)
        flag_dict["cuda_graph"] = use_graphs
        flag_dict["use_graphs"] = None

        # Rename max_seq_length to hp_max_seq_length
        max_seq_length = dict_get(self.args, "max_seq_length", default=None)
        flag_dict["hp_max_seq_length"] = max_seq_length
        flag_dict["max_seq_length"] = None

        # Handle more harness_rnnt knobs
        no_pipelined = dict_get(self.args, "nopipelined_execution", default=False)
        flag_dict["pipelined_execution"] = not no_pipelined
        flag_dict["nopipelined_execution"] = None

        # Handle more harness_rnnt knobs : disable batch sorting by sequence length
        no_sorting = dict_get(self.args, "nobatch_sorting", default=False)
        flag_dict["batch_sorting"] = not no_sorting
        flag_dict["nobatch_sorting"] = None

        # Handle yet another harness_rnnt knob: turning off DALI preprocessing for debug
        no_dali = dict_get(self.args, "noenable_audio_processing", default=False)
        flag_dict["enable_audio_processing"] = not no_dali
        flag_dict["noenable_audio_processing"] = None

        # Handle yet another harness_rnnt knob: disable DALI's scatter gather kernel
        no_copy_kernel = dict_get(self.args, "nouse_copy_kernel", default=False)
        flag_dict["use_copy_kernel"] = not no_copy_kernel
        flag_dict["nouse_copy_kernel"] = None

        # Rename gpu_inference_streams to streams_per_gpu
        num_inference = dict_get(self.args, "gpu_inference_streams", default=None)
        flag_dict["streams_per_gpu"] = num_inference
        flag_dict["gpu_inference_streams"] = None

        audio_fp16_input = dict_get(self.args, "audio_fp16_input", default=True)
        flag_dict["audio_fp16_input"] = audio_fp16_input

        start_from_device = dict_get(self.args, "start_from_device", default=False)
        flag_dict["start_from_device"] = start_from_device

        audio_input_suffix = "fp16" if audio_fp16_input else "fp32"
        flag_dict["audio_serialized_pipeline_file"] = "build/bin/dali" + "/dali_pipeline_gpu_" + audio_input_suffix + ".pth"

        argstr = args_to_string(flag_dict) + " --scenario {:} --model {:}".format(self.scenario, self.name)

        # Handle engine dir
        argstr += " --engine_dir={:}".format(self.engine_dir)

        return argstr
