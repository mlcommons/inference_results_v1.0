#!/usr/bin/env python3
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

from importlib import import_module
from code.common import BENCHMARKS
from code.common.arguments import apply_overrides


def get_benchmark(conf):
    """Return module of benchmark initialized with config."""

    benchmark_name = conf["benchmark"]

    # Do not use a map. We want to import benchmarks as we need them, because some take
    # time to load due to plugins.
    if benchmark_name == BENCHMARKS.ResNet50:
        ResNet50 = import_module("code.resnet50.tensorrt.ResNet50").ResNet50
        return ResNet50(conf)
    elif benchmark_name == BENCHMARKS.SSDResNet34:
        SSDResNet34 = import_module("code.ssd-resnet34.tensorrt.SSDResNet34").SSDResNet34
        return SSDResNet34(conf)
    elif benchmark_name == BENCHMARKS.SSDMobileNet:
        SSDMobileNet = import_module("code.ssd-mobilenet.tensorrt.SSDMobileNet").SSDMobileNet
        return SSDMobileNet(conf)
    elif benchmark_name == BENCHMARKS.BERT:
        # TODO now only BERT uses gpu_inference_streams to generate engines
        conf = apply_overrides(conf, ['gpu_inference_streams'])
        BERTBuilder = import_module("code.bert.tensorrt.bert_var_seqlen").BERTBuilder
        return BERTBuilder(conf)
    elif benchmark_name == BENCHMARKS.RNNT:
        RNNTBuilder = import_module("code.rnnt.tensorrt.rnn-t_builder").RNNTBuilder
        return RNNTBuilder(conf)
    elif benchmark_name == BENCHMARKS.DLRM:
        DLRMBuilder = import_module("code.dlrm.tensorrt.dlrm").DLRMBuilder
        return DLRMBuilder(conf)
    elif benchmark_name == BENCHMARKS.UNET:
        UNETBuilder = import_module("code.3d-unet.tensorrt.3d-unet").UnetBuilder
        return UNETBuilder(conf)
    else:
        raise ValueError("Unknown benchmark: {:}".format(benchmark_name))


def get_harness(config, profile):
    """Refactors harness generation for use by functions other than handle_run_harness."""
    benchmark_name = config['benchmark']
    # Quick path for CPU
    if config.get("use_cpu"):
        TritonHarnessCPU = import_module("code.common.server_harness_cpu").TritonHarnessCPU
        harness = TritonHarnessCPU(config, name=benchmark_name)
        config["inference_server"] = "triton"
        return harness, config

    if config.get("use_triton"):
        TritonHarness = import_module("code.common.server_harness").TritonHarness
        harness = TritonHarness(config, name=benchmark_name)
        config["inference_server"] = "triton"
    elif benchmark_name == BENCHMARKS.BERT:
        BertHarness = import_module("code.bert.tensorrt.harness").BertHarness
        harness = BertHarness(config, name=benchmark_name)
        config["inference_server"] = "custom"
    elif benchmark_name == BENCHMARKS.DLRM:
        DLRMHarness = import_module("code.dlrm.tensorrt.harness").DLRMHarness
        harness = DLRMHarness(config, name=benchmark_name)
        config["inference_server"] = "custom"
    elif benchmark_name == BENCHMARKS.RNNT:
        RNNTHarness = import_module("code.rnnt.tensorrt.harness").RNNTHarness
        harness = RNNTHarness(config, name=benchmark_name)
        config["inference_server"] = "custom"
    else:
        LWISHarness = import_module("code.common.lwis_harness").LWISHarness
        harness = LWISHarness(config, name=benchmark_name)

    # Attempt to run profiler. Note that this is only available internally.
    if profile is not None:
        try:
            ProfilerHarness = import_module("code.internal.profiler").ProfilerHarness
            harness = ProfilerHarness(harness, profile)
        except BaseException:
            logging.info("Could not load profiler: Are you an internal user?")

    return harness, config
