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
    if benchmark_name == BENCHMARKS.BERT:
        # TODO now only BERT uses gpu_inference_streams to generate engines
        conf = apply_overrides(conf, ['gpu_inference_streams'])
        BERTBuilder = import_module("code.bert.tensorrt_sparse.bert_var_seqlen").BERTBuilder
        return BERTBuilder(conf)
    elif benchmark_name == BENCHMARKS.ResNet50:
        ResNet50Builder = import_module("code.resnet50.int4.harness").ResNet50Builder
        return ResNet50Builder(conf)
    else:
        raise ValueError("Unknown benchmark: {:}".format(benchmark_name))


def get_harness(config, profile):
    """Refactors harness generation for use by functions other than handle_run_harness."""
    benchmark_name = config['benchmark']
    if benchmark_name == BENCHMARKS.BERT:
        BertHarness = import_module("code.bert.tensorrt_sparse.harness").BertHarness
        harness = BertHarness(config, name=benchmark_name)
        config["inference_server"] = "custom"
    elif benchmark_name == BENCHMARKS.ResNet50:
        ResNet50Harness = import_module("code.resnet50.int4.harness").ResNet50Harness
        harness = ResNet50Harness(config)
    else:
        raise RuntimeError("Could not find supported harness!")

    return harness, config
