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

import tensorrt as trt
import os
import sys
import platform
import onnx

sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common import logging, dict_get, BENCHMARKS
from code.common import get_system
from code.common.builder import BenchmarkBuilder

RN50Calibrator = import_module("code.resnet50.tensorrt.calibrator").RN50Calibrator
RN50GraphSurgeon = import_module("code.resnet50.tensorrt.rn50_graphsurgeon").RN50GraphSurgeon


class ResNet50(BenchmarkBuilder):
    """Resnet50 engine builder."""

    def __init__(self, args):
        workspace_size = dict_get(args, "workspace_size", default=(1 << 30))
        logging.info("Using workspace size: {:,}".format(workspace_size))

        super().__init__(args, name=BENCHMARKS.ResNet50, workspace_size=workspace_size)

        # Model path
        self.model_path = dict_get(args, "model_path", default="build/models/ResNet50/resnet50_v1.onnx")

        self.cache_file = None
        self.need_calibration = False

        if self.precision == "int8":
            # Get calibrator variables
            calib_batch_size = dict_get(self.args, "calib_batch_size", default=1)
            calib_max_batches = dict_get(self.args, "calib_max_batches", default=500)
            force_calibration = dict_get(self.args, "force_calibration", default=False)
            cache_file = dict_get(self.args, "cache_file", default="code/resnet50/tensorrt/calibrator.cache")
            preprocessed_data_dir = dict_get(self.args, "preprocessed_data_dir", default="build/preprocessed_data")
            calib_data_map = dict_get(self.args, "calib_data_map", default="data_maps/imagenet/cal_map.txt")
            calib_image_dir = os.path.join(preprocessed_data_dir, "imagenet/ResNet50/fp32")

            # Set up calibrator
            self.calibrator = RN50Calibrator(calib_batch_size=calib_batch_size, calib_max_batches=calib_max_batches,
                                             force_calibration=force_calibration, cache_file=cache_file,
                                             image_dir=calib_image_dir, calib_data_map=calib_data_map)
            self.builder_config.int8_calibrator = self.calibrator
            self.cache_file = cache_file
            self.need_calibration = force_calibration or not os.path.exists(cache_file)

    def initialize(self):
        """
        Parse input ONNX file to a TRT network. Apply layer optimizations and fusion plugins on network.
        """

        # Query system id for architecture
        self.system = get_system()
        self.gpu_arch = self.system.arch

        # Create network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Parse from onnx file.
        parser = trt.OnnxParser(self.network, self.logger)

        rn50_gs = RN50GraphSurgeon(self.model_path,
                                   self.gpu_arch, self.device_type,
                                   self.precision,
                                   self.cache_file, self.need_calibration)
        model = rn50_gs.process_onnx()
        success = parser.parse(onnx._serialize(model))
        if not success:
            raise RuntimeError("ResNet50 onnx model processing failed! Error: {:}".format(parser.get_error(0).desc()))
        # unmarking topk_layer_output_value, just leaving topk_layer_output_index
        assert self.network.num_outputs == 2, "Two outputs expected"
        assert self.network.get_output(0).name == "topk_layer_output_value",\
            "unexpected tensor: {}".format(self.network.get_output(0).name)
        assert self.network.get_output(1).name == "topk_layer_output_index",\
            "unexpected tensor: {}".format(self.network.get_output(1).name)
        logging.info("Unmarking output: {:}".format(self.network.get_output(0).name))
        self.network.unmark_output(self.network.get_output(0))

        # Set input dtype and format
        input_tensor = self.network.get_input(0)
        if self.input_dtype == "int8":
            input_tensor.dtype = trt.int8
            input_tensor.dynamic_range = (-128, 127)
        if self.input_format == "linear":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        elif self.input_format == "chw4":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

        self.initialized = True
