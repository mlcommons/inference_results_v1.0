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
import ctypes
import struct
import numpy as np
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common import logging, dict_get, BENCHMARKS
from code.common import get_system
from code.common.builder import BenchmarkBuilder
import pycuda.autoinit 
RN50Calibrator = import_module("code.resnet50.tensorrt.calibrator").RN50Calibrator
AUTOSINIAN_CNN_PLUGIN_LIBRARY = "code/resnet50/tensorrt/libautosiniancnnplugin_ampere.so" if pycuda.autoinit.device.compute_capability()[0] > 7 else "code/resnet50/tensorrt/libautosiniancnnplugin_turing.so"
if not os.path.isfile(AUTOSINIAN_CNN_PLUGIN_LIBRARY):
    raise IOError("{}\n".format(
        "Failed to load library ({}).".format(AUTOSINIAN_CNN_PLUGIN_LIBRARY)
    ))
ctypes.CDLL(AUTOSINIAN_CNN_PLUGIN_LIBRARY)

class ResNet50(BenchmarkBuilder):
    """Resnet50 engine builder."""

    def __init__(self, args):
        workspace_size = dict_get(args, "workspace_size", default=(1 << 30))
        logging.info("Use workspace_size: {:}".format(workspace_size))

        super().__init__(args, name=BENCHMARKS.ResNet50, workspace_size=workspace_size)

        # Model path
        self.model_path = dict_get(args, "model_path", default="code/resnet50/tensorrt/ofa_autosinian_is176.onnx")
        logging.info("Using AutoSinian optimized once-for-all network")
        
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

        with open(self.model_path, "rb") as f:
            model = f.read()
        success = parser.parse(model)
        if not success:
            raise RuntimeError("ofa_autusinian onnx model processing failed! Error: {:}".format(parser.get_error(0).desc()))
        # Set input dtype and format
        input_tensor = self.network.get_input(0)
        if self.input_dtype == "int8":
            input_tensor.dtype = trt.int8
            scale = struct.unpack('!f', bytes.fromhex('3caa5293'))[0]
            input_tensor.dynamic_range = (-scale*127.0, scale*127.0)
        if self.input_format == "linear":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        elif self.input_format == "chw4":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

        # Get the layers we care about.
        nb_layers = self.network.num_layers

        while self.network.num_outputs > 0:
            logging.info("Unmarking output: {:}".format(self.network.get_output(0).name))
            self.network.unmark_output(self.network.get_output(0))
        #add top-k
        last_fc_layer = self.network.get_layer(nb_layers - 1)
        topk_layer = self.network.add_topk(last_fc_layer.get_output(0), trt.TopKOperation.MAX, 1, 2)
        topk_layer.name = "topk_layer"
        topk_layer.get_output(0).name = "topk_layer_output_value"
        topk_layer.get_output(1).name = "topk_layer_output_index"
        self.network.mark_output(topk_layer.get_output(1))

        if self.network.num_outputs != 1:
            logging.warning("num outputs should be 1 after unmarking! Has {:}".format(self.network.num_outputs))
            raise Exception
            
        if self.precision == "int8" and self.batch_size > 1 and (not self.need_calibration):
            self.autosinian_optimize()

        self.initialized = True
    
    def autosinian_optimize(self):
        logging.info("Applying AutoSinian Optimization...")
        optimize_points = [(10,15), (21,26), (27,32), (38,43), (44,49), (55,60), (61,66), (67,72), (78,83), (84,89), (90,95), (0,4), (5,9), (16,20), (33,37), (50,54), (73,77), (96,100)]
        optimizer = AutoSinian_Optimizer(self.cache_file)
        for point in optimize_points:
            optimizer.optimize(self.network, point)
      
class AutoSinian_Optimizer:
    '''AutoSinian optimizer, optimize the hardware implementation of the layers.'''
    def __init__(self, cache_file = None):
        self.plugin_registery = trt.get_plugin_registry()
        foundPlugin = False
        for plugin_creator in self.plugin_registery.plugin_creator_list:
            if plugin_creator.name == self.name:
                self.creator = self.plugin_registery.get_plugin_creator(self.name,'1','')  
                foundPlugin = True if self.creator else False
                break
        assert(foundPlugin), "fail to found %s!" % self.name
        self.scale_map = {}
        with open(cache_file, "r") as f:
            for line in f:
                pair = line.rstrip().split(':')
                if len(pair) == 2:
                    self.scale_map[pair[0]] = struct.unpack('!f', bytes.fromhex(pair[1]))[0]
        self.count = 0
    
    @property
    def name(self):
        return "AutoSinianCNN_TRT"
    
    def optimize(self, network, point):
        fields = trt.PluginFieldCollection()
        saved = [] #values must be alive when creating the plugin.
        inputs = [network.get_layer(point[0]).get_input(0)]
        append_fields(network, point[0], fields, saved, self.scale_map)
        append_fields(network, point[0]+2, fields, saved, self.scale_map)
        append_fields(network, point[0]+4, fields, saved, self.scale_map)

        plugin=self.creator.create_plugin(self.name, fields)
        if plugin is None:
            raise Exception("Plugin creation failed")
        
        plugin_layer = network.add_plugin_v2(inputs, plugin)
        plugin_layer.name = self.name + "_%d" % self.count
        self.count += 1
        origin_output = network.get_layer(point[1]).get_output(0)
        plugin_output = plugin_layer.get_output(0)
        assert(origin_output.name in self.scale_map), "%s not found!" % origin_output.name
        dynamic_range=self.scale_map[origin_output.name]*127.0
        plugin_output.set_dynamic_range(-dynamic_range, dynamic_range)
        for j in range(network.num_layers):
            layer = network.get_layer(j)
            if layer.name==plugin_layer.name :
                continue
            for k in range(layer.num_inputs):
                if layer.get_input(k) == origin_output:
                    layer.set_input(k, plugin_output)

def append_fields(network, index, fields, saved, scale_map):
    layer = network.get_layer(index)
    assert(isinstance(layer, trt.ILayer) and (layer.type == trt.LayerType.CONVOLUTION)), "must be a conv layer"
    layer.__class__ = trt.IConvolutionLayer
    output_layer = layer
    
    npa1 = np.array([layer.kernel_size.h], dtype=np.int32)
    saved.append(npa1)
    
    npa2 = np.array([layer.num_output_maps], dtype=np.int32)
    saved.append(npa2)
    
    npa3 = np.array([layer.num_groups], dtype=np.int32)
    saved.append(npa3)
    
    npa4 = np.array([layer.stride.h], dtype=np.int32)
    saved.append(npa4)
    
    npa5 = np.array([layer.pre_padding[0]], dtype=np.int32)
    saved.append(npa5)
    
    npa6 = np.array([layer.post_padding[0]], dtype=np.int32)
    saved.append(npa6)
    
    npa7 = np.array([layer.get_input(0).shape[1]], dtype=np.int32)
    saved.append(npa7)

    next_layer = network.get_layer(index+1)
    
    elemwise_add = -1
    if (next_layer.type == trt.LayerType.ACTIVATION):
        next_layer.__class__ = trt.IActivationLayer
        output_layer = next_layer
    elif (next_layer.type == trt.LayerType.ELEMENTWISE):
        next_layer.__class__ = trt.IElementWiseLayer
        elemwise_add = 0
        output_layer = next_layer
    npa8 = np.array([elemwise_add], dtype=np.int32)
    saved.append(npa8)
    
    npa9 = np.array([0], dtype=np.int32)
    saved.append(npa9)
    
    npa10 = 0
    if index == 0:
        npa10 = struct.unpack('!f', bytes.fromhex('3caa5293'))[0]
    npa10 = np.array([npa10], dtype=np.float32)
    saved.append(npa10)
    
    npa11 = np.array([6], dtype=np.float32)
    saved.append(npa11)
    
    name = output_layer.get_output(0).name
    assert(name in scale_map), "Missing scale for %s"%name
    scale = scale_map[name]
    npa12 = np.array([scale], dtype=np.float32)
    saved.append(npa12)
    
    fields.append(trt.PluginField("asn_cnn_plg_field", memoryview(npa1), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", memoryview(npa2), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", memoryview(npa3), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", memoryview(npa4), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", memoryview(npa5), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", memoryview(npa6), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", memoryview(npa7), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", memoryview(npa8), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", memoryview(npa9), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", memoryview(npa10), trt.PluginFieldType.FLOAT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", memoryview(npa11), trt.PluginFieldType.FLOAT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", memoryview(npa12), trt.PluginFieldType.FLOAT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", layer.kernel.data, trt.PluginFieldType.FLOAT32))
    fields.append(trt.PluginField("asn_cnn_plg_field", layer.bias.data, trt.PluginFieldType.FLOAT32))
    