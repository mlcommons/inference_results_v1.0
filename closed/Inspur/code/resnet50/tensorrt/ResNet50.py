#!/usr/bin/env python3
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import os, sys
import platform

sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common import logging, dict_get, BENCHMARKS
from code.common.builder import BenchmarkBuilder
RN50Calibrator = import_module("code.resnet50.tensorrt.calibrator").RN50Calibrator
parse_calibration = import_module("code.resnet50.tensorrt.res2_fusions").parse_calibration
fuse_br1_br2c_onnx = import_module("code.resnet50.tensorrt.res2_fusions").fuse_br1_br2c_onnx
fuse_br2b_br2c_onnx = import_module("code.resnet50.tensorrt.res2_fusions").fuse_br2b_br2c_onnx
fuse_res2_onnx = import_module("code.resnet50.tensorrt.res2_fusions").fuse_res2_onnx
fuse_serial_3_conv2dc_onnx = import_module("code.resnet50.tensorrt.res2_fusions").fuse_serial_3_conv2dc_onnx

class ResNet50(BenchmarkBuilder):

    def __init__(self, args):
        workspace_size = dict_get(args, "workspace_size", default=( 1 << 30 ))
        logging.info("Use workspace_size: {:}".format(workspace_size))
        
        super().__init__(args, name=BENCHMARKS.ResNet50, workspace_size=workspace_size)

        # Model path
        self.model_path = dict_get(args, "model_path", default="build/models/ResNet50/resnet50_v1.onnx")

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
        # Create network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Parse from onnx file.
        parser = trt.OnnxParser(self.network, self.logger)
        with open(self.model_path, "rb") as f:
            model = f.read()
        success = parser.parse(model)
        if not success:
            raise RuntimeError("ResNet50 onnx model parsing failed! Error: {:}".format(parser.get_error(0).desc()))

        nb_layers = self.network.num_layers
        for i in range(nb_layers):
            layer = self.network.get_layer(i)
            # ':' in tensor names will screw up calibration cache parsing (which uses ':' as a delimiter)
            for j in range(layer.num_inputs):
                tensor = layer.get_input(j)
                tensor.name = tensor.name.replace(":", "_")
            for j in range(layer.num_outputs):
                tensor = layer.get_output(j)
                tensor.name = tensor.name.replace(":", "_")

        # Post-process the TRT network
        self.postprocess(useConvForFC = (self.precision == "int8"))

        # Query system id for architecture
        self.gpu_arch = None


        self.registry = trt.get_plugin_registry()
        parse_calibration(self.network, self.cache_file)
        fuse_res2_onnx(self.registry, self.network)
        fuse_serial_3_conv2dc_onnx(self.registry, self.network)

        self.fix_layer_names()

        self.initialized = True

    def postprocess(self, useConvForFC=False):
        # Set input dtype and format
        input_tensor = self.network.get_input(0)
        if self.input_dtype == "int8":
            input_tensor.dtype = trt.int8
            input_tensor.dynamic_range = (-128, 127)
        if self.input_format == "linear":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        elif self.input_format == "chw4":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

        # Get the layers we care about.
        nb_layers = self.network.num_layers
        logging.debug(nb_layers)
        for i in range(nb_layers):
            layer = self.network.get_layer(i)
            logging.info("({:}) Layer '{:}' -> Type: {:} ON {:}".format(i, layer.name, layer.type,
                self.builder_config.get_device_type(layer)))

            # Detect the FC layer.
            # if "Fully Connected" in layer.name:
            if "MatMul" in layer.name:
                fc_layer = layer
                assert fc_layer.type == trt.LayerType.FULLY_CONNECTED
                fc_layer.__class__ = trt.IFullyConnectedLayer
                fc_kernel = fc_layer.kernel.reshape(1001, 2048)[1:,:]
                fc_bias = fc_layer.bias[1:]

                # (i-13)th layer should be reduction.
                reduce_layer = self.network.get_layer(i-13)
                assert reduce_layer.type == trt.LayerType.REDUCE
                reduce_layer.__class__ = trt.IReduceLayer

                # (i-14)th layer should be the last ReLU
                last_conv_layer = self.network.get_layer(i-14)
                assert last_conv_layer.type == trt.LayerType.ACTIVATION
                last_conv_layer.__class__ = trt.IActivationLayer

        # Unmark the old output since we are going to add new layers for the final part of the network.
        while self.network.num_outputs > 0:
            logging.info("Unmarking output: {:}".format(self.network.get_output(0).name))
            self.network.unmark_output(self.network.get_output(0))

        # Replace the reduce layer with pooling layer
        pool_layer_new = self.network.add_pooling(last_conv_layer.get_output(0), trt.PoolingType.AVERAGE, (7, 7))
        pool_layer_new.name = "squeeze_replaced"
        pool_layer_new.get_output(0).name = "squeeze_replaced_output"

        # Add fc layer
        fc_kernel = fc_kernel.flatten()
        if useConvForFC:
            fc_layer_new = self.network.add_convolution(pool_layer_new.get_output(0), fc_bias.size, (1, 1), fc_kernel, fc_bias)
        else:
            fc_layer_new = self.network.add_fully_connected(pool_layer_new.get_output(0), fc_bias.size, fc_kernel, fc_bias)
        fc_layer_new.name = "fc_replaced"
        fc_layer_new.get_output(0).name = "fc_replaced_output"

        # Add topK layer.
        topk_layer = self.network.add_topk(fc_layer_new.get_output(0), trt.TopKOperation.MAX, 1, 2)
        topk_layer.name = "topk_layer"
        topk_layer.get_output(0).name = "topk_layer_output_value"
        topk_layer.get_output(1).name = "topk_layer_output_index"

        # Mark the new output.
        self.network.mark_output(topk_layer.get_output(1))

        if self.network.num_outputs != 1:
            logging.warning("num outputs should be 1 after unmarking! Has {:}".format(self.network.num_outputs))
            raise Exception

    def fix_layer_names(self):
        layer_name_map = {
            "resnet_model/conv2d/Conv2D": "conv1",
            "resnet_model/batch_normalization/FusedBatchNorm": "scale_conv1",
            "resnet_model/Relu": "conv1_relu",
            "resnet_model/max_pooling2d/MaxPool": "pool1",

            "Conv__128": "res2a_branch2a",
            "resnet_model/Relu_1": "res2a_branch2a_relu",
            "Conv__129": "res2a_branch2b",
            "resnet_model/Relu_2": "res2a_branch2b_relu",
            "Conv__130": "res2a_branch2c",
            "Conv__123": "res2a_branch1",
            "resnet_model/add": "res2a",
            "resnet_model/Relu_3": "res2a_relu",

            "Conv__131": "res2b_branch2a",
            "resnet_model/Relu_4": "res2b_branch2a_relu",
            "Conv__132": "res2b_branch2b",
            "resnet_model/Relu_5": "res2b_branch2b_relu",
            "Conv__133": "res2b_branch2c",
            "resnet_model/add_1": "res2b",
            "resnet_model/Relu_6": "res2b_relu",

            "Conv__138": "res2c_branch2a",
            "resnet_model/Relu_7": "res2c_branch2a_relu",
            "Conv__139": "res2c_branch2b",
            "resnet_model/Relu_8": "res2c_branch2b_relu",
            "Conv__140": "res2c_branch2c",
            "resnet_model/add_2": "res2c",
            "resnet_model/Relu_9": "res2c_relu",

            "Conv__145": "res3a_branch2a",
            "resnet_model/Relu_10": "res3a_branch2a_relu",
            "Conv__146": "res3a_branch2b",
            "resnet_model/Relu_11": "res3a_branch2b_relu",
            "Conv__147": "res3a_branch2c",
            "Conv__152": "res3a_branch1",
            "resnet_model/add_3": "res3a",
            "resnet_model/Relu_12": "res3a_relu",

            "Conv__153": "res3b_branch2a",
            "resnet_model/Relu_13": "res3b_branch2a_relu",
            "Conv__154": "res3b_branch2b",
            "resnet_model/Relu_14": "res3b_branch2b_relu",
            "Conv__155": "res3b_branch2c",
            "resnet_model/add_4": "res3b",
            "resnet_model/Relu_15": "res3b_relu",

            "Conv__160": "res3c_branch2a",
            "resnet_model/Relu_16": "res3c_branch2a_relu",
            "Conv__161": "res3c_branch2b",
            "resnet_model/Relu_17": "res3c_branch2b_relu",
            "Conv__162": "res3c_branch2c",
            "resnet_model/add_5": "res3c",
            "resnet_model/Relu_18": "res3c_relu",

            "Conv__167": "res3d_branch2a",
            "resnet_model/Relu_19": "res3d_branch2a_relu",
            "Conv__168": "res3d_branch2b",
            "resnet_model/Relu_20": "res3d_branch2b_relu",
            "Conv__169": "res3d_branch2c",
            "resnet_model/add_6": "res3d",
            "resnet_model/Relu_21": "res3d_relu",

            "Conv__174": "res4a_branch2a",
            "resnet_model/Relu_22": "res4a_branch2a_relu",
            "Conv__175": "res4a_branch2b",
            "resnet_model/Relu_23": "res4a_branch2b_relu",
            "Conv__176": "res4a_branch2c",
            "Conv__181": "res4a_branch1",
            "resnet_model/add_7": "res4a",
            "resnet_model/Relu_24": "res4a_relu",

            "Conv__182": "res4b_branch2a",
            "resnet_model/Relu_25": "res4b_branch2a_relu",
            "Conv__183": "res4b_branch2b",
            "resnet_model/Relu_26": "res4b_branch2b_relu",
            "Conv__184": "res4b_branch2c",
            "resnet_model/add_8": "res4b",
            "resnet_model/Relu_27": "res4b_relu",

            "Conv__189": "res4c_branch2a",
            "resnet_model/Relu_28": "res4c_branch2a_relu",
            "Conv__190": "res4c_branch2b",
            "resnet_model/Relu_29": "res4c_branch2b_relu",
            "Conv__191": "res4c_branch2c",
            "resnet_model/add_9": "res4c",
            "resnet_model/Relu_30": "res4c_relu",

            "Conv__196": "res4d_branch2a",
            "resnet_model/Relu_31": "res4d_branch2a_relu",
            "Conv__197": "res4d_branch2b",
            "resnet_model/Relu_32": "res4d_branch2b_relu",
            "Conv__198": "res4d_branch2c",
            "resnet_model/add_10": "res4d",
            "resnet_model/Relu_33": "res4d_relu",

            "Conv__203": "res4e_branch2a",
            "resnet_model/Relu_34": "res4e_branch2a_relu",
            "Conv__204": "res4e_branch2b",
            "resnet_model/Relu_35": "res4e_branch2b_relu",
            "Conv__205": "res4e_branch2c",
            "resnet_model/add_11": "res4e",
            "resnet_model/Relu_36": "res4e_relu",

            "Conv__210": "res4f_branch2a",
            "resnet_model/Relu_37": "res4f_branch2a_relu",
            "Conv__211": "res4f_branch2b",
            "resnet_model/Relu_38": "res4f_branch2b_relu",
            "Conv__212": "res4f_branch2c",
            "resnet_model/add_12": "res4f",
            "resnet_model/Relu_39": "res4f_relu",

            "Conv__217": "res5a_branch1",
            "Conv__222": "res5a_branch2a",
            "resnet_model/Relu_40": "res5a_branch2a_relu",
            "Conv__223": "res5a_branch2b",
            "resnet_model/Relu_41": "res5a_branch2b_relu",
            "Conv__224": "res5a_branch2c",
            "resnet_model/add_13": "res5a",
            "resnet_model/Relu_42": "res5a_relu",

            "Conv__225": "res5b_branch2a",
            "resnet_model/Relu_43": "res5b_branch2a_relu",
            "Conv__226": "res5b_branch2b",
            "resnet_model/Relu_44": "res5b_branch2b_relu",
            "Conv__227": "res5b_branch2c",
            "resnet_model/add_14": "res5b",
            "resnet_model/Relu_45": "res5b_relu",

            "Conv__232": "res5c_branch2a",
            "resnet_model/Relu_46": "res5c_branch2a_relu",
            "Conv__233": "res5c_branch2b",
            "resnet_model/Relu_47": "res5c_branch2b_relu",
            "Conv__234": "res5c_branch2c",
            "resnet_model/add_15": "res5c",
            "resnet_model/Relu_48": "res5c_relu",

            "resnet_model/Mean": "pool5",
            # "reshape__269": "",
            # "resnet_model/Squeeze": "",
            # "(Unnamed Layer* 123) [Shape]": "",
            # "(Unnamed Layer* 124) [Gather]": "",
            # "(Unnamed Layer* 125) [Shuffle]": "",
            # "resnet_model/dense/MatMul": "",
            # "(Unnamed Layer* 127) [Shape]": "",
            # "(Unnamed Layer* 128) [Constant]": "",
            # "(Unnamed Layer* 129) [Concatenation]": "",
            # "(Unnamed Layer* 130) [Constant]": "",
            # "(Unnamed Layer* 131) [Gather]": "",
            # "(Unnamed Layer* 132) [Shuffle]": "",
            # TODO: ONNX Parser change
            # "(Unnamed Layer* 133) [Fully Connected]": "fc1000",
            "resnet_model/dense/MatMul": "fc1000",
            # "(Unnamed Layer* 134) [Constant]": "",
            # "(Unnamed Layer* 135) [Shape]": "",
            # "(Unnamed Layer* 136) [Gather]": "",
            # "(Unnamed Layer* 137) [Shuffle]": "",
            # "resnet_model/dense/BiasAdd": "",
            # "(Unnamed Layer* 139) [Shuffle]": "",
            # "(Unnamed Layer* 140) [ElementWise]": "",
            # "resnet_model/final_dense": "",
            # "softmax_tensor": "",
            # "(Unnamed Layer* 143) [Shape]": "",
            # "(Unnamed Layer* 144) [Gather]": "",
            # "(Unnamed Layer* 145) [Constant]": "",
            # "(Unnamed Layer* 146) [Concatenation]": "",
            # "(Unnamed Layer* 147) [Shuffle]": "",
            # TODO: ONNX Parser change
            # "(Unnamed Layer* 148) [Softmax]": "prob",
            "softmax_tensor": "prob",
            # "(Unnamed Layer* 149) [Shuffle]": "",
            # "(Unnamed Layer* 150) [Shape]": "",
            # "graph_outputs_Identity__6": "",
            "ArgMax": "topk",
            # "(Unnamed Layer* 153) [Constant]": "",
            # "(Unnamed Layer* 154) [Shape]": "",
            # "(Unnamed Layer* 155) [Gather]": "",
            # "(Unnamed Layer* 156) [Shuffle]": "",
            # "graph_outputs_Identity__4": "",
        }

        # rename layers to something more sensible
        nb_layers = self.network.num_layers
        for i in range(nb_layers):
            layer = self.network.get_layer(i)

            if layer.name in layer_name_map:
                new_layer_name = layer_name_map[layer.name]
                logging.debug ("Renaming Layer: {:} -> {:}".format(layer.name, new_layer_name))
                layer.name = new_layer_name
