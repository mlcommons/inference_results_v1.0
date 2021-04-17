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
import numpy as np
import torch

import argparse
import ctypes
import os
import sys
sys.path.insert(0, os.getcwd())

# The plugin .so file has to be loaded at global scope and before `import torch` to avoid cuda version mismatch.
RNNT_OPT_PLUGIN_LIBRARY = "build/plugins/RNNTOptPlugin/librnntoptplugin.so"
if not os.path.isfile(RNNT_OPT_PLUGIN_LIBRARY):
    raise IOError("{}\n{}\n".format(
        "Failed to load library ({}).".format(RNNT_OPT_PLUGIN_LIBRARY),
        "Please build the RNN-T Opt plugin."
    ))
ctypes.CDLL(RNNT_OPT_PLUGIN_LIBRARY)

from code.common import logging, dict_get, run_command, BENCHMARKS
from code.common.builder import BenchmarkBuilder, MultiBuilder
from code.rnnt.dali.pipeline import DALIInferencePipeline

import code.common.arguments as common_args

from importlib import import_module
RNNTCalibrator = import_module("code.rnnt.tensorrt.calibrator").RNNTCalibrator

# Support methods
##


def set_tensor_dtype(tensor, t_dtype, t_format):
    # handle datatype
    if t_dtype == "int8":
        tensor.dtype = trt.int8
        tensor.dynamic_range = (-128, 127)
    elif t_dtype == "int32":
        tensor.dtype = trt.int32
    elif t_dtype == "fp16":
        tensor.dtype = trt.float16
    elif t_dtype == "fp32":
        tensor.dtype = trt.float32
    else:
        assert(False)

    # handle format
    if t_format == "linear":
        tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    elif t_format == "chw4":
        tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)
    elif t_format == "hwc8":
        tensor.allowed_formats = 1 << int(trt.TensorFormat.HWC8)

# Common parameters
##


class RNNHyperParam:
    # alphabet
    labels_size = 29   # alphabet

    # encoder
    encoder_input_size = 240
    encoder_hidden_size = 1024
    enc_pre_rnn_layers = 2
    enc_post_rnn_layers = 3

    # encoder
    decoder_input_size = 320
    decoder_hidden_size = 320
    joint_hidden_size = 512
    dec_rnn_layers = 2

# Parent class
##


class RNNTBaseBuilder(BenchmarkBuilder):

    model_path = "build/models/rnn-t/DistributedDataParallel_1576581068.9962234-epoch-100.pt"
    state_dict = None

    @classmethod
    def _load_model(cls):
        if not cls.state_dict:
            logging.info("Loading RNN-T PyTorch model")
            checkpoint = torch.load(cls.model_path, map_location="cpu")
            cls.state_dict = checkpoint["state_dict"]

    def __init__(self, args):
        workspace_size = dict_get(args, "workspace_size", default=(4 << 30))
        logging.info("Using workspace size: {:,}".format(workspace_size))

        super().__init__(args, name=BENCHMARKS.RNNT, workspace_size=workspace_size)

        self.num_profiles = 1

        self.max_seq_length = dict_get(args, "max_seq_length", default=128)
        self.opt = dict_get(args, "opt", default="greedy")

    def initialize(self):
        RNNTBaseBuilder._load_model()
        self.initialized = True

# Encoder class
##


class EncoderBuilder(RNNTBaseBuilder):
    def __init__(self, args):
        super().__init__(args)

        # Encoder has a special enc_batch_size argument that can override batch size
        self.batch_size = dict_get(args, "enc_batch_size", default=self.batch_size)

        self.name = "encoder.plan"
        self.expose_state = not dict_get(args, "seq_splitting_off", default=False)
        self.unroll = dict_get(args, "calibrate_encoder", default=False)

        if not dict_get(self.args, "disable_encoder_plugin", default=False):
            self.cache_file = dict_get(self.args, "cache_file", default="code/rnnt/tensorrt/calibrator.cache")

        if self.unroll:
            calib_batch_size = dict_get(self.args, "calib_batch_size", default=10)
            calib_max_batches = dict_get(self.args, "calib_max_batches", default=500)
            force_calibration = dict_get(self.args, "force_calibration", default=False)
            cache_file = dict_get(self.args, "cache_file", default="code/rnnt/tensorrt/calibrator.cache")

            calib_data_map = dict_get(self.args, "calib_data_map", default="build/preprocessed_data/rnnt_train_clean_512_fp32/val_map_512.txt")
            preprocessed_data_dir = dict_get(self.args, "preprocessed_data_dir", default="build/preprocessed_data")
            calib_data_dir = os.path.join(preprocessed_data_dir, "rnnt_train_clean_512_fp32/fp32")
            calib_data_dir = dict_get(self.args, "calib_data_dir", default=calib_data_dir)

            # We can't run with expose_state because we don't have stimulus/calibrationData for hidden state.
            if self.expose_state:
                raise NotImplementedError("Can't use --calibrate_encoder without --seq_splitting_off")
            if self.input_dtype != 'fp32':
                print(f"Warning: Not using --input_type=fp32 may result in accuracy degredation and poor calibration performance given fp32 data")
            # If FP32/FP16 was set, unflip that flag in builder_config and set the Int8 flag
            if self.precision != "int8":
                flag_to_flip = trt.BuilderFlag.FP16 if self.precision == "fp16" else trt.BuilderFlag.FP32
                self.builder_config.flags = (self.builder_config.flags) & ~(1 << int(flag_to_flip))
                self.builder_config.flags = (self.builder_config.flags) | (1 << int(trt.BuilderFlag.INT8))
            if calib_batch_size < self.batch_size:
                raise RuntimeError(f"Can't run with calibration batch size less than than network batch size: {calib_batch_size} vs. {self.batch_size}!\nThis is tracked by MLPINF-437")
            self.calibrator = RNNTCalibrator(calib_batch_size, calib_max_batches,
                                             force_calibration, cache_file,
                                             calib_data_map, calib_data_dir, self.input_dtype)
            self.builder_config.int8_calibrator = self.calibrator
            self.cache_file = cache_file
            self.need_calibration = force_calibration or not os.path.exists(cache_file)

    def parse_calibration(self):
        """Parse calibration file to get dynamic range of all network tensors.
        Returns the tensor:range dict.
        """

        if not os.path.exists(self.cache_file):
            return

        with open(self.cache_file, "rb") as f:
            lines = f.read().decode('ascii').splitlines()

        calibration_dict = {}
        for line in lines:
            split = line.split(':')
            if len(split) != 2:
                continue
            tensor = split[0]
            calibration_dict[tensor] = np.uint32(int(split[1], 16)).view(np.dtype('float32')).item()

        return calibration_dict

    def add_unrolled_rnns(self, num_layers, max_seq_length, input_tensor, length_tensor, input_size, hidden_size, hidden_state_tensor, cell_state_tensor, name):
        past_layer = None
        for i in range(num_layers):
            if past_layer is None:
                # For the first layer, set-up inputs
                rnn_layer = self.network.add_rnn_v2(input_tensor, 1, hidden_size, max_seq_length, trt.RNNOperation.LSTM)
                rnn_layer.seq_lengths = length_tensor
                # Note that we don't hook-up argument-state-tensors because
                # calib_unroll can only be called with --seq_splitting_off
            else:
                # Hook-up the past layer
                rnn_layer = self.network.add_rnn_v2(past_layer.get_output(0), 1, hidden_size, max_seq_length, trt.RNNOperation.LSTM)
                rnn_layer.seq_lengths = length_tensor
            rnn_layer.get_output(0).name = f"{name}{i}_output"
            rnn_layer.get_output(1).name = f"{name}{i}_hidden"
            rnn_layer.get_output(2).name = f"{name}{i}_cell"
            # Set the name as expected for weight finding
            rnn_layer.name = name
            self._init_weights_per_layer(rnn_layer, i, True)
            # Now rename the layer for readability
            rnn_layer.name = f"{name}{i}"
            # Move on to the next layer
            past_layer = rnn_layer
        return rnn_layer

    def _init_weights_per_layer(self, layer, idx, is_unrolled=False):
        name = layer.name
        # initialization of the gate weights
        weight_ih = RNNTBaseBuilder.state_dict[name + '.weight_ih_l' + str(idx)]
        weight_ih = weight_ih.chunk(4, 0)

        weight_hh = RNNTBaseBuilder.state_dict[name + '.weight_hh_l' + str(idx)]
        weight_hh = weight_hh.chunk(4, 0)

        bias_ih = RNNTBaseBuilder.state_dict[name + '.bias_ih_l' + str(idx)]
        bias_ih = bias_ih.chunk(4, 0)

        bias_hh = RNNTBaseBuilder.state_dict[name + '.bias_hh_l' + str(idx)]
        bias_hh = bias_hh.chunk(4, 0)

        for gate_type in [trt.RNNGateType.INPUT, trt.RNNGateType.CELL, trt.RNNGateType.FORGET, trt.RNNGateType.OUTPUT]:
            for is_w in [True, False]:
                if is_w:
                    if (gate_type == trt.RNNGateType.INPUT):
                        weights = trt.Weights(weight_ih[0].numpy().astype(np.float32))
                        bias = trt.Weights(bias_ih[0].numpy().astype(np.float32))
                    elif (gate_type == trt.RNNGateType.FORGET):
                        weights = trt.Weights(weight_ih[1].numpy().astype(np.float32))
                        bias = trt.Weights(bias_ih[1].numpy().astype(np.float32))
                    elif (gate_type == trt.RNNGateType.CELL):
                        weights = trt.Weights(weight_ih[2].numpy().astype(np.float32))
                        bias = trt.Weights(bias_ih[2].numpy().astype(np.float32))
                    elif (gate_type == trt.RNNGateType.OUTPUT):
                        weights = trt.Weights(weight_ih[3].numpy().astype(np.float32))
                        bias = trt.Weights(bias_ih[3].numpy().astype(np.float32))
                else:
                    if (gate_type == trt.RNNGateType.INPUT):
                        weights = trt.Weights(weight_hh[0].numpy().astype(np.float32))
                        bias = trt.Weights(bias_hh[0].numpy().astype(np.float32))
                    elif (gate_type == trt.RNNGateType.FORGET):
                        weights = trt.Weights(weight_hh[1].numpy().astype(np.float32))
                        bias = trt.Weights(bias_hh[1].numpy().astype(np.float32))
                    elif (gate_type == trt.RNNGateType.CELL):
                        weights = trt.Weights(weight_hh[2].numpy().astype(np.float32))
                        bias = trt.Weights(bias_hh[2].numpy().astype(np.float32))
                    elif (gate_type == trt.RNNGateType.OUTPUT):
                        weights = trt.Weights(weight_hh[3].numpy().astype(np.float32))
                        bias = trt.Weights(bias_hh[3].numpy().astype(np.float32))

                layer_idx = idx if not is_unrolled else 0
                layer.set_weights_for_gate(layer_idx, gate_type, is_w, weights)
                layer.set_bias_for_gate(layer_idx, gate_type, is_w, bias)

    def add_rnns(self, num_layers, max_seq_length, input_tensor, length_tensor, length_tensor_host, input_size, hidden_size, hidden_state_tensor, cell_state_tensor, name):
        if dict_get(self.args, "disable_encoder_plugin", default=False):
            rnn_layer = self.network.add_rnn_v2(input_tensor, num_layers, hidden_size, max_seq_length, trt.RNNOperation.LSTM)
            rnn_layer.seq_lengths = length_tensor
            rnn_layer.name = name
            # connect the initial hidden/cell state tensors (if they exist)
            if hidden_state_tensor:
                rnn_layer.hidden_state = hidden_state_tensor
            if cell_state_tensor:
                rnn_layer.cell_state = cell_state_tensor

            for i in range(rnn_layer.num_layers):
                self._init_weights_per_layer(rnn_layer, idx=i)

            return rnn_layer
        else:
            layer = None
            plugin = None
            plugin_name = "RNNTEncoderPlugin"
            calibration_dict = self.parse_calibration()

            for plugin_creator in trt.get_plugin_registry().plugin_creator_list:
                if plugin_creator.name == plugin_name:
                    logging.info("RNNTEncoderPlugin Plugin found")

                    fields = []

                    fields.append(trt.PluginField("numLayers", np.array([num_layers], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("hiddenSize", np.array([hidden_size], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("inputSize", np.array([input_size], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("max_seq_length", np.array([max_seq_length], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("max_batch_size", np.array([self.batch_size], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("dataType", np.array([trt.DataType.INT8], dtype=np.int32), trt.PluginFieldType.INT32))

                    for layer in range(num_layers):
                        weightsI = RNNTBaseBuilder.state_dict[name + '.weight_ih_l' + str(layer)]
                        weightsH = RNNTBaseBuilder.state_dict[name + '.weight_hh_l' + str(layer)]

                        if layer == 0:
                            assert(weightsI.numpy().astype(np.float16).size == 4 * hidden_size * input_size)
                        else:
                            assert(weightsI.numpy().astype(np.float16).size == 4 * hidden_size * hidden_size)
                        assert(weightsH.numpy().astype(np.float16).size == 4 * hidden_size * hidden_size)

                        fields.append(trt.PluginField("weightsI", weightsI.numpy().astype(np.float16), trt.PluginFieldType.FLOAT16))
                        fields.append(trt.PluginField("weightsH", weightsH.numpy().astype(np.float16), trt.PluginFieldType.FLOAT16))

                    for layer in range(num_layers):
                        biases = torch.cat((RNNTBaseBuilder.state_dict[name + '.bias_ih_l' + str(layer)], RNNTBaseBuilder.state_dict[name + '.bias_hh_l' + str(layer)]), 0)

                        fields.append(trt.PluginField("bias", biases.numpy().astype(np.float16), trt.PluginFieldType.FLOAT16))

                    scaleFactors = []

                    if name == "encoder.pre_rnn.lstm":
                        scaleFactors.append(1 / calibration_dict["input"])
                    elif name == "encoder.post_rnn.lstm":
                        scaleFactors.append(1 / calibration_dict["encoder_reshape"])

                    else:
                        logging.error("Unrecognised name in add_rnns")

                    fields.append(trt.PluginField("scaleFactors", np.array(scaleFactors, dtype=np.float32), trt.PluginFieldType.FLOAT32))

                    field_collection = trt.PluginFieldCollection(fields)

                    plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)

                    inputs = []
                    inputs.append(input_tensor)
                    inputs.append(hidden_state_tensor)
                    inputs.append(cell_state_tensor)
                    inputs.append(length_tensor)
                    inputs.append(length_tensor_host)

                    layer = self.network.add_plugin_v2(inputs, plugin)
                    layer.name = name

                    break

            if not plugin:
                logging.error("RNNTEncoderPlugin not found")
            if not layer:
                logging.error("Layer {} not set".format(name))

            return layer

    def initialize(self):
        super().initialize()

        # Create network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        input_tensor = self.network.add_input("input", trt.DataType.FLOAT, (-1, self.max_seq_length, RNNHyperParam.encoder_input_size))
        set_tensor_dtype(input_tensor, self.input_dtype, self.input_format)

        length_tensor = self.network.add_input("length", trt.DataType.INT32, (-1,))
        length_tensor_host = self.network.add_input("length_host", trt.DataType.INT32, (-1,))

        # compute (seq_length + 1) // 2
        one_constant = self.network.add_constant((1,), np.array([1]).astype(np.int32))
        one_constant.get_output(0).name = "one_constant"
        length_add_one = self.network.add_elementwise(length_tensor, one_constant.get_output(0), trt.ElementWiseOperation.SUM)
        length_add_one.get_output(0).name = "length_add_one"
        two_constant = self.network.add_constant((1,), np.array([2]).astype(np.int32))
        two_constant.get_output(0).name = "two_constant"
        length_half = self.network.add_elementwise(length_add_one.get_output(0), two_constant.get_output(0), trt.ElementWiseOperation.FLOOR_DIV)
        length_half.get_output(0).name = "length_half"

        # state handling
        enc_tensor_dict = {'lower': dict(), 'upper': dict()}
        for tensor_name in ['hidden', 'cell']:
            if self.expose_state:
                enc_tensor_dict['lower'][tensor_name] = self.network.add_input("lower_" + tensor_name, trt.DataType.FLOAT, (-1, RNNHyperParam.enc_pre_rnn_layers, RNNHyperParam.encoder_hidden_size))
                enc_tensor_dict['upper'][tensor_name] = self.network.add_input("upper_" + tensor_name, trt.DataType.FLOAT, (-1, RNNHyperParam.enc_post_rnn_layers, RNNHyperParam.encoder_hidden_size))
                set_tensor_dtype(enc_tensor_dict['lower'][tensor_name], self.input_dtype, self.input_format)
                set_tensor_dtype(enc_tensor_dict['upper'][tensor_name], self.input_dtype, self.input_format)
            else:
                enc_tensor_dict['lower'][tensor_name] = None
                enc_tensor_dict['upper'][tensor_name] = None

        # instantiate layers
        #

        # pre_rnn
        encoder_add_rnn_dispatch = self.add_unrolled_rnns if self.unroll else self.add_rnns
        encoder_lower = encoder_add_rnn_dispatch(RNNHyperParam.enc_pre_rnn_layers,
                                                 self.max_seq_length,
                                                 input_tensor,
                                                 length_tensor,
                                                 length_tensor_host,
                                                 RNNHyperParam.encoder_input_size,
                                                 RNNHyperParam.encoder_hidden_size,
                                                 enc_tensor_dict['lower']['hidden'],
                                                 enc_tensor_dict['lower']['cell'],
                                                 'encoder.pre_rnn.lstm')
        # reshape (stack time x 2)
        reshape_layer = self.network.add_shuffle(encoder_lower.get_output(0))
        reshape_layer.reshape_dims = trt.Dims((0, self.max_seq_length // 2, RNNHyperParam.encoder_hidden_size * 2))
        reshape_layer.name = 'encoder_reshape'
        reshape_layer.get_output(0).name = 'encoder_reshape'

        # post_nnn
        encoder_upper = encoder_add_rnn_dispatch(RNNHyperParam.enc_post_rnn_layers,
                                                 self.max_seq_length // 2,
                                                 reshape_layer.get_output(0),
                                                 length_half.get_output(0),
                                                 length_tensor_host,
                                                 RNNHyperParam.encoder_hidden_size * 2,
                                                 RNNHyperParam.encoder_hidden_size,
                                                 enc_tensor_dict['upper']['hidden'],
                                                 enc_tensor_dict['upper']['cell'],
                                                 'encoder.post_rnn.lstm')

        # Add expected names for "regular" LSTM layers.
        if not self.unroll:
            encoder_lower.name = 'encoder_pre_rnn'
            encoder_lower.get_output(0).name = "encoder_pre_rnn_output"
            encoder_lower.get_output(1).name = "encoder_pre_rnn_hidden"
            encoder_lower.get_output(2).name = "encoder_pre_rnn_cell"

            encoder_upper.name = 'encoder_post_rnn'
            encoder_upper.get_output(0).name = "encoder_post_rnn_output"
            encoder_upper.get_output(1).name = "encoder_post_rnn_hidden"
            encoder_upper.get_output(2).name = "encoder_post_rnn_cell"

        # mark outputs
        self.network.mark_output(encoder_upper.get_output(0))
        set_tensor_dtype(encoder_upper.get_output(0), self.input_dtype, self.input_format)
        if self.expose_state:
            # lower_hidden
            self.network.mark_output(encoder_lower.get_output(1))
            set_tensor_dtype(encoder_lower.get_output(1), self.input_dtype, self.input_format)
            # upper_hidden
            self.network.mark_output(encoder_upper.get_output(1))
            set_tensor_dtype(encoder_upper.get_output(1), self.input_dtype, self.input_format)
            # lower_cell
            self.network.mark_output(encoder_lower.get_output(2))
            set_tensor_dtype(encoder_lower.get_output(2), self.input_dtype, self.input_format)
            # upper_cell
            self.network.mark_output(encoder_upper.get_output(2))
            set_tensor_dtype(encoder_upper.get_output(2), self.input_dtype, self.input_format)

# Decoder class
##


class DecoderBuilder(RNNTBaseBuilder):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.name = "decoder.plan"

    def add_decoder_rnns(self, num_layers, input_tensor, hidden_size, hidden_state_tensor, cell_state_tensor, name):
        max_seq_length = 1   # processed single step
        if not dict_get(self.args, "decoderPlugin", default=True):
            rnn_layer = self.network.add_rnn_v2(input_tensor, num_layers, hidden_size, max_seq_length, trt.RNNOperation.LSTM)

            # connect the initial hidden/cell state tensors
            rnn_layer.hidden_state = hidden_state_tensor
            rnn_layer.cell_state = cell_state_tensor

            # initialization of the gate weights
            for i in range(num_layers):
                weight_ih = RNNTBaseBuilder.state_dict[name + '.weight_ih_l' + str(i)]
                weight_ih = weight_ih.chunk(4, 0)

                weight_hh = RNNTBaseBuilder.state_dict[name + '.weight_hh_l' + str(i)]
                weight_hh = weight_hh.chunk(4, 0)

                bias_ih = RNNTBaseBuilder.state_dict[name + '.bias_ih_l' + str(i)]
                bias_ih = bias_ih.chunk(4, 0)

                bias_hh = RNNTBaseBuilder.state_dict[name + '.bias_hh_l' + str(i)]
                bias_hh = bias_hh.chunk(4, 0)

                for gate_type in [trt.RNNGateType.INPUT, trt.RNNGateType.CELL, trt.RNNGateType.FORGET, trt.RNNGateType.OUTPUT]:
                    for is_w in [True, False]:
                        if is_w:
                            if (gate_type == trt.RNNGateType.INPUT):
                                weights = trt.Weights(weight_ih[0].numpy().astype(np.float32))
                                bias = trt.Weights(bias_ih[0].numpy().astype(np.float32))
                            elif (gate_type == trt.RNNGateType.FORGET):
                                weights = trt.Weights(weight_ih[1].numpy().astype(np.float32))
                                bias = trt.Weights(bias_ih[1].numpy().astype(np.float32))
                            elif (gate_type == trt.RNNGateType.CELL):
                                weights = trt.Weights(weight_ih[2].numpy().astype(np.float32))
                                bias = trt.Weights(bias_ih[2].numpy().astype(np.float32))
                            elif (gate_type == trt.RNNGateType.OUTPUT):
                                weights = trt.Weights(weight_ih[3].numpy().astype(np.float32))
                                bias = trt.Weights(bias_ih[3].numpy().astype(np.float32))
                        else:
                            if (gate_type == trt.RNNGateType.INPUT):
                                weights = trt.Weights(weight_hh[0].numpy().astype(np.float32))
                                bias = trt.Weights(bias_hh[0].numpy().astype(np.float32))
                            elif (gate_type == trt.RNNGateType.FORGET):
                                weights = trt.Weights(weight_hh[1].numpy().astype(np.float32))
                                bias = trt.Weights(bias_hh[1].numpy().astype(np.float32))
                            elif (gate_type == trt.RNNGateType.CELL):
                                weights = trt.Weights(weight_hh[2].numpy().astype(np.float32))
                                bias = trt.Weights(bias_hh[2].numpy().astype(np.float32))
                            elif (gate_type == trt.RNNGateType.OUTPUT):
                                weights = trt.Weights(weight_hh[3].numpy().astype(np.float32))
                                bias = trt.Weights(bias_hh[3].numpy().astype(np.float32))

                        rnn_layer.set_weights_for_gate(i, gate_type, is_w, weights)
                        rnn_layer.set_bias_for_gate(i, gate_type, is_w, bias)

            return rnn_layer
        else:
            layer = None
            plugin = None
            plugin_name = "RNNTDecoderPlugin"

            # logging.info(trt.get_plugin_registry().plugin_creator_list)

            for plugin_creator in trt.get_plugin_registry().plugin_creator_list:
                if plugin_creator.name == plugin_name:
                    logging.info("Decoder Plugin found")

                    fields = []

                    fields.append(trt.PluginField("numLayers", np.array([num_layers], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("hiddenSize", np.array([hidden_size], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("inputSize", np.array([hidden_size], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("dataType", np.array([trt.DataType.HALF], dtype=np.int32), trt.PluginFieldType.INT32))

                    for layer in range(num_layers):
                        weights = torch.cat((RNNTBaseBuilder.state_dict[name + '.weight_ih_l' + str(layer)], RNNTBaseBuilder.state_dict[name + '.weight_hh_l' + str(layer)]), 0)

                        assert(weights.numpy().astype(np.float16).size == 8 * hidden_size * hidden_size)

                        fields.append(trt.PluginField("weights", weights.numpy().astype(np.float16), trt.PluginFieldType.FLOAT16))

                    for layer in range(num_layers):
                        biases = torch.cat((RNNTBaseBuilder.state_dict[name + '.bias_ih_l' + str(layer)], RNNTBaseBuilder.state_dict[name + '.bias_hh_l' + str(layer)]), 0)

                        fields.append(trt.PluginField("bias", biases.numpy().astype(np.float16), trt.PluginFieldType.FLOAT16))

                    field_collection = trt.PluginFieldCollection(fields)

                    plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)

                    inputs = []
                    inputs.append(input_tensor)
                    inputs.append(hidden_state_tensor)
                    inputs.append(cell_state_tensor)

                    layer = self.network.add_plugin_v2(inputs, plugin)

                    break

            if not plugin:
                logging.error("Plugin not found")

            return layer

    def initialize(self):
        super().initialize()

        # Create network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Decoder
        #    Embedding layer : (29 => 320)
        #    Rnn             : LSTM layers=2, h=320

        # Embedding layer
        #   gather layer with LUT of RNNHyperParam.labels_size=29 entries with RNNHyperParam.decoder_input_size=320 size per entry
        #   blank token does not need to be looked up, whereas the SOS (start-of-sequence) requires all zeros for embed vector
        dec_embedding_input = self.network.add_input("dec_embedding_input", trt.DataType.INT32, (-1, 1))
        dec_embedding_orig = RNNTBaseBuilder.state_dict["prediction.embed.weight"].numpy().astype(np.float32)
        dec_embedding_sos = np.zeros((1, RNNHyperParam.decoder_input_size), dtype=np.float32)
        dec_embedding_weights = trt.Weights(np.concatenate((dec_embedding_orig, dec_embedding_sos), axis=0))
        dec_embedding_lut = self.network.add_constant((RNNHyperParam.labels_size, RNNHyperParam.decoder_input_size), dec_embedding_weights)
        self.dec_embedding = self.network.add_gather(dec_embedding_lut.get_output(0), dec_embedding_input, axis=0)
        self.dec_embedding.name = 'decoder_embedding'

        # Rnn layer
        dec_rnn_layers = RNNHyperParam.dec_rnn_layers

        # Create tensors  [ batch, seq=1, input ]
        dec_tensor_dict = dict()
        # dec_tensor_dict['input']  = self.network.add_input("dec_input", trt.DataType.FLOAT, (-1, 1, RNNHyperParam.decoder_input_size))
        dec_tensor_dict['input'] = self.dec_embedding.get_output(0)
        dec_tensor_dict['hidden'] = self.network.add_input("hidden", trt.DataType.FLOAT, (-1, dec_rnn_layers, RNNHyperParam.decoder_hidden_size))
        dec_tensor_dict['cell'] = self.network.add_input("cell", trt.DataType.FLOAT, (-1, dec_rnn_layers, RNNHyperParam.decoder_hidden_size))
        for dec_tensor_name, dec_tensor_val in dec_tensor_dict.items():
            # RNN input is an internal layer whose type we should let TRT determine
            if dec_tensor_name != 'input':
                set_tensor_dtype(dec_tensor_val, self.input_dtype, self.input_format)

        # Instantiate RNN
        # logging.info("dec_input_size = {:}".format(dec_input_size))
        logging.info("dec_embed_lut OUT tensor shape = {:}".format(dec_embedding_lut.get_output(0).shape))
        logging.info("dec_embedding OUT tensor shape = {:}".format(self.dec_embedding.get_output(0).shape))
        self.decoder = self.add_decoder_rnns(dec_rnn_layers,
                                             dec_tensor_dict['input'],
                                             RNNHyperParam.decoder_hidden_size,
                                             dec_tensor_dict['hidden'],
                                             dec_tensor_dict['cell'],
                                             'prediction.dec_rnn.lstm')
        self.decoder.name = 'decoder_rnn'

        # Determine outputs (and override size)
        #   output
        #   hidden
        #   cell
        for output_idx in range(3):
            output_tensor = self.decoder.get_output(output_idx)
            self.network.mark_output(output_tensor)
            set_tensor_dtype(output_tensor, self.input_dtype, self.input_format)


# Joint class
##

# Famility of network components for Joint
class JointNetComponents():

    def create_split_fc1_layer(layer_name,
                               network,
                               input_tensor,
                               input_size,
                               output_size,
                               weight_offset,
                               joint_fc1_weight_ckpt,
                               joint_fc1_bias_ckpt,
                               add_bias=False):

        # detach weight (using weight_offset)
        joint_fc1_kernel_np = np.zeros((output_size, input_size))
        for i in range(output_size):
            for j in range(input_size):
                joint_fc1_kernel_np[i][j] = joint_fc1_weight_ckpt.numpy()[i][j + weight_offset]
        joint_fc1_kernel = joint_fc1_kernel_np.astype(np.float32)

        # detach bias (if available)
        joint_fc1_bias_np = np.zeros((output_size))
        if add_bias:
            for i in range(output_size):
                joint_fc1_bias_np[i] = joint_fc1_bias_ckpt.numpy()[i]
            joint_fc1_bias = joint_fc1_bias_np.astype(np.float32)

        # instantiate FC layer
        if add_bias:
            joint_fc1 = network.add_fully_connected(
                input_tensor,
                output_size,
                joint_fc1_kernel,
                joint_fc1_bias)
        else:
            joint_fc1 = network.add_fully_connected(
                input_tensor,
                output_size,
                joint_fc1_kernel)

        # epilogue
        joint_fc1.name = layer_name
        return joint_fc1

# Detached FC1_a and FC1_b builder


class JointFc1Builder(RNNTBaseBuilder):
    def __init__(self, name, port, args):
        super().__init__(args)
        self.name = name
        self.name = name + ".plan"
        if (port != 'encoder' and port != 'decoder'):
            logging.info("JointFc1Builder: unrecognized port")
        self.port = port

    def initialize(self):
        super().initialize()

        # Create network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Create tensors  [ batch, seq=1, input ]
        joint_tensor_dict = dict()
        if self.port == 'encoder':
            joint_tensor_dict['enc_input'] = self.network.add_input("enc_input", trt.DataType.FLOAT, (-1, RNNHyperParam.encoder_hidden_size, 1, 1))
        if self.port == 'decoder':
            joint_tensor_dict['dec_input'] = self.network.add_input("dec_input", trt.DataType.FLOAT, (-1, RNNHyperParam.decoder_hidden_size, 1, 1))

        for joint_tensor in joint_tensor_dict.values():
            set_tensor_dtype(joint_tensor, self.input_dtype, "hwc8")  # hwc8 to avoid reformatting

        # FC1 + bias :
        joint_fc1_output_size = RNNHyperParam.joint_hidden_size
        joint_fc1_weight_ckpt = RNNTBaseBuilder.state_dict['joint_net.0.weight']
        joint_fc1_bias_ckpt = RNNTBaseBuilder.state_dict['joint_net.0.bias']

        # Instantiate two split FC1's : one for the encoder and one for the decoder
        if self.port == 'encoder':
            joint_fc1_a = JointNetComponents.create_split_fc1_layer('joint_fc1_a',
                                                                    self.network,
                                                                    joint_tensor_dict['enc_input'],
                                                                    RNNHyperParam.encoder_hidden_size,
                                                                    joint_fc1_output_size,
                                                                    0,
                                                                    joint_fc1_weight_ckpt,
                                                                    joint_fc1_bias_ckpt,
                                                                    True)
            final_output = joint_fc1_a.get_output(0)

        if self.port == 'decoder':
            joint_fc1_b = JointNetComponents.create_split_fc1_layer('joint_fc1_b',
                                                                    self.network,
                                                                    joint_tensor_dict['dec_input'],
                                                                    RNNHyperParam.decoder_hidden_size,
                                                                    joint_fc1_output_size,
                                                                    RNNHyperParam.encoder_hidden_size,
                                                                    joint_fc1_weight_ckpt,
                                                                    joint_fc1_bias_ckpt)
            final_output = joint_fc1_b.get_output(0)

        # set output properties
        self.network.mark_output(final_output)
        set_tensor_dtype(final_output, self.input_dtype, "hwc8")  # hwc8 to avoid reformatting


# fc1_a and fc1_b classes for encoder / decoder
def JointFc1_A_Builder(args): return JointFc1Builder("fc1_a", "encoder", args)
def JointFc1_B_Builder(args): return JointFc1Builder("fc1_b", "decoder", args)

# Detached Joint backed builder (FC1_SUM + FC1_RELU + FC2 + topK)


class JointBackendBuilder(RNNTBaseBuilder):
    def __init__(self, args):
        super().__init__(args)
        self.name = "joint_backend.plan"
        self.dump_joint_fc2_weights = not dict_get(args, "no_dump_joint_fc2_weights", default=False)

    def initialize(self):
        super().initialize()

        # Create network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Create tensors  [ batch, seq=1, input ]
        joint_fc1_output_size = RNNHyperParam.joint_hidden_size
        joint_tensor_dict = dict()
        joint_tensor_dict['joint_fc1_a_output'] = self.network.add_input("joint_fc1_a_output", trt.DataType.FLOAT, (-1, 1, joint_fc1_output_size))
        joint_tensor_dict['joint_fc1_b_output'] = self.network.add_input("joint_fc1_b_output", trt.DataType.FLOAT, (-1, 1, joint_fc1_output_size))
        for joint_tensor in joint_tensor_dict.values():
            set_tensor_dtype(joint_tensor, self.input_dtype, self.input_format)

        # element_wise SUM
        joint_fc1_sum = self.network.add_elementwise(joint_tensor_dict['joint_fc1_a_output'], joint_tensor_dict['joint_fc1_b_output'], trt.ElementWiseOperation.SUM)
        joint_fc1_sum.name = 'joint_fc1_sum'

        # reLU
        joint_relu = self.network.add_activation(joint_fc1_sum.get_output(0), trt.ActivationType.RELU)
        joint_relu.name = 'joint_relu'

        # FC2 + bias :
        joint_fc2_input_size = joint_fc1_output_size
        joint_fc2_output_size = RNNHyperParam.labels_size
        joint_fc2_weight_ckpt = RNNTBaseBuilder.state_dict['joint_net.3.weight']
        joint_fc2_bias_ckpt = RNNTBaseBuilder.state_dict['joint_net.3.bias']
        joint_fc2_kernel = trt.Weights(joint_fc2_weight_ckpt.numpy().astype(np.float32))
        joint_fc2_bias = trt.Weights(joint_fc2_bias_ckpt.numpy().astype(np.float32))

        joint_fc2_shuffle = self.network.add_shuffle(joint_relu.get_output(0))   # Add an extra dimension for FC processing
        joint_fc2_shuffle.reshape_dims = (-1, joint_fc2_input_size, 1, 1)
        joint_fc2_shuffle.name = 'joint_fc2_shuffle'

        joint_fc2 = self.network.add_fully_connected(
            joint_fc2_shuffle.get_output(0),
            joint_fc2_output_size,
            joint_fc2_kernel,
            joint_fc2_bias)
        joint_fc2.name = 'joint_fc2'

        # opt = GREEDY
        # -------------
        #    - Do not use softmax layer
        #    - Use TopK (K=1) GPU sorting

        # TopK (k=1)
        red_dim = 1 << 1
        joint_top1 = self.network.add_topk(joint_fc2.get_output(0), trt.TopKOperation.MAX, 1, red_dim)
        joint_top1.name = 'joint_top1'

        # Final output
        # final_output = joint_fc2.get_output(0)
        final_output = joint_top1.get_output(1)
        self.network.mark_output(final_output)

        # epilogue: dump fc2 weights and bias if required
        if self.dump_joint_fc2_weights:
            joint_fc2_weight_ckpt.numpy().astype(np.float16).tofile(self.engine_dir + '/joint_fc2_weight_ckpt.fp16.dat')
            joint_fc2_bias_ckpt.numpy().astype(np.float16).tofile(self.engine_dir + '/joint_fc2_bias_ckpt.fp16.dat')
            joint_fc2_weight_ckpt.numpy().astype(np.float32).tofile(self.engine_dir + '/joint_fc2_weight_ckpt.fp32.dat')
            joint_fc2_bias_ckpt.numpy().astype(np.float32).tofile(self.engine_dir + '/joint_fc2_bias_ckpt.fp32.dat')

# Full Joint builder: FC1 + FC2 + softmax/topK


class JointBuilder(RNNTBaseBuilder):
    def __init__(self, args):
        super().__init__(args)
        self.name = "joint.plan"

    def initialize(self):
        super().initialize()

        # Create network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Joint: [ BS, seq=1, channel ]
        #    Concat          : (enc:1024 + pred:320 )
        #    FC1             : FC 1344 x 512
        #    bias1           :
        #    reLU            : reLU
        #    FC2             : FC 512 x 29
        #    bias2           :

        # Create tensors  [ batch, seq=1, input ]
        joint_tensor_dict = dict()
        joint_tensor_dict['enc_input'] = self.network.add_input("enc_input", trt.DataType.FLOAT, (-1, 1, RNNHyperParam.encoder_hidden_size))
        joint_tensor_dict['dec_input'] = self.network.add_input("dec_input", trt.DataType.FLOAT, (-1, 1, RNNHyperParam.decoder_hidden_size))
        for joint_tensor in joint_tensor_dict.values():
            set_tensor_dtype(joint_tensor, self.input_dtype, self.input_format)

        # FC1 + bias :
        joint_fc1_output_size = RNNHyperParam.joint_hidden_size
        joint_fc1_weight_ckpt = RNNTBaseBuilder.state_dict['joint_net.0.weight']
        joint_fc1_bias_ckpt = RNNTBaseBuilder.state_dict['joint_net.0.bias']

        # Instantiate two split FC1's : one for the encoder and one for the decoder
        joint_fc1_a = JointNetComponents.create_split_fc1_layer('joint_fc1_a',
                                                                self.network,
                                                                joint_tensor_dict['enc_input'],
                                                                RNNHyperParam.encoder_hidden_size,
                                                                joint_fc1_output_size,
                                                                0,
                                                                joint_fc1_weight_ckpt,
                                                                joint_fc1_bias_ckpt,
                                                                True)

        joint_fc1_b = JointNetComponents.create_split_fc1_layer('joint_fc1_b',
                                                                self.network,
                                                                joint_tensor_dict['dec_input'],
                                                                RNNHyperParam.decoder_hidden_size,
                                                                joint_fc1_output_size,
                                                                RNNHyperParam.encoder_hidden_size,
                                                                joint_fc1_weight_ckpt,
                                                                joint_fc1_bias_ckpt)

        # element_wise SUM
        joint_fc1_sum = self.network.add_elementwise(joint_fc1_a.get_output(0), joint_fc1_b.get_output(0), trt.ElementWiseOperation.SUM)
        joint_fc1_sum.name = 'joint_fc1_sum'

        # reLU
        joint_relu = self.network.add_activation(joint_fc1_sum.get_output(0), trt.ActivationType.RELU)
        joint_relu.name = 'joint_relu'

        # FC2 + bias :
        joint_fc2_input_size = joint_fc1_output_size
        joint_fc2_output_size = RNNHyperParam.labels_size
        joint_fc2_weight_ckpt = RNNTBaseBuilder.state_dict['joint_net.3.weight']
        joint_fc2_bias_ckpt = RNNTBaseBuilder.state_dict['joint_net.3.bias']
        joint_fc2_kernel = trt.Weights(joint_fc2_weight_ckpt.numpy().astype(np.float32))
        joint_fc2_bias = trt.Weights(joint_fc2_bias_ckpt.numpy().astype(np.float32))

        joint_fc2_shuffle = self.network.add_shuffle(joint_relu.get_output(0))   # Add an extra dimension for FC processing
        joint_fc2_shuffle.reshape_dims = (-1, joint_fc2_input_size, 1, 1)
        joint_fc2_shuffle.name = 'joint_fc2_shuffle'

        joint_fc2 = self.network.add_fully_connected(
            joint_fc2_shuffle.get_output(0),
            joint_fc2_output_size,
            joint_fc2_kernel,
            joint_fc2_bias)
        joint_fc2.name = 'joint_fc2'

        # opt = DEFAULT
        # -------------
        #    - Use softmax layer
        #    - No GPU sorting
        if self.opt == 'default':
            # Softmax
            softmax_layer = self.network.add_softmax(joint_fc2.get_output(0))
            softmax_layer.name = 'joint_softmax'

            # Final output
            final_output = softmax_layer.get_output(0)
            self.network.mark_output(final_output)
            set_tensor_dtype(final_output, self.input_dtype, self.input_format)

        # opt = GREEDY
        # -------------
        #    - Do not use softmax layer
        #    - Use TopK (K=1) GPU sorting
        elif self.opt == 'greedy':
            # TopK (k=1)
            red_dim = 1 << 1
            joint_top1 = self.network.add_topk(joint_fc2.get_output(0), trt.TopKOperation.MAX, 1, red_dim)
            joint_top1.name = 'joint_top1'

            # Final output
            # final_output = joint_fc2.get_output(0)
            final_output = joint_top1.get_output(1)
            self.network.mark_output(final_output)

# Isel class
##


class IselBuilder(RNNTBaseBuilder):
    def __init__(self, args):
        super().__init__(args)
        self.name = "isel.plan"

    def initialize(self):
        super().initialize()

        # Create network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Isel:
        #    output_hidden: [ BS, layers=2, decoder_hidden_size=320 ]
        #    output_cell  : [ BS, layers=2, decoder_hidden_size=320 ]
        #
        #    input_select : [ BS, 1, 1 ]
        #    input0_hidden: [ BS, layers=2, decoder_hidden_size=320 ]
        #    input0_cell  : [ BS, layers=2, decoder_hidden_size=320 ]
        #    input1_hidden: [ BS, layers=2, decoder_hidden_size=320 ]
        #    input1_cell  : [ BS, layers=2, decoder_hidden_size=320 ]

        # Declare input tensors: port 0
        input0_hidden = self.network.add_input("input0_hidden", trt.DataType.FLOAT, (-1, RNNHyperParam.dec_rnn_layers, RNNHyperParam.decoder_hidden_size))
        input0_cell = self.network.add_input("input0_cell", trt.DataType.FLOAT, (-1, RNNHyperParam.dec_rnn_layers, RNNHyperParam.decoder_hidden_size))
        if self.opt == "greedy":
            input0_winner = self.network.add_input("input0_winner", trt.DataType.INT32, (-1, 1, 1))

        # Declare input tensors: port 1
        input1_hidden = self.network.add_input("input1_hidden", trt.DataType.FLOAT, (-1, RNNHyperParam.dec_rnn_layers, RNNHyperParam.decoder_hidden_size))
        input1_cell = self.network.add_input("input1_cell", trt.DataType.FLOAT, (-1, RNNHyperParam.dec_rnn_layers, RNNHyperParam.decoder_hidden_size))
        if self.opt == "greedy":
            input1_winner = self.network.add_input("input1_winner", trt.DataType.INT32, (-1, 1, 1))

        # Reformat tensors
        for input_tensor in (input0_hidden, input0_cell, input1_hidden, input1_cell):
            set_tensor_dtype(input_tensor, self.input_dtype, self.input_format)

        # One Iselect layer per component
        if self.input_dtype != "fp16" or self.opt != "greedy":
            # logging.info("Not using select plugin due to input datatype not being fp16 or opt not being greedy")
            # assert(False);
            # Select tensor
            input_select = self.network.add_input("input_select", trt.DataType.BOOL, (-1, 1, 1))

            isel_hidden = self.network.add_select(input_select, input0_hidden, input1_hidden)
            isel_cell = self.network.add_select(input_select, input0_cell, input1_cell)
            isel_hidden.name = 'Iselect Dec hidden'
            isel_cell.name = 'Iselect Dec cell'
            if self.opt == "greedy":
                isel_winner = self.network.add_select(input_select, input0_winner, input1_winner)
                isel_winner.name = 'Iselect Dec winner'

            # Declare outputs
            output_hidden = isel_hidden.get_output(0)
            output_cell = isel_cell.get_output(0)
            self.network.mark_output(output_hidden)
            self.network.mark_output(output_cell)
            set_tensor_dtype(output_hidden, self.input_dtype, self.input_format)
            set_tensor_dtype(output_cell, self.input_dtype, self.input_format)

            if self.opt == "greedy":
                output_winner = isel_winner.get_output(0)
                self.network.mark_output(output_winner)

        else:
            sel3Layer = None
            plugin = None
            plugin_name = "RNNTSelectPlugin"

            # Select tensor
            input_select = self.network.add_input("input_select", trt.DataType.INT32, (-1, 1, 1))

            for plugin_creator in trt.get_plugin_registry().plugin_creator_list:
                if plugin_creator.name == plugin_name:
                    logging.info("Select Plugin found")

                    fields = []

                    field_collection = trt.PluginFieldCollection(fields)

                    plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)

                    inputs = []
                    inputs.append(input_select)
                    inputs.append(input0_hidden)
                    inputs.append(input1_hidden)
                    inputs.append(input0_cell)
                    inputs.append(input1_cell)
                    inputs.append(input0_winner)
                    inputs.append(input1_winner)

                    sel3Layer = self.network.add_plugin_v2(inputs, plugin)

                    sel3Layer.name = 'Select3'

                    break

            if not plugin:
                logging.error("Select plugin not found")

            # Declare outputs
            output_hidden = sel3Layer.get_output(0)
            output_cell = sel3Layer.get_output(1)
            self.network.mark_output(output_hidden)
            self.network.mark_output(output_cell)
            set_tensor_dtype(output_hidden, self.input_dtype, self.input_format)
            set_tensor_dtype(output_cell, self.input_dtype, self.input_format)

            output_winner = sel3Layer.get_output(2)
            self.network.mark_output(output_winner)
            set_tensor_dtype(output_winner, "int32", self.input_format)


# Igather class
##

class IgatherBuilder(RNNTBaseBuilder):
    def __init__(self, args):
        super().__init__(args)
        self.name = "igather.plan"

    def initialize(self):
        super().initialize()

        # Create network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Igather:
        #    encoder_input  : [ BS, SEQ//2=1152//2, encoder_hidden_size=1024 ]  native
        #    t_coordinate   : [ BS, 1, ]                                        int32
        #
        #    igather_output : [ BS, 1,  encoder_hidden_size=1024 ]              native

        # Declare input tensors
        encoder_input = self.network.add_input("encoder_input", trt.DataType.FLOAT, (-1, self.max_seq_length // 2, RNNHyperParam.encoder_hidden_size))
        # t_coordinate  = self.network.add_input("t_coordinate", trt.DataType.INT32, (-1, 1))
        # t_coordinate  = self.network.add_input("t_coordinate", trt.DataType.INT32, trt.Dims([-1]))
        t_coordinate = self.network.add_input("t_coordinate", trt.DataType.INT32, trt.Dims([-1]))
        set_tensor_dtype(encoder_input, self.input_dtype, self.input_format)

        # igather_layer  = self.network.add_gather(encoder_input, t_coordinate, axis=0)
        igather_layer = self.network.add_gather(encoder_input, t_coordinate, axis=1)
        igather_layer.name = "Igather joint cell"
        igather_layer.num_elementwise_dims = 1

        # Declare outputs
        igather_output = igather_layer.get_output(0)
        self.network.mark_output(igather_output)
        set_tensor_dtype(igather_output, self.input_dtype, self.input_format)


# Main methods
##

class DisaggregatedJointBuilder(MultiBuilder):

    builders = {
        "joint_fc1_a": JointFc1_A_Builder,
        "joint_fc2_b": JointFc1_B_Builder,
        "joint_backend": JointBackendBuilder,
    }

    def __init__(self, args):
        super().__init__(DisaggregatedJointBuilder.builders.values(), args)


class RNNTBuilder(MultiBuilder):

    builders = {
        "encoder": EncoderBuilder,
        "decoder": DecoderBuilder,
        "isel": IselBuilder,
        "igather": IgatherBuilder,
    }

    def __init__(self, args):
        super().__init__(RNNTBuilder.builders.values(), args)

        audio_fp16_input = dict_get(args, "audio_fp16_input", default=True)

        # These flags are only exposed if this file is run directly, not through the Makefile pipeline
        topology = dict_get(args, "topology", default="build_all")
        disagg_joint = dict_get(args, "disaggregated_joint", default=True)

        if disagg_joint:
            self.builders.append(DisaggregatedJointBuilder)
        else:
            self.builders.append(JointBuilder)

        # topology overrides which builders we want to build
        if topology in RNNTBuilder.builders:
            self.builders = [RNNTBuilder.builders[topology]]
        elif topology == "joint":
            self.builders = [JointBuilder]
        elif topology == "build_all":
            # This case is here to explicitly say that it is the default case.
            None
        else:
            raise(Exception("Unknown topology: {}".format(topology)))

        if not os.path.exists("build/bin/dali"):
            os.makedirs("build/bin/dali")

        filename = "build/bin/dali/dali_pipeline_gpu_{:}.pth".format("fp16" if audio_fp16_input else "fp32")
        dali_pipeline = DALIInferencePipeline.from_config(
            device="gpu",
            config=dict(),    # Default case
            device_id=0,
            batch_size=16,
            total_samples=16,  # Unused, can be set arbitrarily
            num_threads=2,
            audio_fp16_input=audio_fp16_input
        )
        dali_pipeline.serialize(filename=filename)

    def calibrate(self):
        enc_calib_args = dict(self.args)  # Make copy so we don't overwrite anything

        # These flags are required to run encoder in calibration mode
        enc_calib_args.update({
            "seq_splitting_off": True,
            "calibrate_encoder": True,
            "input_dtype": "fp32",
            "max_seq_length": 512,
            "force_calibration": True,
            "calib_max_batches": 30,
            "batch_size": 100,
            "enc_batch_size": 100,
            "calib_batch_size": 100,
            "calib_data_map": dict_get(self.args, "calib_data_map", default="data_maps/rnnt_train_clean_512/val_map.txt"),
            "preprocessed_data_dir": dict_get(os.environ, "PREPROCESSED_DATA_DIR", default="build/preprocessed_data")
        })

        RNNTBuilder.builders["encoder"](enc_calib_args).calibrate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--enc_batch_size", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--engine_dir", default="build/engines/rnnt")
    parser.add_argument("--config_ver", default="default")
    parser.add_argument("--verbose_nvtx", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--input_dtype", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--audio_fp16_input", type=bool, default=True)
    parser.add_argument("--input_format", choices=["linear", "hwc8", "chw4", "chw32"], default="linear")
    parser.add_argument("--topology", default="build_all", help="Options: encoder/decoder/joint/isel/build_all")
    parser.add_argument("--opt", choices=["default", "greedy"], default="greedy")
    parser.add_argument("--disable_encoder_plugin", default=False, help="Options: True/False")
    parser.add_argument("--decoderPlugin", default=True, help="Options: True/False")
    parser.add_argument("--seq_splitting_off", action="store_true")
    parser.add_argument("--disaggregated_joint", default=True, help="Options: True/False")
    parser.add_argument("--no_dump_joint_fc2_weights", action="store_true")
    parser.add_argument("--system_id", default="TitanRTX")
    parser.add_argument("--scenario", default="Offline")
    parser.add_argument("--calibrate_encoder", action="store_true", help="Overrides precision settings for encoder to int8. Must be used with --seq_splitting_off and --input_dtype=fp32. Ensure that max_seq_length is high enough for calibration data. Uses --calib_* parameters for configuration. Changes network description by expanding LSTMs in encoder")
    parser.add_argument("--calib_max_batches", type=int, default=100)
    parser.add_argument("--calib_batch_size", type=int, default=100)
    parser.add_argument("--force_calibration", action="store_true")
    parser.add_argument("--cache_file", type=str, default="code/rnnt/tensorrt/calibrator.cache")
    parser.add_argument("--calib_data_map", type=str, default="build/preprocessed_data/rnnt_train_clean_512_fp32/val_map_512.txt")
    parser.add_argument("--calib_data_dir", type=str, default="build/preprocessed_data/rnnt_train_clean_512_fp32/fp32")
    args = vars(parser.parse_known_args()[0])

    builder = RNNTBuilder(args)
    builder.build_engines()


if __name__ == "__main__":
    main()
