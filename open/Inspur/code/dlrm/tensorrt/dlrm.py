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

import csv
import ctypes
import os
import sys
import struct
sys.path.insert(0, os.getcwd())

import tensorrt as trt
from code.common import get_system
from code.common.system_list import Architecture

##
# Plugin .so files have to be loaded at global scope and before `import torch` to avoid cuda version mismatch.
##

DLRM_INTERACTIONS_PLUGIN_LIBRARY = "build/plugins/DLRMInteractionsPlugin/libdlrminteractionsplugin.so"
if not os.path.isfile(DLRM_INTERACTIONS_PLUGIN_LIBRARY):
    raise IOError("{}\n{}\n".format(
        "Failed to load library ({}).".format(DLRM_INTERACTIONS_PLUGIN_LIBRARY),
        "Please build the DLRM Interactions plugin."
    ))
ctypes.CDLL(DLRM_INTERACTIONS_PLUGIN_LIBRARY)

DLRM_BOTTOM_MLP_PLUGIN_LIBRARY = "build/plugins/DLRMBottomMLPPlugin/libdlrmbottommlpplugin.so"
if not os.path.isfile(DLRM_BOTTOM_MLP_PLUGIN_LIBRARY):
    raise IOError("{}\n{}\n".format(
        "Failed to load library ({}).".format(DLRM_BOTTOM_MLP_PLUGIN_LIBRARY),
        "Please build the DLRM Bottom MLP plugin."
    ))
ctypes.CDLL(DLRM_BOTTOM_MLP_PLUGIN_LIBRARY)

from importlib import import_module
from code.common import logging, dict_get, BENCHMARKS
from code.common.builder import BenchmarkBuilder
from code.dlrm.tensorrt.calibrator import DLRMCalibrator

import json
import numpy as np
import torch


class DLRMBuilder(BenchmarkBuilder):
    """Calibrate and build engines for DLRM."""

    def __init__(self, args):
        """Set up the config and calibrator for DLRM. Does not initialize."""

        workspace_size = dict_get(args, "workspace_size", default=(4 << 30))
        logging.info("Using workspace size: {:,}".format(workspace_size))

        super().__init__(args, name=BENCHMARKS.DLRM, workspace_size=workspace_size)

        with open("code/dlrm/tensorrt/mlperf_40m.limit.json") as f:
            self.dlrm_config = json.load(f)
        logging.info("DLRM config: {:}".format(self.dlrm_config))
        self.num_numerical_inputs = self.dlrm_config["num_numerical_features"]
        self.num_features = len(self.dlrm_config["categorical_feature_sizes"])
        self.num_interactions = (self.num_features + 1) * self.num_features // 2
        self.embedding_size = self.dlrm_config["embedding_dim"]
        self.embedding_rows = self.dlrm_config["categorical_feature_sizes"]
        self.embedding_rows_bound = 40000000
        self.embedding_rows = [min(i, self.embedding_rows_bound) for i in self.embedding_rows]
        self.embedding_rows_total = np.sum(np.array(self.embedding_rows))
        self.bottom_mlp_channels = self.dlrm_config["bottom_mlp_sizes"]
        self.bottom_mlp_names = ["bot_l.0", "bot_l.2", "bot_l.4"]
        self.output_padding = self.args.get("output_padding_granularity", 32)
        self.top_mlp_input_size = (self.num_interactions + self.embedding_size + self.output_padding - 1) // self.output_padding * self.output_padding
        self.top_mlp_channels = self.dlrm_config["top_mlp_sizes"]
        self.top_mlp_names = ["top_l.0", "top_l.2", "top_l.4", "top_l.6", "top_l.8"]
        self.model_filepath = "build/models/dlrm/tb00_40M.pt"
        self.embedding_weights_binary_filepath = "build/models/dlrm/40m_limit/dlrm_embedding_weights_int8_v3.bin"
        self.model_without_embedding_weights_filepath = "build/models/dlrm/40m_limit/model_test_without_embedding_weights_v3.pt"
        self.row_frequencies_binary_filepath = "build/models/dlrm/40m_limit/row_frequencies.bin"
        self.row_frequencies_src_dir = "build/models/dlrm/40m_limit/row_freq"
        self.embedding_weights_on_gpu_part = self.args.get("embedding_weights_on_gpu_part", 1.0)
        self.use_row_frequencies = True if self.embedding_weights_on_gpu_part < 1.0 else False
        self.num_profiles = self.args.get("gpu_inference_streams", 1)
        self.use_small_tile_gemm_plugin = self.args.get("use_small_tile_gemm_plugin", False)
        self.gemm_plugin_fairshare_cache_size = self.args.get("gemm_plugin_fairshare_cache_size", -1)
        self.enable_interleaved_top_mlp = self.args.get("enable_interleaved_top_mlp", False)

        if self.precision == "fp16":
            self.apply_flag(trt.BuilderFlag.FP16)
        elif self.precision == "int8":
            self.apply_flag(trt.BuilderFlag.INT8)

        if self.precision == "int8":
            # Get calibrator variables
            calib_batch_size = dict_get(self.args, "calib_batch_size", default=512)
            calib_max_batches = dict_get(self.args, "calib_max_batches", default=500)
            force_calibration = dict_get(self.args, "force_calibration", default=False)
            cache_file = dict_get(self.args, "cache_file", default="code/dlrm/tensorrt/calibrator.cache")
            preprocessed_data_dir = dict_get(self.args, "preprocessed_data_dir", default="build/preprocessed_data")
            calib_data_dir = os.path.join(preprocessed_data_dir, "criteo/full_recalib/val_data_128000")

            # Set up calibrator
            self.calibrator = DLRMCalibrator(calib_batch_size=calib_batch_size, calib_max_batches=calib_max_batches,
                                             force_calibration=force_calibration, cache_file=cache_file, data_dir=calib_data_dir)
            self.builder_config.int8_calibrator = self.calibrator
            self.cache_file = cache_file
            self.need_calibration = force_calibration or not os.path.exists(cache_file)
        else:
            self.need_calibration = False

    def calibrate(self):
        """
        Generate a new calibration cache, overriding the input batch size to 2 needed for interleaving
        """

        self.need_calibration = True
        self.calibrator.clear_cache()
        self.initialize()

        # Generate a dummy engine to generate a new calibration cache.
        for input_idx in range(self.network.num_inputs):
            input_shape = self.network.get_input(input_idx).shape
            input_shape[0] = 2  # need even-numbered batch size for interleaving
            self.network.get_input(input_idx).shape = input_shape
        self.builder.build_engine(self.network, self.builder_config)

    def parse_calibration(self):
        """Parse calibration file to get dynamic range of all network tensors.
        Returns the tensor:range dict.
        """

        if not os.path.exists(self.cache_file):
            return

        with open(self.cache_file, "rb") as f:
            lines = f.read().decode('ascii').splitlines()

        calibration_dict = {}
        np127 = np.float32(127.0)
        for line in lines:
            split = line.split(':')
            if len(split) != 2:
                continue
            tensor = split[0]
            dynamic_range = np.uint32(int(split[1], 16)).view(np.dtype('float32')).item() * np127
            calibration_dict[tensor] = dynamic_range

        return calibration_dict

    def add_mlp(self, input_tensor, input_size, num_channels, names, last_relu=False, useConvForFC=False):
        """Add bottom/top MLP part of DLRM network. Return the last FC layer in MLP.

        Args:
            input_tensor (ITensor): Input to MLP.
            input_size (int): Number of numerical features.
            num_channels (list): List of number of channels for each FC layer in MLP.
            names (list): List of names of each FC layer in MLP.
            last_relu (bool): Whether last FC layer in MLP will have ReLU. Rest of FC have ReLU by default.
            useConvForFC (bool): Whether to use 1x1 Conv to implement FC (for better perf).
        """

        for i, num_channel in enumerate(num_channels):
            # use add_single_mlp subroutine
            add_relu = (i != len(num_channels) - 1) or last_relu
            layer = self.add_single_mlp(input_tensor, input_size, num_channel, names[i],
                                   useConvForFC, add_relu)

            input_size = num_channel
            input_tensor = layer.get_output(0)

        return layer

    def add_single_mlp(self, input_tensor, input_size, num_channels, name, useConvForFC=False, add_relu=False):
        """
        Add a single layer of mlp.

        Args:
            input_tensor (ITensor): Input to MLP.
            input_size (int): Number of numerical features (C).
            num_channels (int): Number of channels for each FC layer in MLP (K).
            name (str): Name of the FC layer in MLP.
            useConvForFC (bool): Whether to use 1x1 Conv to implement FC (for better perf).
        """

        weights = self.weights[name + ".weight"].numpy()
        input_size_suggested_by_weights = weights.shape[1]
        if input_size > input_size_suggested_by_weights:
            weights = np.concatenate((weights, np.zeros((weights.shape[0], input_size - input_size_suggested_by_weights), dtype=weights.dtype)), 1)

        if useConvForFC:
            layer = self.network.add_convolution(input_tensor, num_channels, (1, 1),
                                                 weights, self.weights[name + ".bias"].numpy())
        else:
            layer = self.network.add_fully_connected(input_tensor, num_channels,
                                                     weights, self.weights[name + ".bias"].numpy())

        layer.name = name
        layer.get_output(0).name = name + ".output"

        if add_relu:
            layer = self.network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
            layer.name = name + ".relu"
            layer.get_output(0).name = name + ".relu.output"

        return layer

    def add_small_tile_gemm_top_mlp(self, input_tensor, input_channels,
                                    output_channels, layer_name, fairshare_cache_size=-1):
        """
        Use smallTileGEMMPlugin for top_mlp layer 1, 2 and 3

        Args:
            input_tensor (ITensor): Input to the GEMM plugin.
            input_channels (int): Number of input channels (C).
            output_channels (int): Number of output channels (K).
            layer_name (str): Name of the top mlp layer (e.g. "top_l.2")

        """
        plugin_name = "SmallTileGEMM_TRT"
        plugin_layer_name = layer_name + plugin_name
        plugin_version = '1'
        plugin_creator = trt.get_plugin_registry().\
            get_plugin_creator(plugin_name, plugin_version, '')
        if plugin_creator is None:
            raise Exception("Cannot find small tile GEMM plugin creator for top_mlp")

        weight = self.weights[layer_name + ".weight"].numpy()
        # Pad the weight if input size suggests a larger weight
        input_size_suggested_by_weights = weight.shape[1]
        if input_channels > input_size_suggested_by_weights:
            weight = np.concatenate((weight, np.zeros((weight.shape[0],
                                                       input_channels - input_size_suggested_by_weights), dtype=weight.dtype)), 1)

        bias = self.weights[layer_name + ".bias"].numpy()
        # Create a scale vector and pass to the plugin.
        scale = np.ones([output_channels], dtype=np.float32)

        # Get dynamic ranges of the input and output
        dynamic_range_dict = self.parse_calibration()
        input_tensor_name = input_tensor.name
        input_dr = dynamic_range_dict[input_tensor_name]
        output_tensor_name = layer_name + ".relu.output"
        output_dr = dynamic_range_dict[output_tensor_name]

        # Append the attributes to the plugin field collection
        fields = []
        fields.append(trt.PluginField("inputChannels", np.array([input_channels],
                                                                dtype=np.int32), trt.PluginFieldType.INT32))
        fields.append(trt.PluginField("weight", weight, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("bias", bias, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("scale", scale, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("dynamicRanges", np.array([input_dr, output_dr],
                                                                dtype=np.float32), trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("epilogueScaleBiasRelu", np.array([1],
                                                                        dtype=np.int32), trt.PluginFieldType.INT32))
        fields.append(trt.PluginField("fairShareCacheSize", np.array([fairshare_cache_size],
                                                                     dtype=np.int32), trt.PluginFieldType.INT32))
        fields = trt.PluginFieldCollection(fields)

        plugin = plugin_creator.create_plugin(plugin_layer_name, fields)
        if plugin is None:
            raise Exception("Cannot create DLRM Small-Tile GEMM plugin for top mlp.")

        plugin_layer = self.network.add_plugin_v2([input_tensor], plugin)
        plugin_layer.get_output(0).name = output_tensor_name

        return plugin_layer

    def add_fused_bottom_mlp(self, plugin_name, input_tensor, input_size, num_channels, names):
        """Add the MLP part of DLRM network as a fused plugin for better perf. Return the last FC layer in MLP.

        Args:
            plugin_name (str): Name of fused MLP plugin to use.
            input_tensor (ITensor): Input to MLP.
            input_size (int): Number of numerical features.
            num_channels (list): List of number of channels for each FC layer in MLP.
            names (list): List of names of each FC layer in MLP.
        """

        plugin = None
        output_tensor_name = ""
        dynamic_range_dict = self.parse_calibration()
        for plugin_creator in trt.get_plugin_registry().plugin_creator_list:
            if plugin_creator.name == plugin_name:
                plugin_fields = []
                plugin_fields.append(trt.PluginField("inputChannels", np.array([input_size], dtype=np.int32), trt.PluginFieldType.INT32))
                for i, _ in enumerate(num_channels):
                    weights = self.weights[names[i] + ".weight"].numpy()
                    input_size_suggested_by_weights = weights.shape[1]
                    if input_size > input_size_suggested_by_weights:
                        weights = np.concatenate((weights, np.zeros((weights.shape[0], input_size - input_size_suggested_by_weights), dtype=weights.dtype)), 1)
                    plugin_fields.append(trt.PluginField("weights" + str(i), weights, trt.PluginFieldType.FLOAT32))
                    plugin_fields.append(trt.PluginField("biases" + str(i), self.weights[names[i] + ".bias"].numpy(), trt.PluginFieldType.FLOAT32))
                    output_tensor_name = names[i] + ".relu.output"
                    if i != len(num_channels) - 1:
                        plugin_fields.append(trt.PluginField("dynamicRange" + str(i), np.array([dynamic_range_dict[output_tensor_name]], dtype=np.float32), trt.PluginFieldType.FLOAT32))
                plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=trt.PluginFieldCollection(plugin_fields))
        return plugin, output_tensor_name

    def get_dlrm_interactions_plugin(self, plugin_name, tableOffsets, interactionsOutputInterleaved):
        """Create a plugin layer for the DLRM Interactions plugin and return it.

        DLRM Interactions plugin takes two inputs: from bottom MLP and categorical input and looks up their embeddings.
        Since DLRM embeddings can be larger than GPU memory, the plugin keeps the most frequently used embeddings on GPU
        and rest on host and manages the lookup with good performance.
        """

        plugin = None
        for plugin_creator in trt.get_plugin_registry().plugin_creator_list:
            if plugin_creator.name == plugin_name:
                embeddingSize_field = trt.PluginField("embeddingSize", np.array([self.embedding_size], dtype=np.int32), trt.PluginFieldType.INT32)
                embeddingRows_field = trt.PluginField("embeddingRows", np.array([self.embedding_rows_total], dtype=np.int32), trt.PluginFieldType.INT32)
                reducedPrecisionIO_field = trt.PluginField("reducedPrecisionIO", np.array(
                    [0 if self.need_calibration else (1 if self.precision == "fp16" else 2)], dtype=np.int32), trt.PluginFieldType.INT32)
                embeddingWeightsOnGpuPart_field = trt.PluginField("embeddingWeightsOnGpuPart", np.array([self.embedding_weights_on_gpu_part], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                interactionsOutputInterleaved_field = trt.PluginField("interactionsOutputInterleaved", np.array([1 if interactionsOutputInterleaved else 0], dtype=np.int32), trt.PluginFieldType.INT32)
                tableOffsets_field = trt.PluginField("tableOffsets", tableOffsets, trt.PluginFieldType.INT32)
                embeddingWeightsFilepath_field = trt.PluginField("embeddingWeightsFilepath", np.array(list(self.embedding_weights_binary_filepath.encode()), dtype=np.int8), trt.PluginFieldType.CHAR)
                if self.use_row_frequencies:
                    rowFrequenciesFilepath_field = trt.PluginField("rowFrequenciesFilepath", np.array(list(self.row_frequencies_binary_filepath.encode()), dtype=np.int8), trt.PluginFieldType.CHAR)
                else:
                    rowFrequenciesFilepath_field = trt.PluginField("rowFrequenciesFilepath", np.array(list("".encode()), dtype=np.int8), trt.PluginFieldType.CHAR)

                output_padding_field = trt.PluginField("outputPaddingGranularity", np.array([self.output_padding], dtype=np.int32), trt.PluginFieldType.INT32)

                field_collection = trt.PluginFieldCollection([embeddingSize_field, embeddingRows_field, reducedPrecisionIO_field, embeddingWeightsOnGpuPart_field,
                                                              interactionsOutputInterleaved_field, output_padding_field, tableOffsets_field, embeddingWeightsFilepath_field, rowFrequenciesFilepath_field])
                plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
        return plugin

    def dump_embedding_weights_to_binary_file(self):
        """Quantize embedding weights and write to binary file."""

        logging.info("Writing quantized embedding weights to " + self.embedding_weights_binary_filepath)

        with open(self.embedding_weights_binary_filepath, 'wb') as f:

            # Write number of tables
            f.write(struct.pack('i', self.num_features))

            # For each table, calculate max abs value of embedding weights and write it
            mults = np.ndarray(shape=(self.num_features))
            for feature_id in range(self.num_features):
                weight_tensor_name = "emb_l." + str(feature_id) + ".weight"
                embeddings = self.weights[weight_tensor_name].numpy()
                maxAbsVal = abs(max(embeddings.max(), embeddings.min(), key=abs))
                mults[feature_id] = 127.5 / maxAbsVal
                embeddingsScale = 1.0 / mults[feature_id]
                f.write(struct.pack('f', embeddingsScale))

            for feature_id in range(self.num_features):
                weight_tensor_name = "emb_l." + str(feature_id) + ".weight"
                embeddings = self.weights[weight_tensor_name].numpy()
                if (embeddings.shape[0] != self.embedding_rows[feature_id]):
                    raise IOError("Expected " + str(self.embedding_rows[feature_id]) + " embedding rows, but got " + str(embeddings.shape[0]) + " rows for feature " + str(feature_id))

                # Scale and bind to [-127, 127]
                embeddingsQuantized = np.minimum(np.maximum(np.rint(np.multiply(embeddings, mults[feature_id])), -127), 127).astype('int8')

                # Remove the embedding weights, we don't need them any longer
                del self.weights[weight_tensor_name]

                # Write quantized embeddings to file
                embeddingsQuantized.tofile(f)

    def dump_row_frequencies_to_binary_file(self):
        """Dump row frequencies from CSV to binary file."""

        with open(self.row_frequencies_binary_filepath, 'wb') as f:
            f.write(struct.pack('i', self.num_features))
            for feature_id in range(self.num_features):
                f.write(struct.pack('i', self.embedding_rows[feature_id]))
                row_frequencies_source_filepath = self.row_frequencies_src_dir + "/" + "table_" + str(feature_id + 1) + ".csv"
                with open(row_frequencies_source_filepath, 'r') as infile:
                    reader = csv.reader(infile)
                    rowIdToFreqDict = {rows[0]: rows[1] for rows in reader}
                    # if (len(rowIdToFreqDict) != self.embedding_rows[feature_id]):
                    #    raise IOError("Expected " + str(self.embedding_rows[feature_id]) + " embedding rows, but got " + str(len(rowIdToFreqDict)) + " row frequencies for feature " + str(feature_id))
                    for row_id in range(self.embedding_rows[feature_id]):
                        if not str(row_id) in rowIdToFreqDict:
                            f.write(struct.pack('f', 0))
                            # raise IOError("Cannot find frequency for row " + str(row_id) + " for feature " + str(feature_id))
                        else:
                            f.write(struct.pack('f', float(rowIdToFreqDict[str(row_id)])))

    def initialize(self):
        """Create DLRM network using TRT API and plugins and set the weights."""

        useConvForFC_bottom = (self.precision == "int8")
        useConvForFC_top = (self.precision == "int8")
        interactionsOutputInterleaved = False if self.need_calibration or self.input_dtype != "int8" else True

        # Turn off interleaved format if top_mlp use non-interleaved format
        if not self.enable_interleaved_top_mlp:
            interactionsOutputInterleaved = False
        else:
            print("Using batch-interleaved format for top_mlp.")

        # Check if we should split the model into the binary file with embedding weights quantized and model without embeddings
        if not (os.path.isfile(self.embedding_weights_binary_filepath) and os.path.isfile(self.model_without_embedding_weights_filepath)):
            logging.info("Loading checkpoint from " + self.model_filepath)
            self.weights = torch.load(self.model_filepath, map_location="cpu")["state_dict"]
            self.dump_embedding_weights_to_binary_file()
            logging.info("Writing model without embedding weights to " + self.model_without_embedding_weights_filepath)
            torch.save(self.weights, self.model_without_embedding_weights_filepath)
            del self.weights

        # Dump row frequencies to file in binary format
        if self.use_row_frequencies and not os.path.isfile(self.row_frequencies_binary_filepath):
            logging.info("Writing row frequencies to " + self.row_frequencies_binary_filepath)
            self.dump_row_frequencies_to_binary_file()

        # Load weights
        self.weights = torch.load(self.model_without_embedding_weights_filepath, map_location="cpu")

        # Create network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Numerical input
        numerical_input = self.network.add_input("numerical_input", trt.DataType.FLOAT, (-1, self.num_numerical_inputs, 1, 1))
        if not self.need_calibration:
            if self.input_dtype == "int8":
                numerical_input.dtype = trt.int8
            elif self.input_dtype == "fp16":
                numerical_input.dtype = trt.float16
            if self.input_format == "linear":
                numerical_input.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
            elif self.input_format == "chw4":
                numerical_input.allowed_formats = 1 << int(trt.TensorFormat.CHW4)
            elif self.input_format == "chw32":
                numerical_input.allowed_formats = 1 << int(trt.TensorFormat.CHW32)

        # Bottom MLP
        if self.need_calibration or self.input_dtype != "int8":
            bottom_mlp = self.add_mlp(numerical_input, self.num_numerical_inputs, self.bottom_mlp_channels, self.bottom_mlp_names,
                                      last_relu=True, useConvForFC=useConvForFC_bottom)
        else:
            bottom_mlp_plugin, output_tensor_name = self.add_fused_bottom_mlp("DLRM_BOTTOM_MLP_TRT", numerical_input, self.num_numerical_inputs, self.bottom_mlp_channels, self.bottom_mlp_names)
            bottom_mlp = self.network.add_plugin_v2([numerical_input], bottom_mlp_plugin)
            bottom_mlp.get_output(0).name = output_tensor_name
        bottom_mlp_shuffle = self.network.add_shuffle(bottom_mlp.get_output(0))
        bottom_mlp_shuffle.reshape_dims = trt.Dims((-1, 1, self.embedding_size))

        # Index input
        index_input = self.network.add_input("index_input", trt.DataType.INT32, (-1, self.num_features))

        # Embedding lookup and interactions
        dlrm_interactions_plugin = self.get_dlrm_interactions_plugin("DLRM_INTERACTIONS_TRT", np.cumsum(
            np.array([0] + self.embedding_rows[:-1]).astype(np.int32)).astype(np.int32), interactionsOutputInterleaved)
        interaction_output_concat = self.network.add_plugin_v2([bottom_mlp.get_output(0), index_input], dlrm_interactions_plugin)
        interaction_output_concat.name = "interaction_plugin"
        interaction_output_concat.get_output(0).name = "interaction_output_concat_output"

        if self.enable_interleaved_top_mlp and not interactionsOutputInterleaved:
            # Shuffle from [BS, C, 1, 1] to [BS//2, C, 2, 1] before top_mlp
            interleave_pre_top_mlp = self.network.add_shuffle(interaction_output_concat.get_output(0))
            interleave_pre_top_mlp.reshape_dims = trt.Dims((-1, 2, interaction_output_concat.get_output(0).shape[1], 0))
            interleave_pre_top_mlp.second_transpose = trt.Permutation([0, 2, 1, 3])
            interleave_pre_top_mlp.name = "interleave_pre_top_mlp"

            top_mlp_input = interleave_pre_top_mlp.get_output(0)
            top_mlp_input.name = "interleave_pre_top_mlp"
        else:
            top_mlp_input = interaction_output_concat.get_output(0)

        # Insert small-tile GEMM plugin. The plugin supports Ampere-only.
        gpu_arch = get_system().arch
        system_id = get_system().gpu
        if self.use_small_tile_gemm_plugin:
            if gpu_arch != Architecture.Ampere:
                print("Small-Tile GEMM plugin does not support {}. Plugin disabled.".format(system_id))
                self.use_small_tile_gemm_plugin = False

        # Enable gemm plugin with interleaved format is not recommended.
        # Note (2/7/21): GEMM plugin doesn't perform well when H*W > 1
        if self.use_small_tile_gemm_plugin and self.enable_interleaved_top_mlp:
            print("Warning: small-Tile GEMM plugin performance will be "
                  "significantly impacted by interleaved format. Turn off "
                  "interleaved format for the best performance")

        tmp_mlp_input = top_mlp_input
        tmp_input_size = self.top_mlp_input_size

        # Helper function to check whether the provided shape is supported by
        # Small-Tile GEMM plugin
        def support_small_tile_gemm_func(C, K): return \
            (C >= 256) and (C <= 1280) and (C % 128 == 0) and (K % 128 == 0)

        # Split the top_mlp layers, and use GEMM plugin for 2,4,6
        # C, K for top_mlp.0,2,4,6,8: [480,1024],[1024,1024],[1024,512],[512,256],[256,1]
        for i in range(len(self.top_mlp_channels)):
            # Insert plugin if the layer meets the restriction
            if support_small_tile_gemm_func(tmp_input_size, self.top_mlp_channels[i]) and \
                    self.use_small_tile_gemm_plugin:
                print("Replacing {} with Small-Tile GEMM Plugin, with fairshare cache size {}".
                      format(self.top_mlp_names[i], self.gemm_plugin_fairshare_cache_size))
                layer_top_mlp = self.add_small_tile_gemm_top_mlp(
                    tmp_mlp_input, tmp_input_size,
                    self.top_mlp_channels[i], self.top_mlp_names[i],
                    self.gemm_plugin_fairshare_cache_size
                )
            else:
                layer_top_mlp = self.add_single_mlp(
                    tmp_mlp_input, tmp_input_size,
                    self.top_mlp_channels[i], self.top_mlp_names[i],
                    useConvForFC=useConvForFC_top,
                    add_relu=(i != len(self.top_mlp_channels) - 1))

            tmp_mlp_input = layer_top_mlp.get_output(0)
            tmp_input_size = self.top_mlp_channels[i]

        top_mlp = layer_top_mlp

        if self.enable_interleaved_top_mlp:
            # Shuffle [BS//2, 1, 2, 1] back to [BS, 1, 1, 1]
            interleave_post_top_mlp = self.network.add_shuffle(top_mlp.get_output(0))
            interleave_post_top_mlp.reshape_dims = trt.Dims((-1, 0, 1, 0))
            interleave_post_top_mlp.name = "interleave_post_top_mlp"

            sigmoid_input = interleave_post_top_mlp.get_output(0)
            sigmoid_input.name = "interleave_post_top_mlp"
        else:
            sigmoid_input = top_mlp.get_output(0)

        # Sigmoid
        sigmoid_layer = self.network.add_activation(sigmoid_input, trt.ActivationType.SIGMOID)
        sigmoid_layer.name = "sigmoid"
        sigmoid_layer.get_output(0).name = "sigmoid_output"

        # Output
        self.network.mark_output(sigmoid_layer.get_output(0))

        # Make sure we release the memory to system
        del self.weights

        self.initialized = True
