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

import numpy as np
import onnx
import tensorrt as trt
import json


class BertConfig:
    def __init__(self, config_path):

        with open(config_path, 'r') as f:
            cfg = json.load(f)
            self.hidden_size = cfg["hidden_size"]
            self.mid_size = cfg['intermediate_size']
            self.N = cfg['num_attention_heads']
            self.L = cfg['num_hidden_layers']

            self.H = self.hidden_size // self.N
            assert(self.H * self.N == self.hidden_size)
            self.qkv_size = 3 * self.hidden_size


def onnx_to_tf_name(onnx_name):
    """
    Converting variables in the onnx checkpoint to names corresponding to the naming convention used in the TF version, expected by the builder
    """
    onnx_name = onnx_name.lower()
    toks = [t.strip('_') for t in onnx_name.split('.')]
    if toks[0] == 'bert':  # embeddings or encoder
        if toks[1] == 'encoder':  # transformer

            if toks[-2] == 'layernorm':  # bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            elif (toks[-2] == 'dense' or toks[-2] in {'key', 'value', 'query'}) and toks[-1] == 'weight':
                toks[-1] = 'kernel'
            elif (toks[-3] == 'dense' or toks[-3] in {'key', 'value', 'query'}) and toks[-1] == 'amax':
                if toks[-2] == 'weight_quantizer':
                    toks[-2] = 'kernel'
                elif toks[-2] == 'input_quantizer':
                    toks[-2] = 'input'

            if 'final_input_quantizer' not in toks[2]:
                toks = toks[3:]
                toks[0] = 'l{}'.format(int(toks[0]))
        else:
            if toks[-2] == 'layernorm':  # bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            else:  # embeddings: drop "_weight" suffix
                if toks[-1] == 'amax':
                    toks[-2] = 'amax'
                toks = toks[:-1]
    elif 'qa' in onnx_name:
        name = 'cls_squad_output_bias' if toks[-1] == 'bias' else 'cls_squad_output_weights'
        return name
    else:
        print("Encountered unknown case:", onnx_name)
        assert(False)
    parsed = '_'.join(toks)
    return parsed


def get_onnx_fake_quant_weights(path):
    """Return weights from ONNX model file."""
    model = onnx.load(path)
    weights = model.graph.initializer
    weights_dict = dict([(onnx_to_tf_name(w.name), np.frombuffer(w.raw_data, np.float32).reshape(w.dims)) for w in weights])
    return weights_dict


def mark(network, tensor, dtype):
    """Set input dtype on tensor and mark it as an output tensor."""
    tensor.dtype = dtype
    network.mark_output(tensor)


def add_gelu(network, input_tensor):
    """This will trigger FC+GELU fusion in TRT"""
    shape = (1, ) * len(input_tensor.shape)
    POW = network.add_constant(shape, trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
    MULTIPLY = network.add_constant(shape, trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
    SQRT = network.add_constant(shape, trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
    ONE = network.add_constant(shape, trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
    HALF = network.add_constant(shape, trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
    X_pow = network.add_elementwise(input_tensor, POW.get_output(0), trt.ElementWiseOperation.POW)
    X_pow_t = X_pow.get_output(0)
    X_mul = network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
    X_add = network.add_elementwise(input_tensor, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
    X_sqrt = network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
    X_sqrt_tensor = X_sqrt.get_output(0)
    X_tanh = network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
    X_tanh_tensor = X_tanh.get_output(0)
    X_one = network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
    CDF = network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
    gelu_layer = network.add_elementwise(CDF.get_output(0), input_tensor, trt.ElementWiseOperation.PROD)

    # enable elementwise fusing for int8 && fp16
    POW.precision = trt.DataType.FLOAT
    MULTIPLY.precision = trt.DataType.FLOAT
    SQRT.precision = trt.DataType.FLOAT
    ONE.precision = trt.DataType.FLOAT
    HALF.precision = trt.DataType.FLOAT
    X_pow.precision = trt.DataType.FLOAT
    X_mul.precision = trt.DataType.FLOAT
    X_add.precision = trt.DataType.FLOAT
    X_sqrt.precision = trt.DataType.FLOAT
    X_tanh.precision = trt.DataType.FLOAT
    X_one.precision = trt.DataType.FLOAT
    CDF.precision = trt.DataType.FLOAT
    gelu_layer.precision = trt.DataType.FLOAT
    return gelu_layer
