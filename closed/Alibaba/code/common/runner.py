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

import ctypes
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pytest
import tensorrt as trt
import time

from code.common import logging
from glob import glob


class HostDeviceMem(object):
    def __init__(self, host, device):
        self.host = host
        self.device = device


def allocate_buffers(engine, profile_id):
    """Allocate device memory for I/O bindings of engine and return them."""

    d_inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    if engine.has_implicit_batch_dimension:
        max_batch_size = engine.max_batch_size
    else:
        shape = engine.get_binding_shape(0)
        if -1 in list(shape):
            batch_dim = list(shape).index(-1)
            max_batch_size = engine.get_profile_shape(0, 0)[2][batch_dim]
        else:
            max_batch_size = shape[0]
    nb_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    bindings = [0 for i in range(engine.num_bindings)]
    for binding in range(profile_id * nb_bindings_per_profile, (profile_id + 1) * nb_bindings_per_profile):
        logging.info("Binding {:}".format(binding))
        dtype = engine.get_binding_dtype(binding)
        format = engine.get_binding_format(binding)
        shape = engine.get_binding_shape(binding)
        if format == trt.TensorFormat.CHW4:
            shape[-3] = ((shape[-3] - 1) // 4 + 1) * 4
        elif format == trt.TensorFormat.DHWC8:
            shape[-4] = ((shape[-4] - 1) // 8 + 1) * 8
        if not engine.has_implicit_batch_dimension:
            if -1 in list(shape):
                batch_dim = list(shape).index(-1)
                shape[batch_dim] = max_batch_size
            size = trt.volume(shape)
        else:
            size = trt.volume(shape) * max_batch_size
        # Allocate device buffers
        device_mem = cuda.mem_alloc(size * dtype.itemsize)
        # Append device buffer to device bindings.
        bindings[binding] = int(device_mem)
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            d_inputs.append(device_mem)
        else:
            host_mem = cuda.pagelocked_empty(size, trt.nptype(dtype))
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return d_inputs, outputs, bindings, stream


def get_input_format(engine):
    return engine.get_binding_dtype(0), engine.get_binding_format(0)


class EngineRunner:
    """Enable running inference through an engine on each call."""

    def __init__(self, engine_file, verbose=False, plugins=None, profile_id=0):
        """Load engine from file, allocate device memory for its bindings and create execution context."""

        self.engine_file = engine_file
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        if not os.path.exists(engine_file):
            raise ValueError("File {:} does not exist".format(engine_file))

        trt.init_libnvinfer_plugins(self.logger, "")
        if plugins is not None:
            for plugin in plugins:
                ctypes.CDLL(plugin)
        self.engine = self.load_engine(engine_file)

        if profile_id < 0:
            profile_id = self.engine.num_optimization_profiles + profile_id

        self.d_inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, profile_id)
        self.context = self.engine.create_execution_context()

        if profile_id > 0:
            self.context.active_optimization_profile = profile_id

    def load_engine(self, src_path):
        """Deserialize engine file to an engine and return it."""

        with open(src_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        return engine

    def __call__(self, inputs, batch_size=1):
        """Use host inputs to run inference on device and return back results to host."""

        # Copy input data to device bindings of context.
        profile_id = self.context.active_optimization_profile
        nb_bindings_per_profile = self.engine.num_bindings // self.engine.num_optimization_profiles
        [cuda.memcpy_htod_async(d_input, inp, self.stream) for (d_input, inp) in zip(self.d_inputs, inputs)]

        # Run inference.
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            for binding_idx in range(profile_id * nb_bindings_per_profile, (profile_id + 1) * nb_bindings_per_profile):
                if self.engine.binding_is_input(binding_idx):
                    input_shape = self.context.get_binding_shape(binding_idx)
                    if -1 in list(input_shape):
                        input_shape[0] = batch_size
                        self.context.set_binding_shape(binding_idx, input_shape)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output device buffers back to host.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

        # Synchronize the stream
        self.stream.synchronize()

        # Return only the host outputs.
        return [out.host for out in self.outputs]

    def __del__(self):
        # Clean up everything.
        with self.engine, self.context:
            [d_input.free() for d_input in self.d_inputs]
            [out.device.free() for out in self.outputs]
            del self.stream
