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
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

from tqdm import tqdm


class RNNTCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, batch_size, max_batches, force, cache_path, data_map, data_dir, data_type):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.batch_size = batch_size
        self.max_batches = max_batches
        self.force = force
        self.cache_path = cache_path
        if data_type == 'fp32':
            self.ITEM_DTYPE = np.float32
        elif data_type == 'fp16':
            self.ITEM_DTYPE = np.float16
        elif data_type == 'int8':
            self.ITEM_DTYPE = np.int8
        else:
            raise NotImplementedError(f"Data type {data_type} not recognized for calibration")

        ITEMSIZE_BYTES = self.ITEM_DTYPE(0).itemsize

        with open(data_map) as f:
            # Assumes calibration data is just lines of filenames (no extension)
            data_paths = [os.path.join(data_dir, fn.strip() + ".npy") for fn in f.readlines()]

        if max_batches * batch_size < len(data_paths):
            n_samples = max_batches * batch_size
        else:
            print(f"Requested {max_batches * batch_size} samples for calibration, but only {len(data_paths)} are in the dataset. Calibrating with {len(data_paths)} samples instead")
            n_samples = len(data_paths)
        lens = []
        samples = []
        for path in tqdm(data_paths[:n_samples]):
            sample = np.load(path)
            samples.append(np.load(path))
            # Probably very inefficient way to get sequence lengths for each entry
            lens.append(np.nonzero(sample)[0][-1])

        # We should ensure that we've loaded in data that we expect as input, and warn otherwise
        if samples[0].dtype != self.ITEM_DTYPE:
            print(f"Warning: converting input data of type {samples[0].dtype} to {self.ITEM_DTYPE}. This may result in loss of calibration accuracy and increased calibration time")
        # Partitions an array into sub-arrays, each of batch_size length, with the last entry being remainder-sized
        def partition(ar): return np.array_split(ar, np.arange(len(ar))[batch_size::batch_size])

        self.batches = partition(samples)
        self.batch_lens = partition(lens)  # Type conversion happens later

        SEQ_LEN, BINS = self.batches[0].shape[1:3]  # Not sure if "BINS" is the name I'm looking for (how many floats are used for each time step)
        self.device_input = cuda.mem_alloc(self.batch_size * SEQ_LEN * BINS * ITEMSIZE_BYTES)

        INT32_SIZE = np.int32(0).itemsize
        self.device_length = cuda.mem_alloc(self.batch_size * INT32_SIZE)
        self.current_idx = 0
        # If there's a cache, use it instead of calibrating
        if not self.force and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.cache = f.read()
        else:
            self.cache = None

    def get_batch(self, names):
        if self.current_idx < len(self.batches):
            npbatch = np.ascontiguousarray(self.batches[self.current_idx], dtype=self.ITEM_DTYPE)
            cuda.memcpy_htod(self.device_input, npbatch)
            npseqlen = np.ascontiguousarray(self.batch_lens[self.current_idx], dtype=np.int32)
            cuda.memcpy_htod(self.device_length, npseqlen)
            self.current_idx += 1
            return [int(self.device_input), int(self.device_length)]
        else:
            return None

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self):
        return self.cache

    def write_calibration_cache(self, cache):
        with open(self.cache_path, 'wb') as f:
            f.write(cache)
        self.cache = cache

    def clear_cache(self):
        self.cache = None

    def __del__(self):
        self.device_input.free()
        self.device_length.free()
