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
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os
import sys
sys.path.insert(0, os.getcwd())

from code.common import logging


class UNet3DLegacyCalibrator(trt.IInt8LegacyCalibrator):
    def __init__(self, data_dir, cache_file, batch_size,
                 max_batches, force_calibration, calib_data_map, input_shape):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8LegacyCalibrator.__init__(self)

        self.cache_file = cache_file
        self.max_batches = max_batches

        vol_list = []
        with open(calib_data_map) as f:
            for line in f:
                vol_list.append(line.strip())

        self.shape = tuple([batch_size] + input_shape)
        # * 4 is for sizeof(FP32)
        self.device_input = cuda.mem_alloc(trt.volume(self.shape) * 4)

        self.brats_id = 0
        self.force_calibration = force_calibration

        def load_batches():
            """
            Create a generator that will give us batches. We can use next()
            to iterate over the result.
            """
            batch_id = 0
            batch_size = self.shape[0]
            batch_data = np.zeros(shape=self.shape, dtype=np.float32)
            while self.brats_id < len(vol_list) and batch_id < self.max_batches:
                print("Calibrating with batch {}".format(batch_id))
                batch_id += 1
                end_brats_id = min(self.brats_id + batch_size, len(vol_list))

                for i in range(self.brats_id, end_brats_id):
                    batch_data[i - self.brats_id] = np.load(os.path.join(data_dir, vol_list[i] + ".npy"))

                self.brats_id = end_brats_id

                shape = self.shape
                data = batch_data.tobytes()
                labels = bytes(b'')
                yield data

        self.batches = load_batches()

        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if not self.force_calibration and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.cache = f.read()
        else:
            self.cache = None

    def get_batch_size(self):
        return self.shape[0]

    def get_batch(self, names):
        """
        Acquire a single batch 

        Arguments:
        names (string): names of the engine bindings from TensorRT. Useful to understand the order of inputs.
        """
        try:
            data = next(self.batches)
            # Copy to device, then return a list containing pointers to input device buffers.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        return self.cache

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def clear_cache(self):
        self.cache = None

    def get_quantile(self):
        """ returning 99.999% """
        return 0.99999

    def get_regression_cutoff(self):
        return 1.0

    def read_histogram_cache(self, arg0):
        return None

    def write_histogram_cache(self, arg0, arg1):
        return None
