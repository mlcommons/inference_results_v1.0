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


class DLRMCalibrator(trt.IInt8EntropyCalibrator2):
    """Calibrator for DLRM benchmark."""

    def __init__(self, calib_batch_size=512, calib_max_batches=500, force_calibration=False,
                 cache_file="code/dlrm/tensorrt/calibrator.cache",
                 data_dir="build/preprocessed_data/criteo/full_recalib/val_data_128000"):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.calib_batch_size = calib_batch_size
        self.calib_max_batches = calib_max_batches
        self.force_calibration = force_calibration
        self.current_idx = 0
        self.cache_file = cache_file

        num_samples = calib_batch_size * calib_max_batches
        numeric_path = os.path.join(data_dir, "numeric_fp32.npy")
        self.numeric_inputs = np.load(numeric_path)[:num_samples]
        index_path = os.path.join(data_dir, "categorical_int32.npy")
        self.index_inputs = np.load(index_path)[:num_samples]

        self.device_input_numeric = cuda.mem_alloc(self.calib_batch_size * 13 * 4)
        self.device_input_index = cuda.mem_alloc(self.calib_batch_size * 26 * 4)

        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if not self.force_calibration and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.cache = f.read()
        else:
            self.cache = None

    def get_batch_size(self):
        return self.calib_batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_idx < self.calib_max_batches:
            cuda.memcpy_htod(self.device_input_numeric, np.ascontiguousarray(self.numeric_inputs[self.current_idx:self.current_idx + self.calib_batch_size]))
            cuda.memcpy_htod(self.device_input_index, np.ascontiguousarray(self.index_inputs[self.current_idx:self.current_idx + self.calib_batch_size]))
            self.current_idx += 1
            return [int(self.device_input_numeric), int(self.device_input_index)]
        else:
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

    def __del__(self):
        self.device_input_numeric.free()
        self.device_input_index.free()
