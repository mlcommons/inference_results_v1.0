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
import os

import pycuda.driver as cuda
import pycuda.autoinit
#import cv2
import numpy as np
from collections import namedtuple

# For reading size information from batches
import struct
from PIL import Image

_ModelData = namedtuple('_ModelData', ['MODEL_PATH', 'INPUT_SHAPE',  'DTYPE'])
ModelData = _ModelData(MODEL_PATH = "resnet50v1/resnet50v1.onnx",
                       INPUT_SHAPE = (1,3, 224, 224),
                       DTYPE = trt.float32  )

def center_crop(img, output_size):
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))

    def crop(img, top, left, height, width):
        return img.crop((left, top, left + width, top + height))

    return crop(img, crop_top, crop_left, crop_height, crop_width)

def resize_with_aspectratio(img, out_size, interpolation = Image.BILINEAR):
    w, h  = img.size
    size = out_size
    if (w <= h and w == size) or (h <= w and h == out_size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), interpolation)
    else:
        oh = size
        ow = int(size * w / h)

    img = img.resize((ow,oh), interpolation)
    return img

class RN50Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_batch_size=1, calib_max_batches=500, force_calibration=False,
        cache_file="code/resnet50/tensorrt/calibrator.cache",
        image_dir="build/data/",
        calib_data_map="data_maps/imagenet/cal_map.txt"
        ):

        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.calib_batch_size = calib_batch_size
        self.calib_max_batches = calib_max_batches
        self.force_calibration = force_calibration
        self.cache_file = cache_file
       
        # Get a list of all the batch files in the batch folder.
        image_lists = []
        with open(calib_data_map) as f:
            for line in f:
                image_lists.append(line.split()[0])

        _batch_files = list()
        for f in os.listdir(image_dir):
            if f.endswith('.JPEG') and f in image_lists:
                _batch_files.append(os.path.join(image_dir,f))

        self.batch_files = np.array(_batch_files)

        # Find out the shape of a batch and then allocate a device buffer of that size.
        self.shape = ModelData.INPUT_SHAPE
        # Each element of the calibration data is a float32.
        self.device_input = cuda.mem_alloc(trt.volume(self.shape) * trt.float32.itemsize)

        # Create a generator that will give us batches. We can use next() to iterate over the result.
        def load_batches():
            for f in self.batch_files:
                shape, data = self.read_batch_file(f)
                yield shape, data
        self.batches = load_batches()

    # This function is used to load calibration data from the calibration batch files.
    # In this implementation, one file corresponds to one batch, but it is also possible to use
    # aggregate data from multiple files, or use only data from portions of a file.
    def read_batch_file(self, filename):
        data = self.normalize_image(filename)
        shape = ModelData.INPUT_SHAPE
        fn = filename.split('/')[-1]
        print(shape, np.shape(data))
        return shape, data

    def get_batch_size(self):
        return self.shape[0]

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    #def get_batch(self, names, a_p):
    def get_batch(self, names):
        try:
            # Get a single batch.
            _, data = next(self.batches)
            # Copy to device, then return a list containing pointers to input device buffers.
            #cuda.memcpy_htod(self.device_input, data)
            cuda.memcpy_htod(self.device_input, np.ascontiguousarray(data))
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            print('reading calibration file')
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        if not os.path.exists(self.cache_file):
            print('writing calibration file')
            with open(self.cache_file, "wb") as f:
                f.write(cache)

    def clear_cache(self):
        self.cache = None

    def __del__(self):
        self.device_input.free()


    def normalize_image(self, img):

        img = Image.open(img)
        img = img.convert('RGB')
        img = resize_with_aspectratio(img, 256)
        img = center_crop(img, (224, 224))

        img = np.asarray(img, dtype='float32')
        img /= 255.0
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std = np.array([0.229,0.224,0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose([2, 0, 1])
        #img = np.asarray(img.reshape((3,224,224)), dtype='float32')
        return img


