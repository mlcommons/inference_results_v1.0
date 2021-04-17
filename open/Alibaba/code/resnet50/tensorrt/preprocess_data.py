#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to preprocess data for Resnet50 benchmark."""

import argparse
import cv2
import math
import numpy as np
import os
import shutil
import struct
import sys
sys.path.insert(0, os.getcwd())

from code.common import logging, BENCHMARKS
from code.common.image_preprocessor import ImagePreprocessor
from PIL import Image
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
def center_crop(img, output_size):
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))

    def crop(img, top, left, height, width):
        return img.crop((left, top, left + width, top + height))

    return crop(img, crop_top, crop_left, crop_height, crop_width)

def preprocess_imagenet_for_resnet50(data_dir, preprocessed_data_dir, formats, overwrite=False, cal_only=False, val_only=False):
    """Proprocess the raw images for inference."""

    def loader(file):
        """Resize and crop image to required dims and return as FP32 array."""

        img = Image.open(file)
        img = img.convert('RGB')
        img = resize_with_aspectratio(img, 202)
        img = center_crop(img, (176, 176))

        img = np.asarray(img, dtype='float32')
        img /= 255.0
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std = np.array([0.229,0.224,0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose([2, 0, 1])
        #img = np.asarray(img.reshape((3,224,224)), dtype='float32')
        return img

    def quantizer(image):
        """Return quantized INT8 image of input FP32 image."""
        scale = struct.unpack('!f', bytes.fromhex('3caa5293'))[0]
        image_int8 = (image/scale).clip(-127.0, 127.0)
        return image_int8.astype(dtype=np.int8, order='C')

    preprocessor = ImagePreprocessor(loader, quantizer)
    if not val_only:
        # Preprocess calibration set. FP32 only because calibrator always takes FP32 input.
        preprocessor.run(os.path.join(data_dir, "imagenet"), os.path.join(preprocessed_data_dir, "imagenet", "ResNet50"),
                         "data_maps/imagenet/cal_map.txt", ["fp32"], overwrite)
    if not cal_only:
        # Preprocess validation set.
        preprocessor.run(os.path.join(data_dir, "imagenet"), os.path.join(preprocessed_data_dir, "imagenet", "ResNet50"),
                         "data_maps/imagenet/val_map.txt", formats, overwrite)


def main():
    """
    Parse arguments to identify the data directory with the input images
      and the output directory for the preprocessed images.
    The data directory is assumed to have the following structure:
    <data_dir>
     └── imagenet
    And the output directory will have the following structure:
    <preprocessed_data_dir>
     └── imagenet
         └── ResNet50
             ├── fp32
             └── int8_linear
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-d",
        help="Directory containing the input images.",
        default="build/data"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Output directory for the preprocessed data.",
        default="build/preprocessed_data"
    )
    parser.add_argument(
        "--formats", "-t",
        help="Comma-separated list of formats. Choices: fp32, int8_linear, int8_chw4.",
        default="default"
    )
    parser.add_argument(
        "--overwrite", "-f",
        help="Overwrite existing files.",
        action="store_true"
    )
    parser.add_argument(
        "--cal_only",
        help="Only preprocess calibration set.",
        action="store_true"
    )
    parser.add_argument(
        "--val_only",
        help="Only preprocess validation set.",
        action="store_true"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    preprocessed_data_dir = args.preprocessed_data_dir
    formats = args.formats.split(",")
    overwrite = args.overwrite
    cal_only = args.cal_only
    val_only = args.val_only
    default_formats = ["int8_chw4"]

    # Now, actually preprocess the input images
    logging.info("Loading and preprocessing images. This might take a while...")
    if args.formats == "default":
        formats = default_formats
    preprocess_imagenet_for_resnet50(data_dir, preprocessed_data_dir, formats, overwrite, cal_only, val_only)

    logging.info("Preprocessing done.")


if __name__ == '__main__':
    main()
