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

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
import numpy as np
import shutil

from code.common import logging, BENCHMARKS
from code.common.image_preprocessor import ImagePreprocessor, center_crop, resize_with_aspectratio
import cv2
import math


def preprocess_coco_for_ssdmobilenet(data_dir, preprocessed_data_dir, formats, overwrite=False, cal_only=False, val_only=False):
    def loader(file):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR)).astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = (2.0 / 255.0) * image - 1.0
        return image

    def quantizer(image):
        # Dynamic range of image is [-1.0, 1.0]
        image_int8 = image * 127.0
        return image_int8.astype(dtype=np.int8, order='C')
    preprocessor = ImagePreprocessor(loader, quantizer)
    if not val_only:
        # Preprocess calibration set. FP32 only because calibrator always takes FP32 input.
        preprocessor.run(os.path.join(data_dir, "coco", "train2017"),
                         os.path.join(preprocessed_data_dir, "coco", "train2017", "SSDMobileNet"),
                         "data_maps/coco/cal_map.txt", ["fp32"], overwrite)
    if not cal_only:
        # Preprocess validation set.
        preprocessor.run(os.path.join(data_dir, "coco", "val2017"),
                         os.path.join(preprocessed_data_dir, "coco", "val2017", "SSDMobileNet"),
                         "data_maps/coco/val_map.txt", formats, overwrite)


def copy_coco_annotations(data_dir, preprocessed_data_dir):
    src_dir = os.path.join(data_dir, "coco/annotations")
    dst_dir = os.path.join(preprocessed_data_dir, "coco/annotations")
    if not os.path.exists(dst_dir):
        shutil.copytree(src_dir, dst_dir)


def main():
    # Parse arguments to identify the data directory with the input images
    #   and the output directory for the preprocessed images.
    # The data dicretory is assumed to have the following structure:
    # <data_dir>
    #  └── coco
    #      ├── annotations
    #      ├── train2017
    #      └── val2017
    # And the output directory will have the following structure:
    # <preprocessed_data_dir>
    #  └── coco
    #      ├── annotations
    #      ├── train2017
    #      │   └── SSDMobileNet
    #      │       └── fp32
    #      └── val2017
    #          └── SSDMobileNet
    #              ├── int8_chw4
    #              └── int8_linear
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-d",
        help="Specifies the directory containing the input images.",
        default="build/data"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Specifies the output directory for the preprocessed data.",
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
    default_formats = ["int8_linear", "int8_chw4"]

    # Now, actually preprocess the input images
    logging.info("Loading and preprocessing images. This might take a while...")
    if args.formats == "default":
        formats = default_formats
    preprocess_coco_for_ssdmobilenet(data_dir, preprocessed_data_dir, formats, overwrite, cal_only, val_only)

    # Copy annotations from data_dir to preprocessed_data_dir.
    copy_coco_annotations(data_dir, preprocessed_data_dir)

    logging.info("Preprocessing done.")


if __name__ == '__main__':
    main()
