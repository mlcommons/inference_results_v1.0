#! /usr/bin/env python3
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

import ctypes
import os
import sys
sys.path.insert(0, os.getcwd())

UNET_INTERACTIONS_PLUGIN_LIBRARY = "build/plugins/instanceNormalization3DPlugin/libinstancenorm3dplugin.so"
if not os.path.isfile(UNET_INTERACTIONS_PLUGIN_LIBRARY):
    raise IOError("{}\n{}\n".format(
        "Failed to load library ({}).".format(UNET_INTERACTIONS_PLUGIN_LIBRARY),
        "Please build the NMS Opt plugin."
    ))
ctypes.CDLL(UNET_INTERACTIONS_PLUGIN_LIBRARY)

PIXELSHUFFLE3D_PLUGIN_LIBRARY = "build/plugins/pixelShuffle3DPlugin/libpixelshuffle3dplugin.so"
if not os.path.isfile(PIXELSHUFFLE3D_PLUGIN_LIBRARY):
    raise IOError("{}\n{}\n".format(
        "Failed to load library ({}).".format(PIXELSHUFFLE3D_PLUGIN_LIBRARY),
        "Please build the pixelShuffle3D plugin."
    ))
ctypes.CDLL(PIXELSHUFFLE3D_PLUGIN_LIBRARY)

CONV3D1X1X14K_PLUGIN_LIBRARY = "build/plugins/conv3D1X1X1K4Plugin/libconv3D1X1X1K4Plugin.so"
if not os.path.isfile(CONV3D1X1X14K_PLUGIN_LIBRARY):
    raise IOError("{}\n{}\n".format(
        "Failed to load library ({}).".format(CONV3D1X1X14K_PLUGIN_LIBRARY),
        "Please build the conv3D1X1X1K4 plugin."
    ))
ctypes.CDLL(CONV3D1X1X14K_PLUGIN_LIBRARY)

from code.common.accuracy import AccuracyRunner
from code.common.runner import EngineRunner, get_input_format
from code.common import logging
import code.common.arguments as common_args
import json
import numpy as np
import tensorrt as trt
import time


def run_3dunet_accuracy(engine_file, batch_size, num_images, verbose=False):
    runner = EngineRunner(engine_file, verbose=verbose)
    input_dtype, input_format = get_input_format(runner.engine)

    if verbose:
        logging.info("Running UNET accuracy test with:")
        logging.info("    engine_file: {:}".format(engine_file))
        logging.info("    batch_size: {:}".format(batch_size))
        logging.info("    num_images: {:}".format(num_images))
        logging.info("    input_dtype: {:}".format(input_dtype))
        logging.info("    input_format: {:}".format(input_format))

    if input_dtype == trt.DataType.FLOAT:
        format_string = "fp32"
    elif input_dtype == trt.DataType.INT8:
        if input_format == trt.TensorFormat.LINEAR:
            format_string = "int8_linear"
    elif input_dtype == trt.DataType.HALF:
        if input_format == trt.TensorFormat.LINEAR:
            format_string = "fp16_linear"
        elif input_format == trt.TensorFormat.DHWC8:
            format_string = "fp16_dhwc8"
    image_dir = os.path.join(os.getenv("PREPROCESSED_DATA_DIR", "build/preprocessed_data"),
                             "brats", "brats_npy", format_string)

    if num_images is None:
        num_images = 67

    image_list = []
    with open("data_maps/brats/val_map.txt") as f:
        for line in f:
            image_list.append(line.split()[0])

    predictions = []
    batch_idx = 0
    for image_idx in range(0, num_images, batch_size):
        actual_batch_size = batch_size if image_idx + batch_size <= num_images else num_images - image_idx
        batch_images = np.ascontiguousarray(np.stack([np.load(os.path.join(image_dir, name + ".npy")) for name in image_list[image_idx:image_idx + actual_batch_size]]))

        start_time = time.time()
        outputs = runner([batch_images], actual_batch_size)

        print(np.mean(batch_images[0].astype(np.float32)))
        print(np.std(batch_images[0].astype(np.float32)))
        print(np.mean(outputs[0].astype(np.float32)))
        print(np.std(outputs[0].astype(np.float32)))

        if verbose:
            logging.info("Batch {:d} (Size {:}) >> Inference time: {:f}".format(batch_idx, actual_batch_size, time.time() - start_time))

        predictions.extend(outputs[0][:actual_batch_size])

        batch_idx += 1

    logging.warning("3D-Unet standalone accuracy checker does not have accuracy checking yet! Always return 1.0")

    return 1.0


def main():
    args = common_args.parse_args(common_args.ACCURACY_ARGS)
    logging.info("Running accuracy test...")
    acc = run_3dunet_accuracy(args["engine_file"], args["batch_size"], args["num_samples"],
                              verbose=args["verbose"])
    logging.info("Accuracy: {:}".format(acc))


if __name__ == "__main__":
    main()
