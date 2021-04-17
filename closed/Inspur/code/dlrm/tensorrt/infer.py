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

import os
import sys
import ctypes
sys.path.insert(0, os.getcwd())

# The plugin .so file has to be loaded at global scope and before `import torch` to avoid cuda version mismatch.
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

from code.common.runner import EngineRunner, get_input_format
from code.common import logging
import code.common.arguments as common_args
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorrt as trt
import torch
import time


def evaluate(ground_truths, predictions):
    assert len(ground_truths) == len(predictions), "Number of ground truths are different from number of predictions"
    return roc_auc_score(ground_truths, predictions)


def run_dlrm_accuracy(engine_file, batch_size, num_pairs=10000000, verbose=False):
    if verbose:
        logging.info("Running DLRM accuracy test with:")
        logging.info("    engine_file: {:}".format(engine_file))
        logging.info("    batch_size: {:}".format(batch_size))
        logging.info("    num_pairs: {:}".format(num_pairs))

    runner = EngineRunner(engine_file, verbose=verbose)
    pair_dir = os.path.join(os.getenv("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "criteo", "full_recalib")

    input_dtype, input_format = get_input_format(runner.engine)
    if input_dtype == trt.DataType.FLOAT:
        format_string = "fp32"
    elif input_dtype == trt.DataType.HALF:
        format_string = "fp16"
    elif input_dtype == trt.DataType.INT8:
        format_string = "int8"
        if input_format == trt.TensorFormat.CHW4:
            format_string += "_chw4"
    else:
        raise NotImplementedError("Unsupported DataType {:}".format(input_dtype))

    numerical_inputs = np.load(os.path.join(pair_dir, "numeric_{:}.npy".format(format_string)))
    categ_inputs = np.load(os.path.join(pair_dir, "categorical_int32.npy"))

    predictions = []
    refs = []
    batch_idx = 0
    for pair_idx in range(0, int(num_pairs), batch_size):
        actual_batch_size = batch_size if pair_idx + batch_size <= num_pairs else num_pairs - pair_idx
        numerical_input = np.ascontiguousarray(numerical_inputs[pair_idx:pair_idx + actual_batch_size])
        categ_input = np.ascontiguousarray(categ_inputs[pair_idx:pair_idx + actual_batch_size])

        start_time = time.time()
        outputs = runner([numerical_input, categ_input], actual_batch_size)

        if verbose:
            logging.info("Batch {:d} (Size {:}) >> Inference time: {:f}".format(batch_idx, actual_batch_size, time.time() - start_time))

        predictions.extend(outputs[0][:actual_batch_size])

        batch_idx += 1

    ground_truths = np.load(os.path.join(pair_dir, "ground_truth.npy"))[:num_pairs].tolist()

    return evaluate(ground_truths, predictions)


def main():
    args = common_args.parse_args(common_args.ACCURACY_ARGS)
    logging.info("Running accuracy test...")
    acc = run_dlrm_accuracy(args["engine_file"], args["batch_size"], args["num_samples"],
                            verbose=args["verbose"])
    logging.info("Accuracy: {:}".format(acc))


if __name__ == "__main__":
    main()
