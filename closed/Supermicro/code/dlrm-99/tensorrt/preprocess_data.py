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

"""Preprocess data for the DLRM benchmark."""

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
from argparse import Namespace
import json
import numpy as np

from code.common import logging
from code.dlrm.tensorrt.scripts.convert_dlrm_data import process_dlrm_data
from code.dlrm.tensorrt.scripts.data_loader_terabyte import _preprocess
from code.dlrm.tensorrt.scripts.data_utils import loadDataset
from code.dlrm.tensorrt.scripts.get_frequency_data import get_frequency_data


def preprocess_dlrm(data_dir, model_dir, preprocessed_data_dir):

    np.random.seed(123)
    raw_data_file = os.path.join(data_dir, "criteo", "day")
    processed_data_file = os.path.join(preprocessed_data_dir, "criteo", "day")
    output_dir = os.path.join(preprocessed_data_dir, "criteo")
    embedding_rows_bound = 40000000

    print("Converting day_0, day_1, ... to day_0_reordered.npz, day_1_reordered.npz, ...")
    loadDataset(
        "terabyte",
        embedding_rows_bound,
        0.0,
        "total",
        "train",
        raw_data_file,
        processed_data_file,
        True
    )

    print("Converting day_0_reordered.npz, day_1_reordered.npz, ... to train_data.bin, test_data.bin, val_data.bin ...")
    args = Namespace(input_data_prefix=processed_data_file, output_directory=output_dir)
    _preprocess(args)

    print("Converting test_data.bin and val_data.bin to npy files...")
    full_recalib_dir = os.path.join(output_dir, "full_recalib")
    cal_data_dir = os.path.join(full_recalib_dir, "val_data_128000")
    # Test set
    process_dlrm_data(embedding_rows_bound, os.path.join(output_dir, "test_data.bin"), full_recalib_dir, 0)
    # Calibration set: first 128000 pairs in val_data.bin
    process_dlrm_data(embedding_rows_bound, os.path.join(output_dir, "val_data.bin"), cal_data_dir, 128000)

    print("Processing training set to get frequency data...")
    row_freq_dir = os.path.join(model_dir, "dlrm", "40m_limit", "row_freq")
    os.makedirs(row_freq_dir, exist_ok=True)
    get_frequency_data(os.path.join(output_dir, "train_data.bin"), row_freq_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", "-d",
        help="Directory containing the input data.",
        default="build/data"
    )
    parser.add_argument(
        "--model_dir", "-m",
        help="Directory containing the models.",
        default="build/models"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Output directory for the preprocessed data.",
        default="build/preprocessed_data"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    preprocessed_data_dir = args.preprocessed_data_dir

    preprocess_dlrm(data_dir, model_dir, preprocessed_data_dir)

    print("Done!")


if __name__ == '__main__':
    main()
