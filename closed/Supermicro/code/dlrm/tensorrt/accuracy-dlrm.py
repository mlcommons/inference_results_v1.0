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

"""Accuracy checker for DLRM benchmark from LoadGen logs."""

import argparse
import datetime
import json
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.getcwd())


def evaluate(log_path, ground_truth_file, sample_partition_file):
    """Evaluate accuracy by comparing expected to prediction. Print the result."""

    print("Loading ground truths...")
    ground_truths = np.load(ground_truth_file)
    print("Done loading ground truths.")

    print("Loading sample partition...")
    sample_partition = np.load(sample_partition_file)

    print("Parsing LoadGen accuracy log...")
    with open(log_path) as f:
        predictions = json.load(f)

    expected = []
    predicted = []
    for counter, prediction in enumerate(predictions):
        if counter % 1000 == 0:
            print("[{:}] {:} / {:}".format(datetime.datetime.now(), counter, len(predictions)))
        qsl_idx = prediction["qsl_idx"]
        assert qsl_idx < len(sample_partition), "qsl_idx exceeds total number of samples in validation dataset"

        data = np.frombuffer(bytes.fromhex(prediction["data"]), np.float32)
        start_idx = sample_partition[qsl_idx]
        end_idx = sample_partition[qsl_idx + 1]
        assert len(data) == end_idx - start_idx, "Length of predictions does not match number of pairs in sample"

        for i in data:
            predicted.append(np.nan_to_num(i))

        for i in range(start_idx, end_idx):
            expected.append(ground_truths[i])
    print("Done parsing LoadGen accuracy log.")

    print("Evaluating results...")
    score = roc_auc_score(expected, predicted)
    print("Done evaluating results.")
    print("auc={:.3f}%".format(score * 100))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mlperf-accuracy-file", help="Path to LoadGen log produced in AccuracyOnly mode")
    parser.add_argument("--ground-truth-file", help="Path to ground_truth.npy file",
                        default="build/preprocessed_data/criteo/full_recalib/ground_truth.npy")
    parser.add_argument("--sample-partition-file", help="Path to sample partition file",
                        default=os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "criteo", "full_recalib", "sample_partition.npy"))
    args = parser.parse_args()
    evaluate(args.mlperf_accuracy_file, args.ground_truth_file, args.sample_partition_file)


if __name__ == "__main__":
    main()
