#!/usr/bin/env python3
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
import sys
sys.path.insert(0, os.getcwd())

from code.common import logging

import argparse
import json
from typing import List

# translate from LUT indices to chars
glob_results_are_indices = True   # need to translate to ascii
int_2_labels = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\'', '*']


def __levenshtein(a: List, b: List) -> int:
    """Calculates the Levenshtein distance between a and b.
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def word_error_rate(hypotheses: List[str], references: List[str]) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same length.

    Args:
        hypotheses: list of hypotheses
        references: list of references

    Returns:
        (float) average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError("In word error rate calculation, hypotheses and reference"
                         " lists must have the same number of elements. But I got:"
                         "{0} and {1} correspondingly".format(len(hypotheses), len(references)))
    for h, r in zip(hypotheses, references):
        h_list = h.split()
        r_list = r.split()
        words += len(r_list)
        scores += __levenshtein(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer, scores, words


def parse_loadgen_log(acc_log):
    with open(acc_log, "r") as acc_json:
        acc_data = json.load(acc_json)
        acc_json.close()

    # read accuracy log json and create a dictionary of qsl_idx/data pairs
    results_dict = {}
    num_acc_log_duplicate_keys = 0
    num_acc_log_data_mismatch = 0

    sortedTranscripts = [None for i in range(len(acc_data))]
    logging.info("Reading accuracy mode results...")
    for sample in acc_data:
        qsl_idx = sample["qsl_idx"]
        data = sample["data"]
        data = b''.fromhex(data)
        if glob_results_are_indices:
            data = "".join([int_2_labels[idx] for idx in list(data)])
        else:
            data = data.decode('ascii')

        sortedTranscripts[qsl_idx] = data
    return sortedTranscripts


def eval(args):
    logging.info("Start RNN-T accuracy checking")

    # Load ground truths
    with open(args.val_manifest) as f:
        manifest = json.load(f)

    offender_set = {idx for idx, f in enumerate(manifest) if f['original_duration'] > args.max_duration}
    ground_truths = [sample["transcript"] for idx, sample in enumerate(manifest) if idx not in offender_set]

    logging.info("Finished loading the ground truths")

    # Load predictions
    predictions = parse_loadgen_log(args.loadgen_log)
    logging.info("Finished loading the predictions")

    # Make sure predicions have the same number of samples as the ground truths.
    assert len(ground_truths) == len(predictions), "Predictions and ground truths do not have same number of samples"

    if(args.dump_output):
        fp = open("predictions.txt", "w")
        fg = open("ground_truth.txt", "w")

        for p in predictions:
            fp.write(p + "\n")

        for g in ground_truths:
            fg.write(g + "\n")

        fp.close()
        fg.close()

    # Note that here we don't use logging.info and instead use print because we need the output to be in stdout to
    # capture it in the testing harness.
    # Compute WER (word error rate)
    wer, _, _ = word_error_rate(predictions, ground_truths)
    # Report accuracy as well (1-WER) for convenience for the harness
    print("Word Error Rate: {:}%, accuracy={:}%".format(wer * 100, (1 - wer) * 100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loadgen_log", default="build/logs/rnnt_logs_accuracy.json")
    parser.add_argument("--val_manifest", default="build/preprocessed_data/LibriSpeech/dev-clean-wav.json")
    parser.add_argument("--max_duration", default=15.0)
    parser.add_argument("--dump_output", default=False)
    args = parser.parse_args()
    eval(args)


if __name__ == "__main__":
    main()
