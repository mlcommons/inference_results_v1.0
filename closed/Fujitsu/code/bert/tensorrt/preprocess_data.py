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

"""Script to preprocess the data for BERT."""

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
import json
import numpy as np

from code.common import logging
from code.bert.tensorrt.helpers.data_processing import read_squad_json, convert_example_to_features
from code.bert.tensorrt.helpers.tokenization import BertTokenizer


def preprocess_bert(data_dir, model_dir, preprocessed_data_dir):

    max_seq_length = 384
    max_query_length = 64
    doc_stride = 128
    output_dir = os.path.join(preprocessed_data_dir, "squad_tokenized")
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Creating tokenizer...")
    tokenizer = BertTokenizer(os.path.join(model_dir, "bert", "vocab.txt"))
    logging.info("Done creating tokenizer.")

    logging.info("Reading SQuAD examples...")
    eval_examples = read_squad_json(os.path.join(data_dir, "squad", "dev-v1.1.json"))
    logging.info("Done reading SQuAD examples.")

    logging.info("Converting examples to features...")
    eval_features = []
    for example in eval_examples:
        feature = convert_example_to_features(example.doc_tokens, example.question_text,
                                              tokenizer, max_seq_length, doc_stride, max_query_length)
        eval_features.extend(feature)
    logging.info("Done converting examples to features.")

    logging.info("Saving features...")
    eval_features_num = len(eval_features)
    input_ids = np.zeros((eval_features_num, max_seq_length), dtype=np.int32)
    input_mask = np.zeros((eval_features_num, max_seq_length), dtype=np.int32)
    segment_ids = np.zeros((eval_features_num, max_seq_length), dtype=np.int32)
    for idx, feature in enumerate(eval_features):
        print(f"Processing {idx}/{eval_features_num}...")
        input_ids[idx, :] = np.array(feature.input_ids, dtype=np.int32)
        input_mask[idx, :] = np.array(feature.input_mask, dtype=np.int32)
        segment_ids[idx, :] = np.array(feature.segment_ids, dtype=np.int32)
    np.save(os.path.join(output_dir, "input_ids.npy"), input_ids)
    np.save(os.path.join(output_dir, "input_mask.npy"), input_mask)
    np.save(os.path.join(output_dir, "segment_ids.npy"), segment_ids)
    logging.info("Done saving features.")


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

    preprocess_bert(data_dir, model_dir, preprocessed_data_dir)

    print("Done!")


if __name__ == '__main__':
    main()
