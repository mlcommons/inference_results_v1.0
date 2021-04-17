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
sys.path.insert(0, os.getcwd())

import argparse
import collections
import json
import numpy as np

from code.bert.tensorrt.evaluate import f1_score, exact_match_score
from code.bert.tensorrt.helpers.data_processing import get_predictions, read_squad_json, convert_example_to_features
from code.bert.tensorrt.helpers.tokenization import BertTokenizer
from code.common import logging

_NetworkOutput = collections.namedtuple("NetworkOutput", ["start_logits", "end_logits", "feature_index"])


def get_score(predictions):

    logging.info("Evaluating predictions...")

    input_file = "build/data/squad/dev-v1.1.json"

    with open(input_file) as f:
        data = json.load(f)["data"]

    f1_score_total = 0.0
    exact_score_total = 0.0
    sample_idx = 0
    for task in data:
        title = task["title"]
        for paragraph_idx, paragraph in enumerate(task["paragraphs"]):
            context = paragraph["context"]
            for q_idx, qas in enumerate(paragraph["qas"]):
                if sample_idx < len(predictions):
                    answers = qas["answers"]
                    f1_score_this = 0.0
                    exact_score_this = 0.0
                    for answer in answers:
                        f1_score_this = max(f1_score_this, f1_score(predictions[sample_idx], answer["text"]))
                        exact_score_this = max(exact_score_this, exact_match_score(predictions[sample_idx], answer["text"]))
                    f1_score_total += f1_score_this
                    exact_score_total += exact_score_this
                sample_idx += 1

    f1_score_avg = f1_score_total / len(predictions) * 100
    exact_score_avg = exact_score_total / len(predictions) * 100

    return (exact_score_avg, f1_score_avg)


def evaluate(log_path, squad_path):
    logging.info("Creating tokenizer...")
    tokenizer = BertTokenizer("build/models/bert/vocab.txt")
    logging.info("Done creating tokenizer.")

    logging.info("Reading SQuAD examples...")
    eval_examples = read_squad_json(squad_path)
    logging.info("Done reading SQuAD examples.")

    logging.info("Converting examples to features...")
    max_seq_length = 384
    max_query_length = 64
    doc_stride = 128
    eval_features = []
    num_features_per_example = []
    for example_idx, example in enumerate(eval_examples):
        feature = convert_example_to_features(example.doc_tokens, example.question_text,
                                              tokenizer, max_seq_length, doc_stride, max_query_length)
        eval_features.extend(feature)
        num_features_per_example.append(len(feature))
    logging.info("Done converting examples to features.")

    logging.info("Collecting LoadGen results...")
    with open(log_path) as f:
        log_predictions = json.load(f)
    score_total = 0.0
    results = [None for i in range(len(eval_features))]

    logits_padded = np.zeros((max_seq_length, 2), dtype=np.float16)
    for prediction in log_predictions:
        qsl_idx = prediction["qsl_idx"]
        assert qsl_idx < len(eval_features), "qsl_idx exceeds total number of features"

        data = np.frombuffer(bytes.fromhex(prediction["data"]), np.float16)
        data = data.reshape(-1, 2)
        seq_len = data.shape[0]
        logits_padded.fill(-10000.0)
        logits_padded[:seq_len, :] = data
        start_logits = logits_padded[:, 0].copy()
        end_logits = logits_padded[:, 1].copy()
        results[qsl_idx] = _NetworkOutput(start_logits=start_logits, end_logits=end_logits, feature_index=qsl_idx)
    logging.info("Done collecting LoadGen results.")

    logging.info("Evaluating results...")
    predictions = []
    feature_idx = 0
    # Total number of n-best predictions to generate in the nbest_predictions.json output file
    n_best_size = 20
    # The maximum length of an answer that can be generated. This is needed
    # because the start and end predictions are not conditioned on one another
    max_answer_length = 30
    for example_idx, example in enumerate(eval_examples):
        results_per_example = []
        for i in range(num_features_per_example[example_idx]):
            results_per_example.append(results[feature_idx])
            feature_idx += 1

        prediction, _, _ = get_predictions(example.doc_tokens, eval_features, results_per_example, n_best_size, max_answer_length)
        predictions.append(prediction)

    exact_score, f1_score = get_score(predictions)
    print("{{\"exact_match\": {:.3f}, \"f1\": {:.3f}}}".format(exact_score, f1_score))


def main():
    parser = argparse.ArgumentParser("Accuracy checker for BERT benchmark from LoadGen logs")
    parser.add_argument("--mlperf-accuracy-file", help="Path to LoadGen log produced in AccuracyOnly mode")
    parser.add_argument("--squad-val-file", help="Path to SQuAD 1.1 json file", default="build/data/squad/dev-v1.1.json")
    args = parser.parse_args()
    evaluate(args.mlperf_accuracy_file, args.squad_val_file)


if __name__ == "__main__":
    main()
