# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
import pickle

import mlperf_loadgen as lg
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.data import SQuAD
from preprocessing_utils import preprocess_dataset, max_seq_length, doc_stride, max_query_length


## TODO, delete the file
class SQuAD_v1_QSL():
    def __init__(self, mxnet_vocab=None, perf_count=None, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.info("Constructing QSL...")
        test_batch_size = 1
        eval_features = []

        if self.logger:
            self.logger.info("Creating tokenizer...")
        with open(mxnet_vocab, 'r') as f:
            vocab = nlp.vocab.BERTVocab.from_json(f.read())
        tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=True)

        round_to = None
        if self.logger:
            self.logger.info("Reading examples...")
        dev_path = os.path.join(os.getcwd(), 'build/data')
        dev_data = SQuAD('dev', version='1.1', root=dev_path)
        dev_data_transform = preprocess_dataset(tokenizer,
                                                dev_data,
                                                max_seq_length=max_seq_length,
                                                doc_stride=doc_stride,
                                                max_query_length=max_query_length,
                                                input_features=True)

        self.eval_features = dev_data_transform
        self.count = len(self.eval_features)
        self.perf_count = perf_count if perf_count is not None else self.count
        self.qsl = lg.ConstructQSL(self.count, self.perf_count, self.load_query_samples, self.unload_query_samples)
        if self.logger:
            self.logger.info("Finished constructing QSL.")

    def load_query_samples(self, sample_list):
        pass

    def unload_query_samples(self, sample_list):
        pass

    def get_features(self, sample_id):
        return self.eval_features[sample_id]

    def __del__(self):
        if self.logger:
            self.logger.info("Finished destroying QSL.")

def get_squad_QSL(vocab=None, logger=None):
    return SQuAD_v1_QSL(mxnet_vocab=vocab, logger=logger)
