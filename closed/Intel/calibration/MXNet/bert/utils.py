# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Utility functions for BERT."""

import collections
import hashlib
import io

import mxnet as mx
from mxnet.gluon import HybridBlock, nn
import gluonnlp as nlp

__all__ = ['tf_vocab_to_gluon_vocab', 'load_text_vocab']

class BertForQA(HybridBlock):
    """Model for SQuAD task with BERT.

    The model feeds token ids and token type ids into BERT to get the
    pooled BERT sequence representation, then apply a Dense layer for QA task.

    Parameters
    ----------
    bert: BERTModel
        Bidirectional encoder with transformer.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """

    def __init__(self, bert, prefix=None, params=None):
        super(BertForQA, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.span_classifier = nn.Dense(units=2, flatten=False)

    def __call__(self, inputs, token_types, valid_length=None):
        #pylint: disable=arguments-differ, dangerous-default-value
        """Generate the unnormalized score for the given the input sequences."""
        # XXX Temporary hack for hybridization as hybridblock does not support None inputs
        valid_length = [] if valid_length is None else valid_length
        return super(BertForQA, self).__call__(inputs, token_types, valid_length)

    def hybrid_forward(self, F, inputs, token_types, valid_length=None):
        # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or None, shape (batch_size,)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, seq_length, 2)
        """
        # XXX Temporary hack for hybridization as hybridblock does not support None inputs
        if isinstance(valid_length, list) and len(valid_length) == 0:
            valid_length = None
        bert_output = self.bert(inputs, token_types, valid_length)
        output = self.span_classifier(bert_output)
        return output


def tf_vocab_to_gluon_vocab(tf_vocab):
    special_tokens = ['[UNK]', '[PAD]', '[SEP]', '[MASK]', '[CLS]']
    assert all(t in tf_vocab for t in special_tokens)
    counter = nlp.data.count_tokens(tf_vocab.keys())
    vocab = nlp.vocab.BERTVocab(counter, token_to_idx=tf_vocab)
    return vocab


def get_hash(filename):
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest(), str(sha1.hexdigest())[:8]


def read_tf_checkpoint(path):
    """read tensorflow checkpoint"""
    from tensorflow.python import pywrap_tensorflow
    tensors = {}
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        tensor = reader.get_tensor(key)
        tensors[key] = tensor
    return tensors

def profile(curr_step, start_step, end_step, profile_name='profile.json',
            early_exit=True):
    """profile the program between [start_step, end_step)."""
    if curr_step == start_step:
        mx.nd.waitall()
        mx.profiler.set_config(profile_memory=False, profile_symbolic=True,
                               profile_imperative=True, filename=profile_name,
                               aggregate_stats=True)
        mx.profiler.set_state('run')
    elif curr_step == end_step:
        mx.nd.waitall()
        mx.profiler.set_state('stop')
        mx.profiler.dump()
        if early_exit:
            exit()

def load_text_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with io.open(vocab_file, 'r') as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def read_tf_pb(path):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    constant_values = {}
    sess = tf.Session()
    with gfile.FastGFile(path, 'rb') as f:
        graph_def = tf.GraphDef() ## deprecated API
        #graph_def = tf.compat.v1.GraphDef
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
        for constant_op in constant_ops:
            val = sess.run(constant_op.outputs[0])
            if val.ndim < 1:
                continue
            constant_values[constant_op.name] = val

    return constant_values