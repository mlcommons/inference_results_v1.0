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
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import collections
import json
import logging
import os
import io
import random
import time
import warnings
import itertools
import pickle
import multiprocessing as mp
from functools import partial

import numpy as np
import mxnet as mx

import gluonnlp as nlp
from gluonnlp.data import SQuAD
from gluonnlp.calibration import BertLayerCollector
from utils import BertForQA
from gluonnlp.data.bert.glue import concat_sequences
from gluonnlp.data.bert.squad import improve_answer_span, \
        tokenize_and_align_positions, get_doc_spans, align_position2doc_spans, \
        check_is_max_context, convert_squad_examples
from preprocessing_utils import convert_examples_to_features, preprocess_dataset


np.random.seed(8)
random.seed(6)
mx.random.seed(6)

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')

parser = argparse.ArgumentParser(
    description='BERT QA example.'
    'We fine-tune the BERT model on SQuAD dataset.')

parser.add_argument('--model_parameters',
                    type=str,
                    default=None,
                    help='Model parameter file')

parser.add_argument('--bert_model',
                    type=str,
                    default='bert_12_768_12',
                    help='BERT model name. options are bert_12_768_12 and bert_24_1024_16.')

parser.add_argument('--bert_dataset',
                    type=str,
                    default='book_corpus_wiki_en_uncased',
                    help='BERT dataset name.'
                    'options are book_corpus_wiki_en_uncased and book_corpus_wiki_en_cased.')

parser.add_argument('--uncased',
                    action='store_false',
                    help='if not set, inputs are converted to lower case.')

parser.add_argument('--output_dir',
                    type=str,
                    default='./output_dir',
                    help='The output directory where the model params will be written.'
                    ' default is ./output_dir')

parser.add_argument('--test_batch_size',
                    type=int,
                    default=24,
                    help='Test batch size. default is 24')

parser.add_argument('--log_interval',
                    type=int,
                    default=50,
                    help='report interval. default is 50')

parser.add_argument('--max_seq_length',
                    type=int,
                    default=384,
                    help='The maximum total input sequence length after WordPiece tokenization.'
                    'Sequences longer than this will be truncated, and sequences shorter '
                    'than this will be padded. default is 384')

parser.add_argument(
    '--round_to', type=int, default=None,
    help='The length of padded sequences will be rounded up to be multiple of this argument.'
         'When round to is set to 8, training throughput may increase for mixed precision'
         'training on GPUs with tensorcores.')

parser.add_argument('--doc_stride',
                    type=int,
                    default=128,
                    help='When splitting up a long document into chunks, how much stride to '
                    'take between chunks. default is 128')

parser.add_argument('--max_query_length',
                    type=int,
                    default=64,
                    help='The maximum number of tokens for the question. Questions longer than '
                    'this will be truncated to this length. default is 64')

parser.add_argument('--n_best_size',
                    type=int,
                    default=20,
                    help='The total number of n-best predictions to generate in the '
                    'nbest_predictions.json output file. default is 20')

parser.add_argument('--max_answer_length',
                    type=int,
                    default=30,
                    help='The maximum length of an answer that can be generated. This is needed '
                    'because the start and end predictions are not conditioned on one another.'
                    ' default is 30')

parser.add_argument('--version_2',
                    action='store_true',
                    help='SQuAD examples whether contain some that do not have an answer.')

parser.add_argument('--null_score_diff_threshold',
                    type=float,
                    default=0.0,
                    help='If null_score - best_non_null is greater than the threshold predict null.'
                    'Typical values are between -1.0 and -5.0. default is 0.0')

parser.add_argument('--debug',
                    action='store_true',
                    help='Run the example in test mode for sanity checks')

parser.add_argument('--dtype',
                    type=str,
                    default='float32',
                    help='Data type used for training. Either float32 or float16')

parser.add_argument('--num_calib_batches', type=int, default=10,
                    help='number of batches for calibration')
parser.add_argument('--quantized_dtype', type=str, default='auto',
                    choices=['auto', 'int8', 'uint8'],
                    help='quantization destination data type for input data')
parser.add_argument('--calib_mode', type=str, default='customize',
                    choices=['none', 'naive', 'entropy', 'customize'],
                    help='calibration mode used for generating calibration table '
                         'for the quantized symbol.')

parser.add_argument('--vocab_path', type=str,
                    default=None,
                    help='vocab file path.')

parser.add_argument('--scenario', type=str,
                    default="offline",
                    help='model will be optimized for chosen scenario')


args = parser.parse_args()

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

fh = logging.FileHandler(os.path.join(args.output_dir, 'finetune_squad.log'),
                         mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)
log.addHandler(fh)

log.info(args)

model_name = args.bert_model
dataset_name = args.bert_dataset
scenario = args.scenario
if scenario not in ("offline", "server"):
    raise ValueError('\"offline\" and \"server\" are the only valid values for scenario')

model_parameters = args.model_parameters
lower = args.uncased
test_batch_size = args.test_batch_size
ctx = mx.cpu()

log_interval = args.log_interval
version_2 = args.version_2
null_score_diff_threshold = args.null_score_diff_threshold
max_seq_length = args.max_seq_length
doc_stride = args.doc_stride
max_query_length = args.max_query_length
n_best_size = args.n_best_size
max_answer_length = args.max_answer_length

if max_seq_length <= max_query_length + 3:
    raise ValueError('The max_seq_length (%d) must be greater than max_query_length '
                     '(%d) + 3' % (max_seq_length, max_query_length))

pretrained = not model_parameters
bert, vocab = nlp.model.get_model(
    name=model_name,
    dataset_name=dataset_name,
    vocab=None,
    pretrained=pretrained,
    ctx=ctx,
    use_pooler=False,
    use_decoder=False,
    use_classifier=False,
    dropout=0.0)
net = BertForQA(bert=bert)

if args.vocab_path:
    with open(args.vocab_path, 'r') as f:
        vocab = nlp.vocab.BERTVocab.from_json(f.read())
log.info('load external vocab, len(vocab) is: {}'.format(len(vocab)))

tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=lower)

batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Stack(),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=args.round_to),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=args.round_to),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'))

if model_parameters:
    # load complete BertForQA parameters
    nlp.utils.load_parameters(net, model_parameters, ctx=ctx, cast_dtype=True)
else:
    # no checkpoint is loaded
    net.initialize(init=mx.init.Normal(0.02), ctx=ctx)
libpath = os.path.abspath('./bertpass.so')
mx.library.load(libpath)

def run_pass(net, pass_name):
    a = mx.nd.random.uniform(shape=(1,384))
    b = mx.nd.random.uniform(shape=(1,384))
    c = mx.nd.random.uniform(shape=(1,))
    net.hybridize()
    net(a,b,c)
    # TODO: Try to make inmemory
    if pass_name == "custom_pass":
        net.export("original")
    tmp_name = "tmp_symbol"
    net.export(tmp_name)

    sym, arg_params, aux_params = mx.model.load_checkpoint(tmp_name, 0)
    arg_array = arg_params
    arg_array['data0'] = mx.nd.ones((test_batch_size, max_seq_length), dtype='float32')
    arg_array['data1'] = mx.nd.ones((test_batch_size, max_seq_length), dtype='float32')
    arg_array['data2'] = mx.nd.ones((test_batch_size, ), dtype='float32')
    custom_sym = sym.optimize_for(pass_name, arg_array, aux_params)

    arg_array.pop('data0')
    arg_array.pop('data1')
    arg_array.pop('data2')

    mx.model.save_checkpoint(tmp_name, 0, custom_sym, arg_array, aux_params)
    mx.nd.waitall()
    net = mx.gluon.SymbolBlock.imports(tmp_name + "-symbol.json", ["data0", "data1", "data2"], tmp_name + "-0000.params")
    #print(arg_array)
    mx.nd.waitall()
    net.hybridize(static_alloc=True)
    return net

# calibration config
num_calib_batches = args.num_calib_batches
quantized_dtype = args.quantized_dtype
calib_mode = args.calib_mode

def calibration(net, num_calib_batches, quantized_dtype, calib_mode):
    """calibration function on the dev dataset."""
    log.info('Loading dev data...')
    if version_2:
        dev_data = SQuAD('dev', version='2.0')
    else:
        dev_data = SQuAD('dev', version='1.1')
    if args.debug:
        sampled_data = [dev_data[0], dev_data[1], dev_data[2]]
        dev_data = mx.gluon.data.SimpleDataset(sampled_data)
    log.info('Number of records in dev data:{}'.format(len(dev_data)))
    origin_dev_data_len = len(dev_data)
    num_calib_examples = test_batch_size * num_calib_batches
    ### randomly select the calib data from full dataset
    random_indices = np.random.choice(origin_dev_data_len, num_calib_examples)
    print ('random_indices: ', random_indices)
    dev_data=list(dev_data[i] for i in random_indices)
    log.info('Number of records in dev data:{}'.format(len(dev_data)))

    batchify_fn_calib = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=args.round_to),
        nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=args.round_to),
        nlp.data.batchify.Stack('float32'),
        nlp.data.batchify.Stack('float32'))

    dev_data_transform = preprocess_dataset(tokenizer,
                                            dev_data,
                                            max_seq_length=max_seq_length,
                                            doc_stride=doc_stride,
                                            max_query_length=max_query_length,
                                            input_features=True,
                                            for_calibration=True)

    dev_dataloader = mx.gluon.data.DataLoader(
        dev_data_transform,
        batchify_fn=batchify_fn_calib,
        num_workers=4, batch_size=test_batch_size,
        shuffle=True, last_batch='keep')

    net = run_pass(net, 'custom_pass')
    assert ctx == mx.cpu(), \
        'Currently only supports CPU with MKL-DNN backend.'
    log.info('Now we are doing calibration on dev with %s.', ctx)
    collector = BertLayerCollector(clip_min=-50, clip_max=10, logger=log)
    net = mx.contrib.quantization.quantize_net_v2(net, quantized_dtype=quantized_dtype,
                                                  exclude_layers=[],
                                                  quantize_mode='smart',
                                                  quantize_granularity='tensor-wise',
                                                  calib_data=dev_dataloader,
                                                  calib_mode=calib_mode,
                                                  num_calib_examples=num_calib_examples,
                                                  ctx=ctx,
                                                  LayerOutputCollector=collector,
                                                  logger=log)
    if scenario == "offline":
        net = run_pass(net, 'softmax_mask')
    else:
        net = run_pass(net, 'normal_softmax')

    net = run_pass(net, 'bias_to_s32')

    # # save params
    ckpt_name = 'model_bert_squad_quantized_{0}'.format(calib_mode)
    params_saved = os.path.join(output_dir, ckpt_name)
    net.hybridize(static_alloc=True, static_shape=True)
    
    a = mx.nd.ones((test_batch_size, max_seq_length), dtype='float32')
    b = mx.nd.ones((test_batch_size, max_seq_length), dtype='float32')
    c = mx.nd.ones((test_batch_size, ), dtype='float32')
    net(a,b,c)
    mx.nd.waitall()
    net.export(params_saved, epoch=0)
    log.info('Saving quantized model at %s', output_dir)


if __name__ == '__main__':
    try:
        calibration(net,
                    num_calib_batches,
                    quantized_dtype,
                    calib_mode)
    except AttributeError:
        nlp.utils.version.check_version('1.7.0', warning_only=True, library=mx)
        warnings.warn('INT8 Quantization for BERT need mxnet-mkl >= 1.6.0b20200115')
