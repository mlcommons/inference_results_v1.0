# coding=utf-8
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

import array
import os
import sys
import numpy as np
import collections

import mlperf_loadgen as lg
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.model import BERTEncoder, BERTModel
from gluonnlp.model.bert import bert_hparams
from utils import BertForQA, profile
from mxnet_squad_QSL import get_squad_QSL


## TODO, delete the file
class BERT_MXNET_SUT():
    def __init__(self, args, logger=None):
        self.ctx = mx.cpu()
        self.quantized = args.quantized
        self.batch_size = args.batch_size
        self.logger = logger
        if not args.quantized_model_prefix and not args.params:
            raise ValueError('Please only specify either params or quantized_model_prefix'
                             ' to run FP32 or INT8 models, but not both.')

        if self.quantized:
            if self.logger:
                self.logger.info('Loading quantized MXNet model...')
            quantized_model_prefix = args.quantized_model_prefix
            self.net = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(quantized_model_prefix),
                                            ['data0', 'data1', 'data2'],
                                            '{}-0000.params'.format(quantized_model_prefix))
            self.net.hybridize(static_alloc=True, static_shape=True)
        else:
            if self.logger:
                logger.info('Loading MXNet model...')
            mxnet_vocab = args.vocab
            mxnet_params = args.params
            with open(mxnet_vocab, 'r') as f:
                vocab = nlp.vocab.BERTVocab.from_json(f.read())

            bert, vocab = nlp.model.get_model(
                name='bert_24_1024_16',
                dataset_name=None,
                vocab=vocab,
                pretrained=False,
                ctx=self.ctx,
                use_pooler=False,
                use_decoder=False,
                use_classifier=False,
                dropout=0.0)
            self.net = BertForQA(bert=bert)
            nlp.utils.load_parameters(self.net, mxnet_params, ctx=self.ctx, cast_dtype=True)
            self.net.hybridize(static_alloc=True)

        if self.logger:
            logger.info("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        if self.logger:
            logger.info("Finished constructing SUT.")

        self.qsl = get_squad_QSL(args.vocab, logger=logger)

    def issue_queries(self, query_samples):
        def run_one_batch(cur_batch_size=1, base_index=0):
            inputs_list = []
            token_types_list = []
            valid_length_list = []
            for i in range(cur_batch_size):
                idx = base_index + i
                eval_features = self.qsl.get_features(query_samples[idx].index)
                example_ids, inputs, token_types, valid_length, _, _ = eval_features
                inputs_list.append(inputs)
                token_types_list.append(token_types)
                valid_length_list.append(valid_length)

            max_len = max([len(inp) for inp in inputs_list])
            for i in range(len(inputs_list)):
                inputs_list[i] += [0] * (max_len - len(inputs_list[i]))
                token_types_list[i] += [0] * (max_len - len(token_types_list[i]))

            inputs = mx.nd.array(inputs_list).as_in_context(self.ctx)
            token_types = mx.nd.array(token_types_list).as_in_context(self.ctx)
            valid_length = mx.nd.array(valid_length_list).as_in_context(self.ctx).astype('float32')

            ## run with a batch
            out = self.net(inputs, token_types, valid_length)
            out_np = out.asnumpy()

            out_list = np.split(out_np, cur_batch_size, axis=0)
            for i, o in enumerate(out_list):
                idx = base_index + i
                response_array = array.array("B", np.array(o).astype(np.float32).tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(query_samples[idx].id, bi[0], bi[1])
                lg.QuerySamplesComplete([response])

        num_samples = len(query_samples)
        if num_samples == 1:
            eval_features = self.qsl.get_features(query_samples[0].index)
            example_ids, inputs, token_types, valid_length, _, _ = eval_features
            inputs = mx.nd.array(inputs).reshape(1, -1)
            token_types = mx.nd.array(token_types).reshape(1, -1)
            valid_length = mx.nd.array(valid_length).reshape(-1,)

            out = self.net(inputs.as_in_context(self.ctx),
                    token_types.as_in_context(self.ctx),
                    valid_length.as_in_context(self.ctx).astype('float32'))
            out = out.asnumpy()

            response_array = array.array("B", np.array(out).astype(np.float32).tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])
        else:
            ## TODO, used in batch_size tuning
            if num_samples < self.batch_size:
                if self.logger:
                    self.logger.error('batch_size {0} is larger than provided samples {1}, consider'
                                      ' to decrease batch_size.'.format(self.batch_size, num_samples))
                sys.exit(-1)

            num_batch = num_samples // self.batch_size
            remaining_batch = num_samples % self.batch_size
            if self.logger:
                self.logger.info('split the datasets into {0} batches with bs={1} and remaining {2}...'
                                 .format(num_batch, self.batch_size, remaining_batch))

            start_step = 10
            end_step = 30 if num_batch > 30 else num_batch
            for b in range(num_batch):
                base_index = b * self.batch_size
                profile(b, start_step, end_step, profile_name='profile.json', early_exit=False)
                run_one_batch(self.batch_size, base_index)

            if remaining_batch > 0:
                base_index = num_batch * self.batch_size
                run_one_batch(remaining_batch, base_index)

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        if self.logger:
            self.logger.info("Average latency (ms) per query:")
            self.logger.info(np.mean(latencies_ns)/1000000.0)
            self.logger.info("Median latency (ms): ")
            self.logger.info(np.percentile(latencies_ns, 50)/1000000.0)
            self.logger.info("90 percentile latency (ms): ")
            self.logger.info(np.percentile(latencies_ns, 90)/1000000.0)
        else:
            print("Average latency (ms) per query:")
            print(np.mean(latencies_ns)/1000000.0)
            print("Median latency (ms): ")
            print(np.percentile(latencies_ns, 50)/1000000.0)
            print("90 percentile latency (ms): ")
            print(np.percentile(latencies_ns, 90)/1000000.0)

    def __del__(self):
        if self.logger:
            self.logger.info("Finished destroying SUT.")

def get_mxnet_sut(args, logger=None):
    return BERT_MXNET_SUT(args, logger=None)
