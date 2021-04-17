# Copyright (c) 2020, Cerebras Systems, Inc. All rights reserved.
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

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "pytorch"))

import array
import numpy as np
import toml
import mlperf_loadgen as lg
from tqdm import tqdm

from QSL import AudioQSL, AudioQSLInMemory
from decoders import ScriptGreedyDecoder
from helpers import add_blank_label
from preprocessing import AudioPreprocessing
from model_separable_rnnt import RNNT

import multiprocessing as mp
import threading
import time
import torch
import torch.autograd.profiler as profiler

query_count = 0
finish_count = 0
start_time = time.time()
debug = False

def get_num_cores():
    cmd = "lscpu | awk '/^Core\(s\) per socket:/ {cores=$NF}; /^Socket\(s\):/ {sockets=$NF}; END{print cores*sockets}'"
    lscpu = os.popen(cmd).readlines()
    lscpu = int(lscpu[0])
    return lscpu

def block_until(counter, num_ins, t):
    while counter.value < num_ins:
        time.sleep(t)

class Input(object):
    def __init__(self, id_list, idx_list):
        assert isinstance(id_list, list)
        assert isinstance(idx_list, list)
        assert len(id_list) == len(idx_list)
        self.query_id_list = id_list
        self.query_idx_list = idx_list


class Output(object):
    def __init__(self, query_id, transcript):
        self.query_id = query_id
        self.transcript = transcript


class InQueue():
    def __init__(self, in_queue, batch_size=1):
        self.in_queue = in_queue
        self.batch_size = batch_size

    def put(self, query_samples):
        query_len = len(query_samples)
        query_idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]

        if query_len == 1:
            input_item = Input(query_id, query_idx)
            self.in_queue.put(input_item)
        else:
            bs = self.batch_size
            for i in range(0, query_len, bs):
                i_end = min(i + bs, query_len)
                input_item = Input(query_id[i:i_end], query_idx[i:i_end])
                self.in_queue.put(input_item)

class InQueueServer():
    def __init__(self, input_queue_list, qsl, seq_cutoff_list, 
                 batch_size_list, total_query_count):

        self.input_queue_list = input_queue_list
        self.qsl = qsl
        self.seq_cutoff_list = seq_cutoff_list
        self.num_queues = len(input_queue_list) 
        self.batch_size_list = batch_size_list 
        self.query_batcher = [[] for _ in range(self.num_queues)]
        self.total_query_count = total_query_count
        self.curr_query_count = 0

    def put(self, query_samples): 

        assert len(query_samples) == 1  # server scenario
        self.curr_query_count += 1 

        for i in range(self.num_queues): 
            idx = query_samples[0].index  #BS=1
            waveform = self.qsl[idx] 
            if len(waveform) <= self.seq_cutoff_list[i]:
                self.query_batcher[i].append(query_samples[0]) 
                # put queries in queue if BS treshold reached
                if len(self.query_batcher[i]) == self.batch_size_list[i]:
                    qid_list, qidx_list = [], []
                    for q in self.query_batcher[i]:
                       qid_list.append(q.id) 
                       qidx_list.append(q.index)
                    input_item = Input(qid_list, qidx_list)
                    self.input_queue_list[i].put(input_item)
                    self.query_batcher[i] = [] 
                break 

        if self.curr_query_count == self.total_query_count:
            # no more calls to put function
            # submit remaining queries in query batcher to input queues
            # process remaining queries with BS=1    
            for i in range(self.num_queues): 
                for q in self.query_batcher[i]: 
                    input_item = Input([q.id], [q.index])
                    self.input_queue_list[i].put(input_item)            

class Consumer(mp.Process):
    def __init__(self, task_queue, result_queue, lock, init_counter,
                 rank, start_core, end_core, num_cores,
                 qsl, config_toml, checkpoint_path, dataset_dir,
                 manifest_filepath, perf_count, cosim, profile, ipex, bf16,
                 warmup):

        mp.Process.__init__(self)

        ### sub process
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.lock = lock
        self.init_counter = init_counter
        self.rank = rank
        self.start_core = start_core
        self.end_core = end_core

        self.qsl = qsl
        self.config_toml = config_toml
        self.checkpoint_path = checkpoint_path
        self.dataset_dir = dataset_dir
        self.manifest_filepath = manifest_filepath
        self.perf_count = perf_count
        self.cosim = cosim
        self.profile = profile
        self.ipex = ipex
        self.bf16 = bf16
        self.warmup = warmup

        self.model_init = False

    # warmup basically go through samples with different feature lengths so
    # all shapes can be prepared
    def do_warmup(self):
        print ('Start warmup...')
        length_list = {}
        count = 0
        idxs = self.qsl.idxs()
        for i in idxs:
            feature_list = []
            feature_length_list = []
            waveform = self.qsl[i]
            feature_element, feature_length = self.audio_preprocessor.forward(
                                                    (torch.from_numpy(waveform).unsqueeze(0),
                                                     torch.tensor(len(waveform)).unsqueeze(0)))
            feature_list.append(feature_element.squeeze(0).transpose_(0, 1))
            feature_length_list.append(feature_length.squeeze(0))
            feature = torch.nn.utils.rnn.pad_sequence(feature_list, batch_first=True)
            feature_length = torch.tensor(feature_length_list)

            if feature_length[0].item() in length_list:
                continue
            length_list[feature_length[0].item()] = True

            assert feature.ndim == 3
            assert feature_length.ndim == 1
            if self.ipex:
                import intel_pytorch_extension as ipex
                if self.bf16:
                    ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
                ipex.core.enable_auto_dnnl()
                feature = feature.to(ipex.DEVICE)
                feature_length = feature_length.to(ipex.DEVICE)
            feature_ = feature.permute(1, 0, 2)
            _, _, transcripts = self.greedy_decoder.forward_batch(feature_, feature_length, self.rank)

            count += 1
            if self.rank==0 and count % 10 == 0:
                print ('Warmup {} samples'.format(count))
        print ('Warmup done')

    def run_queue(self, debug = False):
        next_task = self.task_queue.get()
        if next_task is None:
            self.task_queue.task_done()
            return False

        query_id_list = next_task.query_id_list
        query_idx_list = next_task.query_idx_list
        query_len = len(query_id_list)
        with torch.no_grad():
            t1 = time.time()
            serial_audio_processor = True
            if serial_audio_processor:
                feature_list = []
                feature_length_list = []
                for idx in query_idx_list:
                    waveform = self.qsl[idx]
                    feature_element, feature_length = self.audio_preprocessor.forward(
                                                            (torch.from_numpy(waveform).unsqueeze(0),
                                                             torch.tensor(len(waveform)).unsqueeze(0)))
                    feature_list.append(feature_element.squeeze(0).transpose_(0, 1))
                    feature_length_list.append(feature_length.squeeze(0))
                feature = torch.nn.utils.rnn.pad_sequence(feature_list, batch_first=True)
                feature_length = torch.tensor(feature_length_list)
            else:
                waveform_list = []
                for idx in query_idx_list:
                    waveform = self.qsl[idx]
                    waveform_list.append(torch.from_numpy(waveform))
                waveform_batch = torch.nn.utils.rnn.pad_sequence(waveform_list, batch_first=True)
                waveform_lengths = torch.tensor([waveform.shape[0] for waveform in waveform_list],
                                                                dtype=torch.int64)

                feature, feature_length = self.audio_preprocessor.forward((waveform_batch, waveform_lengths))

            assert feature.ndim == 3
            assert feature_length.ndim == 1
            if self.ipex:
                import intel_pytorch_extension as ipex
                if self.bf16:
                    ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
                ipex.core.enable_auto_dnnl()
                feature = feature.to(ipex.DEVICE)
                feature_length = feature_length.to(ipex.DEVICE)
            if serial_audio_processor:
                feature_ = feature.permute(1, 0, 2)
            else:
                feature_ = feature.permute(2, 0, 1)
            t3 = time.time()
            if query_len == 1:
                _, _, transcripts = self.greedy_decoder.forward_single_batch(feature_, feature_length, self.ipex, self.rank)
            else:
                _, _, transcripts = self.greedy_decoder.forward_batch(feature_, feature_length, self.ipex, self.rank)
            t4 = time.time()
            # cosim
            if self.cosim:
                _, _, transcripts0 = self.greedy_decoder.forward(feature, feature_length)
                if transcripts0 != transcripts:
                    print ('vvvvvv difference between reference and batch impl. vvvvvv')
                    for i in range(query_len):
                        if transcripts0[i] != transcripts[i]:
                            for j in range(len(transcripts0[i])):
                                if transcripts0[i][j] != transcripts[i][j]:
                                    break
                            print ('[{}] reference'.format(i))
                            print ('{} diff {}'.format(transcripts0[i][0:j], transcripts0[i][j:]))
                            print ('[{}] batch'.format(i))
                            print ('{} diff {}'.format(transcripts[i][0:j], transcripts[i][j:]))
                            print ('')
                    print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                else:
                    print ('.', end='', flush=True)

        t6 = time.time()
        assert len(transcripts) == query_len
        for id, trans in zip(query_id_list, transcripts):
            self.result_queue.put(Output(id, trans))
        t2 = time.time()
        dur = t2 - t1
        if debug:
            print ('Audio {} Infer {} Total {}'.format(t3-t1, t4-t3, t2-t1))
            if query_len > 1:
                print("#### rank {} finish {} sample in {:.3f} sec".format(self.rank, query_len, dur))
            else:
                print("#### rank {} finish sample of feature_len={} in {:.3f} sec".format(self.rank, feature_length[0].item(), dur))

        self.task_queue.task_done()
        return True

    def run(self):
        core_list = range(self.start_core, self.end_core + 1)
        num_cores = len(core_list)
        os.sched_setaffinity(self.pid, core_list)
        cmd = "taskset -p -c %d-%d %d" % (self.start_core, self.end_core, self.pid)
        print (cmd)
        os.system(cmd)
        os.environ['OMP_NUM_THREADS'] = '{}'.format(self.end_core-self.start_core+1)
        print("### set rank {} to cores [{}:{}]; omp num threads = {}"
            .format(self.rank, self.start_core, self.end_core, num_cores))

        torch.set_num_threads(num_cores)

        if not self.model_init:
            print("lazy_init rank {}".format(self.rank))
            config = toml.load(self.config_toml)
            dataset_vocab = config['labels']['labels']
            rnnt_vocab = add_blank_label(dataset_vocab)
            featurizer_config = config['input_eval']
            self.audio_preprocessor = AudioPreprocessing(**featurizer_config)
            self.audio_preprocessor.eval()
            self.audio_preprocessor = torch.jit.script(self.audio_preprocessor)
            self.audio_preprocessor = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(self.audio_preprocessor._c))

            model = RNNT(
                feature_config=featurizer_config,
                rnnt=config['rnnt'],
                num_classes=len(rnnt_vocab)
            )
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            migrated_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                key = key.replace("joint_net", "joint.net")
                migrated_state_dict[key] = value
            del migrated_state_dict["audio_preprocessor.featurizer.fb"]
            del migrated_state_dict["audio_preprocessor.featurizer.window"]
            model.load_state_dict(migrated_state_dict, strict=True)

            if self.ipex:
                import intel_pytorch_extension as ipex
                if self.bf16:
                    ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
                ipex.core.enable_auto_dnnl()
                model = model.to(ipex.DEVICE)

            model.eval()
            if not self.ipex:
                model.encoder = torch.jit.script(model.encoder)
                model.encoder = torch.jit._recursive.wrap_cpp_module(
                    torch._C._freeze_module(model.encoder._c))
                model.prediction = torch.jit.script(model.prediction)
                model.prediction = torch.jit._recursive.wrap_cpp_module(
                    torch._C._freeze_module(model.prediction._c))
            model.joint = torch.jit.script(model.joint)
            model.joint = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(model.joint._c))
            if not self.ipex:
                model = torch.jit.script(model)

            self.greedy_decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, model)

            self.model_init = True

        if self.warmup:
            self.do_warmup()

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        if self.rank == 0 and self.cosim:
            print ('Running with cosim mode, performance will be slow!!!')
        if self.rank == 0 and self.profile:
            print ('Start profiler')
            with profiler.profile(record_shapes=True) as prof:
                self.run_queue(debug=True)
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=40))
            print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=40))
            while self.run_queue():
                pass
        else:
            while self.run_queue():
                pass

def response_loadgen(out_queue):
    global finish_count
    out_queue_cnt = 0
    while True:
        next_task = out_queue.get()
        if next_task is None:
            print("Exiting response thread")
            break

        query_id = next_task.query_id
        transcript = next_task.transcript
        response_array = array.array('q', transcript)
        bi = response_array.buffer_info()
        response = lg.QuerySampleResponse(query_id, bi[0],
                                          bi[1] * response_array.itemsize)
        lg.QuerySamplesComplete([response])
        out_queue_cnt += 1
        finish_count += 1
        if debug:
            print("#### finish {} samples".format(finish_count))

    print("Finish processing {} samples".format(out_queue_cnt))


class PytorchSUT:
    def __init__(self, config_toml, checkpoint_path, dataset_dir,
                 manifest_filepath, perf_count, total_query_count, scenario, machine_conf, batch_size=1,
                 cores_for_loadgen=0, cores_per_instance=1, enable_debug=False,
                 cosim=False, profile=False, ipex=False, bf16=False, warmup=False):
        ### multi instance attributes
        self.batch_size = batch_size
        self.cores_for_loadgen = cores_for_loadgen
        self.cores_per_instance = cores_per_instance
        self.num_cores = get_num_cores()
        self.lock = mp.Lock()
        self.init_counter = mp.Value("i", 0)
        self.output_queue = mp.Queue()
        self.input_queue = mp.JoinableQueue()
        self.cosim = cosim
        self.ipex = ipex
        self.bf16 = bf16
        self.warmup = warmup
        self.scenario = scenario

        #server-specific
        self.num_queues = None       
        self.core_count_list = [] 
        self.num_instance_list = [] 
        self.seq_cutoff_list = []
        self.batch_size_list = []
        self.input_queue_list = []
        self.total_query_count = total_query_count

        if self.scenario == "Server": 
            # read config
            self.read_machine_conf(machine_conf) 
            # create queue list
            for _ in range(self.num_queues):
                self.input_queue_list.append(mp.JoinableQueue())

        config = toml.load(config_toml)

        dataset_vocab = config['labels']['labels']
        rnnt_vocab = add_blank_label(dataset_vocab)
        featurizer_config = config['input_eval']

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries,
                                   self.process_latencies)
        self.qsl = AudioQSLInMemory(dataset_dir,
                                    manifest_filepath,
                                    dataset_vocab,
                                    featurizer_config["sample_rate"],
                                    perf_count)
        
        if self.scenario == "Offline":
            self.issue_queue = InQueue(self.input_queue, batch_size)
        elif self.scenario == "Server":
            self.issue_queue = InQueueServer(self.input_queue_list, self.qsl, 
                                             self.seq_cutoff_list, self.batch_size_list,
                                             self.total_query_count)

        ### worker process
        self.consumers = []
        cur_core_idx = self.cores_for_loadgen
        rank = 0
        if self.scenario == "Offline":
            while cur_core_idx + self.cores_per_instance <= self.num_cores:
                self.consumers.append(
                        Consumer(self.input_queue, self.output_queue,
                                self.lock, self.init_counter, rank, cur_core_idx,
                                cur_core_idx+self.cores_per_instance-1, self.num_cores,
                                self.qsl, config_toml, checkpoint_path, dataset_dir, manifest_filepath,
                                perf_count, cosim, profile, ipex, bf16, warmup))
                rank += 1
                cur_core_idx += self.cores_per_instance
        elif self.scenario == "Server":
            for i in range(self.num_queues): 
                curr_cores_per_instance = self.core_count_list[i]             
                for _ in range(self.num_instance_list[i]):  
                    
                    self.consumers.append(
                        Consumer(self.input_queue_list[i], self.output_queue,
                                self.lock, self.init_counter, rank, cur_core_idx,
                                cur_core_idx + curr_cores_per_instance-1, self.num_cores,
                                self.qsl, config_toml, checkpoint_path, dataset_dir, manifest_filepath,
                                perf_count, cosim, profile, ipex, bf16, warmup))
                    rank += 1
                    cur_core_idx += curr_cores_per_instance
        self.num_instances = len(self.consumers)

        ### start worker process
        for c in self.consumers:
            c.start()

        ### wait until all sub processes are ready
        block_until(self.init_counter, self.num_instances, 2)

        ### start response thread
        self.response_worker = threading.Thread(
            target=response_loadgen, args=(self.output_queue,))
        self.response_worker.daemon = True
        self.response_worker.start()

        ### debug
        global debug
        debug = enable_debug


    def read_machine_conf(self, machine_conf):

        # machine conf format: core_per_instance, num_instances, seq_len_cutoff
        # assuming seq_len_cutoff in increasing order 
        infile = open(machine_conf, "r")
        data = infile.read().splitlines()
        
        self.num_queues = len(data) 
        for d in data:
            core_count, num_instance, cutoff, batch_size = map(int, d.split())
            self.core_count_list.append(core_count) 
            self.num_instance_list.append(num_instance)
            self.seq_cutoff_list.append(cutoff) 
            self.batch_size_list.append(batch_size)
        infile.close()
        #TO DO: validate config

    def issue_queries(self, query_samples):
        global start_time
        global query_count
        if self.batch_size != 1:
            ### make sure samples in the same batch are about the same length
            # qsl must be reversed sorted for best performance
            query_samples.sort(key=lambda k: self.qsl[k.index].shape[0], reverse=True)
        self.issue_queue.put(query_samples)
        end_time = time.time()
        dur = end_time - start_time
        start_time = end_time
        query_count += len(query_samples)
        if debug:
            print('\n#### issue {} samples in {:.3f} sec: total {} samples'.format(len(query_samples), dur, query_count))

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        print("Average latency (ms) per query:")
        print(np.mean(latencies_ns)/1000000.0)
        print("Median latency (ms): ")
        print(np.percentile(latencies_ns, 50)/1000000.0)
        print("90 percentile latency (ms): ")
        print(np.percentile(latencies_ns, 90)/1000000.0)

    def __del__(self):
        ### clear up sub processes

        if self.scenario == "Offline":
            self.input_queue.join()
            for i in range(self.num_instances):
                self.input_queue.put(None)

        elif self.scenario == "Server":

            for i in range(self.num_queues):
                self.input_queue_list[i].join()
                for _ in range(self.num_instance_list[i]): 
                    self.input_queue_list[i].put(None) 

        for c in self.consumers:
            c.join()
        self.output_queue.put(None)

        print("Finished destroying SUT.")
