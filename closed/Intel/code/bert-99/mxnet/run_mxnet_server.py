"""
This is a sample stub of loadgen with multiple processes support.
Each process sets its affinity by a proc list.

Loadgen is a producer, which calls issue_queries(). issue_queries() gets query
from loadgen and puts query id/sample indices into an input queue.

Each Consumer(process)'s run() reads input queue, calls model_predict() to get
inference result, and put result into output queue.

A standalone thread's response_loadgen() reads output queue, and responds
inference result to loadgen.

Server and Offline scenario PerformanceOnly mode are verified.

Each Model needs to implement below
model_predict()
load_query_samples()
unload_query_samples()

For model_predict(), how to return data to loadgen is model specific, the
loadgen CPP API requires a data pointer and length, then it saves the data to
mlperf_log_accuracy.json, which is used to generate accuracy number offline.
"""

import multiprocessing
import threading
import subprocess
import time
import os
import sys
import argparse
import array
import logging

import numpy as np
import mlperf_loadgen as lg
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("MXNet-BERT")

num_cpus = 28
num_ins = 2
NANO_SEC = 1e9
MILLI_SEC = 1000

in_queue_cnt = 0
out_queue_cnt = 0

bs_step = 8


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument("--batching", choices=["Fixed", "Dynamic", "Adaptive"], default="Adaptive", help="Batching method")
    parser.add_argument("--batch-size", default=1, type=int, help="batch_size")
    parser.add_argument("--num-instance", default=2, type=int, help="number of instance")
    parser.add_argument("--num-phy-cpus", default=28, type=int, help="number of physical cpus")
    parser.add_argument("--vocab", default='converted_from_tf_to_mxnet/tf.vocab',
                        type=str, help="vocab file path")
    parser.add_argument("--params", default='converted_from_tf_to_mxnet/tf_fp32.params',
                        type=str, help="FP32 params path")
    parser.add_argument("--quantized_model_prefix",
                        default='converted_from_tf_to_mxnet/quantized_models/model_bert_squad_quantized_customize',
                        type=str, help="quantized model prefix")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--quantized", action="store_true", help="use quantized model")
    parser.add_argument("--mlperf-conf", default="mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user-conf", default="user.conf", help="user rules config")
    parser.add_argument("--perf-count", default=None, help="perf count")
    parser.add_argument("--profile", action="store_true", help="whether enable profiler")
    parser.add_argument("--warmup", action="store_true", help="whether do warmup")
    parser.add_argument("--perf_calibrate", action="store_true", help="whether do performance calibration")
    args = parser.parse_args()
    return args

scenario_map = {
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def load_query_samples(sample_list):
    # This is model specific place holder
    pass


def unload_query_samples(sample_list):
    # This is model specific place holder
    pass


def block_until(counter, num_ins, t=1):
    while counter.value < num_ins:
        time.sleep(t)

batches = None

def load_perf_prof():
    global batches
    global throughputs
    # load performance profile map for offline scenario
    if os.path.exists("prof.py"):
        from prof import prof_map
        from prof import prof_bs_step
    else:
        prof_map = {}
        prof_bs_step = 1
        return
    longest_seq = 0
    for k, v in sorted(prof_map.items()):
        if k > longest_seq:
            longest_seq = k
    batches = [0.0] * (longest_seq+1)
    throughputs = [0.0] * (longest_seq+1)
    for k, v in sorted(prof_map.items()):
        max_throughput = 0.0
        max_bs = 0
        for i in range(1, len(v)):
            current_bs = i * prof_bs_step
            if current_bs/v[i] > max_throughput:
                max_throughput = current_bs/v[i]
                max_bs = current_bs
        batches[k] = max_bs
        throughputs[k] = max_throughput

def get_best_bs(seq_len):
    global batches
    if batches == None:
        load_perf_prof()
    global throughputs
    while batches[seq_len] == 0:
        seq_len += 1
    best_seq_len = seq_len
    best_bs = batches[seq_len]
    best_throughput = throughputs[seq_len]
    seq_len += 1
    while seq_len < 385:
        if throughputs[seq_len] > best_throughput:
            best_seq_len = seq_len
            best_bs = batches[seq_len]
            best_throughput = throughputs[seq_len]
        seq_len += 1
    return best_seq_len, best_bs, best_throughput

class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue, lock, init_counter, calibrate_counter, proc_idx, world_size, args, max_pad_len=384):
        multiprocessing.Process.__init__(self)
        global num_ins
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.lock = lock
        self.init_counter = init_counter
        self.calibrate_counter = calibrate_counter
        self.proc_idx = proc_idx
        self.world_size = world_size
        self.args = args
        self.affinity = range(round(proc_idx * num_cpus / num_ins),
                              round((proc_idx + 1) * num_cpus / num_ins))
        self.start_core_idx = proc_idx * num_cpus // num_ins
        self.end_core_idx = (proc_idx + 1) * num_cpus // num_ins - 1
        self.length_list = {}
        self.length_time_list = {}

        self.max_pad_len = max_pad_len

    def warmup(self, model, data_set, context, scenario):
        if self.proc_idx == 0:
            print ('Start warmup...')
        data_size = len(data_set.eval_features)
        count = 0
        import mxnet as mx

        for start in range(0, data_size):
            inputs_list = []
            token_types_list = []
            valid_length_list = []
            eval_feature = data_set.eval_features[start]
            _, inputs, token_types, valid_length, _, _ = eval_feature
            if len(inputs) in self.length_list:
                continue
            self.length_list[len(inputs)] = True
            max_throughput = 0.0
            best_bs = 0

            if scenario == 'Offline':
                # only support warmup of adaptive batching
                best_len, best_bs, _ = get_best_bs(len(inputs))
                if best_len in self.length_list:
                    continue
                self.length_list[best_len] = True
                inputs += [0] * (best_len - len(inputs))
                token_types += [0] * (best_len - len(token_types))
                for i in range(best_bs):
                    inputs_list.append(inputs)
                    token_types_list.append(token_types)
                    valid_length_list.append(valid_length)
                if self.proc_idx == 0:
                    print ("warmup seqlen {} batchsize {}".format(best_len, best_bs))
            else:
                inputs_list.append(inputs)
                token_types_list.append(token_types)
                valid_length_list.append(valid_length)

            inputs_nd = mx.nd.array(inputs_list).as_in_context(context)
            token_types_nd = mx.nd.array(token_types_list).as_in_context(context)
            valid_length_nd = mx.nd.array(valid_length_list).as_in_context(context).astype('float32')

            # warm up primitive once
            out = model.net(inputs_nd, token_types_nd, valid_length_nd)
            out_np = out.asnumpy()

            count += 1
            if count % 10 == 0 and self.proc_idx == 0:
                print ('Warmup {} samples'.format(count))
        if self.proc_idx == 0:
            print ('Warmup done')

    def calibrate(self, model, data_set, context):
        if self.proc_idx == 0:
            print ('Start calibration...')
        data_size = len(data_set.eval_features)
        count = 0
        global bs_step
        import mxnet as mx

        for start in range(0, data_size):
            inputs_list = []
            token_types_list = []
            valid_length_list = []
            eval_feature = data_set.eval_features[start]
            _, inputs, token_types, valid_length, _, _ = eval_feature
            cur_len = len(inputs)
            if cur_len in self.length_list:
                continue
            self.length_list[cur_len] = True
            if count % self.world_size != self.proc_idx:
                count += 1
                continue
            count += 1
            length_time_list = []
            length_time_list.append(0)
            max_throughput = 0.0
            best_bs = 0
            max_len = len(inputs)
            while True:
                for i in range(bs_step):
                    inputs_list.append(inputs)
                    token_types_list.append(token_types)
                    valid_length_list.append(valid_length)

                inputs_nd = mx.nd.array(inputs_list).as_in_context(context)
                token_types_nd = mx.nd.array(token_types_list).as_in_context(context)
                valid_length_nd = mx.nd.array(valid_length_list).as_in_context(context).astype('float32')

                # warm up primitive once
                out = model.net(inputs_nd, token_types_nd, valid_length_nd)
                out_np = out.asnumpy()

                # measure time for the batch
                t0 = time.time()
                for i in range(8):
                    out = model.net(inputs_nd, token_types_nd, valid_length_nd)
                    out_np = out.asnumpy()
                t1 = time.time()
                duration = (t1 - t0)/8.0
                throughput = len(inputs_list)/duration
                if throughput > max_throughput:
                    max_throughput = throughput
                    best_bs = len(inputs_list)
                if len(inputs_list) >= 256:
                    print ("{} - Best efficiency for seq len {} is BS {} with seq/s {:.5}".format(
                            self.proc_idx, max_len, best_bs, max_throughput))
                    break
                #print ("{} - Best efficiency for seq len {} is BS {} with seq/s {:.5}, current BS {} seq/s {:.5}\r".format(
                #        self.proc_idx, max_len, best_bs, max_throughput, len(inputs_list), throughput), end='')
                length_time_list.append(duration)
            self.length_time_list[cur_len] = length_time_list
        with open('prof_new.py', 'a') as f:
            for k, v in sorted(self.length_time_list.items()):
                print ('    {} : {},'.format(k, v), file=f)
        # keep the processor hot until all instance done calibration
        print ('Calibrate almost done, keep instance hot')
        self.lock.acquire()
        self.calibrate_counter.value += 1
        self.lock.release()
        while self.calibrate_counter.value < 2 * self.world_size:
            out = model.net(inputs_nd, token_types_nd, valid_length_nd)
            out_np = out.asnumpy()
        print ('Calibrate done')

    def run(self):
        global batching
        #os.sched_setaffinity(self.pid, self.affinity)
        cmd = "taskset -p -c %d-%d %d" % (self.start_core_idx, self.end_core_idx, self.pid)
        print (cmd)
        os.system(cmd)
        import mxnet as mx
        ctx = mx.cpu()
        #from numexpr.utils import set_num_threads
        #set_num_threads(28)
        os.environ['OMP_NUM_THREADS'] = '{}'.format(self.end_core_idx-self.start_core_idx+1)

        model = BERTModel(mx.cpu(), self.args.vocab, self.args.params,
                          self.args.quantized, self.args.quantized_model_prefix)
        data_set = BERTDataSet(self.args.vocab, self.args.perf_count)

        self.lock.acquire()
        self.calibrate_counter.value += 1
        self.lock.release()
        block_until(self.calibrate_counter, self.world_size)
        if self.args.perf_calibrate:
            self.calibrate(model, data_set, ctx)
            return

        self.lock.acquire()
        self.calibrate_counter.value += 1
        self.lock.release()
        if self.args.warmup:
            self.warmup(model, data_set, ctx, self.args.scenario)

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        #affinity = os.sched_getaffinity(self.pid)
        #print('Process', self.pid, 'affinity proc list:', affinity)
        cur_step = 0
        start_step = 384
        end_step = -1
        from utils import profile

        while True:
            next_task = self.task_queue.get() #(self.proc_idx)
            if next_task is None:
                # None means shutdown
                log.info('Exiting {}-pid:{}, cur_step={}'.format(self.name, self.pid, cur_step))
                self.task_queue.task_done()
                if self.args.profile and self.proc_idx==0:
                    if end_step == -1:
                        end_step = cur_step
                    profile(cur_step, start_step, end_step, profile_name='profile_{}.json'.format(self.pid), early_exit=False)
                break

            query_id_list = next_task.query_id_list
            sample_index_list = next_task.sample_index_list
            batch_size = len(sample_index_list)
            #print ('pid-{}, query_id_list: {}, sample_index_list: {}'.format(self.pid, query_id_list, sample_index_list))
            inputs_list = []
            token_types_list = []
            valid_length_list = []
            for sample_index in sample_index_list:
                eval_feature = data_set.eval_features[sample_index]
                _, inputs, token_types, valid_length, _, _ = eval_feature
                inputs_list.append(inputs)
                token_types_list.append(token_types)
                valid_length_list.append(valid_length)


            if len(inputs_list) > 1:
                max_len = max([len(inp) for inp in inputs_list])
                new_max_len, bs, best_throughput = get_best_bs(max_len)
                if bs == len(inputs_list):
                    max_len = new_max_len
                #for i in range(len(inputs_list)):
                #    inputs_list[i] += [0] * (max_len - len(inputs_list[i]))
                #    token_types_list[i] += [0] * (max_len - len(token_types_list[i]))
            else:
                max_len = self.max_pad_len #len(inputs_list[0]) #self.max_pad_len #len(inputs_list)

            for i in range(len(inputs_list)):
                inputs_list[i] += [0] * (max_len - len(inputs_list[i]))
                token_types_list[i] += [0] * (max_len - len(token_types_list[i]))

            inputs = mx.nd.array(inputs_list).as_in_context(ctx)
            token_types = mx.nd.array(token_types_list).as_in_context(ctx)
            valid_length = mx.nd.array(valid_length_list).as_in_context(ctx).astype('float32')

            if self.args.profile and self.proc_idx==0:
                profile(cur_step, start_step, end_step, profile_name='profile_{}.json'.format(self.pid), early_exit=False)
                cur_step += 1
            #t0 = time.time()
            out = model.net(inputs, token_types, valid_length)
            out_np = out.asnumpy()
            #t1 = time.time()
            #if self.proc_idx == 0:
            #    cur_throughput = len(inputs_list)/(t1-t0)
            #    if best_throughput != 0:
            #        throughput_diff = (cur_throughput - best_throughput) / best_throughput
            #        print ('inference seq len = {} BS = {} throughput = {:.5f} ({:.3f}%)'.format(max_len, len(inputs_list), cur_throughput, throughput_diff*100))
            #    else:
            #        print ('inference seq len = {} BS = {} throughput = {:.5f})'.format(max_len, len(inputs_list), cur_throughput))
            result = Output(query_id_list, out_np)
            self.result_queue.put(result)
            #print('consumer-{}: output.shape={}, query_id={}'.format(self.pid, out_np.shape, query_id_list[0]))
            self.task_queue.task_done()


class Input(object):
    def __init__(self, id_list, index_list, sample_length_list):
        assert isinstance(id_list, list)
        assert isinstance(index_list, list)
        assert isinstance(sample_length_list, list)
        assert len(id_list) == len(index_list)
        self.query_id_list = id_list
        self.sample_index_list = index_list
        self.sample_length_list = sample_length_list


class Output(object):
    def __init__(self, query_id_list, result):
        self.query_id_list = query_id_list
        self.result = result


class InQueue():
    def __init__(self, in_queue, batch_size, data_set):
        from preprocessing_utils import max_seq_length
        self.in_queue = in_queue
        self.batch_size = batch_size
        self.query_id_list = []
        self.sample_index_list = []
        self.sample_length_list = []
        self.index = 0
        self.data_set = data_set
        self.max_seq_len = max_seq_length

    def put(self, query_samples):
        global in_queue_cnt
        ##TODO, debug
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        query_len = len(query_samples)

        num_samples = len(query_samples)
        def idx_len(e):
            idx = e.index
            feature = self.data_set.eval_features[idx]
            _, inputs, _, _, _, _ = feature
            return len(inputs)


        if num_samples == 1:
            if self.batch_size == 1:
                in_queue_cnt += 1
                self.in_queue.put(Input([query_samples[0].id],
                                        [query_samples[0].index],
                                        [idx_len(query_samples[0])]))
            else:
                self.index += 1
                if self.index < self.batch_size:
                    self.query_id_list.append(query_samples[0].id)
                    self.sample_index_list.append(query_samples[0].index)
                    self.sample_length_list.append(idx_len(query_samples[0]))
                else:
                    self.query_id_list.append(query_samples[0].id)
                    self.sample_index_list.append(query_samples[0].index)
                    self.sample_length_list.append(idx_len(query_samples[0]))
                    self.in_queue.put(Input(self.query_id_list, self.sample_index_list, self.sample_length_list))
                    in_queue_cnt += self.batch_size
                    self.index = 0
                    self.query_id_list = []
                    self.sample_index_list = []
                    self.sample_length_list = []
        else:

            query_samples.sort(key=idx_len, reverse=True)

            def enqueue_batch(cur_batch_size, base_index=0):
                global in_queue_cnt
                id_list = []
                index_list = []
                length_list = []
                for i in range(cur_batch_size):
                    id_list.append(query_samples[base_index + i].id)
                    index_list.append(query_samples[base_index + i].index)
                    length_list.append(idx_len(query_samples[base_index + i]))
                self.in_queue.put(Input(id_list, index_list, length_list))
                in_queue_cnt += cur_batch_size

            global batching
            true_total_len = 0
            total_len = 0
            for i in range(num_samples):
                true_total_len += idx_len(query_samples[i])
            if batching == 'Dynamic':
                batch_seq_len = self.batch_size * self.max_seq_len
                base_index = 0
                num_batches = 0
                while base_index < num_samples:
                    base_len = idx_len(query_samples[base_index])
                    for i in range(base_index, num_samples):
                        current_len = base_len * (i-base_index+1)
                        if i+1 < num_samples:
                            next_len = base_len * (i+1-base_index+1)
                            if next_len > batch_seq_len:
                              if next_len - batch_seq_len > batch_seq_len - current_len:
                                  next_index = i+1
                              else:
                                  next_index = i+2
                              break
                        else:
                            next_index = i+1
                            break
                    total_len += base_len * (next_index-base_index)
                    enqueue_batch(next_index-base_index, base_index)
                    num_batches += 1
                    #print('pid-{2}: enqueue bs={0} and input volume {1}...'
                    #    .format(next_index-base_index, current_len, os.getpid()))
                    base_index = next_index
                print('pid-{1}: enqueued {0} batches, pad ratio = {2}%'
                    .format(num_batches, os.getpid(), (total_len-true_total_len)*100/true_total_len))
            elif batching == 'Adaptive':
                batch_seq_len = self.batch_size * self.max_seq_len
                base_index = 0
                num_batches = 0
                while base_index < num_samples:
                    base_len = idx_len(query_samples[base_index])
                    best_len, best_bs, _ = get_best_bs(base_len)
                    next_index = base_index + best_bs
                    if next_index > num_samples:
                        next_index = num_samples
                    total_len += base_len * (next_index-base_index)
                    enqueue_batch(next_index-base_index, base_index)
                    num_batches += 1
                    #print('pid-{2}: enqueue bs={0} and input volume {1}...'
                    #    .format(next_index-base_index, current_len, os.getpid()))
                    base_index = next_index
                print('pid-{1}: enqueued {0} batches, pad ratio = {2}%'
                    .format(num_batches, os.getpid(), (total_len-true_total_len)*100/true_total_len))
            else:
                num_batch = num_samples // self.batch_size
                remaining_batch = num_samples % self.batch_size
                ## TODO, remove
                print('pid-{3}: split the datasets into {0} batches with bs={1} and remaining {2}...'
                    .format(num_batch, self.batch_size, remaining_batch, os.getpid()))

                for b in range(num_batch):
                    base_index = b * self.batch_size
                    enqueue_batch(self.batch_size, base_index)

                if remaining_batch > 0:
                    base_index = num_batch * self.batch_size
                    enqueue_batch(remaining_batch, base_index)
        #print ('in_queue_cnt=', in_queue_cnt)


class InQueueServer():
    def __init__(self, in_queue, batch_sizes, data_set, expected_total_queries):
        from preprocessing_utils import max_seq_length
        self.in_queues = in_queue
        self.batch_sizes = batch_sizes
        self.query_id_lists = defaultdict(list)
        self.sample_index_lists = defaultdict(list)
        self.indexes = defaultdict(int)
        self.sample_length_lists = defaultdict(list)

        self.data_set = data_set
        self.max_seq_len = max_seq_length

        self.num_buckets = len(in_queue)
        self.cutoffs = sorted(list(batch_sizes.keys()))
        self.expected_total_queries = expected_total_queries

        self.batch_sizes = defaultdict(int)

    def getQueryBucket(self, query_len):
        end = 0
        while end < self.num_buckets and query_len > self.cutoffs[end]:
            end += 1

        return self.cutoffs[end]


    def getQuerySampleLength(self, query ):
        idx = query.index
        return len( self.data_set.eval_features[idx][1] ) # input sequence is the 2nd attribute per ex.

    def put(self, query_samples):
        global in_queue_cnt
        global queries_so_far # Track no. of queries received from loadgen

        ##TODO, debug
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        query_len = len(query_samples)

        num_samples = len(query_samples)
        if num_samples == 1:

            # Use length of the query sample to determine the queue it should be put
            q_length = self.getQuerySampleLength( query_samples[0] )
            bucket = self.getQueryBucket( q_length )

            if self.batch_sizes[bucket] == 1:
                in_queue_cnt += 1
                self.in_queues[bucket].put(Input([query_samples[0].id], [query_samples[0].index], [q_len]))
            else:
                self.indexes[bucket] += 1
                if self.indexes[bucket] < self.batch_sizes[bucket]:
                    self.query_id_lists[bucket].append(query_samples[0].id)
                    self.sample_index_lists[bucket].append(query_samples[0].index)
                    self.sample_length__lists[bucket].append(q_length)
                else:
                    self.query_id_lists[bucket].append(query_samples[0].id)
                    self.sample_index_lists[bucket].append(query_samples[0].index)
                    self.sample_length_lists[bucket].append(q_length)
                    self.in_queues[bucket].put(Input(self.query_id_lists[bucket], self.sample_index_lists[bucket], self.sample_length_lists[bucket]))

                    in_queue_cnt += self.batch_sizes[bucket]
                    self.indexes[bucket] = 0
                    self.query_id_lists[bucket] = []
                    self.sample_index_lists[bucket] = []
                    self.sample_length_lists[bucket] = []

        if queries_so_far == self.expected_total_queries:
            for bucket in self.in_queues:
                query_id_list = self.query_id_lists[bucket]
                sample_index_list = self.sample_index_lists[bucket]
                sample_length_list = self.sample_length_lists[bucket]

                for j, q_id in enumerate(query_id_list):
                    s_idx = sample_index_list[j]
                    s_len = sample_length_list[j]

                    self.in_queues[bucket].put(Input([q_id], [s_idx], [s_len]))
                    in_queue_cnt += 1


def flush_queries():
    pass


def process_latencies(latencies_ns):
    # It's called by loadgen to show us the recorded latencies
    log.info("Average latency (ms) per query:")
    log.info(np.mean(latencies_ns)/1000000.0)
    log.info("Median latency (ms): ")
    log.info(np.percentile(latencies_ns, 50)/1000000.0)
    log.info("90 percentile latency (ms): ")
    log.info(np.percentile(latencies_ns, 90)/1000000.0)


def response_loadgen(out_queue):
    global out_queue_cnt
    while True:
        next_task = out_queue.get()
        if next_task is None:
            # None means shutdown
            log.info('Exiting response thread')
            break
        query_id_list = next_task.query_id_list
        result = next_task.result

        batch_size = len(query_id_list)
        result.reshape(batch_size, -1, 2)

        out_list = np.split(result, batch_size, axis=0)
        #responses = []
        for i, o in enumerate(out_list):
            response_array = array.array("B", np.array(o).astype(np.float32).tobytes())
            bi = response_array.buffer_info()
            #responses.append(lg.QuerySampleResponse(query_id_list[i], bi[0], bi[1]))
            responses = [lg.QuerySampleResponse(query_id_list[i], bi[0], bi[1])]
            out_queue_cnt += 1
            #print('Response loadgen ({}), query_id {}, out_queue_cnt {}'.format(os.getpid(), query_id_list[i], out_queue_cnt))
            lg.QuerySamplesComplete(responses)
        #lg.QuerySamplesComplete(responses)


class BERTModel():
    def __init__(self, ctx, mx_vocab, params, quantized, quantized_model_prefix):
        import gluonnlp as nlp
        from utils import BertForQA
        import mxnet as mx
        if quantized:
            log.info('Loading quantized MXNet model...')
            self.net = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(quantized_model_prefix),
                                                    ['data0', 'data1', 'data2'],
                                                    '{}-0000.params'.format(quantized_model_prefix))
            self.net.hybridize(static_alloc=True, static_shape=True)
        else:
            log.info('Loading MXNet model...')
            with open(mx_vocab, 'r') as f:
                vocab = nlp.vocab.BERTVocab.from_json(f.read())

            bert, vocab = nlp.model.get_model(
                name='bert_24_1024_16',
                dataset_name=None,
                vocab=vocab,
                pretrained=False,
                ctx=ctx,
                use_pooler=False,
                use_decoder=False,
                use_classifier=False)
            self.net = BertForQA(bert=bert)
            nlp.utils.load_parameters(self.net, params, ctx=ctx, cast_dtype=True)
            self.net.hybridize(static_alloc=True)


class BERTDataSet():
    def __init__(self, mx_vocab, perf_count):
        import gluonnlp as nlp
        from preprocessing_utils import preprocess_dataset, max_seq_length, max_query_length, doc_stride
        from gluonnlp.data import SQuAD

        eval_features = []
        with open(mx_vocab, 'r') as f:
            vocab = nlp.vocab.BERTVocab.from_json(f.read())
        log.info("Creating tokenizer...")
        tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=True)

        round_to = None
        log.info("Reading examples...")
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


class MultiprocessShapeBasedQueue(object):
    def __init__(self):
        global num_ins
        self._jq = multiprocessing.JoinableQueue()
        self._instances_queue = [multiprocessing.Queue() for _ in range(num_ins)]
        self._manager = multiprocessing.Manager()
        self.shape_in_instance = self._manager.dict()
        self.finish_status = self._manager.dict()

    def get(self, instance_id=0):
        return self._jq.get()
        # with multiprocessing.Lock():
        #     if self._instances_queue[instance_id].empty():
        #         while True:
        #             item = self._jq.get()
        #             if item != None:
        #                 sample_length = item.sample_length_list[0]
        #                 batch_size = len(item.sample_index_list)
        #                 key = (batch_size, sample_length)

        #                 if key in self.shape_in_instance.keys():
        #                     if self.shape_in_instance[key] == instance_id:
        #                         return item
        #                     else:
        #                         target_instance = self.shape_in_instance[key]
        #                         if target_instance in self.finish_status.keys():
        #                             # target instance already finished execution - get item
        #                             del shape_in_instance[key]
        #                             return item
        #                         else:
        #                             self._instances_queue[target_instance].put(item)
        #                             # reapeat while loop - get new item and check if it's suitable for instance
        #                 else:
        #                     # mark shape with current instance
        #                     self.shape_in_instance[key] = instance_id
        #                     return item
        #             else:
        #                 self.finish_status[instance_id] = True
        #                 return item # return None
        #     else:
        #         item = self._instances_queue[instance_id].get()
        #         return item

    def put(self, obj, block=True, timeout=None):
        return self._jq.put(obj, block, timeout)
        ##print("end put")

    def task_done(self):
        #print("task_done")
        return self._jq.task_done()
        #print("end task_done")

    def join(self):
        #print("join")
        return self._jq.join()
        #print("end join")


def main():
    global num_ins
    global num_cpus
    global in_queue_cnt
    global out_queue_cnt
    global batching
    global queries_so_far
    global Latencies

    queries_so_far = 0

    args = get_args()
    log.info(args)
    scenario = args.scenario
    accuracy_mode = args.accuracy
    perf_count = args.perf_count
    batch_size = args.batch_size
    num_ins = args.num_instance
    num_cpus = args.num_phy_cpus
    batching = args.batching

    # Read Loadgen and workload config parameters
    settings = lg.TestSettings()
    settings.scenario = scenario_map[scenario]
    settings.FromConfig(args.mlperf_conf, "bert", scenario)
    settings.FromConfig(args.user_conf, "bert", scenario)
    settings.mode = lg.TestMode.AccuracyOnly if accuracy_mode else lg.TestMode.PerformanceOnly

    # Establish communication queues
    lock = multiprocessing.Lock()
    init_counter = multiprocessing.Value("i", 0)
    calibrate_counter = multiprocessing.Value("i", 0)
    out_queue = multiprocessing.Queue()

    # Create consumers
    consumers = []
    if scenario == "Server":
        from parse_server_config import configParser

        buckets = configParser( "machine_conf.json")
        cutoffs = list(buckets.keys())
        batch_sizes = {}

        in_queue = {j: multiprocessing.JoinableQueue() for j in buckets}
        proc_idx = 0
        num_cpus = 0
        total_ins = 0
        for cutoff in list(buckets.keys()):
            batch_sizes[ cutoff ] = buckets[ cutoff ]["batch_size"]
            num_ins = buckets[ cutoff ]["instances"]
            cpus_per_instance = buckets[ cutoff ]["cpus_per_instance"]
            num_cpus = num_ins * cpus_per_instance
            total_ins += num_ins


            for j in range(num_ins):
                consumer = Consumer( in_queue[ cutoff ], out_queue, lock, init_counter, calibrate_counter, proc_idx, num_ins, args, cutoff)
                consumer.start_core_idx = proc_idx
                consumer.end_core_idx = proc_idx + cpus_per_instance - 1
                consumers.append(consumer)
                proc_idx = consumer.end_core_idx + 1

        num_ins = total_ins


    else:
        total_ins = num_ins
        in_queue = MultiprocessShapeBasedQueue()
        consumers = [Consumer(in_queue, out_queue, lock, init_counter, calibrate_counter, i, num_ins, args)
                 for i in range(num_ins)]


    for c in consumers:
        c.start()


    # Dataset object used by constructQSL
    data_set = BERTDataSet(args.vocab, args.perf_count)
    if scenario=="Server":
        issue_queue = InQueueServer(in_queue, batch_sizes, data_set, settings.min_query_count)
    else:
        issue_queue = InQueue(in_queue, batch_size, data_set)

    # Wait until all sub-processors are ready
    block_until(init_counter, total_ins, 2)

    # Start response thread
    response_worker = threading.Thread(
        target=response_loadgen, args=(out_queue,))
    response_worker.daemon = True
    response_worker.start()

    def issue_queries(query_samples):
        # It's called by loadgen to send query to SUT
        issue_queue.put(query_samples)

    sut = lg.ConstructSUT(
        issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(
        data_set.count, data_set.perf_count, load_query_samples, unload_query_samples)

    log_path = "build/logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)

    # Wait until outQueue done
    while out_queue_cnt < in_queue_cnt:
        time.sleep(0.2)

    if scenario == "Server":
        for i in in_queue:
            in_queue[i].join()
            for j in  range(buckets[ i ]["cpus_per_instance"]):
                in_queue[i].put(None)
    else:
        for i in range(num_ins):
            in_queue.put(None)

    for c in consumers:
        c.join()
    out_queue.put(None)


    if accuracy_mode:
        cmd = "python accuracy-squad.py --log_file={}/mlperf_log_accuracy.json".format(log_path)
        subprocess.check_call(cmd, shell=True)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


if __name__ == '__main__':
    main()
