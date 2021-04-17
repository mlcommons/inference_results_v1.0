"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import json
import logging
import os
import sys
import multiprocessing
import threading
import time
import collections

import mlperf_loadgen as lg
import numpy as np
from shutil import copyfile 

# add dlrm code path
try:
    dlrm_dir_path = os.environ['DLRM_DIR']
    sys.path.append(dlrm_dir_path)
except KeyError:
    print("ERROR: Please set DLRM_DIR environment variable to the dlrm code location")
    sys.exit(0)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

num_sockets = int(os.getenv('NUM_SOCKETS', 8))
cpus_per_socket = int(os.getenv('CPUS_PER_SOCKET', 28))
cpus_per_instance = int(os.getenv('CPUS_PER_INSTANCE', 14))
cpus_per_process = int(os.getenv('CPUS_PER_PROCESS', 28))
procs_per_socket = cpus_per_socket // cpus_per_process
total_procs = num_sockets * procs_per_socket

NANO_SEC = 1e9
MILLI_SEC = 1000

# pylint: disable=missing-docstring

# the datasets we support
DATASETS_KEYS = ["kaggle", "terabyte"]

# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "terabyte",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
    "dlrm-kaggle-pytorch": {
        "dataset": "kaggle",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 128,
    },
    "dlrm-terabyte-pytorch": {
        "dataset": "terabyte",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
}

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

qcount=0
start_time = 0
item_good = 0
item_total = 0
total_instances = 0
item_timing = []
item_results = []
last_timeing = []


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="name of the mlperf model, ie. dlrm")
    parser.add_argument("--model-path", required=True, help="path to the model file")
    parser.add_argument("--dataset", choices=DATASETS_KEYS, help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles")
    parser.add_argument("--enable-profiling", type=bool, default=False, help="enable pytorch profiling")
    parser.add_argument("--scenario", default="SingleStream",
                        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())))
    parser.add_argument("--test-num-workers", type=int, default=0, help='# of workers reading the data')
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--max-batchsize", type=int, default=2048, help="max batch size in a single inference")
    parser.add_argument("--max-sample-size", type=int, help="max combined samples in a single inference")
    parser.add_argument("--output", help="test results")
    parser.add_argument("--inputs", help="model inputs (currently not used)")
    parser.add_argument("--outputs", help="model outputs (currently not used)")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--use-ipex", action="store_true", default=False)
    parser.add_argument("--threads", default=1, type=int, help="threads")
    parser.add_argument("--cache", type=int, default=0, help="use cache (currently not used)")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--find-peak-performance", action="store_true", help="enable finding peak performance pass")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--config", default="../mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user-config", default="./user.conf", help="mlperf rules user config")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--duration", type=int, help="duration in milliseconds (ms)")
    parser.add_argument("--target-qps", type=int, help="target/expected qps")
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--count-samples", type=int, help="dataset items to use")
    parser.add_argument("--count-queries", type=int, help="number of queries to use")
    parser.add_argument("--samples-per-query-multistream", type=int, help="query length for multi-stream scenario (in terms of aggregated samples)")
    # --samples-per-query-offline is equivalent to perf_sample_count
    parser.add_argument("--samples-per-query-offline", type=int, default=2048, help="query length for offline scenario (in terms of aggregated samples)")
    parser.add_argument("--samples-to-aggregate-fix", type=int, help="number of samples to be treated as one")
    parser.add_argument("--samples-to-aggregate-min", type=int, help="min number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-max", type=int, help="max number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-quantile-file", type=str, help="distribution quantile used to generate number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-trace-file", type=str, default="dlrm_trace_of_aggregated_samples.txt")
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.numpy_rand_seed)

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)
    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args

def get_dataset(args):
    import criteo
    SUPPORTED_DATASETS = {
        "kaggle":
            (criteo.Criteo, criteo.pre_process_criteo_dlrm, criteo.DlrmPostProcess(),
             {"randomize": 'total',  "memory_map": True}),
        "terabyte":
            (criteo.Criteo, criteo.pre_process_criteo_dlrm, criteo.DlrmPostProcess(),
             {"randomize": 'total',  "memory_map": True}),
    }

    # dataset to use
    wanted_dataset, pre_proc, _, kwargs = SUPPORTED_DATASETS[args.dataset]

    # --count-samples can be used to limit the number of samples used for testing
    ds = wanted_dataset(data_path=args.dataset_path,
                        name=args.dataset,
                        pre_process=pre_proc,  # currently an identity function
                        use_cache=args.cache,  # currently not used
                        count=args.count_samples,
                        samples_to_aggregate_fix=args.samples_to_aggregate_fix,
                        samples_to_aggregate_min=args.samples_to_aggregate_min,
                        samples_to_aggregate_max=args.samples_to_aggregate_max,
                        samples_to_aggregate_quantile_file=args.samples_to_aggregate_quantile_file,
                        samples_to_aggregate_trace_file=args.samples_to_aggregate_trace_file,
                        test_num_workers=0,
                        max_ind_range=args.max_ind_range,
                        sub_sample_rate=args.data_sub_sample_rate,
                        mlperf_bin_loader=args.mlperf_bin_loader,
                        **kwargs)
    return ds

def get_backend(backend, dataset, max_ind_range, data_sub_sample_rate, use_gpu, use_ipex):

    if backend == "pytorch-native":
        from backend_pytorch_native import BackendPytorchNative
        # NOTE: pass model parameters here, the following options are available
        if dataset == "kaggle":
            # 1. Criteo Kaggle Display Advertisement Challenge Dataset (see ./bench/dlrm_s_criteo_kaggle.sh)
            backend = BackendPytorchNative(
                m_spa=16,
                ln_emb=np.array([1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572]),
                ln_bot=np.array([13,512,256,64,16]),
                ln_top=np.array([367,512,256,1]),
                use_gpu=use_gpu
            )
        elif dataset == "terabyte":
            if max_ind_range == 10000000:
                # 2. Criteo Terabyte (see ./bench/dlrm_s_criteo_terabyte.sh [--sub-sample=0.875] --max-in-range=10000000)
                backend = BackendPytorchNative(
                    m_spa=64,
                    ln_emb=np.array([9980333,36084,17217,7378,20134,3,7112,1442,61, 9758201,1333352,313829,10,2208,11156,122,4,970,14, 9994222, 7267859, 9946608,415421,12420,101, 36]),
                    ln_bot=np.array([13,512,256,64]),
                    ln_top=np.array([415,512,512,256,1]),
                    use_gpu=use_gpu
                )
            elif max_ind_range == 40000000:
                # 3. Criteo Terabyte MLPerf training (see ./bench/run_and_time.sh --max-in-range=40000000)
                backend = BackendPytorchNative(
                    m_spa=128,
                    ln_emb=np.array([39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36]),
                    ln_bot=np.array([13,512,256,128]),
                    ln_top=np.array([479,1024,1024,512,256,1]),
                    use_gpu=use_gpu,
                    use_ipex=use_ipex,
                    server=True
                )
            else:
                raise ValueError("only --max-in-range 10M or 40M is supported")
        else:
            raise ValueError("only kaggle|terabyte dataset options are supported")

    else:
        raise ValueError("unknown backend: " + backend)
    return backend

class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_id):
        self.query_id = query_id
        self.content_id = content_id
        #self.issues = issues
        self.start = time.time()

class OItem:
    def __init__(self, presults, query_ids=None, array_ref=None, good=0, total=0, timing=0):
        self.good = good
        self.total = total
        self.timing = timing
        self.presults = presults
        self.query_ids = query_ids
        self.array_ref = array_ref

class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue, ds_queue, lock, init_counter, finished_samples, barrier, proc_num, args, min_query_count, first_instance_start_core):
        multiprocessing.Process.__init__(self)
        global cpus_per_instance
        global cpus_per_process
        self.args = args
        self.lock = lock
        self.barrier = barrier
        self.ds_queue = ds_queue
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.init_counter = init_counter
        self.finished_samples = finished_samples
        self.proc_num = proc_num 
        self.min_query_count = min_query_count

        self.workers = []
        self.instances_start_core = []
        self.instances_end_core = []
        #self.instances_affinity = []
        self.instances_core_nums = []
        self.instances_core_id = []

        socket_num = self.proc_num // procs_per_socket
        socket_proc_idx = self.proc_num % procs_per_socket
        self.start_core_idx = socket_num * cpus_per_socket + socket_proc_idx * cpus_per_process
        self.end_core_idx = self.start_core_idx + cpus_per_process
        if self.proc_num == 0:
            self.start_core_idx = self.start_core_idx + first_instance_start_core  #reserved threads for loadgen staff
        self.affinity = range(self.start_core_idx, self.end_core_idx)
        self.core_nums = self.end_core_idx - self.start_core_idx
        self.num_ins = self.core_nums // cpus_per_instance
        if self.proc_num == 0 and (first_instance_start_core > 0) and (cpus_per_instance - first_instance_start_core) >= (cpus_per_instance // 2):
            self.num_ins = self.num_ins + 1

        for i in range(self.num_ins):
            if self.proc_num == 0:
              if (cpus_per_instance - first_instance_start_core) >= (cpus_per_instance // 2):
                  if i == 0:
                      self.instances_start_core.append(first_instance_start_core)
                      self.instances_core_id.append(first_instance_start_core) 
                      self.instances_end_core.append(cpus_per_instance) 
                  else:
                      self.instances_start_core.append(i * cpus_per_instance)
                      self.instances_end_core.append(self.instances_start_core[i] + cpus_per_instance) 
                      self.instances_core_id.append(i * cpus_per_instance) 
              else:
                  self.instances_start_core.append((i + 1) * cpus_per_instance)
                  self.instances_end_core.append(self.instances_start_core[i] + cpus_per_instance) 
                  self.instances_core_id.append((i + 1) * cpus_per_instance) 
            else:
                  self.instances_start_core.append(self.start_core_idx + i * cpus_per_instance)
                  self.instances_end_core.append(self.instances_start_core[i] + cpus_per_instance) 
                  self.instances_core_id.append(i * cpus_per_instance) 
            #self.instances_affinity.append(range(self.instances_start_core[i], self.instances_end_core[i]))
            self.instances_core_nums.append(self.instances_end_core[i] - self.instances_start_core[i])

    def warmup(self, model):
        import torch
        for _ in range(2):
            #for s in range(100, self.args.max_sample_size * 700 + 100, 100):
            for s in range(self.args.max_batchsize - 700, self.args.max_batchsize + 100, 100):
                batch_dense_X = torch.randn((s, 13), dtype=torch.float)
                batch_lS_i = torch.ones([26, s], dtype=torch.long)
                batch_lS_o = torch.stack([torch.arange(s) for _ in range(26)])
                model.predict(batch_dense_X, batch_lS_o, batch_lS_i)

    # def trace(self, ds, model):
    #     ds.load_query_samples([0])
    #     batch_dense_X, batch_lS_o, batch_lS_i, batch_T, idx_offsets = ds.get_samples([0])
    #     model.trace(batch_dense_X, batch_lS_o, batch_lS_i)
    #     ds.unload_query_samples(None)

    def handle_tasks(self, i, ds_queue, task_queue, result_queue, args, pid):
        import torch
        global cpus_per_process
        #os.sched_setaffinity(self.workers[i].pid, self.instances_affinity[i])
        from intel_pytorch_extension import core
        core.thread_bind(self.proc_num, cpus_per_process, self.instances_core_id[i], self.instances_core_nums[i])

        ds = get_dataset(self.args)
        sample_list = ds_queue.get()
        ds.load_query_samples(sample_list)

        if args.enable_profiling:
            filename = "dlrm_mlperf_server_run_" + str(pid) + "_" + str(i) + ".prof"

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        with torch.autograd.profiler.profile(args.enable_profiling, args.use_gpu) as prof:
          while True:
            qitem = task_queue.get()
            if qitem is None:
                #task_queue.task_done()
                break
            
            #get_sample_start = time.time()
            batch_dense_X, batch_lS_o, batch_lS_i, batch_T, idx_offsets = ds.get_samples(qitem.content_id)
            #get_sample_timing = time.time() - get_sample_start
            #print("DS get_samples elapsed time:{} ms ".format(get_sample_timing * 1000))

            good = 0
            total = 0
            presults = []
            result_timing = 0
            try:
                #predict_start = time.time()
                results = self.model.predict(batch_dense_X, batch_lS_o, batch_lS_i)
                #predict_timing = time.time() - predict_start
                #print("batch size = {}, predict elapsed time:{} ms".format(len(batch_dense_X), predict_timing * 1000))
                # post_process
                results = results.detach().cpu()
                total = len(results)
                good = (results.round() == batch_T).nonzero().size(0)
                presults = torch.cat((results, batch_T), dim=1)

                if args.accuracy:
                    result_timing = time.time() - qitem.start

            except Exception as ex:  # pylint: disable=broad-except
                log.error("thread: failed, %s", ex)
                presults = [[]] * len(qitem.query_id)
            finally:
                response_array_refs = []
                for idx, query_id in enumerate(qitem.query_id):
                    # NOTE: processed_results returned by DlrmPostProcess store both
                    # result = processed_results[idx][0] and target = processed_results[idx][1]
                    # also each idx might be a query of samples, rather than a single sample
                    # depending on the --samples-to-aggregate* arguments.
                    s_idx = idx_offsets[idx]
                    e_idx = idx_offsets[idx + 1]
                    # debug prints
                    # print("s,e:",s_idx,e_idx, len(processed_results))
                    response_array = array.array("B", np.array(presults[s_idx:e_idx], np.float32).tobytes())
                    response_array_refs.append(response_array)
                if args.accuracy:
                    result_queue.put(OItem(np.array(presults, np.float32), qitem.query_id, response_array_refs, good, total, result_timing))
                else:
                    result_queue.put(OItem([], qitem.query_id, response_array_refs, good, total, result_timing))

        ds.unload_query_samples(None)
        if args.enable_profiling:
            with open(filename, "w") as prof_f:
                prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))

    def run(self):
        os.sched_setaffinity(self.pid, self.affinity)
        import torch
        import torch.multiprocessing as mp
        torch.set_num_threads(self.core_nums)

        backend = get_backend(self.args.backend, self.args.dataset, self.args.max_ind_range,
                              self.args.data_sub_sample_rate, self.args.use_gpu, self.args.use_ipex)
        self.model = backend.load(self.args.model_path, self.args.inputs, self.args.outputs)
        self.warmup(self.model)

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        if self.num_ins > 1 :
            for i in range(self.num_ins):
                worker = mp.Process(target=self.handle_tasks, args=(i, self.ds_queue, self.task_queue, self.result_queue, self.args, self.pid))
                self.workers.append(worker)
            for w in self.workers:
                w.start()
            for w in self.workers:
                w.join()
        else:
            self.handle_tasks(0, self.ds_queue, self.task_queue, self.result_queue, self.args, self.pid)

class QueueRunner:
    def __init__(self, inQueue, dataset, sample_num, max_sample_size, max_batchsize):
        self.inQueue = inQueue
        self.max_sample_size = max_sample_size
        self.max_batchsize = max_batchsize
        self.ds = dataset
        self.sample_list = range(0, sample_num)
        self.ds.load_query_samples(self.sample_list)
        self.issues = 0
        self.sample_len = 0
        self.queue_id = 0
        self.num_queues = 4#len(self.inQueue)
        self.ids = [[] for i in range(self.num_queues)]
        self.idxes = [[] for i in range(self.num_queues)]
        self.qsize = [0 for i in range(self.num_queues)]
        self.qwaiting = [0 for i in range(self.num_queues)]
        self.qwaiting_flag = False
        self.qorder = collections.deque(range(0, self.num_queues))
        self.qorder.rotate(-1)

    def issue(self, q, query_id, idx):
        # print("Send in queue", q, " with total samples ", len(query_id), " qsize ", self.qsize[q])
        self.inQueue.put(Item(query_id, idx))
        self.ids[q] = []
        self.idxes[q] = []
        self.qsize[q] = 0
        self.qwaiting[q] = 0

    def find_waiting_q(self, sample_len):
        for q in self.qorder:
            if sample_len <= self.qwaiting[q]:
                return q
        return -1

    # Fill query buffers to create constant batch
    def enqueue(self, query_id, idx):
        global qcount
        increment_q = False
        sample_len = self.ds.get_sample_length(idx[0])
        # print("Enqueue sample of size ", sample_len)
        if sample_len % 100:
            sample_len += 100 - (sample_len % 100)

        self.issues += 1

        q = self.queue_id
        if self.qwaiting_flag:
            old_q = self.find_waiting_q(sample_len)
            if old_q >= 0:
                q = old_q
                # print("Found a prev q ", q, " waiting for sample of size", sample_len)
                # print(self.qwaiting)
                self.qwaiting[q] -= sample_len
                if not sum(x for x in self.qwaiting):
                    self.qwaiting_flag = False
                    # print("Reset qwaiting flag to False")

        self.ids[q].append(query_id[0])
        self.idxes[q].append(idx[0])
        self.qsize[q] += sample_len
        batch_size = self.max_batchsize
        # print("Putting sample of size", sample_len, " in queue ", q, " New size ", self.qsize[q])

        if self.qsize[q] == batch_size:
            self.issue(q, self.ids[q], self.idxes[q])
            if q == self.queue_id:
                increment_q = True
                # print("Changing queue as we reached batch size")
        elif batch_size - self.qsize[q] <= 700:
            to_fill = batch_size - self.qsize[q]
            # if self.qwaiting[q] > 0:
            #     print("Filling queue but qwaiting not empty")
            #     print(self.qwaiting)
            self.qwaiting[q] = to_fill
            self.qwaiting_flag = True
            if q == self.queue_id:
                increment_q = True
                # print("Changing queue as current queue ", q, " will wait for size", to_fill)
                # print(self.qwaiting)
                # print(self.qsize[q])

        if increment_q:
            self.qorder.rotate(-1)
            self.queue_id = self.qorder[-1]
            if self.qsize[self.queue_id] > 0:
                # This queue has been waiting for a sample
                self.issue(self.queue_id, self.ids[self.queue_id], self.idxes[self.queue_id])

        if self.issues == qcount:
            # print(self.qsize)
            for q in range(self.num_queues):
                if self.ids[q]:
                    self.inQueue.put(Item(self.ids[q], self.idxes[q]))
                    # print("Sendall in queue", q, " with total samples ", len(self.ids[q]), " qsize ", self.qsize[q])

    def unload_query_samples(self):
        self.ds.unload_query_samples(self.sample_list)

def add_results(final_results, name, result_dict, result_list, took, show_accuracy=False):
    percentiles = [50., 80., 90., 95., 99., 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])

    if result_dict["total"] == 0:
        result_dict["total"] = len(result_list)

    # this is what we record for each run
    result = {
        "took": took,
        "mean": np.mean(result_list),
        "percentiles": {str(k): v for k, v in zip(percentiles, buckets)},
        "qps": len(result_list) / took,
        "count": len(result_list),
        "good_items": result_dict["good"],
        "total_items": result_dict["total"],
    }
    acc_str = ""
    if show_accuracy:
        result["accuracy"] = 100. * result_dict["good"] / result_dict["total"]
        acc_str = ", acc={:.3f}%".format(result["accuracy"])
        if "roc_auc" in result_dict:
            result["roc_auc"] = 100. * result_dict["roc_auc"]
            acc_str += ", auc={:.3f}%".format(result["roc_auc"])

    # add the result to the result dict
    final_results[name] = result

    # to stdout
    print("{} qps={:.2f}, mean={:.4f}, time={:.3f}{}, queries={}, tiles={}".format(
        name, result["qps"], result["mean"], took, acc_str,
        len(result_list), buckets_str))

def response_loadgen(outQueue, accuracy):
    global item_good
    global item_total
    global item_timing
    global item_results

    while True:
        oitem = outQueue.get()
        if oitem is None:
            break

        response = []
        for q_id, arr in zip(oitem.query_ids, oitem.array_ref):
            bi = arr.buffer_info()
            response.append(lg.QuerySampleResponse(q_id, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)

        if accuracy:
            item_good += oitem.good
            item_total += oitem.total
            item_timing.append(oitem.timing)
            item_results.append(oitem.presults)


def main():
    global qcount
    global num_sockets
    global cpus_per_socket
    global cpus_per_process
    global cpus_per_instance
    global total_instances
    global start_time
    global item_total
    global last_timeing

    args = get_args()
    log.info(args)
    config = os.path.abspath(args.config)
    user_config = os.path.abspath(args.user_config)

    if not os.path.exists(config):
        log.error("{} not found".format(config))
        sys.exit(1)

    if not os.path.exists(user_config):
        log.error("{} not found".format(user_config))
        sys.exit(1)

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists("./audit.config"):
            copyfile("./audit.config", output_dir + "/audit.config")
        os.chdir(output_dir)

    settings = lg.TestSettings()
    settings.FromConfig(config, args.model, args.scenario)
    settings.FromConfig(user_config, args.model, args.scenario)
    settings.mode = lg.TestMode.PerformanceOnly

    cpus_for_loadgen = 1
    left_cores = cpus_per_socket * num_sockets - total_procs * cpus_per_process
    first_instance_start_core = cpus_for_loadgen 
    if left_cores > cpus_for_loadgen:
        first_instance_start_core = 0
        cpus_for_loadgen = left_cores

    total_instances = 0
    instances_per_proc = (cpus_per_process // cpus_per_instance)
    for i in range(total_procs):
        if i == 0 and first_instance_start_core > 0:
            total_instances = total_instances + ((cpus_per_process - first_instance_start_core) // cpus_per_instance)
            if (cpus_per_instance - first_instance_start_core) >= (cpus_per_instance // 2):
                total_instances = total_instances + 1
        else:
            total_instances = total_instances + instances_per_proc
    #print("Setup {} Instances !!".format(total_instances))

    lock = multiprocessing.Lock()
    barrier = multiprocessing.Barrier(total_instances)
    init_counter = multiprocessing.Value("i", 0)
    total_samples = multiprocessing.Value("i", 0)
    finished_samples = multiprocessing.Value("i", 0)
    dsQueue = multiprocessing.Queue()
    numOutQ = num_sockets
    outQueues = [multiprocessing.Queue() for i in range(numOutQ)]
    #inQueue = multiprocessing.JoinableQueue()
    inQueue = multiprocessing.Queue()
    consumers = [Consumer(inQueue, outQueues[i%numOutQ], dsQueue, lock, init_counter, finished_samples, barrier, i, args, settings.min_query_count, first_instance_start_core)
                 for i in range(total_procs)]
    for c in consumers:
        c.start()

    # Wait until subprocess ready
    while init_counter.value < total_procs: time.sleep(2)

    import torch
    import criteo
    torch.set_num_threads(cpus_per_socket * num_sockets)

    dlrm_dataset = get_dataset(args)
    total_samples.value = dlrm_dataset.get_item_count()
    scenario = SCENARIO_MAP[args.scenario]
    runner_map = {
        lg.TestScenario.Server: QueueRunner,
        lg.TestScenario.Offline: QueueRunner
    }

    settings.scenario = scenario
    runner = runner_map[scenario](inQueue, dlrm_dataset, total_samples.value, args.max_sample_size, args.max_batchsize)

    # Start response thread
    response_workers = [threading.Thread(
        target=response_loadgen, args=(outQueues[i], args.accuracy)) for i in range(numOutQ)]
    for response_worker in response_workers:
       response_worker.daemon = True
       response_worker.start()


    def issue_queries(response_ids, query_sample_indexes):
        runner.enqueue(response_ids, query_sample_indexes)

    def flush_queries():
        runner.unload_query_samples()

    def process_latencies(latencies_ns):
        # called by loadgen to show us the recorded latencies
        global last_timeing
        last_timeing = [t / NANO_SEC for t in latencies_ns]

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
        settings.performance_sample_count_override = total_samples.value

    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if args.duration:
        settings.min_duration_ms = args.duration
        settings.max_duration_ms = args.duration

    if args.target_qps:
        settings.server_target_qps = float(args.target_qps)
        settings.offline_expected_qps = float(args.target_qps)

    if args.count_queries:
        settings.min_query_count = args.count_queries
        settings.max_query_count = args.count_queries

    if args.samples_per_query_multistream:
        settings.multi_stream_samples_per_query = args.samples_per_query_multistream

    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_target_latency_ns = int(args.max_latency * NANO_SEC)

    if args.accuracy:
        qcount = total_samples.value
    else:
        qcount = settings.min_query_count

    def load_query_samples(sample_list):
        # Wait until subprocess ready
        global start_time
        global total_instances
        for _ in range(total_instances):
            dsQueue.put(sample_list)
        while init_counter.value < total_procs + total_instances: time.sleep(2)
        start_time = time.time()

    def unload_query_samples(sample_list):
        pass

    sut = lg.ConstructFastSUT(issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(total_samples.value, min(total_samples.value, args.samples_per_query_offline), load_query_samples, unload_query_samples)

    log.info("starting {}".format(scenario))
    result_dict = {"good": 0, "total": 0, "roc_auc": 0, "scenario": str(scenario)}

    torch.set_num_threads(cpus_for_loadgen)
    lg.StartTest(sut, qsl, settings)

    if not last_timeing:
        last_timeing = item_timing
    if args.accuracy:
        result_dict["good"] = item_good
        result_dict["total"] = item_total
        result_dict["roc_auc"] = criteo.auc_score(item_results)

    final_results = {
        "runtime": "pytorch-native-dlrm",
        "version": torch.__version__,
        "time": int(time.time()),
        "cmdline": str(args),
    }

    add_results(final_results, "{}".format(scenario),
                result_dict, last_timeing, time.time() - start_time, args.accuracy)

    #inQueue.join()
    for _ in range(total_instances):
        inQueue.put(None)
    for c in consumers:
        c.join()
    for i in range(numOutQ):
        outQueues[i].put(None)

    lg.DestroyQSL(qsl)
    lg.DestroyFastSUT(sut)

    # write final results
    if args.output:
        with open("results.json", "w") as f:
            json.dump(final_results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
