# NVIDIA MLPerf Inference System Under Test (SUT) performance tuning guide

The NVIDIA MLPerf Inference System Under Test implementation has many different parameters which can be tuned to achieve the best performance under the various MLPerf scenarios on a particular system.
However, if starting from a good baseline set of parameters, only a small number of settings will need to be adjusted to achieve good performance.

⚠ **Important**:
Please restrict your performance tuning changes to the settings in the `configs/<BENCHMARK>/<SCENARIO>/config.json` files.
All files in the `measurements/` directory, including `user.conf`, are automatically generated from the `config.json` files.
So please do **not** modify the `user.conf` files.

## Adding a new system

We formally support and fully test the configuration files for only the systems listed in [README.md](README.md).
To run on a different system configuration, follow the steps below:

If you plan to run on NVIDIA A100 or NVIDIA T4 with different numbers of GPUs, you can add your own configurations based
on the provided ones.

First you will need to add a configuration for your system in each benchmark-scenario `config.json`. To do this, you can
copy an existing block for a similar system configuration.

The syntax is as follows:

```
{
    ...
    "your-system-id": {
        <key>: <value>,
        ...
    }
    ...
}
```
Most of these keys are self explanatory and can be copied from existing configs. However, 2 of these keys define
inheritance rules that are worth mentioning:

```
extends: [ <list of system IDs> ],
scales: {
    <system ID>: {
        <key>: <numerical scaling factor>,
        ...
    }
}
```

`extends` inherits the fields from all system IDs specified in the list. System IDs that appear later in `extends` will
override fields set by other extended system IDs before it.

`scales` works similar to `extends`, and will always be processed *after* `extends` (meaning it will override fields
from `extends`). In addition to inheriting the fields, it will also scale specific keys by the specified scaling
factors. For example:

```
scales: {
    A100-SXM4-40GBx1: {
        gpu_offline_expected_qps: 0.2,
        gpu_batch_size: 0.1
    }
}
```

will inherit all the fields from the `A100-SXM4-40GBx1` and then scale the inherited `gpu_offline_expected_qps` by a factor of 0.2, and `gpu_batch_size` by a factor of 0.1.

**Side note**: For BERT benchmark, `server_num_issue_query_threads` field also need to be scaled by the number of GPUs.

**After you add your entry into config.json**, you will have to add your system description into
[code/common/system_list.py](code/common/system_list.py). At the bottom of the file there is a class called
`KnownSystems`. This class defines a list of `SystemClass` objects that describe supported systems as follows:

```
SystemClass(<system ID>, [<list of names reported by nvidia-smi>], [<known PCI IDs of this system>],
    <architecture>, [list of known supported gpu counts>])
```

 - `system ID`: The system ID you would like to identify this system as
 - For the list of names reported by `nvidia-smi`, run `nvidia-smi` and use the name it reports.
 - For PCI IDs, run

   ```
   $ CUDA_VISIBLE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=gpu_name,pci.device_id --format=csv
   name, pci.device_id
   TITAN RTX, 0x1E0210DE
   ...
   ```

   This `pci.device_id` field is in the format `0x<PCI ID>10DE`, where `10DE` is the NVIDIA PCI vendor ID. Use the 4 hex
   digits between `0x` and `10DE` as your PCI ID for the system. In this case, it is `1E02`.

 - `architecture`: There is an architecture `Enum` at the top of the file. Use one of these fields.
 - For the list of known GPU counts, use a list of the number of GPUs of the systems you would like to support (i.e.
   `[1,2,4]` if you want to support 1x, 2x, and 4x GPU variants of this system.)

After you do this, you can run commands as documented in [README.md](README.md). It should work or should be very close
to working. If you see INVALID results in any case, follow the steps in the [Fix INVALID results](#fix-invalid-results)
section below.

If you plan to run on GPUs other than NVIDIA A100 or explicitly supported systems, you can still follow these steps to get a set of baseline config files.
However, more performance tuning will be required to achieve better performance numbers, such as changing batch sizes or number of streams to run inference with.
See the [Tune parameters for better performance](#tune-parameters-for-better-performance) section below.

### Different system configurations that use the same GPU configuration

Sometimes, it may be the case that submitters will have 2 systems with the same GPU configuration, but differing system
configurations elsewhere, such as a different CPU, memory size, etc. In this case, internally, our code will detect
these 2 systems as the same `system_id`. However, the `config.json` parameters might be different for the 2 systems. To
handle this, you can use the `config_ver` feature to define different versions of the `config.json` for a single system
ID. For example, if you have 2 systems that use 2xA100-PCIe, but one system has an Intel CPU, and another has an AMD
CPU, you can declare a `config_ver` block like below:

```
{
    ...
    "A100-PCIex2": {
        "config_ver": {
            "intel_xeon_modelXYZ": {
                "gpu_offline_expected_qps": <throughput on Intel>
            },
            "amd_epyc_modelABC": {
                "gpu_offline_expected_qps": <throughput on AMD>
            },
        },
        "gpu_batch_size": 2048,
        "gpu_copy_streams": 4,
        "gpu_offline_expected_qps": 66240,
        "run_infer_on_copy_streams": true
    },
    ...
}
```

Then to run with a specific config version, simply specify `--config_ver=<name of config version>` in `RUN_ARGS` when
running the harness.

However, these two systems will still have the same 'System name' from `systems/<system_name>.json`. Users should
instead of 2 separate entries in `systems/`, such as `systems/A100-PCIex2_intel_xeon.json` and
`systems/A100-PCIex2_amd_epyc.json`. To use these system descriptions, either:

 1. In the `config_ver` block itself, specify a `system_name` key in each config version, pointing to the appropriate
    System Name.
 2. When running the harness, use `--system_name=<system_name>` in `RUN_ARGS`


## NUMA configuration

If the submission system has multiple CPUs and supports NUMA, then configuring NUMA correctly might be required for optimum performance.
The target system may have NUMA configuration enabled. Users can check if the system is in active NUMA configuration by `numactl --hardware`:
```
$ numactl --hardware

available: 4 nodes (0-3)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
node 0 size: 128825 MB
node 0 free: 124581 MB
node 1 cpus: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
node 1 size: 129016 MB
node 1 free: 127519 MB
node 2 cpus: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111
node 2 size: 129016 MB
node 2 free: 120735 MB
node 3 cpus: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
node 3 size: 128978 MB
node 3 free: 125733 MB
node distances:
node   0   1   2   3 
  0:  10  12  12  12 
  1:  12  10  12  12 
  2:  12  12  10  12 
  3:  12  12  12  10 
```

The above example shows there are 4 NUMA nodes, from 0 to 3. Each NUMA node has one or more CPUs and a memory address space that is closer,
in terms of latency, to those CPUs compared to CPUs of other NUMA nodes. For example, CPU ID 2 is in node 0; this CPU 2 can 
access memory address in memory address space within node 0 with less latency, compared to memory address 
in memory address space within node 3. This CPU may get higher memory bandwidth when accessing memory addresses in the node 0 memory address space.
This memory access latency is shown as `distance` in the `numactl --hardware` output above.

If `numactl --hardware` shows only one node, it means the system is not configured to take advantage of NUMA affinity.

`numactl` does not show the NUMA affinity of the GPUs in the system. In order to find this information, we use `nvidia-smi topo -m`
```
$ nvidia-smi topo -m

GPU0    GPU1    GPU2    GPU3    CPU    Affinity        NUMA Affinity
GPU0     X      SYS     SYS     SYS    48-63,112-127   3
GPU1    SYS      X      SYS     SYS    32-47,96-111    2
GPU2    SYS     SYS      X      SYS    16-31,80-95     1
GPU3    SYS     SYS     SYS      X     0-15,64-79      0

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

The above printout suggests that GPU 3 is in NUMA node 0. Therefore performance would be improved if
CPU 0-16,64-79 would talk to GPU 3, using memory address space in node 0.

In order to exploit the above NUMA affinity information, it is recommended to configure the harness accordingly. The steps to do this differ
based on the type of harness:
1. Triton harness currently does not have a NUMA aware setting.
2. Multi-MIG Triton harness finds NUMA aware settings automatically.
3. LWIS harness requires manual setting by user in config.json -- For details on how to specify numa_config please see documentation of this option in arguments.py.


## Test runtime

The [MLPerf Inference rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc) requires each submission to meet certain requirements to become a valid submission, and one of them is the runtime for the benchmark test.
Below we summarize the expected runtime for each scenario.
Some of them will take several hours to run, so we also provide tips to reduce test runtime for faster debugging cycles.

### Offline

Default test runtime: `max((1.1 * min_duration * offline_expected_qps / actual_qps), (min_query_count / actual_qps))`, where:

- `min_duration`: 600 seconds by default.
- `offline_expected_qps`: set by `config.json`.
- `min_query_count`: 24576 by default.

The typical runtime for Offline scenario should be about 66 seconds, unless on a system with very low QPS, in which case, runtime will be much longer.
To reduce runtime for debugging, simply lower `min_duration` and `min_query_count` by adding flags like `--min_duration=1000 --min_query_count=1` to `RUN_ARGS` in the command.

### Server

Default test runtime: `max((min_duration * server_target_qps / actual_qps), (min_query_count / actual_qps))`, where:

- `min_duration`: 600 seconds by default.
- `server_target_qps`: set by `config.json`.
- `min_query_count`: 270336 by default.

The typical runtime for Server scenario is about 600 seconds if `server_target_qps` is equal to or is lower than actual QPS.
Otherwise, the runtime will be longer since queries start to queue up as the system cannot digest the queries in time.
To reduce runtime for debugging, simply lower `min_duration` and `min_query_count` by adding flags like `--min_duration=1000 --min_query_count=1` to `RUN_ARGS` in the command.

### SingleStream

Default test runtime: `max((min_duration / single_stream_expected_latency_ns * actual_latency), (min_query_count * actual_latency))`, where:

- `min_duration`: 600 seconds by default.
- `single_stream_expected_latency_ns`: set by `config.json`.
- `min_query_count`: 1024 by default.

The typical runtime for SingleStream scenario should be about 600 seconds, unless on a system with very long latency per sample, in which case, runtime will be much longer.
To reduce runtime for debugging, simply lower `min_duration` and `min_query_count` by adding flags like `--min_duration=1000 --min_query_count=1` to `RUN_ARGS` in the command.

### MultiStream

Default test runtime: `max((min_duration), (min_duration / multi_stream_target_qps * multi_stream_samples_per_query / actual_qps), (min_query_count / multi_stream_target_qps))`, where:

- `min_duration`: 600 seconds by default.
- `multi_stream_target_qps`: 20 qps for ResNet50 and SSDMobileNet benchmarks and 16 qps for SSDResNet34 benchmark. Do not change this since this is defined by MLPerf Inference rules.
- `multi_stream_samples_per_query`: set by `config.json`.
- `min_query_count`: 270336 by default.

Note that this results in typical test runtime being roughly 3h45m for ResNet50 and SSDMobileNet benchmarks and 4h45m for SSDResNet34 benchmark regardless of the actual QPS of the system.
To reduce runtime for debugging, simply lower `min_duration` and `min_query_count` by adding flags like `--min_duration=1000 --min_query_count=1` to `RUN_ARGS` in the command.

## Fix INVALID results

### Offline

The most common reason for INVALID results in Offline scenario is that the actual QPS of the system is much higher than the `offline_expected_qps`.
Therefore, simply increase `offline_expected_qps` until the max query latency reported by LoadGen reaches 66 secs, which is when `offline_expected_qps` matches the actual QPS.

### Server

The most common reason for INVALID results in Server scenario is that the actual QPS of the system is lower than the `server_target_qps`.
Therefore, reduce `server_target_qps` until the 99th percentile latency falls below the Server latency targets defined by the [MLPerf Inference rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc). If lowering `server_target_qps` does not reduce the 99th percentile latency,
try reducing `gpu_batch_size` and/or `gpu_inference_streams` instead.

### SingleStream

The most common reason for INVALID results in SingleStream scenario is that the actual latency of the system for each sample is much lower than the `single_stream_expected_latency_ns`.
Therefore, simply lower `single_stream_expected_latency_ns` to match the actual 90th percentile latency reported by LoadGen.

### MultiStream

The most common reason for INVALID results in MultiStream scenario is that the actual QPS of the system cannot handle the target `multi_stream_samples_per_query` we set.
Therefore, reduce `multi_stream_samples_per_query` until the 99th percentile falls below the MultiStream target QPS defined by [MLPerf Inference rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc).

## Tune parameters for better performance

To get better performance numbers, parameters in the config files such as batch sizes can be further tuned.
All settings can be adjusted using the configuration files. Additionally, for interactive experiments, the settings can be adjusted on the command line.

For example:

```
$ make run RUN_ARGS="--benchmarks=ResNet50 --scenarios=offline --gpu_batch_size=128"
```

will run ResNet50 in the offline scenario, overriding the batch size to 128.
To use that batch size for submission, the `gpu_batch_size` setting will need to be changed in the corresponding config.json file.

Below we show some common parameters that can be tuned if necessary.
There are also some benchmark-specific or platform-specific parameters which are not listed here.
For example, on Jetson AGX Xavier or Xavier NX platforms for ResNet50/SSDResNet34/SSDMobileNet, there are also DLA parameters (such as `dla_batch_size`) that can be tuned for performance.
Please look at the code to understand the purpose of those parameters.

### Offline

In the Offline scenario, the default settings should provide good performance.
Optionally, try increasing `gpu_batch_size` to see if it gives better performance. CUDA stream settings like `gpu_copy_streams` and `gpu_inference_streams` can also be tried to better overlap the memory transfers with inference computation.

### Server

Server scenario tuning can be a bit more involved.
The goal is to increase `server_target_qps` to the maximum value that can still satisfy the latency requirements specified by the [MLPerf Inference rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc).

This is typically an iterative process:

1. Increase `server_target_qps` to the maximum value that still meets the latency constraints with the current settings
2. Sweep across many variations of the existing settings, such as `gpu_batch_size`, `deque_timeout_us`, `gpu_copy_streams`, `gpu_inference_streams`, and so on.
3. Replace the current settings with those providing the lowest latency at the target percentile, with the current `server_target_qps` setting
4. Goto 1.

### SingleStream

In the SingleStream scenario, the default settings should provide good performance.

### MultiStream

The multistream scenario is conceptually simpler than the server scenario, but the tuning process is similar.
The goal is to increase the `gpu_multi_stream_samples_per_query` to the largest value that still satisfies the latency constraints.

To start, set `gpu_batch_size` to be equal to `gpu_multi_stream_samples_per_query` and then increase the values to the maximum point where the latency constraint is still met.

Since all samples belonging to a query arrive at the same time, use of a single batch to process the entire query will lead to serialization between the data copies between the host and device and the inference.
This can be mitigated by splitting the query into multiple batches, reducing the amount of data that must be transferred before inference can begin on the device.
If `gpu_batch_size` is less than `gpu_multi_stream_samples_per_query`, the samples will be divided into `ceil(gpu_multi_stream_samples_per_query / gpu_batch_size)` batches.
The data transfers can then be pipelined against computation, reducing overall latency. Other settings like `gpu_copy_streams` and `gpu_inference_streams` can also be tried.

## Other performance tips

- For systems with passive cooling GPUs, especially systems with NVIDIA T4, the cooling system plays an important role in performance.
You can run `nvidia-smi dmon -s pc` to monitor the GPU temperature while harness is running.
To get best performance, GPU temperature should saturate to a reasonable temperature, such as 65C, instead of constantly going up and throttling the GPU clock frequencies.
On Jetson platforms, use `tegrastats` to monitor GPU temperature instead.
- In Server scenario, please make sure that the Transparent Huge Page setting is set to "always".
