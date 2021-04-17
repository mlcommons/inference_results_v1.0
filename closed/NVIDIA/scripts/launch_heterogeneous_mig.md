# Instructions for MIG-Hetero

1. The main script is `mlperf-inference/closed/NVIDIA/scripts/launch_heterogeneous_mig.py`.
It is a wrapper that runs `make run_harness` for 6 "background benchmarks" and one "main benchmark" that we'll actually use the results from.
Please do `launch_heterogeneous_mig.py --help` to see what commandline arguments are supported.

2. It launches the background benchmarks first, monitors their stdouts until it sees `running actual test` from all of them, waits for a configurable period, and then launches the main benchmark.
After it sees that the main benchmark has completed, it waits for all the background benchmarks to complete and confirms that (1) They did not complete too early, and (2) they exit properly with exit code 0.

3. In order to target a specific GPU instance on a device configured with multiple GPU instances, you have to set `CUDA_VISIBLE_DEVICES` with the UUID of the GPU instance you're targeting, so all of the benchmark launch commands look like something like this:
```
CUDA_VISIBLE_DEVICES=MIG-GPU-e86cb44c-6756-fd30-cd4a-1e6da3caf9b0/1/0 make run_harness "... --config_ver=hetero ..."
```

4. The main command line options to use with the `launch_heterogeneous_mig.py` script are `--main_benchmark` to set the main benchmark, `--main_scenario` to tell it which scenario to run the main benchmark in (we'll use offline and server), `--background_benchmarks` to tell it which benchmarks to run in the background (always set to `server` which is an alias for the 6 server benchmarks), `--start_time_buffer` to specify how long to wait for all benchmarks background benchmarks to wait for "running actual test", and `--background_benchmark_duration` to tell it how long to make the background benchmarks run for using the `--min_duration RUN_ARG`.
In some cases, I have also used `--main_benchmark_runargs=--min_query_count=1` to make things run more quickly for debugging or faster measurements.

5. Setting `--background_benchmark_duration` to a high enough value sometimes takes some iteration if, like me, you don't really know how long things run for in their default configurations.

6. Preparing engines before running the `launch_heterogeneous_mig.py` script is important. How to build an engine is similar to run a harness as above:
```
CUDA_VISIBLE_DEVICES=MIG-GPU-e86cb44c-6756-fd30-cd4a-1e6da3caf9b0/1/0 make generate_engines "... --config_ver=hetero ..."
```

- A100 MIG 1g.10gb supports all benchmarks.
- A30 MIG 1g.3gb supports subset of the benchmarks due to its low memory capacity. Benchmarks supported are listed below; note that bert high_accuracy_triton is NOT supported:
    - Datacenter: bert, resnet50, rnnt, ssd-resnet34
    - Edge: bert, resnet50, rnnt, ssd-mobilenet, ssd-resnet34

7. Tips to generate engines and run this script.
- Launch docker with `MIG_CONF=1` for generating engines, i.e. `make prebuild MIG_CONF=1` in `mlperf-inference/closed/NVIDIA`. 
  This automatically sets up the environment where only one MIG instance is visible.
- Launch docker with `MIG_CONF=ALL` for running the script, i.e. `make prebuild MIG_CONF=ALL` in `mlperf-inference/closed/NVIDIA`. 
  This automatically sets up the environment where all the MIG slices are populated. Then `launch_heterogeneous_mig.py` will choose MIG instances automatically for main and background benchmarks.

8. Some commandline examples; from docker container::`/work`
```
# main benchmark: ssd-resnet34, background benchmarks: datacenter, Server scenario, PerformanceOnly run
python3 ./scripts/launch_heterogeneous_mig.py --main_action=run_harness --background_benchmarks=datacenter --main_benchmark=ssd-resnet34 --main_scenario=server
# main benchmark: ssd-resnet34, background benchmarks: datacenter, Server scenario, AccuracyOnly run
python3 ./scripts/launch_heterogeneous_mig.py --main_action=run_harness --background_benchmarks=datacenter --main_benchmark=ssd-resnet34 --main_scenario=server --main_benchmark_args="--test_mode=AccuracyOnly"
# main benchmark: resnet50, background benchmarks: datacenter, Server scenario, AUDIT_TEST01 run
python3 ./scripts/launch_heterogeneous_mig.py --main_action=run_audit_test01 --background_benchmarks=datacenter --main_benchmark=resnet50 --main_scenario=Offline
```

# Tuning Performance and Target QPS of the main benchmark
For tuning, it is recommended to focus on each offline and server benchmark individually first (i.e. not worry about the fact that we'll eventually run with background workloads).
In most cases, we get similar results with and without background workloads.

DLRM, which is ~7% slower in offline scenario with background benchmarks, was first tuned individually and then the target QPS was lowered after adding background benchmarks.

In summary, we tune in isolation, and then, we lower the target QPS if needed as we move to the environment with background benchmarks.
