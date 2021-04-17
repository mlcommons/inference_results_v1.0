# Instructions for MIG-Hetero

1. The main script is `mlperf-inference/closed/Fujitsu/scripts/launch_heterogeneous_mig.py`.
It is a wrapper that runs `make run_harness` for 6 "background benchmarks" and one "main benchmark" that we'll actually use the results from.

2. It launches the background benchmarks first, monitors their stdouts until it sees `running actual test` from all of them, waits for a configurable period, and then launches the main benchmark.
After it sees that the main benchmark has completed, it waits for all the background benchmarks to complete and confirms that (1) They did not complete too early, and (2) they exit properly with exit code 0.

3. In order to target a specific GPU instance on a device configured with multiple GPU instances, you have to set `CUDA_VISIBLE_DEVICES` with the UUID of the GPU instance you're targeting, so all of the benchmark launch commands look like something like this:
```
CUDA_VISIBLE_DEVICES=MIG-GPU-e86cb44c-6756-fd30-cd4a-1e6da3caf9b0/1/0 make run_harness ...
```

4. The main command line options to use with the `launch_heterogeneous_mig.py` script are `--main_benchmark` to set the main benchmark, `--main_scenario` to tell it which scenario to run the main benchmark in (we'll use offline and server), `--background_benchmarks` to tell it which benchmarks to run in the background (always set to `server` which is an alias for the 6 server benchmarks), `--start_time_buffer` to specify how long to wait for all benchmarks background benchmarks to wait for "running actual test", and `--background_benchmark_duration` to tell it how long to make the background benchmarks run for using the `--min_duration RUN_ARG`.
In some cases, I have also used `--main_benchmark_runargs=--min_query_count=1` to make things run more quickly for debugging or faster measurements.

5. Setting `--background_benchmark_duration` to a high enough value sometimes takes some iteration if, like me, you don't really know how long things run for in their default configurations.

For tuning purposes, I think it will work fine to focus on each offline and server benchmark individually (i.e. not worry about the fact that we'll eventually run with background workloads).
In most cases, I expect we'll get the same result with and without background workloads.
For DLRM (which is ~7% slower in offline mode when you add the background benchmarks), the run was invalid initially, but it succeeded when I reduced the target qps by 7%.
So basically, we tune in isolation, if needed, we can lower the target qps a bit when we move to the environment with background benchmarks.