# MLPerf Inference v1.0 Alibaba Cloud Sinian Platform Implementations

This is a repository of Alibaba Cloud Sinian Platform implementations for [MLPerf Inference Benchmark v1.0](https://www.mlperf.org/inference-overview/).

## Platform
### Sinian
Sinian is Alibaba’s compiler-based heterogeneous hardware acceleration platform, targeting extreme performance for machine learning applications. Interfacing with the upper level frameworks such as Alibaba PAI, Tensorflow , MxNet and etc, Sinian enables deep co-optimizations between software and hardware to deliver high execution efficiency for ML applications. Sinian is fully tailorable (“statically and dynamically”) for cloud computing, edge computing, and IoT devices, making it easy to achieve performance portability between training and deploying machine learning models across heterogeneous accelerators. 
### AutoSinian
AutoSinian is the automatic performance optimization framework in Sinian. By auto-tuning and joint-optimizing the heterogeneous system performance across algorithm, system, framework and hardware library layers, AutoSinian serves as the core component in Sinian to maximize performance for machine learning applications with very little engineer efforts in case-by-case performance tuning.

## Benchmarks

The following *benchmarks* are part of our submission for MLPerf Inference v1.0:

 - **ResNet50** (resnet50)

## Scenarios
The above benchmarks can run in one or more of the following three inference *scenarios*:

 - **SingleStream**
 - **Offline**
 - **Server**

Please refer to the [MLPerf Inference official page](https://www.mlperf.org/inference-overview/) for explanations about the scenarios.

## Alibaba Submissions

Our MLPerf Inference v1.0 implementation has the following submissions:

| Benchmark     | Datacenter Submissions                                        | Edge Submissions                                   |
|---------------|---------------------------------------------------------------|----------------------------------------------------------------------------------|
| ResNet50      | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream           |

The benchmark is stored in the [code/](code) directory which contains a `README.md` detailing instructions on how to set up the benchmark, including:

 - Downloading the dataset and model
 - Running any necessary preprocessing
 - Details on the optimizations being performed

For details on how to run the benchmark, see below.

## Alibaba Submission Systems

The systems that Alibaba supports, has tested, and are submitting are:

 - Datacenter systems
   - A100-SXM-40GBx8
   - A10x2
   - T4x8
 - Edge systems
   - A100-SXM-40GBx1
   - A10x1
   - T4x1

## General Instructions

:warning: **IMPORTANT**: Please use [open/Alibaba](open/Alibaba) (this directory) as the working directory when running any of the commands below. :warning:

:warning: IMPORTANT: Please do not execute any commands or clone the repository with user `root` or `sudo`. Doing so may cause errors that involve directories being unwriteable. :warning:

**Note:** Inside the Docker container, [open/Alibaba](open/Alibaba) will be mounted at `/work`.

This section describes the steps needed to run harness with default configurations, weights, and validation datasets on Alibaba submission systems to reproduce.
Please refer to later sections for instructions on auditing.

### Prerequisites

For x86_64 systems:

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) with `libnvidia-container>=1.3.1-1`
- Ampere/Turing-based NVIDIA GPUs
- NVIDIA Driver Version 455.xx or greater

We recommend using Ubuntu 18.04.
Other operating systems have not been tested.

### Before you run commands

Before running any commands detailed below, such as downloading and preprocessing datasets, or running the benchmark, you should
set up the environment by doing the following:

- Run `export MLPERF_SCRATCH_PATH=<path/to/scratch/space>` to set your scratch space path.
We recommend that the scratch space has at least **40GB**.
The scratch space will be used to store models, datasets, and preprocessed datasets.

- For x86_64 systems: Run `make prebuild`.
This launches the Docker container with all the necessary packages installed.

 The docker image will have the tag `mlperf-inference:<USERNAME>-latest`.
 The source codes in the repository are located at `/work` inside the docker image.

:warning: If your system has multiple GPUs, but you only wish to use a subset of them, use `DOCKER_ARGS="--gpus '\"device=<comma separated device IDs>\"'"`

For example, if you have a 4x A100-PCIe system, but you only wish to use 2, you can use:

```
make prebuild DOCKER_ARGS="--gpus '\"device=0,1\"'"
```

As an alternative, you may also specify a comma separated list of GPU UUIDs provided from `nvidia-smi -L`.

### Download and Preprocess Datasets and Download Models

The [benchmark](code) already contains the optimized ofa-autosinian model for the resnet50 benchmark.

- please first download the [ImageNet 2012 Validation set](http://www.image-net.org/challenges/LSVRC/2012/) and unzip the images to `$MLPERF_SCRATCH_PATH/data/imagenet/`.

Quick commands:

```
$ make preprocess_data BENCHMARKS=resnet50 # Preprocess data and saves to $MLPERF_SCRATCH_PATH/preprocessed_data
```

Notes:

- The combined preprocessed data can be huge.
- Please reserve at least **40GB** of storage in `$MLPERF_SCRATCH_PATH` to ensure you can store everything.

### Running the repository

Running models is broken down into 3 steps:

#### Build

Builds the required libraries:

```
$ make build
```

#### Generate TensorRT engines

```
$ make generate_engines RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIOS> --config_ver=default,high_accuracy [OTHER FLAGS]"
```

If `RUN_ARGS` is not specified, all engines for each supported benchmark-scenario pair will be built.
See [command_flags.md](command_flags.md) for information on arguments that can be used with `RUN_ARGS`.
The optimized engine files are saved to `/work/build/engines`.

#### Run harness on engines

:warning: **IMPORTANT**: In MLPerf Inference v1.0, the default `min_duration` of harness runs was extended from 1min to
10min. As this is quite the large increase in runtime duration, there is now a new `--fast` flag that can be specified
in `RUN_ARGS` that is a shortcut to specify `--min_duration=60000`. In Offline scenario, this also sets
`--min_query_count=1`.

```
$ make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIOS> --test_mode=<AccuracyOnly/PerformanceOnly> [OTHER FLAGS]"
```

If `RUN_ARGS` is not specified, all harnesses for each supported benchmark-scenario pair will be run.
See [command_flags.md](command_flags.md) for `RUN_ARGS` options.
Note that if an engine has not already been built for a benchmark-scenario pair (in the earlier step), this will result in an error.

The performance results will be printed to `stdout`.
Other logging will be sent to `stderr`.
LoadGen logs can be found in `/work/build/logs`.

### Notes on runtime and performance

- To achieve maximum performance for Server scenario, please set Transparent Huge Pages (THP) to *always*.
- To achieve maximum performance for Server scenario on T4x8 and T4x20 systems (no longer officially supported), please lock the clock at the max frequency by `sudo nvidia-smi -lgc 1590,1590`.
- As a shortcut, doing `make run RUN_ARGS="..."` will run `generate_engines` and `run_harness` in succession. If multiple benchmark-scenario pairs are specified, engines will only run after all engines are successfully built.

### Run Calibration

The calibration caches generated from default calibration set are already provided in the benchmark directory.

If you would like to re-generate the calibration cache for the benchmark, please run the following command:

```
$ make calibrate RUN_ARGS="--benchmarks=resnet50"
```

See [calibration.md](calibration.md) for an explanation on how calibration is used for Alibaba's submission.

## Instructions for Auditors

Please refer to the `README.md` in each benchmark directory for auditing instructions.

## Other documentations:

- [FAQ.md](FAQ.md): Frequently asked questions.
- [calibration.md](calibration.md): Instructions about how we did post-training quantization on activations and weights.
- [command_flags.md](command_flags.md): List of some flags supported by `RUN_ARGS`.
- [benchmark READMEs](code/README.md): Instructions about how to download and preprocess the models and the datasets for the benchmark and lists of optimizations we did.
