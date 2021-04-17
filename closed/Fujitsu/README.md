# MLPerf Inference v1.0 Implementations for Fujitsu Servers

This is a repository of implementations for Fujitsu Servers, for [MLPerf Inference Benchmark](https://www.mlperf.org/inference-overview/).

## Benchmarks

The following *benchmarks* are part of our submission for MLPerf Inference:

 - **BERT** (bert)
 - **DLRM** (dlrm)
 - **ResNet50** (resnet50)
 - **SSD-ResNet34** (ssd-resnet34)

## Scenarios

Each of the above benchmarks can run in one or both of the following inference *scenarios*:

 - **Offline**
 - **Server**

Please refer to the [MLPerf Inference official page](https://www.mlperf.org/inference-overview/) for explanations about the scenarios.

## Fujitsu Submissions

Our MLPerf Inference implementation has the following submissions:

| Benchmark     | Datacenter Submissions                                        |
|---------------|---------------------------------------------------------------|
| BERT          | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, Server |
| DLRM          | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, Server |
| ResNet50      | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           |
| SSD-ResNet34  | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           |

Benchmarks are stored in the [code/](code) directory.
Every benchmark contains a `README.md` detailing instructions on how to set up that benchmark, including:

 - Downloading the dataset and model
 - Running any necessary preprocessing
 - Details on the optimizations being performed

For details on how to run each benchmark, see below.

## Fujitsu Submission Systems

The systems that Fujitsu supports, has tested, and are submitting are:

 - Datacenter system
   - A100-PCIe-40GBx4 (Fujitsu PRIMERGY GX2460 m1)
   - A100-PCIe-40GBx2 (Fujitsu PRIMERGY RX2540 m5)
   - A10-PCIe-16GBx4 (Fujitsu PRIMERGY GX2460 m1)
   - A10-PCIe-16GBx2 (Fujitsu PRIMERGY GX2460 m1)

## General Instructions

:warning: **IMPORTANT**: Please use [closed/Fujitsu](closed/Fujitsu) (this directory) as the working directory when running any of the commands below. :warning:

:warning: IMPORTANT: Please do not execute any commands or clone the repository with user `root` or `sudo`. Doing so may cause errors that involve directories being unwriteable. :warning:

**Note:** Inside the Docker container, [closed/Fujitsu](closed/Fujitsu) will be mounted at `/work`.

This section describes the steps needed to run harness with default configurations, weights, and validation datasets on Fujitsu submission systems to reproduce.
Please refer to later sections for instructions on auditing.

### Prerequisites

For x86_64 systems:

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) with `libnvidia-container>=1.3.1-1`
- Ampere-based NVIDIA GPUs
- NVIDIA Driver Version 455.xx or greater

We recommend using Ubuntu 18.04.
Other operating systems have not been tested.

### Before you run commands

Before running any commands detailed below, such as downloading and preprocessing datasets, or running any benchmarks, you should
set up the environment by doing the following:

- Run `export MLPERF_SCRATCH_PATH=<path/to/scratch/space>` to set your scratch space path.
We recommend that the scratch space has at least **3TB**.
The scratch space will be used to store models, datasets, and preprocessed datasets.

- For x86_64 systems (not Xavier): Run `make prebuild`.
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

Each [benchmark](code) contains a `README.md` that explains how to download and set up the dataset and model for that benchmark.
The following commands allow you to download all datasets and models, and preprocesses them, except for downloading the datasets needed by `ResNet50` and `DLRM` since they don't have publicly available download links.

- For `ResNet50`, please first download the [ImageNet 2012 Validation set](http://www.image-net.org/challenges/LSVRC/2012/) and unzip the images to `$MLPERF_SCRATCH_PATH/data/imagenet/`.
- For `DLRM`, please first download the [Criteo Terabyte dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) and unzip the files to `$MLPERF_SCRATCH_PATH/data/criteo/`.

Quick commands:

```
$ make download_model # Downloads models and saves to $MLPERF_SCRATCH_PATH/models
$ make download_data # Downloads datasets and saves to $MLPERF_SCRATCH_PATH/data
$ make preprocess_data # Preprocess data and saves to $MLPERF_SCRATCH_PATH/preprocessed_data
```

Notes:

- The combined preprocessed data can be huge.
- Please reserve at least **3TB** of storage in `$MLPERF_SCRATCH_PATH` to ensure you can store everything.
- By default, the `make download_model`/`make download_data`/`make preprocess_data` commands run for all the benchmarks.
Add `BENCHMARKS=resnet50`, for example, to specify a benchmark.

### Running the repository

Running models is broken down into 3 steps:

#### Build

Builds the required libraries and TensorRT plugins:

```
$ make build
```

#### Generate TensorRT engines

```
$ make generate_engines RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> --config_ver=default,high_accuracy [OTHER FLAGS]"
```

If `RUN_ARGS` is not specified, all engines for each supported benchmark-scenario pair will be built.
See [command_flags.md](command_flags.md) for information on arguments that can be used with `RUN_ARGS`.
The optimized engine files are saved to `/work/build/engines`.

#### Run harness on engines

:warning: **IMPORTANT**: The DLRM harness requires around **40GB** of free CPU memory to load the dataset.
Otherwise, running the harness will crash with `std::bad_alloc`. :warning:

:warning: **IMPORTANT**: In current version of MLPerf Inference, the default `min_duration` of harness runs is extended from 1 min to
10 min. As this is quite the large increase in runtime duration, there is now a new `--fast` flag that can be specified
in `RUN_ARGS` that is a shortcut to specify `--min_duration=60000`. In Offline and MultiStream scenarios, this also sets
`--min_query_count=1`.

```
$ make run_harness RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> --config_ver=default,high_accuracy --test_mode=<AccuracyOnly/PerformanceOnly> [OTHER FLAGS]"
```

If `RUN_ARGS` is not specified, all harnesses for each supported benchmark-scenario pair will be run.
See [command_flags.md](command_flags.md) for `RUN_ARGS` options.
Note that if an engine has not already been built for a benchmark-scenario pair (in the earlier step), this will result in an error.

The performance results will be printed to `stdout`.
Other logging will be sent to `stderr`.
LoadGen logs can be found in `/work/build/logs`.

### Notes on runtime and performance

- To achieve maximum performance for Server scenario, please set Transparent Huge Pages (THP) to *always*.
- As a shortcut, doing `make run RUN_ARGS="..."` will run `generate_engines` and `run_harness` in succession. If multiple benchmark-scenario pairs are specified, engines will only run after all engines are successfully built.
- If you get INVALID results, or if the test takes a long time to run, or for more performance tuning guidance, or if you would like to run the benchmarks on an unsupported GPU, please refer to the [performance_tuning_guide.md](performance_tuning_guide.md).

### Run code in Headless mode

If you would like to run the repository without launching the interactive docker container, follow the steps below:

- `make build_docker NO_BUILD=1` to build the docker image.
- `make docker_add_user` to create a user in the docker image. (Skip this if you run as root.)
- Run commands with `make launch_docker DOCKER_COMMAND='<COMMAND>'` where `<COMMAND>` is the command to run inside the docker. For example:
  - `make launch_docker DOCKER_COMMAND='make build'`
  - `make launch_docker DOCKER_COMMAND='make generate_engines RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline"'`
  - `make launch_docker DOCKER_COMMAND='make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline"'`
  - etc.

### Run Calibration

The calibration caches generated from default calibration set are already provided in each benchmark directory.

If you would like to re-generate the calibration cache for a specific benchmark, please run the following command:

```
$ make calibrate RUN_ARGS="--benchmarks=<BENCHMARK>"
```

See [calibration.md](calibration.md) for an explanation on how calibration is used for Fujitsu's submission.

### Update Results

Run the following command to update the LoadGen logs in `results/` with the logs in `build/logs`:

```
$ make update_results
```

Please refer to [submission_guide.md](submission_guide.md) for more detail about how to populate the logs requested by
MLPerf Inference rules under `results/`.

:warning: **IMPORTANT**: MLPerf Inference policies now have an option to allow submitters to submit an encrypted tarball of their
submission repository, and share a SHA1 of the encrypted tarball as well as the decryption password with the MLPerf
Inference results chair. This option gives submitters a more secure, private submission process.

:warning: For instructions on how to encrypt your submission, see the `Encrypting your project for submission` section
of [submission_guide.md](submission_guide.md).

### Run Compliance Tests and Update Compliance Logs

Please refer to [submission_guide.md](submission_guide.md).

## Instructions for Auditors

Please refer to the `README.md` in each benchmark directory for auditing instructions.

## Other documentations:

- [FAQ.md](FAQ.md): Frequently asked questions.
- [performance_tuning_guide.md](performance_tuning_guide.md): Instructions about how to run the benchmarks on your systems using our code base, solve the INVALID result issues, and tune the parameters for better performance.
- [submission_guide.md](submission_guide.md): Instructions about the required steps for a valid MLPerf Inference submission with or without our code base.
- [calibration.md](calibration.md): Instructions about how we did post-training quantization on activations and weights.
- [command_flags.md](command_flags.md): List of some flags supported by `RUN_ARGS`.
- [Per-benchmark READMEs](code/README.md): Instructions about how to download and preprocess the models and the datasets for each benchmarks and lists of optimizations we did for each benchmark.
