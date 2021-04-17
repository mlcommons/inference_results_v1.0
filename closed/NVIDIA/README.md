# MLPerf Inference v1.0 NVIDIA-Optimized Implementations

This is a repository of NVIDIA-optimized implementations for [MLPerf Inference Benchmark](https://www.mlperf.org/inference-overview/).
This README pertains to running the MLPerf benchmark suite on NVIDIA GPU-based platforms.
For instructions on running the MLPerf benchmark suite on CPU using Triton inference server, refer to [README_Triton_CPU.md](README_Triton_CPU.md)

## Benchmarks

The following *benchmarks* are part of our submission for MLPerf Inference:

 - **3D-Unet** (3d-unet)
 - **BERT** (bert)
 - **DLRM** (dlrm)
 - **RNN-T** (rnnt)
 - **ResNet50** (resnet50)
 - **SSD-MobileNet** (ssd-mobilenet)
 - **SSD-ResNet34** (ssd-resnet34)

## Scenarios

Each of the above benchmarks can run in one or more of the following four inference *scenarios*:

 - **SingleStream**
 - **MultiStream**
 - **Offline**
 - **Server**

Please refer to the [MLPerf Inference official page](https://www.mlperf.org/inference-overview/) for explanations about the scenarios.

## Inference Server Types

Our MLPerf submission implements multiple inference harnesses. 

- **Light-weight Inference Server** (LWIS)
- **BERT Harness**
- **DLRM Harness**
- **RNNT Harness**
- **Triton Harness**
- **Triton MIG Harness**
- **Triton CPU Harness**

### Triton Harness
This repository supports running the benchmarks using the [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) with the TensorRT Plan backend. The following benchmarks and scenarios are supported.

 - **3D-Unet** (3d-unet)
 - **BERT** (bert)
 - **DLRM** (dlrm)
 - **ResNet50** (resnet50)
 - **SSD-MobileNet** (ssd-mobilenet)
 - **SSD-ResNet34** (ssd-resnet34)

Scenarios:
 - **Offline**
 - **Server**
 - **SingleStream**
 - **MultiStream**

Please refer to the [Run Triton harness](####Run Triton harness) section for instructions on using the Triton Inference Server on GPUs.


## NVIDIA Submissions

Our MLPerf Inference implementation has the following submissions:

| Benchmark     | Datacenter Submissions                                        | Edge Submissions (Multistream may be optional)                                   |
|---------------|---------------------------------------------------------------|----------------------------------------------------------------------------------|
| 3D-UNET       | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline         | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, SingleStream              |
| BERT          | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, Server | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream                        |
| DLRM          | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, Server | None                                                                             |
| RNN-T         | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream                        |
| ResNet50      | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream, MultiStream           |
| SSD-MobileNet | None                                                          | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream, MultiStream           |
| SSD-ResNet34  | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream, MultiStream           |

Benchmarks are stored in the [code/](code) directory.
Every benchmark contains a `README.md` detailing instructions on how to set up that benchmark, including:

 - Downloading the dataset and model
 - Running any necessary preprocessing
 - Details on the optimizations being performed

For details on how to run each benchmark, see below.

## NVIDIA Submission Systems

The systems that NVIDIA supports, has tested, and are submitting are:

 - Datacenter systems
   - A100-SXM-80GBx8 (NVIDIA DGX A100, 80GB variant)
   - A100-SXM-80GBx4 (NVIDIA DGX Station A100, 80GB variant)
   - A100-PCIex8
   - A30x8 (preview)
   - A10x8 (preview)
 - Edge systems
   - A100-SXM-80GBx1, with full GPU, 1-GPC MIG variants
   - A100-PCIex1
   - A30x1 (preview)
   - A10x1 (preview)
   - AGX Xavier
   - Xavier NX

## General Instructions

:warning: **IMPORTANT**: Please use [closed/NVIDIA](closed/NVIDIA) (this directory) as the working directory when running any of the commands below. :warning:

:warning: IMPORTANT: Please do not execute any commands or clone the repository with user `root` or `sudo`. Doing so may cause errors that involve directories being unwriteable. :warning:

**Note:** Inside the Docker container, [closed/NVIDIA](closed/NVIDIA) will be mounted at `/work`.

If you are working on the MLPerf Inference open submission, use [open/NVIDIA](open/NVIDIA) instead.

This section describes the steps needed to run harness with default configurations, weights, and validation datasets on NVIDIA submission systems to reproduce.
Please refer to later sections for instructions on auditing.

### Prerequisites

For x86_64 systems:

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) with `libnvidia-container>=1.3.1-1`
- Ampere-based NVIDIA GPUs (Turing GPUs have legacy support, but are no longer maintained for optimizations)
- NVIDIA Driver Version 455.xx or greater

We recommend using Ubuntu 18.04.
Other operating systems have not been tested.

For Jetson Xavier:

- [21.03 Jetson CUDA-X AI Developer Preview](https://developer.nvidia.com/embedded/21.03-Jetson-CUDA-X-AI-Developer-Preview)
- Dependencies can be installed by running this script: [install_xavier_dependencies.sh](scripts/install_xavier_dependencies.sh). Note that this might take a while, on the order of several hours.

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
The following commands allow you to download all datasets and models, and preprocesses them, except for downloading the datasets needed by `ResNet50`, `DLRM`, and `3D-Unet` since they don't have publicly available download links.

- For `ResNet50`, please first download the [ImageNet 2012 Validation set](http://www.image-net.org/challenges/LSVRC/2012/) and unzip the images to `$MLPERF_SCRATCH_PATH/data/imagenet/`.
- For `DLRM`, please first download the [Criteo Terabyte dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) and unzip the files to `$MLPERF_SCRATCH_PATH/data/criteo/`.
- For `3D-Unet`, please first download the [BraTS 2019 Training set](https://www.med.upenn.edu/cbica/brats2019/registration.html) and unzip the data to `$MLPERF_SCRATCH_PATH/data/BraTS/MICCAI_BraTS_2019_Data_Training`.

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

#### Run Triton Harness 

To create an engine that can be run using the triton harness, use the `--config_ver=triton` as follows

```
$ make generate_engines RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> --config_ver=triton [OTHER FLAGS]"
```

or to create an engine for the high_accuracy version

```
$ make generate_engines RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> --config_ver=high_accuracy_triton [OTHER FLAGS]"
```

To run the engines using the triton harness

```
$ make run_harness RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> --config_ver=<triton/high_accuracy_triton> --test_mode=<AccuracyOnly/PerformanceOnly> [OTHER FLAGS]"
```

The constraints on CPU memory availability and `min_duration` are the same as mentioned in the [Run harness on engines](####Run harness on engines) section.

### Multi-MIG Harness

Currently, CUDA process can see no more than one Multi-Instance GPU (MIG) instance. In order to use all MIG instances in the system, in the same way multiple GPUs are utilized to speed up the workload, we implemented special harness called Multi-MIG harness. We can distribute work into N-GPUs in the system for better throughput via N or more inference streams. Here we demonstrate similar parallelism that can be obtained from up to NxM MIG instances, if each of the N GPUs is instantiated with M MIG instances.

Multi-MIG harness automatically checks MIG instances populated in the system and forks one CUDA process for each MIG instance. The parent process uses various Inter-Process Communication (IPC) to distribute work and collect results. If the system is aware of NUMA affinity, the harness also takes care of proper NUMA mapping to CPU and memory region automatically.

Limitations
- The harness requires all MIG instances to be of the same type. For example, `A100-SXM-80GBx1` system has one A100 GPU which can instantiate 7 of 1g.10gb instances at the same time. Alternatively it can instantiate 3 of 1g.10gb instances and 2 of 2g.20gb instances at the same time. Multi-MIG harness does not support the latter with mix of different MIG instances.
- The harness only supports Triton Inference Server

How to run
1. Enable MIG on all the GPUs in the system with `sudo nvidia-smi -mig 1`
2. Run `make prebuild MIG_CONF=ALL` to instantiate as many `1` GPC MIG instances as possible on all the GPUs
3. Run `make generate_engines/run_harness ... RUN_ARGS="--config_ver=triton ..."` just as any run with Triton Inference Server
4. Exit the container and the MIG instances will be destoyed automatically
5. Disable MIG with `sudo nvidia-smi -mig 0` on all the GPUs

### Notes on runtime and performance

- MultiStream scenario takes a long time to run (4-5 hours) due to the minimum query count requirement. If you would like to run it for shorter runtime, please add `--fast` to `RUN_ARGS`.
- To achieve maximum performance for Server scenario, please set Transparent Huge Pages (THP) to *always*.
- To achieve maximum performance for Server scenario on T4x8 and T4x20 systems (no longer officially supported), please lock the clock at the max frequency by `sudo nvidia-smi -lgc 1590,1590`.
- As a shortcut, doing `make run RUN_ARGS="..."` will run `generate_engines` and `run_harness` in succession. If multiple benchmark-scenario pairs are specified, engines will only run after all engines are successfully built.
- If you get INVALID results, or if the test takes a long time to run, or for more performance tuning guidance, or if you would like to run the benchmarks on an unsupported GPU, please refer to the [performance_tuning_guide.md](performance_tuning_guide.md).

### Run code on Multi Instance GPU (MIG) slice

The repository supports running on a single 1-GPC, 2-GPC, and 3-GPC MIG slice. Any other partitioning will require adding a new submission MIG system with new parameters.

1. Enable MIG on the desired GPU with `sudo nvidia-smi -mig 1 -i $GPU`
2. Run `make prebuild MIG_CONF=N` where `N` is `1`, `2`, or `3` to specify the number of GPCs per slice. This will
   create MIG instances and surface them in the docker container. If multiple MIG instances are enabled and want to
   use Multi-MIG harness or HeteroMultiUse, use `ALL`, i.e. `make prebuild MIG_CONF=ALL`.
3. Run `make generate_engines/run_harness ...` as normal
4. Exit the container. This will automatically destroy the MIG instances.
5. Disable MIG with `sudo nvidia-smi -mig 0 -i $GPU`

More instructions about running multi-MIG heterogeneous workloads (HeteroMultiUse) can be found in [launch_heterogeneous_mig.md](scripts/launch_heterogeneous_mig.md).
More information about Multi-MIG, please refer to Multi-MIG Harness section.

### Run code with power measurement

To run the harness with power measurement, please follow the steps below:

- Set the machine to the desired power mode. See the `make power_set_maxq_state` target for the power settings we use for our MaxQ submissions.
- Set up a Windows power director machine with the following requirements:
  - PTDaemon is installed in `C:\PTD\ptd-windows-x86.exe`.
  - The MLCommons [power-dev repo](https://github.com/mlcommons/power-dev) is cloned in `C:\power-dev` and is on r1.0 branch.
  - The directory `C:\ptd-logs` is created.
  - There exists an administrator user `lab` with password `labuser` or set `POWER_SERVER_USERNAME` and `POWER_SERVER_PASSWORD` in [Makefile](Makefile) to the correct credentials.
  - OpenSSH server is installed and enabled and the machine allows the user to connect with ssh on port 22.
- Set the power meter configurations in `power/server-$HOSTNAME.cfg` to the desired settings.
- Follow normal commands to generate engines and run the harnesses, except replacing `make run_harness ...` commands with `make run_harness_power ...` commands. The LoadGen logs will be located in `build/power_logs` instead of `build/logs`.
- Other commands like `make update_results` still work. The script will automatically copy the result logs and the power logs in `build/power_logs` into `results/` directory.

### Run code in Headless mode

If you would like to run the repository without launching the interactive docker container, follow the steps below:

- `make build_docker NO_BUILD=1` to build the docker image.
- `make docker_add_user` to create a user in the docker image. (Skip this if you run as root.)
- Run commands with `make launch_docker DOCKER_COMMAND='<COMMAND>'` where `<COMMAND>` is the command to run inside the docker. For example:
  - `make launch_docker DOCKER_COMMAND='make build'`
  - `make launch_docker DOCKER_COMMAND='make generate_engines RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline"'`
  - `make launch_docker DOCKER_COMMAND='make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline"'`
  - etc.
- When running with MIGs, follow these extra steps:
  - Run `make configure_mig MIG_CONF=N` before any `make launch_docker MIG_CONF=N ...` commands to configure MIGs.
  - Add `MIG_CONF=N` to all the `make launch_docker` commands, such as `make launch_docker MIG_CONF=N DOCKER_COMMAND='make generate_engines RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline"'`.
  - Run `make configure_mig MIG_CONF=N` after all the `make launch_docker MIG_CONF=N ...` commands to tear down MIGs.

### Run Calibration

The calibration caches generated from default calibration set are already provided in each benchmark directory.

If you would like to re-generate the calibration cache for a specific benchmark, please run the following command:

```
$ make calibrate RUN_ARGS="--benchmarks=<BENCHMARK>"
```

See [calibration.md](calibration.md) for an explanation on how calibration is used for NVIDIA's submission.

### Update Results

Run the following command to update the LoadGen logs in `results/` with the logs in `build/logs`:

```
$ make update_results
```

Please refer to [submission_guide.md](submission_guide.md) for more detail about how to populate the logs requested by
MLPerf Inference rules under `results/`.

:warning: **IMPORTANT**: MLPerf Inference policies now have an option to allow submitters to submit an encrypted tarball of their
submission repository, and share a SHA1 of the encrypted tarball as well as the decryption password with the MLPerf
Inference results chair. This option gives submitters a more secure, private submission process. NVIDIA highly
recommends using this new submission process to ensure fairness among submitters.

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
- [launch_heterogeneous_mig.md](scripts/launch_heterogeneous_mig.md): Instructions about running multi-MIG heterogeneous workloads (HeteroMultiUse).
- [calibration.md](calibration.md): Instructions about how we did post-training quantization on activations and weights.
- [command_flags.md](command_flags.md): List of some flags supported by `RUN_ARGS`.
- [Per-benchmark READMEs](code/README.md): Instructions about how to download and preprocess the models and the datasets for each benchmarks and lists of optimizations we did for each benchmark.
