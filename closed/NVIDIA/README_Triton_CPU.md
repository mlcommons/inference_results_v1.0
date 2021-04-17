# MLPerf Inference v1.0 NVIDIA-Optimized Implementations of Triton Inference Server running on CPU

In addition to demonstrating inference on NVIDIA GPUs, NVIDIA's [MLPerf Inference v1.0](https://www.mlperf.org/inference-overview/) submission also demonstrates inference on CPUs. This is achieved by using the OpenVINO backend of NVIDIA's [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), an open-source inference serving software for deploying trained AI models at scale in production. 
Triton inference server is optimized for performance and designed to provide a consistent interface when running on GPUs or CPUs.


## Summary

Start by exporting the environment variable in order to properly build and set up the container.
Note that all CPU Triton related dependencies, including backend libraries, model files, and datasets should be aggregated into one directory mapped to `MLPERF_CPU_SCRATCH_PATH` in the Makefile. The models can be built by following the directions in the [calibration_triton_cpu/](calibration_triton_cpu) directory.

```
export USE_CPU=1
make prebuild
```

Once in the container, build the harness with

```
make build_cpu
```

ResNet in offline scenario can be run using the Triton Inference Server harness with OpenVINO backend by executing:

```
make run_cpu_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=offline --config_ver=openvino --verbose" VERBOSE=1
```

To confirm accuracy, use this command:

```
make run_cpu_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=offline --config_ver=openvino --verbose --test_mode=AccuracyOnly" VERBOSE=1
```

## Benchmarks

The following benchmarks are part of our Triton CPU submission for MLPerf Inference v1.0:

 - **3D-Unet** (3d-unet)
 - **BERT** (bert)
 - **ResNet50** (resnet50)
 - **SSD-ResNet34** (ssd-resnet34)

## Scenarios

With the exception of 3D-Unet (Offline only), each of the above benchmarks can run in either of the following two inference *scenarios*:

 - **Offline**
 - **Server**

Please refer to the [MLPerf Inference official page](https://www.mlperf.org/inference-overview/) for an explanation of the scenarios.

## NVIDIA Submissions

Our MLPerf Inference v1.0 Triton CPU implementation supports the following benchmarks:

| Benchmark     | Datacenter Submissions                                        
|---------------|---------------------------------------------------------------
| 3D-UNET       | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline         
| BERT          | Accuracy: 99% of FP32<br>Scenarios: Offline, Server 
| ResNet50      | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           
| SSD-ResNet34  | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           

All models are quantized to INT8. Directions for generating the INT8 models can be found in the [calibration_triton_cpu/](calibration_triton_cpu) directory.

Models generated using the instructions described must then be copied into `MLPERF_CPU_SCRATCH_PATH` as follows:

*  <MLPERF_CPU_SCRATCH_PATH>/models/Triton/3dunet_int8_openvino/1/model.xml
*  <MLPERF_CPU_SCRATCH_PATH>/models/Triton/3dunet_int8_openvino/1/model.bin
*  <MLPERF_CPU_SCRATCH_PATH>/models/Triton/bert_int8_openvino/1/model.xml
*  <MLPERF_CPU_SCRATCH_PATH>/models/Triton/bert_int8_openvino/1/model.bin
*  <MLPERF_CPU_SCRATCH_PATH>/models/Triton/resnet50_int8_openvino/1/model.xml
*  <MLPERF_CPU_SCRATCH_PATH>/models/Triton/resnet50_int8_openvino/1/model.bin
*  <MLPERF_CPU_SCRATCH_PATH>/models/Triton/ssd-resnet34_int8_openvino/1/model.xml
*  <MLPERF_CPU_SCRATCH_PATH>/models/Triton/ssd-resnet34_int8_openvino/1/model.bin
*  <MLPERF_CPU_SCRATCH_PATH>/preprocessed_data/\<preprocessed datasets\>

To generate the preprocessed datasets, follow the benchmark-specific instructions described in the `README.md` files stored in the code/<benchmark>/openvino directories.

For details on how to run each benchmark, see below.

## NVIDIA Submission Systems

The Intel Xeon CPUs that NVIDIA supports, has tested, and are submitting are:

 - Dual Socket Xeon Gold 6258R Cascade Lake
 - Quad Socket Xeon Platinum 8380H Cooper Lake

## General Instructions

:warning: **IMPORTANT**: Please use [closed/NVIDIA](closed/NVIDIA) (this directory) as the working directory when running any of the commands below. :warning:

:warning: IMPORTANT: Please do not execute any commands or clone the repository with user `root` or `sudo`. Doing so may cause errors that involve directories being unwriteable. :warning:

**Note:** Inside the Docker container, [closed/NVIDIA](closed/NVIDIA) will be mounted at `/work`.

This section describes the steps needed to run harness with default configurations, weights, and validation datasets on NVIDIA submission systems to reproduce.
Please refer to later sections for instructions on auditing.

### Prerequisites

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) with `libnvidia-container>=1.3.1-1`
- NVIDIA Driver Version 455.xx or greater

We recommend using Ubuntu 18.04.
Other operating systems have not been tested.

### Before you run commands

Before running any commands detailed below, such as downloading and preprocessing datasets, or running any benchmarks, you should
set up the environment by doing the following:

- Run `export MLPERF_CPU_SCRATCH_PATH=<path/to/scratch/space>` to set your scratch space path.
We recommend that the scratch space has at least **1TB**.
The scratch space will be used to store models, datasets, and preprocessed datasets.

- Run `export USE_CPU=1` to enable the Triton CPU code path and run `make prebuild`.
This launches the Docker container with all the necessary packages installed.

 The docker image will have the tag `mlperf-inference:<USERNAME>-latest`.
 The source codes in the repository are located at `/work` inside the docker image.

### Download and Preprocess Datasets and Download Models

Each [benchmark](code) contains a `README.md` that explains how to download and set up the dataset and model for that benchmark.

Notes:

- The combined preprocessed data can be huge.
- Please reserve at least **1TB** of storage in `$MLPERF_CPU_SCRATCH_PATH` to ensure you can store everything.
- The preprocessed data formats can be different than what our GPU implementation uses. Refer to the respective READMEs in `code/<benchmark>/openvino` for detailed instructions. 

### Running the repository

Running models is broken down into 3 steps:

#### Build Triton Backends

This is done automatically as part of the prebuild step:

```
export USE_CPU=1
make prebuild
```

#### Build Triton Harness

Builds the required libraries and Triton harness binary:

```
$ make build_cpu
```

#### Run harness 

:warning: **IMPORTANT**: In MLPerf Inference v1.0, the default `min_duration` of harness runs was extended from 1min to
10min. As this is quite the large increase in runtime duration, there is now a new `--fast` flag that can be specified
in `RUN_ARGS` that is a shortcut to specify `--min_duration=60000`. In Offline scenario, this also sets
`--min_query_count=1`.

```
$ make run_cpu_harness RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> --config_ver=default,high_accuracy --test_mode=<AccuracyOnly/PerformanceOnly> [OTHER FLAGS]"
```

If `RUN_ARGS` is not specified, all harnesses for each supported benchmark-scenario pair will be run.
See [command_flags.md](command_flags.md) for `RUN_ARGS` options.
Note that if an engine has not already been built for a benchmark-scenario pair (in the earlier step), this will result in an error.

The performance results will be printed to `stdout`.
Other logging will be sent to `stderr`.
LoadGen logs can be found in `/work/build/logs`.

### Notes on runtime and performance

- To achieve maximum performance, please set Transparent Huge Pages (THP) to *always*.

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
- [command_flags.md](command_flags.md): List of some flags supported by `RUN_ARGS`.
- [Per-benchmark dataset READMEs](code/README.md): Instructions about how to download and preprocess the datasets for each benchmarks and lists of optimizations we did for each benchmark.
- [Per-benchmark model READMEs](calibration_triton_cpu/OpenVINO/README.md): Instructions about how to generate the CPU models.
