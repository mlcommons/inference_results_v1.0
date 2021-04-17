# BERT Benchmark

## General Information

:warning: **IMPORTANT**: Please use [closed/NVIDIA](closed/NVIDIA) as the working directory when
running the below commands. :warning:

### Goal of this Benchmark

This benchmark performs language processing using BERT network.

## Dataset

### Downloading the dataset

The dataset used for this benchmark is [SQuAD v1.1 validation set](https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/dev-v1.1.json). Download this alongside the vocabulary list [here].(https://zenodo.org/record/3750364/files/vocab.txt?download=1). You can run `bash code/bert/openvino/download_data.sh` to download the data to their expected directories.

### Preprocessing the dataset

The input contexts and questions are tokenized and converted to token_ids, segment_ids, and masks. The maximum sequence length parameter used is 384. Please run `python3 code/bert/openvino/preprocess_data.py` to run the preprocessing. Note that the preprocessing step requires that the model has been downloaded first.

The file structure expected by the harness are as follows:

```
<MLPERF_CPU_SCRATCH_PATH>/data/squad/vocab.txt
<MLPERF_CPU_SCRATCH_PATH>/data/squad/dev-v1.1.json
<MLPERF_CPU_SCRATCH_PATH>/preprocessed_data/squad_tokenized/input_ids.npy
<MLPERF_CPU_SCRATCH_PATH>/preprocessed_data/squad_tokenized/input_mask.npy
<MLPERF_CPU_SCRATCH_PATH>/preprocessed_data/squad_tokenized/input_segment_ids.npy
```

## Model

### Generating model binaries and running INT8 calibration

For generating the required INT8 OpenVino model binaries, follow the instructions in the [calibration documentation./](../../../calibration_triton_cpu/OpenVINO/bert/README.md) 

## Instructions for Auditors

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen on CPU using [Triton inference server](https://github.com/triton-inference-server/server):

```
make run_cpu_harness RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=openvino --test_mode=PerformanceOnly"
make run_cpu_harness RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=openvino --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

