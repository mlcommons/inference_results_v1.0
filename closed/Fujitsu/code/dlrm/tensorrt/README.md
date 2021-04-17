# DLRM Benchmark Setup and Usage

This benchmark performs recommendations using DLRM network.

:warning: **IMPORTANT**: Please use [closed/Fujitsu](closed/Fujitsu) as the working directory when
running the below commands. :warning:

:warning: **IMPORTANT**: This benchmark requires around 40GB of free CPU memory to load the dataset. Otherwise, running the harness
will crash with `std::bad_alloc`. :warning:

## Dataset

### Downloading / obtaining the dataset

The dataset used for this benchmark is [Criteo Terabyte dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/). Please manually download the dataset and unzip the files to `$MLPERF_SCRATCH_PATH/data/criteo/`. You can run `bash code/dlrm/tensorrt/download_data.sh` to verify if the data files are in the expected locations.

### Preprocessing the dataset for use

Please run `python3 code/dlrm/tensorrt/preprocess_data.py` to run the preprocessing.

The input click logs for 24 days are first preprocessed and saved to compact binary bin files using the standard scripts in the original [DLRM repo](https://github.com/facebookresearch/dlrm), and then converted to npy files. The numerical inputs are converted to INT8 NC/4HW4 format, and the categorical inputs are converted to INT32 format. Also, the access frequencies data are gathered from the training dataset (the first 23 days) and saved to `row_frequencies.bin`.

## Model

### Downloading / obtaining the model

The PyTorch model `tb00_40M.pt` is downloaded from the Zenodo links provided by the [MLPerf inference repository](https://github.com/mlperf/inference/tree/master/recommendation/dlrm/pytorch). We construct TensorRT network by reading layer and weight information from the PyTorch model. Details can be found in [dlrm.py](dlrm.py). You can download this model by running `bash code/dlrm/tensorrt/download_model.sh`.

Note that on the first run of the generate_engines step, the script will split the model into two parts:

- `dlrm_embedding_weights_int8_v3.bin`: the embedding tables only
- `model_test_without_embedding_weights_v3.pt`: the model without embedding tables

## Optimizations

### Plugins

The following TensorRT plugins were used to optimize DLRM benchmark:
- `DLRM_BOTTOM_MLP_TRT`: optimizes fused bottom MLP layers.
- `DLRM_INTERACTIONS_TRT`: optimizes fused embedding lookup and interaction operations.
The source codes of the plugins can be found in [../../plugin](../../plugin).

### Lower Precision

To further optimize performance, with minimal impact on segmentation accuracy, we run the computations in INT8 precision. The embedding tables are quantized by mapping the maximum value of each embedding table to `127.5`, and scaling and quantizing the values in the same embedding table with the same scaling factor. The same computation precision satisfies both accuracy targets, 99% and 99.9% of the reference FP32 accuracy.

### Embedding Table Sorting and Splitting

The quantized embedding tables are ~25GB in total. On GPUs which have sufficient memory, we place the entire embedding tables on device memory. On GPUs which does not have sufficient memory, we first sort all the rows in the embedding tables according to the access frequency data gathered from the training set, place the most frequently accessed rows on device memory, and place the rest of the rows on host memory. The ratio is controlled by the `embedding_weights_on_gpu_part` field in the `config.json` files.

## Instructions for Auditors

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=dlrm --scenarios=<SCENARIO> --config_ver=default --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=dlrm --scenarios=<SCENARIO> --config_ver=default --test_mode=AccuracyOnly"
make run RUN_ARGS="--benchmarks=dlrm --scenarios=<SCENARIO> --config_ver=high_accuracy --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=dlrm --scenarios=<SCENARIO> --config_ver=high_accuracy --test_mode=AccuracyOnly"
```

To run inference through [Triton Inference Server](https://github.com/triton-inference-server/server) and LoadGen:

```
make run RUN_ARGS="--benchmarks=dlrm --scenarios=<SCENARIO> --config_ver=triton --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=dlrm --scenarios=<SCENARIO> --config_ver=triton --test_mode=AccuracyOnly"
make run RUN_ARGS="--benchmarks=dlrm --scenarios=<SCENARIO> --config_ver=high_accuracy_triton --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=dlrm --scenarios=<SCENARIO> --config_ver=high_accuracy_triton --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

### Run with New Weights

Follow these steps to run inference with new weights:

1. Replace `build/models/dlrm/tb00_40M.pt` with new PyTorch model.
2. Remove the `build/models/dlrm/40m_limit` directory.
3. Run `make calibrate RUN_ARGS="--benchmarks=dlrm"` to generate a new calibration cache.
4. Run inference by `make run RUN_ARGS="--benchmarks=dlrm --scenarios=<SCENARIO> --config_ver=high_accuracy"`.

### Run with New Validation or Calibration Dataset

Follow these steps to run inference with new validation or calibration dataset:

1. Put the new dataset under `build/data/criteo/`.
2. Preprocess data by `python3 code/dlrm/tensorrt/preprocess_data.py`.
3. Run `make calibrate RUN_ARGS="--benchmarks=dlrm"` to generate a new calibration cache.
4. Run inference by `make run RUN_ARGS="--benchmarks=dlrm --scenarios=<SCENARIO> --config_ver=high_accuracy"`.
