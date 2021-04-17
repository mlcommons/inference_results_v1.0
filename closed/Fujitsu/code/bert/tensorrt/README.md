# BERT benchmark Setup and Usage

This benchmark performs language processing using BERT network.

:warning: **IMPORTANT**: Please use [closed/Fujitsu](closed/Fujitsu) as the working directory when
running the below commands. :warning:

## Dataset

### Downloading / obtaining the dataset

The dataset used for this benchmark is [SQuAD v1.1 validation set](https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/dev-v1.1.json). You can run `bash code/bert/tensorrt/download_data.sh` to download the dataset.

### Preprocessing the dataset for usage

The input contexts and questions are tokenized and converted to token_ids, segment_ids, and masks. The maximum sequence length parameter used is 384. Please run `python3 code/bert/tensorrt/preprocess_data.py` to run the preprocessing. Note that the preprocessing step requires that the model has been downloaded first.

## Model

### Downloading / obtaining the model

The ONNX model `bert_large_v1_1.onnx`, the quantized ONNX model `bert_large_v1_1_fake_quant.onnx`, and the vocabulary file `vocab.txt` are downloaded from the Zenodo links provided by the [MLPerf inference repository](https://github.com/mlperf/inference/tree/master/language/bert). We construct TensorRT network by reading layer and weight information from the ONNX model. Details can be found in [bert_var_seqlen.py](bert_var_seqlen.py). You can download these models by running `bash code/bert/tensorrt/download_model.sh`.

## Optimizations

### Plugins

The following TensorRT plugins were used to optimize BERT benchmark:
- `CustomEmbLayerNormPluginDynamic` version 2: optimizes fused embedding table lookup and LayerNorm operations.
- `CustomSkipLayerNormPluginDynamic` version 2 and 3: optimizes fused LayerNorm operation and residual connections.
- `CustomQKVToContextPluginDynamic` version 2 and 3: optimizes fused Multi-Head Attentions operation.
These plugins are available in [TensorRT 7.2](https://developer.nvidia.com/tensorrt) release.

### Lower Precision

To further optimize performance, with minimal impact on segmentation accuracy, we run the computations in INT8 precision for lower accuracy target (99% of reference FP32 accuracy).
We run the computations in FP16 precision for higher accuracy target (99.9% of reference FP32 accuracy).

### Batch Sorting

In Offline scenario, we sort the sequences in the incoming query according to the sequence lengths before running the inference to encourage more uniform sequence lengths within a batch and to reduce the wasted computations caused by padding.
The cost of this sorting is included in the latency measurement.

### Variable Sequence Lengths

We truncate the padded part of the input sequences and concatenate the truncated sequences when forming a batch to reduce the wasted computations for the padded part of the sequences.
The cost of this truncation and concatenation is included in the latency measurement.

### Soft Dropping

In Server scenario, we keep track of the histogram of the total sequence lengths of the batches at runtime. When a batch contains sequences whose total length exceeds some configurable percentile threshold (defined by `soft_drop` field in `config.json` files), we delay the inference of the batch until the end of the test.

## Instructions for Auditors

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=default --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=default --test_mode=AccuracyOnly"
make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=high_accuracy --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=high_accuracy --test_mode=AccuracyOnly"
```

To run inference through [Triton Inference Server](https://github.com/triton-inference-server/server) and LoadGen:

```
make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=triton --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=triton --test_mode=AccuracyOnly"
make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=high_accuracy_triton --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=high_accuracy_triton --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

### Run with New Weights

Follow these steps to run inference with new weights:

1. Replace `build/models/bert/bert_large_v1_1.onnx` or `build/models/bert/bert_large_v1_1_fake_quant.onnx` with new ONNX model.
2. Run inference by `make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO>"`.

### Run with New Validation Dataset

Follow these steps to run inference with new validation dataset:

1. Replace `build/data/squad/dev-v1.1.json` with the new validation dataset.
2. Preprocess data by `python3 code/bert/tensorrt/preprocess_data.py`.
3. Run inference by `make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO>"`.
