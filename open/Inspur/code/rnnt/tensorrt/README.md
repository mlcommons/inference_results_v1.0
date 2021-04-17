# RNN-T Benchmark Setup and Usage

This benchmark performs speech recognition using RNN-T network.

:warning: **IMPORTANT**: Please use [closed/NVIDIA](closed/NVIDIA) as the working directory when
running the below commands. :warning:

## Dataset

### Downloading / obtaining the dataset

The dataset used for this benchmark is [LibriSpeech dev-clean set](http://www.openslr.org/resources/12/dev-clean.tar.gz). You can run `bash code/rnnt/tensorrt/download_data.sh` to download the dataset.

### Preprocessing the dataset for usage

The input flac files are first converted to wav files, padded to 512 time steps, and then converted to npy files in FP16 format. Please run `python3 code/rnnt/tensorrt/preprocess_data.py` to run the preprocessing.

## Model

### Downloading / obtaining the model

The PyTorch model `DistributedDataParallel_1576581068.9962234-epoch-100.pt` is downloaded from the Zenodo links provided by the [MLPerf inference repository](https://github.com/mlperf/inference/tree/master/speech_recognition/rnnt). We construct TensorRT network by reading layer and weight information from the PyTorch model. Details can be found in [rnn-t_builder.py](rnn-t_builder.py). You can download these models by running `bash code/rnnt/tensorrt/download_model.sh`.

## Optimizations

### Plugins

The following TensorRT plugins were used to optimize RNN-T benchmark:
- `RNNTSelectPlugin`: optimizes gather operation.
- `RNNTDecoderPlugin`: optimizes single-step LSTM operation for RNN-T Decoder.
The source codes of the plugins can be found in [../../plugin](../../plugin).

### CUDA kernels

The following CUDA kernels were used in the RNN-T harness:
- `greedySearch_ker`: optimizes a step in Greedy Search process, which appends the predicted token to the results, checks if the predicted token is the `<BLANK>` token, and increments the time_step of each sequence if needed.
- `rnnt_sparse_memset_ker`: optimizes conditional memset based on a bool device buffer. This is used to conditionally memset the Encoder and Decoder LSTM hidden/cell states and the time step counters when there are new sequences added to the batch.
- `fc2_top1_ker`: optimizes fused ElementWiseSum + FullyConnected + TopK operations in Joint Net.

The source codes of the CUDA kernels can be found in [../../harness/harness_rnnt/rnnt_kernels.cu](../../harness/harness_rnnt/rnnt_kernels.cu).

### Joint Net FC1 Splitting

We convert the concatenation of the Encoder output and the Decoder output, and the first FullyConnected layer in the Joint Net into two separated FullyConnected operations, one of which takes the Encoder output as the input and the other takes the Decoder output as the input. The two FullyConnected outputs are then added element-wise in the `fc2_top1_ker` kernel. This is mathematically equivalent to the original Concatenation + FullyConnected operations.

### Audio Feature Extraction with DALI

We implement the audio feature extraction (i.e. converting from waveform to the Mel spectrogram) using [NVIDIA Data LoadIng Library (DALI)](https://developer.nvidia.com/DALI). The operations and the parameters match the reference implementation.

### Batch Sorting

In Offline scenario, we sort the sequences in the incoming query according to the sequence lengths before running the inference to encourage more uniform sequence lengths within a batch and to reduce the wasted computations caused by padding. The sorting part is included in the latency measurement.

### Sequence Splitting

For each inference iteration, we only process the Encoder, the Decoder, and the GreedySearch on a batch of sequences up to a number of time steps (defined by the `max_seq_length` field in the `config.json` files) instead of the max time steps (512). This allows us to return the results of shorter sequences within the batch early and insert new sequences into the batch before the next iteration.

## Instructions for Audits

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=rnnt --scenarios=<SCENARIO> --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=rnnt --scenarios=<SCENARIO> --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.

### Run with New Weights

Follow these steps to run inference with new weights:

1. Replace `build/models/rnn-t/DistributedDataParallel_1576581068.9962234-epoch-100.pt` with new TensorFlow frozen graph.
2. Run inference by `make run RUN_ARGS="--benchmarks=rnnt --scenarios=<SCENARIO>"`.

### Run with New Validation Dataset

Follow these steps to run inference with new validation dataset:

1. Put the validation dataset under `build/data/LibriSpeech/dev-clean`.
2. Preprocess data by `python3 code/rnnt/tensorrt/preprocess_data.py`.
3. Run inference by `make run RUN_ARGS="--benchmarks=rnnt --scenarios=<SCENARIO>"`.
