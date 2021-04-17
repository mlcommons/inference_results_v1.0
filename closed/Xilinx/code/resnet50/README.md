# Vitis-AI Benchmarking Application

This Applictions is to run [MLPerf]() classification with ResNet50-v1.5 on Xilinx the VCK5000 Versal device. 

## Setup

### Prerequisites

- Clone and build MLPerf loadgen. Follow this [link](https://github.com/mlperf/inference/blob/master/loadgen/README_BUILD.md) on how to build loadgen. Alternatively follow the below steps.
    ```sh
    git clone https://github.com/mlperf/inference.git mlperf_inference
    cd mlperf_inference
    mkdir loadgen/build/ && cd loadgen/build/
    cmake .. && cmake --build .
    cp libmlperf_loadgen.a ..
    ```
- Clone Vitis-AI Benchmark App
    ```sh
    git clone gits@xcdl190260:vitis/mlperf-vitis-benchmark-app.git
    git checkout dpuv4e
    ```

### Build Application

- make clean; make -j

The above step will generate `app.exe`. 

### Run Application

- Run `./run.sh -h` to print usage.
    ```sh
    ./run.sh -h

    Usage:
    -----------------------------------------------------------------
    ./run.sh --exe <application>
             --mode <mode-of-exe>
             --scenario <mlperf-screnario>
             --dir <image-dir>
             --nsamples <number-of-samples>

    ./run.sh --mode PerformanceOnly
    ./run.sh --mode AccuracyOnly 
    ```
- If AccuracyOnly mode is selected, mlperf will output `mlperf_log_accuracy.json`. A separate script is used to check accuracy. See `./run.sh` for more.

    ```sh
    python ${MLP_INFER_ROOT}/v0.5/classification_and_detection/tools/accuracy-imagenet.py --imagenet-val-file=${DIRECTORY}/val_map.txt --mlperf-accuracy-file=mlperf_log_accuracy.json
    ```
