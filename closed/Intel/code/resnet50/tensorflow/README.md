# Instructions for building TensorFlow and MLPerf loadgen integration

## OS

We tested it on Ubuntu (20.04 LTS) only.


## Prepare dependencies

· Python >= 3.6,

· GCC 10.1

. boost 1.7.4

. opencv 4.5.0

Install boost and opencv in a common directory (e.g., deps-installations).


## Download and build Tensorflow

### These build instructions are for inference with 8-bit integer. 
Download TF 2.3.0 release (https://github.com/tensorflow/tensorflow/releases/tag/v2.3.0)

Build TensorFlow wheel
```
bazel build --config=mkl -c opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package <some_path>/
pip install <some_path>/tensorflow-2.3.0-cp36-cp36m-linux_x86_64.whl
```

Build TensorFlow C++ API library
```
bazel build --config=mkl --config=monolithic -c opt //tensorflow:libtensorflow_cc.so
```

### These build instructions are for inference with Google bfloat16 precision
Download TF source from https://github.com/Intel-tensorflow/tensorflow/tree/bf16/base

Build TensorFlow wheel
```
bazel build --copt=-O3 --copt=-Wformat --copt=-Wformat-security --copt=-fstack-protector --copt=-fPIC --copt=-fpic --linkopt=-znoexecstack --linkopt=-zrelro --linkopt=-znow --linkopt=-fstack-protector --config=mkl --define build_with_mkl_dnn_v1_only=true --copt=-DENABLE_INTEL_MKL_BFLOAT16 --copt=-march=skylake-avx512 //tensorflow/tools/pip_package:build_pip_package
```

Build TensorFlow C++ API library
```
bazel build --copt=-O3 --copt=-Wformat --copt=-Wformat-security --copt=-fstack-protector --copt=-fPIC --copt=-fpic --linkopt=-znoexecstack --linkopt=-zrelro --linkopt=-znow --linkopt=-fstack-protector --config=mkl --config=monolithic --define build_with_mkl_dnn_v1_only=true --copt=-DENABLE_INTEL_MKL_BFLOAT16 --copt=-march=skylake-avx512 //tensorflow:libtensorflow_cc.so
``` 

### Common build instructions for both numeric precisions
Organize TensorFlow C++ library into the common installation directory (deps-installations)
```
cd deps-installations && mkdir tf-cc && cd tf-cc && mkdir ./lib
cp -r <tensorflow_installation_path>/include ./
cp <tensorflow_root_dir>/bazel-bin/tensorflow/libtensorflow_cc.so.2.*.0 ./lib/
ln -s ./lib/libtensorflow_cc.so.2.*.0 ./lib/libtensorflow_cc.so.2
ln -s ./lib/libtensorflow_cc.so.2 ./lib/libtensorflow_cc.so
```
Download and copy into tf-cc/lib/ Intel OpenMP binary and MKL-ML binary (libiomp5.so and libmklml_intel.so) from https://github.com/intel/mkl-dnn/releases/download/v0.21/mklml_lnx_2019.0.5.20190502.tgz
Replace val_map.txt in folder with default ImageNet val_map.txt

## Data

Follow instructions to download data https://github.com/mlperf/inference/tree/master/vision/classification_and_detection


## Setup MLPerf Loadgen
```
git clone --recurse-submodules https://github.com/mlperf/inference.git mlperf_inference

cd mlperf_inference/loadgen

mkdir build && cd build

cmake .. && cmake --build .
```

## Build Tensorflow Backend
```
cd loadrun
make -C ../backend clean
make -C ../backend
make clean
make
```

## Run MLPerf Inference
* Set LD_LIBRARY_PATH for boost, opencv, and TensorFlow C++ api.
* Edit shell scripts(loadrun/loadrun.sh, loadrun/netrun.sh, run_loadrun.sh) for
model and data locations.

**INT8 Offline scenario**
```
./run_loadrun.sh 28 offline 112 resnet50 224 PerformanceOnly
```
**INT8 Server scenario**
To run test with batch size 2 and 14 instances on 56 cores:
```
./run_loadrun.sh 1 server 112 resnet50 224 PerformanceOnly
```

**BF16 Offline scenario**
```
NUM_INTRA_THREADS=7 ./run_loadrun.sh  28 offline 32 resnet50 224 PerformanceOnly
```

**BF16 Server scenario**
```
NUM_INTRA_THREADS=4 ./run_loadrun.sh 1 server 56 resnet50 224 PerformanceOnly
```

For accuracy, replace PerformanceOnly with AccuracyOnly for each scenerio
