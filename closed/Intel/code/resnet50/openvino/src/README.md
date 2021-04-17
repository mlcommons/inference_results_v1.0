# MLPerf Mobile Segment with OpenVINO

## Requirements

+ [cmake](https://cmake.org/download/): 3.15.4 or higher
+ [Boost](https://www.boost.org/users/history/version_1_72_0.html): 1.72.0
+ Numpy

## Build Instructions

```
    SOURCE_DIR=${PWD}
    BUILD_DIR=${PWD}/MLPerf-Intel-openvino
    cd ..
    mkdir ${BUILD_DIR} && cd ${BUILD_DIR}
    
    sudo apt-get install libglib2.0-dev libtbb-dev python3-dev python3-pip cmake
```

### Build OpenVINO Libraries


```
    OPENVINO_DIR=${BUILD_DIR}/openvino-repo
    git clone https://github.com/openvinotoolkit/openvino.git ${OPENVINO_DIR}
    
    cd ${OPENVINO_DIR}
    
    git checkout releases/2021/2
    git submodule update --init --recursive
    mkdir build && cd build
    
    cmake -DENABLE_VPU=OFF \
            -DENABLE_CLDNN=OFF \
            -DTHREADING=OMP \
            -DENABLE_GNA=OFF \
            -DENABLE_DLIA=OFF \
            -DENABLE_TESTS=OFF \
            -DENABLE_VALIDATION_SET=OFF \
            -DNGRAPH_ONNX_IMPORT_ENABLE=OFF \
            -DNGRAPH_DEPRECATED_ENABLE=FALSE \
            -DPYTHON_EXECUTABLE=`which python3` \
            ..
    
    TEMPCV_DIR=${OPENVINO_DIR}/inference-engine/temp/opencv_4*
    OPENCV_DIRS=$(ls -d -1 ${TEMPCV_DIR} )
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OPENCV_DIRS[0]}/opencv/lib
    
    make -j$(nproc)
    cd ${BUILD_DIR}
```

### Build Gflags

```
    GFLAGS_DIR=${BUILD_DIR}/gflags
    git clone https://github.com/gflags/gflags.git ${GFLAGS_DIR}
    cd ${GFLAGS_DIR}
    mkdir gflags-build && cd gflags-build
    cmake ..
    make 
    cd ${BUILD_DIR}
```

### Build Boost-Filesystem
    
```
    BOOST_DIR=${BUILD_DIR}/boost
    cd ${BOOST_DIR}
    wget https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz
    tar -xzvf boost_1_72_0.tar.gz
    cd boost_1_72_0
    ./bootstrap.sh --with-libraries=filesystem 
    ./b2 --with-filesystem
    cd ${BOOST_DIR}
```

### Build MLPerf Loadgen

```
    MLPERF_INFERENCE_REPO=${BUILD_DIR}/mlperf-inference
    pip3 install absl-py numpy pybind11
    git clone --recurse-submodules https://github.com/mlcommons/inference.git${MLPERF_INFERENCE_REPO}
    cd ${MLPERF_INFERENCE_REPO}/loadgen
    git checkout r1.0
    git submodule update --init --recursive
    mkdir build && cd build
    cmake -DPYTHON_EXECUTABLE=$(which python3) .. && make
    cp libmlperf_loadgen.a ../
    cd ${BUILD_DIR}
    
```

### Build MLPerf-OpenVINO Backend (For pre-built version of OpenVINO - See Below for self-compiled OpenVINO)

```
    cd ${SOURCE_DIR}
    mkdir build && cd build
    cmake -DInferenceEngine_DIR=${OPENVINO_DIR}/build/ \
            -DOpenCV_DIR=${OPENCV_DIRS[0]}/opencv/cmake/ \
            -DLOADGEN_DIR=${MLPERF_INFERENCE_REPO}/loadgen \
            -DBOOST_INCLUDE_DIRS=${BOOST_DIR}/boost_1_72_0 \
            -DBOOST_FILESYSTEM_LIB=${BOOST_DIR}/boost_1_72_0/stage/lib/libboost_filesystem.so \
            -DCMAKE_BUILD_TYPE=Release \
            -Dgflags_DIR=${GFLAGS_DIR}/gflags-build/ \
            ..
    
    make
    
```

+ Built binary is located at ```${SOURCE_DIR}/Release/ov_mlperf```

## Known Issues


