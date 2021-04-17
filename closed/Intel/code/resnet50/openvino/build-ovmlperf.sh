#!/bin/bash

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]];then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}"
}
trap 'error ${LINENO}' ERR


sudo apt update
sudo apt-get install libglib2.0-dev libtbb-dev python3-dev python3-pip cmake

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
BUILD_DIRECTORY=${CUR_DIR}
SKIPS=" "
DASHES="================================================"


MLPERF_DIR=${BUILD_DIRECTORY}/MLPerf-Intel-openvino
if [ -e ${MLPERF_DIR} ]; then
	sudo rm -r ${MLPERF_DIR}
fi

DEPS_DIR=${MLPERF_DIR}/dependencies

#====================================================================
# Build OpenVINO library (If not using publicly available openvino)
#====================================================================

echo " ========== Building OpenVINO libraries ==========="
echo ${SKIPS}

OPENVINO_DIR=${DEPS_DIR}/openvino-repo
git clone https://github.com/openvinotoolkit/openvino.git ${OPENVINO_DIR}

cd ${OPENVINO_DIR}

git checkout releases/2021/2
git submodule update --init --recursive
mkdir build && cd build

cmake -DENABLE_VPU=OFF \
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

#=============================================================
#       Build gflags
#=============================================================
echo ${SKIPS}
echo " ============ Building Gflags ==========="
echo ${SKIPS}

GFLAGS_DIR=${DEPS_DIR}/gflags

git clone https://github.com/gflags/gflags.git ${GFLAGS_DIR}
cd ${GFLAGS_DIR}
mkdir gflags-build && cd gflags-build
cmake .. && make 


# Build boost
echo ${SKIPS}
echo "========= Building boost =========="
echo ${SKIPS}

BOOST_DIR=${DEPS_DIR}/boost
if [ ! -d ${BOOST_DIR} ]; then
        mkdir ${BOOST_DIR}
fi

cd ${BOOST_DIR}
wget https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz
tar -xzf boost_1_72_0.tar.gz
cd boost_1_72_0
./bootstrap.sh --with-libraries=filesystem 
./b2 --with-filesystem

#===============================================================

# Build loadgen
#===============================================================
echo ${SKIPS}
echo " =========== Building mlperf loadgenerator =========="
echo ${SKIPS}

MLPERF_INFERENCE_REPO=${DEPS_DIR}/mlperf-inference

if [ -d ${MLPERF_INFERENCE_REPO} ]; then
        rm -r ${MLPERF_INFERENCE_REPO}
fi

python3 -m pip install absl-py numpy pybind11

git clone --recurse-submodules https://github.com/mlcommons/inference.git ${MLPERF_INFERENCE_REPO}

cd ${MLPERF_INFERENCE_REPO}/loadgen
git checkout r1.0
git submodule update --init --recursive

mkdir build && cd build
cmake -DPYTHON_EXECUTABLE=`which python3` \
	..

make

cp libmlperf_loadgen.a ../

cd ${MLPERF_DIR}

# =============================================================
#        Build ov_mlperf
#==============================================================

echo ${SKIPS}
echo " ========== Building ov_mlperf ==========="
echo ${SKIPS}

SOURCE_DIR=${CUR_DIR}/src
cd ${SOURCE_DIR}

if [ -d build ]; then
	rm -r build
fi

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

echo ${SKIPS}
echo ${DASHES}
if [ -e ${SOURCE_DIR}/Release/ov_mlperf ]; then
        echo -e "\e[1;32m ov_mlperf built at ${SOURCE_DIR}/Release/ov_mlperf            \e[0m"
        echo " "
        echo " * * * Important directories * * *"
        echo -e "\e[1;32m OPENVINO_LIBRARIES=${OPENVINO_DIR}/bin/intel64/Release/lib    \e[0m"
        echo -e "\e[1;32m OPENCV_LIBRARIES=${OPENCV_DIRS[0]}/opencv/lib                 \e[0m"
        echo -e "\e[1;32m OMP_LIBRARY=${OPENVINO_DIR}/inference-engine/temp/omp/lib     \e[0m"
        echo -e "\e[1;32m BOOST_LIBRARIES=${BOOST_DIR}/boost_1_72_0/stage/lib           \e[0m"
        echo -e "\e[1;32m GFLAGS_LIBRARIES=${GFLAGS_DIR}/gflags-build/lib               \e[0m"
else
        echo -e "\e[0;31m ov_mlperf not built. Please check logs on screen\e[0m"
fi
echo ${DASHES}
