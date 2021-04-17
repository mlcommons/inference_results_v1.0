# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(mlperf-inference)

include(GNUInstallDirs)
find_package(CUDA REQUIRED)

# Build options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Pass the Loadgen include directory from command line
add_definitions(-DLOADGEN_INCLUDE_DIR=${LOADGEN_INCLUDE_DIR})

# Workaround for TRT header warning
execute_process(COMMAND echo "Warning: setting -Wno-deprecated-declarations to avoid header warnings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# Set sm versions
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_70,code=sm_70")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_72,code=sm_72")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.0)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_80,code=sm_80")
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.1)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_86,code=sm_86")
endif()

project(harness LANGUAGES CXX CUDA)

# Find the static Loadgen library
unset(LOADGEN_LIB CACHE)
find_library(LOADGEN_LIB NAMES libmlperf_loadgen.a PATHS ${LOADGEN_LIB_DIR})

# Set the path to the LWIS library
unset(LWIS_INCLUDE_DIR CACHE)
set(LWIS_INCLUDE_DIR lwis/include)

# Set the path to the Triton library
#unset(TRITON_DIR CACHE)
#set(TRITON_DIR ../../build/triton-inference-server)

# Set NVTX library path
unset(NV_TOOLS_EXT_LIB CACHE)
set(NV_TOOLS_EXT_LIB ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so)

# Build the harness for the Triton harness and for DLRM if not on "aarch64" platform
execute_process(COMMAND uname -p OUTPUT_VARIABLE ARCH)

######### DEFAULT HARNESS ########
# Add the LWIS subdirectory (which will generate a static LWIS library)
add_subdirectory(lwis)

# Build the default harness which covers single_stream, offline, and multi_stream sceanrios on image benchmarks.
execute_process(COMMAND echo "Building default harness...")
add_executable(harness_default
    harness_default/main_default.cc
    common/logger.cpp
)

target_link_libraries(harness_default
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    lwis
    ${LOADGEN_LIB}
    numa
)

target_include_directories(harness_default
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LWIS_INCLUDE_DIR}
        common
)

function(get_dali_paths DALI_INCLUDE_DIR DALI_LIB_DIR DALI_LIBRARIES)

    execute_process(
            COMMAND python3 -c "import nvidia.dali.sysconfig as dali_sc; print(dali_sc.get_include_dir(), end='')"
            OUTPUT_VARIABLE DALI_INCLUDE_DIR_LOC
            RESULT_VARIABLE DALI_INCLUDE_DIR_RESULT
    )

    execute_process(
            COMMAND python3 -c "import nvidia.dali.sysconfig as dali_sc; print(dali_sc.get_lib_dir(), end='')"
            OUTPUT_VARIABLE DALI_LIB_DIR_LOC
            RESULT_VARIABLE DALI_LIB_DIR_RESULT
    )

    if (${DALI_INCLUDE_DIR_RESULT} EQUAL "1" OR ${DALI_LIB_DIR_RESULT} EQUAL "1")
        message(FATAL_ERROR "Error acquiring DALI. Verify, that you have DALI whl installed")
    endif ()
    set(DALI_INCLUDE_DIR ${DALI_INCLUDE_DIR_LOC} PARENT_SCOPE)
    set(DALI_LIB_DIR ${DALI_LIB_DIR_LOC} PARENT_SCOPE)
    set(DALI_LIBRARIES dali dali_operators dali_kernels PARENT_SCOPE)

endfunction()

######### RNN-T HARNESS ########
execute_process(COMMAND echo "Building RNN-T harness...")

get_dali_paths(DALI_INCLUDE_DIR DALI_LIB_DIR DALI_LIBRARIES)

message(STATUS "DALI libraries DIR: " ${DALI_LIB_DIR})
message(STATUS "DALI include DIR: " ${DALI_INCLUDE_DIR})

message(STATUS "DALI linked libraries: " ${DALI_LIBRARIES})

add_executable(harness_rnnt
    harness_rnnt/main_rnnt.cc
    harness_rnnt/rnnt_kernels.cu
    common/logger.cpp
)

if (${ARCH} MATCHES "aarch64")
    target_compile_options(harness_rnnt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -gencode arch=compute_72,code=sm_72>
)
endif()

target_link_directories(harness_rnnt PRIVATE ${DALI_LIB_DIR})

target_link_libraries(harness_rnnt
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    ${LOADGEN_LIB}
    ${DALI_LIBRARIES}
)

target_include_directories(harness_rnnt
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LWIS_INCLUDE_DIR}
        ${DALI_INCLUDE_DIR}
        common
        harness_rnnt
)


######### BERT HARNESS ########
execute_process(COMMAND echo "Building BERT harness...")
add_executable(harness_bert
    harness_bert/main_bert.cc
    harness_bert/bert_server.cc
    harness_bert/bert_core_vs.cc
    common/logger.cpp
)

target_link_libraries(harness_bert
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    ${LOADGEN_LIB}
)

target_include_directories(harness_bert
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LWIS_INCLUDE_DIR}
        common
        harness_bert
)

######### DLRM HARNESS ########
if (NOT ${ARCH} MATCHES "aarch64")
    execute_process(COMMAND echo "Building DLRM harness...")
    add_executable(harness_dlrm
        harness_dlrm/main_dlrm.cc
        harness_dlrm/dlrm_server.cc
        harness_dlrm/batch_maker.cpp
        harness_dlrm/dlrm_kernels.cu
        common/logger.cpp
    )

    target_link_libraries(harness_dlrm
        nvinfer
        nvinfer_plugin
        gflags
        glog
        ${CUDA_LIBRARIES}
        ${LOADGEN_LIB}
        ${NV_TOOLS_EXT_LIB}
        numa
    )

    target_include_directories(harness_dlrm
        PUBLIC
            ${CUDA_INCLUDE_DIRS}
            ${LOADGEN_INCLUDE_DIR}
            ${LWIS_INCLUDE_DIR}
            common
            harness_dlrm
    )
else()
    execute_process(COMMAND echo "Skipping DLRM harness for ${ARCH}")
endif()
