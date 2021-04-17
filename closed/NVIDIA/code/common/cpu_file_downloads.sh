#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

MLPERF_CPU_SCRATCH_PATH=${MLPERF_CPU_SCRATCH_PATH:-/home/scratch.mlperf_inference_triton_cpu_data}

# $1 indicates the subdirectory (i.e. models or data)
# $2 indicates the benchmark name (i.e. ResNet50, dlrm, etc.)
# $3 indicates the URL to download from
# $4 indicates the destination filename
function download_file {
    _SUB_DIR=${MLPERF_CPU_SCRATCH_PATH}/$1/$2

    if [ ! -d ${_SUB_DIR} ]; then
        echo "Creating directory ${_SUB_DIR}"
        mkdir -p ${_SUB_DIR}
    fi
    echo "Downloading $2 $1..." \
        && wget $3 -O ${_SUB_DIR}/$4 \
        && echo "Saved $2 $1 to ${_SUB_DIR}/$4!"
}
