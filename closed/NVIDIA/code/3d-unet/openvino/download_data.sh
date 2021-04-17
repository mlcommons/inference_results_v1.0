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

if [ -e ${MLPERF_CPU_SCRATCH_PATH}/data/BraTS/MICCAI_BraTS_2019_Data_Training/HGG ]
then
    echo "Dataset for 3D-Unet looks good!"
else
    echo "!!!! BraTS 2019 training set cannot be downloaded directly. !!!!" && \
    echo "Please visit https://www.med.upenn.edu/cbica/brats2019/registration.html to download and unzip the data to ${MLPERF_CPU_SCRATCH_PATH}/data/BraTS/MICCAI_BraTS_2019_Data_Training/." && \
    echo "Directory structure:" && \
    echo "    ${MLPERF_CPU_SCRATCH_PATH}/data/BraTS/MICCAI_BraTS_2019_Data_Training/HGG" && \
    echo "    ${MLPERF_CPU_SCRATCH_PATH}/data/BraTS/MICCAI_BraTS_2019_Data_Training/LGG" && \
    echo "    ..." && false
fi
