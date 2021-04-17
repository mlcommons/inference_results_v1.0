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

if [ -e ${MLPERF_CPU_SCRATCH_PATH}/data/imagenet/ILSVRC2012_val_00000001.JPEG ]
then
    echo "Dataset for ResNet50 already exists!"
else
    echo "!!!! ImageNet 2012 validation set cannot be downloaded directly. !!!!" && \
    echo "Please visit http://www.image-net.org/challenges/LSVRC/2012/ to download and place the images under ${MLPERF_CPU_SCRATCH_PATH}/data/imagenet/." && \
    echo "Directory structure:" && \
    echo "    ${MLPERF_CPU_SCRATCH_PATH}/data/imagenet/ILSVRC2012_val_00000001.JPEG" && \
    echo "    ${MLPERF_CPU_SCRATCH_PATH}/data/imagenet/ILSVRC2012_val_00000002.JPEG" && \
    echo "    ${MLPERF_CPU_SCRATCH_PATH}/data/imagenet/ILSVRC2012_val_00000003.JPEG" && \
    echo "    ${MLPERF_CPU_SCRATCH_PATH}/data/imagenet/ILSVRC2012_val_00000004.JPEG" && \
    echo "    ..." && false
fi
