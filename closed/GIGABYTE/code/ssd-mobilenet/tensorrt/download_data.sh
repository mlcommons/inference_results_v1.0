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

source code/common/file_downloads.sh

DATA_DIR=${DATA_DIR:-build/data}

if [[ -e ${DATA_DIR}/coco/train2017/000000000400.jpg ]] && \
    [[ -e ${DATA_DIR}/coco/val2017/000000000139.jpg ]] && \
    [[ -e ${DATA_DIR}/coco/annotations/instances_val2017.json ]]
then
    echo "Dataset for SSD networks (SSDMobileNet,SSDResNet34) already exists!"
else
    download_file data coco http://images.cocodataset.org/zips/train2017.zip train2017.zip \
        && unzip ${MLPERF_SCRATCH_PATH}/data/coco/train2017.zip -d ${MLPERF_SCRATCH_PATH}/data/coco
    download_file data coco http://images.cocodataset.org/zips/val2017.zip val2017.zip \
        && unzip ${MLPERF_SCRATCH_PATH}/data/coco/val2017.zip -d ${MLPERF_SCRATCH_PATH}/data/coco
    download_file data coco http://images.cocodataset.org/annotations/annotations_trainval2017.zip annotations_trainval2017.zip \
        && unzip ${MLPERF_SCRATCH_PATH}/data/coco/annotations_trainval2017.zip -d ${MLPERF_SCRATCH_PATH}/data/coco
fi
