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

DATA_DIR=${DATA_DIR:-build/data}

if [ -e ${DATA_DIR}/imagenet/ILSVRC2012_val_00000001.JPEG ]
then
    echo "Dataset for ResNet50 already exists!"
else
    echo "!!!! ImageNet 2012 validation set cannot be downloaded directly. !!!!" && \
    echo "Please visit http://www.image-net.org/challenges/LSVRC/2012/ to download and place the images under ${DATA_DIR}/imagenet/." && \
    echo "Directory structure:" && \
    echo "    ${DATA_DIR}/imagenet/ILSVRC2012_val_00000001.JPEG" && \
    echo "    ${DATA_DIR}/imagenet/ILSVRC2012_val_00000002.JPEG" && \
    echo "    ${DATA_DIR}/imagenet/ILSVRC2012_val_00000003.JPEG" && \
    echo "    ${DATA_DIR}/imagenet/ILSVRC2012_val_00000004.JPEG" && \
    echo "    ..." && false
fi
