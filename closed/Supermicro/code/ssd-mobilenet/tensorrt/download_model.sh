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

download_file models SSDMobileNet http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz SSDMobileNet.tar.gz \
    && tar -xzvf ${MLPERF_SCRATCH_PATH}/models/SSDMobileNet/SSDMobileNet.tar.gz -C ${MLPERF_SCRATCH_PATH}/models/SSDMobileNet/ --strip-components 1 ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb \
    && rm -f ${MLPERF_SCRATCH_PATH}/models/SSDMobileNet/SSDMobileNet.tar.gz
