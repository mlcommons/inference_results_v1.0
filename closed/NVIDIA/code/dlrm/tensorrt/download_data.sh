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

if [ -e ${DATA_DIR}/criteo/day_23 ]
then
    echo "Dataset for DLRM already exists!"
else
    echo "!!!! Criteo Terabyte dataset cannot be downloaded directly. !!!!" && \
    echo "Please visit https://labs.criteo.com/2013/12/download-terabyte-click-logs/ to download and unzip the files under ${DATA_DIR}/criteo/." && \
    echo "Directory structure:" && \
    echo "    ${DATA_DIR}/criteo/day_0" && \
    echo "    ${DATA_DIR}/criteo/day_1" && \
    echo "    ${DATA_DIR}/criteo/day_2" && \
    echo "    ${DATA_DIR}/criteo/day_3" && \
    echo "    ..." && false
fi
