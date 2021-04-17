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

if [ -e ${DATA_DIR}/squad/dev-v1.1.json ]
then
    echo "Dataset for BERT already exists!"
else
    download_file data squad https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/dev-v1.1.json dev-v1.1.json
fi
