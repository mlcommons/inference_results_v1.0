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

if [[ -e ${DATA_DIR}/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac ]]
then
    echo "Dataset for RNN-T already exists!"
else
    download_file data LibriSpeech http://www.openslr.org/resources/12/dev-clean.tar.gz dev-clean.tar.gz \
        && tar -xzf ${MLPERF_SCRATCH_PATH}/data/LibriSpeech/dev-clean.tar.gz -C ${MLPERF_SCRATCH_PATH}/data
    download_file data LibriSpeech http://www.openslr.org/resources/12/train-clean-100.tar.gz train-clean-100.tar.gz \
        && tar -xzf ${MLPERF_SCRATCH_PATH}/data/LibriSpeech/train-clean-100.tar.gz -C ${MLPERF_SCRATCH_PATH}/data
fi
