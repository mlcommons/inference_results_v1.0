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

MODEL=dlrm

download_file models ${MODEL} https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt tb00_40M.pt
mkdir -p ${MLPERF_SCRATCH_PATH}/models/${MODEL}/40m_limit

download_file preprocessed_data criteo/full_recalib https://zenodo.org/record/3941795/files/dlrm_trace_of_aggregated_samples.txt?download=1 sample_partition_trace.txt \
    && python3 code/dlrm/tensorrt/scripts/convert_dlrm_reference_partition.py \
        -i ${MLPERF_SCRATCH_PATH}/preprocessed_data/criteo/full_recalib/sample_partition_trace.txt \
        -o ${MLPERF_SCRATCH_PATH}/preprocessed_data/criteo/full_recalib/sample_partition.npy
