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

GPU=${1:-0}

if [ $MIG_CONF == "ALL" ]
then
GPU=$(sudo nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
fi

echo "Destroying GPU #$GPU partitions"
sudo nvidia-smi mig -dci -i $GPU
sudo nvidia-smi mig -dgi -i $GPU
