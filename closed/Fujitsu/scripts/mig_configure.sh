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

# This script configures the GPU number #GPU to 7 MIG slices with profile ID 19

GPU=${1:-0}

if [ $MIG_CONF == "ALL" ]
then
GPU=$(sudo nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
MIG_CONF=1
fi

MIG_PROFILE_LIST=(OFF `sudo nvidia-smi mig -lgipp -i 0 | python3 scripts/get_mig_profiles.py`)

MIG_ENABLED=$(nvidia-smi --query-gpu=mig.mode.current --format=noheader,csv -i ${GPU} | paste -sd "," -)
if [[ $MIG_ENABLED =~ ^Enabled[,Enabled]*$ ]]
then
sudo nvidia-smi mig -cgi ${MIG_PROFILE_LIST[$MIG_CONF]} -i $GPU
sudo nvidia-smi mig -cci -i $GPU
else
echo "MIG is not enabled. Please enable MIG on $GPU and rerun"
fi
