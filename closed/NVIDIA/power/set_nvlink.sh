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

read_nvmod () {
  local RET_STR=$( lsmod | grep nvidia )
  if [ -z "$RET_STR" ]; then
    echo "No nv* mod"
  else
    echo "$RET_STR"
  fi
}

stop_nvservices () {
  sudo systemctl stop nvidia-fabricmanager
  sudo systemctl stop dcgm
  sudo systemctl stop nvsm
  sudo systemctl stop nvidia-persistenced.service
}

start_nvservices () {
  sudo systemctl start nvidia-fabricmanager
  sudo systemctl start dcgm
  sudo systemctl start nvsm
  sudo systemctl start nvidia-persistenced.service
}

unload_nvmod () {
  echo "start unload nv* mod"
  sudo rmmod nv_peer_mem
  sudo rmmod nvidia_uvm
  sudo rmmod nvidia_drm
  sudo rmmod nvidia_modeset
  sudo rmmod nvidia
  read_nvmod
}

reload_nvmod () {
  echo "start reload nv* mod"
  sudo modprobe nvidia_modeset
  sudo modprobe nvidia_drm
  sudo modprobe nvidia_uvm
  sudo modprobe nv_peer_mem
  read_nvmod
}

disable_nvlinks () {
  sudo nvidia-smi -pm 0
  stop_nvservices
  unload_nvmod
  sudo modprobe nvidia "NVreg_NvLinkDisable=0x1"
  reload_nvmod
  sudo nvidia-smi -r
  sudo nvidia-smi -pm 1
}

enable_nvlinks () {
  sudo nvidia-smi -pm 0
  stop_nvservices
  unload_nvmod
  sudo modprobe nvidia "NVreg_NvLinkDisable=0x0"
  reload_nvmod
  sudo nvidia-smi -r 
  sudo nvidia-smi -pm 1
}

check_nvlinks () {
  local RET_STR=""
  RET_STR=$( nvidia-smi nvlink -s )
  if [ -z "$RET_STR" ]; then
    echo "NV LINK OFFLINE"
  else
    echo "$RET_STR"
  fi
}

main () {
  if [ $1 -eq 0 ] || [ $1 -eq 1 ]; then
    if [ $1 -eq 1 ]; then
      check_nvlinks
      enable_nvlinks
      check_nvlinks
    elif [ $1 -eq 0 ]; then
      check_nvlinks
      disable_nvlinks
      check_nvlinks
    fi
  else
    echo "source set_nvlink.bash 0 # disable all nvlinks"
    echo "source set_nvlink.bash 1 # enable all nvlinks"
  fi
}

main $1
