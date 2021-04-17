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


power_monitor () {
  local LOGFILE=$1
  local GPULOG=$2
  while [ 1 -eq 1 ]; do
    date >> ${LOGFILE}
    if which ipmitool &> /dev/null; then
      sudo ipmitool sdr list >> ${LOGFILE}
    fi
    nvidia-smi --query-gpu=pstate,clocks.gr,clocks.mem,power.draw,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,display_mode,display_active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.applications_clocks_setting,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sync_boost,memory.used,utilization.gpu,utilization.memory,ecc.mode.current,enforced.power.limit,temperature.gpu --format=csv -i ${NVIDIA_VISIBLE_DEVICES} >> ${GPULOG}
  done
}

set_powercap () {
  local PL=$1

  if [ -z $NVIDIA_VISIBLE_DEVICES ]; then
    echo "set_powercap only runs inside container"
    return
  fi

  sudo nvidia-smi -pm 0 -i ${NVIDIA_VISIBLE_DEVICES}
  sleep 1
  sudo nvidia-smi -pm 1 -i ${NVIDIA_VISIBLE_DEVICES}
  sleep 1
  sudo nvidia-smi -rgc -i ${NVIDIA_VISIBLE_DEVICES}
  sleep 1
  sudo nvidia-smi -pl ${PL} -i ${NVIDIA_VISIBLE_DEVICES}
  sleep 1
  nvidia-smi
}

