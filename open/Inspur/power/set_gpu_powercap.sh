#!/bin/bash



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

