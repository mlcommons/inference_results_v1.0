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

LUNA_PRODUCT_NAME_LIST="DGXA100"
MACHINE_TYPE=""
NVSWITCH_IDs="32 34 36 38 3a 3c"
#########################################################

hex2bin () {
  local HEX_NUM=${1^^}
  echo "obase=2; ibase=16; $HEX_NUM" | bc
}

hex2dec () {
  local HEX_NUM=${1^^}
  echo "obase=10; ibase=16; $HEX_NUM" | bc
}

dec2bin () {
  local DEC_NUM=$1
  echo "obase=2; ibase=10; $DEC_NUM" | bc
}

dec2hex () {
  local DEC_NUM=$1
  echo "obase=16; ibase=10; $DEC_NUM" | bc
}

#least significant bit being 0
read_nth_bit () {
  local DEC_NUM=$( hex2dec $1 )
  local BIN_NUM=$( dec2bin $(( $DEC_NUM >> $2 )) )
  echo "${BIN_NUM: -1}"
}

in_the_list () {
  local LIST=$1
  local ITEM=$2
  local X=""
  for X in ${LIST}; do
    if [ "$ITEM" == "$X" ]; then
      echo $X
    fi
  done
}

find_machine_type () {
  local PRODUCTNAME=""
  local RES=""
  PRODUCTNAME=$( sudo ipmitool fru | grep 'Product Name' | head -n1 | awk '{print $4}' )
  RES=$( in_the_list "$LUNA_PRODUCT_NAME_LIST" "$PRODUCTNAME" )
  if [ ! -z "$RES" ]; then
    echo "LUNA"
    return
  fi
}

set_bmc_polling () {
  local CODE=$1
  local RET_CODE=""
  local FUN_NAME="set_bmc_polling"
  if [ $CODE -eq 1 ] || [ $CODE -eq 0 ]; then
    sudo ipmitool raw 0x30 0x11 9 0xa0 $CODE &> /dev/null
    RET_CODE=$?
    if [ $RET_CODE -eq 0 ]; then
      sudo ipmitool raw 0x30 0x11 18 1 $CODE &> /dev/null
      RET_CODE=$?
      if [ $RET_CODE -eq 0 ]; then
        echo "SUCCESS: $FUN_NAME $CODE"
        return 0
      fi
    fi
  fi
  echo "FAIL: $FUN_NAME $CODE"
  return 1
}

check_driver_is_loaded () {
  local RET_CODE=""
  local LAST_BYTE=""
  local FUN_NAME="check_driver_is_loaded"
  local SWITCH_ID=$1
  local BIT_0=""
  sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5c 0x04 0x01 0x02 0x00 0x80 &> /dev/null
  RET_CODE=$?
  if [ $RET_CODE -eq 0 ]; then
    RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5c )
    echo "$FUN_NAME 1st return code $RET_CODE"
    LAST_BYTE=$( echo $RET_CODE | awk '{print $5}' )
    if [ "$LAST_BYTE" == "1f" ] || [ "$LAST_BYTE" == "5f" ]; then
      RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5d ) 
      echo "$FUN_NAME 2st return code $RET_CODE"
      LAST_BYTE=$( echo $RET_CODE | awk '{print $2}' )
      BIT_0=$( read_nth_bit $LAST_BYTE 0 )
      if [ $BIT_0 -eq 0 ]; then
        return 0
      fi
    fi
  fi
  return 1
}

check_device_disable_is_supported () {
  local RET_CODE=""
  local LAST_BYTE=""
  local FUN_NAME="check_device_disable_is_supported"
  local SWITCH_ID=$1
  local BIT_7=""
  sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5c 0x04 0x01 0x04 0x00 0x80
  RET_CODE=$?
  if [ $RET_CODE -eq 0 ]; then
    RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5d )
    echo "$FUN_NAME return code $RET_CODE"
    LAST_BYTE=$( echo $RET_CODE | awk '{print $2}' )
    BIT_7=$( read_nth_bit $LAST_BYTE 7 )
    if [ $BIT_7 -eq 1 ]; then
      return 0
    fi
  fi
  return 1
}

check_one_nvswitch_status () {
  local RET_CODE=""
  local LAST_BYTE=""
  local FUN_NAME="check_nvswitch_status"
  local SWITCH_ID=$1
  local BIT_7=""
  sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5c 0x04 0x18 0 0 0x80 &> /dev/null
  RET_CODE=$?
  if [ $RET_CODE -eq 0 ]; then
    RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5d )
    LAST_BYTE=$( echo $RET_CODE | awk '{print $2}' )
    BIT_7=$( read_nth_bit $LAST_BYTE 7 )
    if [ $BIT_7 -eq 0 ]; then
      echo "Switch 0x$SWITCH_ID Online"
      return 0
    else
      echo "Switch 0x$SWITCH_ID Offline"
    fi
  fi
  return 1
}

reset_all_nvidia_devices () {
  sudo nvidia-smi -pm 0
  sudo systemctl stop nvidia-fabricmanager
  sudo systemctl stop dcgm
  sudo systemctl stop nvsm
  sudo systemctl stop nvidia-persistenced.service
  sudo nvidia-smi -r
  sudo nvidia-smi -pm 1
}

restart_all_nvidia_services () {
  sudo systemctl start nvidia-fabricmanager
  sudo systemctl start dcgm
  sudo systemctl start nvsm
  sudo systemctl start nvidia-persistenced.service
}

enable_nvswitches () {
  local RET_CODE=""
  local FUN_NAME="enable_nvswitches"
  local SWITCH_ID=$1
  local REQ_ID=""
  check_driver_is_loaded "$SWITCH_ID"
  RET_CODE=$?
  if [ $RET_CODE -eq 0 ]; then
    sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5d 0x04 0x02 0x00 0x00 0x00 &> /dev/null
    RET_CODE=$?
    if [ $RET_CODE -eq 0 ]; then
      sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5c 0x04 0x0e 0 0 0x80 &> /dev/null
      RET_CODE=$?
      if [ $RET_CODE -eq 0 ]; then
        RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5c )
        echo "${FUN_NAME} write_0x2 return code $RET_CODE"
        sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5c 0x04 0x10 0x00c 0 0x80 &> /dev/null
        RET_CODE=$?
        if [ $RET_CODE -eq 0 ]; then
          RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5c )
          echo "${FUN_NAME} write_0x0c 1st return code $RET_CODE"
          RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5d )
          echo "${FUN_NAME} write_0x0c 2nd return code $RET_CODE"
          REQ_ID=$( echo $RET_CODE | awk '{print $2}')
          sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5c 0x04 0x10 0xff 0x$REQ_ID 0x80 &> /dev/null
          RET_CODE=$?
          if [ $RET_CODE -eq 0 ]; then
            RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5c )
            echo "${FUN_NAME} write_0xff return code $RET_CODE"
            return 0
          fi
        fi
      fi
    fi
  fi
  return 1
}


disable_nvswitches () {
  local RET_CODE=""
  local FUN_NAME="disable_nvswitches"
  local SWITCH_ID=$1
  local REQ_ID=""
  local LAST_BYTE=""
  local BIT_1=""
  check_driver_is_loaded "$SWITCH_ID"
  RET_CODE=$?
  if [ $RET_CODE -ne 0 ]; then
    check_driver_is_loaded "$SWITCH_ID"
  fi
  if [ $RET_CODE -eq 0 ]; then
    check_device_disable_is_supported "$SWITCH_ID"
    RET_CODE=$?
    if [ $RET_CODE -eq 0 ]; then
      check_one_nvswitch_status "$SWITCH_ID"
      RET_CODE=$?
      if [ $RET_CODE -eq 0 ]; then
        sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5d 0x04 0x02 0x00 0x01 0x00 &> /dev/null
        RET_CODE=$?
        if [ $RET_CODE -eq 0 ]; then
          sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5c 0x04 0x0e 0 0 0x80 &> /dev/null
          RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5c )
          echo "${FUN_NAME} 1st return code $RET_CODE"
          sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5c 0x04 0x10 0x00c 0 0x80 &> /dev/null
          RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5c )
          echo "${FUN_NAME} 2nd return code $RET_CODE"
          RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5d )
          echo "${FUN_NAME} 3rd return code $RET_CODE"
          REQ_ID=$( echo $RET_CODE | awk '{print $2}')
          sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5c 0x04 0x10 0xff 0x$REQ_ID 0x80 &> /dev/null
          RET_CODE=$?
          if [ $RET_CODE -eq 0 ]; then
            RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5c )
            echo "${FUN_NAME} 4th return code $RET_CODE"
            sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5c 0x04 0x18 0 0 0x80 &> /dev/null
            RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5d )
            echo "${FUN_NAME} 5th return code $RET_CODE"
            LAST_BYTE=$( echo $RET_CODE | awk '{print $4}' )
            BIT_1=$( read_nth_bit $LAST_BYTE 1 )
            if [ $BIT_1 -eq 1 ]; then
              return 0
            fi
          fi
        fi
      fi
    fi
  fi
  return 1
}

unload_nvswitch_driver () {
  for d in /sys/bus/pci/drivers/nvidia-nvswitch/*/remove; do
    echo 1 | sudo tee $d
  done
  lsmod | grep nvidia
  sudo rmmod nvidia_uvm
  sudo rmmod nvidia_drm
  sudo rmmod nvidia_modeset
  sudo rmmod nvidia
  lsmod | grep nvidia
}

reload_nvswitch_driver () {
  lsmod | grep nvidia
  sudo modprobe nvidia
  sudo modprobe nvidia_modeset
  sudo modprobe nvidia_drm
  sudo modprobe nvidia_uvm
  lsmod | grep nvidia
}

check_all_nvswitch_status () {
  local nvswitch_id=""
  for nvswitch_id in $NVSWITCH_IDs; do
    check_one_nvswitch_status "$nvswitch_id" && sleep 1
  done
}


disable_switches_wrapper () {
  local RET_CODE=""
  set_bmc_polling 0
  sleep 3
  for nvswitch_id in $NVSWITCH_IDs; do
    disable_nvswitches "$nvswitch_id"
    RET_CODE=$?
    if [ $RET_CODE -ne 0 ]; then
      echo "disable_nvswitches $nvswitch_id failed"
    fi
    sleep 3
  done
  if [ $RET_CODE -eq 0 ]; then
    reset_all_nvidia_devices
    sleep 10
    check_all_nvswitch_status
    sleep 3
    check_all_nvswitch_status
    sleep 3
    unload_nvswitch_driver
    sleep 10
    reload_nvswitch_driver
  fi
  sleep 10
  set_bmc_polling 1
  sleep 3
  check_all_nvswitch_status || true
}



enable_switches_wrapper () {
  local RET_CODE=""
  set_bmc_polling 0
  for nvswitch_id in $NVSWITCH_IDs; do
    enable_nvswitches "$nvswitch_id"
    RET_CODE=$?
    if [ $RET_CODE -ne 0 ]; then
      echo "enable_nvswitches $nvswitch_id failed"
    fi
  done
  sleep 30
  echo 1 | sudo tee /sys/bus/pci/rescan
  set_bmc_polling 1
  check_all_nvswitch_status
}

main () {
  if which ipmitool &> /dev/null; then
    MACHINE_TYPE=$( find_machine_type )
    if [ "$MACHINE_TYPE" == "LUNA" ]; then
      if [ $1 -eq 0 ] || [ $1 -eq 1 ]; then
        if [ $1 -eq 1 ]; then
          check_all_nvswitch_status
          echo "Enable nvswitches"
          enable_switches_wrapper
        elif [ $1 -eq 0 ]; then
          check_all_nvswitch_status
          echo "Disable nvswitches"
          disable_switches_wrapper
        fi
      else
        echo "source set_nvswitch.bash 0#disable all nvSwitches"
        echo "source set_nvswitch.bash 1# enable all nvSwitches"
      fi
    else
      echo "Script only support LUNA"
    fi
  else
    echo "NO ipmitool"
  fi
}

main $1
