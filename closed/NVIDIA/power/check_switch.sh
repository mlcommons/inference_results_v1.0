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

hex2bin () {
  local hexnum=${1^^}
  echo "obase=2; ibase=16; $hexnum" | bc
}

hex2dec () {
  local hexnum=${1^^}
  echo "obase=10; ibase=16; $hexnum" | bc
}

dec2bin () {
  local decnum=$1
  echo "obase=2; ibase=10; $decnum" | bc
}

#least significant bit being 0
read_nth_bit () {
  local decnum=$( hex2dec $1 )
  local binnum=$( dec2bin $(( $decnum >> $2 )) )
  echo "${binnum: -1}"
}

check_one_nvswitch_status () {
  local RET_CODE=""
  local LAST_BYTE=""
  local FUN_NAME="check_nvswitch_status"
  local SWITCH_ID=$1
  sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 0 0x5c 0x04 0x18 0 0 0x80 &> /dev/null
  RET_CODE=$?
  if [ $RET_CODE -eq 0 ]; then
    RET_CODE=$( sudo ipmitool raw 0x30 0x81 0x0b 0x$SWITCH_ID 5 0x5d )
    echo "$FUN_NAME 1st return code $RET_CODE"
    LAST_BYTE=$( echo $RET_CODE | awk '{print $2}' )
    BIT_7=$( read_nth_bit $LAST_BYTE 7 )
    if [ $BIT_7 -eq 0 ]; then
      echo "Switch 0x$SWITCH_ID Online"
    else
      echo "Switch 0x$SWITCH_ID Offline"
    fi
  fi
}

check_all_nvswitch_status () {
  for i in "32" "34" "36" "38" "3a" "3c"; do
    check_one_nvswitch_status "$i"
  done
}

check_all_nvswitch_status
