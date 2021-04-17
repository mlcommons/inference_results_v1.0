#!/bin/bash

#global vars

GIGABYTE_PRODUCT_NAME_LIST="G482-Z54-ES-NVIDIA-001 G482-Z52-00"
GOOD_RETURN_CODE=" 0a 3c 00"
LUNA_PRODUCT_NAME_LIST="DGXA100"
MACHINE_TYPE=""

#---------------------------------------------------------------------------------------------------------

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
  RES=$( in_the_list "$GIGABYTE_PRODUCT_NAME_LIST" "$PRODUCTNAME" )
  if [ ! -z "$RES" ]; then
    echo "GIGABYTE"
    return
  fi
  RES=$( in_the_list "$LUNA_PRODUCT_NAME_LIST" "$PRODUCTNAME" )
  if [ ! -z "$RES" ]; then
    echo "LUNA"
    return
  fi
}

ipmi_read () {
  date
  sudo ipmitool sdr list
}

check_fan_speed () {
  echo "Fan Label | # RPM | Status"
  echo "1st read"
  ipmi_read | grep RPM
  echo "sleep 5 sec, 2nd read"
  sleep 5
  ipmi_read | grep RPM
}

set_speed_failure () {
  echo "FAIL set $1, reset MC"
  sudo ipmitool mc reset cold
  sleep 300
  check_fan_speed
}

set_speed_success () {
  echo "SUCCEED set fan speed to $1"
  check_fan_speed
}

set_fan_manual_speed () {
  local FAN_DUTY=$1
  local RES=""
  local RET_CODE=""
  if [ "$MACHINE_TYPE" == "GIGABYTE" ]; then
    #duty cycle: 100%, 96%, 87%, 77%, 67%, 57%, 46%, 35%, 23%
    local CODE_LIST="0xFF 0xE6 0xCC 0xB3 0x99 0x80 0x66 0x4D 0x33"
    RES=$( in_the_list "$CODE_LIST" "$FAN_DUTY" )
    if [ -z $RES ]; then
      echo "UNSUPPORTED fan speed code $FAN_DUTY"
      return
    fi
    RET_CODE=$( sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0x00 0x40 0x01 $RES 0xFF )
    if [ "$RET_CODE" == "$GOOD_RETURN_CODE" ]; then
      set_speed_success ${FAN_DUTY}
    else
      set_speed_failure "Manual Fan"
    fi
  elif [ "$MACHINE_TYPE" == "LUNA" ]; then
    local CODE_LIST="90 80 70 60 50 40"
    RES=$( in_the_list "$CODE_LIST" "$FAN_DUTY" )
    if [ -z $RES ]; then
      echo "UNSUPPORTED fan speed code $FAN_DUTY"
      return
    fi
    sudo ipmitool raw 0x3c 0x73 0x1 &> /dev/null
    RET_CODE=$?
    if [ ${RET_CODE} -eq 0 ]; then
      sudo ipmitool raw 0x3c 0x74 ${FAN_DUTY} &> /dev/null
      RET_CODE=$?
      if [ ${RET_CODE} -eq 0 ]; then
        set_speed_success ${FAN_DUTY}
	return
      fi
    fi
    set_speed_failure "Manual Fan"
  else
    echo "UNSUPPORTED machine"
  fi
}

set_fan_auto () {
  local RET_CODE=""
  if [ "$MACHINE_TYPE" == "GIGABYTE" ]; then
    RET_CODE=$( sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0x00 0x40 0x00 )
    if [ "$RET_CODE" == "$GOOD_RETURN_CODE" ]; then
      set_speed_success "Auto"
    else
      set_speed_failure "Auto Fan"
    fi
  elif [ "$MACHINE_TYPE" == "LUNA" ]; then
    sudo ipmitool raw 0x3c 0x73 0x0 &> /dev/null
    RET_CODE=$?
    if [ ${RET_CODE} -eq 0 ]; then
      set_speed_success "Auto"
    else
      set_speed_failure "Auto Fan"
    fi
  else
    echo "UNSUPPORTED machine"
  fi
}


fan_test_main () {
  if which ipmitool &> /dev/null; then
    MACHINE_TYPE=$( find_machine_type )
    set_fan_manual_speed 90
    set_fan_auto
  else
    echo "NO ipmitool"
  fi
}

fan_test_main
