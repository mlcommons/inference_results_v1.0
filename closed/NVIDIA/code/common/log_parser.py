#!/usr/bin/env python3
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

import datetime
import json
import glob
import os
import sys
sys.path.insert(0, os.getcwd())

from code.common.result_parser import from_loadgen_by_keys


def from_timestamp(timestamp):
    return datetime.datetime.strptime(timestamp, "%m-%d-%Y %H:%M:%S.%f")


def get_perf_summary(log_dir):
    """
    Returns the contents of perf_harness_summary.json as a dict with structure:

    {
        <config_name>: {
            <benchmark name>: <result string>,
            ...
        },
        ...
    }
    """
    summary_path = os.path.join(log_dir, "perf_harness_summary.json")
    if not os.path.exists(summary_path):
        return None

    with open(summary_path) as f:
        results = json.load(f)
    return results


def get_acc_summary(log_dir):
    """
    Returns the contents of accuracy_summary.json as a dict with structure:

    {
        <config_name>: {
            <benchmark name>: <result string>,
            ...
        },
        ...
    }
    """
    summary_path = os.path.join(log_dir, "accuracy_summary.json")
    if not os.path.exists(summary_path):
        return None

    with open(summary_path) as f:
        results = json.load(f)
    return results


def get_power_summary(log_dir):
    """
    Returns a list of power wattages from between power_begin and power_end for the spl.txt located in log_dir. Note
    that this does not support directories where there are multiple power harness runs in a single log_dir. Running
    multiple power harnesses in a single harness run is not advised or officially supported.
    """
    spl_path = os.path.join(log_dir, "spl.txt")
    if not os.path.exists(spl_path):
        return None

    detail_log_path = None
    if log_dir.startswith("results"):
        # In results, mlperf_log_summary would be in the same directory as spl.txt at:
        # results/<system name>/<benchmark>/<scenario>/run_1/mlperf_log_detail.txt
        detail_log_path = os.path.join(log_dir, "mlperf_log_detail.txt")
    else:
        # In a harness run log directory, mlperf_log_detail would be at:
        # build/power_logs/<timestamp>/run_1/<system name>/<benchmark>/<scenario>/mlperf_log_detail.txt
        # spl would be in: build/power_logs/<timestamp>/run_1/spl.txt
        detail_logs = glob.glob(os.path.join(log_dir, "**", "mlperf_log_detail.txt"), recursive=True)
        if len(detail_logs) == 0:
            raise RuntimeError("Could not find detail logs for power run!")
        elif len(detail_logs) > 1:
            print("WARNING: Power harness run contains multiple benchmark-scenario runs. This is not advised.")
            scenario = config_name.split("-")[-1]
            for detail_log in detail_logs:
                components = detail_log.split("/")
                if scenario == components[-2] and benchmark in components[-3]:
                    detail_log_path = detail_log
                    break
        else:
            detail_log_path = detail_logs[0]

    if detail_log_path is None or not os.path.exists(detail_log_path):
        return None

    power_times = from_loadgen_by_keys(os.path.dirname(detail_log_path), ["power_begin", "power_end"])
    power_begin = from_timestamp(power_times["power_begin"])
    power_end = from_timestamp(power_times["power_end"])

    # Read power metrics from spl.txt
    with open(os.path.join(log_dir, "spl.txt")) as f:
        lines = f.read().split("\n")

    power_vals = []
    for line in lines:
        data = line.split(",")

        if len(data) < 4:
            continue

        timestamp = data[1]
        watts = float(data[3])
        curr_time = from_timestamp(timestamp)

        if power_begin <= curr_time and curr_time <= power_end:
            power_vals.append(watts)

    return power_vals
