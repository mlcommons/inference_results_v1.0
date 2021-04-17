#! /usr/bin/env python3
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

import argparse
import datetime
import json
import os
import sys
import glob
sys.path.insert(0, os.getcwd())

import code.common.arguments as common_args
from code.common.result_parser import from_loadgen_by_keys


def from_timestamp(timestamp):
    return datetime.datetime.strptime(timestamp, "%m-%d-%Y %H:%M:%S.%f")


def main():
    log_dir = common_args.parse_args(["log_dir"])["log_dir"]

    summary_file = os.path.join(log_dir, "perf_harness_summary.json")
    with open(summary_file) as f:
        results = json.load(f)

    print("")
    print("======================= Perf harness results: =======================")
    print("")

    for config_name in results:
        print("{:}:".format(config_name))
        for benchmark in results[config_name]:
            print("    {:}: {:}".format(benchmark, results[config_name][benchmark]))
        print("")

    summary_file = os.path.join(log_dir, "accuracy_summary.json")
    with open(summary_file) as f:
        results = json.load(f)

    print("")
    print("======================= Accuracy results: =======================")
    print("")

    for config_name in results:
        print("{:}:".format(config_name))
        for benchmark in results[config_name]:
            print("    {:}: {:}".format(benchmark, results[config_name][benchmark]))
        print("")

    # If this is a power run, we should print out the average power
    if os.path.exists(os.path.join(log_dir, "spl.txt")):
        print("")
        print("======================= Power results: =======================")
        print("")
        for config_name in results:
            print("{:}:".format(config_name))
            for benchmark in results[config_name]:
                # Get power_start and power_end
                detail_logs = glob.glob(os.path.join(log_dir, "**", "mlperf_log_detail.txt"), recursive=True)
                if len(detail_logs) == 0:
                    raise RuntimeError("Could not find detail logs for power run!")
                elif len(detail_logs) > 1:
                    print("WARNING: Power harness run contains multiple benchmark-scenario runs. This is not advised.")

                # Select the correct detail log
                scenario = config_name.split("-")[-1]
                detail_log_path = None
                for detail_log in detail_logs:
                    components = detail_log.split("/")
                    if scenario == components[-2] and benchmark == components[-3]:
                        detail_log_path = detail_log
                        break

                if detail_log_path is None:
                    raise RuntimeError("Could not find mlperf_log_detail.txt for {}-{}".format(benchmark, scenario))

                power_times = from_loadgen_by_keys(os.path.dirname(detail_log_path), ["power_begin", "power_end"])
                power_begin = from_timestamp(power_times["power_begin"])
                power_end = from_timestamp(power_times["power_end"])

                # Read power metrics from spl.txt
                with open(os.path.join(log_dir, "spl.txt")) as f:
                    lines = f.read().split("\n")

                power_vals = []
                for line in lines:
                    data = line.split(",")
                    if len(data) != 12:
                        continue

                    timestamp = data[1]
                    watts = float(data[3])
                    curr_time = from_timestamp(timestamp)

                    if power_begin <= curr_time and curr_time <= power_end:
                        power_vals.append(watts)
                avg_power = sum(power_vals) / len(power_vals)
                print("    {}: avg power under load: {:.2f}W with {} power samples".format(benchmark, avg_power, len(power_vals)))
            print("")


if __name__ == "__main__":
    main()
