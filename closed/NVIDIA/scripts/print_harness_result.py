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
import os
import sys
import glob
sys.path.insert(0, os.getcwd())

import code.common.arguments as common_args
from code.common.log_parser import get_perf_summary, get_acc_summary, get_power_summary
from code.common.result_parser import from_loadgen_by_keys


def main():
    log_dir = common_args.parse_args(["log_dir"])["log_dir"]

    results = get_perf_summary(log_dir)
    print("")
    print("======================= Perf harness results: =======================")
    print("")
    for config_name in results:
        print("{:}:".format(config_name))
        for benchmark in results[config_name]:
            print("    {:}: {:}".format(benchmark, results[config_name][benchmark]))
        print("")

    results = get_acc_summary(log_dir)
    print("")
    print("======================= Accuracy results: =======================")
    print("")
    for config_name in results:
        print("{:}:".format(config_name))
        for benchmark in results[config_name]:
            print("    {:}: {:}".format(benchmark, results[config_name][benchmark]))
        print("")

    # If this is a power run, we should print out the average power
    power_vals = get_power_summary(log_dir)
    if power_vals != None:
        print("")
        print("======================= Power results: =======================")
        print("")
        for config_name in results:
            print("{:}:".format(config_name))
            for benchmark in results[config_name]:
                if len(power_vals) > 0:
                    avg_power = sum(power_vals) / len(power_vals)
                    print("    {}: avg power under load: {:.2f}W with {} power samples".format(benchmark, avg_power, len(power_vals)))
                else:
                    print("    {}: cannot find any power samples in the test window. Is the timezone setting correct?".format(benchmark))
            print("")


if __name__ == "__main__":
    main()
