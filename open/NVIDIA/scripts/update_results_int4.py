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

import os
import sys
sys.path.insert(0, os.getcwd())

import re
import sys
import shutil
import glob
import argparse

system_list = ["A10x1", "A10x8"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-d",
        help="Specifies the directory containing the logs.",
        default="build/logs"
    )
    parser.add_argument(
        "--abort_insufficient_runs",
        help="Abort instead if there are not enough perf runs to be considered valid",
        action="store_true"
    )
    parser.add_argument(
        "--abort_missing_accuracy",
        help="Abort instead if there isn't a valid accuracy run",
        action="store_true"
    )
    args = parser.parse_args()

    for system_id in system_list:
        for benchmark in ["resnet50"]:
            for scenario in ["Offline"]:
                print(">>>>>>>> Processing {:}-{:}-{:} <<<<<<<<".format(system_id, benchmark, scenario))

                log_dirs = glob.glob(os.path.join(args.input_dir, "*", system_id + "_TRT", benchmark, scenario))

                perf_list = []
                accu_list = []
                for log_dir in log_dirs:
                    if os.path.exists(os.path.join(log_dir, "accuracy.txt")):
                        # Check if accuracy test passed
                        # Get log directory
                        base_log_dir = os.path.join(*(log_dir.split("/")[:-3]))
                        with open(os.path.join(base_log_dir, "accuracy_summary.json")) as f:
                            contents = f.read()
                        if "Accuracy test PASSED" in contents:
                            accu_list.append(log_dir)
                    else:
                        summary = os.path.join(log_dir, "mlperf_log_summary.txt")
                        with open(summary) as f:
                            for line in f:
                                match = re.match(r"Result is : (VALID|INVALID)", line)
                                if match is not None and match.group(1) == "VALID":
                                    perf_list.append(log_dir)
                                    break

                # Update accuracy run
                if len(accu_list) == 0:
                    if args.abort_missing_accuracy:
                        return
                else:
                    if len(accu_list) > 1:
                        print("WARNING: Found {:d} accuracy runs, which is more than needed. Empirically choose the last one.".format(len(accu_list)))
                        print(accu_list)
                    output_dir = os.path.join("results", system_id, benchmark, scenario, "accuracy")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    for fname in ["mlperf_log_accuracy.json", "mlperf_log_detail.txt", "mlperf_log_summary.txt"]:
                        input_file = os.path.join(accu_list[-1], fname)
                        output_file = os.path.join(output_dir, fname)
                        print("Copy {:} -> {:}".format(input_file, output_file))
                        shutil.copy(input_file, output_file)

                    input_file = os.path.join(accu_list[-1], "accuracy.txt")
                    output_file = os.path.join(output_dir, "accuracy.txt")
                    print("Copy {:} -> {:}".format(input_file, output_file))
                    shutil.copy(input_file, output_file)

                # Update perf run
                # perf_count = 1 if scenario != "Server" else 5
                perf_count = 1  # perf_count changed to 1 in MLPINF v1.0
                if len(perf_list) < perf_count:
                    if args.abort_insufficient_runs:
                        return
                elif len(perf_list) > perf_count:
                    print("WARNING: Found {:d} passing perf runs, which is more than needed. Empirically choose the last passing one(s).".format(len(perf_list)))
                    print(perf_list)
                    perf_list = perf_list[-perf_count:]

                for run_idx in range(0, len(perf_list)):
                    output_dir = os.path.join("results", system_id, benchmark, scenario, "performance", "run_{:d}".format(run_idx + 1))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    for fname in ["mlperf_log_accuracy.json", "mlperf_log_detail.txt", "mlperf_log_summary.txt"]:
                        input_file = os.path.join(perf_list[run_idx], fname)
                        output_file = os.path.join(output_dir, fname)
                        print("Copy {:} -> {:}".format(input_file, output_file))
                        shutil.copy(input_file, output_file)

    print("Done!")


if __name__ == '__main__':
    main()
