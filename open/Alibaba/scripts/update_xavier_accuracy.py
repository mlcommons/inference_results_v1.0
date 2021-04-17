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
import getpass
import glob
import argparse
import json

from code.common import run_command, BENCHMARKS, SCENARIOS
from code.main import check_accuracy

"""
Since we can't install all the dependencies required by the official accuracy scripts for BERT, RNN-T, and 3D-Unet on
Xavier and XavierNX successfully, we developed our own scripts to calculate the accuracy for these benchmarks. Our
scripts should be doing identical computation as the official ones, but to avoid confusion, we will re-run the official
accuracy scripts on the mlperf_log_accuracy.json files of Xavier/XavierNX to ensure that our assumption about our
scripts is true. This script assumes that the mlperf_log_accuracy.json files in results/ are already truncated, and
their original full results have been pushed to Artifact.
"""


def main():
    print("Updating Xavier accuracy.txt files...")

    benchmark_list = [BENCHMARKS.BERT + "-99", BENCHMARKS.RNNT, BENCHMARKS.UNET + "-99", BENCHMARKS.UNET + "-99.9"]
    scenario_list = [SCENARIOS.SingleStream, SCENARIOS.Offline]
    system_list = ["AGX_Xavier_TRT", "Xavier_NX_TRT"]

    # Restore all the mlperf_log_accuracy.json files
    os.makedirs("build/artifacts", exist_ok=True)
    cmd = ("python3 scripts/restore_full_accuracy_logs.py --artifactory_username={:} --artifactory_api_key={:} "
           "--systems={:} --benchmarks={:} --scenarios={:} --test_ids= ").format(
        getpass.getuser(), os.environ["ARTIFACTORY_API_KEY"],
        ",".join(system_list), ",".join(benchmark_list), ",".join(scenario_list)
    )
    run_command(cmd)

    # Re-compute the accuracies
    for system in system_list:
        for benchmark in benchmark_list:
            for scenario in scenario_list:
                print("Processing {:}-{:}-{:}".format(system, benchmark, scenario))
                result_dir = os.path.join("results", system, benchmark, scenario, "accuracy")
                accuracy_path = os.path.join(result_dir, "accuracy.txt")
                log_path = os.path.join(result_dir, "mlperf_log_accuracy.json")

                # Get the hash for accuracy log
                hash = None
                with open(accuracy_path) as f:
                    for line in f:
                        matches = re.match(r"(hash=[0-9a-fA-F]{64})", line.rstrip())
                        if matches is None:
                            continue
                        hash = matches.group(1)
                        break
                if hash is None:
                    raise RuntimeError("Accuracy file {:} does not contain a hash!".format(accuracy_path))

                # Regenerate accuracy.txt
                config = {
                    "benchmark": benchmark.replace("-99.9", "").replace("-99", ""),
                    "accuracy_level": "99.9%" if "99.9" in benchmark else "99%",
                    "precision": "int8"
                }
                check_accuracy(log_path, config, True)

                # Add back hash
                with open(accuracy_path, "a") as f:
                    print(hash, file=f)

                print("Done with {:}-{:}-{:}".format(system, benchmark, scenario))

    print("Done Xavier accuracy.txt files...")


if __name__ == '__main__':
    main()
