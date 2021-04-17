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

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
import json

import code.common.arguments as common_args
from code.common import BENCHMARKS, SCENARIOS

__doc__ = """
Given a path to a config.json, this script will output a list of invalid keys and the lines that they appear on in the
file.

If no invalid keys are found, this process will exit with status code 0. Otherwise, this process will exit with status
code 1.
"""

# Arguments that are used to process config inheritance. Maps the name of the argument to the expected type
CONFIG_META_ARGS = {
    "extends": list,
    "scales": dict,
    "config_ver": dict,
}


def verify_config_keys(config_path):
    """
    Check if all keys inside system_ids are valid.

    Args:
        config: The full contents of a config.json file

    Returns:
        is_valid: A boolean representing if all of the keys in config are valid
        invalid_occurrences: A dict<str -> list<int>> that maps invalid strings to a list of line numbers that they
                             appear on.
    """
    with open(config_path) as f:
        contents = f.read()
    config = json.loads(contents)
    lines = contents.split("\n")

    # Define a helper method to get all invalid keys
    def _find_invalid(d, valid_keys, toplevel=True):
        invalid = []
        for k in d:
            # META_ARGS are only allowed in the top level
            if toplevel and k in CONFIG_META_ARGS:
                # Check if the meta arg is of the correct type
                if type(d[k]) is not CONFIG_META_ARGS[k]:
                    raise RuntimeError("Invalid type {} for key {}".format(type(d[k]), k))

                # Recurse on scales or config_ver, but META_ARGS cannot exist in these
                if k in ("scales", "config_ver"):
                    for nested in d[k]:
                        invalid += _find_invalid(d[k][nested], valid_keys, toplevel=False)

                # openvino parameters handling if it is valid
                if k == "ov_parameters" and k in valid_keys:
                    invalid += _find_invalid(d[k], set(common_args.OPENVINO_ARGS), toplevel=False)
            else:
                if k not in valid_keys:
                    invalid.append(k)
        return invalid

    invalid_keys = []
    for k in config:
        # Assume that config[k] represents the config for system_id 'k'
        if type(config[k]) is dict:
            valid_keys = common_args.GENERATE_ENGINE_ARGS + common_args.HARNESS_ARGS
            prefixes = ["gpu_"]

            # SystemID specific keys
            if k.startswith("Triton_CPU"):
                valid_keys += common_args.CPU_HARNESS_ARGS
                prefixes = [""]
            elif "Xavier" in k:
                prefixes = ["gpu_", "dla_", "concurrent_"]

            # Benchmark specific keys
            if config["benchmark"] == BENCHMARKS.BERT:
                valid_keys += common_args.SMALL_GEMM_KERNEL_ARGS + common_args.BERT_ARGS
                if config["scenario"] == SCENARIOS.Server:
                    valid_keys.append("server_num_issue_query_threads")
            elif config["benchmark"] == BENCHMARKS.DLRM:
                valid_keys += common_args.SMALL_GEMM_KERNEL_ARGS + common_args.DLRM_ARGS
            elif config["benchmark"] == BENCHMARKS.RNNT:
                valid_keys += common_args.RNNT_ARGS
                if config["scenario"] == SCENARIOS.Server:
                    valid_keys.append("server_num_issue_query_threads")

            # Scenario specific keys
            valid_keys += common_args.getScenarioMetricArgs(config["scenario"], prefixes=prefixes)

            invalid_keys += _find_invalid(config[k], set(valid_keys))

    # Filter out duplicates
    invalid_keys = list(set(invalid_keys))

    # Get line numbers
    invalid_occurrences = dict()
    for k in invalid_keys:
        invalid_occurrences[k] = []
        for i, line in enumerate(lines):
            if "\"{}\"".format(k) in line:
                invalid_occurrences[k].append(i + 1)

    return (len(invalid_keys) == 0), invalid_occurrences


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_fpath",
        help="Path to config.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    is_valid, invalid_occurrences = verify_config_keys(args.config_fpath)
    if is_valid:
        return

    print("Invalid keys found in config file: {}".format(args.config_fpath))
    for k in invalid_occurrences:
        print("  \"{}\" on lines: {}".format(k, ", ".join([str(i) for i in invalid_occurrences[k]])))

    # Exit with non-zero return status
    sys.exit(1)


if __name__ == "__main__":
    main()
