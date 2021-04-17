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

"""
This tool is meant to mass-add config.jsons for new system ids. Because this is a mass add, the only options are to
'extend' or 'scale' another existing system_id. Provide the 'extend' or 'scale' in the --json field.
"""

import os
import sys
import argparse
import json


def update_config(benchmark, scenario, new_sys_id, json_string):
    conf_path = "configs/{:}/{:}/config.json".format(benchmark, scenario)
    new = {
        new_sys_id: json.loads(json_string)
    }
    try:
        with open(conf_path, 'r') as f:
            conf = json.load(f)
        conf.update(new)
        with open(conf_path, 'w') as f:
            json.dump(conf, f, indent=4, sort_keys=True)
        return True
    except:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system_id",
        help="String denoting the system_id used by the pipeline"
    )
    parser.add_argument(
        "--mlperf_submission_type",
        help="MLPerf inference submission type - 'DATACENTER' or 'EDGE'",
        choices=["DATACENTER", "EDGE"]
    )
    parser.add_argument(
        "--scenario",
        help="Scenario. Used to override submission type"
    )
    parser.add_argument(
        "--json",
        help="JSON string to use to fill out for the system_id json block"
    )
    args = parser.parse_args()

    if args.mlperf_submission_type == "DATACENTER":
        allowed_scenarios = ["Offline", "Server"]
        allowed_benchmarks = ["3d-unet", "bert", "dlrm", "resnet50", "rnnt", "ssd-resnet34"]
        ignore = {
            "3d-unet": ["Server"],
        }
    elif args.mlperf_submission_type == "EDGE":
        allowed_scenarios = ["Offline", "SingleStream", "MultiStream"]
        allowed_benchmarks = ["3d-unet", "bert", "resnet50", "rnnt", "ssd-mobilenet", "ssd-resnet34"]
        ignore = {
            "3d-unet": ["MultiStream"],
            "bert": ["MultiStream"],
            "rnnt": ["MultiStream"],
        }
    else:
        raise RuntimeError("Invalid submission type")

    if args.scenario is not None and len(args.scenario) > 0:
        if args.scenario not in allowed_scenarios:
            raise RuntimeError("Invalid scenario specified")
        allowed_scenarios = [args.scenario]

    for benchmark in allowed_benchmarks:
        ignored_scenarios = ignore.get(benchmark, list())
        scenarios = list(set(allowed_scenarios) - set(ignored_scenarios))
        for scenario in scenarios:
            if update_config(benchmark, scenario, args.system_id, args.json):
                print("Updated {} {}".format(benchmark, scenario))
            else:
                print("Failed to update {} {}".format(benchmark, scenario))
