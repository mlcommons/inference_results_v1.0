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
import datetime
import json

from scripts.utils import Tree, SortingCriteria, get_system_type

SCENARIO_PERF_RES_METADATA = {
    # scenario: (result regex, SortingCriteria)
    "Offline": (r"Samples per second: (\d+\.?\d*e?[-+]?\d*)", SortingCriteria.Higher),
    "Server": (r"99\.00 percentile latency \(ns\)   : (\d+\.?\d*e?[-+]?\d*)", SortingCriteria.Lower),
    "SingleStream": (r"90th percentile latency \(ns\) : (\d+\.?\d*e?[-+]?\d*)", SortingCriteria.Lower),
    "MultiStream": (r"99\.00 percentile latency \(ns\)   : (\d+\.?\d*e?[-+]?\d*)", SortingCriteria.Lower),
}


def sort_perf_list(perf_file_list, scenario):
    perf_vals = []
    for perf_file in perf_file_list:
        summary_file = perf_file.replace("_accuracy.json", "_summary.txt")
        found_perf = False

        with open(summary_file) as f:
            log = f.read().split("\n")
            for line in log:
                matches = re.match(SCENARIO_PERF_RES_METADATA[scenario][0], line)
                if matches is None:
                    continue

                perf_vals.append((perf_file, float(matches.group(1))))
                found_perf = True
                break

            if not found_perf:
                raise Exception("Could not find perf value in file: " + summary_file)

    sorted_perf_vals = sorted(perf_vals, key=lambda k: k[1],
                              reverse=(SCENARIO_PERF_RES_METADATA[scenario][1] == SortingCriteria.Lower))

    return [k[0] for k in sorted_perf_vals]


def find_valid_runs(input_list, scenario):
    # Check for query constraints documented in https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc#scenarios
    QUERY_METRIC_CONSTRAINTS = {
        "Offline": (r"samples_per_query : (\d+\.?\d*e?[-+]?\d*)", 24576),
        "Server": (r"min_query_count : (\d+\.?\d*e?[-+]?\d*)", 270336),
        "MultiStream": (r"min_query_count : (\d+\.?\d*e?[-+]?\d*)", 270336),
        "SingleStream": (r"min_query_count : (\d+\.?\d*e?[-+]?\d*)", 1024),
    }

    perf_list = []
    accu_list = []
    for input_file in input_list:
        # Check if this is Accuracy run or Performance run.
        if os.path.getsize(input_file) > 4:
            accu_list.append(input_file)

        # Check for valid perf run
        is_valid = False
        satisfies_query_constraint = False
        summary = input_file.replace("_accuracy.json", "_summary.txt")
        with open(summary) as f:
            for line in f:
                # Result validity check
                match = re.match(r"Result is : (VALID|INVALID)", line)
                if match is not None and match.group(1) == "VALID":
                    is_valid = True

                # Query constraint check
                match = re.match(QUERY_METRIC_CONSTRAINTS[scenario][0], line)
                if match is not None and float(match.group(1)) >= QUERY_METRIC_CONSTRAINTS[scenario][1]:
                    satisfies_query_constraint = True

        if is_valid and satisfies_query_constraint:
            perf_list.append(input_file)

    return perf_list, accu_list


def process_results(args, system_ids, metadata):
    time_now = str(datetime.datetime.utcnow())
    result_id = args.result_id if args.result_id is not None else "manual-{:}".format(time_now)

    for system_id in system_ids:
        system_type = get_system_type(system_id)
        for benchmark in system_ids[system_id]:
            # Skip DLRM and BERT-99.9 for Edge
            if system_type == "edge" and (benchmark.startswith("dlrm") or benchmark == "bert-99.9"):
                print("{:} is an edge system. Skipping {:}".format(system_id, benchmark))
                continue

            # Skip SSD MobileNet for datacenter
            if system_type == "datacenter" and benchmark == "ssd-mobilenet":
                print("{:} is a datacenter system. Skipping {:}".format(system_id, benchmark))
                continue

            for scenario in system_ids[system_id][benchmark]:
                # Skip Server for Edge systems
                if system_type == "edge" and scenario in {"Server"}:
                    print("{:} is an edge system. Skipping Server scenario".format(system_id))
                    continue

                # Skip SingleStream and MultiStream for Datacenter systems
                if system_type == "datacenter" and scenario in {"SingleStream", "MultiStream"}:
                    print("{:} is a datacenter system. Skipping {:} scenario".format(system_id, scenario))
                    continue

                print(">>>>>>>> Processing {:}-{:}-{:} <<<<<<<<".format(system_id, benchmark, scenario))
                input_list = system_ids[system_id][benchmark][scenario]
                print("Found {:} log files".format(len(input_list)))

                perf_list, accu_list = find_valid_runs(input_list, scenario)

                # For DLRM and 3d-UNET, the 99.9% and 99% accuracy targets use the same engines. We use the same
                # logs here to make it more prominent that they are the same
                if benchmark in {"dlrm-99", "3d-unet-99"}:
                    perf_list, accu_list = find_valid_runs(system_ids[system_id][benchmark + ".9"][scenario], scenario)

                print("\t{:} perf logs".format(len(perf_list)))
                print("\t{:} acc logs".format(len(accu_list)))

                metadata.insert([system_id, benchmark, scenario, "accuracy", "count"], len(accu_list))
                metadata.insert([system_id, benchmark, scenario, "performance", "count"], len(perf_list))

                # Update accuracy run
                if len(accu_list) == 0:
                    print("WARNING: Cannot find valid accuracy run.")

                    if args.abort_missing_accuracy:
                        return
                else:
                    if len(accu_list) > 1:
                        print("WARNING: Found {:d} accuracy runs, which is more than needed. Empirically choose the last one.".format(len(accu_list)))
                        print(accu_list)
                    output_dir = os.path.join(args.output_dir, system_id, benchmark, scenario, "accuracy")
                    if not args.dry_run:
                        os.makedirs(output_dir, exist_ok=True)
                    for suffix in ["_accuracy.json", "_detail.txt", "_summary.txt"]:
                        input_file = accu_list[-1].replace("_accuracy.json", suffix)
                        output_file = os.path.join(output_dir, "mlperf_log{:}".format(suffix))
                        print("Copy {:} -> {:}".format(input_file, output_file))
                        if not args.dry_run:
                            shutil.copy(input_file, output_file)

                    input_file = os.path.join(os.path.dirname(input_file), "accuracy.txt")
                    output_file = os.path.join(output_dir, "accuracy.txt")
                    print("Copy {:} -> {:}".format(input_file, output_file))
                    if not args.dry_run:
                        shutil.copy(input_file, output_file)

                # Update perf run
                perf_count = 1
                if len(perf_list) < perf_count:
                    print("WARNING: Cannot find enough passing perf runs. Only found {:d} runs.".format(len(perf_list)))
                    if args.abort_insufficient_runs:
                        return
                elif len(perf_list) > perf_count:
                    print("WARNING: Found {:d} passing perf runs, which is more than needed. Choosing the highest perf one(s).".format(len(perf_list)))
                    perf_list = sort_perf_list(perf_list, scenario)[-perf_count:]

                starting_idx = metadata.get([system_id, benchmark, scenario, "performance", "last_updated"])
                if starting_idx is None:
                    starting_idx = 0
                else:
                    # Starting idx is in range 1..perf_count, whereas actual indices are 0..perf_count-1. We wish the
                    # first index we modify to be the one after Starting idx, so taking (N mod perf_count) works.
                    starting_idx = starting_idx % perf_count

                for run_idx in range(0, len(perf_list)):
                    run_num = ((run_idx + starting_idx) % perf_count) + 1
                    output_dir = os.path.join(args.output_dir, system_id, benchmark, scenario, "performance", "run_{:d}".format(run_num))
                    if not args.dry_run:
                        os.makedirs(output_dir, exist_ok=True)
                    for suffix in ["_accuracy.json", "_detail.txt", "_summary.txt"]:
                        input_file = perf_list[run_idx].replace("_accuracy.json", suffix)
                        output_file = os.path.join(output_dir, "mlperf_log{:}".format(suffix))
                        print("Copy {:} -> {:}".format(input_file, output_file))
                        if not args.dry_run:
                            shutil.copy(input_file, output_file)
                    metadata.insert([system_id, benchmark, scenario, "performance", "last_updated"], run_num)

                metadata.insert([system_id, benchmark, scenario, "results_export_timestamp"], time_now)
                metadata.insert([system_id, benchmark, scenario, "result_id"], result_id)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-d",
        help="Specifies the directory containing the logs.",
        default="build/logs"
    )
    parser.add_argument(
        "--output_dir", "-o",
        help="Specifies the directory to output the results/ entries to",
        default="results"
    )
    parser.add_argument(
        "--result_id",
        help="Specifies a unique ID to use for this result",
        default=None
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
    parser.add_argument(
        "--dry_run",
        help="Don't actually copy files, just log the actions taken.",
        action="store_true"
    )
    parser.add_argument(
        "--metadata_file",
        help="File that stores metadata about these results",
        default="results_metadata.json"
    )
    parser.add_argument(
        "--add_metadata",
        help="Save a field as part of metadata to the results directory. Format period.separated.key:value",
        action="append"
    )
    return parser.parse_args()


def main():
    args = get_args()

    glob_to_logs = os.path.join(args.input_dir, "**", "mlperf_log_accuracy.json")
    print("Looking for logs in {:}".format(glob_to_logs))

    all_logs = glob.glob(glob_to_logs, recursive=True)
    print("Found {:} mlperf_log entries".format(len(all_logs)))

    # Loop through input_list to find all the system_ids
    system_ids = Tree()
    for entry in all_logs:
        parts = entry.split("/")
        system_id = parts[-4]  # [input_dir]/<timestamp>/system_id/benchmark/scenario/*.json
        benchmark = parts[-3]
        scenario = parts[-2]
        system_ids.insert([system_id, benchmark, scenario], entry, append=True)

    metadata = None
    if os.path.exists(args.metadata_file):
        with open(args.metadata_file) as f:
            metadata = json.load(f)
    metadata = Tree(starting_val=metadata)

    process_results(args, system_ids, metadata)

    # Write out custom metadata
    if args.add_metadata:
        for md in args.add_metadata:
            tmp = md.split(":")
            if len(tmp) != 2:
                print("WARNING: Invalid metadata \"{:}\"".format(md))
                continue
            keyspace = tmp[0].split(".")
            value = tmp[1]
            metadata.insert(keyspace, value)

    if not args.dry_run:
        with open(args.metadata_file, 'w') as f:
            json.dump(metadata.tree, f, indent=4, sort_keys=True)
    else:
        print(json.dumps(metadata.tree, indent=4, sort_keys=True))

    print("Done!")


if __name__ == '__main__':
    main()
