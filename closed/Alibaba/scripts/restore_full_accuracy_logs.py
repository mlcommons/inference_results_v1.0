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
import subprocess
import shutil
import glob
import argparse
import datetime
import json

from scripts.utils import Tree, SortingCriteria, get_system_type


def run_command(cmd, get_output=False, tee=True, custom_env=None):
    """
    Runs a command.

    Args:
        cmd (str): The command to run.
        get_output (bool): If true, run_command will return the stdout output. Default: False.
        tee (bool): If true, captures output (if get_output is true) as well as prints output to stdout. Otherwise, does
            not print to stdout.
    """
    print("Running command: {:}".format(cmd))
    if not get_output:
        return subprocess.check_call(cmd, shell=True)
    else:
        output = []
        if custom_env is not None:
            print("Overriding Environment")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, env=custom_env)
        else:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        for line in iter(p.stdout.readline, b""):
            line = line.decode("utf-8")
            if tee:
                sys.stdout.write(line)
                sys.stdout.flush()
            output.append(line.rstrip("\n"))
        ret = p.wait()
        if ret == 0:
            return output
        else:
            raise subprocess.CalledProcessError(ret, cmd)


nightly_system_list = [
    "A100", "A100-PCIe", "A100-PCIex2", "A100_MIG_1x1g_5gb", "A100x8", "T4", "T4x8", "T4x20", "Xavier", "XavierNX"
]


def download_artifact(username, api_key, artifacts_dir, artifact_id):

    print("Checking artifact {:}...".format(artifact_id))

    # Check if it's pushed by nightly.
    matches = re.match(r"({:})-[\w-]+-\d+-[\w-]+".format("|".join(nightly_system_list)), artifact_id)
    is_L1 = matches is not None

    if is_L1:
        new_path = os.path.join(artifacts_dir, artifact_id + ".gz")
        remote_path = "L1/{:}/{:}".format(matches.group(1), artifact_id)
    else:
        old_path = os.path.join(artifacts_dir, "full-results_" + artifact_id + ".gz")
        new_path = os.path.join(artifacts_dir, artifact_id + ".gz")
        remote_path = "full_result_logs/full-results_" + artifact_id

    if os.path.exists(new_path):
        print("File {:} already exists.".format(new_path))
        return new_path

    print("Downloading artifact {:}...".format(artifact_id))
    command_fmt = "cd {:} && curl -u{:}:{:} -O \"https://urm.nvidia.com/artifactory/sw-mlpinf-generic/{:}.gz\""
    command = command_fmt.format(artifacts_dir, username, api_key, remote_path)
    run_command(command)

    if not is_L1:
        # Strip the 'full-results_' prefix
        shutil.move(old_path, new_path)

    return new_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifactory_username",
        help="Username for Artifactory",
    )
    parser.add_argument(
        "--artifactory_api_key",
        help="API key for Artifactory",
    )
    parser.add_argument(
        "--artifacts_dir",
        help="Path to directory that stores the artifacts",
        default="build/artifacts"
    )
    parser.add_argument(
        "--result_ids",
        help="Comma separated list of unique result IDs",
        default=None
    )
    parser.add_argument(
        "--metadata_file",
        help="File that stores metadata about these results",
        default="results_metadata.json"
    )
    parser.add_argument(
        "--division",
        help="Division: open/closed",
        choices=["open", "closed"],
        default="closed"
    )
    parser.add_argument(
        "--systems",
        help="Comma separated list of system IDs",
        default="*"
    )
    parser.add_argument(
        "--benchmarks",
        help="Comma separated list of benchmarks. Use official names (i.e. dlrm-99.9 instead of dlrm)",
        default="*"
    )
    parser.add_argument(
        "--scenarios",
        help="Comma separated list of scenarios.",
        default="*"
    )
    parser.add_argument(
        "--test_ids",
        help="Comma separated list of test ids (i.e. TEST01,TEST04-B).",
        default="TEST01"
    )
    return parser.parse_args()


def main():
    args = get_args()

    metadata = None
    if os.path.exists(args.metadata_file):
        with open(args.metadata_file) as f:
            metadata = json.load(f)
    metadata = Tree(starting_val=metadata)

    artifact_ids = set() if args.result_ids is None else set(args.result_ids.split(","))
    test_ids = [i for i in args.test_ids.split(",") if len(i) > 0]

    # Populate the set of all artifacts we should get
    for system in args.systems.split(","):
        for benchmark in args.benchmarks.split(","):
            for scenario in args.scenarios.split(","):
                res_id = metadata.get([system, benchmark, scenario, "accuracy", "result_id"], default=None)
                if res_id is not None:
                    artifact_ids.add(res_id)

                for test_id in test_ids:
                    res_id = metadata.get([system, benchmark, scenario, "compliance", test_id, "accuracy", "result_id"], default=None)
                    if res_id is not None:
                        artifact_ids.add(res_id)

    # Download all
    for artifact_id in artifact_ids:
        download_artifact(args.artifactory_username, args.artifactory_api_key, args.artifacts_dir, artifact_id)

    # Prepare to extract logs into build/full_results
    extract_map = {}
    for system in args.systems.split(","):
        for benchmark in args.benchmarks.split(","):
            for scenario in args.scenarios.split(","):
                res_id = metadata.get([system, benchmark, scenario, "accuracy", "result_id"], default=None)
                if res_id is None:
                    continue

                tarball_path = os.path.join(args.artifacts_dir, res_id + ".gz")
                archive_path = "build/full_results/results/{:}/{:}/{:}/accuracy/mlperf_log_accuracy.json ".format(
                    system,
                    benchmark,
                    scenario
                )

                if tarball_path in extract_map:
                    extract_map[tarball_path] += archive_path
                else:
                    extract_map[tarball_path] = archive_path

                for test_id in test_ids:
                    res_id = metadata.get([system, benchmark, scenario, "compliance", test_id, "accuracy", "result_id"], default=None)
                    if res_id is not None:
                        archive_path = "build/full_results/compliance/{:}/{:}/{:}/{:}/accuracy/mlperf_log_accuracy.json ".format(
                            system,
                            benchmark,
                            scenario,
                            test_id
                        )
                        extract_map[tarball_path] += archive_path

    # Actually extract the files
    for tarball_path in extract_map:
        archive_paths = extract_map[tarball_path]
        print("Extracting files {:} from tarball {:}...".format(archive_paths, tarball_path))
        cmd = "tar -xvzf {:} {:}".format(tarball_path, archive_paths)
        run_command(cmd)

    # Move the files to results/ and compliance/
    glob_to_logs = os.path.join("build/full_results", "**", "mlperf_log_accuracy.json")
    all_logs = glob.glob(glob_to_logs, recursive=True)
    for log in all_logs:
        dst = log.replace("build/full_results/", "")
        print("Moving {:} -> {:}".format(log, dst))
        shutil.move(log, dst)

    print("Done!")


if __name__ == '__main__':
    main()
