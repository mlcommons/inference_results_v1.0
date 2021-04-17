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
import copy
import json
from enum import Enum, unique
from code.common.config_parser import DependencyGraph, update_nested, get_system_benchmark_config

__doc__ = """
This script will output a list of changes to the config.json calculated from the provided `old_fpath` and `new_fpath`.

`old_fpath` should be the path to a file containing the state of the config.json before proposed changes
`new_fpath` should be the path to a file containing the state of the config.json after proposed changes

Users will be shown a set of changes to configs by system_id, and be prompted to confirm that those changes are
expected. If the user enters a negative input (i.e. 'no'), the script will exit with non-zero exit code.
"""

# This script is run from a git hook, which is not supposed to be an interactive setting.
# Force stdin to take user input through command line input
sys.stdin = open("/dev/tty", "r")


@unique
class Change(Enum):
    Addition = "addition"
    Removal = "removal"
    Edit = "edit"


def find_changes(d_before, d_after):
    """
    Returns a dictionary of changes in the format:
        {
            <system id>: {
                <changed key>: <Change type>,
                ...
            },
            ...
        }
    The changes should describe the differences between d_before and d_after.
    """
    changes = dict()
    for k in d_after:
        if k not in d_before:
            changes[k] = Change.Addition
        elif type(d_before[k]) is dict and type(d_after[k]) is dict:
            nested = find_changes(d_before[k], d_after[k])
            if len(nested) > 0:
                changes[k] = nested
        elif d_before[k] != d_after[k]:
            changes[k] = Change.Edit

    # Apply removals
    for k in d_before:
        if k not in d_after:
            changes[k] = Change.Removal

    return changes


def extract_changes(d):
    """
    Returns a sub-dict of d that only contains k-v pairs where v is of type 'Change', not to be confused with
    'find_changes'.
    """
    changes = dict()
    for k in d:
        if type(d[k]) is Change:
            changes[k] = d[k]
        elif type(d[k]) is dict:
            nested = extract_changes(d[k])
            if len(nested) > 0:
                changes[k] = nested
    return changes


def get_affected_configs(config, changes):
    """
    Returns a list of system_ids affected by the diff in changes. Changes should be in the format:
        {
            <system id>: {
                <changed key>: <Change type>,
                ...
            }
        }

    The config provided is the state of config.json *BEFORE* changes are applied.
    <Change type> should be of type Change, representing the type of change that was done on

    A system_id is considered affected if the provided changes causes the final output of
    config_parser.get_system_benchmark_config to be different.
    """
    G = DependencyGraph(create_from=config)
    # candidates is all system_ids that are potentially impacted by changes
    candidates = []
    changed_system_ids = dict()
    for system_id in changes:
        # If a system is an addition or removal, automatically add to changed_system_ids
        if type(changes[system_id]) is Change:
            assert changes[system_id] in (Change.Addition, Change.Removal)
            changed_system_ids[system_id] = changes[system_id]
        else:
            candidates += G.BFS_from(system_id)
    candidates = list(set(candidates))  # Filter out duplicates

    changed_config = copy.deepcopy(config)
    update_nested(changed_config, changes)

    for candidate in candidates:
        new_conf = get_system_benchmark_config(changed_config, candidate, enforce_type_equivalence=False)
        found_changes = extract_changes(new_conf)
        if len(found_changes) > 0:
            changed_system_ids[candidate] = found_changes
    return changed_system_ids


def prompt_user(msg):
    while True:
        user_in = input(msg)
        if user_in.lower().startswith("y"):
            return True
        elif user_in.lower().startswith("n"):
            return False


def print_diffs(diff, A, B, indent_level=1):
    indent_string = " " * (4 * indent_level)

    if A is None:
        A = dict()

    if B is None:
        B = dict()

    if diff == Change.Addition:
        # If diff is an addition, this denotes a completely new system_id that exists in B but not in A. Print out each
        # key in B as an addition.
        diff = copy.deepcopy(B)
        update_nested(diff, Change.Addition)
        print("** This is a newly added system configuration")
    elif diff == Change.Removal:
        # If diff is a removal, this denotes a system_id that exists in A but not in B. Print out each key in A as a
        # removal.
        diff = copy.deepcopy(A)
        update_nested(diff, Change.Removal)
        print("** This is a removed system configuration")

    for k in diff:
        old_val = A.get(k, dict())
        new_val = B.get(k, dict())
        if type(diff[k]) is dict:
            print("{}'{}'".format(indent_string, k))
            print_diffs(diff[k], old_val, new_val, indent_level=indent_level + 1)
        elif diff[k] == Change.Addition:
            print("{}'{}': {} (NEW)".format(indent_string, k, new_val))
        elif diff[k] == Change.Removal:
            print("{}'{}': {} -- REMOVED".format(indent_string, k, old_val))
        elif diff[k] == Change.Edit:
            print("{}'{}': {} -> {}".format(indent_string, k, old_val, new_val))
        else:
            raise RuntimeError("Unexpected value at key '{}': {}".format(k, diff[k]))


def user_verify_changes(path_A, path_B):
    """
    Asks users to verify the changes to the file.

    Args:
        path_A: Path to a temp file containing the original state of the file before changes
        path_B: Path to the file containing the state of the file post-changes. Assumed to be the actual config.json in
                configs/
    """
    # Assume that path_A, the file that was already checked into the repo, exists and is well formed
    with open(path_A) as f:
        config_A = json.load(f)

    # This inheritently checks if path_B is well-formed JSON. If it is not well formed, this will throw a
    # JSONDecodeError with meaningful location on where the syntax error in the JSON.
    with open(path_B) as f:
        config_B = json.load(f)

    changes = find_changes(config_A, config_B)
    affected_sys_ids = get_affected_configs(config_A, changes)

    if len(affected_sys_ids) == 0:
        print("No changes detected")
        return

    print("Changes in {} will affect the following system IDs:".format(path_B))
    for system_id in affected_sys_ids:
        print(system_id + ":")
        old_conf = get_system_benchmark_config(config_A, system_id, enforce_type_equivalence=False)
        new_conf = get_system_benchmark_config(config_B, system_id, enforce_type_equivalence=False)
        print_diffs(affected_sys_ids[system_id], old_conf, new_conf, indent_level=1)
        if not prompt_user("Confirm that these changes are expected [y/n]: "):
            raise RuntimeError("Aborted by user.")
        print("\n============================\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "old_fpath",
        help="Path to old config.json before the changes"
    )
    parser.add_argument(
        "new_fpath",
        help="Path to new config.json after the changes"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    user_verify_changes(args.old_fpath, args.new_fpath)


if __name__ == "__main__":
    main()
