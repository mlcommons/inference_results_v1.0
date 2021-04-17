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
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i",
        help="Specifies the tab-separated file for system descriptions.",
        default="systems/system_descriptions.tsv"
    )
    parser.add_argument(
        "--dry_run",
        help="Don't actually copy files, just log the actions taken.",
        action="store_true"
    )
    return parser.parse_args()


def main():
    args = get_args()

    input_file = args.input
    print("Generating system description json files from {:}".format(input_file))

    open_systems = {}
    closed_systems = {}
    with open(input_file) as f:
        for (idx, line) in enumerate(f):
            line = line.rstrip("\n")
            if idx == 0:
                keys = line.split("\t")
            elif line != "":
                fields = line.split("\t")
                # First field is system id
                system_id = fields[0]
                if fields[2] == "closed":
                    closed_systems[system_id] = {}
                    for (field_idx, field) in enumerate(fields[1:]):
                        closed_systems[system_id][keys[field_idx + 1]] = field
                else:
                    open_systems[system_id] = {}
                    for (field_idx, field) in enumerate(fields[1:]):
                        open_systems[system_id][keys[field_idx + 1]] = field

    print("Found {:d} systems".format(len(open_systems) + len(closed_systems)))

    for system_id in open_systems:
        system = open_systems[system_id]
        output_file = os.path.join("..", "..", system["division"], system["submitter"], "systems", "{:}.json".format(system_id))
        print("Generating {:}".format(output_file))
        if not args.dry_run:
            with open(output_file, 'w') as f:
                json.dump(system, f, indent=4, sort_keys=True)
        else:
            print(json.dumps(system, indent=4, sort_keys=True))

    for system_id in closed_systems:
        system = closed_systems[system_id]
        output_file = os.path.join("..", "..", system["division"], system["submitter"], "systems", "{:}.json".format(system_id))
        print("Generating {:}".format(output_file))
        if not args.dry_run:
            with open(output_file, 'w') as f:
                json.dump(system, f, indent=4, sort_keys=True)
        else:
            print(json.dumps(system, indent=4, sort_keys=True))

    print("Done!")


if __name__ == '__main__':
    main()
