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

__doc__ = """
Given a path to a config.json, this script will auto-format the JSON file and save it to the original path specified.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_fpath",
        help="Path to config.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config_fpath) as f:
        d = json.load(f)

    with open(args.config_fpath, 'w') as f:
        json.dump(d, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
