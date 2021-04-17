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


class SortingCriteria:
    Higher = True
    Lower = False


class Tree:
    def __init__(self, starting_val=None):
        if starting_val is None:
            self.tree = dict()
        else:
            self.tree = starting_val

    def insert(self, keyspace, value, append=False):
        # pop(0) is O(k), but pop(-1) is O(1). Reverse keyspace
        keyspace = keyspace[::-1]

        curr = self.tree
        while len(keyspace) > 0:
            if len(keyspace) == 1:
                if append:
                    if keyspace[-1] not in curr:
                        curr[keyspace[-1]] = [value]
                    else:
                        if type(curr[keyspace[-1]]) is list:
                            curr[keyspace[-1]].append(value)
                        else:
                            curr[keyspace[-1]] = [curr[keyspace[-1]], value]
                else:
                    curr[keyspace[-1]] = value
                keyspace.pop(-1)
            else:
                if keyspace[-1] not in curr:
                    curr[keyspace[-1]] = dict()

                curr = curr[keyspace[-1]]
                keyspace.pop(-1)

    def get(self, keyspace, default=None):
        # pop(0) is O(k), but pop(-1) is O(1). Reverse keyspace
        keyspace = keyspace[::-1]

        curr = self.tree
        while len(keyspace) > 0:
            if keyspace[-1] not in curr:
                return default

            if len(keyspace) == 1:
                return curr[keyspace[-1]]
            else:
                curr = curr[keyspace[-1]]
                keyspace.pop(-1)

    def __getitem__(self, keyspace_str):
        return self.get(keyspace_str.split(","))

    def __setitem__(self, keyspace_str, value):
        self.insert(keyspace_str.split(","), value)

    def __iter__(self):
        return (k for k in self.tree)


def get_system_type(system_name):
    fname = "systems/{:}.json".format(system_name)
    if not os.path.exists(fname):
        raise Exception("Could not locate system.json for {:}".format(system_name))

    with open(fname) as f:
        data = json.load(f)

    if "system_type" not in data:
        raise Exception("{:} does not have 'system_type' key".format(fname))

    return data["system_type"]
