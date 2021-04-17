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
Converts a sample partition .txt file from MLPerf Inference reference to .npy

To generate the .txt file, clone the MLPerf Inference reference repo, and follow the instructions to generate the
partition trace file. You can remove a lot of the imports and comment a lot of their code, or simply extract that
snippet of code if you do not want to run the reference model.
"""

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-i", "--input_file", help="Path to .txt reference file")
    parser.add_argument("-o", "--output_file", help="Path to save .npy file")
    args = parser.parse_args()

    f = open(args.input_file)
    L = []
    last_idx = 0
    for line in f:
        L.append(int(line.split(", ")[0]))
        last_idx = int(line.split(", ")[1])
    L.append(last_idx)
    np.save(args.output_file, np.array(L, dtype=np.int32))


if __name__ == "__main__":
    main()
