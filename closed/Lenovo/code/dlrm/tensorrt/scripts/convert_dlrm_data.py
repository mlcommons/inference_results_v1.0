# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import numpy as np
import argparse
import struct


def process_dlrm_data(embedding_rows_bound, data_file, dest_dir, num_samples=0):
    """Convert embedding data to separate npy files for ground truth, numerical and categorical data."""

    dest_dir = dest_dir + "/build/criteo"
    os.makedirs(dest_dir, exist_ok=True)

    # No of lines in the file
    if num_samples == 0:
        # 40 int_32 values per line
        n_lines = os.path.getsize(data_file) // 40 // 4
    else:
        n_lines = num_samples

    # Write val_map.txt file
    with open(os.path.join(dest_dir, "val_map.txt"), "w") as f:
        for i in range(n_lines):
            print("{:08d}".format(i), file=f)

    ground_truth_list = []
    int_features_list = []
    int_features_int8_list = []
    cat_features_list = []
    with open(str(data_file), "rb") as f:
        for n in range(n_lines):

            # Print status
            if n % 1000 == 0:
                print("Processing No.{:d}/{:d}...".format(n, n_lines))

            # Save one line into list
            nums = struct.unpack_from("40i", f.read(40 * 4))
            ground_truth_list.append(nums[0])
            int_features = nums[1:14]

            # In reference implementation, we do log(max(0, feature) + 1).
            # TODO: should this be in timed path?
            int_features = [np.log(max(0.0, i) + 1.0) for i in int_features]
            int_features_list.append(int_features)

            # Using [-14.2313, 14.2313] as the range for the numerical input according to calibration cache.
            int8_factor = 127.0 / 14.2313
            int_features_int8 = [min(max(i * int8_factor, -128.0), 127.0) for i in int_features]
            int_features_int8_list.append(int_features_int8)
            cat_features = np.array(nums[14:40], dtype=np.int32)
            cat_features = [x % embedding_rows_bound for x in cat_features]
            cat_features_list.append(cat_features)

    np.save(os.path.join(dest_dir, "ground_truth.npy"), np.array(ground_truth_list, dtype=np.int32))
    np.save(os.path.join(dest_dir, "numeric_fp32.npy"), np.array(int_features_list, dtype=np.float32))
    np.save(os.path.join(dest_dir, "numeric_fp16.npy"), np.array(int_features_list, dtype=np.float16))
    np.save(os.path.join(dest_dir, "numeric_int8_linear.npy"), np.array(int_features_int8_list, dtype=np.int8))
    np.save(os.path.join(dest_dir, "numeric_int8_chw4.npy"), np.array([i + [0 for j in range(16 - 13)] for i in int_features_int8_list], dtype=np.int8))
    np.save(os.path.join(dest_dir, "numeric_int8_chw32.npy"), np.array([i + [0 for j in range(32 - 13)] for i in int_features_int8_list], dtype=np.int8))
    np.save(os.path.join(dest_dir, "categorical_int32.npy"), np.array(cat_features_list, dtype=np.int32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_rows_bound", "-b", help="Specifies the upper bound on the number of embedding rows", default=40000000)
    parser.add_argument("--data_file", "-d", help="Specifies the input data file test_data.bin")
    parser.add_argument("--output_dir", "-o", help="Specifies the output directory for the npy files")
    parser.add_argument("--num_samples", "-n", help="Specifies the number of samples to be processed. Default: all", type=int, default=0)

    args = parser.parse_args()
    data_file = args.data_file
    output_dir = args.output_dir
    embedding_rows_bound = args.embedding_rows_bound
    num_samples = args.num_samples

    process_dlrm_data(embedding_rows_bound, data_file, output_dir, num_samples)


if __name__ == "__main__":
    main()
