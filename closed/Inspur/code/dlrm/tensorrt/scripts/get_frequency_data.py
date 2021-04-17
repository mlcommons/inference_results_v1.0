#!/usr/bin/env python3
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

import csv
import struct
import os


def get_frequency_data(bin_file, out_dir):
    """Gather frequency data from training dataset and save to CSV file."""

    n_lines = os.path.getsize(bin_file) // 40 // 4  # 40 int_32 values per line
    out_file = os.path.join(out_dir, "table_")

    embedding_sizes = [
        40000000,
        39060,
        17295,
        7424,
        20265,
        3,
        7122,
        1543,
        63,
        40000000,
        3067956,
        405282,
        10,
        2209,
        11938,
        155,
        4,
        976,
        14,
        40000000,
        40000000,
        40000000,
        590152,
        12973,
        108,
        36
    ]

    for table_no in range(26):
        embed_file = open(out_file + str(table_no + 1) + ".csv", mode='w')
        embed_writer = csv.writer(embed_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        freq_row = {}
        for i in range(embedding_sizes[table_no] + 1):
            freq_row[i] = 0
        print("Process file for table " + str(table_no))

        with open(bin_file, "rb") as f:
            for _ in range(n_lines):
                nums = struct.unpack_from("40i", f.read(40 * 4))
                cat_features = nums[14:40]
                row_accessed = cat_features[table_no]
                freq_row[int(row_accessed)] += 1

        print("Sorting frequencies ..\n")
        sorted_freq = sorted(freq_row.items(), key=lambda x: x[1], reverse=True)
        print("Write to file")
        row = 0
        for x, y in sorted_freq:
            embed_writer.writerow([x, y])
            row += 1
        embed_file.close()
