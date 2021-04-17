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

import numpy as np
import os

in_path = "/home/scratch.mlperf_inference/preprocessed_data/rnnt/fp32/RNNT_input_fp32_"
out_path_fp32 = "/home/scratch.mlperf_inference/preprocessed_data/rnnt_1152/fp32/RNNT_input_1152_"
out_path_fp16 = "/home/scratch.mlperf_inference/preprocessed_data/rnnt_1152/fp16/RNNT_input_1152_"
out_path_int32 = "/home/scratch.mlperf_inference/preprocessed_data/rnnt_1152/int32/RNNT_input_1152_"
num_samples = 2939
val_map = "data_maps/rnnt_1152/val_map.txt"

max_length = 1152

for idx in range(0, num_samples):
    print(idx)
    sample = np.load("{:}{:d}.npy".format(in_path, idx))
    length = np.array([sample.shape[0]]).astype(np.int32)
    np.save("{:}{:d}.npy".format(out_path_int32, idx), length)
    sample_pad = np.zeros((max_length, 240)).astype(np.float32)
    sample_pad[:length[0], :] = sample
    np.save("{:}{:d}.npy".format(out_path_fp32, idx), sample_pad)
    np.save("{:}{:d}.npy".format(out_path_fp16, idx), sample_pad.astype(np.float16))

with open(val_map, "w") as f:
    for idx in range(num_samples):
        print("RNNT_input_1152_{:d}".format(idx), file=f)
