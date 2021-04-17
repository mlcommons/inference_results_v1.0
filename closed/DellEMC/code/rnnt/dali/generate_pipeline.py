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

from pipeline import DALIInferencePipeline
import argparse


config = {}  # Handles empty dict/default case
total_samples = 16  # Unused in class, throwing in a number

parser = argparse.ArgumentParser(description='RNN-T DALIPipelineExport')
parser.add_argument("--output_dir", default="./", type=str, help="pth output path")
parser.add_argument("--device", default="gpu", type=str, help="Device type (gpu/cpu)")
parser.add_argument("--batch_size", default=16)
parser.add_argument("--num_threads", default=2)
parser.add_argument("--audio_fp16_input", action='store_true', help='assume that the raw audio is in fp16 format instead of fp32')

args = parser.parse_args()

audio_fp16_input_suffix = "fp16" if args.audio_fp16_input == True else "fp32"
filename = args.output_dir + "/dali_pipeline_" + args.device + "_" + audio_fp16_input_suffix + ".pth"

pipe = DALIInferencePipeline.from_config(device=args.device,
                                         config=config,
                                         device_id=0,
                                         batch_size=args.batch_size,
                                         total_samples=total_samples,
                                         num_threads=args.num_threads,
                                         audio_fp16_input=args.audio_fp16_input)
pipe.serialize(filename=filename)
