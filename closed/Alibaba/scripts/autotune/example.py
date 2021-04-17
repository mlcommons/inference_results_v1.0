# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


# "Typical" format which doesn't differ much from JSON
gpu_batch_size = [128, 256, 512, 1024, 2048]

# ... But, we can do arbitrary computation to calculate other variables
# Variables with leading underscores are not considered "public", and should be used for private computation
_private_var = 2
audio_batch_size = [x * _private_var for x in gpu_batch_size]

# We can even use external libraries to do wild stuff:
# Note that functions which are not preceeded by META_ are considered "private" and not exposed to grid.py/the scheduler.
import numpy as np


def do_wild_stuff():
    # Generate an array [1, 2]
    complicated_array = np.arange(1, 3, 1)
    return complicated_array.tolist()


dali_pipeline_depth = do_wild_stuff()


# We have some a posteriori knowledge that certain parameters are only meaningful at runtime and not build time
# This meta function let's the scheduler know that to order runs in such a way to minimize the number of rebuilds
def META_get_no_rebuild_params():
    return ["audio_batch_size", "dali_pipeline_depth"]

# It's sometimes easier to just declare some arbitrary list of parameters, and then describe a predicate which filters that list.
# Note well that we never specify "audio_buffer_num_lines" because config is the updated default config.


def META_is_config_valid(config):
    if config['dali_pipeline_depth'] * config['audio_batch_size'] > config['audio_buffer_num_lines']:
        return False
    return True
