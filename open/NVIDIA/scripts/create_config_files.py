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

import sys
import os
import re
sys.path.insert(0, os.getcwd())

from code.common.system_list import system_list
from code.common import dict_get
from code.main import main

if __name__ == "__main__":
    default_args = {
        "action": "generate_conf_files",
        "benchmarks": None,  # Defaults to all benchmarks
        "scenarios": None,  # Defaults to all scenarios
        "config_ver": "all",
        "configs": "",
        "no_gpu": False,
        "gpu_only": False
    }

    systems = set([x[0] for x in system_list])
    for system_id in systems:
        print("Generating measurements/ entries for", system_id)
        if "Xavier" in system_id:
            default_args["config_ver"] = "default,high_accuracy"
        else:
            default_args["config_ver"] = "all"
        main(default_args, system_id)
