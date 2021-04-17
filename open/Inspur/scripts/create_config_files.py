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
sys.path.insert(0, os.getcwd())

import itertools

from code.common.system_list import KnownSystems
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

    systems = KnownSystems.get_all_systems()
    for system in systems:
        print("Generating measurements/ entries for", system)

        config_ver_keywords = ["maxq", "high_accuracy"]
        if "Xavier" not in system.get_id():
            config_ver_keywords += ["triton"]
        cross_products = itertools.product([False, True], repeat=len(config_ver_keywords))
        all_config_vers = []
        for config_product in cross_products:
            if not any(config_product):
                all_config_vers.append("default")
            else:
                all_config_vers.append("_".join(itertools.compress(config_ver_keywords, config_product)))

        if "MIG_1x1g" in system.get_id():
            all_config_vers.extend(["hetero", "hetero_high_accuracy"])

        default_args["config_ver"] = ",".join(all_config_vers)
        main(default_args, system)
