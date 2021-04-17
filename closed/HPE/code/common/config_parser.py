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

import json

import os
import sys
sys.path.insert(0, os.getcwd())

from code.common import logging


def is_same_type(a, b):
    number_types = (int, float, complex)
    return a == b or (a in number_types and b in number_types)


def update_nested(old, new, enforce_type_equivalence=False):
    if new is not None:
        for k in new:
            if k not in old:
                old[k] = new[k]
            else:
                if enforce_type_equivalence and not is_same_type(type(old[k]), type(new[k])):
                    raise RuntimeError("Type mismatch on key={}".format(k))

                if type(old[k]) is dict and type(new[k]) is dict:
                    update_nested(old[k], new[k], enforce_type_equivalence=enforce_type_equivalence)
                else:
                    old[k] = new[k]


def traverse_config(config_name, full, target, seen):
    if target in seen:
        raise RuntimeError("Error in config '{}': cyclical dependency on {}".format(config_name, target))

    target_conf = full.get(target, None)
    if target_conf is None:
        logging.warn("Could not find configuration for {} in {}".format(target, config_name))
        return None

    # The 2 keys that define inheritance dependencies are "extends" and "scales"
    extends = []
    if "extends" in target_conf:
        extends = target_conf["extends"]
        del target_conf["extends"]

    scales = dict()
    if "scales" in target_conf:
        scales = target_conf["scales"]
        del target_conf["scales"]

    # extends and scales cannot share the common elements
    common_keys = set(extends).intersection(set(scales.keys()))
    if len(common_keys) > 0:
        raise RuntimeError("{}:{} cannot both extend and scale {}".format(config_name, target, list(common_keys)[0]))

    conf = dict()

    # Apply extended configs
    for platform in extends:
        parent = traverse_config(config_name, full, platform, seen + [target])
        update_nested(conf, parent, enforce_type_equivalence=True)

    for platform in scales:
        parent = traverse_config(config_name, full, platform, seen + [target])
        for key in scales[platform]:
            if key not in parent:
                raise RuntimeError("{}:{} scales {}:{} which does not exist".format(
                    config_name, target, platform, key))
            parent[key] *= scales[platform][key]
            if "config_ver" in parent:
                for config_ver in parent["config_ver"]:
                    if key in parent["config_ver"][config_ver]:
                        parent["config_ver"][config_ver][key] *= scales[platform][key]
        update_nested(conf, parent, enforce_type_equivalence=True)

    update_nested(conf, target_conf, enforce_type_equivalence=True)
    return conf


def get_system_benchmark_config(config, system_id):
    config_name = "{}/{}/config.json".format(config["benchmark"], config["scenario"])

    benchmark_conf = config.get("default", dict())
    if "default" not in config:
        logging.warn("{} does not have a 'default' setting.".format(config_name))

    system_conf = traverse_config(config_name, config, system_id, [])
    if system_conf is None:
        return None
    benchmark_conf.update(system_conf)

    # Passthrough for top level values
    benchmark_conf["system_id"] = system_id
    benchmark_conf["scenario"] = config["scenario"]
    benchmark_conf["benchmark"] = config["benchmark"]

    return benchmark_conf
