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

import copy
from code.common import logging


def is_same_type(a, b):
    number_types = (int, float, complex)
    return a == b or (a in number_types and b in number_types)


def update_nested(old, new, enforce_type_equivalence=False):
    """
    update_nested applies dict.update, applying fields in new to old. If a field in old is a dictionary as well as new,
    then this method is applied recursively.

    args:
        old - The dict to be updated
        new - A dict containing the new values to be updated into old. If new is not of type dict(), it is assumed to be
              a fill values, and all values in old will be replaced by `new`, applied recursively on nested
              dictionaries
        enforce_type_equivalence - Whether or not matching keys in old and new should be of the same type. If True,
                                   method will throw an error if a mismatch is detected.
    """
    if new is None:
        return

    # If new is not a dict, then assume that is a fill value to replace all the values in old
    if type(new) is not dict:
        for k in old:
            if type(old[k]) is dict:
                update_nested(old[k], new, enforce_type_equivalence=enforce_type_equivalence)
            else:
                old[k] = new
    else:
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


def traverse_config(config_name, full, target, seen, enforce_type_equivalence=True):
    if target in seen:
        raise RuntimeError("Error in config '{}': cyclical dependency on {}".format(config_name, target))

    target_conf = full.get(target, None)
    if target_conf is None:
        logging.warn("Could not find configuration for {} in {}".format(target, config_name))
        return None

    # Do not overwrite existing
    target_conf = copy.deepcopy(target_conf)

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
        parent = traverse_config(config_name, full, platform, seen + [target],
                                 enforce_type_equivalence=enforce_type_equivalence)
        update_nested(conf, parent, enforce_type_equivalence=enforce_type_equivalence)

    for platform in scales:
        parent = traverse_config(config_name, full, platform, seen + [target],
                                 enforce_type_equivalence=enforce_type_equivalence)
        for key in scales[platform]:
            if key not in parent:
                raise RuntimeError("{}:{} scales {}:{} which does not exist".format(
                    config_name, target, platform, key))
            parent[key] *= scales[platform][key]
            if "config_ver" in parent:
                for config_ver in parent["config_ver"]:
                    if key in parent["config_ver"][config_ver]:
                        parent["config_ver"][config_ver][key] *= scales[platform][key]
        update_nested(conf, parent, enforce_type_equivalence=enforce_type_equivalence)

    update_nested(conf, target_conf, enforce_type_equivalence=enforce_type_equivalence)
    return conf


def get_system_benchmark_config(config, system_id, enforce_type_equivalence=True):
    config_name = "{}/{}/config.json".format(config["benchmark"], config["scenario"])

    # Get by value (deepcopy) so that we don't modify the original dict
    benchmark_conf = copy.deepcopy(config.get("default", dict()))
    if "default" not in config:
        logging.warn("{} does not have a 'default' setting.".format(config_name))

    system_conf = traverse_config(config_name, config, system_id, [], enforce_type_equivalence=enforce_type_equivalence)
    if system_conf is None:
        return None
    benchmark_conf.update(system_conf)

    # Passthrough for top level values
    benchmark_conf["system_id"] = system_id
    benchmark_conf["scenario"] = config["scenario"]
    benchmark_conf["benchmark"] = config["benchmark"]

    return benchmark_conf


class GraphNode:
    """
    Node in a directed graph. Contains a set of parent and child keys
    """

    def __init__(self):
        self.parents = set()
        self.children = set()

    def add_parent(self, key):
        self.parents.add(key)

    def add_child(self, key):
        self.children.add(key)


class DependencyGraph:
    """
    Directed dependency graph
    """

    def __init__(self, create_from=None, with_config_vers=False):
        self.graph = dict()

        if type(create_from) is str:
            # Assume that create_from is a filepath to a config.json file
            self.from_config_json(create_from, with_config_vers=with_config_vers)
        elif type(create_from) is dict:
            # Assume that create_from is a well-formed config
            self.from_config(create_from, with_config_vers=with_config_vers)
        else:
            raise RuntimeError("create_from has unrecognized type '{}', expected 'str' or 'dict'".format(type(create_from)))

    def create_edge(self, A, B):
        """
        Creates a directional edge A -> B
        """
        self.graph[A].add_child(B)
        self.graph[B].add_parent(A)

    def create_vertex(self, A):
        """
        Creates a disconnected vertex in graph with name A. If A already exists in graph, this method will error
        """
        if self.contains(A):
            raise RuntimeError("Vertex {} already exists in graph".format(A))
        self.graph[A] = GraphNode()

    def contains(self, A):
        """
        Returns a boolean representing if a vertex with name A exists in the graph
        """
        return A in self.graph

    def from_config(self, config, with_config_vers=False):
        """
        Builds a directed graph of config inheritances.
        """
        config_ids = [k for k in config if type(config[k]) is dict]

        # Insert all vertices
        for config_id in config_ids:
            self.create_vertex(config_id)

            # config_vers should also be vertices
            if with_config_vers and "config_ver" in config[config_id]:
                config_vers = list(config[config_id]["config_ver"].keys())
                for config_ver in config_vers:
                    key = "{}.{}".format(config_id, config_ver)
                    self.create_vertex(key)

                    # config_ver is also inheriting from the config it is a part of, and as such is its child
                    self.create_edge(config_id, key)

        # Create directional edges
        for config_id in config_ids:
            # "default" is special case, and is a parent of all nodes, and has no parent itself
            if config_id == "default":
                continue

            # Add default as a parent in all cases
            self.create_edge("default", config_id)

            # Apply parent edges
            extends = config[config_id].get("extends", list())
            scales = config[config_id].get("scales", dict())
            parent_keys = extends + list(scales.keys())
            for pk in parent_keys:
                self.create_edge(pk, config_id)

                # Due to config_ver inheritance, if a parent contains the same config_ver, it will be inherited.
                if with_config_vers and "config_ver" in config[config_id] and "config_ver" in config[pk]:
                    curr_config_vers = set(config[config_id]["config_ver"].keys())
                    parent_config_vers = set(config[config_id]["config_ver"].keys())
                    common_config_vers = curr_config_vers.intersection(parent_config_vers)
                    for config_ver in common_config_vers:
                        curr_config_ver_key = "{}.{}".format(config_id, config_ver)
                        parent_config_ver_key = "{}.{}".format(pk, config_ver)

                        # Parent might not actually have config_ver. Skip if does not exist
                        if not self.contains(parent_config_ver_key):
                            continue

                        self.create_edge(parent_config_ver_key, curr_config_ver_key)

    def from_config_json(self, json_path, with_config_vers=False):
        with open(json_path) as f:
            tmp_conf = json.load(f)
        self.from_config(tmp_conf, with_config_vers=with_config_vers)

    def BFS_from(self, A):
        """
        Returns a list of names of all visited vertices when performing a directed BFS originating at A
        """
        children = set()
        Q = [A]
        while len(Q) > 0:
            curr = Q.pop(0)
            if curr in children:
                continue

            children.add(curr)

            # Add children
            Q += list(self.graph[curr].children)
        return list(children)
