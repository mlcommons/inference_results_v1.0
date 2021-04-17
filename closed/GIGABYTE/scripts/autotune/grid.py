#!/usr/bin/env python3
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


""" GridSearch-based Autotuning

An automated means of executing an MLPerf benchmark+scenario with a variety of parameters/configurations
and parsing the results of execution.
"""
import argparse
import copy
import glob
import hashlib  # For directory naming
import itertools
import importlib  # For python parsing!
import inspect  # For python parsing!
import json
from numbers import Number
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, os.getcwd())
from code.common import get_system, find_config_files, load_configs
from code.common import SCENARIOS, BENCHMARKS
from code.common import run_command  # used in tee
from code.common.config_parser import get_system_benchmark_config
from code.common.result_parser import from_loadgen_by_keys, scenario_loadgen_log_keys
from code.common.system_list import Architecture

# For metadata and stats we record for each run (in addition to what the harness already records)
METAFILE = "autotuneMETAFILE.json"


class ExecStats:
    """Named tuple that helps us understand how much work each autotuning session is doing."""

    def __init__(self):
        self.builds = 0
        self.cached_builds = 0
        self.runs = 0
        self.cached_runs = 0

    def __str__(self):
        return f"builds={self.builds}, cached_builds={self.cached_builds}, runs={self.runs}, cached_runs={self.cached_runs}"


class PerfStats:
    # List the loadgen keys we want
    scenario_keys_set = {v for k, v in scenario_loadgen_log_keys.items()} | {"result_validity"}
    _extra_key_set = {
        "result_min_latency_ns",
        "result_max_latency_ns",
        "result_mean_latency_ns",
        "result_50.00_percentile_latency_ns",
        "result_90.00_percentile_latency_ns",
        "result_95.00_percentile_latency_ns",
        "result_97.00_percentile_latency_ns",
        "result_99.00_percentile_latency_ns",
        "result_99.90_percentile_latency_ns",
    }
    verbose_stat_key_set = scenario_keys_set | _extra_key_set

    def __init__(self, directory, verbose=False):
        """ Populate self.data from contents of directory which contains:
        - METAFILE at top level
        - mlperf_log_summary.txt in a run/platform-specific subdirectory.
        """
        search_path = os.path.join(directory, "**/mlperf_log_detail.txt")
        paths = [name for name in glob.glob(search_path, recursive=True)]
        if not paths:
            raise RuntimeError(f"Could not find mlperf_log_detail.txt in: \n{directory}\nDid you mean to run with --noparse?")
        key_set = self.verbose_stat_key_set if verbose else self.scenario_keys_set
        result = from_loadgen_by_keys(os.path.dirname(paths[0]), key_set)
        assert len(result) > 0
        to_ret = {}
        to_ret.update(result)

        with open(os.path.join(directory, METAFILE), 'r') as f:
            extra_stats = json.load(f)['run_info']
        to_ret.update(extra_stats)

        self.data = to_ret


class ConfigGrid:
    """ An iterable to enumerate through a cartesian product of configurations

    Actual iterator is referred to as a "cross-term", which is a single
    value in the cartesian product.
    The cross-term on its own isn't very useful, but can be fed to
    cross_to* methods for more useful items.
    """

    def __init__(self, bench, scen, config_dict, config_funcs=None):
        """ Construct a ConfigGrid

        Args:
            bench (str): The benchmark requested (fuzzy match behavior using BENCHMARKS.alias)
            scen (str): The scenario requested (fuzzy match behavior using SCENARIOS.alias)
            config_dict (Dict[str, List]): A config dictionary. Refer to 'Config Schema' in the README for format
            config_funcs (Dict[str, Callable]): A dictionary of META* functions. Refer to 'Config Schema' in the README for requirements.

        """
        if args.spoof_system_id:
            self.system_id = args.spoof_system_id
        else:
            self.system = get_system()
            self.system_id = self.system.get_id()
        self.benchmark = BENCHMARKS.alias(bench)
        self.scenario = SCENARIOS.alias(scen)
        candidate_configs = find_config_files(benchmarks=[self.benchmark],
                                              scenarios=[self.scenario])
        configs = load_configs(candidate_configs)
        assert len(configs) == 1
        # To work with "extends" and "scales", we need to call into another config helper:
        self.base_config = get_system_benchmark_config(configs[0], self.system_id)
        self.default_config = configs[0]['default']
        griddict = config_dict
        self.no_rebuild_params = None
        # No-op
        self.is_config_valid = lambda x: True
        # No-op
        self.search_callback = None
        self.replay = None
        funcs_processed = set()
        if config_funcs:
            if config_funcs.get("META_search_callback"):
                funcs_processed.add("META_search_callback")
                self.search_callback = config_funcs['META_search_callback']
                if not args.use_cached:
                    raise RuntimeError(f"META_search_callback must be used with --use_cached for reproducibility.")
            if config_funcs.get("META_get_no_rebuild_params"):
                funcs_processed.add("META_get_no_rebuild_params")
                norebuild_params = config_funcs.get("META_get_no_rebuild_params")()
                assert isinstance(norebuild_params, list)
                # Make sure these keys all exist in our grid params:
                # But we might not know grid params if a search_callback is being used:
                if self.search_callback is None:
                    missing_keys = set(norebuild_params) - set(griddict.keys())
                    if len(missing_keys) > 0:
                        raise RuntimeError(f"The keys: {missing_keys} were mentioned in META_get_no_rebuild_params, but are not a specified parameter in:\n{griddict.keys()}")
                else:
                    print("WARNING: Not checking get_no_rebuild_params against grid parameters, be careful")
                # For use later, we're gonna turn this into a set:
                self.no_rebuild_params = set(norebuild_params)
            if config_funcs.get("META_is_config_valid"):
                funcs_processed.add("META_is_config_valid")
                self.is_config_valid = config_funcs["META_is_config_valid"]

                # Make sure we aren't scanning other params:
            # Other META handling goes here
            unmatched_funcs = set(config_funcs.keys()) - funcs_processed
            if len(unmatched_funcs) > 0:
                raise RuntimeError(f"Found the following META functions which haven't been implemented, refer to README for proper naming {unmatched_funcs}")

        # Make sure we can set all keys are in our config:
        if not args.no_check_keys:
            for grid_key in griddict.keys():
                if grid_key not in self.base_config:
                    print(f"{grid_key} not found in base config")
                    print(f"{self.base_config}")
                    assert False
        # Make sure all values are non-empty lists of something that isn't a list or a dict
        # TODO expand this to something reasonable to help META_bisect
        if "META_search_callback" in funcs_processed:
            print("WARNING: Skipping parameter validation because META_search_callback was provided")
        else:
            for val in griddict.values():
                assert isinstance(val, list)
                #assert len(val) >= 1
                assert all(not isinstance(el, list) and not isinstance(el, dict) for el in val)
        self.grid = griddict
        # For now, we delay constructing our cross terms
        # Lastly, make a subdirectory to store this run.

    def need_to_rebuild(self, old_cross, new_cross):
        if old_cross is None:
            # If we were told that we can use an engine already present, do that
            return not args.use_existing_engine
        if self.no_rebuild_params is None:
            # If we don't know anything else about our problem, we have to rebuild
            return True
        # Sanity check, make sure all keys of one are in the other
        assert set(old_cross.keys()) == set(new_cross.keys())
        # Now we have to see if all of the changed keys are in the "no_rebuild_needed" set:
        changed_keys = {k for k in old_cross.keys() if old_cross[k] != new_cross[k]}
        return len(changed_keys - self.no_rebuild_params) != 0

    def has_been_run(self, cross):
        """Try and get stats from potentially created artifacts, if we can, we've ran it before"""
        log_dir = self.cross_to_log_dir(cross)
        try:
            stats = PerfStats(directory=log_dir)
        except Exception:
            return False
        return True

    def cross_to_standalone_config(self, cross_term):
        """ Acquire a standalone config which is usable by generate_engines/run_harness targets."""
        # Get a unique copy so we don't clobber:
        dupe = copy.deepcopy(self.base_config)
        dupe.update(cross_term)
        # We need to wrap this in a json with the schema:
        # {"$SYSTEM_ID": {conf}, "benchmark": "$benchmark", "scenario": "$scenario"}
        return {
            f"{self.system_id}": dupe,
            "default": self.default_config,
            "benchmark": self.benchmark,
            "scenario": self.scenario
        }

    def cross_to_log_dir(self, cross_term):
        """ Find the log directory for a given cross-term.

        Contains basic collision detection. It's up to the user if old data wants to be:
        - Overwritten -> Move/remove problematic directory.
        - Kept -> Increase cross_hash length (will invalidate runs with --use_cached).
        """
        # Very arbitrary naming scheme, but at least reproducible with an identical input grid (if we crash, etc.)
        base_str = f"build/logs/grid_{self.system_id}_{self.benchmark}_{self.scenario}_"
        json_str = json.dumps(cross_term, sort_keys=True).encode("utf-8")
        # Ten chars should be "good_enough".
        cross_hash = hashlib.md5(json_str).hexdigest()[:10]
        path = f"{base_str}{cross_hash}"
        # Let's do a quick collision check:
        metafile_path = os.path.join(path, METAFILE)
        # Try opening the previous run metadata and reading it out
        # If the config/cross_term stored doesn't match our current cross_term, panic
        if os.path.exists(metafile_path):
            with open(metafile_path) as f:
                meta_dict = json.load(f)
            if meta_dict['config'] != cross_term:
                raise RuntimeError(f"Hash collision detected with cross_term: {cross_term}\nTry increasing digest length!")

        return path

    def _should_run(self, cross_term):
        full_config = self.cross_to_standalone_config(cross_term)[self.system_id]
        try:
            should_run = self.is_config_valid(full_config)
        except Exception as e:
            print("Unknown error: is META_is_config_valid defined correctly")
            raise e
        return should_run

    def __iter__(self):
        """ Iterates through this Config's cross-terms

        Will potentially take advantage of no_rebuild_params and is_config_valid for better scheduling.
        """
        if self.search_callback is None:
            # We can be fancy:
            # If we choose the noRebuildNeeded params to be on the innerDim of the cross product, that will minimize the number of rebuilds we need
            # (Because we only have the last run cached).
            # Note, in the general case (of scheduling arbitrary jobs where the only thing known is if running job B after job A requires a rebuild), this reduces to finding a minimum cost hamiltonian path, which is NP Complete

            if self.no_rebuild_params:
                sorted_keys = sorted(self.grid.keys(), key=lambda x: x in self.no_rebuild_params)
                sorted_vals = (self.grid[k] for k in sorted_keys)
                cross_terms = (i for i in itertools.product(*sorted_vals))
                named_terms = (dict(zip(sorted_keys, term)) for term in cross_terms)
            else:
                cross_terms = (i for i in itertools.product(*self.grid.values()))
                named_terms = (dict(zip(self.grid.keys(), term)) for term in cross_terms)

            for term_dict in named_terms:
                # Is this a valid term? If so we should yield it, if not, skip it
                if self._should_run(term_dict):
                    yield term_dict
                elif args.dry_run:
                    print(f"DRY-RUN: Not running {term_dict} because it failed user-provided constraint function")
        else:
            if self.replay:
                for term, stats in self.replay:
                    yield term
            else:
                if not args.dry_run:
                    past_terms = []
                    term = self.search_callback(past_terms)
                    while term:
                        print(f"Trying cross_term {term}.\nDone so far:{[c[0] for c in past_terms]}")
                        yield term
                        log_dir = self.cross_to_log_dir(term)
                        stats = PerfStats(directory=log_dir, verbose=True).data
                        past_terms.append((term, stats))
                        term = self.search_callback(past_terms)
                    self.replay = past_terms
                else:
                    print(f"DRY-RUN: Would call into user-provided META_search_callback here.")



    def __len__(self):
        """ Returns number of valid cross terms in this ConfigGrid """
        if self.search_callback:
            # We don't know here
            return None
        else:
            acc = 1
            for li in self.grid.values():
                if self.should_run(li):
                    acc *= len(li)
            return acc


def py_or_json_opener(thing):
    if thing.endswith('.json'):
        with open(thing, 'r') as f:
            try:
                param_dict = json.load(f)
            except Exception as e:
                print(e)
                raise e
            meta_funcs = None
    elif thing.endswith('.py'):
        try:
            loader = importlib.machinery.SourceFileLoader('mod', thing)
            mod = loader.load_module()
            attrs = [(k, v) for k, v in vars(mod).items() if not k.startswith('_')]
            param_dict = {k: v for k, v in attrs if isinstance(v, list)}
            meta_funcs = {k: v for k, v in attrs if callable(v) and k.startswith("META")}
        except Exception as e:
            print(e)
            raise e
    else:
        raise RuntimeError("Config doesn't end in .py or .json")
    return param_dict, meta_funcs


def tee(cmd):
    # Unused return, but we need to request output to get tee effect
    run_command(cmd, get_output=True, tee=True)


class Temperature:
    """A wrapper, non-instance class to do one-time initialization of get_system().

    Initialization is done at the first call of logged_temp_wait akin to 'magic statics'
    We need run-time initialization (rather than module-load-time) to support spoof_system_id"""
    system = None

    @classmethod
    def _get_core_temps(cls):
        if cls.system.arch == Architecture.Xavier:
            # Because we don't have nvidia-smi on xavier, we need to use sysfs to read out the temperature
            # The type of the thermal_zone is in /sys/devices/virtual/thermal/termal_zone<N>/type.
            # To avoid doing a bunch of process spawn to check if a given node is a GPU node, we're gonna hardcode the GPU_therm node:
            # AGX_Xavier: thermal_zone1
            # Xavier_NX: thermal_zone1
            # NOTE, this may change in subsequent/previous submission models.
            try:
                out_text = run_command("cat /sys/devices/virtual/thermal/thermal_zone1/temp", get_output=True, tee=False)
                # The temperature is in units of milli degC, so scale the result:
                temps = [int(str_temp) / 1000 for str_temp in out_text]
            except Exception as e:
                print("Bad temp reading")
                raise e
        else:
            # Non-xavier branch
            try:
                out_text = run_command("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", get_output=True, tee=False)
                # multi-gpu instance return a list of strings corresponding to temp of each core
                temps = [int(str_temp) for str_temp in out_text]
            except Exception as e:
                print("Bad temp reading")
                raise e
        return temps

    @classmethod
    def logged_temp_wait(cls, temp, timeout=None):
        """ Spinwait on GPU temperature with optional timeout. Returns last measured temperature

        For multi-GPU systems, we use mean temperature.
        """
        if cls.system is None:
            cls.system = get_system()
        start_time = time.perf_counter()
        # Poll with timeout
        succ = False
        while (time.perf_counter() - start_time) < timeout:
            temps = cls._get_core_temps()
            mean_temp = sum(temps) / len(temps)
            if mean_temp <= temp:
                print("GPU has finished cooling")
                succ = True
                break
            print(mean_temp)
            time.sleep(2)
        if not succ:
            print("GPU failed to fully cool")
        return mean_temp


def finalize_log_dir(directory, cross_term, extra_info):
    data = {
        "config": cross_term,
        "run_info": extra_info
    }
    with open(os.path.join(directory, METAFILE), 'w') as f:
        json.dump(data, f, indent=2)


def execute(lam, description):
    if args.dry_run:
        print(f"DRY-RUN: {description}")
    else:
        return lam()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=str, help="Fuzzy MLPerf-I Benchmark name. eg: Resnet50, rnnt, or rnn-t")
    parser.add_argument("scenario", type=str, help="Fuzzy MLPerf-I Scenario name. eg: Offline, offline, Server")
    parser.add_argument("--no_check_keys", action="store_true", help="If set, will not check that a key defined in the config file is present in the config, adding it to the temp per-run config if needed rather than updating.")
    run_group = parser.add_argument_group("running")
    run_group.add_argument("--temp_timeout", default=60, type=int, help="Max number of seconds to sleep between runs (and building) if cool_temp isn't reached")
    run_group.add_argument("--cool_temp", default=50, type=int, help="Target temperature in degC to wait for before building or running")
    run_group.add_argument("--extra_run_args", default='', help="String to insert inside of run_harness and generate_engine's RUN_ARGS")
    run_group.add_argument("--use_cached", action='store_true', help="Don't rerun configurations which have existing artifacts")
    run_group.add_argument("--dry_run", action='store_true', help="Prints actions instead of executing, used for verifying config. Implicitly includes --noparse")
    run_group.add_argument("--use_existing_engine", action='store_true',
                           help="Don't unconditionally build an engine for the first build run. (Note, if there a subsequent run requires building a new engine, a new engine will still be built.)")
    parse_group = parser.add_argument_group("parsing")
    parse_group.add_argument("--noparse", action='store_true', help="Skip the parsing/results stage")
    parse_group.add_argument("--verbose_stats", action='store_true', help="Parse all stats, not just scenario-specific fields")
    # TODO, add additional parsing output formats. For now, csv of all fields is good enough.
    parser.add_argument("config", type=py_or_json_opener, help=".json or .py file specifying parameters and values. See README for schema")
    parser.add_argument("--spoof_system_id", type=str, metavar="SYSTEM_ID" ,help="Use a given SYSTEM_ID instead of the host's. Can only be used with --dry_run and/or --use_cached");
    args = parser.parse_args()
    # Argument validation:
    if args.spoof_system_id and (not args.dry_run or not args.use_cached):
        parser.error("--spoof_system_id requires --dry_run --use_cached")
    configpy_vars, configpy_funcs = args.config
    config_grid = ConfigGrid(args.benchmark, args.scenario, configpy_vars, configpy_funcs)
    exec_stats = ExecStats()
    past_cross = None
    for cross_term in config_grid:
        if args.use_cached and config_grid.has_been_run(cross_term):

            # No-op, just doing this for uniformity
            execute(lambda: None,
                    f"Skip build and run for cached item: {cross_term}")
            exec_stats.cached_runs += 1
            continue
        with tempfile.NamedTemporaryFile(mode='w') as tmp_config_file:
            # Write out the config
            config_json = config_grid.cross_to_standalone_config(cross_term)
            json.dump(config_json, tmp_config_file)
            tmp_config_file.flush()

            config_name = tmp_config_file.name

            if config_grid.need_to_rebuild(past_cross, cross_term):
                exec_stats.builds += 1
                # If this is our first pass, we don't need to sleep before building an engine
                if past_cross:
                    execute(lambda: Temperature.logged_temp_wait(args.cool_temp, args.temp_timeout),
                            f"Wait for temp to cool to {args.cool_temp} (timeout at {args.temp_timeout}sec)")
                build_str = (f"make generate_engines "
                             f"RUN_ARGS='--benchmarks={args.benchmark} "
                             f"--scenarios={args.scenario} "
                             f"--configs={config_name} "
                             f"{args.extra_run_args}'")
                execute(lambda: tee(build_str),
                        f"Build")
            else:
                exec_stats.cached_builds += 1
            # We always sleep before a run:
            start_temp = execute(lambda: Temperature.logged_temp_wait(args.cool_temp, args.temp_timeout),
                                 f"Wait for temp to cool to {args.cool_temp} (timeout at {args.temp_timeout}sec)")
            log_dir = config_grid.cross_to_log_dir(cross_term)
            run_str = (f"make run_harness "
                       f"LOG_DIR={log_dir} "
                       f"RUN_ARGS='--benchmarks={args.benchmark} "
                       f"--scenarios={args.scenario} "
                       f"--configs={config_name} "
                       f"--test_mode=PerformanceOnly "
                       f"{args.extra_run_args}'")
            execute(lambda: tee(run_str),
                    f"Run with cross term {cross_term}")

            # For now, we're not recording too much extra run info, but for now it's done here.
            extra_info = {
                'start_temp': start_temp
            }
            execute(lambda: finalize_log_dir(directory=log_dir, cross_term=cross_term, extra_info=extra_info),
                    f"Dump additional run info into log_dir")

            exec_stats.runs += 1
            past_cross = cross_term
    execute(lambda: None,
            f"Session statistics: {exec_stats}")
    if not args.noparse and not args.dry_run:
        for idx, cross_term in enumerate(config_grid):
            log_dir = config_grid.cross_to_log_dir(cross_term)
            # Potentially different values in mlperf_log_summary.txt
            try:
                stats = PerfStats(directory=log_dir, verbose=args.verbose_stats).data
            except Exception as e:
                raise Exception(f"Failed to find performance results for cross-term:\n{cross_term}\n")
            # Throw these stats in the dict to do easy printing. (Use | in python3.9)
            cross_term.update(stats)
            sorted_keys = sorted(cross_term.keys())
            if idx == 0:
                # print schema
                schema_str = ",".join([k for k in sorted_keys])
                print(schema_str)
            out_str = ",".join([str(cross_term[k]) for k in sorted_keys])
            print(out_str)
