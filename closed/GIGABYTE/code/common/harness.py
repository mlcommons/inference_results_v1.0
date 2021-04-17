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

import re
import os
import sys
sys.path.insert(0, os.getcwd())
from code.common import logging, dict_get, run_command, args_to_string
from code.common import BENCHMARKS, SCENARIOS
from code.common.submission import TENSORRT_VERSION, generate_measurements_entry
from code.common.result_parser import from_loadgen_by_keys, scenario_loadgen_log_keys
import code.common.arguments as common_args

plugin_map = {
    BENCHMARKS.DLRM: ["build/plugins/DLRMInteractionsPlugin/libdlrminteractionsplugin.so",
                      "build/plugins/DLRMBottomMLPPlugin/libdlrmbottommlpplugin.so"],
    BENCHMARKS.SSDMobileNet: ["build/plugins/NMSOptPlugin/libnmsoptplugin.so"],
    BENCHMARKS.SSDResNet34: ["build/plugins/NMSOptPlugin/libnmsoptplugin.so"],
    BENCHMARKS.UNET: ["build/plugins/instanceNormalization3DPlugin/libinstancenorm3dplugin.so",
                      "build/plugins/pixelShuffle3DPlugin/libpixelshuffle3dplugin.so",
                      "build/plugins/conv3D1X1X1K4Plugin/libconv3D1X1X1K4Plugin.so"],
}

benchmark_qsl_size_map = {
    # See: https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc#benchmarks-1
    BENCHMARKS.BERT: 10833,
    BENCHMARKS.DLRM: 204800,
    BENCHMARKS.RNNT: 2513,
    BENCHMARKS.ResNet50: 2048,
    BENCHMARKS.SSDMobileNet: 1024,
    BENCHMARKS.SSDResNet34: 64,
    BENCHMARKS.UNET: 16,
}

system_name_map = [
    ("A100-SXM-80GBx4", "DGX-Station-A100"),
    ("A100-SXM", "DGX-A100"),
]


class BaseBenchmarkHarness:
    """Base class for benchmark harnesses."""

    def __init__(self, args, name="", skip_file_checks=False):
        self.args = args
        self.name = name
        self.verbose = dict_get(args, "verbose", default=None)
        if self.verbose:
            logging.info("===== Harness arguments for {:} =====".format(name))
            for key in args:
                logging.info("{:}={:}".format(key, args[key]))

        self.system_id = args["system_id"]
        self.scenario = args["scenario"]
        self.config_ver = args["config_ver"]
        self.engine_dir = "./build/engines/{:}/{:}/{:}".format(self.system_id, self.name, self.scenario)
        self.precision = args["precision"]

        # Detect devices used to set field prefixes
        self.has_gpu = dict_get(args, "gpu_batch_size", default=None) is not None
        self.has_dla = dict_get(args, "dla_batch_size", default=None) is not None
        self.qps_prefix = ""
        if self.has_gpu and self.has_dla:
            self.qps_prefix = "concurrent_"
        elif self.has_gpu:
            self.qps_prefix = "gpu_"
        elif self.has_dla:
            self.qps_prefix = "dla_"

        # Check if we actually need to execute the harness
        self.generate_conf_files_only = False
        if dict_get(self.args, "generate_conf_files_only", False):
            logging.info("Only generating measurements/ configuration entries")
            self.generate_conf_files_only = True
            self.args["generate_conf_files_only"] = None

        # Enumerate engine files
        # Engine not needed if we are only generating measurements/ entries
        self.skip_file_checks = skip_file_checks or self.generate_conf_files_only
        self.gpu_engine = None
        self.dla_engine = None
        self.enumerate_engines()

        # Enumerate harness executable
        self.executable = self._get_harness_executable()
        self.check_file_exists(self.executable)

        self.use_jemalloc = False

        self.env_vars = os.environ.copy()
        self.flag_builder_custom_args = []

    def _get_harness_executable(self):
        raise NotImplementedError("BaseBenchmarkHarness cannot be called directly")

    def _build_custom_flags(self, flag_dict):
        """
        Handles any custom flags to insert into flag_dict. Can return either a flag_dict, or a converted arg string.
        """
        return flag_dict

    def _handle_harness_result(self, result):
        """
        Called on the harness result before it is returned to main.py. Can be used to post-process the result.
        """
        return result

    def _get_engine_fpath(self, device_type, batch_size):
        return "{:}/{:}-{:}-{:}-b{:}-{:}.{:}.plan".format(self.engine_dir, self.name, self.scenario,
                                                          device_type, batch_size, self.precision, self.config_ver)

    def _append_config_ver_name(self, system_name):
        if "maxq" in self.config_ver.lower():
            system_name += "_MaxQ"
        if "hetero" in self.config_ver.lower():
            system_name += "_HeteroMultiUse"
        return system_name

    def get_system_name(self):
        override_system_name = dict_get(self.args, "system_name", default=None)
        if override_system_name not in {None, ""}:
            return override_system_name

        system_name = self.system_id
        for kw, prepend_name in system_name_map:
            if kw in self.system_id:
                system_name = "_".join([prepend_name, system_name])
                break
        full_system_name = "{:}_TRT{:}".format(system_name, TENSORRT_VERSION)
        return self._append_config_ver_name(full_system_name)

    def _get_submission_benchmark_name(self):
        full_benchmark_name = self.name
        if dict_get(self.args, "accuracy_level", "99%") == "99.9%":
            full_benchmark_name += "-99.9"
        elif self.name in BENCHMARKS.HIGH_ACC_ENABLED:
            full_benchmark_name += "-99"
        return full_benchmark_name

    def get_full_log_dir(self):
        return os.path.join(self.args["log_dir"], self.get_system_name(), self._get_submission_benchmark_name(), self.scenario)

    def enumerate_engines(self):
        if self.has_gpu:
            self.gpu_engine = self._get_engine_fpath("gpu", self.args["gpu_batch_size"])
            self.check_file_exists(self.gpu_engine)

        if self.has_dla:
            self.dla_engine = self._get_engine_fpath("dla", self.args["dla_batch_size"])
            self.check_file_exists(self.dla_engine)

    def check_file_exists(self, f):
        """Check if file exists. Complain if configured to do so."""

        if not os.path.isfile(f):
            if self.skip_file_checks:
                print("Note: File {} does not exist. Attempting to continue regardless, as hard file checks are disabled.".format(f))
            else:
                raise RuntimeError("File {:} does not exist.".format(f))

    def build_default_flags(self):
        flag_dict = {}
        flag_dict["verbose"] = self.verbose

        # Handle plugins
        if self.name in plugin_map:
            plugins = plugin_map[self.name]
            for plugin in plugins:
                self.check_file_exists(plugin)
            flag_dict["plugins"] = ",".join(plugins)

        # Generate flags for logfile names.
        log_dir = self.get_full_log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        flag_dict["logfile_outdir"] = log_dir
        flag_dict["logfile_prefix"] = "mlperf_log_"

        # Handle performance sample count
        perf_sample_count = dict_get(self.args, "performance_sample_count", None)
        if perf_sample_count is not None:
            flag_dict["performance_sample_count"] = perf_sample_count
        elif benchmark_qsl_size_map[self.name] > 0:
            flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self.name]
        else:
            flag_dict["performance_sample_count"] = self.args["gpu_batch_size"]

        # Handle custom arguments
        for arg in self.flag_builder_custom_args:
            val = dict_get(self.args, arg, None)
            if val is not None:
                flag_dict[arg] = val

        return flag_dict

    def build_scenario_specific_flags(self):
        """Return flags specific to current scenario."""

        flag_dict = {}

        prefix = self.qps_prefix

        if self.scenario == SCENARIOS.SingleStream:
            scenario_keys = common_args.SINGLE_STREAM_PARAMS
        elif self.scenario == SCENARIOS.Offline:
            scenario_keys = common_args.OFFLINE_PARAMS
        elif self.scenario == SCENARIOS.MultiStream:
            scenario_keys = common_args.MULTI_STREAM_PARAMS
        elif self.scenario == SCENARIOS.Server:
            scenario_keys = common_args.SERVER_PARAMS
        else:
            scenario_keys = []
            raise RuntimeError("Unknown Scenario \"{}\"".format(self.scenario))

        for arg in scenario_keys:
            val = dict_get(self.args, prefix + arg, None)
            if val is None:
                raise ValueError("Missing required key {:}".format(prefix + arg))
            flag_dict[arg] = val

        # Handle RUN_ARGS
        for arg in scenario_keys:
            val = dict_get(self.args, arg, None)
            if val is not None:
                flag_dict[arg] = val

        return flag_dict

    def prepend_ld_preload(self, so_path):
        if "LD_PRELOAD" in self.env_vars:
            self.env_vars["LD_PRELOAD"] = ":".join([so_path, self.env_vars["LD_PRELOAD"]])
        else:
            self.env_vars["LD_PRELOAD"] = so_path

        logging.info("Updated LD_PRELOAD: " + self.env_vars["LD_PRELOAD"])

    def run_harness(self):
        flag_dict = self.build_default_flags()
        flag_dict.update(self.build_scenario_specific_flags())

        # Handle engines
        if self.has_gpu:
            flag_dict["gpu_engines"] = self.gpu_engine

        # MLPINF-853: Special handing of --fast. Use min_duration=60000, and if Multistream, use min_query_count=1.
        if flag_dict.get("fast", False):
            if "min_duration" not in flag_dict:
                flag_dict["min_duration"] = 60000
            if self.scenario in [SCENARIOS.Offline, SCENARIOS.MultiStream]:
                if "min_query_count" not in flag_dict:
                    flag_dict["min_query_count"] = 1
            flag_dict["fast"] = None

        # Generates the entries in the `measurements/` directory, and updates flag_dict accordingly
        generate_measurements_entry(
            self.get_system_name(),
            self.name,
            self._get_submission_benchmark_name(),
            self.scenario,
            self.args["input_dtype"],
            self.args["precision"],
            flag_dict)

        # Stop here if we are only generating .conf files in measurements
        if self.generate_conf_files_only:
            return "Generated conf files"

        argstr = self._build_custom_flags(flag_dict)
        if type(argstr) is dict:
            argstr = args_to_string(flag_dict)

        # Handle environment variables
        if self.use_jemalloc:
            self.prepend_ld_preload("/usr/lib/x86_64-linux-gnu/libjemalloc.so.2")

        cmd = "{:} {:}".format(self.executable, argstr)
        output = run_command(cmd, get_output=True, custom_env=self.env_vars)

        # Return harness result.
        scenario_key = scenario_loadgen_log_keys[self.scenario]
        results = from_loadgen_by_keys(
            os.path.join(
                self.args["log_dir"],
                self.get_system_name(),
                self._get_submission_benchmark_name(),
                self.scenario),
            ["result_validity", scenario_key])

        if scenario_key not in results:
            result_string = "Cannot find performance result. Maybe you are running in AccuracyOnly mode."
        elif "result_validity" not in results:
            result_string = "{}: {}, Result validity unknown".format(scenario_key, results[scenario_key])
        else:
            result_string = "{}: {}, Result is {}".format(scenario_key, results[scenario_key], results["result_validity"])
        return self._handle_harness_result(result_string)
