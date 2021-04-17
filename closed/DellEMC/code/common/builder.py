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

import os
import sys
sys.path.insert(0, os.getcwd())

import tensorrt as trt
from code.common import logging, dict_get, SCENARIOS


class AbstractBuilder(object):
    """Interface base class for calibrating and building engines."""

    def __init__(self):
        raise NotImplementedError("AbstractBuilder cannot be called directly")

    def build_engines(self):
        raise NotImplementedError("AbstractBuilder cannot be called directly")

    def calibrate(self):
        raise NotImplementedError("AbstractBuilder cannot be called directly")


class BenchmarkBuilder(AbstractBuilder):
    """
    Base class for calibrating and building engines for a given benchmark.
    Has the steps common to most benchmarks.
    """

    def __init__(self, args, name="", workspace_size=(1 << 30)):
        """
        Constructor
        :param args: arguments represented by a dictionary
        :param name: name of the benchmark
        """

        self.name = name
        self.args = args

        # Configuration variables
        self.verbose = dict_get(args, "verbose", default=False)
        if self.verbose:
            logging.info("========= BenchmarkBuilder Arguments =========")
            for arg in args:
                logging.info("{:}={:}".format(arg, args[arg]))

        self.system_id = args["system_id"]
        self.scenario = args["scenario"]
        self.config_ver = args["config_ver"]
        self.engine_dir = "./build/engines/{:}/{:}/{:}".format(self.system_id, self.name, self.scenario)

        # Set up logger, builder, and network.
        self.logger = trt.Logger(trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.builder = trt.Builder(self.logger)
        self.builder_config = self.builder.create_builder_config()
        self.builder_config.max_workspace_size = workspace_size
        if dict_get(args, "verbose_nvtx", default=False):
            self.builder_config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE

        # Precision variables
        self.input_dtype = dict_get(args, "input_dtype", default="fp32")
        self.input_format = dict_get(args, "input_format", default="linear")
        self.precision = dict_get(args, "precision", default="int8")
        self.clear_flag(trt.BuilderFlag.TF32)
        if self.precision == "fp16":
            self.apply_flag(trt.BuilderFlag.FP16)
        elif self.precision == "int8":
            self.apply_flag(trt.BuilderFlag.INT8)

        # Device variables
        self.device_type = "gpu"
        self.dla_core = args.get("dla_core", None)
        if self.dla_core is not None:
            logging.info("Using DLA: Core {:}".format(self.dla_core))
            self.device_type = "dla"
            self.apply_flag(trt.BuilderFlag.GPU_FALLBACK)
            self.builder_config.default_device_type = trt.DeviceType.DLA
            self.builder_config.DLA_core = int(self.dla_core)

        if self.scenario == SCENARIOS.SingleStream:
            self.batch_size = 1
        elif self.scenario in [SCENARIOS.Server, SCENARIOS.Offline, SCENARIOS.MultiStream]:
            self.batch_size = self.args.get("batch_size", 1)
        else:
            raise ValueError("Invalid scenario: {:}".format(self.scenario))

        # Currently, TRT has limitation that we can only create one execution
        # context for each optimization profile. Therefore, create more profiles
        # so that LWIS can create multiple contexts.
        self.num_profiles = self.args.get("gpu_copy_streams", 4)

        self.initialized = False

    def initialize(self):
        """Builds the network in preparation for building the engine. This method must be implemented by
        the subclass.

        The implementation should also set self.initialized to True.
        """
        raise NotImplementedError("BenchmarkBuilder.initialize() should build the network")

    def apply_flag(self, flag):
        """Apply a TRT builder flag."""
        self.builder_config.flags = (self.builder_config.flags) | (1 << int(flag))

    def clear_flag(self, flag):
        """Clear a TRT builder flag."""
        self.builder_config.flags = (self.builder_config.flags) & ~(1 << int(flag))

    def _get_engine_fpath(self, device_type, batch_size):
        # Use default if not set
        if device_type is None:
            device_type = self.device_type
        if batch_size is None:
            batch_size = self.batch_size

        # If the name ends with .plan, we assume that it is a custom path / filename
        if self.name.endswith(".plan"):
            return "{:}/{:}".format(self.engine_dir, self.name)
        else:
            return "{:}/{:}-{:}-{:}-b{:}-{:}.{:}.plan".format(
                self.engine_dir, self.name, self.scenario,
                device_type, batch_size, self.precision, self.config_ver)

    def build_engines(self):
        """Calls self.initialize() if it has not been called yet. Builds and saves the engine."""

        if not self.initialized:
            self.initialize()

        # Create output directory if it does not exist.
        if not os.path.exists(self.engine_dir):
            os.makedirs(self.engine_dir)

        engine_name = self._get_engine_fpath(self.device_type, self.batch_size)
        logging.info("Building {:}".format(engine_name))

        if self.network.has_implicit_batch_dimension:
            self.builder.max_batch_size = self.batch_size
        else:
            self.profiles = []
            # Create optimization profiles if on GPU
            if self.dla_core is None:
                for i in range(self.num_profiles):
                    profile = self.builder.create_optimization_profile()
                    for input_idx in range(self.network.num_inputs):
                        input_shape = self.network.get_input(input_idx).shape
                        input_name = self.network.get_input(input_idx).name
                        min_shape = trt.Dims(input_shape)
                        min_shape[0] = 1
                        max_shape = trt.Dims(input_shape)
                        max_shape[0] = self.batch_size
                        profile.set_shape(input_name, min_shape, max_shape, max_shape)
                    if not profile:
                        raise RuntimeError("Invalid optimization profile!")
                    self.builder_config.add_optimization_profile(profile)
                    self.profiles.append(profile)
            else:
                # Use fixed batch size if on DLA
                for input_idx in range(self.network.num_inputs):
                    input_shape = self.network.get_input(input_idx).shape
                    input_shape[0] = self.batch_size
                    self.network.get_input(input_idx).shape = input_shape

        # Build engines
        engine = self.builder.build_engine(self.network, self.builder_config)
        buf = engine.serialize()
        with open(engine_name, 'wb') as f:
            f.write(buf)

    def calibrate(self):
        """Generate a new calibration cache."""

        self.need_calibration = True
        self.calibrator.clear_cache()
        self.initialize()
        # Generate a dummy engine to generate a new calibration cache.
        if self.network.has_implicit_batch_dimension:
            self.builder.max_batch_size = 1
        else:
            for input_idx in range(self.network.num_inputs):
                input_shape = self.network.get_input(input_idx).shape
                input_shape[0] = 1
                self.network.get_input(input_idx).shape = input_shape
        engine = self.builder.build_engine(self.network, self.builder_config)


class MultiBuilder(AbstractBuilder):
    """
    MultiBuilder allows for building multiple engines sequentially. As an example, RNN-T has multiple components, each of
    which have separate engines, which we would like to abstract away.
    """

    def __init__(self, builders, args):
        """
        MultiBuilder takes in a list of Builder classes and args to be passed to these Builders.
        """
        self.builders = list(builders)
        self.args = args

    def build_engines(self):
        for b in self.builders:
            b(self.args).build_engines()

    def calibrate(self):
        for b in self.builders:
            b(self.args).calibrate()
