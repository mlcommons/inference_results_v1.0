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

import json
import platform
import subprocess
import sys

from glob import glob

VERSION = "v0.7"

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s")

from code.common.system_list import KnownSystems, MIGConfiguration


def is_xavier():
    return platform.processor() == "aarch64"


def check_mig_enabled(gpuid):
    """Check if MIG is enabled on input GPU."""

    p = subprocess.Popen("nvidia-smi mig -lgi -i {gpu}".format(gpu=gpuid), universal_newlines=True, shell=True, stdout=subprocess.PIPE)
    for line in p.stdout:
        if "No MIG-enabled devices found" in line:
            return False
    return True


def get_system():
    """Return a System object that describes computer system.
    """

    # Check if system is Xavier
    if is_xavier():
        # Jetson Xavier is the only supported aarch64 platform.
        with open("/sys/firmware/devicetree/base/model") as product_f:
            product_name = product_f.read()
        if "jetson" in product_name.lower():
            if "AGX" in product_name:
                return KnownSystems.AGX_Xavier.get_match("Jetson-AGX", 1)
            elif "NX" in product_name:
                return KnownSystems.Xavier_NX.get_match("Xavier NX", 1)
            else:
                raise RuntimeError("Unrecognized aarch64 device. Only AGX Xavier and Xavier NX are supported.")

    # Check if MIG is enabled on GPU 0
    mig_conf = None
    if check_mig_enabled(0):
        mig_conf = MIGConfiguration.from_nvidia_smi()
        if mig_conf.num_mig_slices() == 0:
            logging.warn("MIG is enabled, but no instances were detected.")
        else:
            logging.info("Found {:} MIG compute instances".format(mig_conf.num_mig_slices()))

    # TODO: Investigate using py-nvml to get this information, instead of nvidia-smi. It may break on aarch64.
    # Get GPU name and count from nvidia-smi
    nvidia_smi_out = run_command("CUDA_VISIBILE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=gpu_name,pci.device_id --format=csv", get_output=True, tee=False)

    # Remove first line (CSV column names) and strip empty lines
    tmp = [line for line in nvidia_smi_out[1:] if len(line) > 0]

    # If CUDA_VISIBLE_DEVICES is set, apply it manually, as nvidia-smi doesn't obey it. Indexing is correct, as we set
    # CUDA_VISIBLE_ORDER to PCI_BUS_ID. Note that this does not support specifying UUIDs with CUDA_VISIBLE_DEVICES.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        indices = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES").split(",")]
        tmp = [tmp[i] for i in indices]

    count_actual = len(tmp)
    if count_actual == 0:
        raise RuntimeError("nvidia-smi did not detect any GPUs:\n{:}".format(nvidia_smi_out))

    name, pci_id = tmp[0].split(", ")
    assert(pci_id[-4:] == "10DE")  # 10DE is NVIDIA PCI vendor ID
    pci_id = pci_id.split("x")[1][:4]  # Get the relevant 4 digit hex

    system = None
    for sysclass in KnownSystems.get_all_system_classes():
        system = sysclass.get_match(name, count_actual, pci_id=pci_id, mig_conf=mig_conf)
        if system:
            break

    if system is None:
        raise RuntimeError("Cannot find valid configs for {:d}x {:}. Please follow performance_tuning_guide.md to add support for a new GPU.".format(count_actual, name))
    return system


class BENCHMARKS:
    """Supported benchmark names and their aliases."""

    #: Official names for benchmarks
    BERT = "bert"
    DLRM = "dlrm"
    RNNT = "rnnt"
    ResNet50 = "resnet50"
    SSDMobileNet = "ssd-mobilenet"
    SSDResNet34 = "ssd-resnet34"
    UNET = "3d-unet"

    ALL = [
        ResNet50,
        SSDResNet34,
        SSDMobileNet,
        BERT,
        DLRM,
        UNET,
        RNNT
    ]

    HIGH_ACC_ENABLED = {
        BERT,
        DLRM,
        UNET
    }

    #: Map common alias to benchmark name
    alias_map = {
        "3D-Unet": UNET,
        "3DUnet": UNET,
        "3d-unet": UNET,
        "BERT": BERT,
        "DLRM": DLRM,
        "RNN-T": RNNT,
        "RNNT": RNNT,
        "ResNet": ResNet50,
        "ResNet50": ResNet50,
        "Resnet": ResNet50,
        "Resnet50": ResNet50,
        "SSD-MobileNet": SSDMobileNet,
        "SSD-ResNet34": SSDResNet34,
        "SSDMobileNet": SSDMobileNet,
        "SSDResNet34": SSDResNet34,
        "UNET": UNET,
        "Unet": UNET,
        "bert": BERT,
        "dlrm": DLRM,
        "resnet": ResNet50,
        "resnet50": ResNet50,
        "rnn-t": RNNT,
        "rnnt": RNNT,
        "ssd-large": SSDResNet34,
        "ssd-mobilenet": SSDMobileNet,
        "ssd-resnet34": SSDResNet34,
        "ssd-small": SSDMobileNet,
        "unet": UNET,
    }

    def alias(name):
        """Return benchmark name of alias."""
        if name not in BENCHMARKS.alias_map:
            raise ValueError("Unknown benchmark: {:}".format(name))
        return BENCHMARKS.alias_map[name]


class SCENARIOS:
    """Supported scenario names and their aliases."""

    #: Official names for scenarios
    MultiStream = "MultiStream"
    Offline = "Offline"
    Server = "Server"
    SingleStream = "SingleStream"
    ALL = [
        SingleStream,
        MultiStream,
        Offline,
        Server
    ]

    #: Map common alias to scenario name
    alias_map = {
        "Multi-Stream": MultiStream,
        "MultiStream": MultiStream,
        "Multistream": MultiStream,
        "Offline": Offline,
        "Server": Server,
        "Single-Stream": SingleStream,
        "SingleStream": SingleStream,
        "Singlestream": SingleStream,
        "multi-stream": MultiStream,
        "multi_stream": MultiStream,
        "multistream": MultiStream,
        "offline": Offline,
        "server": Server,
        "single-stream": SingleStream,
        "single_stream": SingleStream,
        "singlestream": SingleStream,
    }

    def alias(name):
        """Return scenario name of alias."""
        if not name in SCENARIOS.alias_map:
            raise ValueError("Unknown scenario: {:}".format(name))
        return SCENARIOS.alias_map[name]


def run_command(cmd, get_output=False, tee=True, custom_env=None):
    """
    Runs a command.

    Args:
        cmd (str): The command to run.
        get_output (bool): If true, run_command will return the stdout output. Default: False.
        tee (bool): If true, captures output (if get_output is true) as well as prints output to stdout. Otherwise, does
            not print to stdout.
    """
    logging.info("Running command: {:}".format(cmd))
    if not get_output:
        return subprocess.check_call(cmd, shell=True)
    else:
        output = []
        if custom_env is not None:
            logging.info("Overriding Environment")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, env=custom_env)
        else:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        for line in iter(p.stdout.readline, b""):
            line = line.decode("utf-8")
            if tee:
                sys.stdout.write(line)
                sys.stdout.flush()
            output.append(line.rstrip("\n"))
        ret = p.wait()
        if ret == 0:
            return output
        else:
            raise subprocess.CalledProcessError(ret, cmd)


def args_to_string(d, blacklist=[], delimit=True, double_delimit=False):
    flags = []
    for flag in d:
        # Skip unset
        if d[flag] is None:
            continue
        # Skip blacklisted
        if flag in blacklist:
            continue
        if type(d[flag]) is bool:
            if d[flag] is True:
                flags.append("--{:}=true".format(flag))
            elif d[flag] is False:
                flags.append("--{:}=false".format(flag))
        elif type(d[flag]) in [int, float] or not delimit:
            flags.append("--{:}={:}".format(flag, d[flag]))
        else:
            if double_delimit:
                flags.append("--{:}=\\\"{:}\\\"".format(flag, d[flag]))
            else:
                flags.append("--{:}=\"{:}\"".format(flag, d[flag]))
    return " ".join(flags)


def flags_bool_to_int(d):
    for flag in d:
        if type(d[flag]) is bool:
            if d[flag]:
                d[flag] = 1
            else:
                d[flag] = 0
    return d


def dict_get(d, key, default=None):
    """Return non-None value for key from dict. Use default if necessary."""

    val = d.get(key, default)
    return default if val is None else val


def find_config_files(benchmarks, scenarios):
    """For input benchmarks and scenarios, return CSV string of config file paths."""
    config_file_candidates = ["configs/{:}/{:}/config.json".format(benchmark, scenario)
                              for scenario in scenarios
                              for benchmark in benchmarks
                              ]

    # Only return existing files

    config_file_candidates = [i for i in config_file_candidates if os.path.exists(i)]
    return ",".join(config_file_candidates)


def load_configs(config_files):
    """Return list of configs parsed from input config JSON file paths."""
    configs = []
    for config in config_files.split(","):
        file_locs = glob(config)
        if len(file_locs) == 0:
            raise ValueError("Config file {:} cannot be found.".format(config))
        for file_loc in file_locs:
            with open(file_loc) as f:
                logging.info("Parsing config file {:} ...".format(file_loc))
                configs.append(json.load(f))
    return configs
