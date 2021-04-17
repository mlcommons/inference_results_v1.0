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

from enum import Enum, unique
from collections import OrderedDict
import re
import subprocess
import os


@unique
class Architecture(Enum):
    Turing = "Turing"
    Xavier = "Xavier"
    Ampere = "Ampere"

    Unknown = "Unknown"


class MIGSlice:

    def __init__(self, num_gpcs, mem_gb, device_id=None, uuid=None):
        """
        Describes a MIG instance. If optional arguments are set, then this MIGSlice describes an active MIG instance. If
        optional arguments are not set, then this MIGSlice describes an uninstantiated, but supported MIG instance.

        Arguments:
            num_gpcs: Number of GPCs in this MIG slice
            mem_gb: Allocated video memory capacity in this MIG slice in GB

        Optional arguments:
            device_id: Device ID of the GPU this MIG is a part of
            uuid: UUID of this MIG instance in the format MIG-<GPU UUID>/<gpu instance id>/<compute instance id>
        """
        self.num_gpcs = num_gpcs
        self.mem_gb = mem_gb
        self.device_id = device_id
        self.uuid = uuid

        # One cannot be set without the other.
        assert (device_id is None) == (uuid is None)

    def __str__(self):
        return "{:d}g.{:d}gb".format(self.num_gpcs, self.mem_gb)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def is_active_slice(self):
        return self.device_id is None

    def get_gpu_uuid(self):
        return self.uuid.split("/")[0][4:]  # First 4 characters are "MIG-"

    def get_gpu_instance_id(self):
        return int(self.uuid.split("/")[1])

    def get_compute_instance_id(self):
        return int(self.uuid.split("/")[2])


class MIGConfiguration:

    gpu_regex = re.compile(r"GPU (\d+): ([\w\- ]+) \(UUID: (GPU-[0-9a-f\-]+)\)")
    mig_regex = re.compile(r"  MIG (\d+)g.(\d+)gb Device (\d+): \(UUID: (MIG-GPU-[0-9a-f\-]+/\d+/\d+)\)")

    def __init__(self, conf):
        """
        Stores information about a system's MIG configuration.

        conf: An OrderedDict of gpu_id -> { MIGSlice -> Count }
        """
        self.conf = conf

    def check_compatible(self, valid_mig_slices):
        """
        Given a list of valid MIGSlices, checks if this MIGConfiguration only contains MIGSlices that are described in
        the list.
        """
        m = {str(mig) for mig in valid_mig_slices}
        for gpu_id in self.conf:
            for mig in self.conf[gpu_id]:
                if str(mig) not in m:
                    return False
        return True

    def num_gpus(self):
        """
        Returns the number of GPUs with active MIG instances in this MIGConfiguration.
        """
        return len(self.conf)

    def num_mig_slices(self):
        """
        Returns the number of total active MIG instances across all GPUs in this MIGConfiguration
        """
        i = 0
        for gpu_id in self.conf:
            for mig in self.conf[gpu_id]:
                i += self.conf[gpu_id][mig]
        return i

    def __str__(self):
        """
        Returns a string that describes this MIG configuration.

        Examples:
          - For 1x 1-GPC: 1x1g.10gb
          - For 1x 1-GPC, 2x 2-GPC, and 3x 3-GPC: 1x1g.10gb_2x2g.20gb_1x3g.30gb
        """
        # Add up the slices on each GPU by MIGSlice
        flattened = OrderedDict()
        for gpu_id in self.conf:
            for mig in self.conf[gpu_id]:
                if mig not in flattened:
                    flattened[mig] = 0
                flattened[mig] += self.conf[gpu_id][mig]
        return "_".join(sorted(["{}x{}".format(flattened[mig], str(mig))]))

    @staticmethod
    def from_nvidia_smi():
        p = subprocess.Popen("nvidia-smi -L", universal_newlines=True, shell=True, stdout=subprocess.PIPE)
        conf = OrderedDict()
        gpu_id = None
        for line in p.stdout:

            gpu_match = MIGConfiguration.gpu_regex.match(line)
            if gpu_match is not None:
                gpu_id = int(gpu_match.group(1))
                gpu_name = gpu_match.group(2)
                gpu_uuid = gpu_match.group(3)
                conf[gpu_id] = {}

            visible_gpu_instances = set()
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                for g in os.environ.get("CUDA_VISIBLE_DEVICES").split(","):
                    if g.startswith('MIG'):
                        visible_gpu_instances.add(g)

            mig_match = MIGConfiguration.mig_regex.match(line)
            if mig_match is not None:
                num_gpcs = int(mig_match.group(1))
                mem_gb = int(mig_match.group(2))
                mig_gpu_id = int(mig_match.group(3))
                mig_uuid = mig_match.group(4)

                if mig_uuid.split("/")[0] != "MIG-" + gpu_uuid:
                    raise RuntimeError("MIG instance has UUID mismatch with GPU. Got {}, expected MIG-{}".format(mig_uuid, gpu_uuid))

                if (not os.environ.get("CUDA_VISIBLE_DEVICES")) or (mig_uuid in visible_gpu_instances):
                    mig_slice = MIGSlice(num_gpcs, mem_gb, device_id=mig_gpu_id, uuid=mig_uuid)
                    if mig_slice not in conf[gpu_id]:
                        conf[gpu_id][mig_slice] = 0
                    conf[gpu_id][mig_slice] += 1

        return MIGConfiguration(conf)


class System:
    """
    System class contains information on the GPUs used in our submission systems.

    gpu: ID of the GPU being used
    pci_id: PCI ID of the GPU
    arch: Architecture of the GPU
    count: Number of GPUs used on the system
    mig_conf: MIG configuration (if applicable)
    """

    def __init__(self, gpu, pci_id, arch, count, mig_conf=None):
        self.gpu = gpu
        self.pci_id = pci_id
        self.arch = arch
        self.count = count
        self.mig_conf = mig_conf
        self.uses_mig = mig_conf is not None

    def get_id(self):
        sid = "{:}x{:}".format(self.gpu, self.count) if "Xavier" not in self.gpu else self.gpu
        if self.mig_conf is not None:
            sid = "{:}-MIG_{:}".format(self.gpu, str(self.mig_conf))
        return sid

    def __str__(self):
        return self.get_id()


class SystemClass:
    def __init__(self, gpu, aliases, pci_ids, arch, supported_counts, valid_mig_slices=None):
        """
        SystemClass describes classes of submissions systems with different variations. SystemClass objects are
        hardcoded as supported systems and must be defined in KnownSystems below to be recognized a valid system for the
        pipeline.

        Args:
            gpu: ID of the GPU being used, usually the name reported by nvidia-smi
            aliases: Different names of cards reported by nvidia-smi that use the same SKUs, i.e. Titan RTX and Quadro
                     RTX 8000
            pci_ids: PCI IDs of cards that match this system configuration that use the same SKUs
            arch: Architecture of the GPU
            supported_counts: Counts of GPUs for supported multi-GPU systems, i.e. [1, 2, 4] to support 1x, 2x, and 4x
                              GPU systems
            valid_mig_slices: List of supported MIGSlices. None if MIG not supported.
        """
        self.gpu = gpu
        self.aliases = aliases
        self.pci_ids = pci_ids
        self.arch = arch
        self.supported_counts = supported_counts
        self.valid_mig_slices = valid_mig_slices
        self.supports_mig = valid_mig_slices is not None

    def __str__(self):
        return "SystemClass(gpu={}, aliases={}, pci_ids={}, arch={}, counts={})".format(
            self.gpu,
            self.aliases,
            self.pci_ids,
            self.arch,
            self.supported_counts)

    def get_match(self, name, count, pci_id=None, mig_conf=None):
        """
        Attempts to match a certain GPU configuration with this SystemClass. If the configuration does not match,
        returns None. Otherwise, returns a System object with metadata about the configuration.

        mig_conf should be a MIGConfiguration object.
        """
        # PCI ID has precedence over name, as pre-release chips often are not named yet in nvidia-smi
        gpu_match = False
        if pci_id is not None and len(self.pci_ids) > 0:
            gpu_match = pci_id in self.pci_ids
        # Attempt to match a name if PCI ID is not specified, or if the system has no known PCI IDs
        # This is an else block, but we explicitly show condition for clarity
        elif pci_id is None or len(self.pci_ids) == 0:
            gpu_match = name in self.aliases

        if not gpu_match:
            return None

        # If GPU matches, match the count and mig configs (if applicable)
        if count not in self.supported_counts:
            return None

        if self.supports_mig and mig_conf is not None and not mig_conf.check_compatible(self.valid_mig_slices):
            return None

        return System(self.gpu, pci_id, self.arch, count, mig_conf=mig_conf)


class KnownSystems:
    """
    Global List of supported systems
    """

    #A100_PCIe_40GB = SystemClass("A100-PCIe", ["A100-PCIE-40GB"], ["20F1", "20BF"], Architecture.Ampere, [1, 2, 8])
    #A100_SXM4_40GB = SystemClass("A100-SXM4-40GB", ["A100-SXM4-40GB"], ["20B0"], Architecture.Ampere, [1, 8],
    #                             valid_mig_slices=[MIGSlice(1, 5), MIGSlice(2, 10), MIGSlice(3, 20)])
    #A100_SXM_80GB = SystemClass("A100-SXM-80GB", ["A100-SXM-80GB"], ["20B2"], Architecture.Ampere, [1, 4, 8],
    #                            valid_mig_slices=[MIGSlice(1, 10), MIGSlice(2, 20), MIGSlice(3, 40)])
    #GeForceRTX_3080 = SystemClass("GeForceRTX3080", ["GeForce RTX 3080"], ["2206"], Architecture.Ampere, [1])
    #GeForceRTX_3090 = SystemClass("GeForceRTX3090", ["GeForce RTX 3090", "Quadro RTX A6000", "RTX A6000"],
    #                              ["2204", "2230"], Architecture.Ampere, [1])
    #A10 = SystemClass("A10", ["A10"], ["2236"], Architecture.Ampere, [1, 8])
    A40 = SystemClass("A40", ["A40"], ["2235"], Architecture.Ampere, [4])
    T4 = SystemClass("T4", ["Tesla T4", "T4 32GB"], ["1EB8", "1EB9"], Architecture.Turing, [1, 8, 20])
    A100_PCIe_40GB = SystemClass("A100-PCIe", ["A100-PCIE-40GB"], ["20F1"], Architecture.Ampere, [4])
    #T4 = SystemClass("T4", ["Tesla T4", "T4 32GB"], ["1EB8", "1EB9"], Architecture.Turing, [1, 8, 20])
    #TitanRTX = SystemClass("TitanRTX", ["TITAN RTX", "Quadro RTX 8000", "Quadro RTX 6000"], ["1E02", "1E30", "1E36"],
    #                       Architecture.Turing, [1, 4])
    #AGX_Xavier = SystemClass("AGX_Xavier", ["Jetson-AGX"], [], Architecture.Xavier, [1])
    #Xavier_NX = SystemClass("Xavier_NX", ["Xavier NX"], [], Architecture.Xavier, [1])
    #A30 = SystemClass("A30", ["A30"], ["20B7"], Architecture.Ampere, [1, 8],
    #                  valid_mig_slices=[MIGSlice(1, 3), MIGSlice(1, 6), MIGSlice(2, 6), MIGSlice(2, 12)])
    
    # CPU Systems
    #Triton_CPU_2S_6258R = System("Triton_CPU_2S_6258R", "2S_6258R", "", 1, None)
    #Triton_CPU_4S_8380H = System("Triton_CPU_4S_8380H", "4S_8380H", "", 1, None)

    @staticmethod
    def get_all_system_classes():
        return [
            getattr(KnownSystems, attr)
            for attr in dir(KnownSystems)
            if type(getattr(KnownSystems, attr)) == SystemClass
        ]

    @staticmethod
    def get_all_systems():
        all_classes = KnownSystems.get_all_system_classes()
        all_systems = []
        for system_class in all_classes:
            for count in system_class.supported_counts:
                all_systems.append(System(system_class.gpu, "", system_class.arch, count, None))
                if count == 1 and system_class.valid_mig_slices is not None:
                    for mig_slice in system_class.valid_mig_slices:
                        conf = {"DummyGPU": {mig_slice: 1}}
                        mig_conf = MIGConfiguration(conf)
                        all_systems.append(System(system_class.gpu, "", system_class.arch, count, mig_conf))
        return all_systems
