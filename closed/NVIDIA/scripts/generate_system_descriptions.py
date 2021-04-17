#! /usr/bin/env python3
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
import argparse
import json

import collections

trt_version = "TensorRT 7.2.3"
cuda_version = "CUDA 11.1"
xavier_cuda_version = "CUDA 10.2"
jetpack_version = "21.03 Jetson CUDA-X AI Developer Preview"
cudnn_version = "cuDNN 8.1.1"
xavier_cudnn_version = "cuDNN 8.0.0"
dali_version = "DALI 0.30.0"
triton_version = "Triton 21.02"
os_version = "Ubuntu 18.04.4"
driver_version = "Driver 460.32.03"
a30_driver_version = "Driver 460.46"
submitter = "NVIDIA"


class Status:
    AVAILABLE = "available"
    PREVIEW = "preview"
    RDI = "rdi"


class Division:
    CLOSED = "closed"
    OPEN = "open"


class SystemType:
    EDGE = "edge"
    DATACENTER = "datacenter"
    BOTH = "datacenter,edge"

# List of Machines


Machine = collections.namedtuple("Machine", [
    "status",
    "host_processor_model_name",
    "host_processors_per_node",
    "host_processor_core_count",
    "host_memory_capacity",
    "host_storage_capacity",
    "host_storage_type",
    "accelerator_model_name",
    "accelerator_short_name",
    "mig_short_name",
    "accelerator_memory_capacity",
    "accelerator_memory_configuration",
    "hw_notes",
    "system_id_prefix",
    "system_name_prefix",
])

# The DGX-A100-640G
SJC1_LUNA_02 = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="AMD EPYC 7742",
    host_processors_per_node=2,
    host_processor_core_count=64,
    host_memory_capacity="2 TB",
    host_storage_capacity="15 TB",
    host_storage_type="NVMe SSD",
    accelerator_model_name="NVIDIA A100-SXM-80GB",
    accelerator_short_name="A100-SXM-80GB",
    mig_short_name="",
    accelerator_memory_capacity="80 GB",
    accelerator_memory_configuration="HBM2e",
    hw_notes="",
    system_id_prefix="DGX-A100",
    system_name_prefix="NVIDIA DGX-A100",
)
# The DGX-A100-640G MIG (1GPC)
SJC1_LUNA_02_MIG_1 = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="AMD EPYC 7742",
    host_processors_per_node=2,
    host_processor_core_count=64,
    host_memory_capacity="2 TB",
    host_storage_capacity="15 TB",
    host_storage_type="NVMe SSD",
    accelerator_model_name="NVIDIA A100-SXM-80GB",
    accelerator_short_name="A100-SXM-80GB",
    mig_short_name="1g.10gb",
    accelerator_memory_capacity="80 GB",
    accelerator_memory_configuration="HBM2e",
    hw_notes="",
    system_id_prefix="DGX-A100",
    system_name_prefix="NVIDIA DGX-A100",
)
# The A100-PCIex8 machine
IPP1_1469 = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="AMD EPYC 7742",
    host_processors_per_node=2,
    host_processor_core_count=64,
    host_memory_capacity="1 TB",
    host_storage_capacity="4 TB",
    host_storage_type="NVMe SSD",
    accelerator_model_name="NVIDIA A100-PCIe-40GB",
    accelerator_short_name="A100-PCIe",
    mig_short_name="",
    accelerator_memory_capacity="40 GB",
    accelerator_memory_configuration="HBM2",
    hw_notes="",
    system_id_prefix="",
    system_name_prefix="Gigabyte G482-Z54",
)
# The A30x8 machine
IPP1_1470 = Machine(
    status=Status.PREVIEW,
    host_processor_model_name="AMD EPYC 7742",
    host_processors_per_node=2,
    host_processor_core_count=64,
    host_memory_capacity="1 TB",
    host_storage_capacity="4 TB",
    host_storage_type="NVMe SSD",
    accelerator_model_name="NVIDIA A30",
    accelerator_short_name="A30",
    mig_short_name="",
    accelerator_memory_capacity="24 GB",
    accelerator_memory_configuration="HBM2",
    hw_notes="",
    system_id_prefix="",
    system_name_prefix="Gigabyte G482-Z54",
)
# The A30x8 machine (MIG)
IPP1_1470_MIG_1 = Machine(
    status=Status.PREVIEW,
    host_processor_model_name="AMD EPYC 7742",
    host_processors_per_node=2,
    host_processor_core_count=64,
    host_memory_capacity="1 TB",
    host_storage_capacity="4 TB",
    host_storage_type="NVMe SSD",
    accelerator_model_name="NVIDIA A30",
    accelerator_short_name="A30",
    mig_short_name="1g.3gb",
    accelerator_memory_capacity="24 GB",
    accelerator_memory_configuration="HBM2",
    hw_notes="",
    system_id_prefix="",
    system_name_prefix="Gigabyte G482-Z54",
)
# The A10x8 machine
COMPUTELAB_402 = Machine(
    status=Status.PREVIEW,
    host_processor_model_name="Intel(R) Xeon(R) Platinum 8280 CPU @ 2.70GHz",
    host_processors_per_node=2,
    host_processor_core_count=28,
    host_memory_capacity="768 GB",
    host_storage_capacity="4 TB",
    host_storage_type="NVMe SSD",
    accelerator_model_name="NVIDIA A10",
    accelerator_short_name="A10",
    mig_short_name="",
    accelerator_memory_capacity="16 GB",
    accelerator_memory_configuration="GDDR6",
    hw_notes="",
    system_id_prefix="",
    system_name_prefix="Supermicro 4029GP-TRT-OTO-28",
)
# The DGX Station A100 machine
COMPUTELAB_RO_PROD_01 = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="AMD EPYC 7742",
    host_processors_per_node=1,
    host_processor_core_count=64,
    host_memory_capacity="512 GB",
    host_storage_capacity="10 TB",
    host_storage_type="NVMe SSD",
    accelerator_model_name="NVIDIA A100-SXM-80GB",
    accelerator_short_name="A100-SXM-80GB",
    mig_short_name="",
    accelerator_memory_capacity="80 GB",
    accelerator_memory_configuration="HBM2e",
    hw_notes="",
    system_id_prefix="DGX-Station-A100",
    system_name_prefix="NVIDIA DGX Station A100",
)
# The AGX Xavier machine
COMPUTELAB_310 = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="NVIDIA Carmel (ARMv8.2)",
    host_processors_per_node=1,
    host_processor_core_count=8,
    host_memory_capacity="32 GB",
    host_storage_capacity="32 GB",
    host_storage_type="eMMC 5.1",
    accelerator_model_name="NVIDIA AGX Xavier",
    accelerator_short_name="AGX_Xavier",
    mig_short_name="",
    accelerator_memory_capacity="Shared with host",
    accelerator_memory_configuration="SRAM",
    hw_notes="GPU and both DLAs are used in resnet50, ssd-mobilenet, and ssd-resnet34, in Offline and MultiStream scenarios",
    system_id_prefix="",
    system_name_prefix="NVIDIA Jetson AGX Xavier 32GB",
)
# The Xavier NX machine
COMPUTELAB_501 = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="NVIDIA Carmel (ARMv8.2)",
    host_processors_per_node=1,
    host_processor_core_count=6,
    host_memory_capacity="8 GB",
    host_storage_capacity="32 GB",
    host_storage_type="Micro SD Card",
    accelerator_model_name="NVIDIA Xavier NX",
    accelerator_short_name="Xavier_NX",
    mig_short_name="",
    accelerator_memory_capacity="Shared with host",
    accelerator_memory_configuration="SRAM",
    hw_notes="GPU and both DLAs are used in resnet50, ssd-mobilenet, and ssd-resnet34, in Offline and MultiStream scenarios",
    system_id_prefix="",
    system_name_prefix="NVIDIA Jetson Xavier NX",
)

# List of Systems


class System():
    def __init__(self, machine, division, system_type, gpu_count=1, mig_count=0, is_triton=False, is_xavier=False, is_maxq=False, additional_config=""):
        self.attr = {
            "system_id": self._get_system_id(machine, division, gpu_count, mig_count, is_triton, is_xavier, is_maxq, additional_config),
            "submitter": submitter,
            "division": division,
            "system_type": system_type,
            "status": machine.status if division == Division.CLOSED else Status.RDI,
            "system_name": self._get_system_name(machine, division, gpu_count, mig_count, is_triton, is_xavier, is_maxq, additional_config),
            "number_of_nodes": 1,
            "host_processor_model_name": machine.host_processor_model_name,
            "host_processors_per_node": machine.host_processors_per_node,
            "host_processor_core_count": machine.host_processor_core_count,
            "host_processor_frequency": "",
            "host_processor_caches": "",
            "host_processor_interconnect": "",
            "host_memory_configuration": "",
            "host_memory_capacity": machine.host_memory_capacity,
            "host_storage_capacity": machine.host_storage_capacity,
            "host_storage_type": machine.host_storage_type,
            "host_networking": "",
            "host_networking_topology": "",
            "accelerators_per_node": gpu_count,
            "accelerator_model_name": self._get_accelerator_model_name(machine, division, gpu_count, mig_count, is_triton, is_xavier, is_maxq, additional_config),
            "accelerator_frequency": "",
            "accelerator_host_interconnect": "",
            "accelerator_interconnect": "",
            "accelerator_interconnect_topology": "",
            "accelerator_memory_capacity": machine.accelerator_memory_capacity,
            "accelerator_memory_configuration": machine.accelerator_memory_configuration,
            "accelerator_on-chip_memories": "",
            "cooling": "",
            "hw_notes": machine.hw_notes,
            "framework": self._get_framework(machine, division, gpu_count, mig_count, is_triton, is_xavier, is_maxq, additional_config),
            "operating_system": os_version,
            "other_software_stack": self._get_software_stack(machine, division, gpu_count, mig_count, is_triton, is_xavier, is_maxq, additional_config),
            "sw_notes": "",
            "power_management": "",
            "filesystem": "",
            "boot_firmware_version": "",
            "management_firmware_version": "",
            "other_hardware": "",
            "number_of_type_nics_installed": "",
            "nics_enabled_firmware": "",
            "nics_enabled_os": "",
            "nics_enabled_connected": "",
            "network_speed_mbit": "",
            "power_supply_quantity_and_rating_watts": "",
            "power_supply_details": "",
            "disk_drives": "",
            "disk_controllers": "",
        }

    def _get_system_id(self, machine, division, gpu_count, mig_count, is_triton, is_xavier, is_maxq, additional_config):
        return "".join([
            (machine.system_id_prefix + "_") if machine.system_id_prefix != "" else "",
            machine.accelerator_short_name,
            ("x" + str(gpu_count)) if not is_xavier and mig_count == 0 else "",
            "-MIG_{:}x{:}".format(mig_count * gpu_count, machine.mig_short_name) if mig_count > 0 else "",
            "_TRT" if division == Division.CLOSED else "",
            "_Triton" if is_triton else "",
            "_MaxQ" if is_maxq else "",
            "_{:}".format(additional_config) if additional_config != "" else "",
        ])

    def _get_system_name(self, machine, division, gpu_count, mig_count, is_triton, is_xavier, is_maxq, additional_config):
        system_details = []
        if not is_xavier:
            system_details.append("{:d}x {:}{:}".format(
                gpu_count,
                machine.accelerator_short_name,
                "-MIG-{:}x{:}".format(mig_count, machine.mig_short_name) if mig_count > 0 else ""
            ))
        if is_maxq:
            system_details.append("MaxQ")
        if division == Division.CLOSED:
            system_details.append("TensorRT")
        if is_triton:
            system_details.append("Triton")
        if additional_config != "":
            system_details.append(additional_config)
        return "{:} ({:})".format(
            machine.system_name_prefix,
            ", ".join(system_details)
        )

    def _get_accelerator_model_name(self, machine, division, gpu_count, mig_count, is_triton, is_xavier, is_maxq, additional_config):
        return "{:}{:}".format(
            machine.accelerator_model_name,
            " ({:d}x{:} MIG)".format(mig_count, machine.mig_short_name) if mig_count > 0 else "",
        )

    def _get_framework(self, machine, division, gpu_count, mig_count, is_triton, is_xavier, is_maxq, additional_config):
        frameworks = []
        if is_xavier:
            frameworks.append(jetpack_version)
        if division == Division.CLOSED:
            frameworks.append(trt_version)
        if is_xavier:
            frameworks.append(xavier_cuda_version)
        else:
            frameworks.append(cuda_version)
        return ", ".join(frameworks)

    def _get_software_stack(self, machine, division, gpu_count, mig_count, is_triton, is_xavier, is_maxq, additional_config):
        frameworks = []
        if is_xavier:
            frameworks.append(jetpack_version)
        if division == Division.CLOSED:
            frameworks.append(trt_version)
        if is_xavier:
            frameworks.append(xavier_cuda_version)
        else:
            frameworks.append(cuda_version)
        if division == Division.CLOSED:
            if is_xavier:
                frameworks.append(xavier_cudnn_version)
            else:
                frameworks.append(cudnn_version)
        if not is_xavier:
            frameworks.append(driver_version if "A30" not in machine.accelerator_short_name else a30_driver_version)
        if division == Division.CLOSED:
            frameworks.append(dali_version)
        if is_triton:
            frameworks.append(triton_version)
        return ", ".join(frameworks)

    def __getitem__(self, key):
        return self.attr[key]


submission_systems = [
    # Datacenter submissions
    System(IPP1_1469, Division.CLOSED, SystemType.DATACENTER, 8, 0, False, False),  # A100-PCIex8
    System(IPP1_1469, Division.CLOSED, SystemType.DATACENTER, 8, 0, False, False, True),  # A100-PCIex8-MaxQ
    System(COMPUTELAB_402, Division.CLOSED, SystemType.DATACENTER, 8, 0, False, False),  # A10x8
    System(COMPUTELAB_402, Division.CLOSED, SystemType.DATACENTER, 8, 0, True, False),  # A10x8-Triton
    System(IPP1_1470, Division.CLOSED, SystemType.DATACENTER, 8, 0, False, False),  # A30x8
    System(IPP1_1470, Division.CLOSED, SystemType.DATACENTER, 8, 0, True, False),  # A30x8-Triton
    System(SJC1_LUNA_02, Division.CLOSED, SystemType.DATACENTER, 8, 0, False, False),  # A100-SXM-80GBx8
    System(SJC1_LUNA_02, Division.CLOSED, SystemType.DATACENTER, 8, 0, True, False),  # A100-SXM-80GBx8-Triton
    System(SJC1_LUNA_02, Division.CLOSED, SystemType.DATACENTER, 8, 0, False, False, True),  # A100-SXM-80GBx8-MaxQ
    System(SJC1_LUNA_02_MIG_1, Division.CLOSED, SystemType.DATACENTER, 8, 7, True, False),  # A100-SXM-80GB-MIG-56x1g.10gb-Triton
    System(COMPUTELAB_RO_PROD_01, Division.CLOSED, SystemType.DATACENTER, 4, 0, False, False, False),  # A100-SXM-80GBx4
    System(COMPUTELAB_RO_PROD_01, Division.CLOSED, SystemType.DATACENTER, 4, 0, False, False, True),  # A100-SXM-80GBx4-MaxQ
    # Datacenter and Edge submissions
    System(IPP1_1469, Division.CLOSED, SystemType.EDGE, 1, 0, False, False),  # A100-PCIex1
    System(IPP1_1469, Division.CLOSED, SystemType.EDGE, 1, 0, True, False),  # A100-PCIex1-Triton
    System(SJC1_LUNA_02, Division.CLOSED, SystemType.BOTH, 1, 0, False, False),  # A100-SXM-80GBx1
    System(SJC1_LUNA_02, Division.CLOSED, SystemType.BOTH, 1, 0, True, False),  # A100-SXM-80GBx1-Triton
    # Edge submissions
    System(COMPUTELAB_402, Division.CLOSED, SystemType.EDGE, 1, 0, False, False),  # A10x1
    System(COMPUTELAB_402, Division.CLOSED, SystemType.EDGE, 1, 0, True, False),  # A10x1-Triton
    System(IPP1_1470, Division.CLOSED, SystemType.EDGE, 1, 0, False, False),  # A30x1
    System(IPP1_1470, Division.CLOSED, SystemType.EDGE, 1, 0, True, False),  # A30x1-Triton
    System(COMPUTELAB_310, Division.CLOSED, SystemType.EDGE, 1, 0, False, True),  # AGX Xavier
    System(COMPUTELAB_310, Division.CLOSED, SystemType.EDGE, 1, 0, False, True, True),  # AGX Xavier MaxQ
    System(SJC1_LUNA_02_MIG_1, Division.CLOSED, SystemType.EDGE, 1, 1, False, False),  # A100-SXM-80GB-MIG-1x1g.10gb
    System(SJC1_LUNA_02_MIG_1, Division.CLOSED, SystemType.EDGE, 1, 1, True, False),  # A100-SXM-80GB-MIG-1x1g.10gb-Triton
    System(SJC1_LUNA_02_MIG_1, Division.CLOSED, SystemType.DATACENTER, 1, 1, False, False, False, "HeteroMultiUse"),  # A100-SXM-80GB-MIG-1x1g.10gb-HeteroMultiUse
    System(IPP1_1470_MIG_1, Division.CLOSED, SystemType.EDGE, 1, 1, False, False),  # A30-MIG-1x1g.3gb
    System(IPP1_1470_MIG_1, Division.CLOSED, SystemType.EDGE, 1, 1, True, False),  # A30-MIG-1x1g.3gb-Triton
    System(COMPUTELAB_501, Division.CLOSED, SystemType.EDGE, 1, 0, False, True),  # Xavier NX
    System(COMPUTELAB_501, Division.CLOSED, SystemType.EDGE, 1, 0, False, True, True),  # Xavier NX MaxQ
    # Open submissions
    System(COMPUTELAB_402, Division.OPEN, SystemType.DATACENTER, 1, 0, False, False),  # A10x1
    System(COMPUTELAB_402, Division.OPEN, SystemType.DATACENTER, 8, 0, False, False),  # A10x8
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv", "-o",
        help="Specifies the output tab-separated file for system descriptions.",
        default="systems/system_descriptions.tsv"
    )
    parser.add_argument(
        "--dry_run",
        help="Don't actually copy files, just log the actions taken.",
        action="store_true"
    )
    return parser.parse_args()


def main():
    args = get_args()

    tsv_file = args.tsv

    summary = []
    for system in submission_systems:
        json_file = os.path.join("..", "..", system["division"], system["submitter"], "systems", "{:}.json".format(system["system_id"]))
        print("Generating {:}".format(json_file))
        summary.append("\t".join([str(i) for i in [
            system["system_name"],
            system["system_id"],
            system["submitter"],
            system["division"],
            system["system_type"],
            system["status"],
            system["number_of_nodes"],
            system["host_processor_model_name"],
            system["host_processors_per_node"],
            system["host_processor_core_count"],
            system["host_processor_frequency"],
            system["host_processor_caches"],
            system["host_processor_interconnect"],
            system["host_memory_configuration"],
            system["host_memory_capacity"],
            system["host_storage_capacity"],
            system["host_storage_type"],
            system["host_networking"],
            system["host_networking_topology"],
            system["accelerators_per_node"],
            system["accelerator_model_name"],
            system["accelerator_frequency"],
            system["accelerator_host_interconnect"],
            system["accelerator_interconnect"],
            system["accelerator_interconnect_topology"],
            system["accelerator_memory_capacity"],
            system["accelerator_memory_configuration"],
            system["accelerator_on-chip_memories"],
            system["cooling"],
            system["hw_notes"],
            system["framework"],
            system["operating_system"],
            system["other_software_stack"],
            system["sw_notes"],
            system["power_management"],
            system["filesystem"],
            system["boot_firmware_version"],
            system["management_firmware_version"],
            system["other_hardware"],
            system["number_of_type_nics_installed"],
            system["nics_enabled_firmware"],
            system["nics_enabled_os"],
            system["nics_enabled_connected"],
            system["network_speed_mbit"],
            system["power_supply_quantity_and_rating_watts"],
            system["power_supply_details"],
            system["disk_drives"],
            system["disk_controllers"],
        ]]))
        del system.attr["system_id"]
        if not args.dry_run:
            with open(json_file, "w") as f:
                json.dump(system.attr, f, indent=4, sort_keys=True)
        else:
            print(json.dumps(system.attr, indent=4, sort_keys=True))

    print("Generating system description summary to {:}".format(tsv_file))
    if not args.dry_run:
        with open(tsv_file, "w") as f:
            for item in summary:
                print(item, file=f)
    else:
        print("\n".join(summary))

    print("Done!")


if __name__ == '__main__':
    main()
