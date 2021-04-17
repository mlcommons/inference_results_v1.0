# DGX-A100 System Architecture

NVIDIA's [DGX-A100](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/dgx-a100/dgxa100-system-architecture-white-paper.pdf) system supports the [GPUDirect](https://developer.nvidia.com/gpudirect) capability of NVIDIA GPUs to transfer data directly from PCIe devices (e.g. network, storage or camera) to GPU device memory. Each pair of A100 GPUs in the system is connected to a PCIe-Gen4 bridge, which is also connected to two Mellanox CX6 NICs.

The highest bandwidth requirement for any of Nvidia's submissions is 3DUnet-Offline which requires 15GB/s. NVIDIA has measured over 21 GB/s per GPU with our distributed storage partners.

