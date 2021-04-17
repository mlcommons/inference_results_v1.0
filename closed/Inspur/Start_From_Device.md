Regarding GPUDirect memory access, there are three systems of Inspur involved, NF5488A5, NF5488M6, and NF5468M6. All of them support the GPU Direct capability of NVIDIA GPUs to transfer data direct from PCIe devices directly to GPU device memory. 

# NF5488A5 and NF5488M6 system architecture
Each pair of A100(SXM) GPUs in both NF5488A5 and NF5488M6 systems is connected to a PCIe-Gen4 bridge, which is also connected to a Mellanox GDR NIC with bandwidth of 200 Gb/s. Inspur has measured over 11 GB/s per GPU. The highest bandwidth requirement per GPU for Inspur's submissions on NF5488A5 and NF5488M6 is over 14GB/s for the 3D-Unet. The bandwidth of the others were well below 11 GB/s per GPU. 

# NF5468M6 system architecture
Each group of four A100(PCIE) GPUs in NF5468M6 are connected to a PCIe-Gen4 bridge, which is also connected to a Mellanox GDR NIC with bandwidth of 200 Gb/s. Inspur has measured the bandwidth of 23.75 GB/s per group, and the bandwidth of one GPU was estimated about 6 GB/s. Most bandwidth requirements of Inspur submissions on NF5468M6 were well below 6 GB/s per GPU. 
Inspur submitted Resnet50, SSD-ResNet34, and Bert on NF5468M6 with GPUDirect memory access.
