# MLPerf Inference v1.0 Implementations
This is a repository of DellEMC servers using optimized implementations for [MLPerf Inference Benchmark v1.0](https://github.com/mlcommons/inference/).

# CPU Implementations
## Benchmarks
**Please refer to Intel's readme under /closed/Intel for detailed instructions, including performace guides, and instructions on how to run with new systems.**
  
The following benchmarks are part of our submission for MLPerf Inference v1.0:
- [resnet50](/closed/Intel/code/resnet/openvino/README.md)
- [ssd-resnet34](/closed/Intel/code/ssd-resnet34/openvino/README.md)

## Dell EMC Submission Systems

The closed systems that Dell EMC has submitted on using CPUs are:
- Datacenter systems
  - Dell EMC PowerEdge R750
    - Intel(R) Xeon(R) Gold 6330 CPU @ 2.0GHz

# GPU Implementations
## Benchmarks
**Please refer to /closed/NVIDIA for detailed instructions, including performace guides, and instructions on how to run with new systems.** 
  
The following benchmarks are part of our submission for MLPerf Inference v1.0:
- [3d-unet](/closed/NVIDIA/code/3d-unet/tensorrt/README.md)
- [bert](/closed/NVIDIA/code/bert/tensorrt/README.md)
- [dlrm](/closed/NVIDIA/code/dlrm/tensorrt/README.md)
- [rnnt](/closed/NVIDIA/code/rnnt/tensorrt/README.md)
- [resnet50](/closed/NVIDIA/code/resnet50/tensorrt/README.md)
- [ssd-resnet34](/closed/NVIDIA/code/ssd-resnet34/tensorrt/README.md)

## Dell EMC Submission Systems

The closed systems that Dell EMC has submitted on using NVIDIA GPUs are:
- Datacenter systems
  - Dell EMC DSS 8440
    - A100-PCIe-40GB
    - A40
  - Dell EMC PowerEdge R740
    - A100-PCIe-40GB 
  - Dell EMC PowerEdge R750xa
    - A100-PCIe-40GB
  - Dell EMC PowerEdge R7525
    - A100-PCIe-40GB
    - Quadro RTX 8000
  - Dell EMC PowerEdge XE2420
    - Tesla T4
  - Dell EMC PowerEdge XE8545
    - A100-SXM-40GB / 400W
    - A100-SXM-80GB / 500W
- Edge systems
  - Dell EMC PowerEdge XE2420
    - Tesla T4
