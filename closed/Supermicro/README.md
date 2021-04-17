# MLPerf Inference v1.0 Implementations

This is a repository of Supermicro implementations for [MLPerf Inference Benchmark](https://www.mlperf.org/inference-overview/).

## Benchmarks

The following *benchmarks* are part of our submission for MLPerf Inference:

 - **3D-Unet** (3d-unet)
 - **BERT** (bert)
 - **DLRM** (dlrm)
 - **RNN-T** (rnnt)
 - **ResNet50** (resnet50)
 - **SSD-MobileNet** (ssd-mobilenet)
 - **SSD-ResNet34** (ssd-resnet34)

## Scenarios

Each of the above benchmarks can run in one or more of the following four inference *scenarios*:

 - **SingleStream**
 - **MultiStream**
 - **Offline**
 - **Server**

Please refer to the [MLPerf Inference official page](https://www.mlperf.org/inference-overview/) for explanations about the scenarios.

## Supermicro Submissions

Our MLPerf Inference implementation has the following submissions:

| Benchmark     | Datacenter Submissions                                        | Edge Submissions (Multistream may be optional)                                   |
|---------------|---------------------------------------------------------------|----------------------------------------------------------------------------------|
| 3D-UNET       | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline         | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, SingleStream              |
| BERT          | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, Server | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream                        |
| DLRM          | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, Server | None                                                                             |
| RNN-T         | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream                        |
| ResNet50      | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream, MultiStream           |
| SSD-MobileNet | None                                                          | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream, MultiStream           |
| SSD-ResNet34  | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream, MultiStream           |

Benchmarks are stored in the [code/](code) directory.
Every benchmark contains a `README.md` detailing instructions on how to set up that benchmark, including:

 - Downloading the dataset and model
 - Running any necessary preprocessing
 - Details on the optimizations being performed

For details on how to run each benchmark, see below.

## Supermicro Submission Systems

The systems that NVIDIA supports, has tested, and are submitting are:

 - Datacenter systems
   - A100-SXM-80GBx8 (SYS-420GP-TNAR, 80GB variant)
   - A100-SXM-80GBx8 (AS -4124GO-NART, 80GB variant)
   - A100-SXM-40GBx8 (AS -4124GO-NART, 40GB variant)
   - A100-SXM-80GBx4 (AS -2124GQ-NART, 80GB variant)
   - A100-SXM-40GBx4 (AS -2124GQ-NART, 40GB variant)
   - A100-PCIex8 (AS -4124GS-TNR)
 - Edge systems
   - A100-PCIex1 (SBA-4119SG-X)
 

