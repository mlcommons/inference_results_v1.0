# Neuchips Recommender Inference Accelerator: RecAccel

This document describes steps to run Neuchips RecAccel recommender inference
accelerator in general. While realized on Field Programmable Gate Array
(FPGA), RecAccel is initially configurated and then loads specified
embedding tables to on-board DRAM. Once RecAccel is ready, users could
feed input data and retrieve inference results through mlperf-compatible software 
framework. The following instructions show how to use the RecAccel system.

## Execution

The software stack includes framework, runtime libraries, FPGA device
driver, and MLPerf official pytorch benchmark to deploy RecAccel hardware engine.
MLPerf official commands for Criteo dataset are supported within this release.

The following list shows all the binaries used for submission.

- terasic_pcie_qsys.so -- runtime library

- irecaccel.ko -- FPGA device driver

Hash values of each of the binaries are as follows:

- terasic_pcie_qsys.so, SHA256=02247fb6461b43b639b0c62cf221337ae80f234c4202cf17ea99aec78a1d690c

- irecaccel.ko, SHA256=0c54e5fbaef3da158968e46775bc1ffe0420574b19bce553af0aba7c5f9b8729

