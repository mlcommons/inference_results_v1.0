To run this benchmark, first follow the setup steps in `closed/NVIDIA/README_Triton_CPU.md`. Then to generate the TensorRT engines and run the harness:

```
make run_harness RUN_ARGS="--benchmarks=ssd-resnet34 --config_ver=openvino --use_triton --scenarios=Server --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=ssd-resnet34 --config_ver=openvino --use_triton --scenarios=Server --test_mode=PerformanceOnly"
```

For more details, please refer to `closed/NVIDIA/README_Triton_CPU.md`.
