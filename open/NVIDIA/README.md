# MLPerf Inference NVIDIA-Optimized Implementations for Open Division

## ResNet50 INT4 Precision

Documentation on the ResNet50 implementation can be found in
[code/resnet50/int4/README.md](code/resnet50/int4/README.md).

This submission is only supported on A10x1, TitanRTXx4, T4x8, and T4x20 systems.

To run the ResNet50 implementation, first launch the container with:
```
$ make prebuild_int4
```

Then within the container, run:
```
$ make build
$ make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --test_mode=AccuracyOnly"
$ make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --test_mode=PerformanceOnly"

```

If this is part of a submission, you will want to run:
```
$ python3 scripts/update_results_int4.py
```
This will export the logs into the `results/` directory. To truncate the accuracy logs, follow the instructions in the
closed submission (`closed/NVIDIA/README.md`).

