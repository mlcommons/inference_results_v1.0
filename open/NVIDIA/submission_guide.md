# MLPerf Inference Submission Guide

The general steps for submission are the same as documented in [the closed
submission](../../closed/NVIDIA/submission_guide.md). However, there are a few things you should note:

 1. Running the ResNet50 INT4 harness will cause the working directory to change to `code/resnet50/int4`. As such,
    `audit.config` must be placed into `code/resnet50/int4` for audit tests to run correctly.
 2. `make truncate_results` must be run from `closed/NVIDIA`, not `open/NVIDIA`.
 3. To `update_results` for ResNet50 INT4: Use `python3 scripts/update_results_int4.py`.
 4. Compliance / Audit tests are not yet automated in the open submission. See the instructions below for instructions
    on running the compliance tests.

## Running audit tests for ResNet50 INT4

 1. Launch the container with `make prebuild_int4`
 2. `make build` to build the executables. Make sure you ran `git lfs pull` if you downloaded the repo via `git clone`
    so `int4_offline.a` is valid.
 3. Run the accuracy and performance runs: `make run_int4_${SYSTEM_ID}_${TEST_MODE}` where `${SYSTEM_ID}` is `T4x8`,
    `T4x20`, or `TitanRTXx4`, and `${TEST_MODE}` is `accuracy` or `performance`.
 4. Run `python3 scripts/update_results_int4.py` to update `results/`.
 5. Run the audit tests:

```
# Run TEST01
cp build/inference/compliance/nvidia/TEST01/resnet50/audit.config code/resnet50/int4/audit.config
make run_int4_${SYSTEM}_performance LOG_DIR=/work/build/TEST01
python3 build/inference/compliance/nvidia/TEST01/run_verification.py --results=results/${SYSTEM}/resnet50/Offline/ --compliance=build/TEST01/${SYSTEM}/resnet/Offline/performance/run_1/ --output_dir=compliance/${SYSTEM}/resnet50/Offline/
# Cleanup
rm -f verify_accuracy.txt verify_performance.txt code/resnet50/int4/audit.config

# Run TEST04
cp build/inference/compliance/nvidia/TEST04-A/audit.config code/resnet50/int4/audit.config
make run_int4_$(SYSTEM)_performance LOG_DIR=/work/build/TEST04-A
cp build/inference/compliance/nvidia/TEST04-B/audit.config code/resnet50/int4/audit.config
make run_int4_$(SYSTEM)_performance LOG_DIR=/work/build/TEST04-B
python3 build/inference/compliance/nvidia/TEST04-A/run_verification.py --test4A_dir build/TEST04-A/${SYSTEM}/resnet/Offline/performance/run_1/ --test4B_dir build/TEST04-B/${SYSTEM}/resnet/Offline/performance/run_1/ --output_dir=compliance/${SYSTEM}/resnet50/Offline/
rm -f verify_accuracy.txt verify_performance.txt code/resnet50/int4/audit.config

# Run TEST05
cp build/inference/compliance/nvidia/TEST05/audit.config code/resnet50/int4/audit.config
make run_int4_$(SYSTEM)_performance LOG_DIR=/work/build/TEST05
python3 build/inference/compliance/nvidia/TEST05/run_verification.py --results_dir=results/${SYSTEM}/resnet50/Offline/ --compliance_dir=build/TEST05/${SYSTEM}/resnet/Offline/performance/run_1/ --output_dir=compliance/${SYSTEM}/resnet50/Offline/
```

 6. Run `make truncate_results` from `closed/NVIDIA`. This will back up the full accuracy logs to
    `closed/NVIDIA/build/full_results`. If that command fails with a file not found, you might need to run `make
    clone_loadgen` from `closed/NVIDIA`.

