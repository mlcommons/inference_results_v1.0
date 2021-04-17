# MLPerf Inference Submission Guide

This submission guide will help you prepare everything you need for a valid MLPerf Inference submission that satisfies the [inference rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc) and the [submission rules](https://github.com/mlperf/policies/blob/master/submission_rules.adoc). Please check the rules for the latest updates.

Most part of this guide is general and applies to all MLPerf Inference submissions. It is clearly mentioned if any part is specific to NVIDIA's submission code and our directory structure.

## Directory Structures

The most up-to-date directory structure can be found in the [submission rules](https://github.com/mlperf/policies/blob/master/submission_rules.adoc#inference-1).
The required directories and files are as follows:

```
closed/$(SUBMITTER)/
|-- code/
|   |-- <benchmark name>
|   |   `-- <implementation id>
|   |       `-- <code interface with loadgen and other necessary code>
|   `-- ...
|-- measurements/
|   |-- <system_desc_id>
|   |   |-- <benchmark_name>
|   |   |   |-- <scenario>
|   |   |   |   |-- <system_desc_id>_<implementation id>_<scenario>.json
|   |   |   |   |-- README.md
|   |   |   |   |-- calibration_process.adoc
|   |   |   |   |-- mlperf.conf
|   |   |   |   `-- user.conf
|   |   |   `-- ...
|   |   `-- ...
|   `-- ...
|-- results/
|   |-- compliance_checker_log.txt # stdout of submission checker script
|   `-- <system_desc_id>
|       |-- <benchmark_name>
|       |   `-- <scenario>
|       |       |-- accuracy/
|       |       |   |-- accuracy.txt # stdout of reference accuracy script
|       |       |   |-- mlperf_log_accuracy.json # Truncated by truncate_accuracy script
|       |       |   |-- mlperf_log_detail.txt
|       |       |   `-- mlperf_log_summary.txt
|       |       `-- performance/
|       |           |-- ranging                             # (only needed if power submission)
|       |           |   |-- mlperf_log_detail.txt           # ranging run
|       |           |   |-- mlperf_log_summary.txt          # ranging run
|       |           |   `-- spl.txt                         # ranging run
|       |           |-- run_1/ # 1 run for all scenarios
|       |           |   |-- mlperf_log_detail.txt           # testing run
|       |           |   |-- mlperf_log_summary.txt          # testing run
|       |           |   |-- spl.txt                         # testing run (only needed if power submission)
|       |           `-- power                               # (only needed if power submission)
|       |               |-- client.json
|       |               |-- client.log
|       |               |-- ptd_logs.txt
|       |               |-- server.json
|       |               `-- server.log
|       `-- ...
|-- systems/
|   |-- <system_desc_id>.json # combines hardware and software stack information
|   `-- ...
`-- compliance
    `-- <system_desc_id>
        |-- <benchmark_name>
        |   |-- <scenario>
        |   |   |-- <test_id>
        |   |   |   |-- verify_performance.txt
        |   |   |   |-- verify_accuracy.txt # For TEST01 only
        |   |   |-- accuracy/ # For TEST01 only
        |   |   |   |   |-- accuracy.txt # stdout of reference accuracy script
        |   |   |   |   |-- mlperf_log_accuracy.json # Truncated by truncate_accuracy script
        |   |   |   |   |-- baseline_accuracy.txt # only for TEST01 if accuracy check fails
        |   |   |   |   |-- compliance_accuracy.txt # only for TEST01 if accuracy check fails
        |   |   |   |   |-- mlperf_log_detail.txt
        |   |   |   |   `-- mlperf_log_summary.txt
        |   |   |   `-- performance/
        |   |   |       `-- run_1/ # 1 run for all scenarios
        |   |   |           |-- mlperf_log_detail.txt
        |   |   |           `-- mlperf_log_summary.txt
        |   |   `-- ...
        |   `-- ...
        `-- ...
```

Where `<benchmark_name>` is one of **{resnet50, ssd-mobilenet, ssd-resnet34, rnnt, bert-99, bert-99.9, dlrm-99, dlrm-99.9, 3d-unet-99, 3d-unet-99.9}**, `<scenario>` is one of **{Offline, Server, SingleStream, MultiStream}**, and `<test_id>` is one of **{TEST01, TEST04-A, TEST04-B, TEST05}**. You can also find this list in the official [submission rules](https://github.com/mlperf/policies/blob/master/submission_rules.adoc#562-inference).

Other than required files, submitters can put any additional files as long as they don't cause confusion with the required files.

## Benchmark Codes

The benchmark implementation codes and the interfaces with LoadGen, including the QSL and the SUT implementations, should be placed under `code/<BENCHMARK>/<IMPLEMENTATION>` directory. The implementation codes can be a copy of the reference implementation with needed modifications, or a from-scratch implementation so long as it satisfies the [model equivalence](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc#model-equivalence) requirements.

In NVIDIA submission code, the codes under `code` can be used directly as is. The only required modification is to update [code/common/system_list.py](code/common/system_list.py) with the actual submission systems.

To submit under a different company name, you will need to:
 - Move `closed/NVIDIA` to `closed/<your company name>`
 - In your submission system descriptions, change the `submitter` field to your company name.
 - In `closed/<your company name>/Makefile`, redefine the `SUBMITTER` variable to the correct value, or append `SUBMITTER=<your company name>` to all of your `make` commands when running the code.

## System Descriptions

Under `systems` directory, each submission system must have a system description file named `<system_id>.json`.

Below are the required fields in the system description JSON file. These entries cannot be empty strings:

- `accelerator_memory_capacity`
- `accelerator_model_name`
- `accelerators_per_node`
- `division`
- `framework`
- `host_memory_capacity`
- `host_processor_core_count`
- `host_processor_model_name`
- `host_processors_per_node`
- `host_storage_capacity`
- `host_storage_type`
- `number_of_nodes`
- `operating_system`
- `submitter`
- `status`
- `system_name`
- `system_type`

Below are the optional fields in the system description JSON file. These entries can empty strings if they don't apply to your systems:

- `accelerator_frequency`
- `accelerator_host_interconnect`
- `accelerator_interconnect`
- `accelerator_interconnect_topology`
- `accelerator_memory_configuration`
- `accelerator_on-chip_memories`
- `cooling`
- `host_networking`
- `host_networking_topology`
- `host_memory_configuration`
- `host_processor_caches`
- `host_processor_frequency`
- `host_processor_interconnect`
- `hw_notes`
- `other_software_stack`
- `sw_notes`

In NVIDIA submission code, you can use one of the provided system description JSON file as the starting point. Please remove all the other system description JSON files of systems which you don't plan to submit with before the submission.

## User Configurations

MLPerf Inference submission rules required that you put five files under `measurements/<system_id>/<benchmark_name>/<scenario>/`:

- `<system_desc_id>_<implementation_id>_<scenario>.json`: a JSON file containing the following entries:
  - `input_data_types`: data type of the input.
  - `retraining`: whether the weights are modified with retraining.
  - `starting_weights_filename`: the filename of the original reference weights (model) used in the implementation.
  - `weight_data_types`: data type of the weights.
  - `weight_transformations`: transformations applied to the weights.
- `README.md`: instructions about how to run this benchmark.
- `mlperf.conf`: LoadGen config file with rule complying settings. It is a copy of the official [mlperf.conf](https://github.com/mlperf/inference/blob/master/mlperf.conf) file.
- `user.conf`: LoadGen config file with user settings, including `target_qps` for Server and Offline scenarios, `target_latency` for SingleStream scenario, `samples_per_query` in MultiStream scenario, and optionally `performance_sample_count_override` for all scenarios. See the comments in the official [mlperf.conf](https://github.com/mlperf/inference/blob/master/mlperf.conf) file for details.
- `calibration_process.adoc`: documentation about how post-training calibration/quantization is done.

In NVIDIA submission code, **do not modify the files in `measurements/` directly** since they are generated with our script automatically. Please modify the `code/common/system_list.py` to include only the systems you plan to submit with, modify `config.json` files under `configs/` for the systems you are submitting with, remove `measurements` directory, and then run `make generate_conf_files` to generate all the files needed in `measurements`. Alternatively, since the `measurements/` is updated automatically when you run the `make run_harness ...` commands, you can also run all the benchmarks, scenarios, and systems you plan to submit at least once each so that the `measurements/` will be ready for submission.

## Result Logs

For each system, each benchmark, and each scenario, you will need to generate LoadGen logs and place them under `results/<system_id>/<benchmark>/<scenario>/performance/run_x/` and `results/<system_id>/<benchmark>/<scenario>/accuracy/`. The performance run or accuracy run can be controlled by setting the LoadGen flag `mode` to `TestMode::PerformanceOnly` or `TestMode::AccuracyOnly`. One performance log and one accuracy log is required per system-benchmark-scenario combination.

For performance runs, the required files are `mlperf_log_summary.txt` and `mlperf_log_detail.txt` generated by LoadGen. The accuracy runs require these two files, plus the `mlperf_log_accuracy.json` generated by LoadGen and the `accuracy.txt` which is the stdout log of running official accuracy script on the `mlperf_log_accuracy.json` file to compute the accuracy.

In NVIDIA submission code structure, follow the steps below to generate the required files under `results/`:

- Remove any existing logs with `rm -rf build/logs`.
- Run the benchmarks and the scenarios you would like to submit results in with `make run_harness RUN_ARGS="..."`. Please pass in `--config_ver=default,high_accuracy` to RUN_ARGS to make sure that both accuracy targets are covered. Also, for each benchmark and each scenario, you will need a performance run (default) and an accuracy run (with `--test_mode=AccuracyOnly` in RUN_ARGS). The Server scenario requires at least one performance runs with VALID results. The LoadGen logs will be generated in `build/logs`.
- Make sure your system description json files `<SYSYEM_ID>.json` exist in `systems/`.
- To generate `results/`: run `make update_results`. The script will search `build/logs` for all the valid LoadGen logs, and copy them to `results/`.
- At this point, to run the compliance tests to generate the compliance logs, see the section below. You will be unable to run audit `TEST01` if you truncate the accuracy logs.

## Compliance Logs

After completing the result logs in `results/`, you will also need to generate compliance logs for the four different tests listed in the [official compliance tests](https://github.com/mlperf/inference/tree/master/compliance/nvidia). Follow the steps in the compliance test documentation and run the compliance scripts, which will put the compliance logs under `compliance/` directory.
There are three tests:

- `TEST01`: Samples and logs response randomly in PerformanceOnly mode and verifies that the SUT generates the same outputs for the same set of inputs in PerformanceOnly mode and in AccuracyOnly mode. This is to prevent cheating by generating garbage responses in PerformanceOnly mode. However, this test may fail if your SUT has nondeterministic characteristics, such as running on a hybrid of hardwares (such as GPU and DLA) or when the code path is decided on runtime factors (such as choosing different CUDA kernels base on the characteristics of a batch at runtime). In that case, the rules require that you provide an explanation about the nondeterministic behavior and manually check the accuracy of PerformanceOnly mode output accuracy logs by following the steps in [TEST01 Part III](https://github.com/mlperf/inference/tree/master/compliance/nvidia/TEST01#part-iii).
- `TEST04-A/B`: Checks the performance with same samples and with unique samples and detects if SUT is possibly doing result caching. It fails when the performance with same samples are significantly faster than the performance with unique samples. It is expected that this test will show as `INVALID` in the summary logs and it is safe to ignore them.
- `TEST05`: Runs the normal PerformanceOnly runs with a different set of seeds. This ensures the submitter does not hyper-tune the SUT on a specific seed.

The `run_verification.py` script for each test automatically put all the required logs and files under `compliance/` directory with the required directory structure if the tests pass.

Note that you only need to run these tests in PerformanceOnly mode. Don't run them in AccuracyOnly mode. Also note that these tests work by placing the corresponding `audit.conf` at the working directory so that LoadGen can load the test settings, so please remove `audit.conf` from the working directory if you plan to run the normal runs instead of compliance runs.

In NVIDIA submission code, follow the steps below to generate the required files under `compliance/`:

**Make sure you run the below commands after `results/` is populated with `make update_results` command, but before the accuracy logs are truncated.**

- Run the benchmarks and the scenarios you would like to submit results in with `make run_audit_harness RUN_ARGS="..."`, which is identical to how to run the normal runs except for using `run_audit_test01`, `run_audit_test04`, and `run_audit_test05` instead of `run_harness`. Accuracy run is not needed.
- The stdout logs will show if the TEST PASS or TEST FAIL. If any compliance tests fail, follow the instructions in the [official compliance page](https://github.com/mlperf/inference/tree/master/compliance/nvidia) for how to solve the issue.
- In NVIDIA submission code, it is expected that BERT benchmark will fail the first two parts of TEST01 test since some CUDA kernels are chosen at runtime depending on how the sequences are batched, resulting in tiny difference in the output values without affecting accuracy. Our script already implemented the steps in [TEST01 Part III](https://github.com/mlperf/inference/tree/master/compliance/nvidia/TEST01#part-iii) to compare the baseline accuracy and the compliance run accuracy, which should match, and to generate the `baseline_accuracy.txt` and `compliance_accuracy.txt` required by the rules for TEST01 failure.
- To generate `compliance/`: run `make update_compliance`. The script will search `build/compliance_logs` and copy audit test logs to `compliance/`.

## Truncate Accuracy Logs

Since `mlperf_log_accuracy.json` can be large in size, the rules require that the submitters truncate the `mlperf_log_accuracy.json` files with the official [truncation script](https://github.com/mlperf/inference/blob/master/tools/submission/truncate_accuracy_log.py), which replaces the `mlperf_log_accuracy.json` file with only the first 4k and the last 4k characters of the file, and append the hash of the original `mlperf_log_accuracy.json` file at the end of `accuracy.txt` file. The original `mlperf_log_accuracy.json` should be kept in case other submitters request to see the original `mlperf_log_accuracy.json` logs during the review process.

In NVIDIA submission code structure, follow the steps below to truncate the accuracy logs under `compliance/` and `results/`:

- To truncate the accuracy logs: run `make truncate_results SUBMITTER=<your_company>` from `closed/<your_company>` **outside of the container**. This will run the official truncation script, and back up the full, non-truncated accuracy logs in `build/full_results/`.
- **IMPORTANT:** Please keep a copy of `build/full_results` somewhere else since the next `make truncate_results` run will override the existing non-truncated logs. You should keep the non-truncated logs separately in case other submitters have questions about your truncated logs and would like to see the original logs during the review process.

## Submission Checker

The entire submission repository needs to pass the official [submission-checker.py](https://github.com/mlperf/inference/blob/master/tools/submission/submission-checker.py) to be considered as a valid submission. This script checks if there are missing files, errors in LoadGen logs, invalid LoadGen settings, and other common issues which do not satisfy the inference rules or the submission rules. The stdout of this script should be included in `results/submission_checker_log.txt` as the proof of a valid submission.

In NVIDIA submission code, run `make check_submission SUBMITTER=<your_company>` from `closed/<your_company>` **outside of the container** to run the submission checker script. It will save the output of the submission checker script to `results/submission_checker_log.txt` for submission. Please make sure the there are no errors in the `submission_checker_log.txt`.

## Encrypting your project for submission

In this version of MLPerf Inference, there is a new secure submission process where submitters must generate an encrypted
tarball of their submission repository and submit that along with the decryption password and SHA1 hash of the encrypted
tarball to the MLPerf Inference results chair.

In NVIDIA's submission, there is a script located at `scripts/pack_submission.sh`. To create an encrypted tarball and
generate the SHA1 of the tarball, first change the `SUBMITTER` variable in `scripts/pack_submission.sh` to your company
name. Then from the project root, run:

```
bash scripts/pack_submission.sh --pack
```

This command will prompt to enter and then confirm an encryption password. You must relay this password to the MLPerf
Inference results chair. After this command finishes running, there will be at least 2 files:

 - `mlperf_submission_${SUBMITTER}.tar.gz` - The encrypted tarball, encrypted with AES256
 - `mlperf_submission_${SUBMITTER}.sha1` - A text file containing the sha1 hash of the encrypted tarball

If you have run the submission checker, this script will also grab the last 2 lines from
`closed/NVIDIA/results/submission_checker_log.txt` as a summary. This file will be stored at:

 - `mlperf_submission_${SUBMITTER}_checker_summary.txt`

## To submit your submission

To submit, the following actions must be performed *before* the submission deadline.

After you have run the above script:
 - `mlperf_submission_${SUBMITTER}.tar.gz` - The encrypted tarball must be uploaded to some online storage that can be
   accessed by the MLPerf Inference results chair.

The following files must be emailed to the MLPerf Inference results chair:
 - A link to the cloud storage where your encrypted tarball is stored
 - `mlperf_submission_${SUBMITTER}.sha1` - A text file containing the sha1 hash of the encrypted tarball
 - The decryption password
 - Your modified `scripts/pack_submission.sh` to decrypt the tarball, with instructions to run:
 ```
 bash path/to/pack_submission.sh --unpack
 ```
 - `mlperf_submission_${SUBMITTER}_checker_summary.txt` - A summary of the submission checker output

**IMPORTANT**: The current MLPerf Inference Results Chair is Guenther Schmuelling (guschmue@microsoft.com).

## Common issues

Below summarizes some of the common issues when working with MLPerf Inference submission:

### LoadGen Seeds

LoadGen random seeds are released two weeks before submission to prevent submitters from over-tuning on specific set of seeds. The official seeds will be added to the official [mlperf.conf](https://github.com/mlperf/inference/blob/master/mlperf.conf) file once announced. Please make sure you are using the latest `mlperf.conf` in the official [inference repository](https://github.com/mlperf/inference).

In NVIDIA submission code, we automatically use the official [mlperf.conf](https://github.com/mlperf/inference/blob/master/mlperf.conf) file, so no further action is needed.

### Minimal Query Count

[MLPerf Inference rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc#3-scenarios) require that each performance test to run for at least some number of queries or samples to ensure sufficient statistical confidence in the reported metric. If you use the latest official [mlperf.conf](https://github.com/mlperf/inference/blob/master/mlperf.conf) file, these LoadGen settings are automatically applied as long as you don't overwrite the settings in your `user.conf` files. Below is the summary of this requirement:

- Offline: at least 24576 samples.
- SingleStream: at least 1024 queries.
- MultiStream: at least 270336 queries.
- Server: at least 270336 queries.

This means a performance run in MultiStream scenario will always run for 4-5 hours and a performance run Offline, SingleStream, and Server scenarios may be long when the QPS of the system is low. Please refer to the [performance_tuning_guide.md](performance_tuning_guide.md) for how to calculate the expected performance test runtime.

In NVIDIA submission node, no further action is needed.

### Performance Sample Count

[MLPerf Inference rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc#411-constraints-for-the-closed-division) require that the Loadable Set size of the QSL (which is also called `performance_sample_count`) to be above some number to avoid implicit caching effect. These are defined in the official [mlperf.conf](https://github.com/mlperf/inference/blob/master/mlperf.conf) file already. Submitters can override the `performance_sample_count` settings in the `user.conf` file as long as the numbers are greater than the requirement.

In NVIDIA submission node, no further action is needed.

### INVALID Results

If you see `INVALID` results in the LoadGen summary log, it usually means the target QPS settings do not match the actual QPS of your submission systems. Please follow the [performance_tuning_guide.md](performance_tuning_guide.md) to fix the `INVALID` results.

### LoadGen Git Hash and Uncommitted Changes

To ensure that all the submitters use the same LoadGen, the LoadGen build system checks the git status and the hash of the latest git commit when the LoadGen is being built. There are a set of approved git commit hash for each submission round, which is usually announced two weeks before the submission. Therefore, please make sure that you have the up-to-date official [inference](https://github.com/mlperf/inference) repository and the result of `git status` is clean before you build the LoadGen. Also, do not switch environments between where you clone the [inference](https://github.com/mlperf/inference) repository and where you build the LoadGen, such as cloning outside of the container and building inside the container. Check the LoadGen detailed logs to see if the LoadGen git commit hash matches the official approved hashes or if there are any uncommitted changes.

### Other LoadGen ERRORs

The [submission-checker.py](https://github.com/mlperf/inference/blob/master/tools/submission/submission-checker.py) checks if there are any ERRORs in the LoadGen detailed logs. There are a few allowed LoadGen ERRORs which seem to be caused by LoadGen issue, and are waived in the [submission-checker.py](https://github.com/mlperf/inference/blob/master/tools/submission/submission-checker.py) already. If there are other LoadGen ERRORs which you think are caused by LoadGen issue, please create an Issue in the [inference](https://github.com/mlperf/inference) repository so that the WG can discuss about it.
