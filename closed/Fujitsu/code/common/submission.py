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
sys.path.insert(0, os.getcwd())
from code.common import logging, dict_get
from code.common import BENCHMARKS, SCENARIOS
import shutil
import textwrap
import json

TENSORRT_VERSION = ""
TRITON_VERSION = ""

high_acc_benchmarks = {BENCHMARKS.DLRM, BENCHMARKS.BERT, BENCHMARKS.UNET}

# option name to config file map
options_map = {
    "single_stream_expected_latency_ns": "target_latency",
    "single_stream_target_latency_percentile": "target_latency_percentile",
    "offline_expected_qps": "target_qps",
    "multi_stream_samples_per_query": "samples_per_query",
    "multi_stream_target_qps": "target_qps",
    "multi_stream_target_latency_ns": "target_latency",
    "multi_stream_max_async_queries": "max_async_queries",
    "multi_stream_target_latency_percentile": "target_latency_percentile",
    "server_target_qps": "target_qps",
    "server_target_latency_percentile": "target_latency_percentile",
    "server_target_latency_ns": "target_latency",
}

parameter_scaling_map = {
    "target_latency": 1 / 1000000.0,
    "target_latency_percentile": 100.0,
}


def generate_measurements_entry(system_name, short_benchmark_name, full_benchmark_name, scenario, input_dtype, precision, flag_dict):
    measurements_dir = "measurements/{:}/{:}/{:}".format(system_name, full_benchmark_name, scenario)
    os.makedirs(measurements_dir, exist_ok=True)

    mlperf_conf_path = os.path.join(measurements_dir, "mlperf.conf")
    user_conf_path = os.path.join(measurements_dir, "user.conf")
    readme_path = os.path.join(measurements_dir, "README.md")
    calibration_process_path = os.path.join(measurements_dir, "calibration_process.adoc")
    system_json_path = os.path.join(measurements_dir, "{:}_{:}.json".format(system_name, scenario))

    if "mlperf_conf_path" not in flag_dict:
        flag_dict["mlperf_conf_path"] = mlperf_conf_path
    if "user_conf_path" not in flag_dict:
        flag_dict["user_conf_path"] = user_conf_path

    # Override perf_sample_count. Make sure it's larger than the values required by the rules.
    flag_dict["performance_sample_count_override"] = flag_dict["performance_sample_count"]

    # Copy mlperf.conf
    generate_mlperf_conf(mlperf_conf_path)

    # Auto-generate user.conf
    generate_user_conf(user_conf_path, scenario, flag_dict)

    # Write out README.md (required file by MLPerf Submission rules)
    generate_readme(readme_path, short_benchmark_name, scenario)

    # Generate calibration_process.adoc
    generate_calibration_process(calibration_process_path, short_benchmark_name, scenario)

    # Generate system json
    generate_system_json(system_json_path, short_benchmark_name, input_dtype, precision)


def generate_mlperf_conf(mlperf_conf_path):
    shutil.copyfile("build/inference/mlperf.conf", mlperf_conf_path)


def generate_user_conf(user_conf_path, scenario, flag_dict):
    # Required settings for each scenario
    common_required = ["performance_sample_count_override"]
    required_settings_map = {
        SCENARIOS.SingleStream: [] + common_required,  # "single_stream_expected_latency_ns", See: https://github.com/mlperf/inference/issues/471
        SCENARIOS.Offline: ["offline_expected_qps"] + common_required,
        SCENARIOS.MultiStream: ["multi_stream_samples_per_query"] + common_required,
        SCENARIOS.Server: ["server_target_qps"] + common_required,
    }

    # Optional settings that we support overriding
    common_optional = ["min_query_count", "max_query_count", "min_duration", "max_duration"]
    optional_settings_map = {
        SCENARIOS.SingleStream: ["single_stream_target_latency_percentile"] + common_optional,
        SCENARIOS.Offline: [] + common_optional,
        SCENARIOS.MultiStream: ["multi_stream_target_qps", "multi_stream_target_latency_ns", "multi_stream_max_async_queries", "multi_stream_target_latency_percentile"] + common_optional,
        SCENARIOS.Server: ["server_target_latency_percentile", "server_target_latency_ns"] + common_optional,
    }

    with open(user_conf_path, 'w') as f:
        for param in required_settings_map[scenario]:
            param_name = param
            if param in options_map:
                param_name = options_map[param]
            value = flag_dict[param]
            if param_name in parameter_scaling_map:
                value = value * parameter_scaling_map[param_name]
            f.write("*.{:}.{:} = {:}\n".format(scenario, param_name, value))
            flag_dict[param] = None

        for param in optional_settings_map[scenario]:
            if param not in flag_dict:
                continue
            param_name = param
            if param in options_map:
                param_name = options_map[param]
            value = flag_dict[param]
            if param_name in parameter_scaling_map:
                value = value * parameter_scaling_map[param_name]
            f.write("*.{:}.{:} = {:}\n".format(scenario, param_name, value))
            flag_dict[param] = None


def generate_readme(readme_path, short_benchmark_name, scenario):
    readme_str = textwrap.dedent("""\
    To run this benchmark, first follow the setup steps in `closed/Fujitsu/README.md`. Then to generate the TensorRT engines and run the harness:

    ```
    make generate_engines RUN_ARGS="--benchmarks={benchmark} --scenarios={scenario}"
    make run_harness RUN_ARGS="--benchmarks={benchmark} --scenarios={scenario} --test_mode=AccuracyOnly"
    make run_harness RUN_ARGS="--benchmarks={benchmark} --scenarios={scenario} --test_mode=PerformanceOnly"
    ```

    For more details, please refer to `closed/Fujitsu/README.md`.""".format(benchmark=short_benchmark_name, scenario=scenario))
    with open(readme_path, 'w') as f:
        f.write(readme_str)


def generate_calibration_process(calibration_process_path, short_benchmark_name, scenario):
    calibration_process_str = textwrap.dedent("""\
    To calibrate this benchmark, first follow the setup steps in `closed/Fujitsu/README.md`.

    ```
    make calibrate RUN_ARGS="--benchmarks={benchmark} --scenarios={scenario}"
    ```

    For more details, please refer to `closed/Fujitsu/README.md`.""".format(benchmark=short_benchmark_name, scenario=scenario))
    with open(calibration_process_path, 'w') as f:
        f.write(calibration_process_str)


def generate_system_json(system_json_path, short_benchmark_name, input_dtype, precision):
    starting_weights_filename_map = {
        BENCHMARKS.ResNet50: "resnet50_v1.onnx",
        BENCHMARKS.SSDResNet34: "resnet34-ssd1200.pytorch",
        BENCHMARKS.SSDMobileNet: "ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb",
        BENCHMARKS.RNNT: "DistributedDataParallel_1576581068.9962234-epoch-100.pt",
        BENCHMARKS.DLRM: "tb00_40M.pt",
        BENCHMARKS.BERT: "bert_large_v1_1_fake_quant.onnx",
        BENCHMARKS.UNET: "224_224_160_dyanmic_bs.onnx",
    }

    data = {
        "input_data_types": input_dtype,
        "retraining": "N",
        "starting_weights_filename": starting_weights_filename_map[short_benchmark_name],
        "weight_data_types": precision,
        "weight_transformations": "quantization, affine fusion"
    }

    with open(system_json_path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)
