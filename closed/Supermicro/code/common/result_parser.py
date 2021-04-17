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
from code.common import SCENARIOS
import json

MLPERF_LOG_PREFIX = ":::MLLOG"

scenario_loadgen_log_keys = {
    SCENARIOS.MultiStream: "requested_multi_stream_samples_per_query",
    SCENARIOS.Offline: "result_samples_per_second",
    SCENARIOS.Server: "result_scheduled_samples_per_sec",
    SCENARIOS.SingleStream: "result_90.00_percentile_latency_ns",
}


def from_loadgen_by_keys(log_dir, keys, return_list=False):
    """
    Gets values of certain keys from loadgen detailed logs, based on the new logging design.
    """
    detailed_log = os.path.join(log_dir, "mlperf_log_detail.txt")
    with open(detailed_log) as f:
        lines = f.read().strip().split("\n")

    log_entries = []
    for line in lines:
        if line.startswith(MLPERF_LOG_PREFIX):
            buf = line[len(MLPERF_LOG_PREFIX) + 1:]
            log_entries.append(json.loads(buf))

    results = {}
    for entry in log_entries:
        key = entry["key"]
        if key in keys:
            if return_list:
                if key not in results:
                    results[key] = []
                results[key].append(entry["value"])
            else:
                results[key] = entry["value"]
    return results
