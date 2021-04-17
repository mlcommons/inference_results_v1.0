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

from code.common import logging, run_command
import os
import re
import shutil


def verify_test01(harness):
    # Compute path to results dir
    script_path = 'build/inference/compliance/nvidia/TEST01/run_verification.py'
    results_path = os.path.join('results', harness.get_system_name(), harness._get_submission_benchmark_name(), harness.scenario)
    logging.info('AUDIT HARNESS: ' + results_path + '/accuracy' + '\n' + results_path + '/performance')
    verification_command = 'python3 {} --results={} --compliance={} --output_dir={}'.format(
        script_path, results_path, harness.get_full_log_dir(), harness.get_full_log_dir())
    try:
        command_result = run_command(verification_command, get_output=True)
    except:
        # Handle test 01 failure
        logging.info('TEST01 verification failed. Proceeding to fallback approach')
        command_result = 'TEST01 FALLBACK'  # Signal main.py to finish the process
    return command_result


def verify_test04(harness):
    current_path = harness.get_full_log_dir()  # Might be using TEST04-B instead of TEST04-A
    test04a_path = current_path.replace('TEST04-B', 'TEST04-A')  # Make sure it's TEST04-A
    test04b_path = test04a_path.replace('TEST04-A', 'TEST04-B')  # Make sure it's TEST04-B
    output_path = harness.get_full_log_dir()
    script_path = 'build/inference/compliance/nvidia/TEST04-A/run_verification.py'
    verification_command = 'python3 {} --test4A_dir {} --test4B_dir {} --output_dir {}'.format(
        script_path,
        test04a_path,
        test04b_path,
        output_path
    )
    return run_command(verification_command, get_output=True)


def verify_test05(harness):
    # Compute path to results dir
    script_path = 'build/inference/compliance/nvidia/TEST05/run_verification.py'
    results_path = os.path.join('results', harness.get_system_name(), harness._get_submission_benchmark_name(), harness.scenario)
    logging.info('AUDIT HARNESS: ' + results_path + '/accuracy' + '\n' + results_path + '/performance')
    verification_command = 'python3 {} --results_dir={} --compliance_dir={} --output_dir={}'.format(
        script_path,
        results_path,
        harness.get_full_log_dir(),
        harness.get_full_log_dir())
    return run_command(verification_command, get_output=True)


def load(audit_test, benchmark):
    # Calculates path to audit.config
    src_config = os.path.join('build/inference/compliance/nvidia', audit_test, benchmark, 'audit.config')
    logging.info('AUDIT HARNESS: Looking for audit.config in {}...'.format(src_config))
    if not os.path.isfile(src_config):
        # For tests that have one central audit.config instead of per-benchmark
        src_config = os.path.join('build/inference/compliance/nvidia', audit_test, 'audit.config')
        logging.info('AUDIT HARNESS: Search failed. Looking for audit.config in {}...'.format(src_config))
    # Destination is audit.config
    dest_config = 'audit.config'
    # Copy the file
    shutil.copyfile(src_config, dest_config)
    return dest_config


def cleanup():
    """Delete files for audit cleanup."""
    tmp_files = ["audit.config", "verify_accuracy.txt", "verify_performance.txt", "mlperf_log_accuracy_baseline.json", "accuracy.txt", "predictions.json"]
    for fname in tmp_files:
        if os.path.exists(fname):
            logging.info('Audit cleanup: Removing file {}'.format(fname))
            os.remove(fname)
