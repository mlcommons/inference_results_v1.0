#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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



# A sanity test to quickly check if bare-bones features of grid.py work
# Takes ~1min to run on T4x1.

# Start by writing a simple py config file
tmp_py=/tmp/tmp_config.py
tmp_output=/tmp/tmp_output
cat > $tmp_py <<EOF
gpu_batch_size=[64]
audio_batch_size=[64, 128, 256, 512]

def META_get_no_rebuild_params():
    return ["audio_batch_size"]
def META_is_config_valid(config):
    if config['audio_batch_size'] > 128:
        return False
    return True
EOF
# And get ready to be a good citizen and cleanup after ourselves
cleanup() {
    rm $tmp_py
    rm $tmp_output
}
trap cleanup EXIT

# Do a dry run of the aforementioned config. Note that parsing may fail, but we shouldn't trigger it because of --noparse
python3 scripts/autotune/grid.py rnnt offline $tmp_py --dry_run --noparse |& tee $tmp_output
RET=${PIPESTATUS[0]}
if [ $RET -ne 0 ]; then
    echo "FAILED! Failed dry run"
    exit 1
fi

# We expect there to be two full runs, one cached build
grep  "builds=1, cached_builds=1, runs=2, cached_runs=0" $tmp_output
if [ $? -ne 0 ]; then
    echo "FAILED! Pre-run Session statistics not as expected"
    exit 1
fi

# Now we do a normal run, which we expect to succeed. (We add some extra args to make the run go fast)
python3 scripts/autotune/grid.py rnnt offline $tmp_py --temp_timeout=1 --extra_run_args="--min_query_count=1 --min_duration=1"|& tee $tmp_output
RET=${PIPESTATUS[0]}
if [ $RET -ne 0 ]; then
    echo "FAILED! Failed normal run"
    exit 1
fi

# Now do a cached dry-run, which we expect all runs to be cached
python3 scripts/autotune/grid.py rnnt offline $tmp_py --dry_run --use_cached |& tee $tmp_output
RET=${PIPESTATUS[0]}
if [ $RET -ne 0 ]; then
    echo "FAILED! Failed normal run"
    exit 1
fi

grep  "builds=0, cached_builds=0, runs=0, cached_runs=2" $tmp_output
if [ $? -ne 0 ]; then
    echo "FAILED! Post-run Session statistics not as expected"
    exit 1
fi

echo "PASSED"
