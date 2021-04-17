/*
 // Copyright (c) 2020 Intel Corporation
 //
 // Licensed under the Apache License, Version 2.0 (the "License");
 // you may not use this file except in compliance with the License.
 // You may obtain a copy of the License at
 //
 //      http://www.apache.org/licenses/LICENSE-2.0
 //
 // Unless required by applicable law or agreed to in writing, software
 // distributed under the License is distributed on an "AS IS" BASIS,
 // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 // See the License for the specific language governing permissions and
 // limitations under the License.
 */

#include <iostream>

#include "args.h"

static const char help_message[] = "Print a usage message.";

static const char model_message[] = "Path to IR .xml file. Required.";
static const char data_path_message[] = "Path to preprocessed data file in .pkl format.";

static const char device_message[] = "Device name. [CPU]. CPU by default.";
static const char streams_message[] = "Number of inference streams. 1 by default.";
static const char threads_message[] = "Number of thfreads to use. 0 (all available) by default.";

static const char scenario_message[] = "Benchmark scenario. [Offline]. Offline by default";
static const char mode_message[] = "Benchmark mode. [Accuracy | Performance | Submission | FindPeakPerformance]. Performance by default.";

static const char sample_count_message[] = "Number of samples to use.";
static const char perf_sample_count_message[] = "Number of samples to use for performance mode.";

static const char mlperf_config_message[] = "Path to mlperf.conf";
static const char mlperf_user_config_message[] = "Path to user.conf.";

DEFINE_bool(h, false, help_message);

DEFINE_string(m, "", model_message);
DEFINE_string(data, "", data_path_message);

DEFINE_string(device, "CPU", device_message);
DEFINE_uint32(streams, 1, streams_message);
DEFINE_uint32(threads, 0, threads_message);

DEFINE_string(scenario, "Offline", scenario_message);
DEFINE_string(mode, "Performance", mode_message);

DEFINE_uint32(sc, 0, sample_count_message);
DEFINE_uint32(psc, 16, perf_sample_count_message);

DEFINE_string(mlperf_conf, "", mlperf_config_message);
DEFINE_string(user_conf, "", mlperf_user_config_message);

void Args::ShowUsage() {
    std::cout << std::endl;
    std::cout << "ov_mlperf [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                                    " << help_message << std::endl << std::endl;
    
    std::cout << "    -m <path>                             " << model_message << std::endl;
    std::cout << "    -data <path>                          " << data_path_message << std::endl;

    std::cout << "    -device <device name>                 " << scenario_message << std::endl;
    std::cout << "    -streams N                            " << streams_message << std::endl;
    std::cout << "    -threads N                            " << threads_message << std::endl;

    std::cout << "    -scenario <scenario>                  " << scenario_message << std::endl;
    std::cout << "    -mode <mode>                          " << mode_message << std::endl;

    std::cout << "    -sc <samples count>                   " << data_path_message << std::endl;
    std::cout << "    -psc <performance samples count>      " << data_path_message << std::endl;

    std::cout << "    -mlperf_conf <path>                   " << mlperf_config_message << std::endl;
    std::cout << "    -user_conf <path>                     " << mlperf_user_config_message << std::endl;
}

std::string Args::ModelXml() {
    return FLAGS_m;
}

std::string Args::ModelBin() {
    return ModelXml().substr(0, ModelXml().size() - 4) + ".bin";
}

std::string Args::DataPath() {
    return FLAGS_data;
}

std::string Args::Device() {
    return FLAGS_device;
}

uint32_t Args::Streams() {
    return FLAGS_streams;
}

uint32_t Args::Threads() {
    return FLAGS_threads;
}

std::string Args::Scenario() {
    return FLAGS_scenario;
}

std::string Args::Mode() {
    return FLAGS_mode;
}

uint32_t Args::SampleCount() {
    return FLAGS_sc;
}

uint32_t Args::PerfSampleCount() {
    return FLAGS_psc;
}

std::string Args::MlperfConfigPath() {
    return FLAGS_mlperf_conf;
}

std::string Args::MlperfUserConfigPath() {
    return FLAGS_user_conf;
}

bool Args::Help() {
    return FLAGS_h;
}
