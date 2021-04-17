/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "glog/logging.h"
#include "NvInferPlugin.h"

#include "logger.h"
#include "test_settings.h"
#include "loadgen.h"

#include "qsl.hpp"
#include "bert_server.h"

#include "cuda_profiler_api.h"

DEFINE_string(gpu_engines, "", "Engine");
DEFINE_string(devices, "all", "Enable comma separated numbered devices");

DEFINE_string(scenario, "Offline", "Scenario to run for Loadgen (Offline, Server, SingleStream)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "bert", "Model name");
DEFINE_uint32(gpu_batch_size, 64, "Max Batch size to use for all devices and engines");
DEFINE_uint32(gpu_inference_streams, 1, "Number of streams (BERTCores) for inference");
DEFINE_uint32(gpu_copy_streams, 1, "Number of copy streams");
DEFINE_bool(use_graphs, false, "Enable CUDA Graphs for TensorRT engines");
DEFINE_uint32(graphs_max_seqlen, BERT_MAX_SEQ_LENGTH, "Max seqlen is used to control how many CUDA Graphs will be generated");
DEFINE_string(graph_specs, "", "Specify a comma separeated list of (maxSeqLen, min totSeqLen, max totSeqLen, step size) for CUDA graphs to be captured");
DEFINE_bool(verbose, false, "Use verbose logging");
DEFINE_bool(load_plugins, true, "Load TRT NvInfer plugins");

DEFINE_double(soft_drop, 1.0, "The threshold to soft drop requests when total length in a batch is too long");

// configuration files
DEFINE_string(mlperf_conf_path, "", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "", "Path to user.conf");
DEFINE_uint64(single_stream_expected_latency_ns, 100000, "Inverse of desired target QPS");
DEFINE_uint64(server_num_issue_query_threads, 0, "Number of IssueQuery threads used in Server scenario");

// Loadgen logging settings
DEFINE_string(logfile_outdir, "", "Specify the existing output directory for the LoadGen logs");
DEFINE_string(logfile_prefix, "", "Specify the filename prefix for the LoadGen log files");
DEFINE_string(logfile_suffix, "", "Specify the filename suffix for the LoadGen log files");
DEFINE_bool(logfile_prefix_with_datetime, false, "Prefix filenames for LoadGen log files");
DEFINE_bool(log_copy_detail_to_stdout, false, "Copy LoadGen detailed logging to stdout");
DEFINE_bool(disable_log_copy_summary_to_stdout, false, "Disable copy LoadGen summary logging to stdout");
DEFINE_string(log_mode, "AsyncPoll", "Logging mode for Loadgen");
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000, "Specify the poll interval for asynchrounous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

// QSL arguments
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(tensor_path, "", "Path to preprocessed samples in npy format (<full_image_name>.npy). Comma-separated list if there are more than one input.");
DEFINE_uint64(performance_sample_count, 0, "Number of samples to load in performance set.  0=use default");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {
    {"Offline", mlperf::TestScenario::Offline},
    {"SingleStream", mlperf::TestScenario::SingleStream},
    {"MultiStream", mlperf::TestScenario::MultiStream},
    {"Server", mlperf::TestScenario::Server}
};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {
    {"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly},
    {"PerformanceOnly", mlperf::TestMode::PerformanceOnly}
};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {
    {"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly},
    {"Synchronous", mlperf::LoggingMode::Synchronous}
};

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> res;
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        res.push_back(item);
    }
    return res;
}

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string gSampleName = "BERT_HARNESS";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    if (FLAGS_verbose) {
        setReportableSeverity(Severity::kVERBOSE);
    }
    gLogger.reportTestStart(sampleTest);
    if (FLAGS_load_plugins) {
        //TODO I can only prevent this from loading default plugins by commenting this out
        initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    }

    // Scope to force all smart objects destruction before CUDA context resets
    {
        // Configure the test settings
        mlperf::TestSettings testSettings;
        testSettings.scenario = scenarioMap[FLAGS_scenario];
        testSettings.mode = testModeMap[FLAGS_test_mode];
        testSettings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
        testSettings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);
        testSettings.single_stream_expected_latency_ns = FLAGS_single_stream_expected_latency_ns;
        testSettings.server_coalesce_queries = true;
        testSettings.server_num_issue_query_threads = FLAGS_server_num_issue_query_threads;

        // Configure the logging settings
        mlperf::LogSettings logSettings;
        logSettings.log_output.outdir = FLAGS_logfile_outdir;
        logSettings.log_output.prefix = FLAGS_logfile_prefix;
        logSettings.log_output.suffix = FLAGS_logfile_suffix;
        logSettings.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
        logSettings.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
        logSettings.log_output.copy_summary_to_stdout = !FLAGS_disable_log_copy_summary_to_stdout;
        logSettings.log_mode = logModeMap[FLAGS_log_mode];
        logSettings.log_mode_async_poll_interval_ms = FLAGS_log_mode_async_poll_interval_ms;
        logSettings.enable_trace = FLAGS_log_enable_trace;

        std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
        std::vector<bool> start_from_device(tensor_paths.size(), false);
        
        std::vector<int> gpus;
        if (FLAGS_devices == "all") {
            int numDevices = 0;
            cudaGetDeviceCount(&numDevices);
            LOG(INFO) << "Found " << numDevices << " GPUs";
            for (int i = 0; i < numDevices; i++) {
                gpus.emplace_back(i);
            }
        } else {
            LOG(INFO) << "Use GPUs: " << FLAGS_devices;
            auto deviceNames = split(FLAGS_devices, ',');
            for (auto &n : deviceNames) gpus.emplace_back(std::stoi(n));
        }

        auto qsl = std::make_shared<qsl::SampleLibrary>("BERT QSL", FLAGS_map_path, splitString(FLAGS_tensor_path, ","), FLAGS_performance_sample_count, 0, true, start_from_device);
        auto bert_server = std::make_shared<BERTServer>(
            "BERT SERVER",
            FLAGS_gpu_engines,
            qsl,
            gpus,
            FLAGS_gpu_batch_size,
            FLAGS_gpu_copy_streams,
            FLAGS_gpu_inference_streams,
            FLAGS_use_graphs,
            FLAGS_graphs_max_seqlen,
            FLAGS_graph_specs,
            FLAGS_soft_drop,
            testSettings.server_target_latency_percentile,
            testSettings.server_num_issue_query_threads
            );

        LOG(INFO) << "Starting running actual test.";

        cudaProfilerStart();
        StartTest(bert_server.get(), qsl.get(), testSettings, logSettings);
        cudaProfilerStop();

        // Print out equivalent QPS in multi_stream scenario.
        if (testSettings.scenario == mlperf::TestScenario::MultiStream)
        {
            std::cout << "Equivalent QPS computed by samples_per_query*target_qps : "
                << testSettings.multi_stream_samples_per_query * testSettings.multi_stream_target_qps << std::endl;
        }

        LOG(INFO) << "Finished running actual test.";
    }

    return 0;
}
