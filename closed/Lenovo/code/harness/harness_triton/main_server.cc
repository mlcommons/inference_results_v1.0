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

/* Include necessary header files */
// TRT
#include "NvInferPlugin.h"
#include "logger.h"

// TRITON
#include "triton_frontend.hpp"

// QSL
#include "callback.hpp"
#include "qsl.hpp"

// DLRM QSL
#include "dlrm_qsl.hpp"

// Google Logging
#include <glog/logging.h>

// LoadGen
#include "loadgen.h"

// General C++
#include <dlfcn.h>
#include <vector>

#include "cuda_profiler_api.h"

/* Define the appropriate flags */
// General flags
DEFINE_string(plugins, "", "Comma-separated list of shared objects for plugins");
DEFINE_bool(verbose, false, "Use verbose logging");
DEFINE_bool(use_graphs, false, "Enable cudaGraphs for TensorRT engines");
DEFINE_uint64(performance_sample_count, 0,
              "Number of samples to load in performance set. 0=use default");
DEFINE_double(warmup_duration, 5.0, "Minimum duration to run warmup for");
DEFINE_string(response_postprocess, "",
              "Enable imagenet post-processing on query sample responses.");
DEFINE_string(numa_config, "",
              "NUMA settings: each NUMA node contains a pair of GPU indices and CPU indices. Currently ignored.");

// TRITON flags
DEFINE_string(model_store_path, "", "Path to the engines directory for server scenario");
DEFINE_string(model_name, "", "Name of the model to use with TRITON");
DEFINE_uint32(model_version, 1, "Version of the model to use with TRITON");
DEFINE_uint32(buffer_manager_thread_count, 0, "The number of buffer manager thread");
DEFINE_bool(pinned_input, true, "Start inference assuming the data is in pinned memory");

// QSL flags
DEFINE_uint32(batch_size, 1, "Max Batch size to use for all devices and engines");
DEFINE_bool(batch_triton_requests, false, "Request batch size for each triton request");
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(tensor_path, "",
              "Path to preprocessed samples in npy format (<full_image_name>.npy)");
DEFINE_bool(coalesced_tensor, false,
            "Turn on if all the samples are coalesced into one single npy file");
DEFINE_bool(start_from_device, false, "Start inference assuming the data is already in device memory");
// Additional flags for DLRM QSL
DEFINE_string(sample_partition_path, "", "Path to sample partition file in npy format.");
DEFINE_bool(use_dlrm_qsl, false, "Use DLRM specific QSL");
DEFINE_uint32(min_sample_size, 100, "Minimum number of pairs a sample can contain.");
DEFINE_uint32(max_sample_size, 700, "Maximum number of pairs a sample can contain.");

// Loadgen test settings
DEFINE_string(scenario, "Server", "Scenario to run for Loadgen (Offline, Server)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "resnet50", "Model name");
DEFINE_string(mlperf_conf_path, "", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "", "Path to user.conf");
DEFINE_uint64(single_stream_expected_latency_ns, 100000, "Inverse of desired target QPS");

// Loadgen logging settings
DEFINE_string(logfile_outdir, "", "Specify the existing output directory for the LoadGen logs");
DEFINE_string(logfile_prefix, "", "Specify the filename prefix for the LoadGen log files");
DEFINE_string(logfile_suffix, "", "Specify the filename suffix for the LoadGen log files");
DEFINE_bool(logfile_prefix_with_datetime, false, "Prefix filenames for LoadGen log files");
DEFINE_bool(log_copy_detail_to_stdout, false, "Copy LoadGen detailed logging to stdout");
DEFINE_bool(disable_log_copy_summary_to_stdout, false,
            "Disable copy LoadGen summary logging to stdout");
DEFINE_string(log_mode, "AsyncPoll", "Logging mode for Loadgen");
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000,
              "Specify the poll interval for asynchrounous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

// FIXME: this is enabled for MIG, but not used for SingleMIG or non-MIG
DEFINE_uint64(deque_timeout_us, 10000, "Timeout for deque from work queue");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {
    {"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly},
    {"PerformanceOnly", mlperf::TestMode::PerformanceOnly},
    {"FindPeakPerformance", mlperf::TestMode::FindPeakPerformance}};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {
    {"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly},
    {"Synchronous", mlperf::LoggingMode::Synchronous}};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {
    {"Offline", mlperf::TestScenario::Offline},
    {"SingleStream", mlperf::TestScenario::SingleStream},
    {"MultiStream", mlperf::TestScenario::MultiStream},
    {"Server", mlperf::TestScenario::Server}};

/* Helper function to split a string based on a delimiting character */
std::vector<std::string> splitString(const std::string& input, const std::string& delimiter)
{
    std::vector<std::string> result;
    size_t start = 0;
    size_t next = 0;
    while(next != std::string::npos)
    {
        next = input.find(delimiter, start);
        result.emplace_back(input, start, next - start);
        start = next + 1;
    }
    return result;
}

void doInference()
{
    // Configure the test settings
    mlperf::TestSettings test_settings;
    test_settings.scenario = scenarioMap[FLAGS_scenario];
    test_settings.mode = testModeMap[FLAGS_test_mode];
    test_settings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
    test_settings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);
    test_settings.single_stream_expected_latency_ns = FLAGS_single_stream_expected_latency_ns;
    test_settings.server_coalesce_queries = true;

    // Configure the logging settings
    mlperf::LogSettings log_settings;
    log_settings.log_output.outdir = FLAGS_logfile_outdir;
    log_settings.log_output.prefix = FLAGS_logfile_prefix;
    log_settings.log_output.suffix = FLAGS_logfile_suffix;
    log_settings.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
    log_settings.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
    log_settings.log_output.copy_summary_to_stdout = !FLAGS_disable_log_copy_summary_to_stdout;
    log_settings.log_mode = logModeMap[FLAGS_log_mode];
    log_settings.log_mode_async_poll_interval_ms = FLAGS_log_mode_async_poll_interval_ms;
    log_settings.enable_trace = FLAGS_log_enable_trace;

    // Instantiate our SUT and get the status of the server and model
    triton_frontend::ServerSUTPtr_t sut = std::make_shared<triton_frontend::Server_SUT>(
        "Triton_Server", FLAGS_model_store_path, FLAGS_model_name, FLAGS_model_version,
        FLAGS_use_dlrm_qsl, FLAGS_start_from_device, FLAGS_pinned_input);

    if(FLAGS_use_dlrm_qsl)
    {
        sut->Init(FLAGS_min_sample_size, FLAGS_max_sample_size, FLAGS_buffer_manager_thread_count, FLAGS_batch_triton_requests);
    }
    else
    {
        sut->Init(1,1,FLAGS_buffer_manager_thread_count,FLAGS_batch_triton_requests);
    }
    sut->ModelMetadata();
    sut->SetResponseCallback(callbackMap[FLAGS_response_postprocess]); // Set QuerySampleResponse
                                                                       // post-processing callback

    // Instantiate our QSL
    qsl::SampleLibraryPtr_t lib;
    std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
    std::vector<bool> start_from_device(tensor_paths.size(), FLAGS_start_from_device);

    if(FLAGS_use_dlrm_qsl)
    {
        // Load the sample distribution. We do this here to calculate the performance sample count
        // of the underlying LWIS QSL, since the super constructor must be in the constructor
        // initialization list.
        std::vector<int> originalPartition;

        // Scope to automatically close the file
        {
            npy::NpyFile samplePartitionFile(FLAGS_sample_partition_path);
            CHECK_EQ(samplePartitionFile.getDims().size(), 1);

            // For now we do not allow numPartitions == FLAGS_performance_sample_count, since the
            // TotalSampleCount is determined at runtime in the underlying LWIS QSL.
            size_t numPartitions = samplePartitionFile.getDims()[0];
            CHECK_EQ(numPartitions > FLAGS_performance_sample_count, true);

            std::vector<char> tmp(samplePartitionFile.getTensorSize());
            samplePartitionFile.loadAll(tmp);

            originalPartition.resize(numPartitions);
            memcpy(originalPartition.data(), tmp.data(), tmp.size());
            LOG(INFO) << "Loaded " << originalPartition.size() - 1 << " sample partitions. ("
                      << tmp.size() << ") bytes.";
        }

        // Force underlying QSL to load all samples, since we want to be able to grab any partition
        // given the sample index.
        size_t perfPairCount = originalPartition.back();

        lib.reset(new DLRMSampleLibrary("Triton_DLRMSampleLibrary", FLAGS_map_path,
                                        splitString(FLAGS_tensor_path, ","), originalPartition,
                                        FLAGS_performance_sample_count, perfPairCount,
                                        0 /* padding */, FLAGS_coalesced_tensor, FLAGS_start_from_device));
        sut->AddSampleLibrary(lib);
    }
    else
    {
        size_t padding = (test_settings.scenario == mlperf::TestScenario::MultiStream && test_settings.multi_stream_samples_per_query) ? (test_settings.multi_stream_samples_per_query - 1) : 0;
        lib = std::make_shared<qsl::SampleLibrary>("Triton_SampleLibrary", FLAGS_map_path,
                                                   splitString(FLAGS_tensor_path, ","),
                                                   FLAGS_performance_sample_count, padding,
                                                   FLAGS_coalesced_tensor, start_from_device);
        sut->AddSampleLibrary(lib);
    }

    // Warmup our SUT
    double expected_qps{1.0};
    switch(test_settings.scenario)
    {
        case mlperf::TestScenario::Offline:
            expected_qps = test_settings.offline_expected_qps;
            break;
        case mlperf::TestScenario::Server:
            expected_qps = test_settings.server_target_qps;
            break;
        case mlperf::TestScenario::MultiStream:
            expected_qps = test_settings.multi_stream_target_qps;
            break;
        case mlperf::TestScenario::SingleStream:
            expected_qps = 1.0e9 / test_settings.single_stream_expected_latency_ns;
            break;
    }
    sut->Warmup(FLAGS_warmup_duration, expected_qps);

    LOG(INFO) << "Starting actual test" << std::endl;
    
    // Perform the inference testing
    cudaProfilerStart();
    mlperf::StartTest(sut.get(), lib.get(), test_settings, log_settings);
    cudaProfilerStop();

    LOG(INFO) << "Finished actual test" << std::endl;


    // Check SUT end status and inform the SUT that we are done
    sut->ModelStats();
    sut->Done();

    // Print out equivalent QPS in multi_stream scenario.
    if (test_settings.scenario == mlperf::TestScenario::MultiStream)
    {
        std::cout << "Equivalent QPS computed by samples_per_query*target_qps : "
            << test_settings.multi_stream_samples_per_query * test_settings.multi_stream_target_qps << std::endl;
    }

    // Reset the SUT and QSL pointers
    sut.reset();
    lib.reset();
}

int main(int argc, char** argv)
{
    // Initialize the NVInfer plugins
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    // Parse command line flags
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    // Load all the needed shared objects for plugins.
    std::vector<std::string> plugin_files = splitString(FLAGS_plugins, ",");
    for(auto& s : plugin_files)
    {
        void* dlh = dlopen(s.c_str(), RTLD_LAZY);
        if(nullptr == dlh)
        {
            std::cout << "Error loading plugin library " << s << std::endl;
            return 1;
        }
    }

    // Perform Inference
    doInference();

    /* Return pass */
    return 0;
}
