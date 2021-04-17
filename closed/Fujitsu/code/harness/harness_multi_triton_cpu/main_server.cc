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
// TRITON
#include "triton_frontend.hpp"

// QSL
#include "callback.hpp"
// DLRM QSL
#include "dlrm_qsl_cpu.hpp"

// Google Logging
#include <glog/logging.h>

// LoadGen
#include "loadgen.h"

// General C++
#include <dlfcn.h>
#include <vector>

#include "cuda_profiler_api.h"

// fork & execve & wait
#include <sys/wait.h>
#include <unistd.h>

// short namespace for boost::interprocess
namespace bip = boost::interprocess;

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
              "NUMA settings: each NUMA node contains a pair of GPU indices and CPU indices. "
              "Currently ignored.");

// TRITON flags
DEFINE_string(model_store_path, "", "Path to the engines directory for server scenario");
DEFINE_string(model_name, "", "Name of the model to use with TRITON");
DEFINE_uint32(model_version, 1, "Version of the model to use with TRITON");
DEFINE_uint32(buffer_manager_thread_count, 0, "The number of buffer manager thread");
DEFINE_bool(pinned_input, true, "Start inference assuming the data is in pinned memory");

// QSL flags
DEFINE_uint32(batch_size, 1, "Max Batch size to use for all devices and engines");
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(tensor_path, "",
              "Path to preprocessed samples in npy format (<full_image_name>.npy)");
DEFINE_bool(coalesced_tensor, false,
            "Turn on if all the samples are coalesced into one single npy file");
DEFINE_bool(start_from_device, false,
            "Start inference assuming the data is already in device memory");

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

// Multi-triton mode settings
DEFINE_string(instance_num, "N/A",
              "Not intended to be set by user; Internally "
              "used to fork out servers on MIG instances");
DEFINE_uint32(num_instances, 1, "Number of Triton instances to start");

// borrowed from LWIS
DEFINE_uint64(deque_timeout_us, 10000, "Timeout for deque from work queue");

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

/* Define a map to convert test mode input string into its corresponding enum
 * value */
std::map<std::string, mlperf::TestMode> testModeMap = {
    {"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly},
    {"PerformanceOnly", mlperf::TestMode::PerformanceOnly},
    {"FindPeakPerformance", mlperf::TestMode::FindPeakPerformance}};

/* Define a map to convert logging mode input string into its corresponding enum
 * value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {
    {"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly},
    {"Synchronous", mlperf::LoggingMode::Synchronous}};

/* Define a map to convert test mode input string into its corresponding enum
 * value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {
    {"Offline", mlperf::TestScenario::Offline},
    {"SingleStream", mlperf::TestScenario::SingleStream},
    {"MultiStream", mlperf::TestScenario::MultiStream},
    {"Server", mlperf::TestScenario::Server}};

qsl::MIGSampleLibraryPtr_t createMIGSampleLibrary(mlperf::TestSettings test_settings,
                                                  mlperf::LogSettings log_settings)
{
    qsl::MIGSampleLibraryPtr_t lib;
    std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
    std::vector<bool> start_from_device(tensor_paths.size(), FLAGS_start_from_device);
    size_t padding = (test_settings.scenario == mlperf::TestScenario::MultiStream &&
                      test_settings.multi_stream_samples_per_query)
                         ? (test_settings.multi_stream_samples_per_query - 1)
                         : 0;
    lib = std::make_shared<qsl::MIGSampleLibrary>(
        "Triton_MIGSampleLibrary", FLAGS_map_path, splitString(FLAGS_tensor_path, ","),
        FLAGS_performance_sample_count, padding, FLAGS_coalesced_tensor, start_from_device);

    return lib;
}

qsl::SampleLibraryPtr_t createSampleLibrary(mlperf::TestSettings test_settings,
                                            mlperf::LogSettings log_settings)
{
    LOG(INFO) << "Creating Primitive Sample Library";

    // Instantiate our QSL
    qsl::SampleLibraryPtr_t lib;
    std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
    std::vector<bool> start_from_device(tensor_paths.size(), FLAGS_start_from_device);

    if(FLAGS_use_dlrm_qsl)
    {
        // Load the sample distribution. We do this here to calculate the
        // performance sample count
        // of the underlying LWIS QSL, since the super constructor must be in the
        // constructor
        // initialization list.
        std::vector<int> originalPartition;

        // Scope to automatically close the file
        {
            npy::NpyFile samplePartitionFile(FLAGS_sample_partition_path);
            CHECK_EQ(samplePartitionFile.getDims().size(), 1);

            // For now we do not allow numPartitions ==
            // FLAGS_performance_sample_count, since the
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

        // Force underlying QSL to load all samples, since we want to be able to
        // grab any partition
        // given the sample index.
        size_t perfPairCount = originalPartition.back();

        lib.reset(new DLRMSampleLibrary(
            "Triton_DLRMSampleLibrary", FLAGS_map_path, splitString(FLAGS_tensor_path, ","),
            originalPartition, FLAGS_performance_sample_count, perfPairCount, 0 /* padding */,
            FLAGS_coalesced_tensor, FLAGS_start_from_device));
    }
    else
    {
        size_t padding = (test_settings.scenario == mlperf::TestScenario::MultiStream &&
                          test_settings.multi_stream_samples_per_query)
                             ? (test_settings.multi_stream_samples_per_query - 1)
                             : 0;
        lib = std::make_shared<qsl::SampleLibrary>(
            "Triton_SampleLibrary", FLAGS_map_path, splitString(FLAGS_tensor_path, ","),
            FLAGS_performance_sample_count, padding, FLAGS_coalesced_tensor, start_from_device);
    }

    return lib;
}

void runServer(mlperf::TestSettings& test_settings, mlperf::LogSettings& log_settings,
               triton_frontend::ServerSUTPtr_t& sut, qsl::SampleLibraryPtr_t& lib,
               double expected_qps,
               triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> c2s_state_mq,
               triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> s2c_state_mq,
               triton_frontend::sharedMQ_ptr<mlperf::QuerySampleResponse> s2c_resp_mq,
               triton_frontend::sharedMSM_ptr c2s_shmem, triton_frontend::sharedMR_ptr s2c_shmem)
{
    LOG(INFO) << "Running Server on " << FLAGS_instance_num;

    // Instantiate our SUT and get the status of the server and model
    sut = std::make_shared<triton_frontend::Server_SUT>(
        "Triton_Server", FLAGS_model_store_path, FLAGS_model_name, FLAGS_model_version,
        FLAGS_use_dlrm_qsl, FLAGS_start_from_device, FLAGS_pinned_input, FLAGS_instance_num,
        c2s_state_mq, s2c_state_mq, s2c_resp_mq, c2s_shmem, s2c_shmem);
    if(FLAGS_use_dlrm_qsl)
    {
        sut->Init(FLAGS_min_sample_size, FLAGS_max_sample_size, FLAGS_buffer_manager_thread_count);
    }
    else
    {
        sut->Init();
    }
    sut->ModelMetadata();
    sut->SetResponseCallback(callbackMap[FLAGS_response_postprocess]); // Set QuerySampleResponse
                                                                       // post-processing callback
    // hook lib
    sut->AddSampleLibrary(lib);

    triton_frontend::IPCcomm m_;

    // notify client for readiness
    m_ = triton_frontend::IPCcomm::Initialized;
    s2c_state_mq->send_it(m_);

    // Warmup our SUT; at the end of Warmup, it signals SUTshim for being ready
    c2s_state_mq->receive_it(m_);
    assert(m_ == triton_frontend::IPCcomm::StartWarmup);
    sut->Warmup(FLAGS_warmup_duration, expected_qps);
    // Send WarmupDone
    m_ = triton_frontend::IPCcomm::WarmupDone;
    s2c_state_mq->send_it(m_);

    // do work like LoadSamples, IssueQuery, UnloadSamples
    sut->Worker();

    // Check SUT end status and inform the SUT that we are done
    sut->ModelStats();
    sut->Done();

    sut.reset();
    lib.reset();
}

// start thread this right before LoadGen StartTest
void ClientResponseComplete(
    std::vector<std::string>& num_instances, triton_frontend::SUTShim_ptr& sut,
    triton_frontend::sharedMap_ptr<triton_frontend::IPCcomm> c2s_state_mq_map,
    triton_frontend::sharedMap_ptr<mlperf::QuerySampleResponse> s2c_resp_mq_map,
    triton_frontend::sharedMRMap_ptr s2c_shmem_map, triton_frontend::sharedBAMap_ptr s2c_baddr_map)
{
    while(!sut->is_finished())
    {
        // Get resp from a child server and notify LoadGen
        for(auto& instance : num_instances)
        {
            try
            {
                // recover response from s2c_shmem_map and send back to LoadGen
                mlperf::QuerySampleResponse loadgen_response_ipc;
                bool rcvd = s2c_resp_mq_map[instance]->receive(loadgen_response_ipc);
                if(rcvd)
                {
                    // handle offset manually on shared memory
                    uintptr_t base_addr = s2c_baddr_map[instance];
                    mlperf::QuerySampleResponse loadgen_response{
                        loadgen_response_ipc.id, loadgen_response_ipc.data + base_addr,
                        loadgen_response_ipc.size};
                    // We always send one inference response at a time
                    mlperf::QuerySamplesComplete(&loadgen_response, 1);
                }
            }
            catch(bip::interprocess_exception& ex)
            {
                std::cerr << "Met exception while waiting Init msg from " << instance << " : "
                          << ex.what() << std::endl;
                assert(false);
            }
        }
    }
}

void runClient(mlperf::TestSettings& test_settings, mlperf::LogSettings& log_settings,
               qsl::MIGSampleLibraryPtr_t& lib, double expected_qps,
               std::vector<std::string>& num_instances,
               triton_frontend::sharedMap_ptr<triton_frontend::IPCcomm> c2s_state_mq_map,
               triton_frontend::sharedMap_ptr<triton_frontend::IPCcomm> s2c_state_mq_map,
               triton_frontend::sharedMap_ptr<mlperf::QuerySampleResponse> s2c_resp_mq_map,
               triton_frontend::sharedMSMMap_ptr c2s_shmem_map,
               triton_frontend::sharedMRMap_ptr s2c_shmem_map,
               triton_frontend::sharedBAMap_ptr s2c_baddr_map)
{
    LOG(INFO) << "Running Client";

    // Create SUTshim that calls StartTest on LoadGen
    // This SUTshim also distributes reqs to servers
    std::string scenario{FLAGS_scenario};
    boost::algorithm::to_lower(scenario);
    auto sutshim = std::make_shared<triton_frontend::SUTShim>(
        "Triton_MultiMigServer", scenario, FLAGS_batch_size, FLAGS_deque_timeout_us, num_instances,
        c2s_state_mq_map, s2c_state_mq_map, s2c_resp_mq_map, c2s_shmem_map, s2c_shmem_map);

    // Init, setup interprocess communication etc
    sutshim->Init();

    // register callback
    lib->registerLoadCB([&sutshim](std::vector<mlperf::QuerySampleIndex> s) {
        return sutshim->Do_LoadSamplesToRam(s);
    });
    lib->registerUnloadCB([&sutshim](std::vector<mlperf::QuerySampleIndex> s) {
        return sutshim->Do_UnloadSamplesFromRam(s);
    });

    // wait for all the clients being ready
    for(auto& instance : num_instances)
    {
        try
        {
            triton_frontend::IPCcomm m_;
            s2c_state_mq_map[instance]->receive_it(m_);
            assert(m_ == triton_frontend::IPCcomm::Initialized);
        }
        catch(bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while waiting Init msg from " << instance << " : "
                      << ex.what() << std::endl;
            assert(false);
        }
    }

    // Initiate warmups in servers
    for(auto& instance : num_instances)
    {
        try
        {
            // sending warmup signal
            triton_frontend::IPCcomm m_ = triton_frontend::IPCcomm::StartWarmup;
            c2s_state_mq_map[instance]->send_it(m_);
        }
        catch(bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while sending StartWarmup msg to " << instance << " : "
                      << ex.what() << std::endl;
            assert(false);
        }
    }

    // wait for all the clients done with Warmup
    for(auto& instance : num_instances)
    {
        try
        {
            triton_frontend::IPCcomm m_;
            s2c_state_mq_map[instance]->receive_it(m_);
            assert(m_ == triton_frontend::IPCcomm::WarmupDone);
        }
        catch(bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while waiting WarmupDone msg from " << instance << " : "
                      << ex.what() << std::endl;
            assert(false);
        }
    }

    // how many RespWorker threads to start
    const int num_migs = num_instances.size();
    const int max_num_worker_handle = std::min(1, num_migs);
    int num_workers = ceil(float(num_migs) / float(max_num_worker_handle));
    std::thread workerThreads[num_workers];
    std::vector<std::string> mig_uuid_chops[num_workers];
    // map RespWorkers to MIG instances, in an interleaved way
    for(int i_ = 0; i_ < num_workers; i_++)
    {
        std::vector<std::string> s_;
        s_.clear();
        // clang-format off
        boost::copy(num_instances 
                    | boost::adaptors::sliced(i_, num_migs) 
                    | boost::adaptors::strided(num_workers),
                    std::back_inserter(s_));
        // clang-format on
        mig_uuid_chops[i_] = s_;
    }

    // thread for collecting responses
    for(int i_ = 0; i_ < num_workers; i_++)
    {
        workerThreads[i_] =
            std::thread(ClientResponseComplete, std::ref(mig_uuid_chops[i_]), std::ref(sutshim),
                        std::ref(c2s_state_mq_map), std::ref(s2c_resp_mq_map),
                        std::ref(s2c_shmem_map), std::ref(s2c_baddr_map));
    }

    // Start with SUTshim and launch
    mlperf::StartTest(sutshim.get(), lib.get(), test_settings, log_settings);

    LOG(INFO) << "Test is completed by LoadGen";

    // done
    sutshim->Done();

    // Join threads
    for(int i_ = 0; i_ < num_workers; i_++)
    {
        workerThreads[i_].join();
    }

    // send terminate msg to all Servers
    for(auto& instance : num_instances)
    {
        try
        {
            triton_frontend::IPCcomm m_ = triton_frontend::IPCcomm::Terminate;
            c2s_state_mq_map[instance]->send_it(m_);
        }
        catch(bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while sending Terminate msg to " << instance << " : "
                      << ex.what() << std::endl;
            assert(false);
        }
    }

    // Print out equivalent QPS in multi_stream scenario.
    if(test_settings.scenario == mlperf::TestScenario::MultiStream)
    {
        std::cout << "Equivalent QPS computed by samples_per_query*target_qps : "
                  << test_settings.multi_stream_samples_per_query *
                         test_settings.multi_stream_target_qps
                  << std::endl;
    }

    lib.reset();

    // destroy IPC objects
    for(auto& instance : num_instances)
    {
        try
        {
            std::string qstr;
            qstr = boost::replace_all_copy<std::string>(instance + "_C2S", "/", "_");
            bip::message_queue::remove(qstr.c_str());
            qstr = boost::replace_all_copy<std::string>(instance + "_S2C", "/", "_");
            bip::message_queue::remove(qstr.c_str());
            qstr = boost::replace_all_copy<std::string>(instance + "_S2C_RSP", "/", "_");
            bip::message_queue::remove(qstr.c_str());
            qstr = boost::replace_all_copy<std::string>(instance + "_C2S_SHM", "/", "_");
            bip::shared_memory_object::remove(qstr.c_str());
            qstr = boost::replace_all_copy<std::string>(instance + "_S2C_SHM", "/", "_");
            bip::shared_memory_object::remove(qstr.c_str());
        }
        catch(bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while destroying IPC objects related to " << instance
                      << " : " << ex.what() << std::endl;
            assert(false);
        }
    }
}

mlperf::TestSettings get_test_settings()
{
    // Configure the test settings
    mlperf::TestSettings test_settings;
    test_settings.scenario = scenarioMap[FLAGS_scenario];
    test_settings.mode = testModeMap[FLAGS_test_mode];
    test_settings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
    test_settings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);
    test_settings.single_stream_expected_latency_ns = FLAGS_single_stream_expected_latency_ns;
    test_settings.server_coalesce_queries = true;

    return test_settings;
}

mlperf::LogSettings get_log_settings()
{
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

    return log_settings;
}

std::vector<std::string> getInstances(int num)
{
    // TODO: Generalize using numactl to query hardware information
    // 2 nodes on computelab-403 system. Test with 16 triton instances running on 4 cores each.
    // node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 56 57
    // 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 node 1 cpus: 28
    // 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 84 85 86 87
    // 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111
    std::vector<std::string> numInstances;
    for(int i = 0; i < num; i++)
    {
        numInstances.push_back(std::to_string(i));
    }
    return numInstances;
}
int main(int argc, char** argv)
{
    // store original args before GFlags destroys them
    std::vector<const char*> org_args(argv, argv + argc);

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

    // set up some settings
    mlperf::TestSettings test_settings = get_test_settings();
    mlperf::LogSettings log_settings = get_log_settings();

    // initiate SampleLibrary
    qsl::SampleLibraryPtr_t lib = createSampleLibrary(test_settings, log_settings);

    // expected QPS
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

    // main process
    if(FLAGS_instance_num == "N/A")
    {
        // Get number of server instances to launch
        std::vector<std::string> instance_ids = getInstances(FLAGS_num_instances);
        const int numInstances = instance_ids.size();

        // fork Servers as many as instances
        for(auto& instance : instance_ids)
        {
            int c_pid = fork();

            // c_pid == 0 means this is forked child process
            if(c_pid == 0)
            {
                // Push back this child's instance number to actually launch the server
                std::vector<const char*> args(org_args);
                std::string instance_arg = "--instance_num=";
                instance_arg.append(instance);
                args.push_back(const_cast<char*>(instance_arg.data()));
                args.push_back(nullptr);

                // Get NUMA information.
                int numNumaNodes = numa_num_configured_nodes();
                const std::string numactl = "numactl";
                const int instanceNum = stoi(instance);
                const int half = numInstances / 2;

                // TODO: Programitcally figure out number of cores?
                const int numCores = 28;
                const int coresPerInstance = numCores / half;

                // Run first half number of instances on numa 0, second half on numa 1
                std::string numaNode = instanceNum < half ? "0" : "1";

                std::string start = std::to_string(stoi(instance) * coresPerInstance);
                std::string end =
                    std::to_string(stoi(instance) * coresPerInstance + coresPerInstance - 1);
                std::string CPUS = start + "-" + end;

                std::vector<const char*> fullArgs = {numactl.c_str(),  "-C", CPUS.c_str(),    "-N",
                                                     numaNode.c_str(), "-m", numaNode.c_str()};
                fullArgs.insert(fullArgs.end(), args.begin(), args.end());

                LOG(INFO) << "Launching Inference process on " << instance << " with command:";
                std::string cmd;
                cmd = numactl;
                for(auto arg : fullArgs)
                {
                    if(arg)
                    {
                        cmd = cmd + arg + " ";
                    }
                }
                cmd + "\n";

                LOG(INFO) << cmd;

                execvpe(numactl.c_str(), const_cast<char* const*>(fullArgs.data()), environ);
            }
        }

        // create sample libarary for callbacks
        qsl::MIGSampleLibraryPtr_t mig_lib = createMIGSampleLibrary(test_settings, log_settings);

        // Build shared message queue & shared memory segments
        triton_frontend::sharedMap_ptr<triton_frontend::IPCcomm> c2s_state_mq_map;
        triton_frontend::sharedMap_ptr<triton_frontend::IPCcomm> s2c_state_mq_map;
        triton_frontend::sharedMap_ptr<mlperf::QuerySampleResponse> s2c_resp_mq_map;
        triton_frontend::sharedMSMMap_ptr c2s_shmem_map;
        triton_frontend::sharedMRMap_ptr s2c_shmem_map;
        triton_frontend::sharedBAMap_ptr s2c_baddr_map;

        // TODO: Update sizes for CPU?
        size_t comm_mq_size{1};
        size_t resp_mq_size{1};
        if(FLAGS_scenario == "Offline")
        {
            comm_mq_size = 100;
            resp_mq_size = 100;
        }
        else if(FLAGS_scenario == "Server")
        {
            comm_mq_size = 10;
            resp_mq_size = 10;
        }

        // Create shared message queue, as many as MIGs
        for(auto& instance : instance_ids)
        {
            try
            {
                // Client to Server State MQ
                std::string c2s_qstr =
                    boost::replace_all_copy<std::string>(instance + "_C2S", "/", "_");
                triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> c2s_mq =
                    std::make_shared<triton_frontend::msg_q<triton_frontend::IPCcomm>>(
                        c2s_qstr.c_str(), comm_mq_size);
                c2s_state_mq_map.emplace(instance, c2s_mq);

                // Server to Client State MQ
                std::string s2c_qstr =
                    boost::replace_all_copy<std::string>(instance + "_S2C", "/", "_");
                triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> s2c_mq =
                    std::make_shared<triton_frontend::msg_q<triton_frontend::IPCcomm>>(
                        s2c_qstr.c_str(), comm_mq_size);
                s2c_state_mq_map.emplace(instance, s2c_mq);

                // Server to Client Resp MQ
                std::string rsp_qstr =
                    boost::replace_all_copy<std::string>(instance + "_S2C_RSP", "/", "_");
                triton_frontend::sharedMQ_ptr<mlperf::QuerySampleResponse> rsp_mq =
                    std::make_shared<triton_frontend::msg_q<mlperf::QuerySampleResponse>>(
                        rsp_qstr.c_str(), resp_mq_size);
                s2c_resp_mq_map.emplace(instance, rsp_mq);
            }
            catch(bip::interprocess_exception& ex)
            {
                std::cerr << "Met exception while creating msg_q towards " << instance << " : "
                          << ex.what() << std::endl;
                assert(false);
            }

            // Create shared memory segments, as many as instances
            try
            {
                // TODO: Update sizes for CPU?
                const size_t max_in_size = 6000000 * sizeof(mlperf::QuerySample);
                const size_t max_out_size = 1 << 29;

                // Client to Server shmem, used to toss samples
                std::string c2s_qstr =
                    boost::replace_all_copy<std::string>(instance + "_C2S_SHM", "/", "_");
                bip::shared_memory_object::remove(c2s_qstr.c_str());
                triton_frontend::sharedMSM_ptr c2s_shmem;
                c2s_shmem = std::make_shared<bip::managed_shared_memory>(
                    bip::create_only, c2s_qstr.c_str(), max_in_size);
                c2s_shmem_map.emplace(instance, c2s_shmem);

                // Server to Client shmem, used to mmap output tensors
                std::string s2c_qstr =
                    boost::replace_all_copy<std::string>(instance + "_S2C_SHM", "/", "_");
                bip::shared_memory_object::remove(s2c_qstr.c_str());
                bip::shared_memory_object s2c_shm(bip::create_only, s2c_qstr.c_str(),
                                                  bip::read_write);
                s2c_shm.truncate(max_out_size);
                triton_frontend::sharedMR_ptr s2c_shmem;
                s2c_shmem = std::make_shared<bip::mapped_region>(s2c_shm, bip::read_write);
                s2c_shmem_map.emplace(instance, s2c_shmem);
                s2c_baddr_map.emplace(instance,
                                      reinterpret_cast<uintptr_t>(s2c_shmem->get_address()));
            }
            catch(bip::interprocess_exception& ex)
            {
                std::cerr << "Met exception while creqting shared memory towards " << instance
                          << " : " << ex.what() << std::endl;
                assert(false);
            }
        }

        // Run Client
        runClient(test_settings, log_settings, mig_lib, expected_qps, instance_ids,
                  c2s_state_mq_map, s2c_state_mq_map, s2c_resp_mq_map, c2s_shmem_map, s2c_shmem_map,
                  s2c_baddr_map);
    }

    // Server process in MIG instances
    else
    {
        LOG(INFO) << "Performing inference on server instance: " << FLAGS_instance_num;

        // open shared memory and message queues
        std::string c2s_qstr =
            boost::replace_all_copy<std::string>(FLAGS_instance_num + "_C2S", "/", "_");
        triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> c2s_state_mq =
            std::make_shared<triton_frontend::msg_q<triton_frontend::IPCcomm>>(c2s_qstr.c_str());

        std::string s2c_qstr =
            boost::replace_all_copy<std::string>(FLAGS_instance_num + "_S2C", "/", "_");
        triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> s2c_state_mq =
            std::make_shared<triton_frontend::msg_q<triton_frontend::IPCcomm>>(s2c_qstr.c_str());

        std::string rsp_qstr =
            boost::replace_all_copy<std::string>(FLAGS_instance_num + "_S2C_RSP", "/", "_");
        triton_frontend::sharedMQ_ptr<mlperf::QuerySampleResponse> s2c_resp_mq =
            std::make_shared<triton_frontend::msg_q<mlperf::QuerySampleResponse>>(rsp_qstr.c_str());

        std::string c2s_shm_qstr =
            boost::replace_all_copy<std::string>(FLAGS_instance_num + "_C2S_SHM", "/", "_");
        triton_frontend::sharedMSM_ptr c2s_shmem =
            std::make_shared<bip::managed_shared_memory>(bip::open_only, c2s_shm_qstr.c_str());

        std::string s2c_shm_qstr =
            boost::replace_all_copy<std::string>(FLAGS_instance_num + "_S2C_SHM", "/", "_");
        bip::shared_memory_object s2c_shm(bip::open_only, s2c_shm_qstr.c_str(), bip::read_write);
        triton_frontend::sharedMR_ptr s2c_shmem =
            std::make_shared<bip::mapped_region>(s2c_shm, bip::read_write);

        // Run Server
        triton_frontend::ServerSUTPtr_t sut;

        runServer(test_settings, log_settings, sut, lib, expected_qps, c2s_state_mq, s2c_state_mq,
                  s2c_resp_mq, c2s_shmem, s2c_shmem);
    }

    /* Return pass */
    return 0;
}
