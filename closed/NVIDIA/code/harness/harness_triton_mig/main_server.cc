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
DEFINE_uint64(performance_sample_count, 0, "Number of samples to load in performance set. 0=use default");
DEFINE_double(warmup_duration, 5.0, "Minimum duration to run warmup for");
DEFINE_string(response_postprocess, "", "Enable imagenet post-processing on query sample responses.");
DEFINE_string(numa_config, "",
              "NUMA settings: each NUMA node contains a pair of GPU indices and CPU indices. Currently ignored.");

// TRITON flags
DEFINE_string(model_store_path, "", "Path to the engines directory for server scenario");
DEFINE_string(model_name, "", "Name of the model to use with TRITON");
DEFINE_uint32(model_version, 1, "Version of the model to use with TRITON");
DEFINE_uint32(buffer_manager_thread_count, 0, "The number of buffer manager thread");
DEFINE_bool(pinned_input, true, "Start inference assuming the data is in pinned memory");
DEFINE_bool(batch_triton_requests, false, "Batch-requests before they are sent to Triton; Unused in Multi-MIG harness");

// QSL flags
DEFINE_uint32(batch_size, 1, "Max Batch size to use for all devices and engines");
DEFINE_bool(check_contiguity, false, "Check if requests in a single IssueQuery are contiguous");
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(tensor_path, "", "Path to preprocessed samples in npy format (<full_image_name>.npy)");
DEFINE_bool(coalesced_tensor, false, "Turn on if all the samples are coalesced into one single npy file");
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
DEFINE_bool(disable_log_copy_summary_to_stdout, false, "Disable copy LoadGen summary logging to stdout");
DEFINE_string(log_mode, "AsyncPoll", "Logging mode for Loadgen");
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000, "Specify the poll interval for asynchrounous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

// Harness Triton MIG mode settings
DEFINE_string(mig_uuid, "N/A",
              "Not intended to be set by user; Internally "
              "used to fork out servers on MIG instances");
DEFINE_uint32(num_migs, 1,
              "Not intended to be set by user; Internally "
              "used to weight QPS on each MIG instance");
DEFINE_uint32(target_numa, 0,
              "Not intended to be set by user; Internally "
              "used to set NUMA affinity on MIG instances");
DEFINE_uint32(num_numa, 0,
              "Not intended to be set by user; Internally "
              "used to set NUMA affinity on MIG instances");

// borrowed from LWIS
DEFINE_uint64(deque_timeout_usec, 10000, "Timeout for deque from work queue");

// to prevent hang; child process may assert/abort
pid_t parent_pid;
void sigabort_handler(int sig)
{
    assert(sig == SIGABRT);
    pid_t self = getpid();
    std::cerr << "Abort happened from PID: " << self << std::endl;
    // just terminate from parent, all group PIDs
    std::cerr << "Terminating from parent PID: " << parent_pid << std::endl;
    killpg(getpgid(parent_pid), SIGTERM);
}

/* Define a map to convert test mode input string into its corresponding enum
 * value */
std::map<std::string, mlperf::TestMode> testModeMap = {{"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly}, {"PerformanceOnly", mlperf::TestMode::PerformanceOnly},
    {"FindPeakPerformance", mlperf::TestMode::FindPeakPerformance}};

/* Define a map to convert logging mode input string into its corresponding enum
 * value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {{"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly}, {"Synchronous", mlperf::LoggingMode::Synchronous}};

/* Define a map to convert test mode input string into its corresponding enum
 * value */
std::map<std::string, mlperf::TestScenario> scenarioMap
    = {{"Offline", mlperf::TestScenario::Offline}, {"SingleStream", mlperf::TestScenario::SingleStream},
        {"MultiStream", mlperf::TestScenario::MultiStream}, {"Server", mlperf::TestScenario::Server}};

qsl::MIGSampleLibraryPtr_t createMIGSampleLibrary(mlperf::TestSettings test_settings, mlperf::LogSettings log_settings)
{
    qsl::MIGSampleLibraryPtr_t lib;
    std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
    std::vector<bool> start_from_device(tensor_paths.size(), FLAGS_start_from_device);
    size_t padding
        = (test_settings.scenario == mlperf::TestScenario::MultiStream && test_settings.multi_stream_samples_per_query)
        ? (test_settings.multi_stream_samples_per_query - 1)
        : 0;
    lib = std::make_shared<qsl::MIGSampleLibrary>("Triton_MIGSampleLibrary", FLAGS_map_path,
        splitString(FLAGS_tensor_path, ","), FLAGS_performance_sample_count, padding, FLAGS_coalesced_tensor,
        start_from_device);

    return lib;
}

qsl::SampleLibraryPtr_t createSampleLibrary(mlperf::TestSettings test_settings, mlperf::LogSettings log_settings)
{
    LOG(INFO) << "Creating Sample Library";

    // Instantiate our QSL
    qsl::SampleLibraryPtr_t lib;
    std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
    std::vector<bool> start_from_device(tensor_paths.size(), FLAGS_start_from_device);

    if (FLAGS_use_dlrm_qsl)
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
            LOG(INFO) << "Loaded " << originalPartition.size() - 1 << " sample partitions. (" << tmp.size()
                      << ") bytes.";
        }

        // Force underlying QSL to load all samples, since we want to be able to
        // grab any partition
        // given the sample index.
        size_t perfPairCount = originalPartition.back();

        lib.reset(new DLRMSampleLibrary("Triton_DLRMSampleLibrary", FLAGS_map_path, splitString(FLAGS_tensor_path, ","),
            originalPartition, FLAGS_performance_sample_count, perfPairCount, 0 /* padding */, FLAGS_coalesced_tensor,
            FLAGS_start_from_device));
    }
    else
    {
        size_t padding = (test_settings.scenario == mlperf::TestScenario::MultiStream
                             && test_settings.multi_stream_samples_per_query)
            ? (test_settings.multi_stream_samples_per_query - 1)
            : 0;
        lib = std::make_shared<qsl::SampleLibrary>("Triton_SampleLibrary", FLAGS_map_path,
            splitString(FLAGS_tensor_path, ","), FLAGS_performance_sample_count, padding, FLAGS_coalesced_tensor,
            start_from_device);
    }

    return lib;
}

void runServer(mlperf::TestSettings& test_settings, mlperf::LogSettings& log_settings,
    triton_frontend::ServerSUTPtr_t& sut, qsl::SampleLibraryPtr_t& lib, double expected_qps,
    triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> c2s_state_mq,
    triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> s2c_state_mq,
    triton_frontend::sharedMQ_ptr<mlperf::ResponseId> c2s_req_mq,
    triton_frontend::sharedMQ_ptr<mlperf::QuerySampleResponse> s2c_resp_mq, 
    triton_frontend::sharedMSM_ptr c2s_shmem, triton_frontend::sharedMR_ptr s2c_shmem)
{
    LOG(INFO) << "Running Server on " << FLAGS_mig_uuid;

    // Instantiate our SUT and get the status of the server and model
    sut = std::make_shared<triton_frontend::Server_SUT>("Triton_Server", FLAGS_model_store_path, FLAGS_model_name,
        FLAGS_model_version, FLAGS_use_dlrm_qsl, FLAGS_start_from_device, FLAGS_pinned_input, FLAGS_mig_uuid,
        c2s_state_mq, s2c_state_mq, c2s_req_mq, s2c_resp_mq, c2s_shmem, s2c_shmem);
    if (FLAGS_use_dlrm_qsl)
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
    cudaProfilerStart();
    sut->Worker();
    cudaProfilerStop();

    // Check SUT end status and inform the SUT that we are done
    sut->ModelStats();
    sut->Done();

    sut.reset();
    lib.reset();
}

// start thread this right before LoadGen StartTest
void ClientResponseComplete(std::vector<std::string>& mig_uuids, triton_frontend::SUTShim_ptr& sut,
    triton_frontend::sharedMap_ptr<triton_frontend::IPCcomm> c2s_state_mq_map,
    triton_frontend::sharedMap_ptr<mlperf::QuerySampleResponse> s2c_resp_mq_map,
    triton_frontend::sharedMRMap_ptr s2c_shmem_map, triton_frontend::sharedBAMap_ptr s2c_baddr_map
    )
{
    while (!sut->is_finished())
    {
        // Get resp from MIG servers and notify LoadGen
        for (auto& u_ : mig_uuids)
        {
            try
            {
                // recover response from s2c_shmem_map and send back to LoadGen
                mlperf::QuerySampleResponse loadgen_response_ipc;
                bool rcvd = s2c_resp_mq_map[u_]->receive(loadgen_response_ipc);
                if (rcvd)
                {
                    // handle offset manually on shared memory
                    uintptr_t base_addr = s2c_baddr_map[u_];
                    mlperf::QuerySampleResponse loadgen_response{
                        loadgen_response_ipc.id, loadgen_response_ipc.data + base_addr, loadgen_response_ipc.size};
                    // We always send one inference response at a time
                    mlperf::QuerySamplesComplete(&loadgen_response, 1);
                }
            }
            catch (bip::interprocess_exception& ex)
            {
                std::cerr << "Met exception while waiting Init msg from " << u_ << " : " << ex.what() << std::endl;
                assert(false);
            }
        }
    }
}

void runClient(mlperf::TestSettings& test_settings, mlperf::LogSettings& log_settings, qsl::MIGSampleLibraryPtr_t& lib,
    double expected_qps, std::vector<std::string>& mig_uuids,
    triton_frontend::sharedMap_ptr<triton_frontend::IPCcomm> c2s_state_mq_map,
    triton_frontend::sharedMap_ptr<triton_frontend::IPCcomm> s2c_state_mq_map,
    triton_frontend::sharedMap_ptr<mlperf::ResponseId> c2s_req_mq_map,
    triton_frontend::sharedMap_ptr<mlperf::QuerySampleResponse> s2c_resp_mq_map,
    triton_frontend::sharedMSMMap_ptr c2s_shmem_map, triton_frontend::sharedMRMap_ptr s2c_shmem_map,
    triton_frontend::sharedBAMap_ptr s2c_baddr_map)
{
    LOG(INFO) << "Running Client";

    // Create SUTshim that calls StartTest on LoadGen
    // This SUTshim also distributes reqs to servers
    std::string scenario{FLAGS_scenario};
    boost::algorithm::to_lower(scenario);
    auto sutshim = std::make_shared<triton_frontend::SUTShim>(
        "Triton_MultiMigServer", scenario, FLAGS_batch_size, FLAGS_deque_timeout_usec, mig_uuids,
        c2s_state_mq_map, s2c_state_mq_map, c2s_req_mq_map, s2c_resp_mq_map, c2s_shmem_map,
        s2c_shmem_map);

    // Init, setup interprocess communication etc
    sutshim->Init();

    // register callback
    lib->registerLoadCB(
        [&sutshim](std::vector<mlperf::QuerySampleIndex> s) { return sutshim->Do_LoadSamplesToRam(s); });
    lib->registerUnloadCB(
        [&sutshim](std::vector<mlperf::QuerySampleIndex> s) { return sutshim->Do_UnloadSamplesFromRam(s); });

    // wait for all the clients being ready
    for (auto& u_ : mig_uuids)
    {
        try
        {
            triton_frontend::IPCcomm m_;
            s2c_state_mq_map[u_]->receive_it(m_);
            assert(m_ == triton_frontend::IPCcomm::Initialized);
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while waiting Init msg from " << u_ << " : " << ex.what() << std::endl;
            assert(false);
        }
    }

    // Initiate warmups in servers
    for (auto& u_ : mig_uuids)
    {
        try
        {
            // sending warmup signal
            triton_frontend::IPCcomm m_ = triton_frontend::IPCcomm::StartWarmup;
            c2s_state_mq_map[u_]->send_it(m_);
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while sending StartWarmup msg to " << u_ << " : " << ex.what() << std::endl;
            assert(false);
        }
    }

    // wait for all the clients done with Warmup
    for (auto& u_ : mig_uuids)
    {
        try
        {
            triton_frontend::IPCcomm m_;
            s2c_state_mq_map[u_]->receive_it(m_);
            assert(m_ == triton_frontend::IPCcomm::WarmupDone);
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while waiting WarmupDone msg from " << u_ << " : " << ex.what() << std::endl;
            assert(false);
        }
    }

    // how many RespWorker threads to start
    const int num_migs = mig_uuids.size();
    const int max_num_worker_handle = std::min(1, num_migs);
    int num_workers = ceil(float(num_migs) / float(max_num_worker_handle));
    std::thread workerThreads[num_workers];
    std::vector<std::string> mig_uuid_chops[num_workers];
    // map RespWorkers to MIG instances, in an interleaved way
    for (int i_ = 0; i_ < num_workers; i_++)
    {
        std::vector<std::string> s_;
        s_.clear();
        // clang-format off
        boost::copy(mig_uuids 
                    | boost::adaptors::sliced(i_, num_migs) 
                    | boost::adaptors::strided(num_workers),
                    std::back_inserter(s_));
        // clang-format on
        mig_uuid_chops[i_] = s_;
    }

    // thread for collecting responses
    for (int i_ = 0; i_ < num_workers; i_++)
    {
        workerThreads[i_] = std::thread(ClientResponseComplete, std::ref(mig_uuid_chops[i_]), std::ref(sutshim),
            std::ref(c2s_state_mq_map), std::ref(s2c_resp_mq_map), std::ref(s2c_shmem_map), std::ref(s2c_baddr_map));
    }

    // Start with SUTshim and launch
    mlperf::StartTest(sutshim.get(), lib.get(), test_settings, log_settings);

    LOG(INFO) << "Test is completed by LoadGen";

    // done
    sutshim->Done();

    // Join threads
    for (int i_ = 0; i_ < num_workers; i_++)
    {
        workerThreads[i_].join();
    }

    // send terminate msg to all Servers
    for (auto& u_ : mig_uuids)
    {
        try
        {
            triton_frontend::IPCcomm m_ = triton_frontend::IPCcomm::Terminate;
            c2s_state_mq_map[u_]->send_it(m_);
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while sending Terminate msg to " << u_ << " : " << ex.what() << std::endl;
            assert(false);
        }
    }

    // Print out equivalent QPS in multi_stream scenario.
    if (test_settings.scenario == mlperf::TestScenario::MultiStream)
    {
        std::cout << "Equivalent QPS computed by samples_per_query*target_qps : "
                  << test_settings.multi_stream_samples_per_query * test_settings.multi_stream_target_qps << std::endl;
    }

    lib.reset();

    // destroy IPC objects
    for (auto& u_ : mig_uuids)
    {
        try
        {
            std::string qstr;
            qstr = boost::replace_all_copy<std::string>(u_ + "_C2S", "/", "_");
            bip::message_queue::remove(qstr.c_str());
            qstr = boost::replace_all_copy<std::string>(u_ + "_S2C", "/", "_");
            bip::message_queue::remove(qstr.c_str());
            qstr = boost::replace_all_copy<std::string>(u_ + "_S2C_RSP", "/", "_");
            bip::message_queue::remove(qstr.c_str());
            qstr = boost::replace_all_copy<std::string>(u_ + "_C2S_SHM", "/", "_");
            bip::shared_memory_object::remove(qstr.c_str());
            qstr = boost::replace_all_copy<std::string>(u_ + "_S2C_SHM", "/", "_");
            bip::shared_memory_object::remove(qstr.c_str());
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while destroying IPC objects related to " << u_ << " : " << ex.what()
                      << std::endl;
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

std::string run_pipe(const char* cmd, bool print_output=true)
{
    std::array<char, 1024> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe)
    {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }
    if (print_output)
        LOG(INFO) << cmd << " returned: " << std::endl << result;

    return result;
}

std::vector<std::string> get_mig_uuids()
{
    std::vector<std::string> uuids(0);
    std::string cmd = "nvidia-smi -L";
    std::string result = run_pipe(cmd.data());

    std::regex expr{"\\(UUID: (MIG-GPU-\\S+)\\)"};
    std::regex_token_iterator<std::string::iterator> it{result.begin(), result.end(), expr, 1};
    std::regex_token_iterator<std::string::iterator> end;
    while (it != end)
    {
        uuids.push_back(*it);
        LOG(INFO) << "MIG found: " << *it;
        *it++;
    }

    return uuids;
}

boost::unordered::unordered_map<std::string, std::string> get_gpu_numa_map()
{
    boost::unordered::unordered_map<std::string, std::string> idx_to_uuid;
    boost::unordered::unordered_map<std::string, std::string> uuid_to_numa;

    std::sregex_iterator end;
    
    std::regex expr1{"(GPU-\\S+), (\\d+)"};
    std::regex expr2{"GPU(\\d+)\\s+.*\\s+(\\d+)"};
    
    std::string cmd = "nvidia-smi --query-gpu=gpu_uuid,index --format=csv,noheader";
    std::string result = run_pipe(cmd.data());

    // Map GPU IDX to GPU UUID (GPU IDX is by default sorted by PCIE BUS ID)
    std::sregex_iterator it{result.begin(), result.end(), expr1};    
    while (it != end)
    {
        std::smatch m_ = *it;
        assert(m_.size()%2 == 1); // due to first group
        for (auto i_ = 1; i_ < m_.size(); i_+=2)
        {
            idx_to_uuid.emplace(m_[i_+1].str(), m_[i_].str());
        }
        *it++;
    }

    // Map NUMA IDX to GPU UUID
    if (idx_to_uuid.size() == 1)
        for (auto& k_: idx_to_uuid)
            uuid_to_numa.emplace(k_.second, "0");
    else
    {
        std::string cmd = "nvidia-smi topo -m";
        std::string result = run_pipe(cmd.data());

        std::sregex_iterator it{result.begin(), result.end(), expr2};
        while (it != end)
        {
            std::smatch m_ = *it;
            assert(m_.size()%2 == 1);
            for (auto i_ = 1; i_ < m_.size(); i_+=2)
            {
                uuid_to_numa.emplace(idx_to_uuid[m_[i_].str()], m_[i_+1].str());
            }
            *it++;
        }
    }

    // FIXME: hack - LUNA uses PXB which confuses NUMA affinity
    //               This causes half of NUMA node to be unused
    //        WAR  - if numa node is already in numa_node_found,
    //               use numa node - 1 ; assert if result is invalid node
    //        WARNING: This is only tested/worked in LUNA machines
    //                 Other systems w/ PXB may work similarly
    // checking duplicate NUMA node captured from nvidia-smi topo -m
    std::string pxb_cmd = "nvidia-smi topo -m";
    std::string pxb_result = run_pipe(pxb_cmd.data(), false);
    std::regex pxb_expr{"GPU\\d+\\s+.*\\s+PXB\\s+.*\\s+\\d+"};
    if (std::regex_search(pxb_result, pxb_expr))
    {
        std::vector<std::string> numa_node_found;
        numa_node_found.clear();
        for (auto& k_: uuid_to_numa)
        {
            if (std::find(numa_node_found.begin(), numa_node_found.end(), k_.second) 
                    != numa_node_found.end())
            {
                int new_node = std::stoi(k_.second) - 1;
                assert(new_node >= 0);
                std::string new_node_str = std::to_string(new_node);
                k_.second = new_node_str;
                numa_node_found.push_back(new_node_str);
            }
            else
                numa_node_found.push_back(k_.second);
        }
    }
    // FIXME: END of hack

    return uuid_to_numa;
}

boost::unordered::unordered_map<std::string, std::vector<std::string>> get_cpu_numa_map()
{
    boost::unordered::unordered_map<std::string, std::vector<std::string>> numa_to_cpu;
    
    std::sregex_iterator end;
    std::regex expr{"node (\\d+) cpus: ([\\s*\\d+\\s*]+)"};

    std::string cmd = "numactl --hardware";
    std::string result = run_pipe(cmd.data());

    result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());
    
    std::sregex_iterator it{result.begin(), result.end(), expr};
    while (it != end)
    {
        std::smatch m_ = *it;
        assert(m_.size() > 2);
        std::istringstream iss(m_[2].str());
        std::vector<std::string> v{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
        numa_to_cpu.emplace(m_[1].str(), v);
        *it++;
    }

    return numa_to_cpu;
}

int main(int argc, char** argv)
{
    // Initialize the NVInfer plugins
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    // store original args before GFlags destroys them
    std::vector<const char*> org_args(argv, argv + argc);

    // Parse command line flags
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    // Load all the needed shared objects for plugins.
    std::vector<std::string> plugin_files = splitString(FLAGS_plugins, ",");
    for (auto& s : plugin_files)
    {
        void* dlh = dlopen(s.c_str(), RTLD_LAZY);
        if (nullptr == dlh)
        {
            std::cout << "Error loading plugin library " << s << std::endl;
            return 1;
        }
    }

    // safety measure
    signal(SIGABRT, sigabort_handler);
    parent_pid = getpid();

    // set up some settings
    mlperf::TestSettings test_settings = get_test_settings();
    mlperf::LogSettings log_settings = get_log_settings();

    // expected QPS; also need to adjust expected_qps on each MIG instance
    double expected_qps{1.0};
    switch (test_settings.scenario)
    {
    case mlperf::TestScenario::Offline: expected_qps = test_settings.offline_expected_qps/double(FLAGS_num_migs); break;
    case mlperf::TestScenario::Server: expected_qps = test_settings.server_target_qps/double(FLAGS_num_migs); break;
    case mlperf::TestScenario::MultiStream: expected_qps = test_settings.multi_stream_target_qps/double(FLAGS_num_migs); break;
    case mlperf::TestScenario::SingleStream:
        expected_qps = 1.0e9 / test_settings.single_stream_expected_latency_ns;
        break;
    }

    // main process
    if (FLAGS_mig_uuid == "N/A")
    {
        // how many MIG instances are available; avoiding const for mig_uuids, to
        // ease string manipulation
        std::vector<std::string> mig_uuids = get_mig_uuids();
        const int num_migs = mig_uuids.size();

        triton_frontend::sharedMap_ptr<triton_frontend::IPCcomm> bootup_map;
        for (auto& u_: mig_uuids)
        {
            std::string boot_qstr = boost::replace_all_copy<std::string>(u_ + "_BOOT", "/", "_");
            triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> boot_mq
                = std::make_shared<triton_frontend::msg_q<triton_frontend::IPCcomm>>(boot_qstr.c_str(), 1);
            bootup_map.emplace(u_, boot_mq);
        }

        // find NUMA node of GPUs
        boost::unordered::unordered_map<std::string, std::string> gpu_numa_map = get_gpu_numa_map();
        // find NUMA node of CPUs
        boost::unordered::unordered_map<std::string, std::vector<std::string>> numa_cpu_map = get_cpu_numa_map();

        // for graceful exit if fork fails
        bool f_exit = false;

        // fork Servers as many as MIG instances
        for (auto& u_ : mig_uuids)
        {
            // Launch child process
            std::string tgt_numa;
            for (auto& k_ : gpu_numa_map)
            {
                if (u_.find(k_.first) != std::string::npos)
                {
                    tgt_numa = k_.second;
                    break;
                }
            }
            LOG(INFO) << "Launching Inference process on " << u_ << ", NUMA: " << tgt_numa;

            pid_t c_pid = fork();
            f_exit |= c_pid < 0;

            // sleeping seems to help syscalls
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            // c_pid == 0 means this is forked child process
            if (c_pid == 0)
            {
                // set CPU affinity
                // NOTE: I cannot use numatcl --cpunodebind=.. --membind=.. b/c CUDA process is forced to see only one GPU
                //       which conflicts NUMA binding > 0
                if (gpu_numa_map.size() > 1)
                {
                    cpu_set_t mask;
                    CPU_ZERO(&mask);
                    for (auto& c_: numa_cpu_map[tgt_numa])
                    {
                        CPU_SET(std::stoi(c_), &mask);
                    }
                    int set_cpu_affinity_res = sched_setaffinity(c_pid, sizeof(mask), &mask);
                    assert(set_cpu_affinity_res == 0);
                }
               
                // new cmd: org_cmdline + --mig_uuid=... + --target_numa_node=... + --numa_nodemask=...
                // so that MIG instance can be captured and NUMA affinity is exploited
                std::vector<const char*> args(org_args);
                std::string mig_arg = "--mig_uuid=";
                mig_arg.append(u_);
                args.push_back(const_cast<char*>(mig_arg.data()));
                std::string nmig_arg = "--num_migs=";
                nmig_arg.append(std::to_string(num_migs));
                args.push_back(const_cast<char*>(nmig_arg.data()));
                std::string numa_arg = "--target_numa=";
                numa_arg.append(tgt_numa);
                args.push_back(const_cast<char*>(numa_arg.data()));
                std::string nmask_arg = "--num_numa=";
                nmask_arg.append(std::to_string(numa_cpu_map.size()));
                args.push_back(const_cast<char*>(nmask_arg.data()));
                args.push_back(nullptr);
                args.shrink_to_fit();

                // overwrite ENV var
                setenv("CUDA_VISIBLE_DEVICES", u_.c_str(), 1);
                extern char** environ;
                execve(argv[0], const_cast<char* const*>(args.data()), environ);
            }
        }

        // exit if needed
        if (f_exit)
        {
            std::cerr << "Failed to fork process(es), terminating from Parent PID: " << parent_pid << std::endl;
            killpg(getpgid(parent_pid), SIGTERM);
        }
        
        // create sample libarary for callbacks
        qsl::MIGSampleLibraryPtr_t mig_lib = createMIGSampleLibrary(test_settings, log_settings);

        // Build shared message queue & shared memory segments
        triton_frontend::sharedMap_ptr<triton_frontend::IPCcomm> c2s_state_mq_map;
        triton_frontend::sharedMap_ptr<triton_frontend::IPCcomm> s2c_state_mq_map;
        triton_frontend::sharedMap_ptr<mlperf::ResponseId> c2s_req_mq_map;
        triton_frontend::sharedMap_ptr<mlperf::QuerySampleResponse> s2c_resp_mq_map;
        triton_frontend::sharedMSMMap_ptr c2s_shmem_map;
        triton_frontend::sharedMRMap_ptr s2c_shmem_map;
        triton_frontend::sharedBAMap_ptr s2c_baddr_map;

        // Wait until Servers creates IPC objects
        for (auto& u_ : mig_uuids)
        {
            try
            {
                triton_frontend::IPCcomm m_;
                bootup_map[u_]->receive_it(m_);
                assert(m_ == triton_frontend::IPCcomm::Bootup);
            }
            catch (bip::interprocess_exception& ex)
            {
                std::cerr << "Met exception while waiting Bootup from " << u_ << " : " << ex.what() << std::endl;
                assert(false);
            }
        }

        // remove this one time bootup queue
        for (auto& u_ : mig_uuids)
        {
            std::string boot_qstr = boost::replace_all_copy<std::string>(u_ + "_BOOT", "/", "_");
            bip::message_queue::remove(boot_qstr.c_str());
        }

        // Open shared memory and message queues, as many as MIGs
        for (auto& u_ : mig_uuids)
        {
            try
            {
                // Client to Server State MQ
                std::string c2s_qstr = boost::replace_all_copy<std::string>(u_ + "_C2S", "/", "_");
                triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> c2s_mq
                    = std::make_shared<triton_frontend::msg_q<triton_frontend::IPCcomm>>(c2s_qstr.c_str());
                c2s_state_mq_map.emplace(u_, c2s_mq);

                // Server to Client State MQ
                std::string s2c_qstr = boost::replace_all_copy<std::string>(u_ + "_S2C", "/", "_");
                triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> s2c_mq
                    = std::make_shared<triton_frontend::msg_q<triton_frontend::IPCcomm>>(s2c_qstr.c_str());
                s2c_state_mq_map.emplace(u_, s2c_mq);

                // Client to Server Req MQ
                std::string req_qstr = boost::replace_all_copy<std::string>(u_ + "_C2S_REQ", "/", "_");
                triton_frontend::sharedMQ_ptr<mlperf::ResponseId> req_mq
                    = std::make_shared<triton_frontend::msg_q<mlperf::ResponseId>>(req_qstr.c_str());
                c2s_req_mq_map.emplace(u_, req_mq);

                // Server to Client Resp MQ
                std::string rsp_qstr = boost::replace_all_copy<std::string>(u_ + "_S2C_RSP", "/", "_");
                triton_frontend::sharedMQ_ptr<mlperf::QuerySampleResponse> rsp_mq
                    = std::make_shared<triton_frontend::msg_q<mlperf::QuerySampleResponse>>(rsp_qstr.c_str());
                s2c_resp_mq_map.emplace(u_, rsp_mq);
            }
            catch (bip::interprocess_exception& ex)
            {
                std::cerr << "Met exception while creating msg_q towards " << u_ << " : " << ex.what() << std::endl;
                assert(false);
            }

            // Create shared memory segments, as many as MIGs
            try
            {
                // Client to Server shmem, used to toss samples
                std::string c2s_qstr = boost::replace_all_copy<std::string>(u_ + "_C2S_SHM", "/", "_");
                triton_frontend::sharedMSM_ptr c2s_shmem
                    = std::make_shared<bip::managed_shared_memory>(bip::open_only, c2s_qstr.c_str());
                c2s_shmem_map.emplace(u_, c2s_shmem);

                // Server to Client shmem, used to mmap output tensors
                std::string s2c_qstr = boost::replace_all_copy<std::string>(u_ + "_S2C_SHM", "/", "_");
                bip::shared_memory_object s2c_shm(bip::open_only, s2c_qstr.c_str(), bip::read_write);
                triton_frontend::sharedMR_ptr s2c_shmem
                    = std::make_shared<bip::mapped_region>(s2c_shm, bip::read_write);
                s2c_shmem_map.emplace(u_, s2c_shmem);
                s2c_baddr_map.emplace(u_, reinterpret_cast<uintptr_t>(s2c_shmem->get_address()));
            }
            catch (bip::interprocess_exception& ex)
            {
                std::cerr << "Met exception while creqting shared memory towards " << u_ << " : " << ex.what()
                          << std::endl;
                assert(false);
            }
        }

        // Run Client
        runClient(test_settings, log_settings, mig_lib, expected_qps, mig_uuids, 
            c2s_state_mq_map, s2c_state_mq_map, c2s_req_mq_map, s2c_resp_mq_map, 
            c2s_shmem_map, s2c_shmem_map, s2c_baddr_map);
    }

    // Server process in MIG instances
    else
    {
        // Set NUMA memory affinity if needed
        if (FLAGS_num_numa > 1)
            bindNumaMemPolicy(FLAGS_target_numa, FLAGS_num_numa);

        // prep IPC objects
        triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> c2s_state_mq;
        triton_frontend::sharedMQ_ptr<triton_frontend::IPCcomm> s2c_state_mq;
        triton_frontend::sharedMQ_ptr<mlperf::ResponseId> c2s_req_mq;
        triton_frontend::sharedMQ_ptr<mlperf::QuerySampleResponse> s2c_resp_mq;
        triton_frontend::sharedMSM_ptr c2s_shmem;
        triton_frontend::sharedMR_ptr s2c_shmem;

        // using these for tuning
        size_t comm_mq_size {100};
        size_t req_mq_size {100};
        size_t resp_mq_size {100};

        // Instantiate IPC objects
        try
        {
            // Client to Server State MQ
            std::string c2s_qstr = boost::replace_all_copy<std::string>(FLAGS_mig_uuid + "_C2S", "/", "_");
            c2s_state_mq
                = std::make_shared<triton_frontend::msg_q<triton_frontend::IPCcomm>>(c2s_qstr.c_str(), comm_mq_size);

            // Server to Client State MQ
            std::string s2c_qstr = boost::replace_all_copy<std::string>(FLAGS_mig_uuid + "_S2C", "/", "_");
            s2c_state_mq
                = std::make_shared<triton_frontend::msg_q<triton_frontend::IPCcomm>>(s2c_qstr.c_str(), comm_mq_size);

            // Client to Server Req MQ
            std::string req_qstr = boost::replace_all_copy<std::string>(FLAGS_mig_uuid + "_C2S_REQ", "/", "_");
            c2s_req_mq
                = std::make_shared<triton_frontend::msg_q<mlperf::ResponseId>>(req_qstr.c_str(), req_mq_size);

            // Server to Client Resp MQ
            std::string rsp_qstr = boost::replace_all_copy<std::string>(FLAGS_mig_uuid + "_S2C_RSP", "/", "_");
            s2c_resp_mq
                = std::make_shared<triton_frontend::msg_q<mlperf::QuerySampleResponse>>(rsp_qstr.c_str(), resp_mq_size);
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while creating msg_q from " << FLAGS_mig_uuid << " : " << ex.what() << std::endl;
            assert(false);
        }

        // Create shared memory segments, as many as MIGs
        try
        {
            const size_t max_in_size = 6000000 * sizeof(mlperf::QuerySample);
            const size_t max_out_size = 1 << 29;

            // Client to Server shmem, used to toss samples
            std::string c2s_qstr = boost::replace_all_copy<std::string>(FLAGS_mig_uuid + "_C2S_SHM", "/", "_");
            bip::shared_memory_object::remove(c2s_qstr.c_str());
            c2s_shmem
                = std::make_shared<bip::managed_shared_memory>(bip::create_only, c2s_qstr.c_str(), max_in_size);

            // Server to Client shmem, used to mmap output tensors
            std::string s2c_qstr = boost::replace_all_copy<std::string>(FLAGS_mig_uuid + "_S2C_SHM", "/", "_");
            bip::shared_memory_object::remove(s2c_qstr.c_str());
            bip::shared_memory_object s2c_shm(bip::create_only, s2c_qstr.c_str(), bip::read_write);
            s2c_shm.truncate(max_out_size);
            s2c_shmem = std::make_shared<bip::mapped_region>(s2c_shm, bip::read_write);
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while creqting shared memory from " << FLAGS_mig_uuid << " : " << ex.what()
                        << std::endl;
            assert(false);
        }
        
        std::string boot_qstr = boost::replace_all_copy<std::string>(FLAGS_mig_uuid + "_BOOT", "/", "_");
        triton_frontend::msg_q<triton_frontend::IPCcomm> boot_mq = 
            triton_frontend::msg_q<triton_frontend::IPCcomm>(boot_qstr.c_str());
        try
        {
            triton_frontend::IPCcomm m_ = triton_frontend::IPCcomm::Bootup;
            boot_mq.send_it(m_);
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while sending Bootup from " << FLAGS_mig_uuid << " : " << ex.what() << std::endl;
            assert(false);
        }

        LOG(INFO) << "Performing inference on mig_uuid: " << FLAGS_mig_uuid << ", NUMA binding: " << FLAGS_target_numa;

        // Run Server
        triton_frontend::ServerSUTPtr_t sut;

        // initiate SampleLibrary
        qsl::SampleLibraryPtr_t lib = createSampleLibrary(test_settings, log_settings);

        runServer(test_settings, log_settings, sut, lib, expected_qps, 
            c2s_state_mq, s2c_state_mq, c2s_req_mq, s2c_resp_mq,
            c2s_shmem, s2c_shmem);
    }

    /* Return pass */
    return 0;
}
