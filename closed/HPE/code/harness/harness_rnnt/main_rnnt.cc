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

// TensorRT
#include "logger.h"
#include "logging.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

// Google Logging
#include <glog/logging.h>

// General C++
#include <algorithm>
#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>

#include "qsl.hpp"
#include "utils.hpp"
#include "half.h"

// Kernels
#include "rnnt_kernels.h"
#include "loadgen.h"
#include "system_under_test.h"
#include "cuda_profiler_api.h"
#include "nvtx_wrapper.h"

// Utility
#include "SyncWorkQueue.hpp"
#include "CudaBuffer.h"
#include "CudaGraphCache.hpp"

// Parts
#include "preprocessing.hpp" // DALI Components
#include "metadata.hpp" // Metadata tracking
#include "warmup.hpp" // Warmup utility


// RNNT Server
using namespace std;


// =============================
//     Knobs/parameters
// =============================
//

// Infrastructure knobs
DEFINE_bool(verbose, false, "Use verbose logging");
DEFINE_bool(debug_mode, false, "Toggle debug verbose logging");
DEFINE_bool(accuracy_mode, false, "Toggle accuracy_mode");
DEFINE_string(plugins, "build/plugins/RNNTOptPlugin/librnntoptplugin.so", "Comma-separated list of shared objects for plugins");
DEFINE_uint64(batch_size, 2048, "Batch size");
DEFINE_uint64(encoder_batch_size, 0, "Encoder batch size (0 to ignore)");
DEFINE_string(engine_dir, "build/engines/rnnt", "Path to engine directory");
DEFINE_string(val_map, "data_maps/rnnt_dev_clean_512/val_map.txt", "Path to val_map, a list of input file names");
DEFINE_string(preprocessed_data_dir, "build/preprocessed_data/rnnt_dev_clean_512/fp16", "Path to preprocessed data directory");
DEFINE_string(preprocessed_length_dir, "build/preprocessed_data/rnnt_dev_clean_512/int32", "Path to preprocessed data directory storing sample lengths");
DEFINE_string(raw_data_dir, "build/preprocessed_data/rnnt_dev_clean_500_raw", "Path to preprocessed data directory");
DEFINE_string(raw_length_dir, "build/preprocessed_data/rnnt_dev_clean_500_raw/int32", "Path to preprocessed data directory");
DEFINE_uint64(streams_per_gpu, 1, "Number of streams per GPU");
DEFINE_int64(num_warmups, -1, "Number of samples to warmup on. A value of -1 runs two full batches for each stream (2*batch_size*streams_per_gpu*NUM_GPUS), 0 turns off warmups.");
DEFINE_uint64(server_num_issue_query_threads, 0, "Number of IssueQuery threads used in Server scenario");
DEFINE_string(devices, "all", "Enable comma separated numbered devices");

// Hyper parameters
DEFINE_int32(hp_max_seq_length, 128, "max sequence length for audio");
DEFINE_uint64(hp_labels_size, 29, "alphabet symbol size (including blank)");
DEFINE_uint64(hp_encoder_input_size , 240, "encoder audio input size");
DEFINE_uint64(hp_encoder_hidden_size, 1024, "encoder RNNs number of hidden units");
DEFINE_uint64(hp_enc_pre_rnn_layers, 2, "encoder pre RNN # layers");
DEFINE_uint64(hp_enc_post_rnn_layers, 3, "encoder post RNN # layers");
DEFINE_uint64(hp_decoder_input_size, 320, "decoder RNN input size (post-embedding)");
DEFINE_uint64(hp_decoder_hidden_size, 320, "decoder RNN number of hidden units");
DEFINE_uint64(hp_dec_rnn_layers, 2 , "decoder RNN #layers");
DEFINE_uint64(hp_joint_hidden_size, 512, "joint net number of hidden units");
DEFINE_uint64(hp_max_symbols_per_step, 30, "maximum symbols per encoder time step");

// Model knobs
DEFINE_bool(enable_audio_processing, true, "Enable timed audio processing path");
DEFINE_bool(start_from_device, false, "Start processing with samples already on device");
DEFINE_bool(audio_fp16_input, true, "Use raw wav data stored in fp16 format");
DEFINE_bool(use_copy_kernel, true, "Use dali's scatter gather kernel instead of using cudamemcpyAsync");
DEFINE_bool(use_copy_kernel_cudamemcpy, false, "Dali's scatter gather kernel to use cudamemcpyasync under the hood");

DEFINE_uint64(audio_batch_size, 256, "DALI audio processing pipeline  - batch_size parameter");
DEFINE_uint64(audio_num_threads, 2, "DALI audio processing pipeline - num_threads parameter");
DEFINE_uint64(audio_prefetch_queue_depth, 2, "DALI audio processing - number of pipeline instances per GPU");
DEFINE_uint64(dali_pipeline_depth, 4, "Depth of sub-batch processing in DALI pipeline");
DEFINE_uint64(dali_batches_issue_ahead, 4, "Number of batches for which cudamemcpy is issued ahead of dali compute");
DEFINE_uint64(audio_device_type, 1, "DALI audio processing - CPU = 0, GPU = 1. Ensure audio_serialized_pipeline_file is adjusted accordingly");
DEFINE_uint64(audio_buffer_line_size, 240960, "Max tensor output size == buffer line size");
DEFINE_uint64(audio_buffer_num_lines, 4096, "Number of audio samples in flight");
DEFINE_string(audio_serialized_pipeline_file, "build/bin/dali/dali_pipeline_gpu_fp16.pth", "DALI audio processing - serialized filename");

DEFINE_bool(encoder, true, "Set to enable true encoder in the run");
DEFINE_bool(decoder_loop, true, "Set to enable decoder-joint-search loop in the run");
DEFINE_bool(always_advance_time, false, "Force search algorithm to advance encoder pointer even if symbol is not blank");
DEFINE_bool(decoder_isel, true, "Set to enable Isel optimization for decoder inputs");
DEFINE_bool(batch_sorting, true, "Set to enable batch sorting");
DEFINE_bool(seq_splitting, true, "Set to enable sequence splitting");
DEFINE_bool(ref_model, false, "Set to enable reference (non cuda graph) model");
DEFINE_bool(cuda_graph, true, "Set to enable cuda graph");
DEFINE_string(cuda_graph_cache_generation_method, "exp", "Describe cuda graph caching generation method (options: 'exp', 'linear', '1,2,17' [space-less CSV])");
DEFINE_uint64(cuda_graph_cache_generation_scale, 2, "Scale used for cache generation (used in exp and linear)");
DEFINE_bool(cuda_legacy_graph_caching, false, "Set to enable legacy cuda graph caching (cache capacity of 1)");
DEFINE_int64(cuda_graph_unroll, 2, "Determine unrolling degree for decoder loop cuda graphs");
DEFINE_bool(disaggregated_joint, true, "Set to enable disaggregation of joint model");
DEFINE_bool(cuda_joint_backend, true, "Set to enable cuda optimized version of joint backend model");
DEFINE_bool(pipelined_execution, true, "Set to enable encoder/decoder to work in pipelined fashion");
DEFINE_uint64(max_overall_seq_length, 500, "Set the maximum sequence length for the selected dataset");
DEFINE_bool(disable_encoder_plugin, false, "Set to disable int8 encoder plugin");

DEFINE_string(gpu_engines, "", "Engine");

DEFINE_string(scenario, "Offline", "Scenario to run for Loadgen (Offline, SingleStream, Server)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "rnnt", "Model name");

// configuration files
DEFINE_string(mlperf_conf_path, "temp_mlperf.conf", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "code/harness/harness_rnnt/user.conf", "Path to user.conf");
DEFINE_uint64(single_stream_expected_latency_ns, 100000, "Inverse of desired target QPS");

// Loadgen logging settings
DEFINE_string(logfile_outdir, "build/logs", "Specify the existing output directory for the LoadGen logs");
DEFINE_string(logfile_prefix, "rnnt_logs_", "Specify the filename prefix for the LoadGen log files");
DEFINE_string(logfile_suffix, "", "Specify the filename suffix for the LoadGen log files");
DEFINE_bool(logfile_prefix_with_datetime, false, "Prefix filenames for LoadGen log files");
DEFINE_bool(log_copy_detail_to_stdout, false, "Copy LoadGen detailed logging to stdout");
DEFINE_bool(disable_log_copy_summary_to_stdout, false, "Disable copy LoadGen summary logging to stdout");
DEFINE_string(log_mode, "AsyncPoll", "Logging mode for Loadgen");
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000, "Specify the poll interval for asynchrounous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

// QSL arguments
DEFINE_uint64(performance_sample_count, 2513, "Number of samples to load in performance set.");
// DEFINE_uint64(performance_sample_count, 2939, "Number of samples to load in performance set.  0=use default");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {
    {"SingleStream", mlperf::TestScenario::SingleStream},
    {"Offline", mlperf::TestScenario::Offline},
    {"Server", mlperf::TestScenario::Server}};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {
    {"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly},
    {"PerformanceOnly", mlperf::TestMode::PerformanceOnly}};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {
    {"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly},
    {"Synchronous", mlperf::LoggingMode::Synchronous}};

// Compile time defines
#undef PER_BATCH_LOG            // turn on detailed batch timing
#undef CUDA_EVENT               // turn on cuda events profiling
#undef CUDA_EVENT_ENCODER       // turn on cuda events profiling : Encoder
#undef CUDA_EVENT_DECODER       // turn on cuda events profiling : Decoder-greedy loop
#undef RNNT_STATS               // turn on generic RNN-T stats collection

#define _BLANK_ 28

// static model knobs
#define CUDA_SPARSE_MEMSET      // use cuda rnntSparseMemSet


// =============================
//     Support
// =============================
//

// Argument parsing
std::set<size_t> parse_cuda_graph_cache_generation() {
    std::set<size_t> to_ret;
    if (FLAGS_cuda_graph_cache_generation_method == "exp") {
        for (size_t i = 1; i <= FLAGS_batch_size; i *= FLAGS_cuda_graph_cache_generation_scale) {
            to_ret.insert(i);
        }
    } else if (FLAGS_cuda_graph_cache_generation_method == "linear") {
        for (size_t i = 1; i <=FLAGS_batch_size; i += FLAGS_cuda_graph_cache_generation_scale) {
            to_ret.insert(i);
        }
    } else {
        std::vector<std::string> vec_of_strnums;
        // Treat as CSV: turn it into a stream, and push back using commas as delims:
        // Shamelessly stolen from: https://stackoverflow.com/a/10861816
        std::istringstream whole_str(FLAGS_cuda_graph_cache_generation_method);
        while (whole_str.good()) {
            std::string temp;
            std::getline(whole_str, temp, ',');
            vec_of_strnums.push_back(temp);
        }
        for (const auto& strnum : vec_of_strnums) {
            auto result = stoul(strnum);
            to_ret.insert(result);
        }
    }
    // Make sure we can always satisfy max_batch_size:
    to_ret.insert(FLAGS_batch_size);
    return to_ret;
}



// Miscellanous macros and debug functions
// 

// TODO: use SSE intrinsics. Need compiler flags
// f16 = _cvtss_sh(f32, 0);
// f32 = _cvtsh_ss(f16);

// Half to float:
#define macro_fp16tofp32(h) (((h&0x8000)<<16) | (((h&0x7c00)+0x1C000)<<13) | ((h&0x03FF)<<13))

// Float to half:
#define macro_fp32tofp16(x) (((x>>16)&0x8000)|((((x&0x7f800000)-0x38000000)>>13)&0x7c00)|((x>>13)&0x03ff))

using half = half_float::half;


// Encoder/decoder depth (for data privatization)
enum RNNT_PIPE { RNNT_ENC = 0, RNNT_DEC = 1, PIPE_DEPTH };

// =============================
//     Tensor container
// =============================
//

// For convenience, we are packing all HostBuffer tensors in a single object

class RnntTensorContainer
{
private:

    void preload_tensor_from_file(
             std::shared_ptr<CudaBufferRaw> devBuffer,
             size_t numElements,
             size_t esize,
             const std::string& filePrefix)
    {
        // determine file name
        std::string fileName = FLAGS_engine_dir + "/" + filePrefix;
        switch(esize) 
        {
           case 2 : fileName = fileName + ".fp16.dat";
                    break;
           case 4 : fileName = fileName + ".fp32.dat";
                    break;
           default: gLogInfo << "RnntTensorContainer:preload_tensor_from_file: warning! unsupported data size for cuda_joint_backend FC2 weight/bias preloading" << std::endl;
                    CHECK(false);
        }

        // open the file into a temporal buffer
        size_t size{0};
        std::vector<char> buf;
        std::ifstream file(fileName, std::ios::binary);
        CHECK(file.good());
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            CHECK(size == (numElements*esize));
            buf.resize(size);
            file.read(buf.data(), size);
            file.close();
        }

        // transfer to the device buffer
        std::cout << "cudaMemcpy blocking " << std::endl;
        CHECK_EQ(cudaMemcpy(devBuffer->data(), buf.data(), numElements * esize, cudaMemcpyHostToDevice), cudaSuccess);
    }

public:
    // native data size
    size_t esize;

    // full tensor list
    std::shared_ptr<CudaBufferRaw> encoderIn;        // input  => enc
    std::shared_ptr<CudaBufferInt32> encoderInLengths; // input  => enc (seq lengths)
    std::shared_ptr<CudaBufferRaw> encoderOut[PIPE_DEPTH];    // enc    => (greedy) joint
    std::shared_ptr<CudaBufferRaw> encGather;        // greedy => joint
    std::shared_ptr<CudaBufferRaw> decoderSymbol;    // dec    => joint
    std::shared_ptr<CudaBufferRaw> decoderHidden;    // dec    => greedy
    std::shared_ptr<CudaBufferRaw> decoderCell;      // dec    => greedy
    std::shared_ptr<CudaBufferInt32> jointSymbol;      // joint  => greedy
    std::shared_ptr<CudaBufferInt32> greedyWinner;     // greedy => dec
    std::shared_ptr<CudaBufferRaw> greedyHidden;     // greedy => dec
    std::shared_ptr<CudaBufferRaw> greedyCell;       // greedy => dec
    
    // greedy in GPU support
    std::shared_ptr<CudaBufferInt32> encIdx;                   // greedy
    std::shared_ptr<CudaBufferInt32> seqLen;                   // greedy
    std::shared_ptr<CudaBufferInt32> num_symbols_current_step; // greedy
    std::shared_ptr<CudaBufferInt32> outSeq;                   // greedy
    std::shared_ptr<CudaBufferInt32> outSeqLen;                // greedy
    std::shared_ptr<CudaBufferBool> isNotBlank;       // greedy => isel

    // sequence splitting support
    std::shared_ptr<CudaBufferRaw> encoderPreHidden; // enc    => enc
    std::shared_ptr<CudaBufferRaw> encoderPreCell;   // enc    => enc
    std::shared_ptr<CudaBufferRaw> encoderPostHidden; // enc    => enc
    std::shared_ptr<CudaBufferRaw> encoderPostCell;   // enc    => enc

    // disaggregated joint
    std::shared_ptr<CudaBufferRaw> jointFc1Encoder;   // fc1_a => joint backend
    std::shared_ptr<CudaBufferRaw> jointFc1Decoder;   // fc1_b => joint backend

    // cuda_joint_backend
    std::shared_ptr<CudaBufferInt32> jointFc2Output;   // used to hold FC2 output
    std::shared_ptr<CudaBufferRaw> jointFc2Weights;  // used to hold FC2 weights 
    std::shared_ptr<CudaBufferRaw> jointFc2Bias;     // used to hold FC2 bias 

    // Miscellanous temporal buffers
    #ifdef CUDA_SPARSE_MEMSET
    std::shared_ptr<CudaBufferBool> sparseMask[PIPE_DEPTH];   // used for seq_splitting initialization (privatized by polarity)
    #endif

    // Host buffers
    struct host_buffers_t {
        std::shared_ptr<HostBufferInt32> outSeqTmp;     // temporal buffer to host the output sequences
        std::shared_ptr<HostBufferInt32> outSeqLen;     // temporal buffer to host the output sequence lengths
    } host;

    // Done host/device: needs to be privatize per unrolled loop
    // Not moved to cudabuffer because uses ManagedMemory
    struct done_buffers_t {
        int32_t *h_ptr;
        int32_t *d_ptr;
    } done;


    // Cuda graph support
    // Managed exclusively by the decoder
    // Two copies must be maintained because the encoder uses ping-pong buffers
    // (and the graph captures pointers used in a launch)
    // So if we want the graph to capture a different buffer, we need to use a different graph
    CudaGraphCache cg_cache[PIPE_DEPTH];

    // Constructor
    RnntTensorContainer(size_t _esize)
        : esize(_esize)
    {
        // Because sometimes we want to express blobs of data ...
        size_t bytes_per_batch;
        // ...and sometimes we want to just say how many elements
        size_t numElements;
        // Encoder:
        //
        //    encoder input [BS][SEQ][CHAN]
        bytes_per_batch = FLAGS_hp_max_seq_length * FLAGS_hp_encoder_input_size * esize;

        // For now, warn because this code path hasn't been updated in a while
        if (FLAGS_encoder_batch_size != 0) {
            gLogWarning << "encoder_batch_size has not been updated in a while!" << std::endl;
        }
        auto encoder_batch_size = FLAGS_encoder_batch_size ? FLAGS_encoder_batch_size : FLAGS_batch_size;

        encoderIn   = std::make_shared<CudaBufferRaw>(encoder_batch_size, bytes_per_batch);

        if(FLAGS_seq_splitting) {
            //    pre-encoder hidden into tensor [BS][layers=2][ENC_HIDDEN]
            bytes_per_batch = FLAGS_hp_enc_pre_rnn_layers * FLAGS_hp_encoder_hidden_size * esize;
            encoderPreHidden = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);
            //    pre-encoder cell into tensor [BS][layers=2][ENC_HIDDEN]
            encoderPreCell   = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);

            //    post-encoder hidden into tensor [BS][layers=3][ENC_HIDDEN]
            bytes_per_batch = FLAGS_hp_enc_post_rnn_layers * FLAGS_hp_encoder_hidden_size * esize;
            encoderPostHidden = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);
            //    post-encoder cell into tensor [BS][layers=3][ENC_HIDDEN]
            encoderPostCell   = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);
        }

        //    encoder seq length input [BS]
        encoderInLengths = std::make_shared<CudaBufferInt32>(FLAGS_batch_size);

        //    encoder output into tensor [BS][SEQ//2][ENC_HIDDEN]
        bytes_per_batch = (FLAGS_hp_max_seq_length>>1) * FLAGS_hp_encoder_hidden_size * esize;
        encoderOut[0]  = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);
        if(FLAGS_pipelined_execution) {
            encoderOut[1]  = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);
        }

        // compress/gather encoder output used for joint input
        bytes_per_batch = 1 * FLAGS_hp_encoder_hidden_size * esize;
        encGather   = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);

        // Decoder:
        //
        //    decoder symbol into tensor [BS][1][DEC_HIDDEN]
        bytes_per_batch = 1 * FLAGS_hp_decoder_hidden_size * esize;
        decoderSymbol = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);
        //    decoder hidden into tensor [BS][layers=2][DEC_HIDDEN]
        bytes_per_batch = FLAGS_hp_dec_rnn_layers * FLAGS_hp_decoder_hidden_size * esize;
        decoderHidden = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);
        //    decoder cell into tensor [BS][layers=2][DEC_HIDDEN]
        bytes_per_batch = FLAGS_hp_dec_rnn_layers * FLAGS_hp_decoder_hidden_size * esize;
        decoderCell = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);

        // Joint:
        //
        //    joint output into tensor [BS][1]
        jointSymbol = std::make_shared<CudaBufferInt32>(FLAGS_batch_size);

        // Disaggregated Joint
        //
        //    fc1_a : [BS][JOINT_HIDDEN_SIZE]
        //    fc1_b : [BS][JOINT_HIDDEN_SIZE]
        if (FLAGS_disaggregated_joint) {
            bytes_per_batch = FLAGS_hp_joint_hidden_size * esize;
            jointFc1Encoder = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);
            jointFc1Decoder = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);

            if (FLAGS_cuda_joint_backend) {
                // FC2 output
                numElements = FLAGS_batch_size * FLAGS_hp_labels_size;
                jointFc2Output = std::make_shared<CudaBufferInt32>(numElements);
                // FC2 weights (w/ data preload)
                numElements = FLAGS_hp_joint_hidden_size * FLAGS_hp_labels_size;
                jointFc2Weights = std::make_shared<CudaBufferRaw>(numElements * esize);
                preload_tensor_from_file(jointFc2Weights, numElements, esize, "joint_fc2_weight_ckpt");
                // FC2 bias (w/ data preload)
                numElements = FLAGS_hp_labels_size;
                jointFc2Bias = std::make_shared<CudaBufferRaw>(numElements * esize);
                preload_tensor_from_file(jointFc2Bias, numElements, esize, "joint_fc2_bias_ckpt");

            }
        }

        // Greedy:
        //    Note that we need to initialize them to zero
        //
         
        // [BS][1][1]
        numElements = FLAGS_batch_size * 1 * 1;
        encIdx = std::make_shared<CudaBufferInt32>(numElements);
        // [BS][1][1]
        seqLen = std::make_shared<CudaBufferInt32>(numElements);
        // [BS][1][1]
        num_symbols_current_step = make_shared<CudaBufferInt32>(numElements);

        isNotBlank = std::make_shared<CudaBufferBool>(numElements);
        outSeqLen = std::make_shared<CudaBufferInt32>(numElements);

        // [BS * hp_max_symbols_per_step][1][1]
        numElements = FLAGS_batch_size * FLAGS_hp_max_seq_length;
        outSeq = std::make_shared<CudaBufferInt32>(numElements);

        //    greedy symbol into tensor [BS][1][1]
        //    initialized with _BLANK_ so that the first input symbol to the decoder is SOS,
        //    which leads to zero embed vector.
        numElements = FLAGS_batch_size * 1 * 1;
        greedyWinner = std::make_shared<CudaBufferInt32>(numElements);

        //    greedy hidden into tensor [BS][layers=2][DEC_HIDDEN]
        bytes_per_batch = FLAGS_hp_dec_rnn_layers * FLAGS_hp_decoder_hidden_size * esize;
        greedyHidden = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);
        //    greedy cell into tensor [BS][layers=2][DEC_HIDDEN]
        greedyCell = std::make_shared<CudaBufferRaw>(FLAGS_batch_size, bytes_per_batch);

        // Isel
        //      iselSelect [BS][1][1] of type bool
        // numElements = FLAGS_batch_size * 1 * 1;
        // isNotBlank = std::shared_ptr<CudaBufferRaw>(new CudaBufferRaw(numElements * sizeof(bool)));


        // Miscellaneous buffers
        #ifdef CUDA_SPARSE_MEMSET
        for(int i = 0; i < PIPE_DEPTH; i++) {
            sparseMask[i] = std::make_shared<CudaBufferBool>(FLAGS_batch_size);
        }
        #endif

        // Host buffers

        // CHECK THAT THESE ARE BIG ENOUGH!
        // Maybe have HostBuffer do cudaHostAllocMapped
        numElements = FLAGS_batch_size * FLAGS_hp_max_seq_length;
        host.outSeqTmp = std::make_shared<HostBufferInt32>(numElements);
        numElements = FLAGS_batch_size;
        host.outSeqLen = std::make_shared<HostBufferInt32>(numElements);



        // Done buffers
        auto unroll_depth = FLAGS_cuda_graph_unroll;
        cudaHostAlloc(&done.h_ptr, unroll_depth * sizeof(int32_t), 0);
        cudaHostGetDevicePointer(&done.d_ptr, done.h_ptr, 0);
    }

    // Initializer
    // methods

    void InitializeEncoder(cudaStream_t stream)
    {
        auto bound_func = [stream](auto *p_buf) {
                              CHECK_EQ(p_buf->fillWithZeroAsync(stream), cudaSuccess);
                          };
        for (auto ptr : {encoderPreHidden.get(), encoderPreCell.get(), encoderPostHidden.get(), encoderPostCell.get()}) {
            bound_func(ptr);
        }
    }

    void InitializeEncoderSparse(bool *currSparseMask, size_t actualBatchSize, [[maybe_unused]]size_t pol, cudaStream_t stream)
    {
        #ifdef CUDA_SPARSE_MEMSET
        // transfer the sparseMask to the device
        memcpyH2DAsync(*sparseMask[pol], currSparseMask, stream, actualBatchSize *sparseMask[pol]->bytesPerBatch());
        bool *pSparseMask = sparseMask[pol]->data();
        auto bound_func = [pSparseMask, actualBatchSize, stream](auto *p_buf) {
                              p_buf->cudaSparseMemsetAsync(pSparseMask, actualBatchSize, stream);
                          };
        #else
        // Apply initialization element by element (preserve content when currSparseMask is true)
        auto bound_func = [currSparseMask, actualBatchSize, stream](auto *p_buf) {
                              p_buf->slowSparseMemsetAsync(currSparseMask, actualBatchSize, stream);};
        #endif
        for (auto ptr: {encoderPreHidden.get(), encoderPreCell.get(), encoderPostHidden.get(), encoderPostCell.get()}) {
            bound_func(ptr);
        }
    }

    void InitializeDecoder(cudaStream_t stream)
    {
        // Some tensors require initialization before starting a new batch

        //    greedy symbol into tensor [BS][1][1]
        //    initialized with 0 so that the first input symbol to the decoder is SOS,
        //    which leads to zero embed vector.

        auto bound_func = [stream](auto *p_buf) {
                              CHECK_EQ(p_buf->fillWithZeroAsync(stream), cudaSuccess);
                          };
        // No range-for-loop because inconsistent types. (cudabufferRaw and cudabufferint32). Initializer list can't be heterogeneous
        // There's probably some way to do that, though
        bound_func(outSeqLen.get());
        bound_func(greedyWinner.get());
        bound_func(greedyHidden.get());
        bound_func(greedyCell.get());
    }

    // Sequence-splitting mindful initialize method

    void InitializeDecoderSparse([[maybe_unused]]bool *currSparseMask, size_t actualBatchSize, [[maybe_unused]]size_t pol, cudaStream_t stream)
    {
        // Some tensors require initialization before starting a new batch
       
        // outSeqLen does not require merging (TODO: check this)
        CHECK_EQ(outSeqLen->fillWithZeroAsync(stream), cudaSuccess);


        #ifdef CUDA_SPARSE_MEMSET
            // no need to transfer the currSparseMask (already in device)
            bool *pSparseMask = sparseMask[pol]->data();
            auto bound_func = [pSparseMask, actualBatchSize, stream] (auto *p_buf) {
                                  p_buf->cudaSparseMemsetAsync(pSparseMask, actualBatchSize, stream);
                              };
        #else
            // Apply initialization element by element (preserve content when currSparseMask is true)
            auto bound_func = [currSparseMask, actualBatchSize, stream] (auto *p_buf) {
                                  p_buf->slowSparseMemsetAsync(currSparseMask, actualBatchSize, stream);
                              };
        #endif
        // No rnage-for-loop because inconsistent types. (cudabufferRaw and cudabufferint32)
        // There's probably some way to do that, though
        bound_func(greedyWinner.get());
        bound_func(greedyHidden.get());
        bound_func(greedyCell.get());
    }

    void InitializeGreedy(size_t actualBatchSize, int32_t* hostLengthBuf, cudaStream_t stream)
    {
        // the greedy sequence length is the encoder sequence length divided by 2 (time compression)
        std::vector<int32_t>  tmpSeqLen(actualBatchSize);
        for (size_t bs = 0 ; bs < actualBatchSize; bs++) {
            tmpSeqLen[bs] = (hostLengthBuf[bs] + 1) / 2;
        }
        memcpyH2DAsync(*seqLen, tmpSeqLen.data(), stream, tmpSeqLen.size()*seqLen->bytesPerBatch());
               
        // initialize greedy structures:
        //    num_symbols_current_step : count of sequential predictions from the decoder (no time advance)
        //    encIdx                   : encoder time pointer/index

        // Maybe we can reduce work by using actualBatchSize, instead of full buffer? Maybe it doesn't matter?
        auto bound_func = [stream](auto *p_buf) {
                              CHECK_EQ(p_buf->fillWithZeroAsync(stream), cudaSuccess);
                          };
        for (auto ptr : {num_symbols_current_step.get(), encIdx.get()}) {
            bound_func(ptr);
        }

        #ifdef RNNT_STATS
        // stat collection variables
        std::cout << "RNNT_STATS:SEQ_LEN:";
        for (size_t bs = 0 ; bs < actualBatchSize; bs++) std::cout << hostLengthBuf[bs] << ','; // seqLen debug
        std::cout << std::endl; // seqLen debug
        #endif
    }

    void InitializeGreedySparse([[maybe_unused]]bool *currSparseMask, size_t actualBatchSize, int32_t* hostLengthBuf, [[maybe_unused]]size_t pol, cudaStream_t stream)
    {
        // the greedy sequence length is the encoder sequence lenght divided by 2 (time compression)
        std::vector<int32_t>  tmpSeqLen(actualBatchSize);  
        for (size_t bs = 0 ; bs < actualBatchSize; bs++) {
            tmpSeqLen[bs] = (hostLengthBuf[bs] + 1) / 2;
        }
        memcpyH2DAsync(*seqLen, tmpSeqLen.data(), stream, tmpSeqLen.size()*seqLen->bytesPerBatch());

        // initialize greedy structures:
        //    encIdx                   : encoder time pointer/index
        CHECK_EQ(encIdx->fillWithZeroAsync(stream), cudaSuccess);

        //    num_symbols_current_step : count of sequential predictions from the decoder (no time advance)
        //       this is state that may need to be preserved from previous passes/batches
        #ifdef CUDA_SPARSE_MEMSET
            // No need to transfer sparseMask (already in device)
            bool *pSparseMask = sparseMask[pol]->data();
            num_symbols_current_step->cudaSparseMemsetAsync(pSparseMask, actualBatchSize, stream);
        #else
            num_symbols_current_step->slowSparseMemsetAsync(currSparseMask, actualBatchSize, stream);
        #endif

        #ifdef RNNT_STATS
        // stat collection variables
        std::cout << "RNNT_STATS:SEQ_LEN:";
        for (size_t bs = 0 ; bs < actualBatchSize; bs++) std::cout << hostLengthBuf)[bs] << ','; // seqLen debug
        std::cout << std::endl; // seqLen debug
        // stat collection variables
        std::cout << "RNNT_STATS:MAX_SEQ_LEN:";
        size_t max_seq_len = *std::max_element(hostLengthBuf, hostLengthBuf+actualBatchSize);
        std::cout << max_seq_len << std::endl; // seqLen debug
        #endif

    }
    // Destructor
    ~RnntTensorContainer()
    {
        cudaFreeHost(done.h_ptr);
    }

};

// =============================
//     Engine
// =============================
//

// Generic engine runner.
class EngineRunner
{
private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::vector<std::shared_ptr<nvinfer1::IExecutionContext>> mContexts;
public:
    EngineRunner(const std::string& engineFile, size_t numContexts = 1) : mContexts(numContexts)
    {
        initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

        // Deserialize engine
        auto runtime = InferObject(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));
        size_t size{0};
        std::vector<char> buf;
        std::ifstream file(engineFile, std::ios::binary);
        CHECK(file.good());
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            buf.resize(size);
            file.read(buf.data(), size);
            file.close();
        }
        mEngine = InferObject(runtime->deserializeCudaEngine(buf.data(), size, nullptr));

        // Make sure the engine has enough optimization profiles, assuming the engine has dynamic shape (batch).
        CHECK_LE(numContexts, mEngine->getNbOptimizationProfiles());

        // Create execution contexts
        int profileIdx{0};
        for (auto& context : mContexts)
        {
            context = InferObject(mEngine->createExecutionContext());

            // Set optimization profile if necessary.
            CHECK_EQ(profileIdx < mEngine->getNbOptimizationProfiles(), true);
            if (context->getOptimizationProfile() < 0)
            {
                CHECK_EQ(context->setOptimizationProfile(profileIdx), true);
            }
            CHECK_EQ(context->getOptimizationProfile() == profileIdx, true);
            ++profileIdx;
        }
    }

    ~EngineRunner() = default;

    // Get number of bindings per optimization profile.
    size_t getNumBindings()
    {
        return mEngine->getNbBindings() / mEngine->getNbOptimizationProfiles();
    }

    // Get binding sizes of the engine. Return in (inputSizes, outputSizes).
    std::pair<std::vector<size_t>, std::vector<size_t>> getBindingSizes()
    {
        std::pair<std::vector<size_t>, std::vector<size_t>> bindingSizes;
        // Assuming the first dimension of the first input is batch dim.
        int maxBatchSize = mEngine->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0];
        int numBindings = getNumBindings();
        for (int i = 0; i < numBindings; i++)
        {
            size_t vol = volume(mEngine->getBindingDimensions(i), mEngine->getBindingFormat(i), mEngine->hasImplicitBatchDimension());
            size_t elementSize = getElementSize(mEngine->getBindingDataType(i));
            size_t allocationSize = static_cast<size_t>(maxBatchSize) * vol * elementSize;
            // std::cout << "getBindingSizes[" << i << "]  : vol=" << vol << " elementSize= " << elementSize << " allocationSize= " << allocationSize << std::endl;
            if (mEngine->bindingIsInput(i))
            {
                bindingSizes.first.push_back(allocationSize);
            }
            else
            {
                bindingSizes.second.push_back(allocationSize);
            }
        }
        return bindingSizes;
    }

    void enqueue(int contextIdx, int batchSize, void** bindings, cudaStream_t stream, cudaEvent_t* inputConsumed = nullptr)
    {
        // Get the corresponding context.
        CHECK_GE(contextIdx, 0);
        CHECK_LT(contextIdx, mContexts.size());
        auto& context = mContexts[contextIdx];

        // Set the first dimension to batch size. Assign bindings to the correct slots for current optimization profile.
        std::vector<void*> allBindings(mEngine->getNbBindings(), nullptr);
        int profileNum = context->getOptimizationProfile();
        CHECK_EQ(profileNum >= 0 && profileNum < mEngine->getNbOptimizationProfiles(), true);
        int numBindings = getNumBindings();
        for (int i = 0; i < numBindings; i++) {
            int bindingIdx = numBindings * profileNum + i;
            if (mEngine->bindingIsInput(i))
            {
                auto inputDims = context->getBindingDimensions(bindingIdx);
                inputDims.d[0] = batchSize;
                CHECK_EQ(context->setBindingDimensions(bindingIdx, inputDims), true);
            }
            allBindings[bindingIdx] = bindings[i];
        }
        CHECK_EQ(context->allInputDimensionsSpecified(), true);

        // Run inference asynchronously
        CHECK_EQ(context->enqueueV2(allBindings.data(), stream, inputConsumed), true);
    }

    size_t getElementDataSize(size_t idx) 
    {
        size_t elementSize = getElementSize(mEngine->getBindingDataType(idx));
        return elementSize;
    }
};

// =============================
//     Encoder runner
// =============================
//

class RnntEncoder
{
  private:
    size_t mBatchSize;
    std::unique_ptr<EngineRunner> mEncoder;

    #ifdef CUDA_EVENT_ENCODER
    cudaEvent_t eventStart, eventStop;
    #endif

  public:
    // constructor/destructor
    RnntEncoder(const std::string& engineDir, size_t batchSize)
        : mBatchSize(batchSize), mEncoder(new EngineRunner(engineDir + "/encoder.plan"))
    {
        // Cuda Events
        #ifdef CUDA_EVENT_ENCODER
        CHECK_EQ(cudaEventCreate(&eventStart), cudaSuccess);
        CHECK_EQ(cudaEventCreate(&eventStop), cudaSuccess);
        #endif
    }

    ~RnntEncoder()
    {
        // Cuda Events
        #ifdef CUDA_EVENT_ENCODER
        CHECK_EQ(cudaEventDestroy(eventStart), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(eventStop), cudaSuccess);
        #endif
    }

    // main method
    // void batchInfer(size_t batchSize, cudaStream_t stream, void* hostBuf, void* hostLengthBuf, size_t sampleSize,
    //     std::shared_ptr<CudaBufferRaw> encoderInputBuf, std::shared_ptr<CudaBufferRaw> encoderLengthsBuf, std::shared_ptr<CudaBufferRaw> encoderOutputBuf, size_t batchOffset=0)
    void batchInfer(size_t batchSize, cudaStream_t stream, uint8_t* hostBuf, int32_t* hostLengthBuf, size_t sampleSize, RnntTensorContainer &tc, size_t pol, size_t batchOffset=0)
    {
        #ifdef CUDA_EVENT_ENCODER
        CHECK_EQ(cudaEventRecord(eventStart, stream), cudaSuccess);
        #endif

        // pointer pre-processing
        size_t esize = getNativeDataSize();

        uint8_t *pHostBuf          = hostBuf       + (batchOffset * (mBatchSize * sampleSize));
        int32_t *pHostLengthBuf    = hostLengthBuf + (batchOffset * mBatchSize);
        // Check that indexing is right for !
        auto get_offset_ptr = [batch_idx=batchOffset](auto *p_buf) {
                                  return p_buf->get_ptr_from_idx(batch_idx);
                       };
        uint8_t *pEncoderOutputBuf = get_offset_ptr(tc.encoderOut[pol].get());
        // Null-initialize for louder failing
        void *pEncoderPreHidden = nullptr;
        void *pEncoderPostHidden = nullptr;
        void *pEncoderPreCell = nullptr;
        void *pEncoderPostCell = nullptr;
        if (FLAGS_seq_splitting) {
            pEncoderPreHidden = get_offset_ptr(tc.encoderPreHidden.get());
            pEncoderPostHidden= get_offset_ptr(tc.encoderPostHidden.get());
            pEncoderPreCell   = get_offset_ptr(tc.encoderPreCell.get());
            pEncoderPostCell  = get_offset_ptr(tc.encoderPostCell.get());
        }

        // Copy host input to Encoder input device buffer, assuming data is contiguous.
        bool dali_d2d = FLAGS_enable_audio_processing && FLAGS_seq_splitting;
        if (dali_d2d) {
            // If true, we've already done copied to the TRT bound buffers inside of processBatchStateIntoTensor, so we can proceed directly to inference.
            //gLogInfo << "No-Op" << std::endl;
            //std::cout << *tc.encoderInLengths << std::endl;
        } else {
            memcpyH2DAsync(*tc.encoderIn, pHostBuf, stream, batchSize * sampleSize);
            memcpyH2DAsync(*tc.encoderInLengths, pHostLengthBuf, stream, batchSize * sizeof(int32_t));
        }


        // Encoder binding and enqueue
        // if (false) 
        if (FLAGS_seq_splitting) 
        {
            // Hidden/cell of pre and post RNNs need to be captured in the bindings as inputs and outputs
            std::vector<void*> encoderBindings{
                tc.encoderIn->data(),        // in
                tc.encoderInLengths->data(),
                pEncoderPreHidden,
                pEncoderPostHidden,
                pEncoderPreCell,
                pEncoderPostCell,
                pEncoderPreHidden, // out
                pEncoderPreCell,
                pEncoderOutputBuf,
                pEncoderPostHidden,
                pEncoderPostCell
                };

            if (!FLAGS_disable_encoder_plugin)
            {
                encoderBindings.insert(encoderBindings.begin() + 2, pHostLengthBuf);
            }

            CHECK_EQ(encoderBindings.size(), mEncoder->getNumBindings());
            mEncoder->enqueue(0, batchSize, encoderBindings.data(), stream);
        }
        else 
        {
            assert(FLAGS_disable_encoder_plugin);
            // Prepare bindings for Encoder, which should have one input and one output.
            std::vector<void*> encoderBindings{tc.encoderIn->data(), tc.encoderInLengths->data(), pEncoderOutputBuf};
            CHECK_EQ(encoderBindings.size(), mEncoder->getNumBindings());
            mEncoder->enqueue(0, batchSize, encoderBindings.data(), stream);
        }

        #ifdef CUDA_EVENT_ENCODER
        CHECK_EQ(cudaEventRecord(eventStop, stream), cudaSuccess);
        float elapsed_time_ms;
        CHECK_EQ(cudaEventSynchronize(eventStop), cudaSuccess);
        CHECK_EQ(cudaEventElapsedTime(&elapsed_time_ms, eventStart, eventStop), cudaSuccess);
        gLogInfo << "CUDA_EVENT::encoder: " << elapsed_time_ms << std::endl;
        #endif
    }

    // Accessors
    size_t getNativeDataSize()
    {
        size_t elementSize = mEncoder->getElementDataSize(0);
        return elementSize;
    }
};

// =============================
//     Joint runner
// =============================
//

class RnntJoint
{
  private:
    size_t mBatchSize;
    std::unique_ptr<EngineRunner> mJoint;

    #ifdef CUDA_EVENT
    cudaEvent_t eventStart, eventStop;
    #endif

  public:
    RnntJoint(const std::string& engineDir, size_t batchSize)
        : mBatchSize(batchSize), mJoint(new EngineRunner(engineDir + "/joint.plan"))
    {
        // Cuda Events
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventCreate(&eventStart), cudaSuccess);
        CHECK_EQ(cudaEventCreate(&eventStop), cudaSuccess);
        #endif
    }

    ~RnntJoint()
    {
        // Cuda Events
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventDestroy(eventStart), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(eventStop), cudaSuccess);
        #endif
    }

    void step(size_t batchSize, cudaStream_t stream,
              CudaBufferRaw *jointEncInputBuf,
              CudaBufferRaw *jointDecInputBuf,
              CudaBufferInt32 *jointOutputBuf)
    {
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStart, stream), cudaSuccess);
        #endif

        // Prepare bindings for Joint, which should have two inputs and one output.
        std::vector<void*> jointBindings{jointEncInputBuf->data(), jointDecInputBuf->data(), jointOutputBuf->data()};
        CHECK_EQ(jointBindings.size(), mJoint->getNumBindings());
        mJoint->enqueue(0, batchSize, jointBindings.data(), stream);

        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStop, stream), cudaSuccess);
        float elapsed_time_ms;
        CHECK_EQ(cudaEventSynchronize(eventStop), cudaSuccess);
        CHECK_EQ(cudaEventElapsedTime(&elapsed_time_ms, eventStart, eventStop), cudaSuccess);
        gLogInfo << "CUDA_EVENT::joint: " << elapsed_time_ms << std::endl;
        #endif
    }
};


// ==================================
//    Disaggregated Joint runner
// ==================================
//

class RnntJointFc1
{
  private:
    size_t mBatchSize;
    std::unique_ptr<EngineRunner> mJointFc1;
    std::string uniqueName;

    #ifdef CUDA_EVENT
    cudaEvent_t eventStart, eventStop;
    #endif

  public:
    RnntJointFc1(const std::string& engineDir, const std::string& jointFc1Name, size_t batchSize)
        : mBatchSize(batchSize), mJointFc1(new EngineRunner(engineDir + "/" + jointFc1Name + ".plan"))
    {
        // std::string plan_file = engineDir + "/" + jointFc1Name + ".plan";
        // // mJointFc1 = new EngineRunner(engineDir + "/" + jointFc1Name + ".plan");
        // mJointFc1 = new EngineRunner(plan_file);
        uniqueName = jointFc1Name;

        // Cuda Events
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventCreate(&eventStart), cudaSuccess);
        CHECK_EQ(cudaEventCreate(&eventStop), cudaSuccess);
        #endif
    }

    ~RnntJointFc1()
    {
        // Cuda Events
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventDestroy(eventStart), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(eventStop), cudaSuccess);
        #endif
    }

    void step(size_t batchSize,
              cudaStream_t stream,
              CudaBufferRaw *jointInputBuf,
              CudaBufferRaw *jointOutputBuf)
    {
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStart, stream), cudaSuccess);
        #endif

        // Prepare bindings for Joint, which should have two inputs and one output.
        std::vector<void*> jointBindings{jointInputBuf->data(), jointOutputBuf->data()};
        CHECK_EQ(jointBindings.size(), mJointFc1->getNumBindings());
        mJointFc1->enqueue(0, batchSize, jointBindings.data(), stream);

        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStop, stream), cudaSuccess);
        float elapsed_time_ms;
        CHECK_EQ(cudaEventSynchronize(eventStop), cudaSuccess);
        CHECK_EQ(cudaEventElapsedTime(&elapsed_time_ms, eventStart, eventStop), cudaSuccess);
        gLogInfo << "CUDA_EVENT::" <<  uniqueName << ": " << elapsed_time_ms << std::endl;
        #endif
    }
};

class RnntJointBackend
{
  private:
    size_t mBatchSize;
    std::unique_ptr<EngineRunner> mJointBackend;

    #ifdef CUDA_EVENT
    cudaEvent_t eventStart, eventStop;
    #endif

  public:
    RnntJointBackend(const std::string& engineDir, size_t batchSize)
        : mBatchSize(batchSize), mJointBackend(new EngineRunner(engineDir + "/joint_backend.plan"))
    {
        // Cuda Events
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventCreate(&eventStart), cudaSuccess);
        CHECK_EQ(cudaEventCreate(&eventStop), cudaSuccess);
        #endif
    }

    ~RnntJointBackend()
    {
        // Cuda Events
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventDestroy(eventStart), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(eventStop), cudaSuccess);
        #endif
    }

    void step(size_t batchSize,
              cudaStream_t stream,
              CudaBufferRaw *jointEncFc1InputBuf,
              CudaBufferRaw *jointDecFc1InputBuf,
              CudaBufferInt32 *jointOutputBuf)
    {
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStart, stream), cudaSuccess);
        #endif

        // Prepare bindings for Joint, which should have two inputs and one output.
        std::vector<void*> jointBindings{jointEncFc1InputBuf->data(), jointDecFc1InputBuf->data(), jointOutputBuf->data()};
        CHECK_EQ(jointBindings.size(), mJointBackend->getNumBindings());
        mJointBackend->enqueue(0, batchSize, jointBindings.data(), stream);

        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStop, stream), cudaSuccess);
        float elapsed_time_ms;
        CHECK_EQ(cudaEventSynchronize(eventStop), cudaSuccess);
        CHECK_EQ(cudaEventElapsedTime(&elapsed_time_ms, eventStart, eventStop), cudaSuccess);
        gLogInfo << "CUDA_EVENT::joint: " << elapsed_time_ms << std::endl;
        #endif
    }
};






// =============================
//     Decoder runner
// =============================
//

class RnntDecoder
{
  private:
    size_t mBatchSize;
    std::unique_ptr<EngineRunner> mDecoder;

    #ifdef CUDA_EVENT
    cudaEvent_t eventStart, eventStop;
    #endif

  public:

    RnntDecoder(const std::string& engineDir, size_t batchSize)
        : mBatchSize(batchSize), mDecoder(new EngineRunner(engineDir + "/decoder.plan"))
    {
        // Cuda Events
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventCreate(&eventStart), cudaSuccess);
        CHECK_EQ(cudaEventCreate(&eventStop), cudaSuccess);
        #endif
    }

    ~RnntDecoder()
    {
        // Cuda Events
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventDestroy(eventStart), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(eventStop), cudaSuccess);
        #endif
    }

    void step(size_t batchSize,
              cudaStream_t stream,
              CudaBufferInt32 *decoderSymbolInputBuf,
              CudaBufferRaw *decoderHiddenInputBuf,
              CudaBufferRaw *decoderCellInputBuf,
              CudaBufferRaw *decoderSymbolOutputBuf,
              CudaBufferRaw *decoderHiddenOutputBuf,
              CudaBufferRaw *decoderCellOutputBuf)
    {
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStart, stream), cudaSuccess);
        #endif

        // Prepare bindings for Decoder, which should have two inputs and one output.
        std::vector<void*> decoderBindings{
            decoderSymbolInputBuf->data(),
            decoderHiddenInputBuf->data(),
            decoderCellInputBuf->data(),
            decoderSymbolOutputBuf->data(),
            decoderHiddenOutputBuf->data(),
            decoderCellOutputBuf->data()
        };

        CHECK_EQ(decoderBindings.size(), mDecoder->getNumBindings());
        mDecoder->enqueue(0, batchSize, decoderBindings.data(), stream);

        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStop, stream), cudaSuccess);
        float elapsed_time_ms;
        CHECK_EQ(cudaEventSynchronize(eventStop), cudaSuccess);
        CHECK_EQ(cudaEventElapsedTime(&elapsed_time_ms, eventStart, eventStop), cudaSuccess);
        gLogInfo << "CUDA_EVENT::decoder: " << elapsed_time_ms << std::endl;
        #endif
    }
};

// =============================
//     Isel runner
// =============================
//

class RnntIsel
{
  private:
    size_t mBatchSize;
    std::unique_ptr<EngineRunner> mIsel;

    #ifdef CUDA_EVENT
    cudaEvent_t eventStart, eventStop;
    #endif

  public:
    RnntIsel(const std::string& engineDir, size_t batchSize)
        : mBatchSize(batchSize), mIsel(new EngineRunner(engineDir + "/isel.plan"))
    {
        // Cuda Events
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventCreate(&eventStart), cudaSuccess);
        CHECK_EQ(cudaEventCreate(&eventStop), cudaSuccess);
        #endif
    }

    ~RnntIsel()
    {
        // Cuda Events
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventDestroy(eventStart), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(eventStop), cudaSuccess);
        #endif
    }

    void step(size_t batchSize,
              cudaStream_t stream,
              CudaBufferRaw *iselHiddenInput0Buf,
              CudaBufferRaw *iselCellInput0Buf,
              CudaBufferRaw *iselHiddenInput1Buf,
              CudaBufferRaw *iselCellInput1Buf,
              CudaBufferRaw *iselSelectInputBuf,
              CudaBufferRaw *iselHiddenOutputBuf,
              CudaBufferRaw *iselCellOutputBuf)
    {
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStart, stream), cudaSuccess);
        #endif

        // Prepare bindings for Isel, which should have two inputs and one output.
        std::vector<void*> iselBindings{
            iselHiddenInput0Buf->data(),
            iselCellInput0Buf->data(),
            iselHiddenInput1Buf->data(),
            iselCellInput1Buf->data(),
            iselSelectInputBuf->data(),
            iselHiddenOutputBuf->data(),
            iselCellOutputBuf->data()
        };

        CHECK_EQ(iselBindings.size(), mIsel->getNumBindings());
        mIsel->enqueue(0, batchSize, iselBindings.data(), stream);

        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStop, stream), cudaSuccess);
        float elapsed_time_ms;
        CHECK_EQ(cudaEventSynchronize(eventStop), cudaSuccess);
        CHECK_EQ(cudaEventElapsedTime(&elapsed_time_ms, eventStart, eventStop), cudaSuccess);
        gLogInfo << "CUDA_EVENT::isel: " << elapsed_time_ms << std::endl;
        #endif
    }

    void step_3way(size_t batchSize, cudaStream_t stream,
                   CudaBufferRaw *iselHiddenInput0Buf,
                   CudaBufferRaw *iselCellInput0Buf,
                   CudaBufferInt32 *iselWinnerInput0Buf,
                   CudaBufferRaw *iselHiddenInput1Buf,
                   CudaBufferRaw *iselCellInput1Buf,
                   CudaBufferInt32 *iselWinnerInput1Buf,
                   CudaBufferBool *iselSelectInputBuf,
                   CudaBufferRaw *iselHiddenOutputBuf,
                   CudaBufferRaw *iselCellOutputBuf,
                   CudaBufferInt32 *iselWinnerOutputBuf)
        {
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStart, stream), cudaSuccess);
        #endif

        // Prepare bindings for Isel, which should have two inputs and one output.
        std::vector<void*> iselBindings{
            iselHiddenInput0Buf->data(),
            iselCellInput0Buf->data(),
            iselWinnerInput0Buf->data(),
            iselHiddenInput1Buf->data(),
            iselCellInput1Buf->data(),
            iselWinnerInput1Buf->data(),
            iselSelectInputBuf->data(),
            iselHiddenOutputBuf->data(),
            iselCellOutputBuf->data(),
            iselWinnerOutputBuf->data()
        };

        CHECK_EQ(iselBindings.size(), mIsel->getNumBindings());
        mIsel->enqueue(0, batchSize, iselBindings.data(), stream);

        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStop, stream), cudaSuccess);
        float elapsed_time_ms;
        CHECK_EQ(cudaEventSynchronize(eventStop), cudaSuccess);
        CHECK_EQ(cudaEventElapsedTime(&elapsed_time_ms, eventStart, eventStop), cudaSuccess);
        gLogInfo << "CUDA_EVENT::isel: " << elapsed_time_ms << std::endl;
        #endif
    }
};


// =============================
//     Igather runner
// =============================
//

class RnntIgather
{
  private:
    size_t mBatchSize;
    std::unique_ptr<EngineRunner> mIgather;

    #ifdef CUDA_EVENT
    cudaEvent_t eventStart, eventStop;
    #endif

  public:
    RnntIgather(const std::string& engineDir, size_t batchSize)
        : mBatchSize(batchSize), mIgather(new EngineRunner(engineDir + "/igather.plan"))
    {
        // Cuda Events
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventCreate(&eventStart), cudaSuccess);
        CHECK_EQ(cudaEventCreate(&eventStop), cudaSuccess);
        #endif
    }

    ~RnntIgather()
    {
        // Cuda Events
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventDestroy(eventStart), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(eventStop), cudaSuccess);
        #endif
    }

    void step(size_t batchSize,
              cudaStream_t stream,
              CudaBufferRaw *igatherEncoderBuf,
              CudaBufferInt32 *igatherCoordBuf,
              CudaBufferRaw *igatherOutputBuf)
    {
        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStart, stream), cudaSuccess);
        #endif

        // Prepare bindings for Igather, which should have two inputs and one output.
        std::vector<void*> igatherBindings{
            igatherEncoderBuf->data(),
            igatherCoordBuf->data(),
            igatherOutputBuf->data()
        };

        CHECK_EQ(igatherBindings.size(), mIgather->getNumBindings());
        mIgather->enqueue(0, batchSize, igatherBindings.data(), stream);

        #ifdef CUDA_EVENT
        CHECK_EQ(cudaEventRecord(eventStop, stream), cudaSuccess);
        float elapsed_time_ms;
        CHECK_EQ(cudaEventSynchronize(eventStop), cudaSuccess);
        CHECK_EQ(cudaEventElapsedTime(&elapsed_time_ms, eventStart, eventStop), cudaSuccess);
        gLogInfo << "CUDA_EVENT::igather: " << elapsed_time_ms << std::endl;
        #endif
    }
};

// =============================
//     Sequence
// =============================

// A simply container for the symbol sequence output
//

class sequenceClass {
    public:
        std::vector<uint32_t> data;    // sequence data

        // constructor/destructor
        sequenceClass() { }
        ~sequenceClass() { }

        // main methods/accessors
        inline size_t last() { return data.back(); }
        inline size_t size() { return data.size(); }
        inline size_t operator [](size_t index) { return data[index]; }

        bool push(size_t symbol)
        {
            if(symbol == _BLANK_) return false;
            data.push_back(symbol);
            return true;
        }

        void print(const char* prefix="")
        {
            std::cout << prefix << ": " << to_string() << std::endl;
        }

        std::string to_string()
        {
            char labels[29] = { ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\'', '*'};
            std::string output(size(), '?');
            for (size_t c = 0; c < size(); c++) {
                uint32_t index = data.at(c);
                if (index < std::size(labels)) {
                    output[c] = labels[index];
                }
            }
            return output;
        }

        std::vector<RETURN_T> to_data()
        {
            std::vector<RETURN_T> return_data; 
            for (auto in_elem : data) return_data.push_back(in_elem);
            return return_data;
        }
};

// =============================
//     Engine container
// =============================
//

// For convenience, we pack all the runners into a single struct
class RnntEngineContainer {
    public:
        // core engines
        std::unique_ptr<RnntEncoder> encoder;
        std::unique_ptr<RnntDecoder> decoder;
        std::unique_ptr<RnntJoint>   joint;
        std::unique_ptr<RnntIsel>    isel;
        std::unique_ptr<RnntIgather> igather;

        // disaggregated joint
        std::unique_ptr<RnntJointFc1> jointFc1a;
        std::unique_ptr<RnntJointFc1> jointFc1b;
        std::unique_ptr<RnntJointBackend> jointBackend;

    // constructor
    RnntEngineContainer() =delete;
    RnntEngineContainer(const std::string& engineDir, const size_t batchSize) 
    {
        // Create the encoder runner.
        size_t  enc_batch_size = FLAGS_batch_size;
        if ( FLAGS_encoder_batch_size && (FLAGS_encoder_batch_size < enc_batch_size) ) {
            enc_batch_size = FLAGS_encoder_batch_size;  // sub-batch size override
        }
        encoder.reset(new RnntEncoder(engineDir, enc_batch_size));
        gLogInfo << "Created RnntEncoder runner: encoder" << std::endl;

        // Create the decoder runner.
        decoder.reset(new RnntDecoder(engineDir, batchSize));
        gLogInfo << "Created RnntDecoder runner: decoder" << std::endl;

        // Create the joint runner.
        if(FLAGS_disaggregated_joint) {
            jointFc1a.reset(new RnntJointFc1(engineDir, "fc1_a", batchSize));
            gLogInfo << "Created RnntJointFc1 runner: fc1_a" << std::endl;
            jointFc1b.reset(new RnntJointFc1(engineDir, "fc1_b", batchSize));
            gLogInfo << "Created RnntJointFc1 runner: fc1_b" << std::endl;
            jointBackend.reset(new RnntJointBackend(engineDir, batchSize));
            gLogInfo << "Created RnntJointBackend runner: joint_backend" << std::endl;
        } else {
            joint.reset(new RnntJoint(engineDir, batchSize));
            gLogInfo << "Created RnntJoint runner: joint" << std::endl;
        }

        // Create the Isel & Igather engine
        isel.reset(new RnntIsel(engineDir, batchSize));
        gLogInfo << "Created RnntIsel runner: isel" << std::endl;

        igather.reset(new RnntIgather(engineDir, batchSize));
        gLogInfo << "Created RnntIgather runner: igather" << std::endl;
    }
    // destructor
    ~RnntEngineContainer() {}
};


// =============================
//     Thread synch support
// =============================

void notify_thread(std::atomic<size_t> &credits, std::mutex &mt, std::condition_variable &cv, const std::string& header, size_t pol, size_t iCnt)
{
    if (FLAGS_debug_mode) gLogInfo << header <<  "pol=" << pol <<  " icnt=" << iCnt << " | cr=" << (credits+1) << std::endl;
    // go = true;
    // credits++;
    credits.fetch_add(1);
    std::lock_guard<std::mutex> lkg(mt);
    cv.notify_one();
}

void wait_for_thread(bool &allDone, std::atomic<size_t> &credits, std::mutex &mt, std::condition_variable &cv, const std::string& header, size_t pol, size_t iCnt)
{
    if (FLAGS_debug_mode) gLogInfo << header  << "pol=" << pol << " icnt=" << iCnt << ": waiting... " << std::endl;
    while(!credits && !allDone) 
    {
        std::unique_lock<std::mutex> lk(mt);
        cv.wait(lk, []{return true;});
        lk.unlock();
    }
    // go = false;
    // credits--;
    credits.fetch_add(-1);
    if (FLAGS_debug_mode) gLogInfo <<  header  << "icnt=" << iCnt << " wake up | cr=" << credits << std::endl;
}



// =============================
//     Inference algorithms
// =============================
//

// Encoder
//
//     Separate method/wrapper for the encoder phase
void doBatchEncoder (
    RnntEncoder  &encoder,
    size_t       actualBatchSize,
    cudaStream_t stream,
    uint8_t*     hostBuf,
    int32_t*     hostLengthBuf,
    size_t       sampleSize,
    size_t       pol,
    RnntTensorContainer &tc)
{
    if (FLAGS_debug_mode) gLogInfo << "Running encoder batchSize " << actualBatchSize <<  "..." << std::endl;

    if (FLAGS_encoder_batch_size == 0 || (FLAGS_encoder_batch_size>= FLAGS_batch_size)) 
    {
        // Run the encoder for the whole batch
        // encoder.batchInfer(actualBatchSize, stream, hostBuf, hostLengthBuf, sampleSize, tc.encoderIn, tc.encoderInLengths, tc.encoderOut[pol]);
        encoder.batchInfer(actualBatchSize, stream, hostBuf, hostLengthBuf, sampleSize, tc, pol);
    } 
    else 
    {
        // Run the encoder in sub-batches
        for(size_t batchOffset = 0, batchCount = 0; batchCount < actualBatchSize; batchOffset++, batchCount +=  FLAGS_encoder_batch_size)
        {
            // Run the encoder for the sub batch
            size_t actualSubBatchSize = std::min(FLAGS_encoder_batch_size, actualBatchSize - batchCount);
            if (FLAGS_debug_mode) gLogInfo << "Running encoder sub batch " << batchOffset <<  " with subBatchSize " << actualSubBatchSize << "..." << std::endl;
            // encoder.batchInfer(actualSubBatchSize, stream, hostBuf, hostLengthBuf, sampleSize, tc.encoderIn, tc.encoderInLengths, tc.encoderOut[pol], batchOffset);
            encoder.batchInfer(actualSubBatchSize, stream, hostBuf, hostLengthBuf, sampleSize, tc, pol, batchOffset);

            if (FLAGS_verbose) gLogInfo << "Ran encoder sub batch " << batchOffset <<  "..." << std::endl;
        }
    } 

    if (FLAGS_verbose) gLogInfo << "Ran encoder..." << std::endl;
}

// Decoder (single iteration)
//
//     Separate method/wrapper for the decoder phase (single iteration)
void doBatchDecoderIteration(
    size_t       iter,
    int32_t      *done_d,
    size_t       actualBatchSize,
    RnntEngineContainer &ec,
    RnntTensorContainer &tc,
    cudaEvent_t  gatherDecoderEvent,  // encGather | decoder
    cudaStream_t decoderStream,
    cudaEvent_t  jointFc1Event,       // fc1a | fc1b
    cudaStream_t jointFc1bStream,
    size_t       pol,
    cudaStream_t mainStream)
{
    // gLogInfo << "   [doBatchDecoderIteration] bs= " << actualBatchSize << " pol=" << pol << " iter= " << iter << std::endl;

    // synchronize decoder
    CHECK_EQ(cudaEventRecord(gatherDecoderEvent, mainStream), cudaSuccess); 
    CHECK_EQ(cudaStreamWaitEvent(decoderStream, gatherDecoderEvent, 0), cudaSuccess);           

    // invoke the predictor (decoder)
    ec.decoder->step(actualBatchSize, decoderStream,
                     tc.greedyWinner.get(),
                     tc.greedyHidden.get(),
                     tc.greedyCell.get(),
                     tc.decoderSymbol.get(),
                     tc.decoderHidden.get(),
                     tc.decoderCell.get());
                
    // transfer the vector of t coordinates to the igather layer
    #if 1
        ec.igather->step(actualBatchSize, mainStream,
                         tc.encoderOut[pol].get(),
                         tc.encIdx.get(),
                         tc.encGather.get());
    #else 
        rnntIgatherStep(
            tc.encoderOut[pol]->data(),
            tc.encIdx->data(),
            tc.encGather->data(),
                 tc.esize,
                 FLAGS_hp_encoder_hidden_size,
                 (FLAGS_hp_max_seq_length >> 1),  // mind the time compression
                 actualBatchSize,
                 mainStream);
    #endif

    // Optimization trick: we can start encoder fc1 right after gather finishes (in parallel with the decoder)
    if(FLAGS_disaggregated_joint) {
        ec.jointFc1a->step(actualBatchSize, mainStream, tc.encGather.get(), tc.jointFc1Encoder.get());
    }
            
    cudaEventRecord(gatherDecoderEvent, decoderStream);
    cudaStreamWaitEvent(mainStream, gatherDecoderEvent, 0);       
            
    // invoke the joint network
    if(FLAGS_disaggregated_joint) {
        #if 0
        ec.jointFc1b->step(actualBatchSize, mainStream, tc.decoderSymbol.get(), tc.jointFc1Decoder.get());
            ec.jointBackend->step(actualBatchSize, mainStream, tc.jointFc1Encoder.get(), tc.jointFc1Decoder.get(), tc.jointSymbol.get());
        #else
            // synchronize jointFc1bStream
            CHECK_EQ(cudaEventRecord(jointFc1Event, mainStream), cudaSuccess); 
            CHECK_EQ(cudaStreamWaitEvent(jointFc1bStream, jointFc1Event, 0), cudaSuccess);           
            
            // run concurrenly 
            ec.jointFc1b->step(actualBatchSize,
                               jointFc1bStream,
                               tc.decoderSymbol.get(),
                               tc.jointFc1Decoder.get());
    
            // join both streams 
            CHECK_EQ(cudaEventRecord(jointFc1Event, jointFc1bStream), cudaSuccess);
            CHECK_EQ(cudaStreamWaitEvent(mainStream, jointFc1Event, 0), cudaSuccess);       
    
            // run backend
            if(FLAGS_cuda_joint_backend) {
                // cuda optimized version
                rnntFc2Top1(tc.jointFc1Encoder->data(),
                            tc.jointFc1Decoder->data(),
                            tc.jointFc2Weights->data(),
                            tc.jointFc2Bias->data(),
                            tc.jointFc2Output->data(),
                            tc.jointSymbol->data(),
                            actualBatchSize,
                            mainStream);
            } else {
                // reference TRT version
                ec.jointBackend->step(actualBatchSize, mainStream,
                                      tc.jointFc1Encoder.get(),
                                      tc.jointFc1Decoder.get(),
                                      tc.jointSymbol.get());
            }
        #endif
    } else {
        ec.joint->step(actualBatchSize, mainStream, tc.encGather.get(), tc.decoderSymbol.get(), tc.jointSymbol.get());
    }
    
    // So long as we're careful about the done pointer we don't have to sync before this.

    greedySearch(tc.jointSymbol->data(),
                 tc.num_symbols_current_step->data(),
                 tc.encIdx->data(),
                 tc.seqLen->data(),
                 tc.isNotBlank->data(),
                 tc.outSeq->data(),
                 tc.outSeqLen->data(),
                 done_d,
                 actualBatchSize,
                 iter,
                 _BLANK_,
                 FLAGS_hp_max_symbols_per_step,
                 FLAGS_hp_max_seq_length,
                 mainStream); 
                 
    ec.isel->step_3way(actualBatchSize, mainStream,
                       tc.decoderHidden.get(), tc.decoderCell.get(), tc.jointSymbol.get(),  // input0
                       tc.greedyHidden.get(), tc.greedyCell.get(), tc.greedyWinner.get(),   // input1
                       tc.isNotBlank.get(),                                                 // select
                       tc.greedyHidden.get(), tc.greedyCell.get(), tc.greedyWinner.get()    // output
        ); 
}

// Greedy search with optimized engine and cuda graphs: encoder front-end
// 
void doBatchInferenceEncoder(
    uint8_t*            hostBuf,       // Assuming contiguous
    int32_t*            hostLengthBuf, // Assuming contiguous
    size_t              actualSampleSize,
    size_t              actualBatchSize,
    RnntEngineContainer &ec,
    RnntTensorContainer &tc,
    bool*               stateMask,    // determines when to preserve state across invocations
    size_t              pol,
    cudaStream_t stream)
{
    // gLogInfo << "[doBatchInferenceEncoder] bs= " << actualBatchSize << " pol=" << pol << std::endl;

    // Reset tensors
    if(FLAGS_seq_splitting) {
        tc.InitializeEncoderSparse(stateMask, actualBatchSize, pol, stream);  
    } else {
        // FIXME: not actually needed 
        //tc.InitializeEncoder(stream);
    }

    // Run the encoder (may have several iterations) for the whole batch
    doBatchEncoder(*ec.encoder, actualBatchSize, stream, hostBuf, hostLengthBuf, actualSampleSize, pol, tc);
}


void batchInfDecoderInitializeTensors(
    int32_t*               hostLengthBuf, // Assuming contiguous
    size_t              actualBatchSize,
    RnntTensorContainer &tc,
    bool*               stateMask,    // determines when to preserve state across invocations
    size_t              pol,
    cudaStream_t stream)
{
    // Initialize tensors
    if(FLAGS_seq_splitting) {
        tc.InitializeDecoderSparse(stateMask, actualBatchSize, pol, stream);
        tc.InitializeGreedySparse(stateMask, actualBatchSize, hostLengthBuf, pol, stream);
    } else {
        tc.InitializeDecoder(stream);
        tc.InitializeGreedy(actualBatchSize, hostLengthBuf, stream);
    }
}


CudaGraphPair makeDecoderGraph(size_t requested_batch_size, RnntEngineContainer &ec, RnntTensorContainer& tc, size_t polarity, cudaStream_t stream) {
    // Create new stream for decoder and an event to synchronize with encoderGather
    cudaEvent_t gatherDecoderEvent;
    cudaStream_t decoderStream;
    cudaEventCreate(&gatherDecoderEvent, cudaEventDisableTiming);
    cudaStreamCreate(&decoderStream);

    // create new event and stream to allow FC1 concurrent execution
    cudaEvent_t jointFc1Event;
    cudaStream_t jointFc1bStream;
    cudaEventCreate(&jointFc1Event, cudaEventDisableTiming);
    cudaStreamCreate(&jointFc1bStream);

    auto unroll_depth = FLAGS_cuda_graph_unroll;

    // pre-run to prevent the cuda graph from capturing the recomputeResource(), thereby avoiding extra D2D copies during runtime
    for (int i = 0; i < unroll_depth; i++) {
        doBatchDecoderIteration(i, tc.done.d_ptr, requested_batch_size, ec, tc,
                                gatherDecoderEvent, decoderStream,
                                jointFc1Event, jointFc1bStream,
                                polarity, stream);
    }
    cudaGraph_t tempGraph;
    cudaGraphExec_t tempExecGraph;
    CHECK_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), cudaSuccess);
    for (int i = 0; i < unroll_depth; i++) {
        doBatchDecoderIteration(i, tc.done.d_ptr, requested_batch_size, ec, tc,
                                gatherDecoderEvent, decoderStream,
                                jointFc1Event, jointFc1bStream,
                                polarity, stream);
    }
    CHECK_EQ(cudaStreamEndCapture(stream, &tempGraph), cudaSuccess);
    CHECK_EQ(cudaGraphInstantiate(&tempExecGraph, tempGraph, NULL, NULL, 0), cudaSuccess);
    return {tempGraph, tempExecGraph};
}

// Greedy search with optimized engine and cuda graphs: decoder back-end
// 
std::vector<std::vector<RETURN_T>> doBatchInferenceDecoder(
    size_t              actualBatchSize,
    RnntEngineContainer &ec,
    RnntTensorContainer &tc,
    size_t              pol,
    cudaStream_t stream)
{
    //  Greedy loop
    //
    // gLogInfo << "[doBatchInferenceDecoder] bs= " << actualBatchSize << " pol=" << pol << std::endl;

    // Create host resident structures
    std::vector<sequenceClass> outSeq(actualBatchSize); // Output sequences
    int32_t ut_iter = 0;

    // Determine how much we unroll
    auto unroll_depth = FLAGS_cuda_graph_unroll;

    // Create new stream for decoder and an event to synchronize with encoderGather 
    cudaEvent_t gatherDecoderEvent;
    cudaStream_t decoderStream;
    cudaEventCreate(&gatherDecoderEvent, cudaEventDisableTiming);
    cudaStreamCreate(&decoderStream);

    // create new event and stream to allow FC1 concurrent execution
    cudaEvent_t jointFc1Event;
    cudaStream_t jointFc1bStream;
    cudaEventCreate(&jointFc1Event, cudaEventDisableTiming);
    cudaStreamCreate(&jointFc1bStream);

    // Graph instantiation
    cudaGraphExec_t activeGraph;
    if (FLAGS_cuda_graph) {
        if (FLAGS_cuda_legacy_graph_caching) {
            // Bind callback parameters
            auto miss_callback = [&](size_t bsParam) {return makeDecoderGraph(bsParam, ec, tc,  pol, stream);};
            activeGraph = tc.cg_cache[pol].get(actualBatchSize, miss_callback);
        } else {
            activeGraph = tc.cg_cache[pol].get(actualBatchSize);
        }

    }

    // Main time loop:
    //

    // initialize local variables for the loop
    bool are_we_there_yet = false;     // we are finished when all the batch instances reach their sequence size
    std::fill(tc.done.h_ptr, tc.done.h_ptr+unroll_depth, true);

    // profiling
    #ifdef CUDA_EVENT_DECODER
    cudaEvent_t tEventStart, tEventStop;
    CHECK_EQ(cudaEventCreate(&tEventStart), cudaSuccess);
    CHECK_EQ(cudaEventCreate(&tEventStop), cudaSuccess);
    CHECK_EQ(cudaEventRecord(tEventStart, stream), cudaSuccess);
    #endif
    
    // synchronize and run
    CHECK_EQ(cudaStreamSynchronize(stream), cudaSuccess); 
    if (FLAGS_verbose) gLogInfo << "Running greedy loop " << std::endl;

    while(!are_we_there_yet)
    {
        if (FLAGS_debug_mode) gLogInfo << "Running ut iteration = " << ut_iter << std::endl;

        if (FLAGS_cuda_graph) {
            cudaGraphLaunch(activeGraph, stream);
        } else {
            for (auto unroll_idx = 0; unroll_idx < unroll_depth; ++unroll_idx) {
                doBatchDecoderIteration(unroll_idx,
                                        tc.done.d_ptr,
                                        actualBatchSize,
                                        ec,
                                        tc,
                                        gatherDecoderEvent,
                                        decoderStream,
                                        jointFc1Event,
                                        jointFc1bStream,
                                        pol,
                                        stream);
            }
        }

        CHECK_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        //printf("%d %d\n", tc.done.h_ptr[0], tc.done.h_ptr[1]);

        for(auto unroll =0 ; unroll < unroll_depth; unroll++) {
            if (tc.done.h_ptr[unroll]) {
                are_we_there_yet = true;
            }
            tc.done.h_ptr[unroll] = true;
            ut_iter++;
        }
    }

    // RNN-t stats: ut iteration profiling
    #ifdef RNNT_STATS
    // get UT vs T Inflation
    float ut_inflation = (float) ut_iter / (float) (FLAGS_hp_max_seq_length >> 1);
    // std::cout << "RNNT_STATS:UT_INFLATION:\t" << ut_inflation << std::endl;
    std::cout << "RNNT_STATS:UT_ITER:\t" << ut_iter << std::endl;
    #endif

    // Profiling
    #ifdef CUDA_EVENT_DECODER
    CHECK_EQ(cudaEventRecord(tEventStop, stream), cudaSuccess);
    float elapsed_time_ms;
    CHECK_EQ(cudaEventSynchronize(tEventStop), cudaSuccess);
    CHECK_EQ(cudaEventElapsedTime(&elapsed_time_ms, tEventStart, tEventStop), cudaSuccess);
    gLogInfo << "CUDA_EVENT::decoder: " << elapsed_time_ms << std::endl;
    #endif

    
    // Transfer greedy seqLen to host
    memcpyD2HAsync(*tc.host.outSeqLen, *tc.outSeqLen, stream, actualBatchSize * tc.outSeqLen->bytesPerBatch());
    cudaStreamSynchronize(stream);
           
    // Obtain max sequence length 
    int maxOutSeqLen = 0;

    auto outSeqLenPtr = tc.host.outSeqLen->data();
    // Could just do std::max_element, but per-element printf is a nice debug scream for now
    for (size_t bs = 0; bs < actualBatchSize; bs++) {
        if (outSeqLenPtr[bs] >= FLAGS_hp_max_seq_length) {
            printf("Output sequence too long %d/%d\n", outSeqLenPtr[bs], FLAGS_hp_max_seq_length);
        }
        if (outSeqLenPtr[bs] > maxOutSeqLen) {
            maxOutSeqLen = outSeqLenPtr[bs];
        }        
    }

    // Transfer greedy sequences to host
    // CHECK_EQ(cudaMemcpy2DAsync((void*)(tc.host.outSeqTmp->data()),
    //        FLAGS_hp_max_seq_length *  sizeof(int32_t),
    //        (void*) tc.outSeq->data(),
    //        FLAGS_hp_max_seq_length *  sizeof(int32_t),
    //        maxOutSeqLen *  sizeof(int32_t),
    //        actualBatchSize,
    //        cudaMemcpyDeviceToHost,
    //        stream), cudaSuccess);
    // cudaStreamSynchronize(stream);
    // Instead of the above, we do this (copies everything) TODO: is this okay?
    memcpyD2HAsync(*tc.host.outSeqTmp, *tc.outSeq, stream);
    cudaStreamSynchronize(stream);
    // push to output structures
    auto outSeqPtr = tc.host.outSeqTmp->data();
    for (size_t bs = 0; bs < actualBatchSize; bs++) {
        for (int i = 0; i < outSeqLenPtr[bs]; i++) {
            outSeq[bs].push(outSeqPtr[bs * FLAGS_hp_max_seq_length + i]);
        }
    }

    // Debug: show contents of the sequences
    if(FLAGS_verbose) {
        for (size_t bs = 0; bs < actualBatchSize ; bs++)
        {
            std::ostringstream prefix;
            prefix << " BS idx"
                   << "[" << bs << "] "
                   << "text sequence: ";
            outSeq[bs].print(prefix.str().c_str());
        }
    }

    // Record results in accuracy mode
    std::vector<std::vector<RETURN_T>> results;
    results.resize(actualBatchSize);
    for (size_t bs = 0; bs < actualBatchSize ; bs++)
    {
        // results[bs] = outSeq[bs].to_string();
        results[bs] = outSeq[bs].to_data();
    }

    // Structure/primitive clean up
    cudaStreamDestroy(decoderStream);
    cudaEventDestroy(gatherDecoderEvent);
    cudaStreamDestroy(jointFc1bStream);
    cudaEventDestroy(jointFc1Event);

    #ifdef CUDA_EVENT_DECODER
    CHECK_EQ(cudaEventDestroy(tEventStart), cudaSuccess);
    CHECK_EQ(cudaEventDestroy(tEventStop), cudaSuccess);
    #endif
    
    // Epilogue
    return results;
}


// =============================
//     Stream
// =============================
//

class Stream
{
public:
    Stream(
        const int deviceId,
        std::shared_ptr<qsl::SampleLibrary> qsl,
        std::shared_ptr<WarmupSampleLibrary> wqsl,
        std::shared_ptr<SyncWorkQueue> workQueue,
        std::shared_ptr<AudioBufferManagement> audioBufManager,
        const std::string& engineDir,
        const size_t batchSize);
    Stream(const Stream&) = delete;
    ~Stream();
    void launchThread();
    void joinThread();
    void setWarmup(bool off_on);

private:
    // tensor input generator
    void processBatchStateIntoTensor(uint8_t* &, int32_t* &, uint8_t*, int32_t*, size_t , size_t);
    
    // single batch components
    bool frontEndBatchPrologue(size_t,  size_t&);
    void frontEndBatch(size_t); 
    void backEndBatchPrologue(size_t);
    void backEndBatch(size_t);

    // advance processBatches
    void processBatches();
    void processBatchesFrontEnd();
    void processBatchesBackEnd();

    // internal structures
    bool mDone{false};              // shared finished indication
    int mDeviceId{-1};
    size_t mBatchSize;
    std::shared_ptr<qsl::SampleLibrary> mQsl;
    std::shared_ptr<WarmupSampleLibrary> mWqsl;
    qsl::LookupableQuerySampleLibrary *mActiveQsl;
    bool is_warmup{false};

    std::shared_ptr<AudioBufferManagement> mAudioBufManager;
    std::shared_ptr<SyncWorkQueue> mWorkQueue;


    // threads
    std::thread mThProcessSample;     // single thread process
    std::thread mThProcessSampleFe;   // front-end process (pipelined execution)
    std::thread mThProcessSampleBe;   // back-end process (pipelined execution)

    // fe/be synchronization
    // bool decoderGo{false};
    // bool encoderGo{false};
    // size_t decoderCredits{0};
    // size_t encoderCredits{1}; // encoder needs a look-ahead of 1
    std::atomic<size_t> decoderCredits{0};
    std::atomic<size_t> encoderCredits{1}; // encoder needs a look-ahead of 1

    std::mutex mutex_f2b;
    std::condition_variable cv_f2b;
    std::mutex mutex_b2f;
    std::condition_variable cv_b2f;

    // gpu-wise thread communication
    cudaEvent_t cudaEventEncToDec;

    // execution streams (frontend/backend)
    cudaStream_t mStream[PIPE_DEPTH];

    // tensor container
    std::unique_ptr<RnntTensorContainer> mTc;

    EncoderState mEncoderState;
    DecoderState mDecoderState;

    // engine/runner container
    std::unique_ptr<RnntEngineContainer> mEc;

    // sample sizes needed for data structures;
    size_t singleSampleSize;  // size of a single encoder sample (internal tensor)
    size_t fullseqSampleSize;  // size of a encoder sequence (internal tensor)

    // shared values across (polarity)
    size_t   actualBatchSize[PIPE_DEPTH];
    uint8_t *hostBufStaging;
    int32_t *hostLengthBufStaging[PIPE_DEPTH];

    dali::kernels::ScatterGatherGPU *mScatterGatherD2HData;
    dali::kernels::ScatterGatherGPU *mScatterGatherD2DData;
};

Stream::Stream(
    const int deviceId,
    std::shared_ptr<qsl::SampleLibrary> qsl,
    std::shared_ptr<WarmupSampleLibrary> wqsl,
    std::shared_ptr<SyncWorkQueue> workQueue,
    std::shared_ptr<AudioBufferManagement> audioBufManager,
    const std::string& engineDir,
    const size_t batchSize)
    : mDeviceId(deviceId)
    , mBatchSize(batchSize)
    , mQsl(qsl)
    , mWqsl(wqsl)
    , mActiveQsl(qsl.get())
    , mWorkQueue(workQueue)
    , mEncoderState(batchSize, FLAGS_hp_max_seq_length)
    , mDecoderState(batchSize)
    , mAudioBufManager(audioBufManager)
{
    // we want higher priority for decoder stream
    int leastPriority, greatestPriority;
    CHECK_EQ(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority),cudaSuccess);
    // Create a single private cuda stream for each pipeline stage
    for (int i = 0; i < PIPE_DEPTH; i++) {
        int priority;
        switch (i) {
        case RNNT_ENC:
            priority = leastPriority;
            break;
        case RNNT_DEC:
            priority = greatestPriority;
            break;
        default:
            gLogInfo << "Stream::Stream error! Depth iteration out of range" << std::endl;
            CHECK(false);
        }
        CHECK_EQ(cudaStreamCreateWithPriority(&mStream[i], cudaStreamDefault, priority), cudaSuccess);
        //CHECK_EQ(cudaStreamCreate(&mStream[i]), cudaSuccess);

        // create inter-thread specific events
        cudaEventCreate(&cudaEventEncToDec, cudaEventDisableTiming);
    }

    // Create a container for the engines
    mEc.reset(new RnntEngineContainer(engineDir, batchSize));
    gLogInfo << "Instantiated RnntEngineContainer runner" << std::endl;

    // Instantiate tensors in memory
    mTc.reset(new RnntTensorContainer(mEc->encoder->getNativeDataSize()));
    gLogInfo << "Instantiated RnntTensorContainer host memory" << std::endl;

    // Determine single sample/full sequence sizes
    singleSampleSize  = FLAGS_hp_encoder_input_size * mEc->encoder->getNativeDataSize();
    fullseqSampleSize = FLAGS_hp_max_seq_length * singleSampleSize;
    // size_t sampleSize{mQsl->GetSampleSize(0)};   // load gen sample size
    size_t sampleSize = fullseqSampleSize; // FIXME: verify this

    std::cout << "Stream::Stream sampleSize: " << sampleSize << std::endl;
    std::cout << "Stream::Stream singleSampleSize: " << singleSampleSize << std::endl;
    std::cout << "Stream::Stream fullseqSampleSize: " << fullseqSampleSize << std::endl;
    std::cout << "Stream::Stream mBatchSize: " << mBatchSize << std::endl;

    if (!FLAGS_seq_splitting) {
        sampleSize = mActiveQsl->GetSampleSize(0);   // use load gen to provide this information
    }

    // It is worth to pin the staging buffers for more efficient H2D transactions through cuda host alloc
    // TODO: Should we keep this staging, or right kernels to do device-side seq-len calculations
    #define HOSTBUFSTAGING_CUDA_HOST_ALLOC_MAPPED
    #ifdef HOSTBUFSTAGING_CUDA_HOST_ALLOC_MAPPED
       CHECK_EQ(cudaHostAlloc((void**)&hostBufStaging, sampleSize * mBatchSize * sizeof(uint8_t), cudaHostAllocMapped), cudaSuccess);
       CHECK_EQ(cudaHostAlloc((void**)&hostLengthBufStaging[0], mBatchSize * sizeof(int32_t), cudaHostAllocMapped), cudaSuccess);
       CHECK_EQ(cudaHostAlloc((void**)&hostLengthBufStaging[1], mBatchSize * sizeof(int32_t), cudaHostAllocMapped), cudaSuccess);
    #else
       hostBufStaging       = (uint8_t*) malloc(sampleSize * mBatchSize * sizeof(uint8_t));
       hostLengthBufStaging[0] = (int32_t*) malloc(mBatchSize * sizeof(int32_t));
       hostLengthBufStaging[1] = (int32_t*) malloc(mBatchSize * sizeof(int32_t));
    #endif

    // Deferred construction for sampleSize. TODO: Change this after MLPINF-500
    mScatterGatherD2HData = new dali::kernels::ScatterGatherGPU (sampleSize, mBatchSize);
    mScatterGatherD2DData= new dali::kernels::ScatterGatherGPU (sampleSize, mBatchSize);

    // Do graph warmup
    auto doDecoderGraphWarmup =
        [this](size_t num_pol, const auto &batch_sizes) {
            for (size_t pol = 0; pol < num_pol; ++pol) {
                gLogVerbose << "Warmup pol: " << pol << std::endl;
                for (auto batch_size : batch_sizes) {
                    gLogVerbose << "\tBS= " << batch_size << std::endl;
                    auto cg_pair = makeDecoderGraph(batch_size, *mEc, *mTc, pol, mStream[pol]);
                    mTc->cg_cache[pol].put(batch_size, cg_pair);
                }
            }
        };
    if (FLAGS_cuda_graph && !FLAGS_cuda_legacy_graph_caching) {
        auto num_pol = FLAGS_pipelined_execution == true? PIPE_DEPTH : 1;
        auto batch_sizes = parse_cuda_graph_cache_generation();
        doDecoderGraphWarmup(num_pol, batch_sizes);
    }
}

Stream::~Stream()
{
    joinThread();
    for (int i = 0; i < PIPE_DEPTH; i++) {
        CHECK_EQ(cudaStreamDestroy(mStream[i]), cudaSuccess);
    }
    delete mScatterGatherD2HData;
    delete mScatterGatherD2DData;
}

// true to turn on warmup mode, false to turn off warmup mode
void Stream::setWarmup(bool off_on) {
    // It's suspicious if we set to a state we're already in, so make sure we aren't doing that
    CHECK(off_on != is_warmup);
    if (off_on) {
        // We're turning on warmup mode
        is_warmup = true;
        mActiveQsl = mWqsl.get();
        if(FLAGS_enable_audio_processing)
            mAudioBufManager->setWarmup(true);
    } else {
        is_warmup = false;
        mActiveQsl = mQsl.get();
        if(FLAGS_enable_audio_processing)
            mAudioBufManager->setWarmup(false);
    }
}

// Process the batchState into dense tensor(s) for the encoder
//
void Stream::processBatchStateIntoTensor(
    uint8_t* &hostBuf,
    int32_t* &hostLengthBuf,
    uint8_t* hostBufStaging,
    int32_t* hostLengthBufStaging,
    size_t actualBatchSize,
    size_t polarity)
{
    // TODO: Use a different dispatch depending on settings for DRY, or just remove uncommon codepaths.

    if (FLAGS_seq_splitting) 
    {
        // Compress/sparse reformulation (for sequence splitting)
        hostBuf = hostBufStaging;
        hostLengthBuf = hostLengthBufStaging;
        // Because our batch may be sparse, let's enforce a safe default value before writing lengths
        std::fill(hostLengthBufStaging, hostLengthBufStaging+mBatchSize, 1);
        if (FLAGS_enable_audio_processing) {
            for (const auto &el : mEncoderState) {
                const auto &sample_id = el.query_sample.index;
                const auto &offset = el.splitOffset;
                const auto &seq_len = el.sampleSeqLen;

                auto len_to_consider= std::clamp(seq_len-offset, 1, FLAGS_hp_max_seq_length);
                auto bytes_to_write = len_to_consider*singleSampleSize;

                auto devDataSrcPtr = static_cast<uint8_t*>(el.devDataPtr) + (offset * singleSampleSize);
                auto devDataDestPtr = mTc->encoderIn->get_ptr_from_idx(el.batchIdx);
                //auto devDataDestPtr = ((uint8_t*) mTc->encoderIn->data()) + (el.batchIdx * fullseqSampleSize);
                mScatterGatherD2DData->AddCopy(devDataDestPtr, devDataSrcPtr, bytes_to_write);
                hostLengthBufStaging[el.batchIdx] = len_to_consider;
            }
            // Batch copy sequence lengths (decoder needs this too, so that's why we were writing to staging buffer instead of directly to encoderInLengths)
            mScatterGatherD2DData->Run(mStream[RNNT_ENC], true, dali::kernels::ScatterGatherGPU::Method::Kernel);
            memcpyH2DAsync(*(mTc->encoderInLengths), hostLengthBufStaging, mStream[RNNT_ENC]);
        } else {
            for (const auto& el : mEncoderState) {
                const auto &sample_id= el.query_sample.index;
                const auto &offset = el.splitOffset;
                const auto &seq_len = el.sampleSeqLen;

                auto currHostPtr = static_cast<uint8_t*>(mActiveQsl->GetSampleAddress(sample_id, 0));
                auto len_to_consider = std::clamp(seq_len-offset, 1, FLAGS_hp_max_seq_length);
                auto bytes_to_write = len_to_consider * singleSampleSize;
                auto effective_start = currHostPtr + (offset * singleSampleSize);
                auto effective_end = effective_start + bytes_to_write;
                // NOTE! We're doing some pretty cache-unfriendly things here because we're not iterating through batch indices in order (we're at the whim of encoderstate's ordering)
                auto write_location = hostBuf + (el.batchIdx * fullseqSampleSize);

                std::copy(effective_start, effective_end, write_location);
                // Write to lengthbuf too
                hostLengthBufStaging[el.batchIdx] = len_to_consider;
            }
        }
    } else {
        gLogWarning << "Non-Seq-Splitting codepath hasn't been updated in a while" << std::endl;

        if(FLAGS_enable_audio_processing){
            for (auto &el: mEncoderState) {
                hostLengthBufStaging[el.batchIdx] = el.sampleSeqLen;
                if (FLAGS_use_copy_kernel) {
                    mScatterGatherD2HData->AddCopy(hostBufStaging + el.batchIdx * fullseqSampleSize, el.devDataPtr, fullseqSampleSize);
                } else {
                    CHECK_EQ(cudaMemcpyAsync(hostBufStaging + el.batchIdx * fullseqSampleSize, el.devDataPtr, fullseqSampleSize, cudaMemcpyDeviceToHost, mStream[RNNT_ENC]), cudaSuccess);
                }
            }
            if(FLAGS_use_copy_kernel){
                mScatterGatherD2HData->Run(mStream[RNNT_ENC], true, dali::kernels::ScatterGatherGPU::Method::Kernel);
            }
            // Below is a un-necessary synchronize IF all copies are on the encoder stream
            //cudaStreamSynchronize(mStream[RNNT_ENC]);
        } else {

            for (const auto &el : mEncoderState) {
                auto currHostPtr = static_cast<uint8_t*>(mActiveQsl->GetSampleAddress(el.query_sample.index, 0));
                auto effective_start = currHostPtr;
                auto effective_end = effective_start + fullseqSampleSize;
                auto write_location = hostBufStaging + el.batchIdx*fullseqSampleSize;

                std::copy(effective_start, effective_end, write_location);
                hostLengthBufStaging[el.batchIdx] = el.sampleSeqLen;
            }
        }
        hostBuf = hostBufStaging;
        hostLengthBuf = hostLengthBufStaging;
    }
}

bool Stream::frontEndBatchPrologue(
    size_t p, // polarity
    size_t &availBatchSlots)
{
    std::vector<sampleAttributes> daliSamples;
    std::vector<mlperf::QuerySample> samples;
    size_t sampleSize;

    // if (FLAGS_enable_audio_processing) {
    //     sampleSize = FLAGS_audio_buffer_line_size;
    // } else {
    //     sampleSize = mActiveQsl->GetSampleSize(0);
    // }

    // Get a batch up to mBatchSize.
    if (FLAGS_enable_audio_processing) {
        mAudioBufManager->getBatch(daliSamples, availBatchSlots);
        actualBatchSize[p] = std::max(daliSamples.size() + mEncoderState.size(), mEncoderState.max_idx_rng());
    } else {
        mWorkQueue->getBatch(samples, availBatchSlots);
        // for actualBatchSize, we need to account for potential sparse fragmentation (maximum ac)
        actualBatchSize[p] = std::max(samples.size() + mEncoderState.size(), mEncoderState.max_idx_rng());
    }

    // Nothing to process?
    if (actualBatchSize[p] == 0) {
        return false;
    }

    // gLogInfo << "actualBatchSize[p] = " << actualBatchSize[p] << std::endl;
    // gLogInfo << "   availBatchSlots = " << availBatchSlots << std::endl;
    // gLogInfo << "   samples.size    = " << samples.size() << std::endl;
    // gLogInfo << "   pending.size    = " << mBatchState[RNNT_ENC]->size() << std::endl;

    // allocate batchState entries
    if(FLAGS_enable_audio_processing){
        for (const auto &ds : daliSamples) {
            mEncoderState.allocate(ds);
        }
    } else {
        for (const auto &sample: samples) {
            auto seqLen = *static_cast<int32_t*>(mActiveQsl->GetSampleAddress(sample.index, 1));
            mEncoderState.allocate(sample, seqLen);
        }
    }

    return true;
}


void Stream::frontEndBatch(
    size_t p) // polarity
{
    uint8_t* hostBuf{nullptr};
    int32_t* hostLengthBuf{nullptr};

    // create new batch tensor input (based on allocated batch instances)
    processBatchStateIntoTensor(
        hostBuf,
        hostLengthBuf,
        hostBufStaging,
        hostLengthBufStaging[p],
        actualBatchSize[p],
        p);

    auto statemask = mEncoderState.get_state_mask();

    // Run encoder-side inference on the batch.
    doBatchInferenceEncoder(
        hostBuf, hostLengthBuf,
        fullseqSampleSize, actualBatchSize[p], 
        *mEc, *mTc, statemask.get(),
        p, mStream[0]);
}

void Stream::backEndBatchPrologue(
    size_t pol // polarity
    )
{
    // Deciding stream:
    //   0 : if do not have pipelining execution (need to serialize encoder/decoder)
    //   1 : if have pipelining execution (need private stream to avoid encoder mess with graph capture)_
    cudaStream_t backEndStream = mStream[0];
    if (FLAGS_pipelined_execution) backEndStream = mStream[1];

    // Compare our generated statemask to the reference:
    auto statemask = mDecoderState.get_state_mask();
    batchInfDecoderInitializeTensors(
        hostLengthBufStaging[pol],
        actualBatchSize[pol], 
        *mTc, 
        statemask.get(),
        pol,
        backEndStream);
}


void Stream::backEndBatch (
    size_t p // polarity
    )
{
    // Run decoder/joint/greedy-side inference on the batch.
    std::vector<std::vector<RETURN_T>> batchResults;

    // Deciding stream:
    //   0 : if do not have pipelining execution (need to serialize encoder/decoder)
    //   1 : if have pipelining execution (need private stream to avoid encoder mess with graph capture)_
    cudaStream_t backEndStream = FLAGS_pipelined_execution ? mStream[1] : mStream[0];

    batchResults = doBatchInferenceDecoder(
        actualBatchSize[p], 
        *mEc,
        *mTc,
        p,
        backEndStream);

    mDecoderState.update_text_batch(batchResults);

    // Return results to LoadGen (or WarmupManager)
    auto completion_function = !is_warmup ? mlperf::QuerySamplesComplete : WarmupManager::warmupSamplesComplete;
    mDecoderState.send_responses_and_clear(completion_function);
}


void Stream::processBatches()
{
    CHECK_EQ(cudaSetDevice(mDeviceId), cudaSuccess);

    #ifdef PER_BATCH_LOG
    auto timeStart = std::chrono::high_resolution_clock::now();
    auto last_count = timeStart;
    #endif

    // prologue
    size_t p = 0; // polarity
    size_t availBatchSlots  = mBatchSize;  // available batch slots
    bool avail_samples = false;
    while (!mDone)
    {
        // run encoder porting
        avail_samples = frontEndBatchPrologue(p, availBatchSlots);

        // do not bother with the back-end if there is nothing to process
        if (!avail_samples) continue;

        // run encoder
        NVTX_START(nvtxRunEnc, "ENC:run", COLOR_BLUE_0);
        frontEndBatch(p);
        NVTX_END(nvtxRunEnc);

        auto daliIdxsToClear = transfer_state_from_encoder_to_decoder(mEncoderState, mDecoderState);

        if(FLAGS_enable_audio_processing){
            // std::cout << "Clearing idxs: " << std::endl;
            // for (const auto &idx : daliIdxsToClear) {
            //     std::cout << "\t" << idx << std::endl;
            // }
            mAudioBufManager->releaseBatch(daliIdxsToClear);
        }

        availBatchSlots = mEncoderState.num_available();

        // run decoder portion
        NVTX_START(nvtxRunDec, "DEC: run", COLOR_GREEN_0);
        backEndBatchPrologue(p);
        backEndBatch(p);
        NVTX_END(nvtxRunDec);


        #ifdef PER_BATCH_LOG
            // Partial performance
            auto curr_count = std::chrono::high_resolution_clock::now();
            auto partialTimeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(curr_count - last_count);
            gLogInfo << "Batch Throughput = " << (static_cast<double>(actualBatchSize[p]) / partialTimeSpan.count()) << " infer/s." << std::endl;
            gLogInfo << "Batch Time = " << partialTimeSpan.count() << " s." << std::endl;
            last_count = curr_count;
        #endif
    }
}

void Stream::processBatchesFrontEnd()
{
    CHECK_EQ(cudaSetDevice(mDeviceId), cudaSuccess);

    // prologue
    size_t p = 0; // polarity
    size_t availBatchSlots  = mBatchSize;  // available batch slots
    bool avail_samples = false;

    size_t iCnt = 0;

    while (!mDone)
    {
        // run encoder 
        NVTX_START(nvtxWaitFe, "ENC: wait for FE", COLOR_PINK_0);
        do {
            avail_samples = frontEndBatchPrologue(p, availBatchSlots);
        } while(!avail_samples && !mDone);
        NVTX_END(nvtxWaitFe);

        nvtxRangeId_t NVTX_START_WITH_PAYLOAD(nvtxRunEnc, "ENC:run", COLOR_BLUE_0, actualBatchSize[p]);
        if (avail_samples) {
            frontEndBatch(p);
        }
        CHECK_EQ(cudaEventRecord(cudaEventEncToDec, mStream[0]), cudaSuccess); // Notify the gpu portion of the encoder is done
        NVTX_END(nvtxRunEnc);


        // wait for decoder (decoder loop)
        NVTX_START(nvtxWaitDecEng, "ENC: wait for decoder engine", COLOR_RED_0);
        wait_for_thread(mDone, encoderCredits,  mutex_b2f, cv_b2f, "threadSynch: processBatchesFrontEnd[0]: ", p, iCnt);
        NVTX_END(nvtxWaitDecEng);


        // transfer the state to the decoder
        NVTX_START(nvtxTransfer, "ENC: transfer state", COLOR_BLUE_2);
        auto daliIdxsToClear = transfer_state_from_encoder_to_decoder(mEncoderState, mDecoderState);
        if(FLAGS_enable_audio_processing){
            mAudioBufManager->releaseBatch(daliIdxsToClear);
        }

        availBatchSlots = mEncoderState.num_available();
        NVTX_END(nvtxTransfer);

        // wake up the decoder
        notify_thread(decoderCredits, mutex_f2b, cv_f2b, "threadSynch: processBatchesFrontEnd: notify decoder:", p, iCnt);

        // wait for decoder (init tensors)
        NVTX_START(nvtxWaitDecInit, "ENC: wait for decoder init", COLOR_RED_1);
        wait_for_thread(mDone, encoderCredits,  mutex_b2f, cv_b2f, "threadSynch: processBatchesFrontEnd[1]: ", p, iCnt);
        NVTX_END(nvtxWaitDecInit);

        // switch polarity
        p = (p + 1) % PIPE_DEPTH;

        iCnt++;


    }
    if (FLAGS_debug_mode) gLogInfo << "threadSynch:" << iCnt << " processBatchesFrontEnd: signing off!" << std::endl;
}

void Stream::processBatchesBackEnd()
{
    CHECK_EQ(cudaSetDevice(mDeviceId), cudaSuccess);

    #ifdef PER_BATCH_LOG
    auto timeStart = std::chrono::high_resolution_clock::now();
    auto last_count = timeStart;
    #endif

    // prologue
    size_t p = 0; // polarity

    size_t iCnt = 0;

    while (!mDone)
    {
       // wait for encoder (to transfer state)
        NVTX_START(nvtxWaitEncTransfer, "DEC: wait for encoder transfer", COLOR_RED_0);
        wait_for_thread(mDone, decoderCredits,  mutex_f2b, cv_f2b, "threadSynch: processBatchesBackEnd: ", p, iCnt);
        NVTX_END(nvtxWaitEncTransfer);

        // initialize tensors
        NVTX_START(nvtxDecInit, "DEC: tensor init", COLOR_GREEN_1);
        if(actualBatchSize[p]) {
            backEndBatchPrologue(p);
        }
        NVTX_END(nvtxDecInit);

        // wake up the encoder
        notify_thread(encoderCredits, mutex_b2f, cv_b2f, "threadSynch: processBatchesBackEnd[0]: notify encoder:", p, iCnt);
        
        // run decoder portion
        nvtxRangeId_t NVTX_START_WITH_PAYLOAD(nvtxRunDec, "DEC: run", COLOR_GREEN_0, actualBatchSize[p]);
        CHECK_EQ(cudaStreamWaitEvent(mStream[1], cudaEventEncToDec, 0), cudaSuccess); // wait for GPU portion of the encoder to finish
        if(actualBatchSize[p]) {
            backEndBatch(p);
        }
        NVTX_END(nvtxRunDec);

        // wake up the encoder
        // notify_thread(encoderGo, encoderCredits, mutex_b2f, cv_b2f, "threadSynch: processBatchesBackEnd: notify encoder:", iCnt);
        notify_thread(encoderCredits, mutex_b2f, cv_b2f, "threadSynch: processBatchesBackEnd[1]: notify encoder:", p, iCnt);

        // switch polarity
        p = (p + 1) % PIPE_DEPTH;

        #ifdef PER_BATCH_LOG
            // Partial performance
            auto curr_count = std::chrono::high_resolution_clock::now();
            auto partialTimeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(curr_count - last_count);
            gLogInfo << "Batch Throughput = " << (static_cast<double>(actualBatchSize[p]) / partialTimeSpan.count()) << " infer/s." << std::endl;
            gLogInfo << "Batch Time = " << partialTimeSpan.count() << " s." << std::endl;
            last_count = curr_count;
        #endif
        iCnt++;
    }

    if (FLAGS_debug_mode) gLogInfo << "threadSynch:" << iCnt << " processBatchesBackEnd: signing off!" << std::endl;
}

void Stream::launchThread()
{
    if(FLAGS_pipelined_execution) {
        mThProcessSampleFe = std::thread(&Stream::processBatchesFrontEnd, this);
        mThProcessSampleBe = std::thread(&Stream::processBatchesBackEnd, this);
    } else {
        mThProcessSample = std::thread(&Stream::processBatches, this);
    }

}

void Stream::joinThread()
{
    mDone = true;
    if (FLAGS_enable_audio_processing) {
        mAudioBufManager->Done();
    }

    if(FLAGS_pipelined_execution) {
        if (mThProcessSampleFe.joinable() && mThProcessSampleBe.joinable()) {
            mThProcessSampleFe.join();
            mThProcessSampleBe.join();
        }
    } else {
        if (mThProcessSample.joinable()) {
            mThProcessSample.join();
        }
    }
}

// =============================
//     Server
// =============================
//
class RNNTServer : public mlperf::SystemUnderTest
{
public:
    RNNTServer(
        const std::string name,
        const std::string enginePath,
        std::shared_ptr<qsl::SampleLibrary> qsl,
        std::shared_ptr<WarmupSampleLibrary> wqsl,
        const std::vector<int>& gpus,
        const size_t streamsPerGpu,
        const size_t maxBatchSize,
        bool useBatchSort
    );

    virtual ~RNNTServer() = default;

    const std::string& Name() const override { return mName; };

    void setWarmup(bool off_on) {
        CHECK(is_warmup != off_on);
        if (off_on) {
            is_warmup = true;
            mActiveQsl = mWqsl.get();
            for (auto& stream_ptr : mStreams) {
                stream_ptr->setWarmup(true);
            }
        } else {
            is_warmup = false;
            mActiveQsl = mQsl.get();
            for (auto& stream_ptr : mStreams) {
                stream_ptr->setWarmup(false);
            }
        }
    }
    void Warmup() {
        //First, we inform all streams that they should be put in warmup mode (don't send response to loadgen, and use warmup query library instead). The stream will handle setting the preprocessing pipeline into warmup mode too
        // We (RNNTServer) also uses QSL, so we need to make sure IssueQuery uses the correct qsl
        setWarmup(true);

        // WarmupManager handles the case of (num_samples > wqsl->TotalSampleCount())
        WarmupManager wm(mWqsl);

        this->IssueQuery(wm.query_to_send);

        wm.sleep_until_done();

        // We finish by taking all streams and preprocessing pipelines out of warmup mode
        setWarmup(false);
    };
    void Done();

    // Loadgen SUT
    void StartIssueThread(int threadIdx);
    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;
    void FlushQueries() override {};
    void ReportLatencyResults([[maybe_unused]]const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {};

private:
    const std::string mName;
    std::vector<std::shared_ptr<AudioBufferManagement>> mAudioBufManagers;
    std::vector<std::unique_ptr<DaliPipeline>> mDaliPipelines;
    std::shared_ptr<qsl::SampleLibrary> mQsl;
    std::shared_ptr<WarmupSampleLibrary> mWqsl;
    qsl::LookupableQuerySampleLibrary *mActiveQsl;
    bool mUseBatchSort;
    std::vector<std::shared_ptr<SyncWorkQueue>> mWorkQueues;

    std::vector<std::unique_ptr<Stream>> mStreams;
    size_t mNumDevices;
    bool is_warmup{false};
    size_t warmup_counter{0};

    // data members to support multiple IssueQuery() threads if server_num_issue_query_threads != 0
    std::mutex mMtx;
    std::map<std::thread::id, int> mThreadMap;
    std::vector<std::thread> mIssueQueryThreads;
};

RNNTServer::RNNTServer(
    const std::string name,
    const std::string enginePath,
    std::shared_ptr<qsl::SampleLibrary> qsl,
    std::shared_ptr<WarmupSampleLibrary> wqsl,
    const std::vector<int>& gpus,
    const size_t streamsPerGpu,
    const size_t maxBatchSize,
    bool useBatchSort)
    : mName{name}
    , mQsl{qsl}
    , mWqsl(wqsl)
    , mActiveQsl(qsl.get())
    , mUseBatchSort(useBatchSort)
{
    auto numIQThreads = FLAGS_server_num_issue_query_threads;
    if (numIQThreads) {
        CHECK_EQ(gpus.size() % numIQThreads, 0);
        for (uint64_t i = 0; i < numIQThreads; ++i) {
            mWorkQueues.emplace_back(new SyncWorkQueue);
        }
    } else {
        mWorkQueues.emplace_back(new SyncWorkQueue);
    }

    size_t numGpusPerWorkQueue = (numIQThreads == 0) ? INT_MAX : gpus.size() / numIQThreads;
    size_t workQueueCounter = 0;
    size_t workQueueIdx = 0;
    size_t numDevices = 0;
    for (auto deviceId : gpus) {
        gLogInfo << "Set to device " << deviceId << std::endl;
        CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
        if(FLAGS_enable_audio_processing)
        {
            mAudioBufManagers.emplace_back(
                std::make_shared<AudioBufferManagement>(FLAGS_audio_batch_size,
                                                        FLAGS_audio_buffer_line_size,
                                                        FLAGS_audio_buffer_num_lines,
                                                        gpus.size(),
                                                        mQsl,
                                                        mWqsl,
                                                        mWorkQueues[workQueueIdx],
                                                        FLAGS_dali_pipeline_depth));

            mDaliPipelines.emplace_back(
                std::make_unique<DaliPipeline>(deviceId, FLAGS_audio_serialized_pipeline_file,
                                               mAudioBufManagers[numDevices],
                                               FLAGS_audio_batch_size,
                                               static_cast<device_type_t>(FLAGS_audio_device_type),
                                               FLAGS_audio_prefetch_queue_depth,
                                               FLAGS_dali_pipeline_depth,
                                               FLAGS_dali_batches_issue_ahead));
        }

        for (size_t streamIdx = 0; streamIdx < streamsPerGpu; streamIdx++) {
            gLogInfo << "Creating stream " << streamIdx << "/" << streamsPerGpu << std::endl;
            if(FLAGS_enable_audio_processing) {
                mStreams.emplace_back(new Stream(deviceId, mQsl, mWqsl, mWorkQueues[workQueueIdx], mAudioBufManagers[numDevices], enginePath, maxBatchSize));
            } else {
                mStreams.emplace_back(new Stream(deviceId, mQsl, mWqsl, mWorkQueues[workQueueIdx], nullptr, enginePath, maxBatchSize));
            }
        }
        ++numDevices;
        ++workQueueCounter;
        if (workQueueCounter == numGpusPerWorkQueue) {
            ++workQueueIdx;
            workQueueCounter = 0;
        }
    }
    mNumDevices = numDevices;
    for (auto& stream : mStreams) {
        stream->launchThread();
    }

    if(FLAGS_enable_audio_processing)
        for(auto& daliPipeline : mDaliPipelines){
            daliPipeline->launchThread();
        }

    if (numIQThreads) {
        gLogInfo << "Start " << numIQThreads << " IssueQuery() threads" << std::endl;
        for (int i = 0; i < numIQThreads; ++i) {
            mIssueQueryThreads.emplace_back(&RNNTServer::StartIssueThread, this, i);
        }
    } else {
        // store the main thread id
        mThreadMap[std::this_thread::get_id()] = 0;
    }
}

void RNNTServer::Done()
{
    for (auto& stream : mStreams) {
        stream->joinThread();
    }
    for (auto& thread : mIssueQueryThreads) {
        thread.join();
    }
}

void RNNTServer::StartIssueThread(int threadIdx) {
    {
        std::lock_guard<std::mutex> lock(mMtx);
        mThreadMap[std::this_thread::get_id()] = threadIdx;
    }
    mlperf::RegisterIssueQueryThread();
}

void RNNTServer::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    int threadIdx;
    if (is_warmup) {
        threadIdx = warmup_counter;
        warmup_counter = (warmup_counter + 1) % mWorkQueues.size();
    } else {
        std::unique_lock<std::mutex> lock(mMtx);
        threadIdx = mThreadMap[std::this_thread::get_id()];
        lock.unlock();
    }

    if (mUseBatchSort) {
        // Sort samples in the descending order of sentence length
        std::vector<std::pair<int, int>> sequenceSamplePosAndLength(samples.size());
        for (size_t samplePos = 0; samplePos < samples.size(); ++samplePos) {
            sequenceSamplePosAndLength[samplePos] = std::make_pair(samplePos, *static_cast<int32_t*>(mActiveQsl->GetSampleAddress(samples[samplePos].index, 1)));
        }
        std::sort(
            sequenceSamplePosAndLength.begin(),
            sequenceSamplePosAndLength.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.second > b.second; });
        std::vector<mlperf::QuerySample> sortedSamples(samples.size());
        std::transform(
            sequenceSamplePosAndLength.begin(),
            sequenceSamplePosAndLength.end(),
            sortedSamples.begin(),
            [samples](const std::pair<int, int>& a) { return samples[a.first]; });

        mWorkQueues[threadIdx]->insertItems(sortedSamples);
    }
    else {
        mWorkQueues[threadIdx]->insertItems(samples);
    }
}

// =============================
//     Main method
// =============================
 

int main(int argc, char* argv[])
{
    // Initialize logging
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string gSampleName = "RNN-T_Harness";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    gLogger.reportTestStart(sampleTest);

    // Set verbosity
    if (FLAGS_verbose) {
        setReportableSeverity(Severity::kVERBOSE);
    }

    if (FLAGS_accuracy_mode)
    {
        gLogInfo << "Running accuracy mode." << std::endl;
    }

    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    // Load all the needed shared objects for plugins.
    std::vector<std::string> plugin_files = splitString(FLAGS_plugins, ",");
    for (auto& s : plugin_files)
    {
        void* dlh = dlopen(s.c_str(), RTLD_LAZY);
        if (nullptr == dlh)
        {
            gLogError << "Error loading plugin library " << s << std::endl;
            return 1;
        }
    }

    // Perform inference
    // Scope to force all smart objects destruction before CUDA context resets
    {
        // Configure the test settings
        mlperf::TestSettings testSettings;
        testSettings.scenario = scenarioMap[FLAGS_scenario];
        testSettings.mode = testModeMap[FLAGS_accuracy_mode ? "AccuracyOnly" : FLAGS_test_mode];
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

        std::string raw_data_dir;
        if(FLAGS_audio_fp16_input){
            raw_data_dir = FLAGS_raw_data_dir + "/fp16";
        }
        else{
            raw_data_dir = FLAGS_raw_data_dir + "/fp32";
        }
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
            auto deviceNames = splitString(FLAGS_devices, ",");
            for (auto &n : deviceNames) gpus.emplace_back(std::stoi(n));
        }

        std::vector<std::string> tensorPaths;
        if (FLAGS_enable_audio_processing) {
            tensorPaths.push_back(raw_data_dir);
            tensorPaths.push_back(FLAGS_raw_length_dir);
        } else {
            tensorPaths.push_back(FLAGS_preprocessed_data_dir);
            tensorPaths.push_back(FLAGS_preprocessed_length_dir);
        }

        std::vector<bool> start_from_device = {FLAGS_start_from_device, false};
        gLogInfo << "Starting creating QSL." << std::endl;
        auto qsl = std::make_shared<qsl::SampleLibrary>("RNN-T QSL", FLAGS_val_map, tensorPaths, FLAGS_performance_sample_count, 0, false, start_from_device);

        // Here we figure out how many samples to generate for warmup:
        // Determine number of samples to run
        CHECK_GE(FLAGS_num_warmups, -1);
        // Arbitrarily chosen. See "Multistream" in MLPINF-420 for more information.
        auto default_num_warmup_samples = 2*FLAGS_batch_size * FLAGS_streams_per_gpu;
        auto num_warmup_samples = FLAGS_num_warmups == -1 ? default_num_warmup_samples : FLAGS_num_warmups;
        auto wqsl = std::make_shared<WarmupSampleLibrary>(num_warmup_samples, FLAGS_enable_audio_processing);
        gLogInfo << "Finished creating QSL." << std::endl;

        gLogInfo << "Starting creating SUT." << std::endl;
        auto rnnt_server = std::make_shared<RNNTServer>(
            "RNNT SERVER",
            FLAGS_engine_dir,
            qsl,
            wqsl,
            gpus,
            FLAGS_streams_per_gpu,
            FLAGS_batch_size,
            FLAGS_batch_sorting);
        gLogInfo << "Finished creating SUT." << std::endl;

        if (FLAGS_num_warmups) {
            gLogInfo << "Starting warming up SUT." << std::endl;
            NVTX_START(nvtxWarmUp, "Main::Warmup", COLOR_BLUE_3);
            rnnt_server->Warmup();
            gLogInfo << "Finished warming up SUT." << std::endl;
            NVTX_END(nvtxWarmUp);
        }

        gLogInfo << "Starting running actual test." << std::endl;
        cudaProfilerStart();
        StartTest(rnnt_server.get(), qsl.get(), testSettings, logSettings);
        cudaProfilerStop();
        gLogInfo << "Finished running actual test." << std::endl;
        rnnt_server->Done();
    }

    // Report pass
    return gLogger.reportPass(sampleTest);
}
