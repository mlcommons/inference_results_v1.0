/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "triton_frontend.hpp"

// Triton
#include "src/core/constants.h"
#include "src/core/model_config.pb.h"
#include "src/servers/common.h"

// Protobuf
#include <google/protobuf/util/json_util.h>

// Google Logging
#include <glog/logging.h>

// General C++
#include <deque>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <vector>

// LoadGen
#include "loadgen.h"

// DLRM QSL
#include "dlrm_qsl.hpp"

/* Use Triton namespace */
namespace ni = nvidia::inferenceserver;

/* Define macro for enabling tracing */
#ifndef TRITON_FRONTEND_TRACE
#define TRITON_FRONTEND_TRACE 0
#endif

namespace
{
// Callback function for released request
void RequestRelease(TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
    auto request_block = reinterpret_cast<triton_frontend::RequestPool::Block*>(userp);
    triton_frontend::RequestPool::Release(request_block);
}
// Callback function for completed response
void ResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
    /* Pass the response (and various items from the userp field) to the member
     * Completion function
     */
    auto response_metadata = reinterpret_cast<triton_frontend::ResponseMetaData*>(userp);
    triton_frontend::Server_SUT* sut = response_metadata->m_ServerPtr;
    sut->Completion(response, response_metadata);
#if TRITON_FRONTEND_TRACE
    if (response_metadata->m_TracePtr != nullptr)
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t trace_id = 0;
        TRITONSERVER_InferenceTraceId(response_metadata->m_TracePtr, &trace_id);
        sut->m_TraceManager->CaptureTimestamp(
            trace_id, TRITONSERVER_TRACE_LEVEL_MIN, "MLPerf Request Response RECV", TIMESPEC_TO_NANOS(ts));
    }
#endif // TRITON_FRONTEND_TRACE
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(response), "deleting resonse");
}

// Callback function for inference requests during warmup phase
void WarmupResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flag, void* userp)
{
    /* Make sure we have a valid response */
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(response), "response");

    /* Increment the number of warmup responses received via member function call
     */
    auto sut = reinterpret_cast<triton_frontend::Server_SUT*>(userp);
    sut->IncrementWarmupResponses();
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(response), "deleting resonse");
}

std::string MemoryTypeString(TRITONSERVER_MemoryType memory_type)
{
    return (memory_type == TRITONSERVER_MEMORY_CPU) ? "CPU memory" : "GPU memory";
}

TRITONSERVER_DataType DataTypeToTriton(const inference::DataType dtype)
{
    switch (dtype)
    {
    case inference::DataType::TYPE_BOOL: return TRITONSERVER_TYPE_BOOL;
    case inference::DataType::TYPE_UINT8: return TRITONSERVER_TYPE_UINT8;
    case inference::DataType::TYPE_UINT16: return TRITONSERVER_TYPE_UINT16;
    case inference::DataType::TYPE_UINT32: return TRITONSERVER_TYPE_UINT32;
    case inference::DataType::TYPE_UINT64: return TRITONSERVER_TYPE_UINT64;
    case inference::DataType::TYPE_INT8: return TRITONSERVER_TYPE_INT8;
    case inference::DataType::TYPE_INT16: return TRITONSERVER_TYPE_INT16;
    case inference::DataType::TYPE_INT32: return TRITONSERVER_TYPE_INT32;
    case inference::DataType::TYPE_INT64: return TRITONSERVER_TYPE_INT64;
    case inference::DataType::TYPE_FP16: return TRITONSERVER_TYPE_FP16;
    case inference::DataType::TYPE_FP32: return TRITONSERVER_TYPE_FP32;
    case inference::DataType::TYPE_FP64: return TRITONSERVER_TYPE_FP64;
    case inference::DataType::TYPE_STRING: return TRITONSERVER_TYPE_BYTES;
    default: break;
    }

    return TRITONSERVER_TYPE_INVALID;
}

size_t GetDataTypeByteSize(const inference::DataType dtype)
{
    switch (dtype)
    {
    case inference::TYPE_BOOL: return 1;
    case inference::TYPE_UINT8: return 1;
    case inference::TYPE_UINT16: return 2;
    case inference::TYPE_UINT32: return 4;
    case inference::TYPE_UINT64: return 8;
    case inference::TYPE_INT8: return 1;
    case inference::TYPE_INT16: return 2;
    case inference::TYPE_INT32: return 4;
    case inference::TYPE_INT64: return 8;
    case inference::TYPE_FP16: return 2;
    case inference::TYPE_FP32: return 4;
    case inference::TYPE_FP64: return 8;
    case inference::TYPE_STRING: return 0;
    default: break;
    }

    return 0;
}

TRITONSERVER_Error* ResponseAlloc(TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name, size_t byte_size,
    TRITONSERVER_MemoryType preferred_memory_type, int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type, int64_t* actual_memory_type_id)
{
    // If 'byte_size' is zero just return 'buffer'==nullptr, we don't
    // need to do any other book-keeping.
    if (byte_size == 0)
    {
        *buffer = nullptr;
        *buffer_userp = nullptr;
        // LOG_INFO << "allocated " << byte_size << " bytes for result tensor "
        //         << tensor_name;
    }
    else
    {
        auto pool = reinterpret_cast<triton_frontend::PinnedMemoryPool*>(userp);
        auto block = pool->Obtain();
        // Use CPU instead of CPU_PINNED to trigger internal pinned memory buffer
        // in Triton
        *actual_memory_type = TRITONSERVER_MEMORY_CPU;
        *actual_memory_type_id = 0;
        *buffer = block->m_Data;
        *buffer_userp = new triton_frontend::PoolBlockPair(pool, block);
    }

    return nullptr; // Success
}

TRITONSERVER_Error* ResponseRelease(TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
{
    if (buffer_userp != nullptr)
    {
        auto pool_block = reinterpret_cast<triton_frontend::PoolBlockPair*>(buffer_userp);
        pool_block->first->Release(pool_block->second);
        delete pool_block;
    }
    return nullptr; // Success
}

size_t GetSampleLength(qsl::SampleLibraryPtr_t qsl_ptr, const mlperf::QuerySampleIndex idx)
{
    // Get sample length by checking where the input_mask change from 1 to 0
    size_t start{0};
    size_t end{BERT_MAX_SEQ_LENGTH};
    size_t cursor{(start + end) / 2};
    auto& input_mask = *static_cast<std::array<int32_t, BERT_MAX_SEQ_LENGTH>*>(qsl_ptr->GetSampleAddress(idx, 2));
    while (cursor != start)
    {
        if (input_mask[cursor])
        {
            start = cursor;
        }
        else
        {
            end = cursor;
        }
        cursor = (start + end) / 2;
    }
    return end;
}
} // namespace

namespace triton_frontend
{

std::vector<std::unique_ptr<RequestPool>> RequestPool::instances_;

PinnedMemoryPool::PinnedMemoryPool(const size_t element_count, const size_t element_byte_size, sharedMR_ptr s2c_shmem)
    : m_Head(nullptr)
    , m_Blocks(element_count)
    , m_Buffer(nullptr)
    , m_s2c_shmem(s2c_shmem)
{
    assert(element_count * element_byte_size <= m_s2c_shmem->get_size());
    m_Buffer = (char*) m_s2c_shmem->get_address();
    FAIL_IF_CUDA_ERR(cudaHostRegister((void*) m_Buffer, element_count * element_byte_size, cudaHostRegisterPortable),
        "failed to register pinned memory");

    char* next_buffer = m_Buffer;
    for (auto& block : m_Blocks)
    {
        if (m_Head != nullptr)
        {
            block.m_NextBlock = m_Head;
        }
        m_Head = &block;
        block.m_Data = next_buffer;
        next_buffer += element_byte_size;
    }
}

PinnedMemoryPool::~PinnedMemoryPool()
{
    FAIL_IF_CUDA_ERR(cudaHostUnregister(m_Buffer), "failed to unregister pinned memory");
}

void RequestPool::Create(const size_t initial_element_count, TRITONSERVER_Server* server, Server_SUT* server_sut,
    const std::string& model_name, const uint32_t model_version, const std::vector<InputMetaData>& inputs,
    const std::vector<std::string>& outputs)
{
    for (size_t count = 0; count < 2 /* REQUEST_POOL_COUNT */; count++)
    {
        instances_.emplace_back(
            new RequestPool(initial_element_count, server, server_sut, model_name, model_version, inputs, outputs));
    }
}

void RequestPool::Destroy()
{
    for (auto& instance : instances_)
    {
        instance.reset(nullptr);
    }
}

RequestPool::RequestPool(const size_t initial_element_count, TRITONSERVER_Server* server, Server_SUT* server_sut,
    const std::string& model_name, const uint32_t model_version, const std::vector<InputMetaData>& inputs,
    const std::vector<std::string>& outputs)
    : m_Head(nullptr)
    , m_ReleasedHead(nullptr)
    , m_Blocks(initial_element_count)
    , m_Server(server)
    , m_ServerSUT(server_sut)
    , m_ModelName(model_name)
    , m_ModelVersion(model_version)
    , m_Inputs(inputs)
    , m_Outputs(outputs)
{
    Block* prev_block = nullptr;
    // Build free block list in the same order as in std list
    for (auto& block : m_Blocks)
    {
        if (m_Head == nullptr)
        {
            m_Head = &block;
        }
        if (prev_block != nullptr)
        {
            prev_block->m_NextBlock = &block;
        }
        prev_block = &block;
        InternalInitInferenceRequest(&block);
    }
}

RequestPool::~RequestPool()
{
    for (auto& block : m_Blocks)
    {
        if (block.m_Data != nullptr)
        {
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestDelete(block.m_Data), "deleting inference request");
        }
    }
}

void RequestPool::InternalInitInferenceRequest(RequestPool::Block* block)
{
    block->m_AssignedPool = this;
    block->m_ResponseMetadata = ResponseMetaData(m_ServerSUT);

    // Init m_Data (TRITONSERVER_InferenceRequest)
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestNew(&block->m_Data, m_Server, m_ModelName.c_str(), m_ModelVersion),
        "creating new inference request");
    for (const auto& input : m_Inputs)
    {
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(block->m_Data, std::get<0>(input).c_str(), std::get<1>(input),
                        std::get<2>(input).data(), std::get<2>(input).size()),
            "setting input meta-data for the request");
    }
    for (const auto& output : m_Outputs)
    {
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(block->m_Data, output.c_str()),
            "requesting output for the request");
    }
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(block->m_Data, RequestRelease, block),
        "setting request release callback");
}

void Server_SUT::Init(size_t min_sample_size, size_t max_sample_size, size_t buffer_manager_thread_count)
{
#if TRITON_FRONTEND_TRACE
    /* Set up trace manager */
    ni::TraceManager* manager = nullptr;
    FAIL_IF_ERR(nvidia::inferenceserver::TraceManager::Create(
                    &manager, TRITONSERVER_TRACE_LEVEL_MAX, 80 /* rate, one sample per batch*/, "triton_trace.log"),
        "creating trace manger");
    m_TraceManager.reset(manager);
#endif // TRITON_FRONTEND_TRACE

    /* Create the options for the server */
    TRITONSERVER_ServerOptions* server_options = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsNew(&server_options), "creating server options");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options, m_ModelRepositoryPath.c_str()),
        "setting model repository path");
    // FIXME currently don't need to pass in valid directory as TensorRT backend
    // has not yet decoupled from Triton core, will need to fix once it is
    // separated as dynamically loaded library
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetBackendDirectory(server_options, ""), "setting backend directory");
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(server_options, ((uint64_t) 1) << 29 /* 512 MB */),
        "setting pinned memory pool size");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(server_options, buffer_manager_thread_count),
        "setting buffer manager thread count");

    /* Actually create the server now */
    TRITONSERVER_Server* server_ptr = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerNew(&server_ptr, server_options), "creating server");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsDelete(server_options), "deleting server options");
    m_Server = std::shared_ptr<TRITONSERVER_Server>(server_ptr, TRITONSERVER_ServerDelete);

    /* Wait until the server is both live and ready, and the model is ready. */
    size_t health_iters = 0;
    while (true)
    {
        bool live, ready, model_ready;
        FAIL_IF_ERR(TRITONSERVER_ServerIsLive(m_Server.get(), &live), "unable to get server liveness");
        FAIL_IF_ERR(TRITONSERVER_ServerIsReady(m_Server.get(), &ready), "unable to get server readiness");
        FAIL_IF_ERR(TRITONSERVER_ServerModelIsReady(m_Server.get(), m_ModelName.c_str(), m_ModelVersion, &model_ready),
            "unable to get model readiness");
        std::cout << "Server Health status: live " << live << ", ready " << ready << ", model ready " << model_ready
                  << std::endl;
        if (live && ready && model_ready)
        {
            std::cout << "Server is live and ready. Model is ready" << std::endl;
            break;
        }

        if (++health_iters >= 200)
        {
            FAIL("failed to find healthy inference server within 200 tries");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Initalize enough pinned output buffer in front
    // We want to have instance # sets of output buffers, each set has
    // output # buffers, and each buffer has max batch bytes for the output
    TRITONSERVER_Message* model_config_message = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerModelConfig(
                    m_Server.get(), m_ModelName.c_str(), m_ModelVersion, 1 /* config_version */, &model_config_message),
        "unable to get model config message");
    auto lcm = std::unique_ptr<TRITONSERVER_Message, decltype(&TRITONSERVER_MessageDelete)>(
        model_config_message, TRITONSERVER_MessageDelete);

    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_config_message, &buffer, &byte_size),
        "unable to serialize model metadata message to JSON");

    inference::ModelConfig config;
    ::google::protobuf::util::JsonStringToMessage({buffer, (int) byte_size}, &config);
    std::cout << "Model Config:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;
    size_t max_batch1_byte_size = 0;
    std::vector<std::string> output_names;
    // Pre-allocate a memory pool for output buffers
    // Use the largest possible size among the outputs as size for each block
    for (const auto& output : config.output())
    {
        int batch1_byte_size = 1;
        for (const auto& dim : output.dims())
        {
            // FIXME: hard-coded value for variable dims
            if (dim == -1)
            {
                batch1_byte_size *= BERT_MAX_SEQ_LENGTH;
            }
            else
            {
                batch1_byte_size *= dim;
            }
        }
        batch1_byte_size *= GetDataTypeByteSize(output.data_type());
        if (batch1_byte_size <= 0)
        {
            FAIL("can't preallocate memory for variable size data type");
        }
        max_batch1_byte_size = std::max(max_batch1_byte_size, (size_t) batch1_byte_size);
        output_names.emplace_back(output.name());
    }

    size_t instance_count = 0;
    for (const auto& instance_group : config.instance_group())
    {
        // Don't really care about which gpu it is since we are allocating
        // pinned memory buffer with cudaHostAllocPortable.
        for (const auto& gpu : instance_group.gpus())
        {
            instance_count += instance_group.count();
        }
    }
    size_t max_batch_size = config.max_batch_size() == 0 ? 1 : config.max_batch_size();
    size_t pool_item_count = 2 * instance_count * (max_batch_size / min_sample_size + 1) * output_names.size();
    size_t pool_item_size = max_batch1_byte_size * max_sample_size;

    m_OutputBufferPool.reset(new PinnedMemoryPool(pool_item_count, pool_item_size, m_s2c_shmem));

    // Pre-allocate a growable request pool for inference requests
    std::vector<InputMetaData> inputs;
    m_IsDynamic = false;
    for (const auto& io : config.input())
    {
        InputMetaData input;
        std::get<0>(input) = io.name();
        std::get<1>(input) = DataTypeToTriton(io.data_type());
        auto& shape = std::get<2>(input);
        if (config.max_batch_size() != 0)
        {
            shape.push_back(1);
        }
        for (const auto& dim : io.dims())
        {
            m_IsDynamic |= (dim == -1);
            shape.push_back(dim);
        }
        m_InputTensors.emplace_back(input);
        inputs.emplace_back(std::move(input));
    }

    /*  Create the allocator that will be used to allocate buffers for
        the result tensors. */
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorNew(&m_Allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
        "creating response allocator");
    RequestPool::Create(
        2500000 /* initial_element_count */, m_Server.get(), this, m_ModelName, m_ModelVersion, inputs, output_names);

    // Prepare padding buffer in the case of DLRM. The model assumes
    // even batch size but some sample has odd batch size
    if (m_UseDlrmQsl)
    {
        for (const auto& io : config.output())
        {
            int64_t batch1_byte_size = TRITONSERVER_DataTypeByteSize(DataTypeToTriton(io.data_type()));
            for (const auto& dim : io.dims())
            {
                batch1_byte_size *= dim;
            }
            m_OutputPaddingSize = (size_t) batch1_byte_size;
        }
    }

    /* Set the number of warmup responses to 0 to prepare for next warmup */
    m_NumWarmupResponses = 0;
}

void Server_SUT::ModelStats()
{
    TRITONSERVER_Message* model_stats_message = nullptr;
    auto err
        = TRITONSERVER_ServerModelStatistics(m_Server.get(), m_ModelName.c_str(), m_ModelVersion, &model_stats_message);
    if (err != nullptr)
    {
        std::cerr << "failed to obtain stats message: " << TRITONSERVER_ErrorMessage(err) << std::endl;
        TRITONSERVER_ErrorDelete(err);
        return;
    };
    auto lms = std::unique_ptr<TRITONSERVER_Message, decltype(&TRITONSERVER_MessageDelete)>(
        model_stats_message, TRITONSERVER_MessageDelete);
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRITONSERVER_MessageSerializeToJson(model_stats_message, &buffer, &byte_size), "serializing stats message");

    std::cout << "Model Stats:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;
}

void Server_SUT::ModelMetadata()
{
    TRITONSERVER_Message* model_metadata_message = nullptr;
    FAIL_IF_ERR(
        TRITONSERVER_ServerModelMetadata(m_Server.get(), m_ModelName.c_str(), m_ModelVersion, &model_metadata_message),
        "obtaining metadata message");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_metadata_message, &buffer, &byte_size),
        "serializing model metadata message");

    std::cout << "Model Metadata:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;
    auto lmm = std::unique_ptr<TRITONSERVER_Message, decltype(&TRITONSERVER_MessageDelete)>(
        model_metadata_message, TRITONSERVER_MessageDelete);
}

void Server_SUT::Warmup(double duration_sec, double expected_qps)
{
    /* Notify user that we are starting the warmup */
    std::cout << "Starting Triton warmup" << std::endl;

    /* Calculate the number of inferences to send */
    int num_inferences = (int) (duration_sec * expected_qps);

    /* Keep track of the number of inferences that we have sent so far */
    int inferences_sent = 0;

    // Load a sample to RAM to use
    mlperf::QuerySampleIndex index{0}; // Arbitrary sample index
    std::vector<mlperf::QuerySampleIndex> samples;
    samples.push_back(index);
    m_SampleLibrary->LoadSamplesToRam(samples);

    while (inferences_sent < num_inferences)
    {
        /* Create the inference request provider, which provides the request
            header information as well as the actual data. */
        auto request_block = RequestPool::Obtain(0);
        if (m_UseDlrmQsl)
        {
            // Inputs will need to be re-added as the shape is different from run
            // to run
            TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

            auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibrary.get());
            // new batch size for the request
            auto num_pairs = qsl->GetNumUserItemPairs(index);
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*) qsl->GetSampleAddress(index, idx); // Get address of the query
                size_t single_sample_size = qsl->GetSampleSize(idx);

                auto& shape = std::get<2>(m_InputTensors[idx]);
                shape[0] = num_pairs;
                if (num_pairs % 2)
                {
                    shape[0] += 1;
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(request_block->m_Data,
                                    std::get<0>(m_InputTensors[idx]).c_str(), std::get<1>(m_InputTensors[idx]),
                                    shape.data(), shape.size()),
                        "re-adding input");
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                                    std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                                    single_sample_size * num_pairs, m_InputMemoryType, 0),
                        "appending input data");
                    // Add padding buffer
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                                    std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size,
                                    TRITONSERVER_MEMORY_CPU, 0),
                        "appending input data padding");
                }
                else
                {
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(request_block->m_Data,
                                    std::get<0>(m_InputTensors[idx]).c_str(), std::get<1>(m_InputTensors[idx]),
                                    shape.data(), shape.size()),
                        "re-adding input");
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                                    std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                                    single_sample_size * num_pairs, m_InputMemoryType, 0),
                        "appending input data");
                }
            }
        }
        else if (m_IsDynamic)
        {
            // Special handling as BERT is the only model uses dynamic shape
            //
            // Inputs will need to be re-added as the shape is different from run
            // to run
            TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

            size_t seq_len = GetSampleLength(m_SampleLibrary, index);
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data
                    = (int8_t*) m_SampleLibrary->GetSampleAddress(index, idx); // Get address of the query
                // Need to calculate the shape from data for dynamic case
                size_t input_size = seq_len * TRITONSERVER_DataTypeByteSize(std::get<1>(m_InputTensors[idx]));

                thread_local std::vector<int64_t> shape{1, 0};
                shape[1] = seq_len;
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), std::get<1>(m_InputTensors[idx]),
                                shape.data(), shape.size()),
                    "re-adding input");
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0),
                    "appending input data");
            }
        }
        else
        {
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data
                    = (int8_t*) m_SampleLibrary->GetSampleAddress(index, idx); // Get address of the query
                size_t input_size = m_SampleLibrary->GetSampleSize(idx);

                FAIL_IF_ERR(TRITONSERVER_InferenceRequestRemoveAllInputData(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str()),
                    "removing input data");
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0),
                    "appending input data");
            }
        }

        /* Set response callback for warmup */
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
                        request_block->m_Data, m_Allocator, m_OutputBufferPool.get(), WarmupResponseComplete, this),
            "appending input data");

        /* Actually perform inferences (asynchronously) */
        FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, nullptr), "running inference");
        inferences_sent += 1;
    }

    /* Wait for all the warmup inferences to complete */
    while (m_NumWarmupResponses < inferences_sent)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    /* Unload sample from RAM */
    m_SampleLibrary->UnloadSamplesFromRam(samples);

    /* Reset the number of warmup responses */
    m_NumWarmupResponses = 0;

    /* Notify user that we are done with the warmup */
    std::cout << "Finished Triton warmup" << std::endl;
}

void Server_SUT::TraceCaptureTimeStamp(TRITONSERVER_InferenceTrace* trace_ptr, const std::string comment)
{
    if (m_TraceManager != nullptr)
    {
        if (trace_ptr != nullptr)
        {
            uint64_t trace_id = 0;
            TRITONSERVER_InferenceTraceId(trace_ptr, &trace_id);
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            m_TraceManager->CaptureTimestamp(trace_id, TRITONSERVER_TRACE_LEVEL_MIN, comment, TIMESPEC_TO_NANOS(ts));
        }
    }
}

void Server_SUT::HandleSingleQuery(
    const std::vector<mlperf::QuerySample>& samples, int indexIntoQuerySample, int pool_idx)
{
    TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
    if (m_TraceManager != nullptr)
    {
        trace = m_TraceManager->SampleTrace();
        TraceCaptureTimeStamp(trace, "MLPerf Request START");
    }
#endif // TRITON_FRONTEND_TRACE

    auto request_block = RequestPool::Obtain(0);
    request_block->m_ResponseMetadata.m_ResponseId = samples[indexIntoQuerySample].id;
    request_block->m_ResponseMetadata.m_QuerySampleIdx = samples[indexIntoQuerySample].index;
    request_block->m_ResponseMetadata.m_TracePtr = trace;
    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
    {
        // Get a pointer to the input data
        int8_t* input_data = (int8_t*) m_SampleLibrary->GetSampleAddress(
            samples[indexIntoQuerySample].index, idx); // Get address of the query
        size_t input_size = m_SampleLibrary->GetSampleSize(idx);

        FAIL_IF_ERR(TRITONSERVER_InferenceRequestRemoveAllInputData(
                        request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str()),
            "removing input data");
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                        std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0),
            "appending input data");
    }

    /* Actually perform inference (asynchronously) */
    // For the userp field, pass in a pointer to a tuple with a pointer to the
    // SUT, a pointer to
    // the request provider, and the LoadGen response ID
    auto buffer_pool = m_OutputBufferPool.get();
    /* Set response callback for this request */
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                    m_OutputBufferPool.get(), ResponseComplete, &request_block->m_ResponseMetadata),
        "appending input data");

    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");

#if TRITON_FRONTEND_TRACE
    TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif // TRITON_FRONTEND_TRACE
}

void Server_SUT::HandleSingleBertQuery(
    const std::vector<mlperf::QuerySample>& samples, int indexIntoQuerySample, int pool_idx)
{
    // If its bert used samples in the sorted order
    TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
    if (m_TraceManager != nullptr)
    {
        trace = m_TraceManager->SampleTrace();
        TraceCaptureTimeStamp(trace, "MLPerf Request START");
    }
#endif // TRITON_FRONTEND_TRACE
       // Set the Request Provider
    auto request_block = RequestPool::Obtain(pool_idx);

    request_block->m_ResponseMetadata.m_ResponseId = samples[indexIntoQuerySample].id;
    request_block->m_ResponseMetadata.m_QuerySampleIdx = samples[indexIntoQuerySample].index;
    request_block->m_ResponseMetadata.m_TracePtr = trace;

    // Special handling as BERT is the only model uses dynamic shape
    //
    // Inputs will need to be re-added as the shape is different from run
    // to run
    TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

    size_t seq_len = GetSampleLength(m_SampleLibrary, samples[indexIntoQuerySample].index);
    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
    {
        // Get a pointer to the input data
        int8_t* input_data = (int8_t*) m_SampleLibrary->GetSampleAddress(
            samples[indexIntoQuerySample].index, idx); // Get address of the query
        // Need to calculate the shape from data for dynamic case
        size_t input_size = seq_len * TRITONSERVER_DataTypeByteSize(std::get<1>(m_InputTensors[idx]));

        thread_local std::vector<int64_t> shape{1, 0};
        shape[1] = seq_len;
        FAIL_IF_ERR(
            TRITONSERVER_InferenceRequestAddInput(request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
            "re-adding input");
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                        std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0),
            "appending input data");
    }
    /* Actually perform inference (asynchronously) */
    // For the userp field, pass in a pointer to a tuple with a pointer to the
    // SUT, a pointer to
    // the request provider, and the LoadGen response ID
    auto buffer_pool = m_OutputBufferPool.get();
    /* Set response callback for this request */
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                    m_OutputBufferPool.get(), ResponseComplete, &request_block->m_ResponseMetadata),
        "appending input data");

    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");

#if TRITON_FRONTEND_TRACE
    TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif // TRITON_FRONTEND_TRACE
}

void Server_SUT::HandleSingleDlrmQuery(
    const std::vector<mlperf::QuerySample>& samples, int indexIntoQuerySample, int pool_idx)
{
    TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
    if (m_TraceManager != nullptr)
    {
        trace = m_TraceManager->SampleTrace();
        TraceCaptureTimeStamp(trace, "MLPerf Request START");
    }
#endif // TRITON_FRONTEND_TRACE
       // Set the Request Provider
    auto request_block = RequestPool::Obtain(pool_idx);

    request_block->m_ResponseMetadata.m_ResponseId = samples[indexIntoQuerySample].id;
    request_block->m_ResponseMetadata.m_QuerySampleIdx = samples[indexIntoQuerySample].index;
    request_block->m_ResponseMetadata.m_TracePtr = trace;
    // Inputs will need to be re-added as the shape is different from run
    // to run
    TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

    auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibrary.get());
    // new batch size for the request
    auto num_pairs = qsl->GetNumUserItemPairs(samples[indexIntoQuerySample].index);
    auto l_InputTensors = m_InputTensors;

    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
    {
        // Get a pointer to the input data
        int8_t* input_data
            = (int8_t*) qsl->GetSampleAddress(samples[indexIntoQuerySample].index, idx); // Get address of the query
        const size_t single_sample_size = qsl->GetSampleSize(idx);

        auto& shape = std::get<2>(l_InputTensors[idx]);
        shape[0] = num_pairs;
        if (num_pairs % 2)
        {
            shape[0] += 1;
            FAIL_IF_ERR(
                TRITONSERVER_InferenceRequestAddInput(request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                    std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                "re-adding input");
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                            std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size * num_pairs,
                            m_InputMemoryType, 0),
                "appending input data");
            // Add padding buffer
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                            std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size,
                            TRITONSERVER_MEMORY_CPU, 0),
                "appending input data padding");
        }
        else
        {
            FAIL_IF_ERR(
                TRITONSERVER_InferenceRequestAddInput(request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                    std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                "re-adding input");
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                            std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size * num_pairs,
                            m_InputMemoryType, 0),
                "appending input data");
        }
    }
    request_block->m_ResponseMetadata.m_PaddingSize = (num_pairs % 2) ? m_OutputPaddingSize : 0;

    /* Actually perform inference (asynchronously) */
    // For the userp field, pass in a pointer to a tuple with a pointer to the
    // SUT, a pointer to
    // the request provider, and the LoadGen response ID
    auto buffer_pool = m_OutputBufferPool.get();
    /* Set response callback for this request */
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                    m_OutputBufferPool.get(), ResponseComplete, &request_block->m_ResponseMetadata),
        "appending input data");

    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");

#if TRITON_FRONTEND_TRACE
    TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif // TRITON_FRONTEND_TRACE
}

void Server_SUT::Worker()
{
    // Grab samples from shared memory
    std::vector<mlperf::QuerySample> samples;
    std::vector<mlperf::QuerySampleIndex> sample_ids;

    while (true)
    {
        IPCcomm m_;
        bool rcvd = m_c2s_state_mq->receive(m_);
        if (rcvd)
        {
            switch (m_)
            {
            // terminate condition
            case IPCcomm::Terminate: { return;
            }
            // IssueQuery
            case IPCcomm::IssueQuery:
            {
                samples.clear();
                QuerySample_vec* QS_vec_data = m_c2s_shmem->find<QuerySample_vec>("QS").first;
                for (auto& s_ : *QS_vec_data)
                {
                    samples.push_back(s_);
                }
                m_c2s_shmem->destroy<QuerySampleIndex_vec>("QS");

                // Currently only DLRM will suffer from the IssueQuery bottleneck,
                // so always use the current thread for other benchmarks
                if (!m_UseDlrmQsl || samples.size() < 100)
                {
                    IssueQueryInternal(samples, 0, samples.size());
                }
                else
                {
                    auto a1 = std::async(
                        std::launch::async, &Server_SUT::IssueQueryInternal, this, samples, 0, samples.size() / 2);
                    auto a2 = std::async(std::launch::async, &Server_SUT::IssueQueryInternal, this, samples,
                        samples.size() / 2, samples.size());
                    a1.get();
                    a2.get();
                }
                break;
            }
            // LoadSamples
            case IPCcomm::IssueLoad:
            {
                sample_ids.clear();
                QuerySampleIndex_vec* QSI_vec_data = m_c2s_shmem->find<QuerySampleIndex_vec>("QSIL").first;
                for (auto& s_ : *QSI_vec_data)
                {
                    sample_ids.push_back(s_);
                }
                m_c2s_shmem->destroy<QuerySampleIndex_vec>("QSIL");

                m_SampleLibrary->LoadSamplesToRam(sample_ids);

                m_ = IPCcomm::LoadDone;
                m_s2c_state_mq->send_it(m_);
                break;
            }
            // UnloadSamples
            case IPCcomm::IssueUnload:
            {
                sample_ids.clear();
                QuerySampleIndex_vec* QSI_vec_data = m_c2s_shmem->find<QuerySampleIndex_vec>("QSIU").first;
                for (auto& s_ : *QSI_vec_data)
                {
                    sample_ids.push_back(s_);
                }
                m_c2s_shmem->destroy<QuerySampleIndex_vec>("QSIU");

                m_SampleLibrary->UnloadSamplesFromRam(sample_ids);

                m_ = IPCcomm::UnloadDone;
                m_s2c_state_mq->send_it(m_);
                break;
            }
            default: assert(false);
            }
        }
    }
}

void Server_SUT::IssueQueryInternal(const std::vector<mlperf::QuerySample>& samples, size_t start_idx, size_t end_idx)
{
    // Currently BERT is the only model where dynamic
    bool isBertBenchmark = m_IsDynamic;
    bool isSingleQuery = samples.size() == 1;
    bool isDLRMBenchmark = m_UseDlrmQsl;

    // Avoid allocations in single-stream code-path, handle single query and
    // return
    if (isSingleQuery && !isDLRMBenchmark)
    {
        if (isBertBenchmark)
        {
            HandleSingleBertQuery(samples, 0, 0);
        }
        else
        {
            HandleSingleQuery(samples, 0, 0);
        }
        return;
    }

    size_t pool_idx = start_idx == 0 ? 0 : 1;

    if (isBertBenchmark)
    {
        std::vector<std::pair<int, int>> sequenceSamplePosAndLength(samples.size());
        for (int samplePos = 0; samplePos < samples.size(); ++samplePos)
        {
            sequenceSamplePosAndLength[samplePos] = std::make_pair(
                samplePos, static_cast<int>(GetSampleLength(m_SampleLibrary, samples[samplePos].index)));
        }
        // Sort the samples according to sequence length
        // Sort samples in the descending order of sentence length
        std::sort(sequenceSamplePosAndLength.begin(), sequenceSamplePosAndLength.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool { return a.second > b.second; });
        for (size_t i = start_idx; i < end_idx; i++)
        {
            // If its bert used samples in the sorted order
            int indexIntoQuerySample = isBertBenchmark ? sequenceSamplePosAndLength[i].first : i;
            HandleSingleBertQuery(samples, indexIntoQuerySample, pool_idx);
        }
    }
    else
    {
        for (size_t i = start_idx; i < end_idx; i++)
        {
            if (isDLRMBenchmark)
            {
                HandleSingleDlrmQuery(samples, i, pool_idx);
            }
            else
            {
                HandleSingleQuery(samples, i, pool_idx);
            }
        }
    }
}

void Server_SUT::Completion(TRITONSERVER_InferenceResponse* response, const ResponseMetaData* response_metadata)
{
    /* Make sure we have a valid response */
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(response), "response");

    /* Extract and process the response data */
    const char* name;
    TRITONSERVER_DataType datatype;
    void* userp;
    const int64_t* shape;
    uint64_t dim_count;
    const void* output0_content;
    size_t output0_byte_size;
    TRITONSERVER_MemoryType output0_memory_type;
    int64_t output0_memory_type_id;
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutput(response, 0 /* index */, &name, &datatype, &shape, &dim_count,
                    &output0_content, &output0_byte_size, &output0_memory_type, &output0_memory_type_id, &userp),
        "getting output0 result");
    // Recast the output pointer as a uintptr_t (for LoadGen)
    uintptr_t output0_result = reinterpret_cast<uintptr_t>(output0_content);

    /* Call QuerySamplesComplete */
    mlperf::QuerySampleResponse loadgen_response{
        response_metadata->m_ResponseId, output0_result, output0_byte_size - response_metadata->m_PaddingSize};

    // callback if it exists
    if (m_ResponseCallback)
    {
        std::vector<::mlperf::QuerySampleIndex> response_indices = {response_metadata->m_QuerySampleIdx};
        m_ResponseCallback(&loadgen_response, response_indices, 1);
    }
    
    // put samples vector into shared memory segment
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(m_OutputBufferPool->get_base_address());
    mlperf::QuerySampleResponse loadgen_response_ipc{
        loadgen_response.id, loadgen_response.data - base_addr, loadgen_response.size};
    m_s2c_resp_mq->send_it(loadgen_response_ipc);
}

void Server_SUT::Done()
{
    RequestPool::Destroy();

    /* Delete the response allocator since we are done with it */
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(m_Allocator), "deleting response allocator");

    /* Reset the server pointer to nullptr to ensure Init() is called before the
     * server is used
     * again */
    m_Server = nullptr;
}

void Server_SUT::FlushQueries()
{
}

void Server_SUT::ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns)
{
}

// connect real LoadGen with Server_SUTs
void SUTShim::Init()
{
    m_finished = false;

    if (m_scenario == "singlestream")
        m_server_sticky_counter_cap = 1;
    
    if (m_scenario == "server")
    {
        m_server_sticky_counter_cap = 1;

        // set up sync queue
        for (auto& u_ : m_mig_uuids)
        {
            SampleSQ_ptr sq = std::make_shared<lwis::SyncQueue<mlperf::QuerySample>>();
            m_SQ_map.emplace(u_, sq);
            m_SQ_handlers[u_] = std::thread(&SUTShim::ServerQueueHandler, this,
                                            u_,
                                            m_max_server_buffer_size,
                                            m_server_queue_timeout);
        }
    }
        
}

void SUTShim::Done()
{
    m_finished = true;

    if (m_scenario == "server")
        for (auto& u_ : m_mig_uuids)
            m_SQ_handlers[u_].join();
}

void SUTShim::ServerQueueHandler(std::string mig_uuid,
                                 uint64_t max_sq_size, 
                                 std::chrono::microseconds sq_timeout)
{
    while (!is_finished())
    {
        std::deque<mlperf::QuerySample> samples;
        while (samples.empty())
        {
            m_SQ_map[mig_uuid]->acquire(samples, sq_timeout, max_sq_size, false);
            if (is_finished())
                break;
        }

        if (!samples.empty())
        {
            try
            {
                // put samples vector into shared memory segment
                const QuerySample_alloc alloc_QS(m_c2s_shmem_map[mig_uuid]->get_segment_manager());
                QuerySample_vec* QS_vec_data = m_c2s_shmem_map[mig_uuid]->construct<QuerySample_vec>("QS")(alloc_QS);
                for (auto& s_ : samples)
                    QS_vec_data->push_back(s_);

                // and send IssueQuery cmd
                IPCcomm m_ = IPCcomm::IssueQuery;
                m_c2s_state_mq_map[mig_uuid]->send_it(m_);
            }
            catch (bip::interprocess_exception& ex)
            {
                std::cerr << "Met exception while sending Samples/IssueQuery to " << mig_uuid << " : " << ex.what()
                            << std::endl;
                assert(false);
            }
        }
    }
}

void SUTShim::Do_LoadSamplesToRam(std::vector<mlperf::QuerySampleIndex>& samples)
{
    // Initiate MIGs to read data associated with sample indices into Memory
    // Grab c2s shmem & c2s mq, for each MIG and initiate IssueLoad
    for (auto& u_ : m_mig_uuids)
    {
        try
        {
            // put samples vector into shared memory segment
            const QuerySampleIndex_alloc alloc_QS(m_c2s_shmem_map[u_]->get_segment_manager());
            QuerySampleIndex_vec* QSI_vec_data = m_c2s_shmem_map[u_]->construct<QuerySampleIndex_vec>("QSIL")(alloc_QS);
            for (auto& s_ : samples)
                QSI_vec_data->push_back(s_);

            // and send IssueLoad cmd
            IPCcomm m_ = IPCcomm::IssueLoad;
            m_c2s_state_mq_map[u_]->send_it(m_);
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while sending Samples/LoadSamples to " << u_ << " : " << ex.what() << std::endl;
            assert(false);
        }
    }

    // Wait for LoadDone
    for (auto& u_ : m_mig_uuids)
    {
        IPCcomm m_;
        try
        {
            m_s2c_state_mq_map[u_]->receive_it(m_);
            assert(rcvd && m_ == IPCcomm::LoadDone);
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while waiting LoadDone from " << u_ << " : " << ex.what() << std::endl;
            assert(false);
        }
    }
}

void SUTShim::Do_UnloadSamplesFromRam(std::vector<mlperf::QuerySampleIndex>& samples)
{
    // Grab c2s shmem & c2s mq, for each MIG and initiate IssueLoad
    for (auto& u_ : m_mig_uuids)
    {
        try
        {
            // put samples vector into shared memory segment
            const QuerySampleIndex_alloc alloc_QS(m_c2s_shmem_map[u_]->get_segment_manager());
            QuerySampleIndex_vec* QSI_vec_data = m_c2s_shmem_map[u_]->construct<QuerySampleIndex_vec>("QSIU")(alloc_QS);
            for (auto& s_ : samples)
                QSI_vec_data->push_back(s_);

            // and send IssueLoad cmd
            IPCcomm m_ = IPCcomm::IssueUnload;
            m_c2s_state_mq_map[u_]->send_it(m_);
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while sending Samples/UnloadSamples to " << u_ << " : " << ex.what()
                      << std::endl;
            assert(false);
        }
    }

    // Wait for UnloadDone
    for (auto& u_ : m_mig_uuids)
    {
        IPCcomm m_;
        try
        {
            m_s2c_state_mq_map[u_]->receive_it(m_);
            assert(rcvd && m_ == IPCcomm::UnloadDone);
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while waiting UnloadDone from " << u_ << " : " << ex.what() << std::endl;
            assert(false);
        }
    }
}

void SUTShim::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    // per Scenario, it may be different chops of samples
    if (m_scenario == "offline")
    {
        // put samples into shared mem, in a splitted form towards all MIGs
        boost::unordered::unordered_map<std::string, QuerySampleVec_ptr> samples_map;

        // round robin on MIGs on samples
        uint i_ = 0;
        for (auto& u_ : m_mig_uuids)
        {
            std::vector<mlperf::QuerySample> s_;
            s_.clear();
            // clang-format off
            if (i_ < samples.size())
                boost::copy(samples 
                    | boost::adaptors::sliced(i_, samples.size()) 
                    | boost::adaptors::strided(m_num_migs),
                    std::back_inserter(s_));
            // clang-format on
            QuerySampleVec_ptr s_ptr = std::make_shared<std::vector<mlperf::QuerySample>>(s_);
            samples_map.emplace(u_, s_ptr);
            i_++;
        }

        // Grab c2s shmem & c2s mq, for each MIG and initiate IssueQuery
        for (auto& u_ : m_mig_uuids)
        {
            if (samples_map[u_]->size() > 0)
            {
                try
                {
                    // put samples vector into shared memory segment
                    bip::shared_memory_object::remove("QS");
                    m_c2s_shmem_map[u_]->destroy<QuerySampleIndex_vec>("QS");
                    const QuerySample_alloc alloc_QS(m_c2s_shmem_map[u_]->get_segment_manager());
                    QuerySample_vec* QS_vec_data = m_c2s_shmem_map[u_]->find_or_construct<QuerySample_vec>("QS")(alloc_QS);
                    for (auto& s_ : *samples_map[u_])
                        QS_vec_data->push_back(s_);

                    // and send IssueQuery cmd
                    IPCcomm m_ = IPCcomm::IssueQuery;
                    m_c2s_state_mq_map[u_]->send_it(m_);
                }
                catch (bip::interprocess_exception& ex)
                {
                    std::cerr << "Met exception while sending Samples/IssueQuery to " << u_ << " : " << ex.what()
                              << std::endl;
                    assert(false);
                }
            }
        }
    }
    else if (m_scenario == "singlestream")
    {
        // round robin on MIGs on samples
        std::string tgt_mig_uuid = m_mig_uuids[getServerTurn()];
        // Grab c2s shmem & c2s mq, for target MIG and initiate IssueQuery
        try
        {
            // put samples vector into shared memory segment
            bip::shared_memory_object::remove("QS");
            m_c2s_shmem_map[tgt_mig_uuid]->destroy<QuerySampleIndex_vec>("QS");
            const QuerySample_alloc alloc_QS(m_c2s_shmem_map[tgt_mig_uuid]->get_segment_manager());
            QuerySample_vec* QS_vec_data = m_c2s_shmem_map[tgt_mig_uuid]->construct<QuerySample_vec>("QS")(alloc_QS);
            for (auto& s_ : samples)
                QS_vec_data->push_back(s_);

            // and send IssueQuery cmd
            IPCcomm m_ = triton_frontend::IPCcomm::IssueQuery;
            m_c2s_state_mq_map[tgt_mig_uuid]->send_it(m_);
        }
        catch (bip::interprocess_exception& ex)
        {
            std::cerr << "Met exception while sending Samples/IssueQuery to " << tgt_mig_uuid << " : " << ex.what()
                        << std::endl;
            assert(false);
        }
        moveServerTurn();
    }
    else if (m_scenario == "server")
    {
        // round robin on MIGs on samples
        std::string tgt_mig_uuid = m_mig_uuids[getServerTurn()];
        // Grab c2s shmem & c2s mq, for target MIG and initiate IssueQuery
        queue_to_SQ(tgt_mig_uuid, samples);
        moveServerTurn();
    }
    else
    {
        std::cerr << "Scenario other than Offline/Single-Stream/Server is supported yet"
                  << std::endl;
        assert(false);
    }
}

}; // namespace triton_frontend
