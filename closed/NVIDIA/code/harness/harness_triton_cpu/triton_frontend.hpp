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

#ifndef __TRITON_FRONTEND_HPP__
#define __TRITON_FRONTEND_HPP__

// TRITON
#define TRITON_ENABLE_GPU 0
#include "src/servers/tracer.h"
#include "triton/core/tritonserver.h"

// QSL
#include "qsl_cpu.hpp"

// LoadGen
#include "system_under_test.h"

// General C++
#include <atomic>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <thread>

constexpr size_t BERT_MAX_SEQ_LENGTH{384};

namespace triton_frontend {

using InputMetaData = std::tuple<std::string, TRITONSERVER_DataType, std::vector<int64_t>>;

class Server_SUT;
typedef std::shared_ptr<Server_SUT> ServerSUTPtr_t;

class PinnedMemoryPool
{
  public:
    struct MemoryBlock
    {
        MemoryBlock() : m_NextBlock(nullptr), m_Data(nullptr) {}
        MemoryBlock* m_NextBlock;
        char* m_Data;
    };

    PinnedMemoryPool(const size_t element_count, const size_t element_byte_size);
    ~PinnedMemoryPool();

    // Note that there is no checking for empty list
    MemoryBlock* Obtain()
    {
        MemoryBlock* res;
        {
            std::lock_guard<std::mutex> lk(m_ListMtx);
            res = m_Head;
            m_Head = m_Head->m_NextBlock;
        }
        return res;
    }

    void Release(MemoryBlock* block)
    {
        {
            std::lock_guard<std::mutex> lk(m_ListMtx);
            block->m_NextBlock = m_Head;
            m_Head = block;
        }
    }

  private:
    std::mutex m_ListMtx;
    MemoryBlock* m_Head;
    std::vector<MemoryBlock> m_Blocks;
    char* m_Buffer;
};

struct ResponseMetaData
{
    ResponseMetaData() : m_ServerPtr(nullptr), m_TracePtr(nullptr), m_PaddingSize(0) {}
    ResponseMetaData(Server_SUT* server_ptr)
        : m_ServerPtr(server_ptr), m_TracePtr(nullptr), m_PaddingSize(0)
    {
    }
    Server_SUT* m_ServerPtr;
    mlperf::ResponseId m_ResponseId;
    mlperf::QuerySampleIndex m_QuerySampleIdx;
    TRITONSERVER_InferenceTrace* m_TracePtr;
    // FIXME assuming that there is only one output
    size_t m_PaddingSize;
};

// A singlenton class of request pool for all inference requests,
// completed requests will be reused instead of being deleted.
class RequestPool
{
  public:
    struct Block
    {
        Block() : m_AssignedPool(nullptr), m_NextBlock(nullptr), m_Data(nullptr) {}
        RequestPool* m_AssignedPool;
        Block* m_NextBlock;
        TRITONSERVER_InferenceRequest* m_Data;
        // Not a great place for holding response metadata as this imposes
        // release order, response then request. But it is handy as request
        // and response are not decoupled.
        ResponseMetaData m_ResponseMetadata;
    };

    ~RequestPool();

    static void Create(const size_t initial_element_count, TRITONSERVER_Server* server,
                       Server_SUT* server_sut, const std::string& model_name,
                       const uint32_t model_version, const std::vector<InputMetaData>& inputs,
                       const std::vector<std::string>& outputs);

    static void Destroy();

    // Will create new block if the pool is currently empty, function caller
    // should check if the block is initialized (m_Data != nullptr) and call
    // InitInferenceRequest() if not.
    static Block* Obtain(size_t pool_idx)
    {
        auto& instance = instances_[pool_idx];
        Block* res;
        // The implementation ensures that there is no concurrent obtain of the same instance
        if(instance->m_Head == nullptr)
        {
            std::lock_guard<std::mutex> lk(instance->m_ReleasedMtx);
            instance->m_Head = instance->m_ReleasedHead;
            instance->m_ReleasedHead = nullptr;
        }
        res = instance->m_Head;
        if(res != nullptr)
        {
            instance->m_Head = instance->m_Head->m_NextBlock;
        }
        else
        {
            instance->m_Blocks.emplace_back();
            res = &instance->m_Blocks.back();
            instance->InternalInitInferenceRequest(res);
        }
        return res;
    }

    static void Release(Block* block)
    {
        auto instance = block->m_AssignedPool;
        std::lock_guard<std::mutex> lk(instance->m_ReleasedMtx);
        block->m_NextBlock = instance->m_ReleasedHead;
        instance->m_ReleasedHead = block;
    }

  private:
    RequestPool(const size_t initial_element_count, TRITONSERVER_Server* server,
                Server_SUT* server_sut, const std::string& model_name, const uint32_t model_version,
                const std::vector<InputMetaData>& inputs, const std::vector<std::string>& outputs);

    void InternalInitInferenceRequest(RequestPool::Block* block);

    static std::vector<std::unique_ptr<RequestPool>> instances_;

    std::mutex m_ReleasedMtx;
    Block* m_Head;
    Block* m_ReleasedHead;
    // Use list so that we may add new blocks without invalidating
    // pointer / reference to existing blocks
    std::list<Block> m_Blocks;

    // Metadata to construct an TRTServerV2_InferenceRequest object
    TRITONSERVER_Server* m_Server;
    Server_SUT* m_ServerSUT;
    const std::string m_ModelName;
    const int64_t m_ModelVersion;

    std::vector<InputMetaData> m_Inputs;
    std::vector<std::string> m_Outputs;
};

using PoolBlockPair = std::pair<PinnedMemoryPool*, PinnedMemoryPool::MemoryBlock*>;

class Server_SUT : public mlperf::SystemUnderTest
{
  public:
    Server_SUT(std::string name, std::string model_repo_path, std::string model_name,
               uint32_t model_version, bool use_dlrm_qsl, bool start_from_device, bool pinned_input)
        : m_Name(name), m_ModelRepositoryPath(model_repo_path), m_ModelName(model_name),
          m_ModelVersion(model_version), m_UseDlrmQsl(use_dlrm_qsl)
    {
        m_InputMemoryType = TRITONSERVER_MEMORY_CPU_PINNED;
    }
    ~Server_SUT() {}

    void Init(size_t min_sample_size = 1, size_t max_sample_size = 1,
              size_t buffer_manager_thread_count = 0);
    void ModelMetadata();
    void ModelStats();
    void AddSampleLibrary(qsl::SampleLibraryPtr_t sl)
    {
        m_SampleLibrary = sl;
    }
    void Warmup(double duration_sec, double expected_qps);
    void IncrementWarmupResponses()
    {
        m_NumWarmupResponses += 1;
    }
    void Completion(TRITONSERVER_InferenceResponse* response,
                    const ResponseMetaData* response_metadata);
    void Done();
    void SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses,
                                                std::vector<::mlperf::QuerySampleIndex>& sample_ids,
                                                size_t response_count)>
                                 callback)
    {
        m_ResponseCallback = callback;
    }

    // SUT virtual interface
    virtual const std::string& Name() const
    {
        return m_Name;
    }
    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples);
    virtual void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns);
    virtual void FlushQueries();

    std::shared_ptr<nvidia::inferenceserver::TraceManager> m_TraceManager;

    void TraceCaptureTimeStamp(TRITONSERVER_InferenceTrace* trace_ptr, const std::string comment);

  private:
    void IssueQueryInternal(const std::vector<mlperf::QuerySample>& samples, size_t start_idx,
                            size_t end_idx);

    void HandleSingleDlrmQuery(const std::vector<mlperf::QuerySample>& samples,
                               int indexIntoQuerySample, int pool_idx);
    void HandleSingleBertQuery(const std::vector<mlperf::QuerySample>& samples,
                               int indexIntoQuerySample, int pool_idx);
    void HandleSingleQuery(const std::vector<mlperf::QuerySample>& samples,
                           int indexIntoQuerySample, int pool_idx);
    const std::string m_ModelRepositoryPath;
    const std::string m_Name;
    // The tensors must be in the same order as how the I/O are presented.
    // In other words, the index for a name must match the index of the
    // tensor in the sample library.
    bool m_IsDynamic = false;
    std::vector<InputMetaData> m_InputTensors;
    size_t m_OutputPaddingSize;

    // Currently only one output is needed, this may subject to change.
    // But note that MLPerf harness expects output to be contiguous (inferred
    // from the response struct), so need to revisit the struct and whether
    // we need to change the way to allocate output buffer if TRITON harness
    // needs to support multiple outputs.
    qsl::SampleLibraryPtr_t m_SampleLibrary{nullptr};
    std::shared_ptr<TRITONSERVER_Server> m_Server = nullptr;
    TRITONSERVER_ResponseAllocator* m_Allocator = nullptr;
    std::string m_ModelName;
    const uint32_t m_ModelVersion;
    std::atomic<uint64_t> m_NumWarmupResponses{0};
    // Query sample response callback.
    std::function<void(::mlperf::QuerySampleResponse* responses,
                       std::vector<::mlperf::QuerySampleIndex>& sample_ids, size_t response_count)>
        m_ResponseCallback;
    // Pinned memory pool for output buffers:
    // we take advantage of the fact that the model has only one output and
    // the batch 1 size is fixed.
    // # buffer = 2 * # instance * (max_batch_size // min_sample_size + 1),
    // since each instance may ask for two sets of buffers in advance
    // in extreme case.
    std::unique_ptr<PinnedMemoryPool> m_OutputBufferPool;

    // The memory type of the sample data,
    // ideally the memory type should be retrieved from QSL.
    TRITONSERVER_MemoryType m_InputMemoryType;
    bool m_UseDlrmQsl = false;
};
}; // namespace triton_frontend

#endif
