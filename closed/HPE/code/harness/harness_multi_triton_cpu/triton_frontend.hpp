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
#include "multi_qsl.hpp"

// LoadGen
#include "system_under_test.h"

// TODO: remove LWIS for SyncQueue
// #include "lwis.hpp"

// General C++
#include <atomic>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <regex>
#include <thread>

// shared memory interprocess communication
#include "boost/date_time/posix_time/posix_time_types.hpp"
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>

// other boost utils
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/unordered_map.hpp>

// NUMA
#include <numa.h>

// short namespace for boost::interprocess
namespace bip = boost::interprocess;

constexpr size_t BERT_MAX_SEQ_LENGTH{384};

// #FIXME: COPIED LWIS DEPENDENCIES
#define TIMER_START(s)
#define TIMER_END(s)
#include <chrono>
namespace lwis {
template<typename T>
class SyncQueue
{
  public:
    typedef typename std::deque<T>::iterator iterator;

    SyncQueue() {}

    bool empty()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        return m_Queue.empty();
    }

    void insert(const std::vector<T>& values)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.insert(m_Queue.end(), values.begin(), values.end());
        }
        m_Condition.notify_one();
    }
    void acquire(std::deque<T>& values, std::chrono::microseconds duration = 10000, size_t size = 1,
                 bool limit = false)
    {
        size_t remaining = 0;

        TIMER_START(m_Mutex_create);
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            TIMER_END(m_Mutex_create);
            TIMER_START(m_Condition_wait_for);
            m_Condition.wait_for(l, duration, [=] { return m_Queue.size() >= size; });
            TIMER_END(m_Condition_wait_for);

            if(!limit || m_Queue.size() <= size)
            {
                TIMER_START(swap);
                values.swap(m_Queue);
                TIMER_END(swap);
            }
            else
            {
                auto beg = m_Queue.begin();
                auto end = beg + size;
                TIMER_START(values_insert);
                values.insert(values.end(), beg, end);
                TIMER_END(values_insert);
                TIMER_START(m_Queue_erase);
                m_Queue.erase(beg, end);
                TIMER_END(m_Queue_erase);
                remaining = m_Queue.size();
            }
        }

        // wake up any waiting threads
        if(remaining)
            m_Condition.notify_one();
    }

    void push_back(T const& v)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.push_back(v);
        }
        m_Condition.notify_one();
    }
    void emplace_back(T const& v)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.emplace_back(v);
        }
        m_Condition.notify_one();
    }
    T front()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Condition.wait(l, [=] { return !m_Queue.empty(); });
        T r(std::move(m_Queue.front()));
        return r;
    }
    T front_then_pop()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Condition.wait(l, [=] { return !m_Queue.empty(); });
        T r(std::move(m_Queue.front()));
        m_Queue.pop_front();
        return r;
    }
    void pop_front()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Queue.pop_front();
    }

  private:
    mutable std::mutex m_Mutex;
    std::condition_variable m_Condition;

    std::deque<T> m_Queue;
};
} // namespace lwis

namespace triton_frontend {

using InputMetaData = std::tuple<std::string, TRITONSERVER_DataType, std::vector<int64_t>>;

class Server_SUT;
typedef std::shared_ptr<Server_SUT> ServerSUTPtr_t;

class SUTShim;
typedef std::shared_ptr<SUTShim> SUTShim_ptr;

// minimal IPC communication through message_queue; helps shared_mem without
// lock
enum IPCcomm
{
    None = 0,
    StartWarmup,
    WarmupDone,
    IssueQuery = 10,
    IssueLoad,
    IssueUnload,
    LoadDone = 20,
    UnloadDone,
    Initialized = 30,
    Terminate
};

template<class T>
class msg_q
{
  public:
    msg_q(const char* const name) : m_queue{bip::open_only, name} {}

    msg_q(const char* const name, const unsigned max_queue_size)
        : msg_q{name, max_queue_size, remove_queue(name)}
    {
    }

    bool send(T& m_)
    {
        return m_queue.try_send(&m_, sizeof(m_), 0);
    }
    void send_it(T& m_)
    {
        while(!m_queue.try_send(&m_, sizeof(m_), 0))
        {
        };
    }

    bool receive(T& result)
    {
        bip::message_queue::size_type recvsize;
        unsigned recvpriority;
        return m_queue.try_receive(&result, sizeof(result), recvsize, recvpriority);
    }

    void receive_it(T& result)
    {
        bip::message_queue::size_type recvsize;
        unsigned recvpriority;
        while(!m_queue.try_receive(&result, sizeof(result), recvsize, recvpriority))
        {
        };
    }

    std::size_t get_num_msg()
    {
        return m_queue.get_num_msg();
    }

    bool is_avail()
    {
        return (m_queue.get_max_msg() > m_queue.get_num_msg());
    }

  private:
    struct delete_it
    {
    };

    delete_it remove_queue(const char* const name)
    {
        bip::message_queue::remove(name);
        return delete_it{};
    }

    msg_q(const char* const name, const unsigned max_queue_size, delete_it)
        : m_queue{bip::create_only, name, max_queue_size, sizeof(T)}
    {
    }

    bip::message_queue m_queue;
};

// some pointers
typedef std::shared_ptr<bip::managed_shared_memory> sharedMSM_ptr;
typedef std::shared_ptr<bip::shared_memory_object> sharedSMO_ptr;
typedef std::shared_ptr<bip::mapped_region> sharedMR_ptr;
typedef std::shared_ptr<std::vector<mlperf::QuerySample>> QuerySampleVec_ptr;
typedef std::shared_ptr<lwis::SyncQueue<mlperf::QuerySample>> SampleSQ_ptr;
typedef boost::unordered::unordered_map<std::string, sharedMSM_ptr> sharedMSMMap_ptr;
typedef boost::unordered::unordered_map<std::string, sharedMR_ptr> sharedMRMap_ptr;
typedef boost::unordered::unordered_map<std::string, uintptr_t> sharedBAMap_ptr;
template<class T>
using sharedMQ_ptr = std::shared_ptr<msg_q<T>>;
template<class T>
using sharedMap_ptr = boost::unordered::unordered_map<std::string, sharedMQ_ptr<T>>;

// shared memory related setup
typedef bip::allocator<mlperf::QuerySample, bip::managed_shared_memory::segment_manager>
    QuerySample_alloc;
typedef bip::allocator<mlperf::QuerySampleIndex, bip::managed_shared_memory::segment_manager>
    QuerySampleIndex_alloc;

typedef bip::vector<mlperf::QuerySample, QuerySample_alloc> QuerySample_vec;
typedef bip::vector<mlperf::QuerySampleIndex, QuerySampleIndex_alloc> QuerySampleIndex_vec;

class PinnedMemoryPool
{
  public:
    struct MemoryBlock
    {
        MemoryBlock() : m_NextBlock(nullptr), m_Data(nullptr) {}
        MemoryBlock* m_NextBlock;
        char* m_Data;
    };

    PinnedMemoryPool(const size_t element_count, const size_t element_byte_size,
                     sharedMR_ptr s2c_shmem);
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
    char* get_base_address()
    {
        return m_Buffer;
    }

  private:
    std::mutex m_ListMtx;
    MemoryBlock* m_Head;
    std::vector<MemoryBlock> m_Blocks;
    char* m_Buffer;
    sharedMR_ptr m_s2c_shmem;
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
        // The implementation ensures that there is no concurrent obtain of the same
        // instance
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
               uint32_t model_version, bool use_dlrm_qsl, bool start_from_device, bool pinned_input,
               std::string mig_uuid, sharedMQ_ptr<IPCcomm> c2s_state_mq,
               sharedMQ_ptr<IPCcomm> s2c_state_mq,
               sharedMQ_ptr<mlperf::QuerySampleResponse> s2c_resp_mq, sharedMSM_ptr c2s_shmem,
               sharedMR_ptr s2c_shmem)
        : m_Name(name), m_ModelRepositoryPath(model_repo_path), m_ModelName(model_name),
          m_ModelVersion(model_version), m_UseDlrmQsl(use_dlrm_qsl), m_mig_uuid(mig_uuid),
          m_c2s_state_mq(c2s_state_mq), m_s2c_state_mq(s2c_state_mq), m_s2c_resp_mq(s2c_resp_mq),
          m_c2s_shmem(c2s_shmem), m_s2c_shmem(s2c_shmem)
    {
        // Set input memory type accordingly, with start_from_device as highest
        // priority
        // m_InputMemoryType = pinned_input ? TRITONSERVER_MEMORY_CPU_PINNED :
        // TRITONSERVER_MEMORY_CPU; m_InputMemoryType = start_from_device ? TRITONSERVER_MEMORY_GPU
        // : m_InputMemoryType;
        // TODO: Always use pinned cpu memory?
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
    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples){};
    virtual void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns);
    virtual void FlushQueries();

    std::shared_ptr<nvidia::inferenceserver::TraceManager> m_TraceManager;

    void TraceCaptureTimeStamp(TRITONSERVER_InferenceTrace* trace_ptr, const std::string comment);
    void Worker();

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
    std::string m_mig_uuid;
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

    // IPC objects
    sharedMQ_ptr<IPCcomm> m_c2s_state_mq;
    sharedMQ_ptr<IPCcomm> m_s2c_state_mq;
    sharedMQ_ptr<mlperf::QuerySampleResponse> m_s2c_resp_mq;
    sharedMSM_ptr m_c2s_shmem;
    sharedMR_ptr m_s2c_shmem;
};

// Request/Response shim between 'real LoadGen' and SUT children
class SUTShim : public mlperf::SystemUnderTest
{
  public:
    SUTShim(std::string name, std::string scenario, uint32_t batch_size, uint64_t timeout,
            std::vector<std::string>& mig_uuids, sharedMap_ptr<IPCcomm>& c2s_state_mq_map,
            sharedMap_ptr<IPCcomm>& s2c_state_mq_map,
            sharedMap_ptr<mlperf::QuerySampleResponse>& s2c_resp_mq_map,
            sharedMSMMap_ptr& c2s_shmem_map, sharedMRMap_ptr& s2c_shmem_map)
        : m_Name(name), m_mig_uuids(mig_uuids), m_max_server_buffer_size(batch_size),
          m_server_queue_timeout(std::chrono::microseconds(timeout)), m_num_migs(mig_uuids.size()),
          m_scenario(scenario), m_finished(false), m_server_turn(0), m_server_sticky_counter(0),
          m_c2s_state_mq_map(c2s_state_mq_map), m_s2c_state_mq_map(s2c_state_mq_map),
          m_s2c_resp_mq_map(s2c_resp_mq_map), m_c2s_shmem_map(c2s_shmem_map),
          m_s2c_shmem_map(s2c_shmem_map){};

    ~SUTShim(){};

    // SUT virtual interface
    virtual const std::string& Name() const
    {
        return m_Name;
    }
    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples);

    void Do_LoadSamplesToRam(std::vector<mlperf::QuerySampleIndex>& samples);
    void Do_UnloadSamplesFromRam(std::vector<mlperf::QuerySampleIndex>& samples);

    // null override
    void FlushQueries() override{};
    void ReportLatencyResults(
        const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

    // other utils
    void Init();
    void Done();

    void ServerQueueHandler(std::string mig_uuid, uint64_t max_sq_size,
                            std::chrono::microseconds sq_timeout);

    int getServerTurn()
    {
        return m_server_turn;
    }

    int get_max_sq_size()
    {
        return m_max_server_buffer_size;
    }

    std::chrono::microseconds get_sq_timeout()
    {
        return m_server_queue_timeout;
    }

    void moveServerTurn()
    {
        m_server_sticky_counter++;
        if(m_server_sticky_counter >= m_server_sticky_counter_cap)
        {
            m_server_sticky_counter = 0;
            m_server_turn++;
            if(m_server_turn >= m_num_migs)
                m_server_turn = 0;
        }
    }

    // if true is returned, it's expected to send query vector to server, and flush
    void queue_to_SQ(std::string mig_uuid, const std::vector<mlperf::QuerySample>& samples)
    {
        m_SQ_map[mig_uuid]->insert(samples);
    }

    bool is_finished()
    {
        return m_finished.load();
    };

  private:
    const std::string m_Name;
    const std::vector<std::string> m_mig_uuids;
    const int m_num_migs;
    const std::string m_scenario;
    std::atomic<bool> m_finished;
    uint8_t m_server_turn;
    uint8_t m_server_sticky_counter;
    uint8_t m_server_sticky_counter_cap;
    const uint32_t m_max_server_buffer_size;
    const std::chrono::microseconds m_server_queue_timeout;
    boost::unordered::unordered_map<std::string, SampleSQ_ptr> m_SQ_map;
    boost::unordered::unordered_map<std::string, std::thread> m_SQ_handlers;
    // pointers to IPC objects
    sharedMap_ptr<IPCcomm> m_c2s_state_mq_map;
    sharedMap_ptr<IPCcomm> m_s2c_state_mq_map;
    sharedMap_ptr<mlperf::QuerySampleResponse> m_s2c_resp_mq_map;
    sharedMSMMap_ptr m_c2s_shmem_map;
    sharedMRMap_ptr m_s2c_shmem_map;
};

}; // namespace triton_frontend

#endif
