/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef __LWIS_HPP__
#define __LWIS_HPP__

#include <map>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <deque>
#include <chrono>
#include <functional>

#include <cuda.h>
#include <cuda_runtime.h>
#include "NvInfer.h"

#include "system_under_test.h"
#include "test_settings.h"
#include "qsl.hpp"
#include "utils.hpp"

// For debugging the timing of each part
class Timer
{
public:
  Timer(const std::string& tag_) : tag(tag_) { std::cout << "Timer " << tag << " created." << std::endl; }
  void add(const std::chrono::duration<double, std::milli>& in) { ++count; total += in; }
  ~Timer() { std::cout << "Timer " << tag << " reports " << total.count() / count << " ms per call for " << count << " times." << std::endl; }
private:
  std::string tag;
  std::chrono::duration<double, std::milli> total{0};
  size_t count{0};
};

#define TIMER_ON 0

#if TIMER_ON
#define TIMER_START(s) static Timer timer##s(#s); auto start##s = std::chrono::high_resolution_clock::now();
#define TIMER_END(s) timer##s.add(std::chrono::high_resolution_clock::now() - start##s);
#else
#define TIMER_START(s)
#define TIMER_END(s)
#endif

// LWIS (Light Weight Inference Server) is a simple server composed of the following components:
// Simple batcher
// Engine creation
// Host/Device Memory management
// Batch management and data reporting
//
// The goals of LWIS are:
// - Thin layer between MLPerf interface and Cuda/TRT APIs.
// - Single source file and minimal dependencies.

//#define LWIS_DEBUG_DISABLE_INFERENCE // disables all device operations including data copies, most events, and compute
//#define LWIS_DEBUG_DISABLE_COMPUTE // disables the compute (TRT enqueue) portion of inference

namespace lwis {
  class BufferManager;

  using namespace std::chrono_literals;

  class Device;
  class Engine;
  class Server;

  typedef std::shared_ptr<Device> DevicePtr_t;
  typedef std::shared_ptr<Engine> EnginePtr_t;
  typedef std::shared_ptr<Server> ServerPtr_t;

  typedef std::shared_ptr<::nvinfer1::ICudaEngine> ICudaEnginePtr_t;

  struct Batch { 
    std::vector<mlperf::QuerySampleResponse> Responses;
    std::vector<mlperf::QuerySampleIndex> SampleIds;
    cudaEvent_t Event;
    cudaStream_t Stream;
  };

  struct ServerSettings {
    bool EnableCudaGraphs{false};
    bool EnableSyncOnEvent{false};
    bool EnableSpinWait{false};
    bool EnableDeviceScheduleSpin{false};
    bool EnableDma{true};
    bool EnableDmaStaging{false};
    bool EnableDirectHostAccess{false};
    bool EnableDLADirectHostAccess{true};
    bool EnableResponse{true};
    bool EnableDequeLimit{false};
    bool EnableBatcherThreadPerDevice{false};
    bool EnableCudaThreadPerDevice{false};
    bool EnableStartFromDeviceMem{false};
    bool RunInferOnCopyStreams{false};
    bool UseSameContext{false};

    size_t GPUBatchSize{1};
    size_t GPUCopyStreams{4};
    size_t GPUInferStreams{4};
    int32_t MaxGPUs{-1};

    size_t DLABatchSize{1};
    size_t DLACopyStreams{4};
    size_t DLAInferStreams{1};
    int32_t MaxDLAs{-1};

    bool ForceContiguous{false};
    size_t CompleteThreads{1};

    std::chrono::microseconds Timeout{10000us};
    NumaConfig m_NumaConfig;
    GpuToNumaMap m_GpuToNumaMap;
  };

  struct ServerParams {
    std::string DeviceNames;
    // <perDeviceType, perDevice, perIteration>
    std::vector<std::vector<std::vector<std::string>>> EngineNames;
  };

  template<typename T>
  class SyncQueue {
  public:
    typedef typename std::deque<T>::iterator iterator;

    SyncQueue() {}

    bool empty() { std::unique_lock<std::mutex> l(m_Mutex); return m_Queue.empty(); }

    void insert(const std::vector<T> &values) { { std::unique_lock<std::mutex> l(m_Mutex); m_Queue.insert(m_Queue.end(), values.begin(), values.end()); } m_Condition.notify_one(); }
    void acquire(std::deque<T> &values, std::chrono::microseconds duration = 10000us, size_t size = 1, bool limit = false) {
      size_t remaining = 0;

      TIMER_START(m_Mutex_create);
      {
        std::unique_lock<std::mutex> l(m_Mutex);
        TIMER_END(m_Mutex_create);
        TIMER_START(m_Condition_wait_for);
        m_Condition.wait_for(l, duration, [=]{ return m_Queue.size() >= size; });
        TIMER_END(m_Condition_wait_for);

        if (!limit || m_Queue.size() <= size) {
          TIMER_START(swap);
          values.swap(m_Queue);
          TIMER_END(swap);
        }
        else {
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
      if (remaining) m_Condition.notify_one();
    }

    void push_back(T const &v) { { std::unique_lock<std::mutex> l(m_Mutex); m_Queue.push_back(v); } m_Condition.notify_one(); }
    void emplace_back(T const &v) { { std::unique_lock<std::mutex> l(m_Mutex); m_Queue.emplace_back(v); } m_Condition.notify_one(); }
    T front() { std::unique_lock<std::mutex> l(m_Mutex); m_Condition.wait(l, [=]{ return !m_Queue.empty(); }); T r(std::move(m_Queue.front())); return r; }
    T front_then_pop() { std::unique_lock<std::mutex> l(m_Mutex); m_Condition.wait(l, [=]{ return !m_Queue.empty(); }); T r(std::move(m_Queue.front())); m_Queue.pop_front(); return r; }
    void pop_front() { std::unique_lock<std::mutex> l(m_Mutex); m_Queue.pop_front(); }

  private:
    mutable std::mutex m_Mutex;
    std::condition_variable m_Condition;

    std::deque<T> m_Queue;
  };

  // captures execution engine for performing inference
  class Device {
    friend Server;

  public:
    Device(size_t id, size_t numCopyStreams, size_t numInferStreams, size_t numCompleteThreads, bool enableSpinWait, bool enableDeviceScheduleSpin, size_t batchSize, bool dla, bool useSameContext)
      : m_Id(id), m_BatchSize(batchSize), m_EnableSpinWait(enableSpinWait), m_EnableDeviceScheduleSpin(enableDeviceScheduleSpin),
        m_UseSameContext(useSameContext), m_DLA(dla), m_CopyStreams(numCopyStreams), m_InferStreams(numInferStreams) {
      for (size_t i = 0; i < numCompleteThreads; i++) {
        m_Threads.emplace_back(&Device::Completion, this);
      }
    }

    void AddEngine(EnginePtr_t engine);
    std::map<size_t, std::vector<EnginePtr_t>> &GetEngines() { return m_Engines; }

    size_t GetBatchSize() { return m_BatchSize; }
    void SetBatchSize(size_t batchSize) { m_BatchSize = batchSize; }

    std::string GetName() { return m_Name; }
    size_t GetId() { return m_Id; }

    void SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex> &sample_ids, size_t response_count)> callback) { m_ResponseCallback = callback; }

    struct Stats {
      std::map<size_t, uint64_t> m_BatchSizeHistogram;
      uint64_t m_MemcpyCalls{0};
      uint64_t m_PerSampleCudaMemcpyCalls{0};
      uint64_t m_BatchedCudaMemcpyCalls{0};

      void reset() {
        m_BatchSizeHistogram.clear();
        m_MemcpyCalls = 0;
        m_PerSampleCudaMemcpyCalls = 0;
        m_BatchedCudaMemcpyCalls = 0;
      }
    };

    const Stats &GetStats() const { return m_Stats; }

  private:
    void BuildGraphs(bool directHostAccess);

    void Setup();
    void Issue();
    void Done();

    void Completion();

    std::map<size_t, std::vector<EnginePtr_t>> m_Engines;
    const size_t m_Id{0};
    std::string m_Name{""};
    size_t m_BatchSize{0};

    const bool m_EnableSpinWait{false};
    const bool m_EnableDeviceScheduleSpin{false};
    const bool m_DLA{false};
    const bool m_UseSameContext{false};

    std::vector<cudaStream_t> m_CopyStreams;
    std::vector<cudaStream_t> m_InferStreams;
    size_t m_InferStreamNum{0};
    std::map<cudaStream_t, std::tuple<std::shared_ptr<BufferManager>, cudaEvent_t, cudaEvent_t, cudaEvent_t, ::nvinfer1::IExecutionContext*>> m_StreamState;

    // Graphs
    typedef std::pair<cudaStream_t, size_t> t_GraphKey;
    std::map<t_GraphKey, cudaGraphExec_t> m_CudaGraphExecs;

    // Completion management
    SyncQueue<Batch> m_CompletionQueue;
    std::vector<std::thread> m_Threads;

    // Stream management
    SyncQueue<cudaStream_t> m_StreamQueue;

    // Query sample response callback.
    std::function<void(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex> &sample_ids, size_t response_count)> m_ResponseCallback;

    // Batch management
    SyncQueue<std::pair<std::deque<mlperf::QuerySample>, cudaStream_t>> m_IssueQueue;

    Stats m_Stats;
  };

  // captures execution engine for performing inference
  class Engine {
  public:
    Engine(::nvinfer1::ICudaEngine *cudaEngine) : m_CudaEngine(cudaEngine) { }

    ::nvinfer1::ICudaEngine *GetCudaEngine() const { return m_CudaEngine; }

  private:
    ::nvinfer1::ICudaEngine *m_CudaEngine;
  };

  // Create buffers and other execution resources.
  // Perform queuing, batching, and manage execution resources.
  class Server : public mlperf::SystemUnderTest {
  public:
    Server(std::string name) : m_Name(name) { }
    ~Server() { }

    void AddSampleLibrary(qsl::SampleLibraryPtr_t sl) { m_SampleLibraries.emplace_back(sl); }
    void Setup(ServerSettings &settings, ServerParams &params);
    void Warmup(double duration);
    void Done();

    std::vector<DevicePtr_t> &GetDevices() { return m_Devices; }

    // SUT virtual interface
    virtual const std::string &Name() const { return m_Name; }
    virtual void IssueQuery(const std::vector<mlperf::QuerySample> &samples);
    virtual void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency> &latencies_ns);
    virtual void FlushQueries();

    // Set query sample response callback to all the devices.
    void SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex> &sample_ids, size_t response_count)> callback);

  private:
    void ProcessSamples();
    void ProcessBatches();

    void IssueBatch(DevicePtr_t device, size_t batchSize, std::deque<mlperf::QuerySample>::iterator begin, std::deque<mlperf::QuerySample>::iterator end, cudaStream_t copyStream);
    std::vector<void *> CopySamples(DevicePtr_t device, size_t batchSize, std::deque<mlperf::QuerySample>::iterator begin, std::deque<mlperf::QuerySample>::iterator end, cudaStream_t stream, bool directHostAccess, bool staging);

    DevicePtr_t GetNextAvailableDevice(size_t deviceId);

    void Reset();

    const std::string m_Name;

    std::vector<DevicePtr_t> m_Devices;
    size_t m_DeviceIndex{0};
    std::vector<qsl::SampleLibraryPtr_t> m_SampleLibraries;

    ServerSettings m_ServerSettings;

    // Query management
    SyncQueue<mlperf::QuerySample> m_WorkQueue;
    std::vector<std::thread> m_Threads;
    std::atomic<size_t> m_DeviceNum{0};

    // Batch management
    std::vector<std::thread> m_IssueThreads;
    std::atomic<size_t> m_IssueNum{0};

    // Check if NUMA is used
    bool UseNuma() { return !m_ServerSettings.m_NumaConfig.empty(); };

    // Get number of NUMA nodes
    int GetNbNumas() { return m_ServerSettings.m_NumaConfig.size(); };

    // Get NUMA node index of a GPU
    int GetNumaIdx(const int deviceId) { return UseNuma() ? m_ServerSettings.m_GpuToNumaMap[deviceId] : 0; }

    // Get closest CPUs to a GPU
    std::vector<int> GetClosestCpus(const int deviceId)
    {
      CHECK(UseNuma());
      return m_ServerSettings.m_NumaConfig[GetNumaIdx(deviceId)].second;
    }
  };

};

#endif
