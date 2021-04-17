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

#pragma once

#include "half.h"
#include "qsl.hpp"
#include "system_under_test.h"
#include "utils.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "lwis_buffers.h"
#include "dlrm_qsl.hpp"

#include "batch_maker.hpp"

struct DLRMResult
{
    std::shared_ptr<std::vector<DLRMOutputType>> outputs;
    std::vector<DLRMTask> tasks;
    DLRMInferCallback callback;
};

using DLRMResultProcessingCallback = std::function<void(const DLRMResult& r)>;

using DLRMNumericPtrBuffer = DLRMDataBuffer<DLRMNumericInputType*>;
using DLRMCategoricalPtrBuffer = DLRMDataBuffer<DLRMCategoricalInputType*>;
using DLRMSampleSizeBuffer = DLRMDataBuffer<size_t>;

struct DLRMDeferredResult
{
    size_t batchSize;
    const DLRMOutputType * outputs;
    std::vector<DLRMTask> tasks;
    DLRMInferCallback callback;
    DLRMResultProcessingCallback resultCallback;
};

class DLRMResultHandlerPool
{
  public:
    DLRMResultHandlerPool(size_t numThreads) :
        mStopWork(false)
    {
        for (int i = 0; i < numThreads; ++i)
        {
            mThreads.emplace_back(&DLRMResultHandlerPool::HandleResult, this);
        }
    }

    ~DLRMResultHandlerPool()
    {
        {
            std::unique_lock<std::mutex> lock(mMtx);
            mStopWork = true;
            mCondVar.notify_all();
        }
        for (auto& t : mThreads) { t.join(); }
    }

    void Enqueue(const DLRMResult& r)
    {
        NVTX_RANGE_PUSH("DLRMResultHandlerPool::Enqueue");
        std::unique_lock<std::mutex> lock(mMtx);
        mResultQ.emplace_back(r);
        mCondVar.notify_one();
        NVTX_RANGE_POP();
    }

    void HandleResult()
    {
        while (true)
        {
            NVTX_RANGE_PUSH("DLRMResultHandlerPool::HandleResult iteration");
            DLRMResult res;
            {
                NVTX_RANGE_PUSH("Extract result from queue");
                std::unique_lock<std::mutex> lock(mMtx);
                mCondVar.wait(
                    lock,
                    [&]() { return (!mResultQ.empty()) || mStopWork; });

                if (mStopWork) { NVTX_RANGE_POP(); NVTX_RANGE_POP(); break; }

                res = mResultQ.front();
                mResultQ.pop_front();
                mCondVar.notify_one();
                NVTX_RANGE_POP();
            }

            NVTX_RANGE_PUSH("Handle result");
            std::vector<mlperf::QuerySampleResponse> responses;
            int offset = 0;
            for (const auto& task : res.tasks)
            {
                mlperf::QuerySampleResponse response {
                    task.querySample.id,
                    reinterpret_cast<uintptr_t>(&res.outputs->at(offset)),
                    sizeof(DLRMOutputType) * task.numIndividualPairs
                };
                responses.emplace_back(response);

                offset = offset + task.numIndividualPairs;
            }
            CHECK_EQ(responses.size(), res.tasks.size());
            res.callback(responses);
            DLOG(INFO) << "Handled " << offset << " pairs";
            NVTX_RANGE_POP();
            NVTX_RANGE_POP();
        }
    }

  private:
    std::vector<std::thread> mThreads;
    std::deque<DLRMResult> mResultQ;
    std::mutex mMtx;
    std::condition_variable mCondVar;
    bool mStopWork;
};

// Convenience bundle for {cuda events, input/output device buffers, output host buffers}
// We have 2 sets of such bundles per GPU, owned by DLRMCore, to correspond to a foreground
// task being readied on the host and a background task running inference on GPU
class DLRMEventBufferBundle
{
  public:
    DLRMEventBufferBundle(size_t bundleIdx,
                          size_t numInVol,
                          size_t catInVol,
                          size_t outVol,
                          size_t maxBatchSize) :
        idx(bundleIdx),
        numericInputBuf(numInVol, maxBatchSize, false),
        categoricalInputBuf(catInVol, maxBatchSize, false),
        outputBuf(outVol, maxBatchSize, true),
        numericInputPtrBuf(1, maxBatchSize, true),
        categoricalInputPtrBuf(1, maxBatchSize, true),
        sampleSizesBuf(1, maxBatchSize, true),
        sampleOffsetsBuf(1, maxBatchSize, true)
    {
        unsigned int flags = cudaEventDefault | cudaEventDisableTiming;
        CHECK_EQ(cudaEventCreateWithFlags(&h2dEvent, flags), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&inferEvent, flags), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&d2hEvent, flags), cudaSuccess);
    }
    ~DLRMEventBufferBundle()
    {
        CHECK_EQ(cudaEventDestroy(h2dEvent), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(inferEvent), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(d2hEvent), cudaSuccess);
    }

    void recordH2D(cudaStream_t& h2dStream) { CHECK_EQ(cudaEventRecord(h2dEvent, h2dStream), cudaSuccess); }
    void makeAwaitH2D(cudaStream_t& inferStream) { CHECK_EQ(cudaStreamWaitEvent(inferStream, h2dEvent, 0), cudaSuccess); }

    void recordInfer(cudaStream_t& inferStream) { CHECK_EQ(cudaEventRecord(inferEvent, inferStream), cudaSuccess); }
    void makeAwaitInfer(cudaStream_t& d2hStream) { CHECK_EQ(cudaStreamWaitEvent(d2hStream, inferEvent, 0), cudaSuccess); }

    void recordD2H(cudaStream_t& d2hStream) { CHECK_EQ(cudaEventRecord(d2hEvent, d2hStream), cudaSuccess); }
    void syncD2H() { CHECK_EQ(cudaEventSynchronize(d2hEvent), cudaSuccess); }

    size_t idx;
    cudaEvent_t h2dEvent;
    cudaEvent_t d2hEvent;
    cudaEvent_t inferEvent;
    DLRMNumericBuffer numericInputBuf;
    DLRMCategoricalBuffer categoricalInputBuf;
    DLRMOutputBuffer outputBuf;

    // The next 4 buffers are used only in start_from_device mode
    DLRMNumericPtrBuffer numericInputPtrBuf;
    DLRMCategoricalPtrBuffer categoricalInputPtrBuf;
    DLRMSampleSizeBuffer sampleSizesBuf;
    DLRMSampleSizeBuffer sampleOffsetsBuf;
};

class DLRMCore
{
  public:
    DLRMCore(std::shared_ptr<nvinfer1::ICudaEngine> engine, int maxBatchSize, int numBundles,
             int numCompleteThreads, int profileIdx);
    ~DLRMCore();
    void infer(
        std::shared_ptr<DLRMEventBufferBundle> ebBundle,
        size_t batchSize,
        std::vector<DLRMTask>& tasks,
        Batch* batch,
        void(*h2dCallBack)(void*),
        DLRMInferCallback resultCallback,
        DLRMNumericInputType* numericInputPtr,
        DLRMCategoricalInputType* categoricalInputPtr);
    void inferFromDevice(
        std::shared_ptr<DLRMEventBufferBundle> ebBundle,
        size_t batchSize,
        std::vector<DLRMTask>& tasks,
        DLRMInferCallback resultCallback);
    void WarmUp(double duration);
    size_t GetMaxBatchSize() { return mMaxBatchSize; };

    size_t mNumInVol;
    size_t mCatInVol;
    size_t mOutVol;

    std::shared_ptr<DLRMEventBufferBundle> NextForegroundBundle();

  private:
    void SetBatchSize(int batchSize);

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext;
    size_t mMaxBatchSize;

    DLRMResultHandlerPool mResHandlerPool;

    cudaStream_t mH2DStream;
    cudaStream_t mComputeStream;
    cudaStream_t mD2HStream;

    std::vector<std::shared_ptr<DLRMEventBufferBundle>> mEventBufferBundle;
    size_t mBundleCounter;

    std::vector<std::vector<void*>> mBindings;
};

class DLRMServer : public mlperf::SystemUnderTest
{
  public:
    DLRMServer(const std::string name,
               const std::string enginePath,
               std::vector<DLRMSampleLibraryPtr_t> qsls,
               const std::vector<int>& gpus,
               int maxBatchSize,
               int numBundles,
               int numCompleteThreads,
               int numDLRMCores,
               double warmupDuration,
               int numStagingThreads,
               int numStagingBatches,
               int maxPairsPerThread,
               bool checkContiguity,
               bool startFromDevice,
               NumaConfig numaConfig);

    virtual ~DLRMServer();

    const std::string& Name() const override;

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;
    void FlushQueries() override;
    void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override;

    bool UseNuma() { return !mNumaConfig.empty(); };

  private:

    void ProcessTasks(std::shared_ptr<DLRMCore>, int deviceId, int profileIdx);
    void ProcessTasksFromDevice(std::shared_ptr<DLRMCore>, int deviceId, int profileIdx);
    void SetupDevice(const std::string enginePath, int numBundles, int numCompleteThreads, int numDLRMCores, int warmupDuration, int deviceId);
    std::shared_ptr<nvinfer1::ICudaEngine> DeserializeEngine(const std::string enginePath);
    std::vector<DLRMTask> GetBatch();

    // Batch Makers for the start from host (start_from_device = false) case. One per NUMA node.
    std::vector<std::shared_ptr<BatchMaker>> mBatchMakers;

    const std::string mName;
    int mMaxBatchSize;
    std::vector<DLRMSampleLibraryPtr_t> mQsls;
    bool mStartFromDevice;

    // Queue to be used for the start_from_device case, each sample is accompanied by the pair count
    std::deque<DLRMTask> mTasks;

    // mutex to serialize access to mTasks member variable
    std::mutex mMtx;

    // The object to allow threads to avoid spinning on mMtx and mTasks for the new work to arrive
    std::condition_variable mCondVar;

    // Indicates that there will no new tasks and the worker threads should stop processing samples
    bool mStopWork;

    std::vector<std::thread> mWorkerThreads;
    std::vector<std::shared_ptr<DLRMCore>> mDLRMCores;
    size_t mNumInVol;
    size_t mCatInVol;

    // NUMA configs of the machine: list of CPUs for each NUMA node, assuming each GPU corresponds to one NUMA node.
    NumaConfig mNumaConfig;
    GpuToNumaMap mGpuToNumaMap;
    // When NUMA is used, we issue queries around NUMA nodes in round-robin.
    int mPrevBatchMakerIdx{-1};
};
