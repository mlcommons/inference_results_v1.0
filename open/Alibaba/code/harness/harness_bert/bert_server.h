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
#include <unordered_map>
#include <vector>
#include <set>

#include "lwis_buffers.h"

constexpr size_t BERT_MAX_SEQ_LENGTH{384};
constexpr int NUM_RESPONSE_THREADS = 1;

using BERTInputType = int32_t;
using BERTInput = std::array<BERTInputType, BERT_MAX_SEQ_LENGTH>;
using BERTOutputType = half_float::half;
using BERTOutput = std::array<BERTOutputType, BERT_MAX_SEQ_LENGTH * 2>;

bool operator==(const nvinfer1::Dims& d1, const nvinfer1 ::Dims& d2);

class CopyStream
{
  public:
    CopyStream()
    {
        unsigned int flags = cudaEventDefault | cudaEventDisableTiming;
        CHECK_EQ(cudaStreamCreate(&s), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&h2d, flags), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&d2h, flags), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&infer, flags), cudaSuccess);
    }

    ~CopyStream()
    {
        CHECK_EQ(cudaStreamDestroy(s), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(h2d), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(d2h), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(infer), cudaSuccess);
    }
    void recordH2D(){
        CHECK_EQ(cudaEventRecord(h2d, s), cudaSuccess);
    }

    void recordInferDone(cudaStream_t inferenceStream){
        CHECK_EQ(cudaEventRecord(infer, inferenceStream), cudaSuccess);
    }

    void makeAwaitH2D(cudaStream_t inferenceStream){
        CHECK_EQ(cudaStreamWaitEvent(inferenceStream, h2d,0), cudaSuccess);
    }

    void awaitInfer(){
        CHECK_EQ(cudaStreamWaitEvent(s, infer, 0), cudaSuccess);
    }
    void recordD2H(){
        CHECK_EQ(cudaEventRecord(d2h, s), cudaSuccess);
    }
    void syncD2H(){
        CHECK_EQ(cudaEventSynchronize(d2h), cudaSuccess);
    }
    cudaStream_t get() const{
        return s;
    }

  //private:
    cudaStream_t s;
    cudaEvent_t h2d;
    cudaEvent_t d2h;
    cudaEvent_t infer;
};

template<typename T>
class BERTManagedBuffer
{
  public:
    BERTManagedBuffer(size_t sz)
        : mSize(sz), mBytes(sizeof(T) * sz), mDeviceBuffer(mBytes), mHostBuffer(mBytes),
          mHostPtr(static_cast<T*>(mHostBuffer.data())),
          mDevicePtr(static_cast<T*>(mDeviceBuffer.data()))
    {
        CHECK_EQ(mHostPtr != nullptr, true);
        CHECK_EQ(mDevicePtr != nullptr, true);
        CHECK_EQ(mSize > 0, true);
        CHECK_EQ(mBytes > 0, true);
    }
    // memset device buffer
    void memsetD(int value)
    {
        CHECK_EQ(cudaMemset(mDeviceBuffer.data(), value, mBytes), cudaSuccess);
    }

    // transfer between pair of managed buffers
    void H2DAsync(size_t effectiveSize, cudaStream_t stream)
    {
        CHECK_EQ(cudaMemcpyAsync(mDeviceBuffer.data(), mHostBuffer.data(),
                                 effectiveSize * sizeof(T), cudaMemcpyHostToDevice, stream),
                 cudaSuccess);
    }

    void D2HAsync(size_t effectiveSize, cudaStream_t stream)
    {
        CHECK_EQ(cudaMemcpyAsync(mHostBuffer.data(), mDeviceBuffer.data(),
                                 effectiveSize * sizeof(T), cudaMemcpyDeviceToHost, stream),
                 cudaSuccess);
    }

    // stage one sequence of length seqLen at batchIdx in the page-locked host memory
    void H2H(void* ptr, const size_t offset, const size_t elems)
    {
        memcpy(mHostPtr + offset, ptr, sizeof(T) * elems);
    }

    T* const HostData()
    {
        return mHostPtr;
    }
    T* const DeviceData()
    {
        return mDevicePtr;
    }

  private:
    size_t mSize;
    size_t mBytes;
    lwis::DeviceBuffer mDeviceBuffer;
    lwis::HostBuffer mHostBuffer;
    T* const mHostPtr;
    T* const mDevicePtr;
};

using BERTBufferIn = BERTManagedBuffer<BERTInputType>;
using BERTBufferOut = BERTManagedBuffer<BERTOutputType>;

using BERTTask_t = std::vector<std::pair<mlperf::QuerySample,std::chrono::high_resolution_clock::time_point>>;

struct BERTResponse{
    std::vector<mlperf::QuerySampleResponse> QSRs;
    cudaEvent_t resultReady;
    size_t copyStreamIdx;
};

// This is a data structure used to track all the accumulated sequence lengths appeared
// in each batch that has been or will be processed.
// A pointer will point to the `threshold`th percentile of all accumulated sequence lengths, so
// this threshold sequence length can be used as a reference to drop requests that are outside of
// this length in a coming batch.
class TotalLengthMultiSet {
public:
    TotalLengthMultiSet(double threshold): mThreshold(threshold), mMinSampleCount(1000) {
        // default threshold of total length is INT_MAX
        mItrPos = 1;
        mSet.insert(INT_MAX);
        mThresholdItr = mSet.begin();
    }

    void InsertTasks(const BERTTask_t& tasks, std::shared_ptr<qsl::SampleLibrary> qsl) {
        std::unique_lock<std::mutex> lck(mMtx);
        // insert each accumulated sequence length into the multiset
        int totalLength = 0;
        for(int i = 0; i < tasks.size(); ++i)
        {
            BERTInput* mask = static_cast<BERTInput*>(qsl->GetSampleAddress(tasks[i].first.index, 2));
            totalLength += std::accumulate(mask->begin(), mask->end(), 0);
            Insert(totalLength);
        }
    }

    inline int GetThresholdLength() {
        std::unique_lock<std::mutex> lck(mMtx);
        // return INT_MAX if samples are not enough
        if (mSet.size() < mMinSampleCount) return INT_MAX;
        return *mThresholdItr;
    }

private:
    void Insert(int totalLength) {
        if (totalLength <= *mThresholdItr) {
            mSet.insert(mThresholdItr, totalLength);
            ++mItrPos;
        } else {
            mSet.insert(totalLength);
        }
        // move the iterator to the correct position
        while (mItrPos <= mSet.size() * mThreshold) {
            ++mThresholdItr;
            ++mItrPos;
        }
        while (mItrPos-1 > mSet.size() * mThreshold) {
            --mThresholdItr;
            --mItrPos;
        }
    }

private:
    double mThreshold;
    size_t mMinSampleCount;
    size_t mItrPos;
    std::multiset<int> mSet;
    std::multiset<int>::iterator mThresholdItr;

    // mutex to make the APIs thread-safe
    std::mutex mMtx;
};

class BERTServer : public mlperf::SystemUnderTest
{
  public:
    BERTServer(const std::string name, const std::string enginePath,
               std::shared_ptr<qsl::SampleLibrary> qsl, const std::vector<int>& gpus,
               int maxBatchSize, int numCopyStreams, int numBERTCores, bool useGraphs,
               int graphsMaxSeqLen, const std::string& graphSpecs, double softDrop, double targetLatencyPercentile,
               uint64_t serverNumIssueQueryThreads);

    virtual ~BERTServer();

    const std::string& Name() const override;

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;
    void FlushQueries() override;
    void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override;

  private:
    // If the function returns empty vector then there are no tasks remained and the caller should
    // exit
    std::vector<std::pair<mlperf::QuerySample, std::chrono::high_resolution_clock::time_point>>
        GetTasks(int maxSampleCount, int qThreadIdx);

    template<typename T>
    void ProcessTasks(std::shared_ptr<T>, int deviceId, int qThreadIdx);
    void StartIssueThread(int threadIdx);

  private:
    size_t GetSampleLength(mlperf::QuerySampleIndex idx);
    void CreateEnginesPerGPU(int deviceId, std::shared_ptr<std::mutex> pMtx, const std::vector<std::vector<char>>& trtModelStreams);

    const std::string mName;
    const std::string mEnginePath;

    // For each GPU device id, create a vector of ptrs to ICudaEngine
    std::unordered_map<int, std::vector<std::shared_ptr<nvinfer1::ICudaEngine>>> mEnginesPerGPU;

    int mMaxBatchSize;
    std::shared_ptr<qsl::SampleLibrary> mQsl;

    // Each query sample is accompanied by the time query arrived
    std::vector<std::deque<std::pair<mlperf::QuerySample, std::chrono::high_resolution_clock::time_point>>>
        mTasksVec;

    // mutex to serialize access to mTasks member variable
    std::unique_ptr<std::vector<std::mutex>> mMtxs;

    // The object to allow threads to avoid spinning on mMtx and mTasks for the new work to arrive
    std::unique_ptr<std::vector<std::condition_variable>> mCondVars;

    // Indicates that there will no new tasks and the worker threads should stop processing samples
    bool mStopGetTasks;
    bool mStopProcessResponse;

    // Max seqlen to be used for creating CUDA graphs
    int mGraphMaxSeqLen;

    // Whether to apply soft drop policy
    double mSoftDrop;
    double mTargetLatencyPercentile;
    TotalLengthMultiSet mTotalLengthSet;
    uint64_t mTotalTasksCount;
    uint64_t mSoftDropCount;

    // mutex for both mTotalTasksCount and mSoftDropCount
    std::mutex mSoftDropMtx;

    std::vector<std::thread> mIssueQueryThreads;
    std::vector<std::thread> mWorkerThreads;
    std::map<std::thread::id, int> mThreadMap;
};
