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

#include <set>
#include <tuple>
#include "bert_server.h"

// Use (copyStreamIdx, (max seqLen, batch size, total seqLen)) as key to get CUDA Graphs
using BERTGraphSpec_t = std::tuple<int, int, int>;
using BERTGraphKey_t = std::pair<int, BERTGraphSpec_t>;

template<typename T>
class SyncQueue {
public:
    SyncQueue() {}
    ~SyncQueue() {}

    bool empty() { std::unique_lock<std::mutex> l(mMutex); return mQueue.empty(); }
    void push_back(T const &v) { { std::unique_lock<std::mutex> l(mMutex); mQueue.push_back(v); } mCondVar.notify_one(); }
    void emplace_back(T const &v) { { std::unique_lock<std::mutex> l(mMutex); mQueue.emplace_back(v); } mCondVar.notify_one(); }
    T front() { std::unique_lock<std::mutex> l(mMutex); mCondVar.wait(l, [=]{ return !mQueue.empty(); }); T r(std::move(mQueue.front())); return r; }
    T front_then_pop() { std::unique_lock<std::mutex> l(mMutex); mCondVar.wait(l, [=]{ return !mQueue.empty(); }); T r(std::move(mQueue.front())); mQueue.pop_front(); return r; }
    void pop_front() { std::unique_lock<std::mutex> l(mMutex); mQueue.pop_front(); }

private:
    mutable std::mutex mMutex;
    std::condition_variable mCondVar;
    std::deque<T> mQueue;
};

class BERTCoreVS 
{
  public:
    BERTCoreVS(const std::vector<std::shared_ptr<nvinfer1::ICudaEngine>>& engines, int numCopyStreams, int numDupProfiles,
        int profileIdx, bool useGraphs, int deviceId);
    ~BERTCoreVS();
    void infer(const BERTTask_t& tasks, std::shared_ptr<qsl::SampleLibrary> qsl);
    void WarmUp();
    size_t GetMaxBatchSize() { return mMaxBatchSize; };

    inline int StreamIdx() {
        return mCounter % mCopyStreams.size();
    }
    void ** GetBindings() {
        return mBindings[StreamIdx()].data();
    }

    BERTBufferIn& GetBufferInputIds()
    {
        return mInputIdBufs[mCounter % mCopyStreams.size()];
    }
    BERTBufferIn& GetBufferInputMask()
    {
        return mInputMaskBufs[mCounter % mCopyStreams.size()];
    }
    BERTBufferIn& GetBufferSegmentIds()
    {
        return mSegmentIdBufs[mCounter % mCopyStreams.size()];
    }
    BERTBufferOut& GetBufferOutput()
    {
        return mOutputBufs[mCounter % mCopyStreams.size()];
    }
    
    BERTBufferIn& GetDummy()
    {
        return mDummy[mCounter % mCopyStreams.size()];
    }

    int GetClosestSeqLen(int seqLen) ;

    void BuildGraphs(int graphMaxSeqLen, const std::string& graphSpecs, int numBERTCores);

    static int CountTotalLength(const BERTTask_t& tasks, std::shared_ptr<qsl::SampleLibrary> qsl);

    void WaitUntilQueueEmpty() {
        std::unique_lock<std::mutex> lck(mMtx);
        mCondVar.wait(lck, [&]() { return mResultQ.empty(); });
    }

  private:
    std::shared_ptr<nvinfer1::IExecutionContext> GetContext(int batchSize, int seqLen);

    void SetInputShapes(std::shared_ptr<nvinfer1::IExecutionContext> context, int sumS,
                                    int B, int maxS);

    void InitializeGraphSpecs(int graphMaxSeqLen, const std::string& graphSpecs, int numBERTCores);
    std::vector<BERTGraphSpec_t> ParseGraphSpecs(const std::string& graphSpecs, int graphSeqLenUpperBound);

    BERTGraphSpec_t GetClosestGraphSpec(int maxSeqLen, int batchSize, int totalSeqLen);

    size_t GetTotalGPUMemoryInMiB();

    std::map<std::pair<int, int>, std::shared_ptr<nvinfer1::IExecutionContext>> mMaxDimsToContext;
    std::vector<std::pair<int, std::vector<std::pair<int, std::shared_ptr<nvinfer1::IExecutionContext>>>>> mSeqBatchCtx;
    size_t mMaxBatchSize;
    cudaStream_t mStream; //compute stream
    int mDeviceId;

    // data members to support CUDA Graphs
    bool mUseGraphs;
    std::vector<BERTGraphSpec_t> mGraphSpecs;
    std::map<BERTGraphKey_t, cudaGraphExec_t> mCudaGraphExecs;

    // statistics for CUDA Graphs
    std::map<BERTGraphSpec_t, uint64_t> mCudaGraphCounts;
    std::map<int, uint64_t> mMaxSeqLenCounts;
    std::set<BERTGraphSpec_t> mQuantizedPoints;

    //we create a context per profile but share the device memory between them, assuming that we use only one context at a time
    std::shared_ptr<lwis::DeviceBuffer> mContextBuf;

    std::vector<CopyStream> mCopyStreams;
    size_t mCounter;
    SyncQueue<size_t> mCopyStreamIdxQueue;

    // use pairs of page-locked host mem and device mem allocated to max_batch_len x max_seq_len
    // we need one buffer per copy stream
    std::vector<BERTBufferIn> mInputIdBufs;
    std::vector<BERTBufferIn> mSegmentIdBufs;
    std::vector<BERTBufferIn> mInputMaskBufs;
    std::vector<BERTBufferIn> mDummy;
    std::vector<BERTBufferOut> mOutputBufs;

    std::vector<std::vector<void*>> mBindings;

    // BERTCore manages a pool of threads to process the responses asynchronously wrt. the inference
    std::vector<std::thread> mResponseThreads;
    std::deque<BERTResponse> mResultQ;
    std::mutex mMtx;
    std::condition_variable mCondVar;
    bool mStopWork{false};

    void EnqueueResponse(const BERTResponse & r) {
        std::unique_lock<std::mutex> lck(mMtx);
        mResultQ.emplace_back(r);
        mCondVar.notify_one();
    }

    // Logic of a response thread. Calls to QSL to write out the response
    static void ProcessResponse(BERTCoreVS* bertCore);
    
};
