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

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <string>

#include <nvToolsExt.h>

#include "qsl.hpp"

#include "dlrm_qsl.hpp"
#include "lwis_buffers.h"

using DLRMNumericInputType = int8_t;
using DLRMCategoricalInputType = int32_t;
using DLRMOutputType = float;
using DLRMInferCallback = std::function<void(std::vector<mlperf::QuerySampleResponse>&)>;

#define NVTX 0

#if NVTX
#define NVTX_MARK(message) nvtxMarkA(message);
#define NVTX_NAME_THIS_THREAD(name) nvtxNameOsThreadA(pthread_self(), name)
#define NVTX_RANGE_PUSH(message) nvtxRangePushA(message);
#define NVTX_RANGE_POP() nvtxRangePop();
#else
#define NVTX_MARK(message)
#define NVTX_NAME_THIS_THREAD(name)
#define NVTX_RANGE_PUSH(message)
#define NVTX_RANGE_POP()
#endif

constexpr size_t kMIN_NUM_PAIRS = 100;

struct DLRMTask
{
    mlperf::QuerySample querySample;
    size_t numIndividualPairs;
};

/* DLRMDataBuffer abstracts a host and device buffer and provides H2DAsync and D2HAsync. However, it does not assume
 * that `host` refers to its own internal buffer to allow passing in arbitrary host memory addresses. To use its
 * internal buffer, use `GetHostPtr()`.
 */
template<typename T>
class DLRMDataBuffer
{
  public:
    DLRMDataBuffer(size_t length, size_t maxBatchSize, bool allocHost=true, bool allocDevice=true) :
        mLength(length),
        mMaxBatchSize(maxBatchSize),
        mMaxBytes(maxBatchSize * sizeof(T) * mLength),
        mHostBuffer(allocHost ? mMaxBytes : 0),
        mDeviceBuffer(allocDevice ? mMaxBytes : 0),
        mHostPtr(static_cast<T*>(mHostBuffer.data())),
        mDevicePtr(static_cast<T*>(mDeviceBuffer.data()))
    {
        CHECK_EQ(mMaxBatchSize > 0, true);
        CHECK_EQ(mMaxBytes > 0, true);
        CHECK_EQ(!allocHost || (mHostPtr != nullptr), true);
        CHECK_EQ(!allocDevice || (mDevicePtr != nullptr), true);
    }

    void H2H(const T* hostPtr, size_t size, size_t offset = 0)
    {
        CHECK_EQ(mHostPtr != nullptr, true);
        memcpy(GetHostPtr() + offset * mLength, hostPtr, ElemByteSize() * size);
    }

    void H2DAsync(const T* hostPtr, size_t size, cudaStream_t stream, bool oddBatch = false)
    {
        // if oddBatch, only the first (size - 1) inputs on host have valid data
        // the last one will not be copied
        auto unpaddedBatchSize = oddBatch ? size - 1 : size;
        CHECK_EQ(mDevicePtr != nullptr, true);
        CHECK_EQ(cudaMemcpyAsync(
                mDevicePtr,
                hostPtr,
                ElemByteSize() * unpaddedBatchSize,
                cudaMemcpyHostToDevice,
                stream
            ),
            cudaSuccess);
        if (oddBatch)
        {
          CHECK_EQ(cudaMemcpyAsync(
                mDevicePtr + unpaddedBatchSize * mLength,
                hostPtr,
                ElemByteSize() * 1,
                cudaMemcpyHostToDevice,
                stream
            ),
            cudaSuccess);
        }
    }

    void D2HAsync(T* hostPtr, size_t size, cudaStream_t stream)
    {
        CHECK_EQ(mDevicePtr != nullptr, true);
        CHECK_EQ(cudaMemcpyAsync(
                hostPtr,
                mDevicePtr,
                ElemByteSize() * size,
                cudaMemcpyDeviceToHost,
                stream
            ),
            cudaSuccess);
    }

    size_t ElemByteSize() { return mLength * sizeof(T); }
    T* const GetDevicePtr() { return mDevicePtr; }
    T* const GetHostPtr() { return mHostPtr; }
    void SetHostPtr(T* ptr) { mHostPtr = ptr; }
    void ResetHostPtr() { mHostPtr = static_cast<T*>(mHostBuffer.data()); }
    bool isDefaultHostPtr() { return mHostPtr == static_cast<T*>(mHostBuffer.data()); }

  private:
    size_t mLength;
    size_t mMaxBatchSize;
    size_t mMaxBytes;
    lwis::HostBuffer mHostBuffer;
    lwis::DeviceBuffer mDeviceBuffer;
    T* const mDevicePtr;
    T* mHostPtr;
};

// Aliases for convenience
using DLRMNumericBuffer = DLRMDataBuffer<DLRMNumericInputType>;
using DLRMCategoricalBuffer = DLRMDataBuffer<DLRMCategoricalInputType>;
using DLRMOutputBuffer = DLRMDataBuffer<DLRMOutputType>;

class BatchMaker;

class Batch
{
  public:
    Batch(std::string id, size_t maxBatchSize, size_t minBatchSize, size_t numericVolume, size_t categoricalVolume, BatchMaker* batchMaker);

    // Copy task data from QSL memory to this Batch's host buffers
    void doCopy(size_t tasksOffset, size_t pairsOffset, size_t pairsToCopy);
    // Reset batch state after consumption in inference
    void reset();
    // pad batch size to be even if needed, and set a flag to indicate this
    void padToEvenSize();

    void pushTask(DLRMTask task) { mTasks.push_back(task); }
    std::vector<DLRMTask> const getTasks() const { return mTasks; }

    void commitCopies(size_t numCopies) { mCommittedCopies += numCopies; }
    size_t getCommittedCopies() const { return mCommittedCopies; }

    void completeCopies(size_t numCopies)  { mCompletedCopies += numCopies; }
    size_t getCompletedCopies() const { return mCompletedCopies; }

    void markReadyWhenComplete() { mReadyWhenComplete = true; }
    bool isReadyWhenComplete() const { return mReadyWhenComplete; }
    bool isComplete() const { return mCompletedCopies == mCommittedCopies && mReadyWhenComplete; }

    bool isOddBatch() const { return mOddBatch; }

    size_t getFreeSpace() const { return mMaxBatchSize - mCommittedCopies; }

    std::string mDebugId;
    BatchMaker* mBatchMaker;

    DLRMNumericInputType* const getNumericHostPtr() { return mNumericInputBuf.GetHostPtr(); }
    DLRMCategoricalInputType* const getCategoricalHostPtr() { return mCategoricalInputBuf.GetHostPtr(); }

  private:
    void ContiguityAwareH2H(size_t tasksOffset, size_t pairsOffset, size_t pairsToCopy);
    void IndividualH2H(size_t tasksOffset, size_t pairsOffset, size_t pairsToCopy);

    std::vector<DLRMTask> mTasks;

    DLRMNumericBuffer mNumericInputBuf;
    DLRMCategoricalBuffer mCategoricalInputBuf;

    size_t mCommittedCopies;
    size_t mCompletedCopies;
    size_t mMinBatchSize;
    size_t mMaxBatchSize;

    bool mReadyWhenComplete;
    bool mOddBatch;
};

class BatchMaker
{
  public:
    BatchMaker(size_t numStagingThreads,
               size_t numStagingBatches,
               size_t maxBatchSize,
               size_t maxPairsPerThread,
               size_t numericVolume,
               size_t categoricalVolume,
               bool checkContiguity,
               std::shared_ptr<DLRMSampleLibrary> qsl,
               std::vector<int> cpus);
    ~BatchMaker();

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples, int offset, int count);
    void FlushQueries();
    void StopWork();

    // Interface to fetch the earliest Batch from the queue of Batches ready for inference,
    // blocking if no batch is ready
    Batch* GetBatch();

    // Interface to notify that the inference thread has completed H2D transfer
    // of a `batch` and it can be returned to the pool of idle batches
    void NotifyH2D(Batch *batch);

    std::shared_ptr<DLRMSampleLibrary> mQsl;

    bool mCheckContiguity;
  private:

    // Greedily fetches tasks, and copies them to the active staging batch
    void StageBatch(int threadIdx);

    // Pushes a staged batch to the queue of batches ready for inference
    // NOTE: Should be called under mutex lock
    void HandleReadyBatch(Batch *batch);

    // Close this batch for new copies, and handle related bookkeeping
    void CloseBatch(Batch* batch);
    // Check if batch satisfies conditions to be marked as ready
    void ReadyBatchIfComplete(Batch* batch);

    // All Batches owned by this BatchMaker
    std::vector<Batch> mInputBatches;

    // Pointers to idle empty Batches, which are not ready, not staging, and not being used for inference
    std::vector<Batch*> mIdleBatches;
    // Queue of staged Batches that are ready to be sent for inference
    std::deque<Batch*> mReadyBatches;
    // Pointer to Batch currently being used for inference by all StageBatch threads
    Batch* mStagingBatch;

    // Queue of task IDs coming from LoadGen which are to be staged
    std::deque<DLRMTask> mTasksQ;

    // Threads running StageBatch
    std::vector<std::thread> mStagingThreads;
    // Mutex to serialize access mStagingBatch & mIdleBatches
    std::mutex mMutex;

    // Condition variable on which StageBatch will wait to produce batches
    std::condition_variable mProducerCV;
    // Condition variable on which GetBatch will wait to consume batches
    std::condition_variable mConsumerCV;

    // Indicates that there will no new tasks and the worker threads should stop processing samples
    bool mStopWork;
    // Indicates FlushQueries has been called by LoadGen
    bool mFlushQueries;
    // Total number of readied pairs sent for inference
    size_t mReadiedPairs;
    // Total number of pairs issued by LoadGen in IssueQuery
    size_t mIssuedPairs;

    size_t mMinBatchSize;
    size_t mMaxBatchSize;
    size_t mMaxPairsPerThread;
    size_t mWaitingBatches;

    // CPUs on which staging threads should run. Empty implies no constraint.
    std::vector<int> mCpus;
    bool UseNuma() { return !mCpus.empty(); };
};
