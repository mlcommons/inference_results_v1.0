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

#include "batch_maker.hpp"
#include "utils.hpp"

Batch::Batch(std::string id,
             size_t minBatchSize,
             size_t maxBatchSize,
             size_t numericVolume,
             size_t categoricalVolume,
             BatchMaker* batchMaker)
    : mDebugId(id),
      mBatchMaker(batchMaker),
      mMinBatchSize(minBatchSize),
      mMaxBatchSize(maxBatchSize),
      mNumericInputBuf(numericVolume, maxBatchSize, true, false),
      mCategoricalInputBuf(categoricalVolume, maxBatchSize, true, false)
{
    reset();
}

void Batch::ContiguityAwareH2H(size_t tasksOffset, size_t pairsOffset, size_t pairsToCopy)
{
    // We assume tasks before `tasksOffset` (if any) have already been processed by this function and are contiguous.
    // We check for 2 things:
    // - internal contiguity: whether all tasks [tasksOffset, end) are contiguous, by comparing the expected vs actual address of last task
    // - external contiguity: whether task at `tasksOffset` is contiguous with that at tasksOffset-1
    // If both are contiguous, we skip H2H and point the input buffer's to QSL host memory instead.

    CHECK(mBatchMaker->mCheckContiguity);

    NVTX_RANGE_PUSH("Contiguity H2H");
    auto qsl = mBatchMaker->mQsl;
    size_t numericInputVolume = mNumericInputBuf.ElemByteSize() / sizeof(DLRMNumericInputType);

    DLRMTask& firstTask = mTasks[tasksOffset];
    DLRMTask& lastTask = mTasks.back();

    // first address in the contiguous block starting from mTasks[0]
    DLRMNumericInputType* globalContigStart = static_cast<DLRMNumericInputType*>(qsl->GetSampleAddress(mTasks[0].querySample.index, 0));
    // address right after the end of last contiguous block
    DLRMNumericInputType* expectedFirstTaskStart = globalContigStart + (pairsOffset * numericInputVolume);

    // first address of current subset of mTtasks
    DLRMNumericInputType* actualFirstTaskStart = static_cast<DLRMNumericInputType*>(qsl->GetSampleAddress(firstTask.querySample.index, 0));
    // expected end of lastTaskStart, if current tasks are contiguous
    DLRMNumericInputType* expectedLastTaskStart = actualFirstTaskStart + (pairsToCopy * numericInputVolume);
    // starting address of last batch in this set of tasks
    DLRMNumericInputType* actualLastTaskStart = static_cast<DLRMNumericInputType*>(qsl->GetSampleAddress(lastTask.querySample.index, 0)) +
                                                (qsl->GetNumUserItemPairs(lastTask.querySample.index) * numericInputVolume);

    // is there contiguity from mTasks[tasksOffset] to mTasks.back
    bool internalContiguity = (expectedLastTaskStart == actualLastTaskStart);
    // is there contiguity from mTasks[0] to mTasks[tasksOffset]
    bool externalContiguity = (actualFirstTaskStart == expectedFirstTaskStart);

    if(!(externalContiguity && internalContiguity))
    {
        if(!mNumericInputBuf.isDefaultHostPtr())
        {
            // First transfer existing contiguous block from QSL to own buffer
            mNumericInputBuf.ResetHostPtr();
            mCategoricalInputBuf.ResetHostPtr();
            mNumericInputBuf.H2H(globalContigStart, pairsOffset, 0);
            mCategoricalInputBuf.H2H(
                static_cast<DLRMCategoricalInputType*>(qsl->GetSampleAddress(mTasks[0].querySample.index, 1)),
                pairsOffset, 0);
        }

        if(internalContiguity)
        {
            // current set of tasks is contiguous, so can append pairsToCopy to mNumericInputBuf in one go
            mNumericInputBuf.H2H(actualFirstTaskStart, pairsToCopy, pairsOffset);
            mCategoricalInputBuf.H2H(static_cast<DLRMCategoricalInputType*>(qsl->GetSampleAddress(firstTask.querySample.index, 1)),
            pairsToCopy,
            pairsOffset);
        }
        else
        {
            // discontiguous, but don't know where. Copy tasks one by one
            IndividualH2H(tasksOffset, pairsOffset, pairsToCopy);
        }
    }
    else
    {
        // all contiguous, point to QSL memory
        mNumericInputBuf.SetHostPtr(globalContigStart);
        mCategoricalInputBuf.SetHostPtr(static_cast<DLRMCategoricalInputType*>(qsl->GetSampleAddress(mTasks[0].querySample.index, 1)));
    }

    NVTX_RANGE_POP();
}

void Batch::IndividualH2H(size_t tasksOffset, size_t pairsOffset, size_t pairsToCopy)
{
    auto qsl = mBatchMaker->mQsl;
    size_t copiedPairs = 0;

    while (copiedPairs < pairsToCopy)
    {
        const auto& task = mTasks[tasksOffset++];
        auto qslIdx = task.querySample.index;
        auto numIndividualPairs = task.numIndividualPairs;

        mNumericInputBuf.H2H(
            static_cast<DLRMNumericInputType*>(qsl->GetSampleAddress(qslIdx, 0)),
            numIndividualPairs,
            pairsOffset + copiedPairs);
        mCategoricalInputBuf.H2H(
            static_cast<DLRMCategoricalInputType*>(qsl->GetSampleAddress(qslIdx, 1)),
            numIndividualPairs,
            pairsOffset + copiedPairs);

        copiedPairs += numIndividualPairs;
    }
}

void Batch::doCopy(size_t tasksOffset, size_t pairsOffset, size_t pairsToCopy)
{
    NVTX_RANGE_PUSH(("Copying batch " + mDebugId).c_str());
    DLOG(INFO) << mDebugId << " doCopy : " << pairsToCopy;

    if (mBatchMaker->mCheckContiguity)
    {
        ContiguityAwareH2H(tasksOffset, pairsOffset, pairsToCopy);
    }
    else
    {
        IndividualH2H(tasksOffset, pairsOffset, pairsToCopy);
    }
    NVTX_RANGE_POP();
}

void Batch::reset()
{
    mCommittedCopies = 0;
    mCompletedCopies = 0;
    mReadyWhenComplete = false;
    mOddBatch = false;
    mTasks.clear();
    mTasks.reserve(mMaxBatchSize);
    mNumericInputBuf.ResetHostPtr();
    mCategoricalInputBuf.ResetHostPtr();
}

void Batch::padToEvenSize()
{
    size_t actualBatchSize = mCommittedCopies;
    mOddBatch = (actualBatchSize % 2) == 1;
    if (mOddBatch)
    {
        // push a dummy task & update completed copies, but no actual H2H copy is performed
        // at time of H2D, only the first `actualBatchSize` pairs will be copied to device
        CHECK(!mTasks.empty());
        DLRMTask padBatch = mTasks.back();
        padBatch.numIndividualPairs = 1;
        mTasks.push_back(padBatch);
        commitCopies(1);
        completeCopies(1);
    }
}

BatchMaker::BatchMaker(size_t numStagingThreads,
                       size_t numStagingBatches,
                       size_t maxBatchSize,
                       size_t maxPairsPerThread,
                       size_t numericVolume,
                       size_t categoricalVolume,
                       bool checkContiguity,
                       std::shared_ptr<DLRMSampleLibrary> qsl,
                       std::vector<int> cpus)
    : mMaxBatchSize(maxBatchSize),
      mMaxPairsPerThread(maxPairsPerThread),
      mCheckContiguity(checkContiguity),
      mQsl(qsl),
      mStagingBatch(nullptr),
      mIssuedPairs(0),
      mReadiedPairs(0),
      mFlushQueries(false),
      mStopWork(false),
      mWaitingBatches(0),
      mCpus(cpus)
{
    DLOG(INFO) << "BatchMaker - numStagingThreads = " << numStagingThreads;
    DLOG(INFO) << "BatchMaker - numStagingBatches = " << numStagingBatches;
    DLOG(INFO) << "BatchMaker - maxPairsPerThread = " << maxPairsPerThread;

    if (mCheckContiguity)
    {
        LOG(INFO) << "Contiguity-Aware H2H : ON";
        if (mMaxPairsPerThread == 0) { mMaxPairsPerThread = mMaxBatchSize; }
    }
    else
    {
        LOG(INFO) << "Contiguity-Aware H2H : OFF";
        if (mMaxPairsPerThread == 0) { mMaxPairsPerThread = mMaxBatchSize / numStagingThreads; }
    }

    mInputBatches.reserve(numStagingBatches);
    for(int i = 0; i < numStagingBatches; ++i)
    {
        mInputBatches.emplace_back("Batch#"+std::to_string(i), 1, mMaxBatchSize, numericVolume, categoricalVolume, this);
    }

    // All batches idle initially
    mIdleBatches.reserve(numStagingBatches);
    for(auto& batch : mInputBatches)
    {
        mIdleBatches.push_back(&batch);
    }

    // start StageBatchthreads
    mStagingThreads.reserve(numStagingThreads);
    for(int i = 0; i < numStagingThreads; ++i)
    {
        mStagingThreads.emplace_back(&BatchMaker::StageBatch, this, i);
    }

    // Limit the staging threads to the closest CPUs.
    if (UseNuma())
    {
        for(auto& stagingThread : mStagingThreads)
        {
            bindThreadToCpus(stagingThread, mCpus);
        }
    }
}

BatchMaker::~BatchMaker()
{
    DLOG(INFO) << "~BatchMaker";
    StopWork();
    for(auto& stagingThread : mStagingThreads)
    {
        stagingThread.join();
    }
}

void BatchMaker::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    NVTX_RANGE_PUSH("BatchMaker::IssueQuery");
    mFlushQueries = false;
    for (const auto& sample : samples)
    {
        {
            std::unique_lock<std::mutex> lock(mMutex);

            size_t numPairs = mQsl->GetNumUserItemPairs(sample.index);
            mTasksQ.push_back({sample, numPairs});
            mIssuedPairs += numPairs;
        }
    }
    mProducerCV.notify_one();
    NVTX_RANGE_POP();
}

void BatchMaker::FlushQueries()
{
    NVTX_RANGE_PUSH("BatchMaker::FlushQueries");
    DLOG(INFO) << "FlushQueries";
    std::unique_lock<std::mutex> mMutex;
    if (mTasksQ.empty() && mStagingBatch && mStagingBatch->getCommittedCopies() > 0)
    {
        // close this buffer on my own, since no thread will do it until new tasks arrive
        CloseBatch(mStagingBatch);
    }
    mFlushQueries = true;
    NVTX_RANGE_POP();
}

void BatchMaker::StopWork()
{
    NVTX_RANGE_PUSH("BatchMaker::StopWork");
    DLOG(INFO) << "Stop Work";
    std::unique_lock<std::mutex> lock(mMutex);
    mStopWork = true;
    mProducerCV.notify_all();
    mConsumerCV.notify_all();
    NVTX_RANGE_POP();
}

Batch* BatchMaker::GetBatch()
{
    NVTX_RANGE_PUSH("BatchMaker::GetBatch");
    DLOG(INFO) << "GetBatch";
    NVTX_RANGE_PUSH("BatchMaker::GetBatch lock acquire");
    std::unique_lock<std::mutex> lock(mMutex);
    NVTX_RANGE_POP();

    if (mReadyBatches.empty())
    {
        NVTX_RANGE_PUSH("Waiting for ready batch");
        if (mStagingBatch)
        {
            CloseBatch(mStagingBatch);
        }
        ++mWaitingBatches;
        DLOG(INFO) << "GetBatch Waiting";
        mConsumerCV.wait(lock, [&] { return !mReadyBatches.empty() || mStopWork; });
        DLOG(INFO) << "GetBatch notified";
        --mWaitingBatches;
        NVTX_RANGE_POP();
    }

    if (mStopWork)
    {
        NVTX_RANGE_POP();
        LOG(INFO) << "GetBatch Done";
        return nullptr;
    }

    Batch* readyBatch = mReadyBatches.front();
    NVTX_RANGE_PUSH(("Preparing batch " + std::string(readyBatch->mDebugId)).c_str());
    mReadyBatches.pop_front();
    auto completedCopies = readyBatch->getCompletedCopies();
    mReadiedPairs += readyBatch->isOddBatch() ? (completedCopies - 1) : completedCopies;

    DLOG(INFO) << "Sending " << readyBatch->mDebugId << "; batchSize : " << readyBatch->getCompletedCopies() << " mReadiedPairs : " << mReadiedPairs << " mIssuedPairs : " << mIssuedPairs;
    CHECK_LE(mReadiedPairs, mIssuedPairs);

    if (!mReadyBatches.empty())
    {
        mConsumerCV.notify_one();
    }
    NVTX_RANGE_POP();
    NVTX_RANGE_POP();

    return readyBatch;
}

void BatchMaker::HandleReadyBatch(Batch* batch)
{
    DLOG(INFO) << "Ready : " << batch->mDebugId << " " << batch->getCompletedCopies();

    auto actualBatchSize = batch->getCompletedCopies();
    CHECK(batch->isComplete());
    CHECK_GT(actualBatchSize, 0);

    batch->padToEvenSize();

    mReadyBatches.push_back(batch);
    mConsumerCV.notify_one();
}

void BatchMaker::NotifyH2D(Batch* batch)
{
    NVTX_RANGE_PUSH(("BatchMaker::NotifyH2D for batch " + batch->mDebugId).c_str());
    DLOG(INFO) << "Notify " << batch->mDebugId << "; flush = " << mFlushQueries;
    batch->reset();

    {
        std::unique_lock<std::mutex> lock(mMutex);
        mIdleBatches.push_back(batch);
    }

    mProducerCV.notify_one();
    NVTX_RANGE_POP();
}

void BatchMaker::StageBatch(int threadIdx)
{
    while(true)
    {
        NVTX_RANGE_PUSH("BatchMaker::StageBatch iteration");
        Batch* batchPtr;
        std::vector<DLRMTask> committedTasks;
        size_t pairsToCopy = 0;
        size_t tasksOffset = 0;
        size_t pairsOffset = 0;

        // de-queue tasks and commit copies, under lock
        {
            NVTX_RANGE_PUSH("BatchMaker::StageBatch aquire lock");
            std::unique_lock<std::mutex> lock(mMutex);
            DLOG(INFO) << "StageBatch waiting";
            mProducerCV.wait(lock, [&] { return ((mStagingBatch || !mIdleBatches.empty()) && !mTasksQ.empty()) || mStopWork; });
            DLOG(INFO) << "StageBatch notified";
            NVTX_RANGE_POP();

            if (mStopWork) { NVTX_RANGE_POP(); return; }

            if (!mStagingBatch)
            {
                CHECK(!mIdleBatches.empty());
                mStagingBatch = mIdleBatches.back();
                mIdleBatches.pop_back();
                DLOG(INFO) << "Stage Batch set to" << mStagingBatch->mDebugId;
            }

            // mStagingBatch may change once lock is released, so store a copy
            batchPtr = mStagingBatch;
            tasksOffset = batchPtr->getTasks().size();
            pairsOffset = batchPtr->getCommittedCopies();
            size_t maxPairs = std::min(mMaxPairsPerThread, batchPtr->getFreeSpace());
            NVTX_RANGE_PUSH(("Comitting max " + std::to_string(maxPairs) + " pairs to batch " + batchPtr->mDebugId).c_str());

            DLOG(INFO) << "thread " << threadIdx << " with " << batchPtr->mDebugId << ":+" << pairsOffset;

            while(!mTasksQ.empty())
            {
                auto task = mTasksQ.front();
                size_t numPairs = task.numIndividualPairs;

                if(numPairs + pairsToCopy > maxPairs)
                {
                    if(numPairs >= batchPtr->getFreeSpace())
                    {
                        // batch can't fit next sample
                        DLOG(INFO) << pairsToCopy << " : Break because full";
                        CloseBatch(batchPtr);
                    }
                    else
                    {
                        DLOG(INFO) << pairsToCopy << " : Break because maxPairs";
                    }
                    // Let some other thread commit remaining tasks
                    break;
                }

                pairsToCopy += numPairs;
                batchPtr->commitCopies(numPairs);
                batchPtr->pushTask(task);
                mTasksQ.pop_front();
            }

            if (pairsToCopy == 0)
            {
                NVTX_RANGE_POP();
                NVTX_RANGE_POP();
                continue;
            }

            DLOG(INFO) << "Commit " << pairsToCopy << " pairs in " << committedTasks.size() << " tasks";

            if ((mTasksQ.empty() && mFlushQueries) || mWaitingBatches)
            {
                // no more queries
                CloseBatch(batchPtr);
            }
            mProducerCV.notify_one();
            NVTX_RANGE_POP();
        }

        // do H2H, without lock
        batchPtr->doCopy(tasksOffset, pairsOffset, pairsToCopy);

        // mark copy as complete, under lock
        {
            NVTX_RANGE_PUSH("BatchMaker::StageBatch complete copy aquire lock");
            std::unique_lock<std::mutex> lock(mMutex);
            NVTX_RANGE_POP();
            NVTX_RANGE_PUSH("BatchMaker::StageBatch complete copy");
            batchPtr->completeCopies(pairsToCopy);
            ReadyBatchIfComplete(batchPtr);
            mConsumerCV.notify_one();
            NVTX_RANGE_POP();
        }

        NVTX_RANGE_POP();
    }
}

void BatchMaker::CloseBatch(Batch* batch)
{
    if (!batch->isReadyWhenComplete())
    {
        DLOG(INFO) << batch->mDebugId << " closing";
        // batch will move to mReadyBatches once copies complete
        batch->markReadyWhenComplete();
        // if already complete, move to ready
        ReadyBatchIfComplete(batch);
        // next thread will set new staging batch
        mStagingBatch = nullptr;
    }
}

void BatchMaker::ReadyBatchIfComplete(Batch* batch)
{
    if (batch->isComplete())
    {
        HandleReadyBatch(batch);
    }
}
