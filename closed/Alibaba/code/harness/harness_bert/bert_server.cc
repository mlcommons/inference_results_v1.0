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

#include "bert_server.h"
#include "bert_core_vs.h"

#include "glog/logging.h"
#include "loadgen.h"

#include <fstream>
#include <set>

#include <nvtx3/nvToolsExt.h> // For NVTX annotations

// ================================
//     Debug support: nvtx ranges
// ================================

// #define NVTX_ON

#ifdef NVTX_ON
   enum nvtx_color  {
      COLOR_BLUE_0  = 255,     COLOR_BLUE_1  = 200,     COLOR_BLUE_2  = 150,     COLOR_BLUE_3  = 100,
      COLOR_GREEN_0 = 255<<8,  COLOR_GREEN_1 = 200<<8,  COLOR_GREEN_2 = 150<<8,  COLOR_GREEN_3 = 100<<8,
      COLOR_RED_0   = 255<<16, COLOR_RED_1   = 200<<16, COLOR_RED_2   = 150<<16, COLOR_RED_3   = 100<<16,
   };
   #define NVTX_GLOBAL_START(A, B, C)   nvtxRangeId_t A = global_event_start(B, C)
   #define NVTX_GLOBAL_END(A)           global_event_end(A)
   #define NVTX_THREAD_START(B, C)      thread_event_start(B, C)
   #define NVTX_THREAD_END()            thread_event_end()
   #define NVTX_MARK(B, C)              mark_event(B, C)
#else
   #define NVTX_GLOBAL_START(A, B, C)
   #define NVTX_GLOBAL_END(A)
   #define NVTX_THREAD_START(B, C)
   #define NVTX_THREAD_END()
   #define NVTX_MARK(B, C)
#endif


#ifdef NVTX_ON
#define CREAT_NVTX_EVENT_ATTRIB(A) \
    nvtxEventAttributes_t A = {0}; \
    A.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    A.version = NVTX_VERSION; \
    A.colorType = NVTX_COLOR_ARGB; \
    A.color = eventColor; \
    A.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    A.message.ascii = eventName.data()

nvtxRangeId_t global_event_start(const std::string &eventName, const nvtx_color eventColor)
{
    CREAT_NVTX_EVENT_ATTRIB(eventAttrib);
    nvtxRangeId_t corrId = nvtxRangeStartEx(&eventAttrib);
    return (corrId);
}

void global_event_end(nvtxRangeId_t corrId)
{
    nvtxRangeEnd(corrId);
}

void thread_event_start(const std::string &eventName, const nvtx_color eventColor)
{
    CREAT_NVTX_EVENT_ATTRIB(eventAttrib);
    nvtxRangeId_t corrId = nvtxRangePushEx(&eventAttrib);
}

void thread_event_end()
{
    nvtxRangePop();
}

void mark_event(const std::string &eventName, const nvtx_color eventColor)
{
    CREAT_NVTX_EVENT_ATTRIB(eventAttrib);
    nvtxMarkEx(&eventAttrib);
}
#endif

template<typename T>
void BERTServer::ProcessTasks(std::shared_ptr<T> bertCore, int deviceId, int qThreadIdx)
{
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
    uint64_t totalCountInThread = 0;

    // hold soft drop tasks if any
    BERTTask_t holdedTasks;

    // Process samples in batches
    NVTX_THREAD_START("GetTasks", COLOR_BLUE_1);
    auto tasks = GetTasks(mMaxBatchSize, qThreadIdx);
    NVTX_THREAD_END();

    while(!tasks.empty())
    {
        totalCountInThread += tasks.size();
        if (mSoftDrop < 1.0) {
            std::unique_lock<std::mutex> lck(mSoftDropMtx);
            mTotalTasksCount += tasks.size();

            mTotalLengthSet.InsertTasks(tasks, mQsl);

            // Drop requests until the total length is not greater than the threshold
            // Use target latency percentile as a hard limit on how many requests we can drop
            while(BERTCoreVS::CountTotalLength(tasks, mQsl) > mTotalLengthSet.GetThresholdLength() &&
                  mSoftDropCount <= std::floor(static_cast<double>(mTotalTasksCount) * (1.0 - mTargetLatencyPercentile)) - 1) {
                holdedTasks.push_back(tasks.front());
                tasks.erase(tasks.begin());
                ++mSoftDropCount;
            }
        }
        NVTX_THREAD_START("infer:" + std::to_string(tasks.size()) + " tasks", COLOR_BLUE_0);
        bertCore->infer(tasks, mQsl);
        NVTX_THREAD_END();

        NVTX_THREAD_START("GetTasks", COLOR_BLUE_1);
        tasks = GetTasks(mMaxBatchSize, qThreadIdx);
        NVTX_THREAD_END();
    }

    if (mSoftDrop < 1.0) {
        // Process soft drop tasks if any
        LOG(INFO) << "Total number of soft drop tasks: " << holdedTasks.size() << " out of " << totalCountInThread << " total tasks";
        while(holdedTasks.size() != 0) {
            std::vector<std::pair<mlperf::QuerySample, std::chrono::high_resolution_clock::time_point>> tasks;
            tasks.reserve(mMaxBatchSize);
            // Consume up to mMaxBatchSize tasks
            for(int i = 0; (i < mMaxBatchSize) && !holdedTasks.empty(); ++i) {
                tasks.push_back(holdedTasks.back());
                holdedTasks.pop_back();
            }
            NVTX_THREAD_START("infer:" + std::to_string(tasks.size()) + " tasks", COLOR_BLUE_0);
            bertCore->infer(tasks, mQsl);
            NVTX_THREAD_END();
        }
        // This is necessary to avoid a race condition if the bertCore is destructed before we process all responses
        bertCore->WaitUntilQueueEmpty();
    }

    using CLK = std::chrono::high_resolution_clock;
    VLOG(1) << "End of ProcessTasks: " << std::chrono::duration_cast<std::chrono::microseconds>(CLK::now().time_since_epoch()).count();
}

void BERTServer::StartIssueThread(int threadIdx) {
    {
        CHECK_EQ(!mMtxs->empty(), true);
        std::lock_guard<std::mutex> lock((*mMtxs)[0]);
        mThreadMap[std::this_thread::get_id()] = threadIdx;
    }
    mlperf::RegisterIssueQueryThread();
}

static void createModelStreams(const std::string& enginePath, std::vector<std::vector<char>>& trtModelStreams) {
    //we get a comma-separated list of engine paths
    std::vector<std::string> paths;
    int from = 0;
    int to;
    while((to = enginePath.find(',', from)) != std::string::npos) {
        paths.emplace_back(enginePath.substr(from, to - from));
        from = to + 1;
    }

    if(from < enginePath.size()) {
        paths.emplace_back(enginePath.substr(from, enginePath.size() - from));
    }

    for(auto &p : paths) {
        LOG(INFO) << "Engine Path: " << p;
    }

    trtModelStreams.resize(paths.size());
    for(size_t i = 0; i < trtModelStreams.size(); ++i) {
        lwis::GetModelStream(trtModelStreams[i], paths[i]);
    }
}

void BERTServer::CreateEnginesPerGPU(int deviceId, std::shared_ptr<std::mutex> pMtx, const std::vector<std::vector<char>>& trtModelStreams) {
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);

    auto runtime = InferObject(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));

    // load all the engines
    std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> inferObjects(trtModelStreams.size());
    std::transform(trtModelStreams.begin(), trtModelStreams.end(), inferObjects.begin(), [&](const std::vector<char>& trtModelStream) {
        return InferObject(runtime->deserializeCudaEngine(trtModelStream.data(), trtModelStream.size(), nullptr));
    });

    {
        std::unique_lock<std::mutex> lck(*pMtx.get());
        mEnginesPerGPU[deviceId] = std::move(inferObjects);
    }
}

BERTServer::BERTServer(const std::string name, const std::string enginePath,
                       std::shared_ptr<qsl::SampleLibrary> qsl, const std::vector<int>& gpus,
                       int maxBatchSize, int numCopyStreams, int numBERTCores, bool useGraphs,
                       int graphMaxSeqLen, const std::string& graphSpecs, double softDrop, double targetLatencyPercentile,
                       uint64_t serverNumIssueQueryThreads)
    : mName{name}, mQsl{qsl}, mStopGetTasks{false}, mStopProcessResponse{false}, mMaxBatchSize{maxBatchSize}, mGraphMaxSeqLen{graphMaxSeqLen},
      mSoftDrop{softDrop}, mTargetLatencyPercentile{targetLatencyPercentile}, mTotalLengthSet{mSoftDrop}, mTotalTasksCount{0}, mSoftDropCount{0}
{
    {
        // only create one model streams
        std::vector<std::vector<char>> trtModelStreams;
        createModelStreams(enginePath, trtModelStreams);

        // create TRT engines in parallel
        std::shared_ptr<std::mutex> pMtx = std::make_shared<std::mutex>();
        std::vector<std::thread> engineCreationThreads;
        for(auto deviceId : gpus) {
            engineCreationThreads.emplace_back(&BERTServer::CreateEnginesPerGPU, this, deviceId, pMtx, trtModelStreams);
        }
        for(auto& thread : engineCreationThreads) {
            thread.join();
        }
        LOG(INFO) << "Engines Creation Completed";
    }

    if (useGraphs) {
        LOG(INFO) << "Use CUDA graphs";
    }

    // Create BERTCoreVS and store in temporary vector, capture CUDA graphs in parallel
    using BERTCoreVSPtrVec = std::vector<std::shared_ptr<BERTCoreVS>>;
    std::vector<BERTCoreVSPtrVec> tmpBERTCores(gpus.size());
    for(int profileIdx = 0; profileIdx < numBERTCores; ++profileIdx) {
        std::vector<std::thread> cudaGraphsCapturingThreads;
        for(int idx = 0; idx < gpus.size(); ++idx) {
            auto deviceId = gpus[idx];
            CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
            tmpBERTCores[idx].push_back(std::make_shared<BERTCoreVS>(mEnginesPerGPU.at(deviceId), numCopyStreams, numBERTCores, profileIdx, useGraphs, deviceId));

            // Capture CUDA graphs in parallel
            if (useGraphs) {
                cudaGraphsCapturingThreads.emplace_back(&BERTCoreVS::BuildGraphs, tmpBERTCores[idx][profileIdx].get(), mGraphMaxSeqLen, graphSpecs, numBERTCores);
            }
        }
        if (useGraphs) {
            for(auto& thread : cudaGraphsCapturingThreads) {
                thread.join();
            }
            // LOG(INFO) << "Sleep for 5 minute";
            // std::this_thread::sleep_for(std::chrono::seconds(360));
        }
    }

    if (mSoftDrop < 1.0) {
        LOG(INFO) << "Apply soft drop policy with threshold = " << mSoftDrop;
    }

    if (serverNumIssueQueryThreads > 0) {
        CHECK_EQ((gpus.size() * numBERTCores) % serverNumIssueQueryThreads == 0, true);
        LOG(INFO) << "Use number of server IssueQuery threads = " << serverNumIssueQueryThreads;
        mTasksVec.resize(serverNumIssueQueryThreads);
        mMtxs = std::make_unique<std::vector<std::mutex>>(serverNumIssueQueryThreads);
        mCondVars = std::make_unique<std::vector<std::condition_variable>>(serverNumIssueQueryThreads);
        for (int i = 0; i < serverNumIssueQueryThreads; ++i) {
            mIssueQueryThreads.emplace_back(&BERTServer::StartIssueThread, this, i);
        }
    } else {
        mTasksVec.resize(1);
        mMtxs = std::make_unique<std::vector<std::mutex>>(1);
        mCondVars = std::make_unique<std::vector<std::condition_variable>>(1);
    }

    // Warm up BERTCoreVS and launch threads for processing tasks
    int BERTCoresPerQThread = (serverNumIssueQueryThreads == 0) ? INT_MAX : (gpus.size() * numBERTCores) / serverNumIssueQueryThreads;
    int counter = 0;
    int qThreadIdx = 0;

    mWorkerThreads.reserve(gpus.size());
    for(int idx = 0; idx < gpus.size(); ++idx) {
        auto deviceId = gpus[idx];
        CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
        for(int profileIdx = 0; profileIdx < numBERTCores; ++profileIdx) {
            auto bertCore = tmpBERTCores[idx][profileIdx];
            bertCore->WarmUp();
            CHECK_EQ(mMaxBatchSize <= bertCore->GetMaxBatchSize(), true);
            mWorkerThreads.emplace_back(&BERTServer::ProcessTasks<BERTCoreVS>, this, bertCore, deviceId, qThreadIdx);

            ++counter;
            if (counter == BERTCoresPerQThread) {
                ++qThreadIdx;
                counter = 0;
            }
        }
    }
}

BERTServer::~BERTServer()
{
    {
        std::vector<std::unique_lock<std::mutex>> lcks;
        for (int i = 0; i < mMtxs->size(); ++i) {
            lcks.emplace_back((*mMtxs)[i]);
        }
        mStopGetTasks = true;
        mStopProcessResponse = true;
        for (int i = 0; i < mCondVars->size(); ++i) {
            (*mCondVars)[i].notify_all();
        }
    }
    for(auto& workerThread : mWorkerThreads)
    {
        workerThread.join();
    }
    for(auto& issueQueryThread : mIssueQueryThreads)
    {
        issueQueryThread.join();
    }
}

const std::string& BERTServer::Name() const
{
    return mName;
}

void BERTServer::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    NVTX_MARK("IssueQuery:" + std::to_string(samples.size()) + " tasks", COLOR_BLUE_2);
    auto queryArrivedTime = std::chrono::high_resolution_clock::now();

    // Sort samples in the descending order of sentence length
    std::vector<std::pair<int, int>> sequenceSamplePosAndLength(samples.size());
    for(int samplePos = 0; samplePos < samples.size(); ++samplePos)
    {
        sequenceSamplePosAndLength[samplePos] =
            std::make_pair(samplePos, static_cast<int>(GetSampleLength(samples[samplePos].index)));
    }

    std::sort(sequenceSamplePosAndLength.begin(), sequenceSamplePosAndLength.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool {
                  return a.second > b.second;
              });

    int qThreadIdx = mThreadMap[std::this_thread::get_id()];
    for(int beginSamplePos = 0; beginSamplePos < sequenceSamplePosAndLength.size();
        beginSamplePos += mMaxBatchSize)
    {
        int actualBatchSize = std::min(
            mMaxBatchSize, static_cast<int>(sequenceSamplePosAndLength.size()) - beginSamplePos);
        static int totalBatchSize = 0;
        totalBatchSize += actualBatchSize;
        {
            std::unique_lock<std::mutex> lck((*mMtxs)[qThreadIdx]);
            for(int i = 0; i < actualBatchSize; ++i)
            {
                int samplePosInOriginalRequest =
                    sequenceSamplePosAndLength[beginSamplePos + i].first;
                mTasksVec[qThreadIdx].push_back({samples[samplePosInOriginalRequest], queryArrivedTime});
            }

            // Let some worker thread to consume tasks
            (*mCondVars)[qThreadIdx].notify_one();
        }
    }
}

void BERTServer::FlushQueries() {
    if (mSoftDrop < 1.0) {
        std::vector<std::unique_lock<std::mutex>> lcks;
        for (int i = 0; i < mMtxs->size(); ++i) {
            lcks.emplace_back((*mMtxs)[i]);
        }
        mStopGetTasks = true;
        for (int i = 0; i < mCondVars->size(); ++i) {
            (*mCondVars)[i].notify_all();
        }
    }
}

void BERTServer::ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns)
{
    // Nothing to do for the function
}

std::vector<std::pair<mlperf::QuerySample, std::chrono::high_resolution_clock::time_point>>
    BERTServer::GetTasks(int maxSampleCount, int qThreadIdx)
{
    std::vector<std::pair<mlperf::QuerySample, std::chrono::high_resolution_clock::time_point>> res;
    res.reserve(maxSampleCount);
    // Wait for the new work to arrive
    std::unique_lock<std::mutex> lck((*mMtxs)[qThreadIdx]);
    (*mCondVars)[qThreadIdx].wait(lck, [&] { return (!mTasksVec[qThreadIdx].empty()) || mStopGetTasks; });

    // Consume up to maxSampleCount tasks
    for(int i = 0; (i < maxSampleCount) && !mTasksVec[qThreadIdx].empty(); ++i)
    {
        res.push_back(mTasksVec[qThreadIdx].front());
        mTasksVec[qThreadIdx].pop_front();
    }

    // Let some other thread consume remaining tasks
    if(!mTasksVec[qThreadIdx].empty())
    {
        (*mCondVars)[qThreadIdx].notify_one();
    }

    return res;
}

size_t BERTServer::GetSampleLength(mlperf::QuerySampleIndex idx)
{
    // Get sample length by checking where the input_mask change from 1 to 0
    size_t start{0};
    size_t end{BERT_MAX_SEQ_LENGTH};
    size_t cursor{(start + end) / 2};
    BERTInput input_mask = *static_cast<BERTInput*>(mQsl->GetSampleAddress(idx, 2));
    while(cursor != start)
    {
        if(input_mask[cursor])
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
