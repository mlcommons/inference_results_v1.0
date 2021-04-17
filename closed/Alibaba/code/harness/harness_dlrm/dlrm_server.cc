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

#include "dlrm_server.h"

#include "glog/logging.h"
#include "loadgen.h"

#include <fstream>
#include <set>

#include "dlrm_qsl.hpp"
#include "dlrm_kernels.h"

bool operator==(const nvinfer1::Dims& d1, const nvinfer1 ::Dims& d2)
{
    if(d1.nbDims != d2.nbDims)
        return false;
    for(int it = 0; it < d1.nbDims; it++)
    {
        if(d1.d[it] != d2.d[it])
            return false;
    }
    return true;
}

size_t bindingVolume(std::shared_ptr<nvinfer1::ICudaEngine> engine, int idx)
{
    return lwis::volume(
        engine->getBindingDimensions(idx),
        engine->getBindingFormat(idx),
        engine->hasImplicitBatchDimension());
}

size_t GetEffectiveBatchSize(const std::vector<DLRMTask>& tasks)
{
    return std::accumulate(tasks.begin(), tasks.end(), 0ULL, [](const size_t curr, const DLRMTask& t) { return curr + t.numIndividualPairs; });
}

DLRMCore::DLRMCore(std::shared_ptr<nvinfer1::ICudaEngine> engine, int maxBatchSize, int numBundles,
                   int numCompleteThreads, int profileIdx) :
    mEngine(engine),
    mMaxBatchSize(maxBatchSize),
    mResHandlerPool(numCompleteThreads),
    mBundleCounter(0)
{
    mContext = InferObject(mEngine->createExecutionContext());
    LOG(INFO) << "Setting profile = " << profileIdx;
    mContext->setOptimizationProfile(profileIdx);
    SetBatchSize(maxBatchSize);
    LOG(INFO) << "Context creation complete";

    CHECK_EQ(cudaStreamCreate(&mH2DStream), cudaSuccess);
    CHECK_EQ(cudaStreamCreate(&mComputeStream), cudaSuccess);
    CHECK_EQ(cudaStreamCreate(&mD2HStream), cudaSuccess);
    LOG(INFO) << "Created streams";

    size_t numBindings = mEngine->getNbBindings();
    CHECK_EQ(numBindings / mEngine->getNbOptimizationProfiles(), 3) << "Harness expects 3 bindings per engine profile";
    size_t firstBinding = profileIdx * numBindings / mEngine->getNbOptimizationProfiles();

    mNumInVol = bindingVolume(mEngine, firstBinding + 0);
    LOG(INFO) << "Profile - Numeric Input Volume: " << mNumInVol;
    mCatInVol = bindingVolume(mEngine, firstBinding + 1);
    LOG(INFO) << "Profile - Categorical Input Volume: " << mCatInVol;
    mOutVol = bindingVolume(mEngine, firstBinding + 2);
    LOG(INFO) << "Profile - Output Volume: " << mOutVol;

    mBindings.resize(numBundles);
    for (int i = 0; i < numBundles; ++i)
    {
        auto bundle = std::make_shared<DLRMEventBufferBundle>(i, mNumInVol, mCatInVol, mOutVol, mMaxBatchSize);
        mEventBufferBundle.push_back(bundle);

        // set this profile's bindings to DevicePtrs, set rest to nullptr
        mBindings[i].assign(numBindings, nullptr);
        mBindings[i][firstBinding + 0] = bundle->numericInputBuf.GetDevicePtr();
        mBindings[i][firstBinding + 1] = bundle->categoricalInputBuf.GetDevicePtr();
        mBindings[i][firstBinding + 2] = bundle->outputBuf.GetDevicePtr();
    }

    LOG(INFO) << "Created copy streams and buffers";
    LOG(INFO) << "Setup complete";
}

void DLRMCore::SetBatchSize(int batchSize)
{
    int profileNum = mContext->getOptimizationProfile();
    CHECK_EQ(profileNum >= 0 && profileNum < mEngine->getNbOptimizationProfiles(), true);
    int numBindings = mEngine->getNbBindings() / mEngine->getNbOptimizationProfiles();
    for (int i = 0; i < numBindings; i++)
    {
        if (mEngine->bindingIsInput(i))
        {
            int bindingIdx = numBindings * profileNum + i;
            auto inputDims = mContext->getBindingDimensions(bindingIdx);
            if (inputDims.d[0] != batchSize)
            {
                inputDims.d[0] = batchSize;
                CHECK_EQ(mContext->setBindingDimensions(bindingIdx, inputDims), true);
            }
        }
    }
    CHECK_EQ(mContext->allInputDimensionsSpecified(), true);
}

void DLRMCore::infer(
        std::shared_ptr<DLRMEventBufferBundle> ebBundle,
        size_t batchSize, // We assume that this batchsize is the effective batch size (padded to even value)
        std::vector<DLRMTask>& tasks,
        Batch* batch,
        void(*h2dCallBack)(void*),
        DLRMInferCallback resultCallback,
        DLRMNumericInputType* numericInputPtr,
        DLRMCategoricalInputType* categoricalInputPtr)
{
    DLOG(INFO) << "infer() batch = " << batch;
    NVTX_RANGE_PUSH(("DLRMCore::infer: batchSize="+std::to_string(batchSize)).c_str());
    CHECK_EQ((batchSize % 2 == 0) && (batchSize <= mMaxBatchSize), true);
    SetBatchSize(batchSize);

    bool oddBatch = batch ? batch->isOddBatch() : false; // batch is nullptr in WarmUp run
    // Copy buffers
    ebBundle->numericInputBuf.H2DAsync(numericInputPtr, batchSize, mH2DStream, oddBatch);
    ebBundle->categoricalInputBuf.H2DAsync(categoricalInputPtr, batchSize, mH2DStream, oddBatch);
    ebBundle->recordH2D(mH2DStream);

    // callback upon H2D completion
    CHECK_EQ(cudaLaunchHostFunc(mH2DStream, h2dCallBack, batch), cudaSuccess);

    void **bindings = mBindings[ebBundle->idx].data();

    // Run inference
    ebBundle->makeAwaitH2D(mComputeStream);
    CHECK_EQ(mContext->enqueueV2(bindings, mComputeStream, nullptr), true);
    ebBundle->recordInfer(mComputeStream);

    // Get output
    ebBundle->makeAwaitInfer(mD2HStream);
    ebBundle->outputBuf.D2HAsync(ebBundle->outputBuf.GetHostPtr(), batchSize, mD2HStream);
    ebBundle->recordD2H(mD2HStream);

    DLRMDeferredResult * deferredResult = new DLRMDeferredResult {
        batchSize,
        ebBundle->outputBuf.GetHostPtr(),
        std::move(tasks),
        resultCallback,
        [=] (const DLRMResult& r) { mResHandlerPool.Enqueue(r); }
    };
    CHECK_EQ(cudaLaunchHostFunc(
        mD2HStream,
        [] (void * deferredResult) -> void
        {
            NVTX_RANGE_PUSH("deferredResult processing");
            DLRMDeferredResult * res = reinterpret_cast<DLRMDeferredResult*>(deferredResult);
            DLRMResult r = {
                std::make_shared<std::vector<DLRMOutputType>>(res->outputs, res->outputs + res->batchSize),
                std::move(res->tasks),
                res->callback
            };
            res->resultCallback(r);
            delete res;
            NVTX_RANGE_POP();
        },
        deferredResult), cudaSuccess);

    NVTX_RANGE_POP();
}

void DLRMCore::inferFromDevice(
    std::shared_ptr<DLRMEventBufferBundle> ebBundle,
    size_t batchSize,
    std::vector<DLRMTask>& tasks,
    DLRMInferCallback resultCallback)
{
    NVTX_RANGE_PUSH(("DLRMCore::inferFromDevice: batchSize="+std::to_string(batchSize)).c_str());

    CHECK_EQ((batchSize % 2 == 0) && (batchSize <= mMaxBatchSize), true);
    SetBatchSize(batchSize);

    bool contiguousData = true;
    for(size_t i = 1; (i < tasks.size()) && contiguousData; ++i)
    {
        contiguousData = contiguousData && (ebBundle->numericInputPtrBuf.GetHostPtr()[i] == ebBundle->numericInputPtrBuf.GetHostPtr()[i - 1] + ebBundle->sampleSizesBuf.GetHostPtr()[i - 1] * mNumInVol);
        contiguousData = contiguousData && (ebBundle->categoricalInputPtrBuf.GetHostPtr()[i] == ebBundle->categoricalInputPtrBuf.GetHostPtr()[i - 1] + ebBundle->sampleSizesBuf.GetHostPtr()[i - 1] * mCatInVol);
    }

    if (!contiguousData)
    {
        ebBundle->numericInputPtrBuf.H2DAsync(ebBundle->numericInputPtrBuf.GetHostPtr(), tasks.size(), mH2DStream);
        ebBundle->categoricalInputPtrBuf.H2DAsync(ebBundle->categoricalInputPtrBuf.GetHostPtr(), tasks.size(), mH2DStream);
        ebBundle->sampleSizesBuf.H2DAsync(ebBundle->sampleSizesBuf.GetHostPtr(), tasks.size(), mH2DStream);
        ebBundle->sampleOffsetsBuf.H2DAsync(ebBundle->sampleOffsetsBuf.GetHostPtr(), tasks.size(), mH2DStream);
    }
    ebBundle->recordH2D(mH2DStream);

    // Run inference
    ebBundle->makeAwaitH2D(mComputeStream);

    // Run gather kernel to prepare input data
    if (!contiguousData)
    {
        runGatherKernel(
            (const int8_t **)(ebBundle->numericInputPtrBuf.GetDevicePtr()),
            (const int32_t **)(ebBundle->categoricalInputPtrBuf.GetDevicePtr()),
            (const size_t *)(ebBundle->sampleSizesBuf.GetDevicePtr()),
            (const size_t *)(ebBundle->sampleOffsetsBuf.GetDevicePtr()),
            ebBundle->numericInputBuf.GetDevicePtr(),
            ebBundle->categoricalInputBuf.GetDevicePtr(),
            static_cast<int>(tasks.size()),
            static_cast<int>(mNumInVol),
            static_cast<int>(mCatInVol),
            mComputeStream);
    }

    void **bindings = mBindings[ebBundle->idx].data();

    std::vector<void *> actualBindings;
    if (contiguousData)
    {
        actualBindings.push_back(ebBundle->numericInputPtrBuf.GetHostPtr()[0]);
        actualBindings.push_back(ebBundle->categoricalInputPtrBuf.GetHostPtr()[0]);
        actualBindings.push_back(bindings[2]);
        bindings = actualBindings.data();
    }

    CHECK_EQ(mContext->enqueueV2(bindings, mComputeStream, nullptr), true);
    ebBundle->recordInfer(mComputeStream);

    // Get output
    ebBundle->makeAwaitInfer(mD2HStream);
    ebBundle->outputBuf.D2HAsync(ebBundle->outputBuf.GetHostPtr(), batchSize, mD2HStream);
    ebBundle->recordD2H(mD2HStream);

    DLRMDeferredResult * deferredResult = new DLRMDeferredResult {
        batchSize,
        ebBundle->outputBuf.GetHostPtr(),
        std::move(tasks),
        resultCallback,
        [=] (const DLRMResult& r) { mResHandlerPool.Enqueue(r); }
    };
    CHECK_EQ(cudaLaunchHostFunc(
        mD2HStream,
        [] (void * deferredResult) -> void
        {
            NVTX_RANGE_PUSH("deferredResult processing");
            DLRMDeferredResult * res = reinterpret_cast<DLRMDeferredResult*>(deferredResult);
            DLRMResult r = {
                std::make_shared<std::vector<DLRMOutputType>>(res->outputs, res->outputs + res->batchSize),
                std::move(res->tasks),
                res->callback
            };
            res->resultCallback(r);
            delete res;
            NVTX_RANGE_POP();
        },
        deferredResult), cudaSuccess);

    NVTX_RANGE_POP();
}

void DLRMCore::WarmUp(double duration)
{
    double elapsed = 0.0;
    auto tStart = std::chrono::high_resolution_clock::now();

    std::vector<DLRMTask> dummyTasks(mMaxBatchSize, { {0, 0}, 1 });
    std::vector<DLRMNumericInputType> dummyNumIn(mMaxBatchSize * mNumInVol);
    std::vector<DLRMCategoricalInputType> dummyCatIn(mMaxBatchSize * mCatInVol);

    LOG(INFO) << "Running warmup for " << duration << "s.";
    do {
        auto bundle = NextForegroundBundle();
        bundle->syncD2H();
        infer(
                bundle,
                mMaxBatchSize,
                dummyTasks,
                nullptr,
                [](void* batch) -> void { return; },
                [](std::vector<mlperf::QuerySampleResponse>&) { return; },
                dummyNumIn.data(),
                dummyCatIn.data());
        elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - tStart).count();
    } while (elapsed < duration);
    for(size_t i = 0; i < mEventBufferBundle.size(); ++i)
        NextForegroundBundle()->syncD2H();
    LOG(INFO) << "Warmup complete, ran for " << elapsed << "s.";
}

std::shared_ptr<DLRMEventBufferBundle> DLRMCore::NextForegroundBundle()
{
    size_t idx = mBundleCounter;
    mBundleCounter = (mBundleCounter + 1) % mEventBufferBundle.size();
    return mEventBufferBundle[idx];
}

DLRMCore::~DLRMCore()
{
    CHECK_EQ(cudaStreamDestroy(mH2DStream), cudaSuccess);
    CHECK_EQ(cudaStreamDestroy(mComputeStream), cudaSuccess);
    CHECK_EQ(cudaStreamDestroy(mD2HStream), cudaSuccess);
}

DLRMServer::DLRMServer(const std::string name,
                       const std::string enginePath,
                       std::shared_ptr<DLRMSampleLibrary> qsl,
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
                       NumaConfig numaConfig)
    : mName{name},
      mQsl{qsl},
      mStartFromDevice(startFromDevice),
      mStopWork{false},
      mDLRMCores{gpus.size()*numDLRMCores},
      mNumInVol(0),
      mCatInVol(0),
      mNumaConfig(numaConfig),
      mGpuToNumaMap(getGpuToNumaMap(mNumaConfig))
{
    NVTX_NAME_THIS_THREAD("DLRMServer");
    LOG(INFO) << "Using " << numDLRMCores << " DLRM Core(s)";

    if (UseNuma())
    {
        LOG(INFO) << "Using NUMA nodes";
        CHECK(mNumaConfig.size() == gpus.size()) << "Currently only support one GPU per NUMA node!";
    }

    mMaxBatchSize = maxBatchSize;
    // Enforce that max batch size is even due to Top MLP plugin
    if (mMaxBatchSize % 2 == 1) { mMaxBatchSize = mMaxBatchSize - 1; }

    std::vector<std::thread> setupThreads;
    for(const auto& deviceId : gpus)
    {
        setupThreads.emplace_back(&DLRMServer::SetupDevice, this, 
                                  enginePath, numBundles, numCompleteThreads, numDLRMCores, warmupDuration, deviceId);
    }
    for (auto& t : setupThreads) { t.join(); }

    if (!startFromDevice)
    {
        int numBatchMakers = UseNuma() ? mNumaConfig.size() : 1;
        for (int i = 0; i < numBatchMakers; ++i)
        {
            // Allocate the memory (including staging buffers) for the BatchMaker on specific NUMA node.
            if (UseNuma())
            {
                bindNumaMemPolicy(i, mNumaConfig.size());
            }

            // Construct BatchMaker
            mBatchMakers.emplace_back(std::make_shared<BatchMaker>(
                /* numStagingThreads = */ numStagingThreads,
                /* numBatches = */ numStagingBatches,
                /* maxBatchSize = */ maxBatchSize,
                /* maxPairsPerThread = */ maxPairsPerThread,
                /* numericVolume = */ mNumInVol,
                /* categoricalVolume = */ mCatInVol,
                /* checkContiguity = */ checkContiguity,
                /* qsl = */ qsl,
                /* cpus = */ UseNuma() ? mNumaConfig[i].second : std::vector<int>()));

            // Reset memory allocation setting
            if (UseNuma())
            {
                resetNumaMemPolicy();
            }
        }
    }

    for(const auto& deviceId : gpus)
    {
        for (size_t profileIdx = 0; profileIdx < numDLRMCores; ++profileIdx)
        {
            auto dlrmCore = mDLRMCores[deviceId*numDLRMCores + profileIdx];
            mWorkerThreads.emplace_back(startFromDevice ? &DLRMServer::ProcessTasksFromDevice : &DLRMServer::ProcessTasks, this, dlrmCore, deviceId, profileIdx);
            // Limit the worker thread to the closest CPUs.
            if (UseNuma())
            {
                bindThreadToCpus(mWorkerThreads.back(), mNumaConfig[mGpuToNumaMap[deviceId]].second);
            }
        }
    }
}

void DLRMServer::SetupDevice(const std::string enginePath, int numBundles, int numCompleteThreads, int numDLRMCores, int warmupDuration, int deviceId)
{
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
    auto engine = DeserializeEngine(enginePath);
    CHECK_LE(numDLRMCores, engine->getNbOptimizationProfiles());

    mNumInVol = bindingVolume(engine, 0);
    mCatInVol = bindingVolume(engine, 1);

    for (size_t profileIdx = 0; profileIdx < numDLRMCores; ++profileIdx)
    {
        auto dlrmCore = std::make_shared<DLRMCore>(engine, mMaxBatchSize, numBundles, numCompleteThreads, profileIdx);
        mDLRMCores[deviceId*numDLRMCores + profileIdx] = dlrmCore;
        CHECK_LE(mMaxBatchSize, dlrmCore->GetMaxBatchSize());
        dlrmCore->WarmUp(warmupDuration);
    }
};

DLRMServer::~DLRMServer()
{
    DLOG(INFO) << "~DLRMServer";
    {
        std::unique_lock<std::mutex> lck(mMtx);
        mStopWork = true;
        mCondVar.notify_all();
    }
    for (auto batchMaker : mBatchMakers)
    {
        if (batchMaker)
        {
            batchMaker->StopWork();
        }
    }

    for (auto& workerThread : mWorkerThreads)
    {
        workerThread.join();
    }
}

const std::string& DLRMServer::Name() const
{
    return mName;
}

void DLRMServer::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    NVTX_RANGE_PUSH(("DLRMServer::IssueQuery for " + std::to_string(samples.size()) + " samples").c_str());
    if (mStartFromDevice)
    {
        std::vector<size_t> numPairs(samples.size());
        std::transform(samples.begin(), samples.end(), numPairs.begin(), [&] (const mlperf::QuerySample& x) { return mQsl->GetNumUserItemPairs(x.index); } );

        std::unique_lock<std::mutex> lck(mMtx);

        for(size_t i = 0; i < samples.size(); ++i)
            mTasks.push_back({samples[i], numPairs[i]});

        mCondVar.notify_one();
    }
    else
    {
        int nextBatchMakerIdx = UseNuma() ? ((mPrevBatchMakerIdx + 1) % mNumaConfig.size()) : 0;
        mBatchMakers[nextBatchMakerIdx]->IssueQuery(samples);
        mPrevBatchMakerIdx = nextBatchMakerIdx;
    }
    NVTX_RANGE_POP();
}

void DLRMServer::FlushQueries()
{
    NVTX_RANGE_PUSH("DLRMServer::FlushQueries");
    if (!mStartFromDevice)
    {
        for (auto batchMaker : mBatchMakers)
        {
            batchMaker->FlushQueries();
        }
    }
    NVTX_RANGE_POP();
}

void DLRMServer::ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns)
{
    // Nothing to do for the function
}

void DLRMServer::ProcessTasks(std::shared_ptr<DLRMCore> dlrmCore, int deviceId, int profileIdx)
{
    NVTX_NAME_THIS_THREAD(("ProcessTasks"+std::to_string(profileIdx)).c_str());
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);

    // Process samples in batches
    while(true)
    {
        auto ebBundle = dlrmCore->NextForegroundBundle();

        // Only grab tasks when the copy stream is ready. This allows `tasks` to fill up with more batches while the
        // stream is still working. The GPU should still be busy with inference on other bundle(s)
        NVTX_RANGE_PUSH("DLRMServer::ProcessTasks syncing for the foreground bundle to complete");
        ebBundle->syncD2H();
        NVTX_RANGE_POP();
        NVTX_RANGE_PUSH(("DLRMServer::ProcessTasks iteration, profile" + std::to_string(profileIdx)).c_str());

        int batchMakerIdx = UseNuma() ? mGpuToNumaMap[deviceId] : 0;
        Batch* batch = mBatchMakers[batchMakerIdx]->GetBatch();

        if (!batch) { NVTX_RANGE_POP(); break; }

        size_t actualBatchSize = batch->getCommittedCopies();
        auto tasks = batch->getTasks();
        auto numericHostPtr = batch->getNumericHostPtr();
        auto categoricalHostPtr = batch->getCategoricalHostPtr();
        bool oddBatch = batch->isOddBatch();
        DLOG(INFO) << "Batch Size : " << actualBatchSize;

        dlrmCore->infer(
            ebBundle,
            actualBatchSize,
            tasks,
            batch,
            [] (void* batch) -> void { reinterpret_cast<Batch*>(batch)->mBatchMaker->NotifyH2D(reinterpret_cast<Batch*>(batch)); },
            [=] (std::vector<mlperf::QuerySampleResponse>& responses) {
                if (oddBatch) { responses.pop_back(); }
                mlperf::QuerySamplesComplete(responses.data(), responses.size());
            },
            numericHostPtr,
            categoricalHostPtr
        );

        NVTX_RANGE_POP();
    }
}

void DLRMServer::ProcessTasksFromDevice(std::shared_ptr<DLRMCore> dlrmCore, int deviceId, int profileIdx)
{
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);

    // Process samples in batches
    while(true)
    {
        auto ebBundle = dlrmCore->NextForegroundBundle();

        // Only grab tasks when the copy stream is ready. This allows `tasks` to fill up with more batches while the
        // stream is still working. The GPU should still be busy with inference on other bundle(s)
        NVTX_RANGE_PUSH("DLRMServer::ProcessTasksFromDevice syncing for the foreground bundle to complete");
        ebBundle->syncD2H();
        NVTX_RANGE_POP();
        NVTX_RANGE_PUSH(("DLRMServer::ProcessTasksFromDevice iteration, profile" + std::to_string(profileIdx)).c_str());

        auto tasks = GetBatch();
        if (tasks.empty()) { NVTX_RANGE_POP(); break; }

        NVTX_RANGE_PUSH("DLRMServer::ProcessTasksFromDevice preparing batch host buffers");
        size_t originalBatchSize = std::accumulate(tasks.begin(), tasks.end(), (size_t)0, [] (size_t x, const DLRMTask& y) { return x + y.numIndividualPairs; } );
        // Pas the tasks so that the batch size is even
        bool isBatchPadded = false;
        if ((originalBatchSize % 2) != 0)
        {
            tasks.push_back({tasks.back().querySample, 1});
            isBatchPadded = true;
        }

        NVTX_RANGE_PUSH("DLRMServer::ProcessTasksFromDevice preparing numericInputPtrBuf");
        std::transform(tasks.begin(), tasks.end(), ebBundle->numericInputPtrBuf.GetHostPtr(),
            [&] (const DLRMTask& x) { return reinterpret_cast<DLRMNumericInputType *>(mQsl->GetSampleAddress(x.querySample.index, 0, 0, deviceId)); } );
        NVTX_RANGE_POP();
        NVTX_RANGE_PUSH("DLRMServer::ProcessTasksFromDevice preparing categoricalInputPtrBuf");
        std::transform(tasks.begin(), tasks.end(), ebBundle->categoricalInputPtrBuf.GetHostPtr(),
            [&] (const DLRMTask& x) { return reinterpret_cast<DLRMCategoricalInputType *>(mQsl->GetSampleAddress(x.querySample.index, 1, 0, deviceId)); } );
        NVTX_RANGE_POP();
        NVTX_RANGE_PUSH("DLRMServer::ProcessTasksFromDevice preparing sampleSizesBuf");
        std::transform(tasks.begin(), tasks.end(), ebBundle->sampleSizesBuf.GetHostPtr(),
            [&] (const DLRMTask& x) { return x.numIndividualPairs; } );
        NVTX_RANGE_POP();
        NVTX_RANGE_PUSH("DLRMServer::ProcessTasksFromDevice preparing sampleOffsetsBuf");
        ebBundle->sampleOffsetsBuf.GetHostPtr()[0] = 0;
        std::partial_sum(ebBundle->sampleSizesBuf.GetHostPtr(), ebBundle->sampleSizesBuf.GetHostPtr() + tasks.size() - 1, ebBundle->sampleOffsetsBuf.GetHostPtr() + 1);
        NVTX_RANGE_POP();

        size_t batchSize = std::accumulate(ebBundle->sampleSizesBuf.GetHostPtr(), ebBundle->sampleSizesBuf.GetHostPtr() + tasks.size(), (size_t)0);
        NVTX_RANGE_POP();

        dlrmCore->inferFromDevice(
            ebBundle,
            batchSize,
            tasks,
            [=] (std::vector<mlperf::QuerySampleResponse>& responses) {
                if (isBatchPadded) { responses.pop_back(); }
                mlperf::QuerySamplesComplete(responses.data(), responses.size());
            }
        );

        NVTX_RANGE_POP();
    }
}

std::shared_ptr<nvinfer1::ICudaEngine> DLRMServer::DeserializeEngine(std::string enginePath)
{
    int whichDevice;
    CHECK_EQ(cudaGetDevice(&whichDevice), cudaSuccess);
    LOG(INFO) << "Deserializing Engine on GPU#" << whichDevice;

    auto runtime = InferObject(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));
    std::vector<char> trtModelStream;
    auto size = lwis::GetModelStream(trtModelStream, enginePath);
    auto engine = InferObject(runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr));

    LOG(INFO) << "Engine - Device Memory requirements: " << engine->getDeviceMemorySize();
    LOG(INFO) << "Engine - Number of Optimization Profiles: "
              << engine->getNbOptimizationProfiles();
    return engine;
}

std::vector<DLRMTask> DLRMServer::GetBatch()
{
    NVTX_RANGE_PUSH("DLRMServer::GetBatch");
    std::vector<DLRMTask> res;
    // Wait for the new work to arrive
    std::unique_lock<std::mutex> lck(mMtx);
    mCondVar.wait(lck, [&] {return (!mTasks.empty()) || mStopWork;} );

    NVTX_RANGE_PUSH(("Extracting tasks from queue with length " + std::to_string(mTasks.size())).c_str());
    // Consume up to mMaxBatchSize pairs
    int currentBatchSize = 0;
    while(!mTasks.empty())
    {
        const auto& topTask = mTasks.front();
        currentBatchSize += topTask.numIndividualPairs;
        if (currentBatchSize > mMaxBatchSize)
            break;
        res.push_back(topTask);
        mTasks.pop_front();
    }

    // Let some other thread to consume more tasks if this one got any
    if (!res.empty())
        mCondVar.notify_one();

    NVTX_RANGE_POP();
    NVTX_RANGE_POP();

    return res;
}