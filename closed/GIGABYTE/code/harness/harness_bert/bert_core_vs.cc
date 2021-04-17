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

#include "bert_core_vs.h"

#include "bert_server.h"
#include "glog/logging.h"
#include "loadgen.h"

#include <fstream>
#include <unordered_set>
#include <set>

#undef CUDA_GRAPH_STATS

constexpr int BIDX = 0;
constexpr int SIDX = 1;
constexpr int BERT_CUDA_GRAPH_SIZE = 43; // in MiB ranges from ~[38,45]

void BERTCoreVS::SetInputShapes(std::shared_ptr<nvinfer1::IExecutionContext> context, int sumS, int B, int maxS) {
    
    auto& engine = context->getEngine();
    int profileNum = context->getOptimizationProfile();
    CHECK_EQ(profileNum >= 0 && profileNum < engine.getNbOptimizationProfiles(), true);
    int numBindings = engine.getNbBindings() / engine.getNbOptimizationProfiles();

    int bindingOffset = numBindings * profileNum;
    int idxInputIds = engine.getBindingIndex("input_ids");
    int idxSegmentIds = engine.getBindingIndex("segment_ids");
    int idxCuSeqlens = engine.getBindingIndex("cu_seqlens");
    int idxMaxSeqlen = engine.getBindingIndex("max_seqlen");
    CHECK_EQ(idxInputIds , 0);
    CHECK_EQ(idxSegmentIds , 1);
    CHECK_EQ(idxCuSeqlens , 2);
    CHECK_EQ(idxMaxSeqlen , 3);
    
    if(context->getBindingDimensions(bindingOffset + idxInputIds).d[0] != sumS){
        CHECK_EQ(context->setBindingDimensions(bindingOffset + idxInputIds, {1, sumS}), true);
    }
    
    if(context->getBindingDimensions(bindingOffset + idxSegmentIds).d[0] != sumS){
        CHECK_EQ(context->setBindingDimensions(bindingOffset + idxSegmentIds, {1, sumS}), true);
    }
    
    if(context->getBindingDimensions(bindingOffset + idxCuSeqlens).d[0] != B+1){
        CHECK_EQ(context->setBindingDimensions(bindingOffset + idxCuSeqlens, {1, B+1}), true);
    }
    
    if(context->getBindingDimensions(bindingOffset + idxMaxSeqlen).d[0] != maxS){
        CHECK_EQ(context->setBindingDimensions(bindingOffset + idxMaxSeqlen, {1, maxS}), true);
    }

    CHECK_EQ(context->allInputDimensionsSpecified(), true);
}

BERTGraphSpec_t BERTCoreVS::GetClosestGraphSpec(int maxSeqLen, int batchSize, int totalSeqLen) {
    CHECK_EQ(mUseGraphs, true);
    CHECK_EQ(!mGraphSpecs.empty(), true);

    // mGraphSpecs needs to have values in ascending order.
    for (const auto& spec : mGraphSpecs) {
        if (maxSeqLen <= std::get<0>(spec) && batchSize <= std::get<1>(spec) && totalSeqLen <= std::get<2>(spec)) {
            return spec;
        }
    }

    // return -1's if no graph spec found
    return std::make_tuple(-1, -1, -1);
}

size_t BERTCoreVS::GetTotalGPUMemoryInMiB() {
    // return total memory in MiB
    cudaDeviceProp properties;
    CHECK_EQ(cudaGetDeviceProperties(&properties, mDeviceId), CUDA_SUCCESS);
    // totalGlobalMem is in Bytes
    return properties.totalGlobalMem / (1024*1024);

}

std::vector<BERTGraphSpec_t> BERTCoreVS::ParseGraphSpecs(const std::string& graphSpecs, int graphSeqLenUpperBound) {
    // Parse a list of (maxSeqLen, min totSeqLen, max totSeqLen, step size), simply ignore parenthesis and spaces
    std::vector<BERTGraphSpec_t> retGraphSpecs;
    BERTGraphSpec_t newGraphSpec;
    size_t prev = 0;
    int count = 0;

    std::unordered_set<int> maxSeqLens;
    std::array<int, 4> arr;
    for(size_t pos = 0; pos < graphSpecs.size(); ++pos) {
        if (!std::isdigit(graphSpecs[pos])) {
            if (std::isdigit(graphSpecs[prev])) {
                arr[count] = std::stoi(graphSpecs.substr(prev, pos-prev));
                ++count;

                // do processing if we get all 4 numbers
                if (count == 4) {
                    const auto& [maxSeqLen, minTotSeqLen, maxTotSeqLen, stepSize] = arr;
                    maxSeqLens.insert(maxSeqLen);
                    for (auto curSeqLen = minTotSeqLen; curSeqLen <= maxTotSeqLen; curSeqLen += stepSize) {
                        if (curSeqLen > mMaxBatchSize * BERT_MAX_SEQ_LENGTH) {
                            break;
                        }

                        // one graph spec is (max seqLen, batch size, total seqLen)
                        // the batch size will remain at 1 when curSeqLen <= maxSeqLen
                        size_t batchSize = 1;
                        if (curSeqLen > maxSeqLen) {
                            // adjust batch size according to upper bound and max batch size
                            // multiply by 2 since the batch size needs to be increased faster than average
                            batchSize += (curSeqLen / std::min(graphSeqLenUpperBound, maxSeqLen)) * 2;
                            batchSize = std::min(batchSize, mMaxBatchSize);
                        }
                        retGraphSpecs.emplace_back(maxSeqLen, batchSize, curSeqLen);
                    }
                    count = 0;
                }
            }
            prev = pos;
        } else if (!std::isdigit(graphSpecs[prev])) {
            prev = pos;
        }
    }
    return retGraphSpecs;
}

void BERTCoreVS::InitializeGraphSpecs(int graphMaxSeqLen, const std::string& graphSpecs, int numBERTCores) {
    // initialize graph specs, which are a list of (max seqlen, batch size, total seqlen)
    std::vector<BERTGraphSpec_t> tmpGraphSpecs;

    if (!graphSpecs.empty()) {
        LOG(INFO) << "CUDA graph specs in a form of list of (maxSeqLen, min totSeqLen, max totSeqLen, step size):";
        LOG(INFO) << graphSpecs;
        tmpGraphSpecs = ParseGraphSpecs(graphSpecs, graphMaxSeqLen);
    } else {
        // add graph specs for batch size = 1 if max batch size <= 8
        if (mMaxBatchSize <= 8) {
            int step = 4;
            for (int i = step; i < BERT_MAX_SEQ_LENGTH; i += step) {
                tmpGraphSpecs.emplace_back(i, 1, i);
            }
            tmpGraphSpecs.emplace_back(BERT_MAX_SEQ_LENGTH, 1, BERT_MAX_SEQ_LENGTH);
        }

        if (mMaxBatchSize != 1) {
            int maxSeqLen = 1;
            for (const auto& maxDimsAndContext : mMaxDimsToContext) {
                maxSeqLen = std::max(maxSeqLen, maxDimsAndContext.first.second);
            }

            size_t totalMemoryInMiB = GetTotalGPUMemoryInMiB();
            size_t maxNumGraphs = 400 / numBERTCores;
            size_t numGraphs = std::min(maxNumGraphs, totalMemoryInMiB / BERT_CUDA_GRAPH_SIZE / numBERTCores);

            CHECK_EQ(numGraphs >= 3, true);
            if (graphMaxSeqLen == BERT_MAX_SEQ_LENGTH) {
                // all batches have size of max batch size
                size_t start = mMaxBatchSize;
                size_t end = BERT_MAX_SEQ_LENGTH * mMaxBatchSize;
                size_t step = (end - start) / (numGraphs - 2);
                for (size_t totSeqLen = start; totSeqLen < end; totSeqLen += step) {
                    tmpGraphSpecs.emplace_back(maxSeqLen, mMaxBatchSize, totSeqLen);
                }
                tmpGraphSpecs.emplace_back(maxSeqLen, mMaxBatchSize, end);
            } else {
                // Using a linear relation and truncated at max batch size
                size_t maxTotalSeqLen = mMaxBatchSize * graphMaxSeqLen;
                size_t maxStep = 100;
                size_t step = std::min(maxStep, maxTotalSeqLen / numGraphs);
                size_t totSeqLen = 2 * step;
                for (size_t i = 2; i <= mMaxBatchSize; ++i) {
                    // tmpGraphSpecs.emplace_back(maxSeqLen, i, totSeqLen);
                    tmpGraphSpecs.emplace_back(maxSeqLen, mMaxBatchSize, totSeqLen);
                    totSeqLen += step;
                }
                for ( ; totSeqLen <= maxTotalSeqLen; totSeqLen += step) {
                    tmpGraphSpecs.emplace_back(maxSeqLen, mMaxBatchSize, totSeqLen);
                }
            }
        }
    }

    // mGraphSpecs needs to be sorted
    std::sort(std::begin(tmpGraphSpecs), std::end(tmpGraphSpecs));
    tmpGraphSpecs.erase(std::unique(tmpGraphSpecs.begin(), tmpGraphSpecs.end()), tmpGraphSpecs.end());
    mGraphSpecs = std::move(tmpGraphSpecs);

#ifdef CUDA_GRAPH_STATS
    for(auto& spec : mGraphSpecs) {
        LOG(INFO) << std::get<0>(spec) << ", " << std::get<1>(spec) << ", " << std::get<2>(spec);
    }
#endif
}

void BERTCoreVS::BuildGraphs(int graphMaxSeqLen, const std::string& graphSpecs, int numBERTCores)
{
    CHECK_EQ(cudaSetDevice(mDeviceId), cudaSuccess);
    mGraphSpecs.clear();
    InitializeGraphSpecs(graphMaxSeqLen, graphSpecs, numBERTCores);
    CHECK_EQ(!mGraphSpecs.empty(), true);

    // set all input device buffers to zero to avoid error when calling first dummy enqueueV2 later
    // use mCounter because GetBuffer APIs depend on mCounter
    int numCopyStreams = mCopyStreams.size();
    for (mCounter = 0; mCounter < numCopyStreams; ++mCounter) {
        GetBufferInputIds().memsetD(0);
        GetBufferSegmentIds().memsetD(0);
        GetBufferInputMask().memsetD(0);
    }
    mCounter = 0;

    int numGraphs = 0;
    for (int sidx = 0; sidx < numCopyStreams; ++sidx) {
        // get bindings
        void** bindings = mBindings[sidx].data();
        for (const auto& spec : mGraphSpecs) {
            // Now we can do per-spec initialization of cuda graphs
            const auto [maxSeqLen, batchSize, totalSeqLen] = spec;
            // set batch size
            auto context = GetContext(batchSize, GetClosestSeqLen(maxSeqLen));
            SetInputShapes(context, totalSeqLen, batchSize, maxSeqLen);
            // Do a dry run. NOTE, this breaks certain CASK tactics which require onShapeChange to be captured.
            // These tactics should be disabled in the builder through IAlgorithmSelector.
            CHECK_EQ(context->enqueueV2(bindings, mStream, nullptr), true);
            CHECK_EQ(cudaStreamSynchronize(mStream), CUDA_SUCCESS);
            // capture graph
            cudaGraph_t graph;
            CHECK_EQ(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal), CUDA_SUCCESS);
            CHECK_EQ(context->enqueueV2(bindings, mStream, nullptr), true);
            CHECK_EQ(cudaStreamEndCapture(mStream, &graph), CUDA_SUCCESS);

            // create graphExec
            cudaGraphExec_t graphExec;
            CHECK_EQ(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), CUDA_SUCCESS);

            // store graphExec
            BERTGraphKey_t key = std::make_pair(sidx, spec);
            mCudaGraphExecs[key] = graphExec;
            ++numGraphs;

            CHECK_EQ(cudaGraphDestroy(graph), CUDA_SUCCESS);
        }
    }
    LOG(INFO) << "Created " << numGraphs << " CUDA graphs";
}

BERTCoreVS::BERTCoreVS(const std::vector<std::shared_ptr<nvinfer1::ICudaEngine>>& engines, int numCopyStreams, int numProfiles, int profileIdx,
    bool useGraphs, int deviceId):
  mCopyStreams(numCopyStreams), mCounter(0), mBindings(numCopyStreams), mUseGraphs(useGraphs), mDeviceId(deviceId)
{
    for(int it=0;it < NUM_RESPONSE_THREADS;it++){
        mResponseThreads.emplace_back(BERTCoreVS::ProcessResponse, this);
    }

    // get the maximum device memory requirement
    auto maxDevMemSize =
        (*std::max_element(engines.begin(), engines.end(), [](auto ePtr1, auto ePtr2) {
            return (ePtr1->getDeviceMemorySize() < ePtr2->getDeviceMemorySize());
        }))->getDeviceMemorySize();

    mContextBuf = std::make_shared<lwis::DeviceBuffer>(maxDevMemSize);
    LOG(INFO) << "Engine - Device Memory requirements: " << maxDevMemSize;
    
    int maxBS = 0;
    int maxNumBindings = 0;

    // create contexts for all profiles for all engines
    for(auto engine : engines)
    {
        LOG(INFO) << "Engine - Number of Optimization Profiles: "
                  << engine->getNbOptimizationProfiles();

        maxNumBindings = std::max(maxNumBindings, engine->getNbBindings());
        int bindingsPerProfile = engine->getNbBindings() / engine->getNbOptimizationProfiles();
        CHECK_EQ(bindingsPerProfile, 5);

        CHECK_EQ(engine->getNbOptimizationProfiles() % numProfiles == 0, true);
        int profilesPerCore = engine->getNbOptimizationProfiles() / numProfiles;
        int startIt = profileIdx * profilesPerCore;
        int endIt = (profileIdx + 1) * profilesPerCore;

        for(int it = startIt, bidx = startIt * bindingsPerProfile; it < endIt;
            it++, bidx += bindingsPerProfile)
        {
            auto maxDims =
                engine->getProfileDimensions(bidx, it, nvinfer1::OptProfileSelector::kMAX);
            auto maxDimsSIds =
                engine->getProfileDimensions(bidx + 1, it, nvinfer1::OptProfileSelector::kMAX);
            auto maxDimsCuSeqlens =
                engine->getProfileDimensions(bidx + 2, it, nvinfer1::OptProfileSelector::kMAX);
            const int B_ = maxDimsCuSeqlens.d[0] - 1;
            const int S_ = maxDims.d[0] / B_;
            assert(B_ * S_ == maxDims.d[0] && "iids: must be multiple of max batch size");
            assert(maxDimsSIds.d[0] == maxDims.d[0] && "sids: must have the same max length as iids");

            assert(maxDims.nbDims == 1 && "expecting one packed dimension");
            LOG(INFO) << "Engine - Profile " << it << " maxDims " << maxDims.d[0] << " Bmax=" << B_
                      << " Smax=" << S_;
            maxBS = std::max(maxBS, B_);
            auto tmpContext = InferObject(engine->createExecutionContextWithoutDeviceMemory());
            LOG(INFO) << "Setting Opt.Prof. to " << it;
            tmpContext->setOptimizationProfile(it);

            tmpContext->setDeviceMemory(mContextBuf->data());
            SetInputShapes(tmpContext, maxDims.d[0], B_, S_);

            CHECK_EQ(maxDims.nbDims == 1, true);
            mMaxDimsToContext.insert({{B_, S_}, tmpContext});
            CHECK_EQ(tmpContext->getOptimizationProfile(), it);
        }
    }

    CHECK_EQ(maxBS > 0, true);
    LOG(INFO) << "Context creation complete. Max supported batchSize: " << maxBS;
    mMaxBatchSize = maxBS;

    CHECK_EQ(cudaStreamCreate(&mStream), cudaSuccess);

    // Allocate buffers
    const size_t bufferSize = BERT_MAX_SEQ_LENGTH * mMaxBatchSize;

    CHECK_EQ(maxNumBindings % 5 , 0);
    for(int it = 0; it < mCopyStreams.size(); it++)
    {
        mInputIdBufs.emplace_back(bufferSize);
        mSegmentIdBufs.emplace_back(bufferSize);
        //TODO need to rename this
        mInputMaskBufs.emplace_back(maxBS+1);
        mDummy.emplace_back(BERT_MAX_SEQ_LENGTH);
        mOutputBufs.emplace_back(BERT_MAX_SEQ_LENGTH * mMaxBatchSize * 2);
        
        mInputIdBufs.back().memsetD(0);
        mSegmentIdBufs.back().memsetD(0);
        for(int jt=0;jt < maxNumBindings;jt+=5){
            mBindings[it].push_back(mInputIdBufs.back().DeviceData());
            mBindings[it].push_back(mSegmentIdBufs.back().DeviceData());
            mBindings[it].push_back(mInputMaskBufs.back().DeviceData());
            mBindings[it].push_back(mDummy.back().DeviceData());
            mBindings[it].push_back(mOutputBufs.back().DeviceData());
        }
        mCopyStreamIdxQueue.push_back(it);
    }
    
    LOG(INFO) << "Setup complete";
}

std::shared_ptr<nvinfer1::IExecutionContext> BERTCoreVS::GetContext(int batchSize, int seqLen)
{
    auto iter = mMaxDimsToContext.find({batchSize, seqLen});
    if(iter != mMaxDimsToContext.end())
    {
        return iter->second;
    }
    //we assume that seqLen was determined by GetClosestSeqLen and therefore corresponds to actual seqLens in the map
    //so we need to find the smallest batch size >= batchSize with seqLen
    int effectiveBatchSize = INT_MAX;
    for(auto kv : mMaxDimsToContext){
        if(kv.first.second != seqLen) continue;
        if(kv.first.first < batchSize) continue;
        effectiveBatchSize = std::min(effectiveBatchSize, kv.first.first);
    }
    CHECK_NE(effectiveBatchSize, INT_MAX);
    iter = mMaxDimsToContext.find({effectiveBatchSize, seqLen});
    CHECK_EQ(iter!= mMaxDimsToContext.end(), true);
    // LOG(INFO) << "Context maxdims: " << iter->first.first << ", " << iter->first.second;
    return  iter->second;
}

int  BERTCoreVS::GetClosestSeqLen(int seqLen) 
{
    // find the first context whose max batch size is larger or equal to the given one
    auto current = mMaxDimsToContext.begin();
    while(current != mMaxDimsToContext.end() && seqLen > current->first.second)
    {
        current++;
    }
    CHECK_EQ(current == mMaxDimsToContext.end(), false);
    return current->first.second;
}

void BERTCoreVS::ProcessResponse(BERTCoreVS* BERTCoreVS)
{
    size_t totSamples = 0;
    while(true)
    {
        std::unique_lock<std::mutex> lck(BERTCoreVS->mMtx);
        BERTCoreVS->mCondVar.wait(
            lck, [&]() { return !BERTCoreVS->mResultQ.empty() || BERTCoreVS->mStopWork; });
        if(BERTCoreVS->mStopWork)
            break;
        auto& resp = BERTCoreVS->mResultQ.front();

        CHECK_EQ(cudaEventSynchronize(resp.resultReady), cudaSuccess);

        for(auto& qsr : resp.QSRs)
        {
            mlperf::QuerySamplesComplete(&qsr, 1);
        }
        totSamples += resp.QSRs.size();

        BERTCoreVS->mCopyStreamIdxQueue.push_back(resp.copyStreamIdx);
        BERTCoreVS->mResultQ.pop_front();
        BERTCoreVS->mCondVar.notify_one();
    }

    VLOG(1) << "QuerySamplesCompelete " << totSamples << " samples.";
    using CLK = std::chrono::high_resolution_clock;
    VLOG(1) << "End of ProcessResponse: " << std::chrono::duration_cast<std::chrono::microseconds>(CLK::now().time_since_epoch()).count();
}

int BERTCoreVS::CountTotalLength(const BERTTask_t& tasks, std::shared_ptr<qsl::SampleLibrary> qsl) {
    const int actualBatchSize = tasks.size();
    int totalLength = 0;
    for(int i = 0; i < actualBatchSize; ++i)
    {
        BERTInput* mask = static_cast<BERTInput*>(qsl->GetSampleAddress(tasks[i].first.index, 2));
        int Si = std::accumulate(mask->begin(), mask->end(), 0);
        totalLength += Si;
    }
    return totalLength;
}

void BERTCoreVS::infer(const BERTTask_t& tasks, std::shared_ptr<qsl::SampleLibrary> qsl)
{
    // get free copy stream idx from sync queue
    mCounter = mCopyStreamIdxQueue.front_then_pop();

    const int actualBatchSize = tasks.size();
    static int totalBatchSize = 0;
    totalBatchSize += actualBatchSize;
    // iterate through the batch and find largest seqLen
    int maxSeqLen = 0;
    // accumulate the sequence lengths
    std::vector<int> cuSeqlens(actualBatchSize + 1, 0);
    for(int i = 0; i < actualBatchSize; ++i)
    {
        BERTInput* mask = static_cast<BERTInput*>(qsl->GetSampleAddress(tasks[i].first.index, 2));
        int Si = std::accumulate(mask->begin(), mask->end(), 0);
        cuSeqlens[i + 1] = cuSeqlens[i] + Si;
        maxSeqLen = std::max(maxSeqLen, Si);
    }
    // find the closest one that is supported
    int seqLen = this->GetClosestSeqLen(maxSeqLen);
    VLOG(1) << "Max SeqLen found in Batch: " << maxSeqLen << " chosen: " << seqLen;

    // stage batch using correct batchSize and seqLen
    for(int i = 0; i < actualBatchSize; ++i)
    {
        const int offset = cuSeqlens[i]; // offset
        const int numElements = cuSeqlens[i + 1] - offset; // numElements
        this->GetBufferInputIds().H2H(qsl->GetSampleAddress(tasks[i].first.index, 0), offset,
                                      numElements);
        this->GetBufferSegmentIds().H2H(qsl->GetSampleAddress(tasks[i].first.index, 1), offset,
                                        numElements);
    }

    this->GetBufferInputMask().H2H(cuSeqlens.data(), 0, cuSeqlens.size());

    // size of packed sequences
    const size_t sumS = cuSeqlens.back();

    // Pad dummy values if using CUDA Graphs and there is a valid Graph Spec.
    int dummyBatchSize = 0;
    std::vector<int> dummySeqlens;
    bool launchGraph = false;
    if (mUseGraphs) {
        auto graphSpecs = this->GetClosestGraphSpec(maxSeqLen, actualBatchSize, sumS);
        // do not launch CUDA graph if we cannot find appropriate graph spec
        if (std::get<1>(graphSpecs) != -1) {
            launchGraph = true;
            dummyBatchSize = std::get<1>(graphSpecs) - actualBatchSize;
            // the dummy seqlens are always 0.
            dummySeqlens.assign(dummyBatchSize, sumS);
            // copy dummy values if we want to launch CUDA graph
            // No need to copy dummy values to InputIds and SegmentIds, only copy dummySeqlens to ensure
            // the kernels do not do extra work.
            this->GetBufferInputMask().H2H(dummySeqlens.data(), cuSeqlens.size(), dummySeqlens.size());
        } else {
            VLOG(1) << "Cannot find appropriate CUDA graph for " << maxSeqLen << "," << actualBatchSize << "," << sumS;
        }
    }

    // a BERTCoreVS object per thread, so no contention for its resources
    const int sidx = mCounter % mCopyStreams.size();
    auto& copyStream = mCopyStreams[sidx];
    auto& inputIds = GetBufferInputIds();
    auto& segmentIds = GetBufferSegmentIds();
    auto& inputMask = GetBufferInputMask();
    auto& outputBuf = GetBufferOutput();

    // Set batch size
    CHECK_EQ(actualBatchSize <= mMaxBatchSize, true);

    VLOG(2) << "MaxSeqlen: " << seqLen << " Input length: " << cuSeqlens.back()
            << " Batch size: " << actualBatchSize;

#ifdef CUDA_GRAPH_STATS
    // TODO: need to update if MHA kernels change
    for (auto v : {257, 193, 129, 1}) {
        if (maxSeqLen >= v) {
            ++mMaxSeqLenCounts[v];
            break;
        }
    }
    auto getQuantizedMaxSeqLen = [](int seqLen) {
        for (auto v : {128, 192, 256, 384}) {
            if (seqLen <= v) {
                return v;
            }
        }
    };
    auto getQuantizedTotalSeqLen= [](int seqLen) {
        // TODO: granularity of quantization is dependent on GPU and dataset
        static const int ROUND = 1;
        return ((seqLen + ROUND - 1) / ROUND) * ROUND;
    };
    mQuantizedPoints.emplace(std::make_tuple(getQuantizedMaxSeqLen(maxSeqLen), actualBatchSize, getQuantizedTotalSeqLen(cuSeqlens.back())));
#endif

    // Copy buffers, only copy dummy value to inputMasks
    inputIds.H2DAsync(sumS, copyStream.get());
    segmentIds.H2DAsync(sumS, copyStream.get());
    inputMask.H2DAsync(actualBatchSize + dummyBatchSize + 1, copyStream.get());

    copyStream.recordH2D();
    copyStream.makeAwaitH2D(mStream);

    // Enqueue kernels
    if (!launchGraph) {
        // Set batch size
        // TODO support only for fixed sequence length
        auto context = GetContext(actualBatchSize, seqLen);
        SetInputShapes(context, sumS, actualBatchSize, maxSeqLen);

        // Run inference
        CHECK_EQ(context->enqueueV2(GetBindings(), mStream, nullptr), true);
    } else {
        auto graphSpec = this->GetClosestGraphSpec(maxSeqLen, actualBatchSize, sumS);
        CHECK_EQ(std::get<0>(graphSpec) != -1, true);
#ifdef CUDA_GRAPH_STATS
        ++mCudaGraphCounts[graphSpec];
#endif

        // retrieve graph using sid and graphSpec
        BERTGraphKey_t key = std::make_pair(StreamIdx(), graphSpec);

        // the key must be stored in the map
        CHECK_EQ(mCudaGraphExecs.find(key) != mCudaGraphExecs.end(), true);
        CHECK_EQ(cudaGraphLaunch(mCudaGraphExecs[key], mStream), CUDA_SUCCESS);
    }

    copyStream.recordInferDone(mStream);

    // Get output
    copyStream.awaitInfer();
    const size_t outputSize = 2 * sumS;
    outputBuf.D2HAsync(outputSize, copyStream.get());
    copyStream.recordD2H();

    // prepare the response
    BERTResponse resp;
    char * ptr = reinterpret_cast<char*>(outputBuf.HostData());

    resp.resultReady = copyStream.d2h;
    resp.QSRs.reserve(actualBatchSize);
    for(size_t i = 0; i < actualBatchSize; i++)
    {
        const int s_b = cuSeqlens[i + 1] - cuSeqlens[i];
        const size_t logitSizeInBytes = 2 * s_b * sizeof(BERTOutputType);
        // this is to handle warmup - is it robust?
        if(tasks[i].first.id == 0) continue;
        mlperf::QuerySampleResponse response{tasks[i].first.id, reinterpret_cast<uintptr_t>(ptr),
                                             logitSizeInBytes};
        resp.QSRs.emplace_back(response);
        resp.copyStreamIdx = mCounter;

        ptr += logitSizeInBytes;
    }

    EnqueueResponse(resp);
}

void BERTCoreVS::WarmUp()
{
    CHECK_EQ(mCounter, 0);
    auto context = GetContext(mMaxBatchSize, BERT_MAX_SEQ_LENGTH);
    SetInputShapes(context, mMaxBatchSize * BERT_MAX_SEQ_LENGTH, mMaxBatchSize, BERT_MAX_SEQ_LENGTH);
    
    GetBufferInputIds().memsetD(0);
    GetBufferSegmentIds().memsetD(0);
    GetBufferInputMask().memsetD(0);
    void** bindings = mBindings[0].data();
    CHECK_EQ(context->enqueueV2(bindings, mStream, nullptr), true);
    CHECK_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    
    CHECK_EQ(mCounter, 0);
}

BERTCoreVS::~BERTCoreVS()
{
    CHECK_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    {
        std::unique_lock<std::mutex> lck(mMtx);
        mStopWork = true;
        mCondVar.notify_all();
    }
    for(auto & rt :mResponseThreads){
        rt.join();
    }
    for (auto & kv : mCudaGraphExecs) {
        CHECK_EQ(cudaGraphExecDestroy(kv.second), CUDA_SUCCESS);
    }

#ifdef CUDA_GRAPH_STATS
    LOG(INFO) << "CUDA Graphs Statistics:";
    if (!mCudaGraphCounts.empty()) {
        for (auto & kv : mCudaGraphCounts) {
            auto& [maxSL, BS, totSL] = kv.first;
            LOG(INFO) << "(" << maxSL << ", " << BS << ", " << totSL << "): " << kv.second;
        }
    }
    if (!mMaxSeqLenCounts.empty()) {
        LOG(INFO) << "Max seqlens:";
        for (auto it = mMaxSeqLenCounts.begin(); it != mMaxSeqLenCounts.end(); ++it) {
            auto nt = std::next(it);
            if (nt != mMaxSeqLenCounts.end()) {
                LOG(INFO) << it->first << " <= maxSeqLen < " << nt->first << ": " << it->second;
            } else {
                LOG(INFO) << it->first << " <= maxSeqLen < " << BERT_MAX_SEQ_LENGTH << ": " << it->second;
            }
        }
    }
    if (!mQuantizedPoints.empty()) {
        std::unordered_map<int, std::tuple<int, int, int>> maxSeqLenToMinMaxCount;
        for (auto & v : mQuantizedPoints) {
            auto& [maxSL, BS, totSL] = v;
            if (maxSeqLenToMinMaxCount.find(maxSL) == maxSeqLenToMinMaxCount.end()) {
                maxSeqLenToMinMaxCount[maxSL] = std::make_tuple(totSL, totSL, 1);
            } else {
                auto [prevMin, prevMax, prevCount] = maxSeqLenToMinMaxCount[maxSL];
                maxSeqLenToMinMaxCount[maxSL] = std::make_tuple(std::min(totSL, prevMin),
                                                                std::max(totSL, prevMax),
                                                                prevCount + 1);
            }
        }

        LOG(INFO) << "Quantized points statistics:";
        for (auto & pr : maxSeqLenToMinMaxCount) {
            LOG(INFO) << pr.first << "," << std::get<0>(pr.second) << "," << std::get<1>(pr.second) << "," << std::get<2>(pr.second);
        }
    }
#endif
}
