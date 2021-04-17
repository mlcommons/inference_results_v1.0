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
// General C++
#include <chrono> // Time-UDLs

#include "nvtx_wrapper.h"
#include "qsl.hpp" // QuerySample
#include "SyncWorkQueue.hpp"
#include "warmup.hpp" // For seeing WarmupSampleLibrary

// DALI items
#include "dali/c_api.h"
#include "dali/operators.h"
#include "dali/kernels/common/scatter_gather.h"

// REMOVE ONCE COMMANDLINE PARAMS STABALIZE
DECLARE_bool(use_copy_kernel);
DECLARE_bool(use_copy_kernel_cudamemcpy);
DECLARE_bool(enable_audio_processing);
DECLARE_bool(start_from_device);
DECLARE_bool(audio_fp16_input);

// =============================
//     Audio processing
// =============================

// Reads in the wav files and processes it to extract audio features
//

struct sampleAttributes{
    mlperf::QuerySample querySample;
    void* devDataPtr;
    void* devSeqLenPtr;
    int64_t hostSeqLen;
    size_t daliBufIdx;

    sampleAttributes() {
        devDataPtr = nullptr;
        devSeqLenPtr = nullptr;
    }
};

struct DaliOutTuple {
        size_t numFeatures;
        size_t seqLen;
};

class AudioOutBuf
{
    private:
    size_t mNumBufLines;
    size_t mBufLineSize;

    CudaBuffer<DaliOutTuple> mDaliOutTupleBuf;

    CudaBufferRaw mDeviceDataBuf;
    HostBuffer<int64_t> mHostSeqLenBuf;
    std::vector<mlperf::QuerySample> querySamples;

    public:
        AudioOutBuf(size_t bufferLineSize, size_t numBufLines) :
            mBufLineSize(bufferLineSize),
            mNumBufLines(numBufLines),
            mDeviceDataBuf(numBufLines, bufferLineSize),
            mDaliOutTupleBuf(numBufLines),
            mHostSeqLenBuf(numBufLines)
        {
            querySamples.resize(mNumBufLines);
        }

        auto getDataAddr(size_t bufferLineIndex) { return mDeviceDataBuf.get_ptr_from_idx(bufferLineIndex);}
        auto getDevSeqLenAddr(size_t bufferLineIndex) { return (size_t*) mDaliOutTupleBuf.get_ptr_from_idx(bufferLineIndex) + 1;}
        auto getDevTupleAddr(size_t bufferLineIndex) { return mDaliOutTupleBuf.get_ptr_from_idx(bufferLineIndex);}
        auto getHostSeqLenAddr(size_t bufferLineIndex) { return mHostSeqLenBuf.get_ptr_from_idx(bufferLineIndex);}

        // We are only inserting samples.. deleting means overwriting the old one with the new one
        void insertSample(size_t bufferLineIndex, mlperf::QuerySample &sample, dali::kernels::ScatterGatherGPU* &SGSeqLen, cudaStream_t &cudaStream) {
            querySamples[bufferLineIndex].id = sample.id;
            querySamples[bufferLineIndex].index = sample.index;

            auto dstHostSeqLenAddr = mHostSeqLenBuf.get_ptr_from_idx(bufferLineIndex);
            if(FLAGS_use_copy_kernel)
                SGSeqLen->AddCopy(dstHostSeqLenAddr, getDevSeqLenAddr(bufferLineIndex), mHostSeqLenBuf.bytesPerBatch());
            else
                CHECK_EQ(cudaMemcpyAsync(dstHostSeqLenAddr, getDevSeqLenAddr(bufferLineIndex), mHostSeqLenBuf.bytesPerBatch(), cudaMemcpyDeviceToHost, cudaStream), cudaSuccess);
        }
};

struct DaliBufferAttributes{
    std::vector<void *> daliInHostSizeBufPtr;
    std::vector<void *> daliInDevDataBufPtr;
    std::vector<void *> daliOutDataBufPtr;
    std::vector<void *> daliOutTupleBufPtr;
    size_t buffferLineSize;
};

class AudioBufferManagement
{
    private:
        // AudioOutBuf and its state
        // -----------------------------------------------------------------------------------------------------------------------
        // Line               0 -> | mValid | mAllocated | mlperf::QuerySample | mSampleSize | --Data -> mAudioOutBuf[0] --|
        // -----------------------------------------------------------------------------------------------------------------------
        // Line               1 -> | mValid | mAllocated | mlperf::QuerySample | mSampleSize | --Data -> mAudioOutBuf[1] --|    <- mRdPtr
        // -----------------------------------------------------------------------------------------------------------------------
        //                                         |
        //                                         |
        // -----------------------------------------------------------------------------------------------------------------------
        // Line               N -> | mValid | mAllocated | mlperf::QuerySample | mSampleSize | --Data -> mAudioOutBuf[1] --|    <- mWrPtr
        // -----------------------------------------------------------------------------------------------------------------------
        //                                         |
        //                                         |
        // -----------------------------------------------------------------------------------------------------------------------
        // Line mNumBufLines -> | mValid | mAllocated | mlperf::QuerySample | mSampleSize | --Data -> mAudioOutBuf[1] --|
        // -----------------------------------------------------------------------------------------------------------------------
        std::unique_ptr<AudioOutBuf> mAudioOutBuf;

        struct AudioOutBufState
        {
            std::vector<bool> mValid;
            std::vector<bool> mReserved;
            std::vector<bool> mAllocated;
            std::vector<mlperf::QuerySample> qs;
            size_t mNumBufLines;
            size_t mBufLineSize;
            volatile size_t mRdPtr;
            volatile size_t mWrPtr;
            volatile size_t mReservedIndexes;
            volatile size_t mPendingIndexes;
            std::mutex mMutex;
            std::condition_variable	mCVSamplesAvailable;    // Consumers wait till samples are available
            std::condition_variable	mCVSpaceAvailable;      // DALI process waits till atleast DALI batch worth of spaces available
        };
        struct AudioOutBufState mBufState;
        // QSL
        std::shared_ptr<qsl::SampleLibrary> mQsl;
        std::shared_ptr<WarmupSampleLibrary> mWqsl;
        qsl::LookupableQuerySampleLibrary *mActiveQsl;
        bool is_warmup{false};

        size_t mSampleSize; // Size of audio .npy file - obtained from QSL

        // DALI related parameters
        // ----------------------------------------
        size_t mDaliBatchSize;
        size_t mDaliPipelineDepth;

        // DALI Input Buffer for H2D memcpy
        std::vector<HostBuffer<size_t>> mDaliInHostSizeBuf;
        std::vector<CudaBufferRaw> mDaliInDevDataBuf;

        // DALI Output Buffer for storing DALI output tensors
        std::vector<CudaBufferRaw> mDaliOutDataBuf;
        // Dali gives output as tuple of (NUM_FEATURES=240, SEQ_LEN), we only want the second entry
        struct DaliOutTuple {
            size_t numFeatures;
            size_t seqLen;
        };
        std::vector<CudaBuffer<DaliOutTuple>> mDaliOutTupleBuf;
        std::vector<std::vector<mlperf::QuerySample> > mDaliOutSamples;

        // ----------------------------------------
        size_t mRequestedDaliBatchSize {0};
        size_t mGetBatchSize {0};
        size_t mNumDevices;

        bool mDone {false};

        // Global WorkQueue
        std::shared_ptr<SyncWorkQueue> mWorkQueue;

    public:
        AudioBufferManagement(size_t daliBatchSize,
                              size_t maxBufLineSize,
                              size_t numBufLines,
                              size_t numDevices,
                              std::shared_ptr<qsl::SampleLibrary> qsl,
                              std::shared_ptr<WarmupSampleLibrary> wqsl,
                              std::shared_ptr<SyncWorkQueue> workQueue,
                              size_t daliPipelineDepth) :
            mQsl(qsl),
            mWqsl(wqsl),
            mActiveQsl(qsl.get()),
            mSampleSize(mActiveQsl->GetSampleSize(0)),
            mNumDevices(numDevices),
            mDaliBatchSize(daliBatchSize),
            mDaliPipelineDepth(daliPipelineDepth),

            mDaliOutSamples(mDaliPipelineDepth),
            mWorkQueue(workQueue)
        {
            for (auto i=0; i < mDaliPipelineDepth; ++i) {
                mDaliInHostSizeBuf.emplace_back(mDaliBatchSize);
                mDaliInDevDataBuf.emplace_back(mDaliBatchSize, mSampleSize);
                mDaliOutDataBuf.emplace_back(mDaliBatchSize, maxBufLineSize);
                mDaliOutTupleBuf.emplace_back(mDaliBatchSize);
            }

            mBufState.mNumBufLines = numBufLines;
            mBufState.mBufLineSize = maxBufLineSize;

            mAudioOutBuf = std::make_unique<AudioOutBuf>(maxBufLineSize, numBufLines);

            mBufState.mValid.resize(numBufLines);
            mBufState.mReserved.resize(numBufLines);
            mBufState.mAllocated.resize(numBufLines);
            mBufState.qs.resize(numBufLines);

            mBufState.mRdPtr = 0;
            mBufState.mWrPtr = 0;
            mBufState.mReservedIndexes = 0;
            mBufState.mPendingIndexes = 0;

            for(auto lineNum=0; lineNum < numBufLines; lineNum++){
                mBufState.mValid[lineNum] = false;
                mBufState.mReserved[lineNum] = false;
                mBufState.mAllocated[lineNum] = false;
                mBufState.qs[lineNum].id = 0;
                mBufState.qs[lineNum].index = 0;
            }
        }

        ~AudioBufferManagement(){
            mDone = true;
        }

        int8_t *getDataAddr(size_t bufferLineIndex) { return ((int8_t*)mAudioOutBuf->getDataAddr(bufferLineIndex));}
        int64_t *getDevSeqLenAddr(size_t bufferLineIndex) { return ((int64_t*)mAudioOutBuf->getDevSeqLenAddr(bufferLineIndex));}
        DaliOutTuple *getDevTupleAddr(size_t bufferLineIndex) { return ((DaliOutTuple*)mAudioOutBuf->getDevTupleAddr(bufferLineIndex));}
        int64_t *getHostSeqLenAddr(size_t bufferLineIndex) { return ((int64_t*)mAudioOutBuf->getHostSeqLenAddr(bufferLineIndex));}

        size_t getBufLineSize()
        {
            return mBufState.mBufLineSize;
        }

        void setWarmup(bool off_on)
        {
            //CHECK(off_on != is_warmup);
            if (off_on) {
                is_warmup = true;
                mActiveQsl = mWqsl.get();
            } else {
                is_warmup = false;
                mActiveQsl = mQsl.get();
            }
        }
        size_t getSampleSize()
        {
            return mSampleSize;
        }

        void Done(){
            mDone = true;
            //mBufState.mCVSpaceAvailable.notify_all();
            //mBufState.mCVSamplesAvailable.notify_all();
        }

        // APIs for AudioOutBuf
        // Writing new data to the buffer
        void getFreeBufIndexes(std::vector<size_t> &bufIndex)
        {
            nvtxRangeId_t nvtxDaliWaitForBufSpace;
            NVTX_START_WITH_PAYLOAD(nvtxDaliWaitForBufSpace, "Dali::WaitForFreeBufIndexes", COLOR_YELLOW_4, mRequestedDaliBatchSize);
            while(!(mBufState.mNumBufLines - mBufState.mReservedIndexes >= mRequestedDaliBatchSize)) {};
            NVTX_END(nvtxDaliWaitForBufSpace);
   		    std::unique_lock<std::mutex> lock(mBufState.mMutex);
            //mBufState.mCVSpaceAvailable.wait(lock, [this](){return ((mBufState.mNumBufLines - mBufState.mReservedIndexes >= mRequestedDaliBatchSize) || (mDone == true));});

            if(mDone == true) return;

            mBufState.mReservedIndexes += mRequestedDaliBatchSize;
            std::vector<size_t> availableIndexes;
            size_t numAvailableSpaces = 0;
            size_t index = mBufState.mWrPtr;
            for(auto i = 0; i < mBufState.mNumBufLines; i++){
                if(mBufState.mReserved[index] == false){
                    availableIndexes.push_back(index);
                    mBufState.mReserved[index] = true;

                    numAvailableSpaces++;

                    if(numAvailableSpaces >= mRequestedDaliBatchSize)
                        break;
                }
                index = (index + 1) % mBufState.mNumBufLines;
            }

            bufIndex = std::move(availableIndexes);
            mBufState.mWrPtr = (index + 1) % mBufState.mNumBufLines;
        }

        void allocateDaliBatch(cudaStream_t &cudaStream, size_t pingPongId, std::vector<size_t> &bufIndex, dali::kernels::ScatterGatherGPU* &SGData, dali::kernels::ScatterGatherGPU* &SGSeqLen){
            size_t samples = 0;
            size_t offset = 0;
            // transfer the DaliOuput buffer contents to AudioOutBuf
            for(auto index : bufIndex){
                mAudioOutBuf->insertSample(index, mDaliOutSamples[pingPongId][samples], SGSeqLen, cudaStream);

                mBufState.qs[index].id =  mDaliOutSamples[pingPongId][samples].id;
                mBufState.qs[index].index =  mDaliOutSamples[pingPongId][samples].index;
                samples++;
            }

            if(FLAGS_use_copy_kernel){
                SGSeqLen->Run(cudaStream, true, dali::kernels::ScatterGatherGPU::Method::Kernel);
            }

            cudaStreamSynchronize(cudaStream);

            mDaliOutSamples[pingPongId].clear();

            {
       		    std::unique_lock<std::mutex> lock(mBufState.mMutex);
                for(auto index : bufIndex){
                    mBufState.mValid[index] = true;
                }
                mBufState.mPendingIndexes += bufIndex.size();

                //mBufState.mCVSamplesAvailable.notify_all();
            }
        }

        // Removing processed entries from the buffer
        void releaseBatch(std::vector<size_t> &bufIndex){
            nvtxRangeId_t nvtxReleaseBatch;
            NVTX_START_WITH_PAYLOAD(nvtxReleaseBatch, "DALI::ReleaseBatch", COLOR_YELLOW_2, bufIndex.size());
		    std::unique_lock<std::mutex> lock(mBufState.mMutex);
            for(auto i = 0; i < bufIndex.size(); i++)
            {
                auto index = bufIndex[i];
                mBufState.mValid[index] = false;
                mBufState.mReserved[index] = false;
                mBufState.mAllocated[index] = false;
            }

            mBufState.mReservedIndexes -= bufIndex.size();
            NVTX_END(nvtxReleaseBatch);
            //mBufState.mCVSpaceAvailable.notify_one();
        }

        // The caller needs to provide std::vector<size_t> &bufIndex which contains the allocated buffer indexes for this batch
        // Once the consumer consumes the batch, it shoudl call releaseBatch and pass along the above vector
        void getBatch(std::vector<sampleAttributes>& batch, size_t maxBatchSize){
            nvtxRangeId_t nvtxGetEncoderBatch;

            if (maxBatchSize == 0) return;
		    std::unique_lock<std::mutex> lock(mBufState.mMutex);
            mGetBatchSize = maxBatchSize;
		    /*mBufState.mCVSamplesAvailable.wait(lock, [this]() {
                return ((mBufState.mPendingIndexes > 0) || (mDone == true));
            });*/
            if(mBufState.mPendingIndexes == 0) return;

            if(mDone == true) return;

            size_t samples = 0;
            size_t samplesInThisBatch = mGetBatchSize > mBufState.mPendingIndexes ? mBufState.mPendingIndexes : mGetBatchSize;
            NVTX_START_WITH_PAYLOAD(nvtxGetEncoderBatch, "DALI::GetBatch", COLOR_YELLOW_3, samplesInThisBatch);
            size_t initial_RdPtr = mBufState.mRdPtr;
            batch.clear();
            batch.resize(samplesInThisBatch);
            for(auto i = 0; i < mBufState.mNumBufLines; i++){
                auto index = ( initial_RdPtr + i ) % mBufState.mNumBufLines;
                if(mBufState.mValid[index] && !mBufState.mAllocated[index]){
                    batch[samples].querySample.id = mBufState.qs[index].id;
                    batch[samples].querySample.index = mBufState.qs[index].index;
                    batch[samples].devDataPtr = mAudioOutBuf->getDataAddr(index);
                    batch[samples].devSeqLenPtr = mAudioOutBuf->getDevSeqLenAddr(index);
                    batch[samples].hostSeqLen = *mAudioOutBuf->getHostSeqLenAddr(index);
                    batch[samples].daliBufIdx = index;

                    mBufState.mRdPtr = (index + 1) % mBufState.mNumBufLines;
                    mBufState.mAllocated[index] = true;
                    samples++;

                    if(samples >= samplesInThisBatch)
                        break;
                }
            }

            mBufState.mPendingIndexes -= samplesInThisBatch;
            NVTX_END(nvtxGetEncoderBatch);
        }

        // APIs for DALI Pipeline
        //
        void getDaliBufferAttributes(DaliBufferAttributes &attr){
            attr.daliInHostSizeBufPtr.resize(mDaliPipelineDepth);
            attr.daliInDevDataBufPtr.resize(mDaliPipelineDepth);
            attr.daliOutDataBufPtr.resize(mDaliPipelineDepth);
            attr.daliOutTupleBufPtr.resize(mDaliPipelineDepth);

            for(auto i = 0 ; i < mDaliPipelineDepth; i++){
                attr.daliInHostSizeBufPtr[i] = mDaliInHostSizeBuf[i].data();
                attr.daliInDevDataBufPtr[i] = mDaliInDevDataBuf[i].data();
                attr.daliOutDataBufPtr[i] = mDaliOutDataBuf[i].data();
                attr.daliOutTupleBufPtr[i] = mDaliOutTupleBuf[i].data();
            }
            attr.buffferLineSize = mBufState.mBufLineSize;
        }

        bool makeDaliBatch(cudaStream_t &cudaStream, size_t pingPongId, std::vector<size_t> &bufIndexes, dali::kernels::ScatterGatherGPU* &scatterGatherH2D, std::vector<std::vector<void*> > &daliInHostQSLSamplePtr, size_t deviceIdx )
        {
            std::vector<mlperf::QuerySample> samples;

            mWorkQueue->getBatch(samples, mDaliBatchSize);

            nvtxRangeId_t nvtxMakeDaliBatch;
            if(samples.size() > 0){
                NVTX_START_WITH_PAYLOAD(nvtxMakeDaliBatch, "DALI:makeDaliBatch", COLOR_YELLOW_0, samples.size());
                // Store a copy of samples to pass it along the pipeline
                mDaliOutSamples[pingPongId].resize(samples.size());
                auto index = 0;
                for(auto sample : samples){
                    mDaliOutSamples[pingPongId][index].id = sample.id;
                    mDaliOutSamples[pingPongId][index].index = sample.index;

                    void* qslSampleData = mActiveQsl->GetSampleAddress(sample.index, 0, deviceIdx);
                    size_t qslSampleSize = *static_cast<int32_t*>(mActiveQsl->GetSampleAddress(sample.index, 1));

                    *mDaliInHostSizeBuf[pingPongId].get_ptr_from_idx(index) = qslSampleSize;

                    index++;
                }
            }

            if(samples.size() == 0){
                return false;
            }
            else if(samples.size() < mDaliBatchSize){
                for(auto i = samples.size(); i < mDaliBatchSize; i++){
                    *mDaliInHostSizeBuf[pingPongId].get_ptr_from_idx(i) = 0;
                }
            }

            // else we already have mDaliBatchSize worth of samples
            size_t offset = 0;
            for(auto i=0; i < mDaliBatchSize; i++){
                if(i < samples.size()){
                    void* qslSampleData = mActiveQsl->GetSampleAddress(samples[i].index, 0, deviceIdx);
                    size_t qslSampleSize = *static_cast<int32_t*>(mActiveQsl->GetSampleAddress(samples[i].index, 1));
                    size_t dataSize = FLAGS_audio_fp16_input ? 2 : sizeof(float);

                    if(!FLAGS_start_from_device){
                        cudaMemcpyAsync((int8_t*)mDaliInDevDataBuf[pingPongId].data() + offset, qslSampleData, qslSampleSize * dataSize, cudaMemcpyHostToDevice, cudaStream);
                        cudaStreamSynchronize(cudaStream);
                        offset += (qslSampleSize * dataSize);
                    }
                    daliInHostQSLSamplePtr[pingPongId].emplace_back(qslSampleData);
                }
                else
                {
                    void* qslSampleData = mActiveQsl->GetSampleAddress(samples[samples.size()-1].index, 0, deviceIdx);
                    daliInHostQSLSamplePtr[pingPongId].emplace_back(qslSampleData);
                }
            }

            mRequestedDaliBatchSize = mDaliOutSamples[pingPongId].size();
            getFreeBufIndexes(bufIndexes);
            NVTX_END(nvtxMakeDaliBatch);
            return true;
        }

};

// DALI pipeline processing
class DaliPipeline {
    private:
        std::shared_ptr<AudioBufferManagement> mAudioBufManager;

        device_type_t mDeviceType;

        nvinfer1::DataType mPrecision{nvinfer1::DataType::kFLOAT};
        int mDeviceId{-1};
        size_t mNumInputs{0};
        size_t mNumThreads{1};  // FIXME - make input parameter
        size_t mBytesPerSample{0};  // FIXME - make input parameter
        size_t mBytesPerSampleHint{0}; // FIXME - make input parameter
        size_t mDaliPipelineDepth;
        std::vector<cudaEvent_t> mEventBatchReady;
        std::vector<cudaEvent_t> mEventH2DBufReady;
        std::vector<cudaEvent_t> mEventBatchAck;
        cudaStream_t mH2DStream;
        cudaStream_t mD2DStream;
        cudaStream_t mCopyOutStream;
        std::string mSerializedFileName;
        size_t mDaliBatchSize;
        size_t mDaliBatchesIssueAhead;
        bool mDone{false};

        void *mDaliInSeqLen;
        void *mDaliOutTensorSizes;
        void *mDaliTensorListNToOutput;

        // DALI Output Buffer for handling partial dali batches
        std::vector<CudaBufferRaw> mDaliOutPartialBatch;

        daliPipelineHandle mDaliPipelineHandle;

        std::thread mDaliProcessingThread;

        std::mutex mMutex;
        std::condition_variable mCVBatchReady;
        size_t mBatchesReady {0};      // How many dali batches are ready for allocation ?  Possible values 0, 1, 2.
        volatile size_t mRdPtr {0};
        volatile size_t mWrPtr {0};

        size_t mSGBlockSize = 1<<18;  // 256kB per block
        dali::kernels::ScatterGatherGPU *mScatterGatherH2D;
        dali::kernels::ScatterGatherGPU *mScatterGatherD2DData;
        dali::kernels::ScatterGatherGPU *mScatterGatherD2DSeqLen;

    public:
        DaliPipeline(
            const int deviceId,
            std::string &serializedFileName,
            std::shared_ptr<AudioBufferManagement> audioBufManager,
            size_t daliBatchSize,
            device_type_t deviceType,
            size_t prefetch_queue_depth,
            size_t daliPipelineDepth,
            size_t daliBatchesIssueAhead
            ) : mDeviceId(deviceId), mAudioBufManager(audioBufManager), mDeviceType(deviceType), mSerializedFileName(serializedFileName), mDaliBatchSize(daliBatchSize), mDaliPipelineDepth(daliPipelineDepth), mDaliBatchesIssueAhead(daliBatchesIssueAhead)
        {
            CHECK_EQ(cudaMallocHost(&mDaliInSeqLen, mDaliBatchSize * sizeof(int64_t)), cudaSuccess);
            CHECK_EQ(cudaMallocHost(&mDaliOutTensorSizes, mDaliBatchSize * sizeof(size_t)), cudaSuccess);
            CHECK_EQ(cudaMallocHost(&mDaliTensorListNToOutput, mDaliBatchSize * sizeof(size_t) * 2), cudaSuccess);

            for (auto i=0; i < mDaliPipelineDepth; ++i)
                mDaliOutPartialBatch.emplace_back(mDaliBatchSize, audioBufManager->getBufLineSize());

            mEventBatchReady.resize(mDaliPipelineDepth);
            mEventBatchAck.resize(mDaliPipelineDepth);
            mEventH2DBufReady.resize(mDaliPipelineDepth);

            for(auto i = 0; i < mDaliPipelineDepth; i++){
                CHECK_EQ(cudaEventCreateWithFlags(&mEventBatchReady[i], cudaEventDisableTiming), cudaSuccess);
                CHECK_EQ(cudaEventCreateWithFlags(&mEventBatchAck[i], cudaEventDisableTiming), cudaSuccess);
                CHECK_EQ(cudaEventCreateWithFlags(&mEventH2DBufReady[i], cudaEventDisableTiming), cudaSuccess);
            }

            daliInitialize();
            daliInitOperators();

            auto estimatedNBlocks = (mDaliBatchSize * mAudioBufManager->getSampleSize() / mSGBlockSize) + 1;
            mScatterGatherH2D = new dali::kernels::ScatterGatherGPU (mSGBlockSize, estimatedNBlocks);

            mScatterGatherD2DData = new dali::kernels::ScatterGatherGPU (mAudioBufManager->getBufLineSize(), mDaliBatchSize);

            mScatterGatherD2DSeqLen = new dali::kernels::ScatterGatherGPU (sizeof(int64_t), mDaliBatchSize);

            std::stringstream ss;
            std::ifstream fin(mSerializedFileName, std::ios::binary);
            ss << fin.rdbuf();
            auto pipe = ss.str();
            std::cout << "Dali pipeline creating.." <<  std::endl;

            daliCreatePipeline(&mDaliPipelineHandle,
                                pipe.c_str(),
                                pipe.length(),
                                mDaliBatchSize,
                                mNumThreads,
                                mDeviceId,
                                false,
                                prefetch_queue_depth,
                                prefetch_queue_depth,
                                prefetch_queue_depth,
                                1);
            std::cout << "Dali pipeline created" << std::endl;

            CHECK_EQ(cudaStreamCreateWithFlags(&mH2DStream, cudaStreamNonBlocking), cudaSuccess);
            CHECK_EQ(cudaStreamCreateWithFlags(&mD2DStream, cudaStreamNonBlocking), cudaSuccess);
            CHECK_EQ(cudaStreamCreateWithFlags(&mCopyOutStream, cudaStreamNonBlocking), cudaSuccess);
        }

        ~DaliPipeline(){
            if(FLAGS_enable_audio_processing)
                joinThread();

            CHECK_EQ(cudaStreamDestroy(mH2DStream), cudaSuccess);
            CHECK_EQ(cudaStreamDestroy(mD2DStream), cudaSuccess);
            CHECK_EQ(cudaStreamDestroy(mCopyOutStream), cudaSuccess);
            daliDeletePipeline(&mDaliPipelineHandle);
            CHECK_EQ(cudaFreeHost(mDaliInSeqLen), cudaSuccess);
            CHECK_EQ(cudaFreeHost(mDaliOutTensorSizes), cudaSuccess);
            CHECK_EQ(cudaFreeHost(mDaliTensorListNToOutput), cudaSuccess);

            for(auto i = 0; i < mDaliPipelineDepth; i++){
                CHECK_EQ(cudaEventDestroy(mEventBatchReady[i]), cudaSuccess);
                CHECK_EQ(cudaEventDestroy(mEventBatchAck[i]), cudaSuccess);
                CHECK_EQ(cudaEventDestroy(mEventH2DBufReady[i]), cudaSuccess);
            }
        }

        void processDaliBatches()
        {
            CHECK_EQ(cudaSetDevice(mDeviceId), cudaSuccess);
            DaliBufferAttributes attr;

            mAudioBufManager->getDaliBufferAttributes(attr);
            std::vector<size_t> bufIndex[mDaliPipelineDepth];
            std::vector< std::vector<void *> > daliInHostQSLSamplePtr;
            std::vector< std::vector<void *> > daliOutDevBufIndexPtr;
            std::vector< std::vector<void *> > daliOutTupleBufPtr;
            daliInHostQSLSamplePtr.resize(mDaliPipelineDepth);
            daliOutDevBufIndexPtr.resize(mDaliPipelineDepth);
            daliOutTupleBufPtr.resize(mDaliPipelineDepth);

            size_t pingPongId;

            std::vector<bool> batchIssued;

            batchIssued.resize(mDaliPipelineDepth);

            while(!mDone)
            {
                for(auto i = 0 ; i < mDaliPipelineDepth; i++){
                    bufIndex[i].clear();
                    daliInHostQSLSamplePtr[i].clear();
                    daliOutDevBufIndexPtr[i].clear();
                    daliOutTupleBufPtr[i].clear();
                    batchIssued[i] = false;
                }

                size_t issueAhead = mDaliBatchesIssueAhead;
                size_t copyIdx = 0;

                auto issueBatch = [&](int i)
                {
                    if(mAudioBufManager->makeDaliBatch(mH2DStream, i, bufIndex[i], mScatterGatherH2D, daliInHostQSLSamplePtr, mDeviceId)){
                        CHECK_EQ(cudaEventRecord(mEventH2DBufReady[i], mH2DStream), cudaSuccess);
                        CHECK_EQ(cudaStreamWaitEvent(mD2DStream, mEventH2DBufReady[i], 0), cudaSuccess);                        

                        if(!FLAGS_start_from_device){
                            if(FLAGS_audio_fp16_input)
                                daliSetExternalInputAsync(&mDaliPipelineHandle, "INPUT_0", device_type_t::GPU, attr.daliInDevDataBufPtr[i], DALI_FLOAT16, (const int64_t*) attr.daliInHostSizeBufPtr[i], 1, nullptr, mD2DStream, 0);
                            else
                                daliSetExternalInputAsync(&mDaliPipelineHandle, "INPUT_0", device_type_t::GPU, attr.daliInDevDataBufPtr[i], DALI_FLOAT, (const int64_t*) attr.daliInHostSizeBufPtr[i], 1, nullptr, mD2DStream, 0);
                        }
                        else{
                            // Now using daliSetExternalInputTensorsAsync as it avoids un-necessary D2D copies
                            if(FLAGS_audio_fp16_input)
                                daliSetExternalInputTensorsAsync(&mDaliPipelineHandle, "INPUT_0", device_type_t::GPU, daliInHostQSLSamplePtr[i].data(), DALI_FLOAT16, (const int64_t*) attr.daliInHostSizeBufPtr[i], 1, nullptr, mH2DStream, DALI_use_copy_kernel);
                            else
                                daliSetExternalInputTensorsAsync(&mDaliPipelineHandle, "INPUT_0", device_type_t::GPU, daliInHostQSLSamplePtr[i].data(), DALI_FLOAT, (const int64_t*) attr.daliInHostSizeBufPtr[i], 1, nullptr, mH2DStream, DALI_use_copy_kernel);
                        }

                        bool isPartialDaliBatch = bufIndex[i].size() < mDaliBatchSize;
                        
                        for(auto &index : bufIndex[i]){
                            daliOutDevBufIndexPtr[i].emplace_back(mAudioBufManager->getDataAddr(index));
                            daliOutTupleBufPtr[i].emplace_back(mAudioBufManager->getDevTupleAddr(index));
                        }

                        if(isPartialDaliBatch){
                            for(auto j = 0; j < mDaliBatchSize - bufIndex[i].size(); j++){
                                daliOutDevBufIndexPtr[i].emplace_back(nullptr);
                                daliOutTupleBufPtr[i].emplace_back(nullptr);
                            }
                        }

                        batchIssued[i] = true;
                    }
                };

                for(copyIdx = 0 ; copyIdx < std::min(issueAhead, mDaliPipelineDepth); copyIdx++){
                    issueBatch(copyIdx);
                }

                for(auto bufId = 0; bufId < mDaliPipelineDepth; bufId++)
                {
                    if (copyIdx < mDaliPipelineDepth)
                        issueBatch(copyIdx++);

                    if(batchIssued[bufId] == true)
                    {
                        NVTX_START(nvtxExecuteDaliBatch, "DALI:RunDaliBatch", COLOR_YELLOW_1);
                        daliRun(&mDaliPipelineHandle);

                        daliOutput(&mDaliPipelineHandle);
                        size_t tensor_size;
                        for(auto n=0; n < 2; n++) // num_outputs is expected to be equal to 2.
                        {
                            auto non_blocking = 1;

                            if(n==0){
                                daliOutputCopySamples(&mDaliPipelineHandle, daliOutDevBufIndexPtr[bufId].data(), n, mDeviceType, mD2DStream, DALI_ext_default | DALI_use_copy_kernel);
                            }
                            if(n==1){
                                daliOutputCopySamples(&mDaliPipelineHandle, daliOutTupleBufPtr[bufId].data(), n, mDeviceType, mD2DStream, DALI_ext_default | DALI_use_copy_kernel);
                            }
                        }
                        NVTX_END(nvtxExecuteDaliBatch);

                    }
                }
                for(auto bufId = 0; bufId < mDaliPipelineDepth; bufId++)
                {
                    if(batchIssued[bufId] == true)
                    {
                        NVTX_START(nvtxallocateDaliBatch, "DALI:allocateDaliBatch", COLOR_GREEN_3);
                        mAudioBufManager->allocateDaliBatch(mD2DStream, bufId, bufIndex[bufId], mScatterGatherD2DData, mScatterGatherD2DSeqLen);
                        NVTX_END(nvtxallocateDaliBatch);
                    }
                }
            }
        }

        void launchThread(){
            mDaliProcessingThread = std::thread(&DaliPipeline::processDaliBatches, this);
        }

        void joinThread(){
            mDone = true;
            mCVBatchReady.notify_all();
            if(mDaliProcessingThread.joinable()){
                mDaliProcessingThread.join();
            }
        }
};
