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

#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>

#include <cudnn.h>
#include "dlrmInteractionsPlugin.h"
#include "dlrmHelper.h"
#include "dlrmInteractionsPluginKernel.h"

using namespace nvinfer1;

namespace
{
const char* DLRM_INTERACTIONS_PLUGIN_VERSION{"1"};
const char* DLRM_INTERACTIONS_PLUGIN_NAME{"DLRM_INTERACTIONS_TRT"};
const int MAX_FILEPATH_LENGTH{1024};
const int MAX_TABLE_OFFSETS{1024};
}

std::mutex DLRMInteractionsPlugin::mSharedDataMutex;
DLRMInteractionsPlugin::HostData DLRMInteractionsPlugin::mHostData;
std::map<int, DLRMInteractionsPlugin::DeviceData *> DLRMInteractionsPlugin::mDeviceData;

PluginFieldCollection DLRMInteractionsPluginCreator::mFC{};
std::vector<PluginField> DLRMInteractionsPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(DLRMInteractionsPluginCreator);

DLRMInteractionsPlugin::DLRMInteractionsPlugin(
    int embeddingSize,
    int embeddingRows,
    int reducedPrecisionIO,
    float embeddingWeightsOnGpuPart,
    int interactionsOutputInterleaved,
    int outputPaddingGranularity,
    const std::vector<int>& tableOffsets,
    const std::string& embeddingWeightsFilepath,
    const std::string& rowFrequenciesFilepath)
    : mInitialized(false)
    , mLocalDeviceData(nullptr)
    , mHelperData(nullptr)
    , mEmbeddingSize(embeddingSize)
    , mEmbeddingRows(embeddingRows)
    , mReducedPrecisionIO(reducedPrecisionIO)
    , mEmbeddingWeightsOnGpuPart(embeddingWeightsOnGpuPart)
    , mInteractionsOutputInterleaved(interactionsOutputInterleaved)
    , mOutputPaddingGranularity(outputPaddingGranularity)
    , mTableOffsets(tableOffsets)
    , mEmbeddingWeightsFilepath(embeddingWeightsFilepath)
    , mRowFrequenciesFilepath(rowFrequenciesFilepath)
{
}

DLRMInteractionsPlugin::DLRMInteractionsPlugin(const void* data, size_t length)
    : mInitialized(false)
    , mLocalDeviceData(nullptr)
    , mHelperData(nullptr)
{
    const char* d = reinterpret_cast<const char *>(data);
    const char* a = d;

    mTotalInteractionFeatures = read<int>(d);
    mInScale = read<float>(d);
    mOutScale = read<float>(d);
    mEmbeddingSize = read<int>(d);
    mEmbeddingRows = read<int>(d);
    mReducedPrecisionIO = read<int>(d);
    mEmbeddingWeightsOnGpuPart = read<float>(d);
    mInteractionsOutputInterleaved = read<int>(d);
    mOutputPaddingGranularity = read<int>(d);

    int tableOffsetsSize = read<int>(d);
    mTableOffsets.resize(tableOffsetsSize);
    std::copy((const int *)d, (const int *)d + tableOffsetsSize, mTableOffsets.data());
    d += MAX_TABLE_OFFSETS * sizeof(int);

    int embeddingWeightsFilepathSize = read<int>(d);
    mEmbeddingWeightsFilepath = std::string(d, embeddingWeightsFilepathSize);
    d += MAX_FILEPATH_LENGTH;

    int rowFrequenciesFilepathSize = read<int>(d);
    mRowFrequenciesFilepath = std::string(d, rowFrequenciesFilepathSize);
    d += MAX_FILEPATH_LENGTH;

    ASSERT(d == a + length);
}

int DLRMInteractionsPlugin::getNbOutputs() const
{
    return 1;
}

int DLRMInteractionsPlugin::initialize()
{
    if (!mInitialized)
    {
        std::lock_guard<std::mutex> lck(mSharedDataMutex);

        CUDA_ASSERT(cudaGetDevice(&mDeviceId));
        auto it = mDeviceData.find(mDeviceId);
        if (it != mDeviceData.end())
        {
            // The device data was already loaded on this GPU
            mLocalDeviceData = it->second;
            mLocalDeviceData->mCounter++;
        }
        // Current GPU doesn't have device data initialized, but some other GPU has.
        else if (mDeviceData.size() > 0)
        {
            // Host data should be initialized by this point.
            ASSERT(mHostData.mCounter > 0);

            auto otherDeviceData = mDeviceData.begin()->second;
            mLocalDeviceData = new DeviceData();
            mDeviceData.insert(std::make_pair(mDeviceId, mLocalDeviceData));
            mLocalDeviceData->mCounter++;

            // Copy embedding data from other GPU to current GPU.
            if (otherDeviceData->mIndexRemapping != nullptr)
            {
                CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mIndexRemapping, (uint64_t)mEmbeddingRows * sizeof(int)));
                CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mIndexRemapping, otherDeviceData->mIndexRemapping, (uint64_t)mEmbeddingRows * sizeof(int), cudaMemcpyDeviceToDevice));
            }
            CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mEmbeddings, (uint64_t)mEmbeddingSize * (uint64_t)mHostData.mEmbeddingRowsOnDevice * sizeof(int8_t)));
            CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mEmbeddings, otherDeviceData->mEmbeddings, (uint64_t)mEmbeddingSize * (uint64_t)mHostData.mEmbeddingRowsOnDevice * sizeof(int8_t), cudaMemcpyDeviceToDevice));
            CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mTableOffsets, mTableOffsets.size() * sizeof(int)));
            CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mTableOffsets, otherDeviceData->mTableOffsets, mTableOffsets.size() * sizeof(int), cudaMemcpyDeviceToDevice));
        }
        // No GPUs have device data initialized
        else
        {
            // Insert device data entry for current GPU into map.
            mLocalDeviceData = new DeviceData();
            mDeviceData.insert(std::make_pair(mDeviceId, mLocalDeviceData));
            mLocalDeviceData->mCounter++;

            // No host data should be initialized at this point
            ASSERT(mHostData.mCounter == 0);

            bool useRowFrequencies = !mRowFrequenciesFilepath.empty();
            std::vector<int> positionByActualRowIdList;
            // Process file with embedding row frequencies
            if (useRowFrequencies)
            {
                std::ifstream input(mRowFrequenciesFilepath, std::ios::in | std::ios::binary);
                ASSERT(input.is_open());

                int numEmbeddingTables;
                input.read((char *)&numEmbeddingTables, sizeof(int));

                std::vector<std::pair<int, float>> rowIdFreqList(mEmbeddingRows);
                int baseRowId = 0;
                for(int feature_id = 0; feature_id < numEmbeddingTables; ++feature_id)
                {
                    int numTableRows;
                    input.read((char *)&numTableRows, sizeof(int));
                    double freqSum = 0.0;
                    for(int tableRowId = 0; tableRowId < numTableRows; ++tableRowId)
                    {
                        rowIdFreqList[baseRowId + tableRowId].first = baseRowId + tableRowId;
                        float freq;
                        input.read((char *)&freq, sizeof(float));
                        rowIdFreqList[baseRowId + tableRowId].second = freq;
                        freqSum += static_cast<double>(freq);
                    }
                    float mult = static_cast<float>(1.0 / freqSum);
                    // Normalize frequencies for each table as each sample accesses exactly 1 row from each table
                    for(int tableRowId = 0; tableRowId < numTableRows; ++tableRowId)
                        rowIdFreqList[baseRowId + tableRowId].second *= mult;
                    baseRowId += numTableRows;
                }
                std::sort(rowIdFreqList.begin(), rowIdFreqList.end(),
                    [] (const std::pair<int, float>& x, const std::pair<int, float>& y)
                    {
                        if (x.second > y.second)
                            return true;
                        else if (x.second < y.second)
                            return false;
                        else
                            return (x.first < y.first); 
                    });

                // Prepare embedding index remapping table on GPU, to be used by custom interactions kernel
                positionByActualRowIdList.resize(rowIdFreqList.size());
                for(int position_id = 0; position_id < mEmbeddingRows; ++position_id)
                    positionByActualRowIdList[rowIdFreqList[position_id].first] = position_id;
                CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mIndexRemapping, positionByActualRowIdList.size() * sizeof(int)));
                CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mIndexRemapping, &positionByActualRowIdList.front(), sizeof(int) * positionByActualRowIdList.size(), cudaMemcpyHostToDevice));
            }
            
            {
                // Number of embedding rows to keep on GPU and to keep on host.
                const int embeddingRowsOnHost = static_cast<int>(mEmbeddingRows * (1.0F - mEmbeddingWeightsOnGpuPart));
                mHostData.mEmbeddingRowsOnDevice = mEmbeddingRows - embeddingRowsOnHost;

                // Allocate memory for GPU embeddings and host embeddings.
                CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mEmbeddings, (uint64_t)mEmbeddingSize * (uint64_t)mHostData.mEmbeddingRowsOnDevice * sizeof(int8_t)));
                if (embeddingRowsOnHost > 0)
                {
                    ASSERT(useRowFrequencies);
                    CUDA_ASSERT(cudaHostAlloc(&mHostData.mHostEmbeddings, (uint64_t)mEmbeddingSize * (uint64_t)embeddingRowsOnHost * sizeof(int8_t), cudaHostAllocMapped));
                    CUDA_ASSERT(cudaHostGetDevicePointer(&mHostData.mHostEmbeddingsDevicePtr, mHostData.mHostEmbeddings, 0));
                }

                // Open embeddings file for reading
                std::ifstream input(mEmbeddingWeightsFilepath, std::ios::in | std::ios::binary);
                ASSERT(input.is_open());

                int numEmbeddingTables;
                input.read((char *)&numEmbeddingTables, sizeof(int));

                mHostData.mEmbeddingsScales.resize(numEmbeddingTables);
                input.read((char *)&mHostData.mEmbeddingsScales.front(), sizeof(float) * numEmbeddingTables);

                // Load embeddings from file to GPU memory in batches to reduce peak usage of host memory
                int rowsInBatch = std::min(mEmbeddingRows, 1024 * 1024);
                std::vector<int8_t> embeddingsInt8((uint64_t)rowsInBatch * (uint64_t)mEmbeddingSize);
                void * embeddingsStaged = nullptr;
                if (useRowFrequencies)
                {
                    CUDA_ASSERT(cudaMalloc(&embeddingsStaged, embeddingsInt8.size() * sizeof(int8_t)));
                }
                int rowsLoaded = 0;
                while(rowsLoaded < mEmbeddingRows)
                {
                    int rowsToLoad = std::min(rowsInBatch, mEmbeddingRows - rowsLoaded);
                    input.read((char *)(&embeddingsInt8.front()), sizeof(int8_t) * (uint64_t)mEmbeddingSize * (uint64_t)rowsToLoad);
                    CUDA_ASSERT(cudaMemcpy(
                        useRowFrequencies ? embeddingsStaged : ((int8_t * )mLocalDeviceData->mEmbeddings) + ((uint64_t)rowsLoaded * (uint64_t)mEmbeddingSize),
                        &embeddingsInt8.front(),
                        sizeof(int8_t) * (uint64_t)mEmbeddingSize * (uint64_t)rowsToLoad,
                        cudaMemcpyHostToDevice));
                    if (useRowFrequencies)
                    {
                        remapEmbeddingRows(
                            0,
                            (const int8_t * )embeddingsStaged,
                            (int8_t * )mLocalDeviceData->mEmbeddings,
                            ((const int *)mLocalDeviceData->mIndexRemapping) + rowsLoaded,
                            mEmbeddingSize,
                            rowsToLoad,
                            mHostData.mEmbeddingRowsOnDevice);
                    }
                    if (mHostData.mCounter == 0)
                    {
                        for(int rowInBatch = 0; rowInBatch < rowsToLoad; ++rowInBatch)
                        {
                            int actualRowPos = useRowFrequencies ? positionByActualRowIdList[rowsLoaded + rowInBatch] : rowsLoaded + rowInBatch;
                            if (actualRowPos >= mHostData.mEmbeddingRowsOnDevice)
                                memcpy(
                                    ((int8_t * )mHostData.mHostEmbeddings) + (uint64_t)(actualRowPos - mHostData.mEmbeddingRowsOnDevice) * (uint64_t)mEmbeddingSize,
                                    &embeddingsInt8[(uint64_t)rowInBatch * (uint64_t)mEmbeddingSize],
                                    sizeof(int8_t) * (uint64_t)mEmbeddingSize);
                        }
                    }
                    rowsLoaded += rowsToLoad;
                }
                if (useRowFrequencies)
                {
                    CUDA_ASSERT(cudaFree(embeddingsStaged));
                }
            }
            {
                CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mTableOffsets, mTableOffsets.size() * sizeof(int)));
                CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mTableOffsets, mTableOffsets.data(), mTableOffsets.size() * sizeof(int), cudaMemcpyHostToDevice));
            }
        }

        if ((mHelperData == nullptr) && (mReducedPrecisionIO == 2))
        {
            int outputInteractionsPadded = mHostData.mEmbeddingsScales.size() * (mHostData.mEmbeddingsScales.size() + 1) / 2 + 31;
            mHelperData = allocateHelperDataForBatchedSyrkInt8(
                outputInteractionsPadded,
                mHostData.mEmbeddingsScales,
                mInScale,
                mOutScale);
        }
        CUDA_ASSERT(cudaStreamSynchronize(0));

        mInitialized = true;
        mHostData.mCounter++;
    }

    return 0;
}

void DLRMInteractionsPlugin::terminate()
{
    if (mInitialized)
    {
        std::lock_guard<std::mutex> lck(mSharedDataMutex);
        mHostData.mCounter--;
        if (mHostData.mCounter == 0)
        {
            if (mHostData.mHostEmbeddings != nullptr)
            {
                cudaFreeHost(mHostData.mHostEmbeddings);
                mHostData.mHostEmbeddings = nullptr;
                mHostData.mHostEmbeddingsDevicePtr = nullptr;
            }
        }
        mLocalDeviceData->mCounter--;
        if (mLocalDeviceData->mCounter == 0)
        {
            if (mLocalDeviceData->mEmbeddings != nullptr)
            {
                cudaFree(mLocalDeviceData->mEmbeddings);
                mLocalDeviceData->mEmbeddings = nullptr;
            }
            if (mLocalDeviceData->mIndexRemapping != nullptr)
            {
                cudaFree(mLocalDeviceData->mIndexRemapping);
                mLocalDeviceData->mIndexRemapping = nullptr;
            }
            if (mLocalDeviceData->mTableOffsets != nullptr)
            {
                cudaFree(mLocalDeviceData->mTableOffsets);
                mLocalDeviceData->mTableOffsets = nullptr;
            }
            delete mLocalDeviceData;
            mLocalDeviceData = nullptr;
            mDeviceData.erase(mDeviceId);
        }
        mInitialized = false;
    }
    if (mHelperData != nullptr)
    {
        deallocateHelperDataForBatchedSyrkInt8(mHelperData);
        mHelperData = nullptr;
    }
}

DimsExprs DLRMInteractionsPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    ASSERT(outputIndex == 0);
    ASSERT(nbInputs == 2);

    ASSERT(inputs[1].nbDims == 2);
    ASSERT(inputs[1].d[1]->isConstant());
    int sparseFeatures = inputs[1].d[1]->getConstantValue();
    int interactionFeatures = sparseFeatures + 1;

    ASSERT(inputs[0].nbDims >= 2);
    int embeddingSize = 1;
    for(int i = 1; i < inputs[0].nbDims; ++i)
    {
        ASSERT(inputs[0].d[i]->isConstant());
        embeddingSize *= inputs[0].d[i]->getConstantValue();
    }
    ASSERT(embeddingSize == mEmbeddingSize);

    int padOutputToMultiple = mOutputPaddingGranularity; 

    if (mInteractionsOutputInterleaved)
        return DimsExprs{4, {exprBuilder.operation(DimensionOperation::kCEIL_DIV, *inputs[0].d[0], *exprBuilder.constant(2)),
            exprBuilder.constant((embeddingSize + (interactionFeatures * (interactionFeatures - 1) / 2) + padOutputToMultiple - 1) / padOutputToMultiple * padOutputToMultiple),
            exprBuilder.constant(2), exprBuilder.constant(1)}};
    else
        return DimsExprs{4, {inputs[0].d[0],
            exprBuilder.constant((embeddingSize + (interactionFeatures * (interactionFeatures - 1) / 2) + padOutputToMultiple - 1) / padOutputToMultiple * padOutputToMultiple),
            exprBuilder.constant(1), exprBuilder.constant(1)}};
}

size_t DLRMInteractionsPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}

int DLRMInteractionsPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream)
{
    int res;
    if (mReducedPrecisionIO == 2)
        res = runBatchedSyrkInt8(
            stream,
            mTotalInteractionFeatures,
            mEmbeddingSize,
            inputDesc[0].dims.d[0],
            mHostData.mEmbeddingRowsOnDevice,
            outputDesc[0].dims.d[1],
            inputs[0],
            (const int *)inputs[1],
            mLocalDeviceData->mEmbeddings,
            mHostData.mHostEmbeddingsDevicePtr,
            (const int *)(mLocalDeviceData->mIndexRemapping),
            (const int *)(mLocalDeviceData->mTableOffsets),
            mHelperData,
            outputs[0],
            mInScale,
            mOutScale,
            mInteractionsOutputInterleaved != 0);
    else
        res = runBatchedSyrk(
            stream,
            mTotalInteractionFeatures,
            mEmbeddingSize,
            inputDesc[0].dims.d[0],
            mHostData.mEmbeddingRowsOnDevice,
            outputDesc[0].dims.d[1],
            inputs[0],
            (const int *)inputs[1],
            mLocalDeviceData->mEmbeddings,
            mHostData.mHostEmbeddingsDevicePtr,
            (const int *)(mLocalDeviceData->mIndexRemapping),
            (const int *)(mLocalDeviceData->mTableOffsets),
            mHostData.mEmbeddingsScales,
            outputs[0],
            (mReducedPrecisionIO == 0) ? InOutDataType::FLOAT : InOutDataType::HALF);
    ASSERT(res == 0);

    return 0;
}

size_t DLRMInteractionsPlugin::getSerializationSize() const
{
    return sizeof(int) * 9 + sizeof(float) * 3 + MAX_TABLE_OFFSETS * sizeof(int) + MAX_FILEPATH_LENGTH * 2;
}

void DLRMInteractionsPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char *>(buffer);
    const char *a = d;

    write(d, mTotalInteractionFeatures);
    write(d, mInScale);
    write(d, mOutScale);
    write(d, mEmbeddingSize);
    write(d, mEmbeddingRows);
    write(d, mReducedPrecisionIO);
    write(d, mEmbeddingWeightsOnGpuPart);
    write(d, mInteractionsOutputInterleaved);
    write(d, mOutputPaddingGranularity);

    ASSERT(mTableOffsets.size() <= MAX_TABLE_OFFSETS);
    write(d, (int)(mTableOffsets.size()));
    std::copy(mTableOffsets.data(), mTableOffsets.data() + mTableOffsets.size(), (int *)d);
    d += MAX_TABLE_OFFSETS * sizeof(int);

    ASSERT(mEmbeddingWeightsFilepath.size() <= MAX_FILEPATH_LENGTH);
    write(d, (int)(mEmbeddingWeightsFilepath.size()));
    std::copy(mEmbeddingWeightsFilepath.data(), mEmbeddingWeightsFilepath.data() + mEmbeddingWeightsFilepath.size(), d);
    d += MAX_FILEPATH_LENGTH;

    ASSERT(mRowFrequenciesFilepath.size() <= MAX_FILEPATH_LENGTH);
    write(d, (int)(mRowFrequenciesFilepath.size()));
    std::copy(mRowFrequenciesFilepath.data(), mRowFrequenciesFilepath.data() + mRowFrequenciesFilepath.size(), d);
    d += MAX_FILEPATH_LENGTH;

    ASSERT(d == a + getSerializationSize());
}

void DLRMInteractionsPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs)
{
    ASSERT(in && nbInputs == 2);
    ASSERT(out && nbOutputs == 1);

    ASSERT(in[0].desc.dims.nbDims >= 2); // Dense features

    ASSERT(in[1].desc.dims.nbDims == 2); // Sparse features

    mTotalInteractionFeatures = in[1].desc.dims.d[1] + 1;
    mInScale = in[0].desc.scale;
    mOutScale = out[0].desc.scale;
}

bool DLRMInteractionsPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);

    switch (pos)
    {
        case 0:
            if ((inOut[pos].format == TensorFormat::kLINEAR) && (mReducedPrecisionIO == 0))
                return (inOut[pos].type == DataType::kFLOAT);
            else if ((inOut[pos].format == TensorFormat::kLINEAR) && (mReducedPrecisionIO == 1))
                return (inOut[pos].type == DataType::kHALF);
            else if ((inOut[pos].format == TensorFormat::kCHW32) && (mReducedPrecisionIO == 2))
                return (inOut[pos].type == DataType::kINT8);
            break;
        case 1:
            return ((inOut[pos].format == TensorFormat::kLINEAR) && (inOut[pos].type == DataType::kINT32));
            break;
        case 2:
            return ((inOut[pos].format == inOut[0].format) && (inOut[pos].type == inOut[0].type));
            break;
    }

    return false;
}

DataType DLRMInteractionsPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const 
{
    ASSERT(nbInputs == 2);
    ASSERT(index == 0);

    return inputTypes[0];
}

const char* DLRMInteractionsPlugin::getPluginType() const { return DLRM_INTERACTIONS_PLUGIN_NAME; }

const char* DLRMInteractionsPlugin::getPluginVersion() const { return DLRM_INTERACTIONS_PLUGIN_VERSION; }

void DLRMInteractionsPlugin::destroy() { delete this; }

IPluginV2DynamicExt* DLRMInteractionsPlugin::clone() const
{
    IPluginV2DynamicExt* plugin = new DLRMInteractionsPlugin(*this);
    return plugin;
}

DLRMInteractionsPluginCreator::DLRMInteractionsPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("embeddingSize", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("embeddingRows", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("reducedPrecisionIO", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("embeddingWeightsOnGpuPart", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("interactionsOutputInterleaved", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("outputPaddingGranularity", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("reducedPrecisionIO", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("embeddingWeightsFilepath", nullptr, PluginFieldType::kCHAR));
    mPluginAttributes.emplace_back(PluginField("rowFrequenciesFilepath", nullptr, PluginFieldType::kCHAR));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DLRMInteractionsPluginCreator::getPluginName() const
{
    return DLRM_INTERACTIONS_PLUGIN_NAME;
}

const char* DLRMInteractionsPluginCreator::getPluginVersion() const
{
    return DLRM_INTERACTIONS_PLUGIN_VERSION;
}

const PluginFieldCollection* DLRMInteractionsPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* DLRMInteractionsPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;

    int embeddingSize = 0;
    int embeddingRows = 0;
    int reducedPrecisionIO = 0;
    float embeddingWeightsOnGpuPart = 0.0F;
    int interactionsOutputInterleaved = 0;
    std::vector<int> tableOffsets;
    std::string embeddingWeightsFilepath;
    std::string rowFrequenciesFilepath;
    int outputPaddingGranularity = 0;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "embeddingRows"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            embeddingRows = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "embeddingSize"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            embeddingSize = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "reducedPrecisionIO"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            reducedPrecisionIO = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "embeddingWeightsOnGpuPart"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            embeddingWeightsOnGpuPart = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "interactionsOutputInterleaved"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            interactionsOutputInterleaved = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "outputPaddingGranularity"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            outputPaddingGranularity = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "tableOffsets"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            tableOffsets.resize(fields[i].length);
            std::copy((const int *)(fields[i].data), (const int *)(fields[i].data) + fields[i].length, tableOffsets.data());
        }
        else if (!strcmp(attrName, "embeddingWeightsFilepath"))
        {
            ASSERT(fields[i].type == PluginFieldType::kCHAR);
            embeddingWeightsFilepath = std::string((const char *)(fields[i].data), fields[i].length);
        }
        else if (!strcmp(attrName, "rowFrequenciesFilepath"))
        {
            ASSERT(fields[i].type == PluginFieldType::kCHAR);
            rowFrequenciesFilepath = std::string((const char *)(fields[i].data), fields[i].length);
        }
    }

    return new DLRMInteractionsPlugin(
        embeddingSize,
        embeddingRows,
        reducedPrecisionIO,
        embeddingWeightsOnGpuPart,
        interactionsOutputInterleaved,
        outputPaddingGranularity,
        tableOffsets,
        embeddingWeightsFilepath,
        rowFrequenciesFilepath);
}

IPluginV2* DLRMInteractionsPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    return new DLRMInteractionsPlugin(serialData, serialLength);
}
