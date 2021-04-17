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

#include <vector>

#include <cub/cub.cuh> 

#include "ssdOpt.h"
#include "ssdOptMacros.h"

namespace nvinfer1
{
namespace plugin
{

template <typename T_BBOX, typename T_SCORE, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
    __global__ void gatherTopDetectionsOpt_kernel(
        const bool shareLocation,
        const int numImages,
        const int numPredsPerClass,
        const int numClasses,
        const int topK,
        const int keepTopK,
        const int* indices,
        const T_SCORE* scores,
        const T_BBOX* bboxData,
        T_BBOX* topDetections)
{
    typedef cub::BlockReduce<int, nthds_per_cta> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    assert(keepTopK <= topK);

    const int imgId = blockIdx.x;
    const int imgBase = imgId * (7 * keepTopK + 1);
    const int offset = imgId * numClasses * topK;
    const int bboxOffset = imgId * (shareLocation ? numPredsPerClass : (numClasses * numPredsPerClass));

    int isValid = 0;
    int aggregate = 0;

    int finish = ((keepTopK + nthds_per_cta - 1) / nthds_per_cta) * nthds_per_cta;
    for (int detId = threadIdx.x; detId < finish; detId += nthds_per_cta)
    {
        if (detId < keepTopK) {
            const int index = indices[offset + detId];
            const T_SCORE score = scores[offset + detId];

            isValid = (index == -1)? 0 : 1;

            const int bboxId = ((shareLocation ? (index % numPredsPerClass)
                        : index % (numClasses * numPredsPerClass)) + bboxOffset) * 4;
            topDetections[imgBase + detId * 7] = imgId;                                                    // image id
            // clipped bbox ymin
            topDetections[imgBase + detId * 7 + 1] = (isValid)? max(min(bboxData[bboxId + 1], T_BBOX(1.)), T_BBOX(0.)) : 0;
            // clipped bbox xmin
            topDetections[imgBase + detId * 7 + 2] = (isValid)? max(min(bboxData[bboxId], T_BBOX(1.)), T_BBOX(0.)) : 0;
            // clipped bbox ymax
            topDetections[imgBase + detId * 7 + 3] = (isValid)? max(min(bboxData[bboxId + 3], T_BBOX(1.)), T_BBOX(0.)) : 0;
            // clipped bbox xmax
            topDetections[imgBase + detId * 7 + 4] = (isValid)? max(min(bboxData[bboxId + 2], T_BBOX(1.)), T_BBOX(0.)) : 0;
            topDetections[imgBase + detId * 7 + 5] = (isValid)? score : 0;                               // confidence score
            topDetections[imgBase + detId * 7 + 6] = (isValid)? (index % (numClasses * numPredsPerClass)) / numPredsPerClass : -1; // label
        } else {
            isValid = 0;
        }

        aggregate += BlockReduce(temp_storage).Reduce(isValid, cub::Sum());
    }
    if (threadIdx.x == 0)
    {
        ((int*) topDetections)[imgBase + 7 * keepTopK] = aggregate;
    }
}

template <typename T_BBOX, typename T_SCORE>
ssdStatus_t gatherTopDetectionsOpt_gpu(
    cudaStream_t stream,
    const bool shareLocation,
    const int numImages,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const void* indices,
    const void* scores,
    const void* bboxData,
    void* topDetections)
{
    cudaMemsetAsync(topDetections, 0, numImages * (7 * keepTopK + 1) * sizeof(float), stream);
    const int BS = 128;
    int GS = numImages;
    gatherTopDetectionsOpt_kernel<T_BBOX, T_SCORE, BS><<<GS, BS, 0, stream>>>(shareLocation, numImages, numPredsPerClass,
                                                                           numClasses, topK, keepTopK,
                                                                           (int*) indices, (T_SCORE*) scores, (T_BBOX*) bboxData,
                                                                           /*(int*) keepCount,*/ (T_BBOX*) topDetections);

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// gatherTopDetectionsOpt LAUNCH CONFIG {{{
typedef ssdStatus_t (*gtdFunc)(cudaStream_t,
                               const bool,
                               const int,
                               const int,
                               const int,
                               const int,
                               const int,
                               const void*,
                               const void*,
                               const void*,
                               void*);
struct gtdLaunchConfig
{
    DType_t t_bbox;
    DType_t t_score;
    gtdFunc function;

    gtdLaunchConfig(DType_t t_bbox, DType_t t_score)
        : t_bbox(t_bbox)
        , t_score(t_score)
    {
    }
    gtdLaunchConfig(DType_t t_bbox, DType_t t_score, gtdFunc function)
        : t_bbox(t_bbox)
        , t_score(t_score)
        , function(function)
    {
    }
    bool operator==(const gtdLaunchConfig& other)
    {
        return t_bbox == other.t_bbox && t_score == other.t_score;
    }
};

using nvinfer1::DataType;

static std::vector<gtdLaunchConfig> gtdFuncVec;

bool gtdOptInit()
{
    gtdFuncVec.push_back(gtdLaunchConfig(DataType::kFLOAT, DataType::kFLOAT,
                                         gatherTopDetectionsOpt_gpu<float, float>));
    return true;
}

static bool initialized = gtdOptInit();

//}}}

ssdStatus_t gatherTopDetectionsOpt(
    cudaStream_t stream,
    const bool shareLocation,
    const int numImages,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const DType_t DT_BBOX,
    const DType_t DT_SCORE,
    const void* indices,
    const void* scores,
    const void* bboxData,
    void* topDetections)
{
    gtdLaunchConfig lc = gtdLaunchConfig(DT_BBOX, DT_SCORE);
    for (unsigned i = 0; i < gtdFuncVec.size(); ++i)
    {
        if (lc == gtdFuncVec[i])
        {
            DEBUG_PRINTF("gatherTopDetectionsOpt kernel %d\n", i);
            return gtdFuncVec[i].function(stream,
                                          shareLocation,
                                          numImages,
                                          numPredsPerClass,
                                          numClasses,
                                          topK,
                                          keepTopK,
                                          indices,
                                          scores,
                                          bboxData,
                                          //keepCount,
                                          topDetections);
        }
    }
    return STATUS_BAD_PARAM;
}

} // namespace plugin
} // namespace nvinfer1
