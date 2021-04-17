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


#ifndef __CALLBACK_HPP__
#define __CALLBACK_HPP__

#include <functional>
#include <map>
#include <iostream>

#include "query_sample_library.h"

void cocoCallback(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex> &sample_ids, size_t response_count)
{
    for (size_t i = 0; i < response_count; i++)
    {
        auto &r = responses[i];
        size_t size = r.size;
        size_t maxKeepCount = (size / sizeof(float) - 1) / 7;
        int32_t keepCount = *(reinterpret_cast<int32_t*>(r.data) + maxKeepCount * 7);
        r.size = static_cast<size_t>(keepCount * sizeof(float) * 7);
        float image_id = sample_ids[i];
        for (int32_t k = 0; k < keepCount; k++) {
          *(reinterpret_cast<float*>(r.data) + k * 7) = image_id;
        }
    }
}

// Call back for OVRN50 - need to run argmax on raw softmax tensor follwed by a shift
void ovrn50Callback(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex> &sample_ids, size_t response_count)
{
    const int32_t offset = 1;
    for (size_t i = 0; i < response_count; i++)
    {
        auto &r = responses[i];
        size_t size = r.size;
        const size_t numClasses = size / sizeof(float);

        float* softmaxValues = reinterpret_cast<float*>(r.data);
        float max = 0.f;
        int index = 0;
        for (int i = 0; i < numClasses; i++)
        {
            if (softmaxValues[i] > max)
            {
                max = softmaxValues[i];
                index = i;
            }
        }
        // Do background class offset shift
        index = index - offset;
        // Only save the argmax value
        *(reinterpret_cast<int32_t*>(r.data)) = index;
        r.size = 4;
    }
}

void ovcocoCallback(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex> &sample_ids, size_t response_count)
{
    int object_size_ = 4;
    int max_proposal_count_ = 200;

    for (size_t i = 0; i < response_count; i++)
    {
        auto &r = responses[i];
        size_t size = r.size;
        auto output = reinterpret_cast<float*>(r.data);
        size_t count = 0;
        float image_id = sample_ids[i];
        for (int curProposal = 0; curProposal < max_proposal_count_;
                curProposal++) {
            float confidence = output[7*curProposal+5];
            if (confidence > 0.05) {
                float label = output[7*curProposal+4];
                float xmin = output[7*curProposal+0];
                float ymin = output[7*curProposal+1];
                float xmax = output[7*curProposal+2];
                float ymax = output[7*curProposal+3];
                /** Add only objects with > 0.05 probability **/
                *(reinterpret_cast<float*>(r.data) + count*7 + 0 ) = image_id;
                *(reinterpret_cast<float*>(r.data) + count*7 + 1 ) = ymin;
                *(reinterpret_cast<float*>(r.data) + count*7 + 2 ) = xmin;
                *(reinterpret_cast<float*>(r.data) + count*7 + 3 ) = ymax;
                *(reinterpret_cast<float*>(r.data) + count*7 + 4 ) = xmax;
                *(reinterpret_cast<float*>(r.data) + count*7 + 5 ) = confidence;
                *(reinterpret_cast<float*>(r.data) + count*7 + 6 ) = label;

                ++count;
            }
        }
        r.size = static_cast<size_t>(sizeof(float)*7*count);
    }
}

/* Define a map for post-processing callback functions */
std::map<std::string, std::function<void(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex> &sample_ids, size_t response_count)>> callbackMap = {
    {"", nullptr},
    {"coco", cocoCallback},
    {"ovrn50", ovrn50Callback},
    {"ovcoco", ovcocoCallback}
};

#endif /* __CALLBACK_HPP__ */
