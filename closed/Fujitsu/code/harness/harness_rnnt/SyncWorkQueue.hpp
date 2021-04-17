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

#include <deque>
#include <mutex>
#include "qsl.hpp"

// =============================
//     SyncWorkQueue
// =============================
//
class SyncWorkQueue
{
public:
    SyncWorkQueue() = default;
    ~SyncWorkQueue() = default;

    void getBatch(std::vector<mlperf::QuerySample>& batch, size_t maxBatchSize)
    {
        nvtxRangeId_t nvtxGetBatch;
        std::unique_lock<std::mutex> l(mMutex);
        size_t actualBatchSize = std::min(mWorkQueue.size(), maxBatchSize);
        if(actualBatchSize > 0){
            NVTX_START_WITH_PAYLOAD(nvtxGetBatch, "SyncWorkQueue::GetBatch", COLOR_YELLOW_7, mWorkQueue.size());
            batch.resize(actualBatchSize);
            for (int i = 0; i < actualBatchSize; ++i) {
                batch[i] = mWorkQueue.front();
                mWorkQueue.pop_front();
            }
            NVTX_END(nvtxGetBatch);
        }
    }
    
    void insertItems(const std::vector<mlperf::QuerySample>& items)
    {
        std::unique_lock<std::mutex> l(mMutex);
        for (auto item : items) {
            mWorkQueue.emplace_back(item);
        }
    }

private:
    std::mutex mMutex;
    std::deque<mlperf::QuerySample> mWorkQueue;
};
