/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include <map>


struct CudaGraphPair {
    cudaGraph_t graph;
    cudaGraphExec_t execGraph;
    // TODO: Maybe add destructors? This means we need to be MUCH more careful about copying
};
struct CudaGraphCache {
    void put(size_t batch_size, CudaGraphPair cg_pair) {
        // We should only be registering as a first time, never overriding:
        auto existing_el = mCache.find(batch_size);
        CHECK(existing_el == mCache.end());
        gLogVerbose << "Registering graph of BS: " << batch_size << std::endl;
        mCache[batch_size] = cg_pair;
        mUsage[batch_size] = 0;
    }
    cudaGraphExec_t get(size_t requested_batch_size) {
        auto it = mCache.lower_bound(requested_batch_size);
        CHECK(it != mCache.end()); // We're asking for a batch bigger than however much we've stored
        auto &[registered_batch_size, cg_pair] = *it;
        ++mUsage[registered_batch_size];
        return cg_pair.execGraph;
    }

    // The below mimics Legacy caching behavior:
    template<typename FuncT>
    cudaGraphExec_t get(size_t requested_batch_size, FuncT miss_callback) {
        auto it = mCache.find(requested_batch_size);
        if (it == mCache.end()) {
            // Clear the cache
            for (auto &[bs, cg_pair] : mCache) {
                CHECK_EQ(cudaGraphDestroy(cg_pair.graph), cudaSuccess);
                CHECK_EQ(cudaGraphExecDestroy(cg_pair.execGraph), cudaSuccess);
            }
            mCache.clear();
            // Put new entry inside
            CudaGraphPair cg_pair = miss_callback(requested_batch_size);
            put(requested_batch_size, cg_pair);
            // "Optimized" return
            return cg_pair.execGraph;
        } else {
            auto &[registered_batch_size, cg_pair] = *it;
            return cg_pair.execGraph;
        }
    }
    ~CudaGraphCache() {
        gLogVerbose << "Usage for cuda graph cache is as follows:" << std::endl;
        for (const auto &[bs, use_count]: mUsage) {
            gLogVerbose << "\tBS: " << bs << " Use: " << use_count << std::endl;
        }
    }
private:
    std::map<size_t, CudaGraphPair> mCache;
    std::map<size_t, size_t> mUsage;
    // Could add more complex usage tracking like tracking the average "wasted work" by giving too big a graph.
};
