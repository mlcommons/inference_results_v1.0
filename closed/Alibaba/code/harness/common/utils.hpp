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

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "NvInfer.h"

#include <numaif.h>
#include <pthread.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

/* Helper function to split a string based on a delimiting character */
inline std::vector<std::string> splitString(const std::string& input, const std::string& delimiter)
{
    std::vector<std::string> result;
    size_t start = 0;
    size_t next = 0;
    while(next != std::string::npos)
    {
        next = input.find(delimiter, start);
        result.emplace_back(input, start, next - start);
        start = next + 1;
    }
    return result;
}

// Get element size of a data type.
inline unsigned int getElementSize(nvinfer1::DataType t) {
    switch (t) {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    // Fall through to error
    default: break;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

// Return m rounded up to nearest multiple of n
inline int roundUp(int m, int n)
{
    return ((m + n - 1) / n) * n;
}

// Get the size of a binding dimensions
inline int64_t volume(const nvinfer1::Dims& d, const nvinfer1::TensorFormat& format, const bool hasImplicitBatch = false)
{
    nvinfer1::Dims d_new = d;
    // Get number of scalars per vector.
    int spv{1};
    switch(format)
    {
        case nvinfer1::TensorFormat::kCHW2: spv = 2; break;
        case nvinfer1::TensorFormat::kCHW4: spv = 4; break;
        case nvinfer1::TensorFormat::kHWC8: spv = 8; break;
        case nvinfer1::TensorFormat::kCHW16: spv = 16; break;
        case nvinfer1::TensorFormat::kCHW32: spv = 32; break;
        case nvinfer1::TensorFormat::kLINEAR:
        default: spv = 1; break;
    }
    if (spv > 1)
    {
        assert(d.nbDims >= 3); // Vectorized format only makes sense when nbDims>=3.
        d_new.d[d_new.nbDims - 3] = roundUp(d_new.d[d_new.nbDims - 3], spv);
    }
    // Skip the first dimension, which is batch dim.
    return std::accumulate(d_new.d + (hasImplicitBatch ? 0 : 1), d_new.d + d_new.nbDims, 1, std::multiplies<int64_t>());
}

inline int64_t volume(const nvinfer1::Dims& d, const bool hasImplicitBatch = false)
{
    return volume(d, nvinfer1::TensorFormat::kLINEAR, hasImplicitBatch);
}

// Create a shared pointer of an nvinfer1:: object which will be automatically destroyed when going out of scope.
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
inline std::shared_ptr<T> InferObject(T* obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, InferDeleter());
}

// NUMA config. Each NUMA node contains a pair of GPU indices and CPU indices.
using NumaConfig = std::vector<std::pair<std::vector<int>, std::vector<int>>>;

// The NUMA node idx for each GPU.
using GpuToNumaMap = std::vector<int>;

// Helper to converts the range string (like "0,2-5,13-17") to a vector of ints.
inline std::vector<int> parseRange(const std::string& s)
{
    std::vector<int> results;
    auto ranges = splitString(s, ",");
    for (const auto& range : ranges)
    {
        auto startEnd = splitString(range, "-");
        CHECK(startEnd.size() <= 2) << "Invalid numa_config setting. Expects zero or one '-'.";
        if (startEnd.size() == 1)
        {
            results.push_back(std::stoi(startEnd[0]));
        }
        else
        {
            int start = std::stoi(startEnd[0]);
            int last = std::stoi(startEnd[1]);
            for (int i = start; i <= last; ++i)
            {
                results.push_back(i);
            }
        }
    }
    return results;
}

// Example of the format: "0,2:0-63&1,3:64-127" for 4 GPUs, 128 CPU, 2 NUMA node system.
inline NumaConfig parseNumaConfig(const std::string& s)
{
    NumaConfig config;
    if (!s.empty())
    {
        auto nodes = splitString(s, "&");
        for (const auto& node : nodes)
        {
            auto pair = splitString(node, ":");
            CHECK(pair.size() == 2) << "Invalid numa_config setting. Expects one ':'.";
            auto gpus = parseRange(pair[0]);
            auto cpus = parseRange(pair[1]);
            config.emplace_back(std::make_pair(gpus, cpus));
        }
    }
    return config;
}

// Convert NumaConfig to GpuToNumaMap for easier look-up.
inline GpuToNumaMap getGpuToNumaMap(const NumaConfig& config)
{
    std::vector<int> map;
    for (int32_t numaIdx = 0; numaIdx < config.size(); numaIdx++)
    {
        for (const auto gpuIdx : config[numaIdx].first)
        {
            if (gpuIdx >= map.size())
            {
                map.resize(gpuIdx + 1);
            }
            map[gpuIdx] = numaIdx;
        }
    }
    return map;
}

// Restrict mem allocation to specific NUMA node.
inline void bindNumaMemPolicy(const int32_t numaIdx, const int32_t nbNumas)
{
    unsigned long nodeMask = 1UL << numaIdx;
    long ret = set_mempolicy(MPOL_BIND, &nodeMask, nbNumas + 1);
    CHECK(ret >= 0) << strerror(errno);
}

// Reset mem allocation setting.
inline void resetNumaMemPolicy()
{
    long ret = set_mempolicy(MPOL_DEFAULT, nullptr, 0);
    CHECK(ret >= 0) << strerror(errno);
}

// Limit a thread to be on specific cpus.
inline void bindThreadToCpus(std::thread& th, const std::vector<int>& cpus)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for(int cpu : cpus)
    {
        CPU_SET(cpu, &cpuset);
    }
    CHECK_EQ(pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset), 0);
}

#endif // __UTILS_HPP__
