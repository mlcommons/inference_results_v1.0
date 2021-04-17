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
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <variant>
#include <vector>
#include "qsl.hpp"

DECLARE_uint64(max_overall_seq_length);

class WarmupSampleLibrary : public qsl::LookupableQuerySampleLibrary {
    struct InternalSample {
        float *host_data{nullptr};
        int length{-1};
        bool is_allocated() const {return host_data != nullptr && length != -1;}
    };

    // Below are observations if we were to use random arrays, but we can just used fixed, max length instead
    // Raw samples are 240,000 samples long of dtype float32 with Peak-To-Peak ranges around [0.5, 1.5]. Mean of zero, and std-dev in a rough range of [0.02, 0.04]. Length ranges in dev clean are empirically [23120,239920], but let's generalize to [20000, 240000]
    // We'll generate random noise with std-dev
    // Use random_device for random seeding if we care.
    size_t mSampleCount;
    std::string mName;
    std::default_random_engine mRandGen;
    std::uniform_real_distribution<float> mStdDist;
    std::uniform_int_distribution<int> mLenDist;
    std::vector<InternalSample> mSampleData;
    bool mEnableAudioProcessing {false};
    static constexpr auto nbElemsPerSample = 240000;
    static constexpr auto memSizeToAlloc = nbElemsPerSample*sizeof(float);
public:
    WarmupSampleLibrary(size_t sampleCount, bool enableAudioProcessing)
        : mSampleCount(sampleCount)
        , mName("Warmup Sample Library of Size: " + std::to_string(sampleCount))
        , mRandGen(0)
        , mStdDist(0.02, 0.04)
        , mLenDist(20000, 240000)
        , mSampleData(sampleCount)
        , mEnableAudioProcessing(enableAudioProcessing) {}
    const std::string& Name() const final {
        return mName;
    }
    size_t TotalSampleCount() final { return mSampleCount; }
    // Should maybe correct this?
    size_t PerformanceSampleCount() final { return mSampleCount; }
    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) final {
        // Check to make sure that we can satisfy this request:
        CHECK(*std::max_element(samples.begin(), samples.end()) < mSampleCount);
        // For each element, alloc some data
        for (const auto &idx : samples) {
            auto& data = mSampleData[idx];
            // Make sure we haven't already allocated (maybe just continue?)
            CHECK(!data.is_allocated());
            CHECK_EQ(cudaMallocHost(&data.host_data, memSizeToAlloc), cudaSuccess);
            //randomGenerate(idx);
            constantGenerate(idx);
        }
    }
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex> &samples) final {
        // Unlike SampleLibrary, we're gonna actually destroy these samples:
        for (const auto &idx: samples) {
            auto &data = mSampleData[idx];
            // Make sure we're unregistering a valid sample:
            CHECK(data.is_allocated());
            // Now we return to uninitialized state
            cudaFreeHost(data.host_data);
            data.host_data = nullptr;
            data.length = -1;
        }
    }
    size_t GetSampleSize([[maybe_unused]]size_t input_idx) const final {
        // Unused param to keep consistent interface with SampleLibrary
        return memSizeToAlloc;
    }
    void* GetSampleAddress(mlperf::QuerySampleIndex sample_index, size_t input_idx, [[maybe_unused]]size_t device_idx=0) final {
        // We only support input indices of 0 (data) and 1 (size)
        CHECK(input_idx == 0 || input_idx == 1);
        // Bounds and allocation checking
        CHECK(sample_index < mSampleData.size());
        auto &data = mSampleData[sample_index];
        CHECK(data.is_allocated());

        return input_idx == 0 ? static_cast<void*>(data.host_data) : static_cast<void*>(&data.length);
    }
    ~WarmupSampleLibrary() {
        // Make sure we've deallocated all outstanding pointers, and scream if we didn't:
        for (const auto& data : mSampleData) {
            CHECK(!data.is_allocated());
        }
    }

private:
    void randomGenerate(size_t idx) {
        auto &data = mSampleData[idx];
        // For consistency, but check results to see if they're "random enough"
        mRandGen.seed(idx);
        data.length = mLenDist(mRandGen);
        auto range_begin = data.host_data;
        auto range_end = range_begin + data.length;
        std::normal_distribution normal_dist(0.f, mStdDist(mRandGen));
        std::generate(range_begin, range_end,
                      [&gen=mRandGen, &dist=normal_dist]() {
                          return dist(gen);
                      });
    }
    void constantGenerate(size_t idx) {
        auto &data = mSampleData[idx];
        data.length = FLAGS_max_overall_seq_length;
        auto range_begin = data.host_data;
        auto range_end = range_begin + data.length;
        std::fill(range_begin, range_end, 0.5f);
    }
};



class WarmupManager {
    // Resource manager for warmup samples using QSL instead of dummy data.
    // Usage: Construct, use query_to_send, and let automatically delete. Lifetime of query_to_send should not exceed this object.
public:
    // Only going to handle one ctor interface
    WarmupManager(const WarmupManager &) = delete;
    WarmupManager& operator=(const WarmupManager &) = delete;
    //Move assignment/ctor already deleted due to user-provided dtor

    WarmupManager(std::shared_ptr<WarmupSampleLibrary> qsl)
        : mNumCompleted(0)
        , mNumToComplete(qsl->TotalSampleCount())
        , mWqsl(qsl)
        , mSampleIdxs(qsl->TotalSampleCount()){

        // Uses sample indices [0, num_samples).
        std::iota(mSampleIdxs.begin(), mSampleIdxs.end(), 0);

        for (size_t i = 0; i < mNumToComplete; ++i) {
            // Get our querySample index in a satisfiable range
            auto qsi = i % mSampleIdxs.size();

            // Our harness DOES use response ID for book keeping, so we need to keep them unique
            // (Instead of instanced object-tagging like in loadgen's QueryMetadata,
            //  we use pointer to member functions.

            // This is non-trivial! We're constructing a function inside of a struct in this vector (which will now be somewhere random in memory)
            // We have "dummy_callbacks" as storage which is guaranteed to persist for lifetime of WarmupManager
            dummy_callbacks.push_back({[this](){increment_completion();}, qsi});
        }
        for (auto &s : dummy_callbacks){
            // This is also non-trivial! We're taking a pointer to that function-containing-struct (which is in a vector.)
            // This is a roundabout way of getting both a callback function AND a unique ID.
            query_to_send.push_back({reinterpret_cast<mlperf::ResponseId>(&s), s.qsi});
        }
        mWqsl->LoadSamplesToRam(mSampleIdxs);
    }
    ~WarmupManager() {
        mWqsl->UnloadSamplesFromRam(mSampleIdxs);
    }

    void increment_completion() {
        std::unique_lock l(mCountMutex);
        ++mNumCompleted;
        //std::cout << mNumCompleted << std::endl;
        if (is_done()) {
            mCV.notify_one();
        }
    }
    void sleep_until_done() {
        std::unique_lock l(mCountMutex);
        mCV.wait(l, [this] {return is_done();});
    }

    // Identical interface to mlperf::QuerySamplesComplete
    static void warmupSamplesComplete(mlperf::QuerySampleResponse* responses, size_t response_count) {
        // By having this function be static, we can call it as a global function (ie our inference framework doesn't need to be told about any particular object).
        const mlperf::QuerySampleResponse* end = responses + response_count;
        for (mlperf::QuerySampleResponse* response = responses; response < end; response++) {
            // Wacky stuff: we take our function pointer which is disguised as a response ID, dereference it...
            auto callback = reinterpret_cast<CallbackStorage*>(response->id);
            // ... and then call it. (lol type safety)
            callback->f();
        }
    }
private:
    bool is_done() {
        return mNumCompleted == mNumToComplete;
    }

    std::recursive_mutex mCountMutex;
    std::condition_variable_any mCV;
    std::atomic<size_t> mNumCompleted; // Atomic to be safe, and performance isn't critical in warmup
    size_t mNumToComplete;
    std::shared_ptr<WarmupSampleLibrary> mWqsl;
    std::vector<mlperf::QuerySampleIndex> mSampleIdxs;
    struct CallbackStorage {
        std::function<void()> f;
        mlperf::QuerySampleIndex qsi;
    };
    std::vector<CallbackStorage> dummy_callbacks;
public:
    std::vector<mlperf::QuerySample> query_to_send;
};
