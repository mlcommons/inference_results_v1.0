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

// General C++
#include <algorithm>
#include <functional> // greater
#include <list> // BusyList
#include <queue> // FreeQueue
#include <stdexcept>
#include <unordered_map> // WorkMap

// Project Includes
#include "qsl.hpp" // For QuerySample
#include "preprocessing.hpp" // For DALI tracking (just sampleAttributes as of now)

#define RETURN_T uint8_t       // type of data we want to return to loadgen (actual LUT indices instead of ascii)


struct ElementMetadata {
    // Tracks details of a given batch element
    size_t batchIdx;
    size_t processedStepSize;
    size_t daliBufIdx;
    void *devDataPtr;
    void *devSeqLenPtr;
    int32_t sampleSeqLen{0};
    int32_t splitOffset{0};
    int32_t encoder_amount_processed{0};
    int32_t numberOfTimesProcessed{0}; // Seems like useful debug info
    bool persisted_in_encoder{false};
    bool persisted_in_decoder{false};
    mlperf::QuerySample query_sample{};

    // Maybe consider rewriting this as a constructor (currently, items are copied which maintains batchIdx)
    // and other fields are set
    void reassign_to_encoder(const mlperf::QuerySample &sample, const size_t &seqLen) {
        sampleSeqLen = seqLen;
        splitOffset = 0;
        numberOfTimesProcessed = 0;
        query_sample = sample;
        encoder_amount_processed = 0;
        persisted_in_encoder = false;
    }
    // Nice little comparator to facilitate a priority queue
    bool operator >(const ElementMetadata &other) const {
        return batchIdx > other.batchIdx;
    }

};

// Helper for DRY
// Gets a list of iterators whose pointed-to elements satisfy a predicate
template <class Iter, class Pred>
auto get_matching_iters(const Iter &start, const Iter &end, Pred pred) {
    std::vector<Iter> ready_list;
    for (auto it = start; it != end; ++it) {
        if (pred(*it)) ready_list.push_back(it);
    }
    return ready_list;
}

// Forward declare all subsequent states for friend functions!
class DecoderState;

class EncoderState {
    size_t mBatchSize;
    size_t mMaxSeqLen; // Model max sequence length
    std::list<ElementMetadata> mBusyList;

    // Need to state all template parameters because we want to override comparator
    // We use a priority queue because _in theory_ we want the least batch index when we make any allocation
    // (To ensure a minimal value of actualBatchSize)
    // If we use a priority queue we keep the number of cycles for ordering pretty and spread out rather than
    // having a single "stop the world and sort" phase baked into state transfer.
    std::priority_queue<ElementMetadata,
                        std::vector<ElementMetadata>,
                        std::greater<ElementMetadata>> mFreeQueue;

public:

    friend auto transfer_state_from_encoder_to_decoder(EncoderState &es, DecoderState &ds);


    static bool is_element_done(const ElementMetadata &el) {
        return el.encoder_amount_processed > el.sampleSeqLen;
    }
    EncoderState(size_t batchSize, size_t maxSeqLen)
        : mBatchSize(batchSize)
        , mMaxSeqLen(maxSeqLen){
        for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            ElementMetadata em{.batchIdx=batchIdx, .processedStepSize=maxSeqLen};
            mFreeQueue.push(em);
        }
    }

    // Basic accessors
    bool is_full() const { return mFreeQueue.empty(); }
    size_t num_available() const { return mFreeQueue.size(); }
    size_t size() const {return mBusyList.size();}
    size_t max_idx_rng() const {
        auto idx = std::max_element(mBusyList.cbegin(), mBusyList.cend(),
                                    [](const auto& a, const auto& b) {
                                        return a.batchIdx < b.batchIdx;
                                    })->batchIdx;
        return size() == 0 ? 0 : idx+1;
    }

    //Barebones iterator provisioning (cbegin, cend)
    auto begin() const {
        return mBusyList.cbegin();
    }
    auto end() const {
        return mBusyList.cend();
    }

    // Temporary!
    auto get_offset_for_dali(const mlperf::QuerySample &qs) const {
        auto list = get_matching_iters(mBusyList.begin(), mBusyList.end(),
                                       [qs](const auto &el) {
                                           return el.query_sample.id == qs.id;
                                       });
        assert(list.size() == 1);
        return list[0]->splitOffset;
    }

    void allocate(const mlperf::QuerySample& sample, const size_t &sampleSeqLen) {
        if (is_full()) throw std::runtime_error("EncoderState is full but allocation was attempted");

        // Copy the queue element (maybe look into moving)
        mBusyList.push_back(mFreeQueue.top());
        // Pop it from the queue (destructs queue's copy)
        mFreeQueue.pop();
        // Assign it to the new data (maybe should be more idiosyncratic and use a constructor maybe)
        mBusyList.back().reassign_to_encoder(sample, sampleSeqLen);
    }

    void allocate(const sampleAttributes &ds) {
        allocate(ds.querySample, ds.hostSeqLen);
        auto &back = mBusyList.back();
        back.daliBufIdx = ds.daliBufIdx;
        back.devDataPtr = ds.devDataPtr;
        back.devSeqLenPtr = ds.devSeqLenPtr;
    }

    auto get_state_mask() const{
        auto to_ret = std::make_unique<bool[]>(mBatchSize);
        // Probably a very silly optimization: figure out if more persisted than completed from some heuristic to make filling not take as long?
        std::fill(to_ret.get(), to_ret.get() + mBatchSize, false);
        for (const auto& el: mBusyList) {
            // This isn't a very cache-friendly pattern
            to_ret[el.batchIdx] = el.persisted_in_encoder;
        }
        return to_ret;
    }
private:
    void update() {
        for (auto &el: mBusyList) {
            if (is_element_done(el)) {
                // Should never get here because transfer to decoder should have happened before our most recent call to update (this function)
                throw std::runtime_error("State invariants violated inside of EncoderState's update()");
            }

            // Otherwise, do a normal update
            el.persisted_in_encoder = true;
            ++el.numberOfTimesProcessed;
            el.splitOffset += el.processedStepSize;
            el.encoder_amount_processed += el.processedStepSize;
        }
    }
    // Returns daliIdxs of finished elements
    auto deallocate_finished() {
        std::vector<size_t> to_ret;
        for (const auto &it : get_matching_iters(mBusyList.begin(), mBusyList.end(),
                                                 [](const auto &el) {return is_element_done(el);})) {
            // Save metadata for returning
            to_ret.push_back(it->daliBufIdx);
            // Copyto free queue (maybe use a move, because this constructs)
            mFreeQueue.push(*it);
            // Remove from busy list (deletes its copy)
            mBusyList.erase(it);
        }
        return to_ret;
    }

};
class DecoderState {
    size_t mBatchSize;
    // No need to track free list because decoder can only be as busy as encoder
    std::unordered_map<mlperf::ResponseId, std::tuple<ElementMetadata, std::vector<RETURN_T>>> mWorkMap;
public:
    friend auto transfer_state_from_encoder_to_decoder(EncoderState &es, DecoderState &ds);

    DecoderState(size_t batchSize)
        : mBatchSize(batchSize) {
        // Maybe store textmap separately so we can do simpler buffer reuse, ala:
        // for (size_t i = 0; i < batchSize; ++i) {
        //     mTextMap[i].reserve(200);
        // }
    }
    auto size() const { return mWorkMap.size();}
    auto get_state_mask() const{
        auto to_ret = std::make_unique<bool[]>(mBatchSize);
        std::fill(to_ret.get(), to_ret.get() + mBatchSize, false);
        for (const auto& [resp_id, value]: mWorkMap) {
            auto &[el, str] = value;
            to_ret[el.batchIdx] = el.persisted_in_decoder;
        }
        return to_ret;
    }
    void update_text_batch(const std::vector<std::vector<RETURN_T>> &batch) {
        for (auto &[resp_id, value]: mWorkMap) {
            auto &[el, idx_vec] = value;
            auto batch_el = batch.at(el.batchIdx);
            idx_vec.insert( idx_vec.end(), batch_el.begin(), batch_el.end() );
        }
    }
    // For debugging
    std::vector<std::pair<size_t,size_t>> get_active_idxs() const {
        std::vector<std::pair<size_t, size_t>> to_ret;
        for (const auto &it : mWorkMap) {
            auto &[resp_id, value] = it;
            auto &[el, str] = value;
            to_ret.push_back(std::make_pair(resp_id, el.batchIdx));
        }
        std::sort(to_ret.begin(), to_ret.end(), [](const auto& a, const auto& b) {return a.second < b.second;});
        return to_ret;
    }
    template <class SendFunc>
    void send_responses_and_clear(SendFunc sf) {
        mlperf::QuerySampleResponse response;
        auto done_iter_list = get_matching_iters(mWorkMap.begin(), mWorkMap.end(),
                                                 [](const auto &pair) {
                                                     auto &[resp_id, value] = pair;
                                                     auto &[el, str] = value;
                                                     return EncoderState::is_element_done(el);
                                                 });
        for (const auto &it : done_iter_list) {
            auto &[resp_id, value] = *it;
            auto &[el, str] = value;
            response.id = resp_id;
            response.data = reinterpret_cast<uintptr_t>(str.data());
            response.size = str.size() * sizeof(RETURN_T);
            sf(&response, 1); // Send Response
            mWorkMap.erase(it);
        }
    }
private:
    void update(const ElementMetadata &updating_em) {
        auto &[el, str] = mWorkMap.at(updating_em.query_sample.id);
        el = updating_em;

        // Do state-specific overrides here:
        el.persisted_in_decoder = true;
    }
    void allocate(const ElementMetadata &em) {
        // Maybe use pool allocator
        auto &key = em.query_sample.id;
        std::vector<RETURN_T> empty_idxvec;
        auto [it, was_inserted] = mWorkMap.insert(std::make_pair(key, std::make_tuple(em, empty_idxvec)));
        if (!was_inserted) {
            throw std::runtime_error("Inserted existing element when allocating for DecoderState");
        }
        // State-specific overrides go here:
        auto &[k, value] = *it;
        auto &[el, str] = value;
        // Note persistence:
        el.persisted_in_decoder = false;
    }
    bool hasId(mlperf::ResponseId id) {
        return mWorkMap.find(id) != mWorkMap.end();
    }
};

// Serves as a "clock-edge" update where all encoder state is updated,
// then transferred to the decoder
auto transfer_state_from_encoder_to_decoder(EncoderState &es, DecoderState &ds) {
    // We explicitly update encoder state because this is called immediately after the encoder runs.
    es.update();
    // For now, let's just move everything in the EncoderState (if it's not there already)
    for (auto &el: es) {

        if (ds.hasId(el.query_sample.id)) {
            // We're updating an existing sample
            ds.update(el);
        } else {
            // We need to allocate this in the decoder
            ds.allocate(el);
        }
    }
    return es.deallocate_finished();
}
