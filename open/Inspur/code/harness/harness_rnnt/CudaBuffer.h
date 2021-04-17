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

// TODO: Replace with include guard.
#pragma once
#include "cuda_runtime_api.h"
#include <cassert>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <vector>
#include "rnnt_kernels.h"

enum ManagedBufferLocation{CUDA, HOST_PINNED};
/* Helper for clean static asserts:
 * Discarded statements (unused statements in "if constexpr") must be well-formed.
 * static_assert(false) is not well formed.
 * BUT, we can make it well-formed if it's dependent on a template parameter (Loc used as dummy parameter)
 * For example:
 * if constexpr (CONDITION_HERE) {
 *     // Do some stuff here
 * } else {
 *     // Pretend this is illegal!
 *     static_assert(constexpr_false<DummyTemplateParam>, "This is illegal");
 * }
 *
 */
template <class...> constexpr std::false_type constexpr_false{};

template <typename ValueT, ManagedBufferLocation Loc>
class ManagedBuffer
{
public:
    // Should rearange for proper delegating constructors, but this works
    ManagedBuffer(size_t n_batches, size_t batch_stride)
        : ManagedBuffer(n_batches*batch_stride) {
        // The specialization of uint8_t is a special case where we have some arbitrary "blob" of size
        // batch_stride. This changes the behavior of certain methods below.
        static_assert(std::is_same<ValueT, uint8_t>::value, "Can only use 2 arg constructor for *BufferRaw");
        mStride = batch_stride;
        mNbBatches = n_batches;
    }


    //!
    //! \brief Allocate a buffer of size nbElems * sizeof(T) on Cuda enabled device.
    //!
    ManagedBuffer(size_t nbElems = 0)
        : mNbElems(nbElems) {

        if (nbElems) {
            if constexpr (Loc == ManagedBufferLocation::CUDA) {
                CHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&mData), nbElems * sizeof(ValueT)), cudaSuccess);
            } else if constexpr (Loc == ManagedBufferLocation::HOST_PINNED) {
                CHECK_EQ(cudaMallocHost(reinterpret_cast<void**>(&mData), nbElems * sizeof(ValueT)), cudaSuccess);
            } else {
                static_assert(constexpr_false<Loc>, "Unimplemented ManagedBufferLocation for Managed Buffer");
            }

            if (mData == nullptr)
                throw std::runtime_error("ManagedBuffer Allocation Failed.");
        }
        // Set mStride if needed
        mStride = sizeof(ValueT);
        mNbBatches = nbElems;
        }

    //!
    //! \brief Make a copy of other with new memory, but the same elements. Shallow!
    //!
    ManagedBuffer(const ManagedBuffer& other) = delete;

    //!
    //! \brief Move ownership of buffer to new buffer.
    //!
    ManagedBuffer(ManagedBuffer&& other) noexcept {
        moveManagedBuffer(std::forward<ManagedBuffer&&>(other));
    }


    //!
    //! \brief Free the underlaying data when the CudaBuffer goes out of scope.
    //!
    ~ManagedBuffer() {
        if constexpr(Loc == ManagedBufferLocation::CUDA) {
            CHECK_EQ(cudaFree(mData), cudaSuccess);
        } else if constexpr (Loc == ManagedBufferLocation::HOST_PINNED) {
            CHECK_EQ(cudaFreeHost(mData), cudaSuccess);
        } else {
            static_assert(constexpr_false<Loc>, "Unimplemented ManagedBufferLocation for Managed Buffer");
        }
    }

    //!
    //! \brief Get pointer to data.
    //!
    ValueT* data() {
        return mData;
    }

    //!
    //! \brief Get const pointer to data.
    //!
    const ValueT* data() const {
        return mData;
    }


    // "size"" is ambiguous, use nbElems or nbBytes instead
    // //!
    // //! \brief Get size of buffer as the number of elements.
    // //!
    // size_t size() const
    // {
    //     return mNbElems;
    // }


    // Below we have multiple accessors to element and container size attributes. The bifurcation for Raw (dynamic type, a char/uint8_t buffer of length known at runtime, typically from config/CLI arguments) and "normal" (static type known at compile time) types are necessary given that we want to serve both dynamic and static types with the same interface.
    // Example 1: static type
    /* CudaBuffer<int32_t>(5) (cuda buffer of 5 integers)
     * nbElems = 5
     * bytesPerBatch = sizeof(int32_t)
     * nbBatches = 5
     * nbBytes = 5*sizeof(int32_t)
     */
    // Example 2: dynamic type
    /* CudaBufferRaw(6, 100) (cuda buffer of 6 elements, each being 100 bytes big)
     * nbElems = 600 (not particularly useful value)
     * bytesPerBatch = 100
     * nbBatches = 6
     * nbBytes = 600
     */
    // As you can see, the "consistent"/desired interface is brought by using bytesPerBatch (for element size inquiries) and nbBatches (for container length inquiries).

    // Total number of elements of type T in this buffer. If using RAW, this is not preferred externally.
    size_t nbElems() const {
        return mNbElems;
    }
    // This is either sizeof(T) (if we're not Raw) OR batch_stride (if we are Raw)
    // This is the prefered way to determine element size using the public interface.
    size_t bytesPerBatch() const {
        return mStride;
    }
    // This is either mNbElems (if we're Raw) OR n_batches (if we are Raw)
    // This is the prefered way to determine the number of elements in the buffer
    size_t nbBatches() const {
        return mNbBatches;
    }

    // Total number of bytes in this buffer. Useful to know absolute size of the buffer
    size_t nbBytes() const  {
        return mNbElems * sizeof(ValueT);
    }


    void fillWithZero() {
        if constexpr (Loc == ManagedBufferLocation::CUDA) {
            if (mNbElems){
                CHECK_EQ(cudaMemset(mData, 0, nbBytes()), cudaSuccess);
            }
        } else {
            static_assert(constexpr_false<Loc>, "fillWithZero only implemented for CudaBuffer");
        }

    }
    auto fillWithZeroAsync(cudaStream_t stream) {
        if constexpr (Loc == ManagedBufferLocation::CUDA) {
            return cudaMemsetAsync(this->data(), 0, this->nbBytes(), stream);
        } else {
            static_assert(constexpr_false<Loc>, "fillWithZeroAsync only implemented for CudaBuffer");
        }

    }
    ValueT* get_ptr_from_idx(size_t idx) {
        CHECK(idx < nbBatches());
        if constexpr(std::is_same<ValueT, uint8_t>::value) {
            // Use batch byte calculations
            return mData + bytesPerBatch()*idx;
        } else {
            // Use built-in pointer arithmetic
            return mData + idx;
        }
    }

    void cudaSparseMemsetAsync(bool* sparseMask, size_t actualBatchSize, cudaStream_t stream) {
        // We know that this kernel launch will attempt to cast the data to an int32 buffer and assign
        // to the buffer in chunks of sizeof(int32_t)
        // What we need to ensure is that the number of bytes we're trying to assign is cleanly divisible by an int (so we don't overrun our buffers):
        if constexpr(!std::is_same<ValueT, uint8_t>::value) {
            // Checking time!
            assert(this->bytesPerBatch() % sizeof(int32_t) == 0);
        }
        rnntSparseMemSet(reinterpret_cast<uint8_t*>(this->data()), sparseMask, this->bytesPerBatch(), actualBatchSize, stream);
    }
    void slowSparseMemsetAsync(bool* sparseMask, size_t actualBatchSize, cudaStream_t stream) {
        // We want to know how to add to our pointer to get to the next entry referred to by our sparse mask
        for (size_t bs = 0; bs < actualBatchSize; ++bs) {
            if (sparseMask[bs] == false) {
                CHECK_EQ(cudaMemsetAsync(this->data() + (bytesPerBatch() * bs), 0, bytesPerBatch(), stream), cudaSuccess);
            }
        }
    }

private:
        //!
        //! \brief Helper function use to move ownership from one CudaBuffer to the other.
        //!
    void moveManagedBuffer(ManagedBuffer&& other) {
        if constexpr (Loc == ManagedBufferLocation::CUDA) {
            CHECK_EQ(cudaFree(mData), cudaSuccess);
        } else if constexpr (Loc == ManagedBufferLocation::HOST_PINNED) {
            CHECK_EQ(cudaFreeHost(mData), cudaSuccess);
        } else {
            static_assert(constexpr_false<Loc>, "Unimplemented ManagedBufferLocation for ManagedBuffer");
        }
        mData = other.mData;
        mNbElems = other.mNbElems;
        mStride = other.mStride;
        mNbBatches = other.mNbBatches;
        other.mData = nullptr;
        other.mNbElems = 0;
        other.mStride = 0;
        other.mNbBatches = 0;
    }

    ValueT* mData = nullptr;
    size_t mNbElems = 0;
    // Will be set to sizeof(T) if no batch_stride given
    size_t mStride = 0;
    size_t mNbBatches = 0;
};

template <typename ValueT>
using CudaBuffer = ManagedBuffer<ValueT, ManagedBufferLocation::CUDA>;
template <typename ValueT>
using HostBuffer = ManagedBuffer<ValueT, ManagedBufferLocation::HOST_PINNED>;

using CudaBufferRaw   = CudaBuffer<uint8_t>;
using CudaBufferInt8  = CudaBuffer<int8_t>;
using CudaBufferInt32 = CudaBuffer<int32_t>;
using CudaBufferFP16  = CudaBuffer<uint16_t>;
using CudaBufferFP32  = CudaBuffer<float>;
using CudaBufferBool  = CudaBuffer<bool>;
using HostBufferInt32 = HostBuffer<int32_t>;
using HostBufferFP32  = HostBuffer<float>;

//!
//! \brief Dump the CudaBuffer into the stream.
//!
template <typename T>
inline std::ostream& operator<<(std::ostream& stream, const CudaBuffer<T>& other)
{
    auto hostPtr = std::make_unique<T[]>(other.nbElems());
    CHECK_EQ(cudaMemcpy(hostPtr.get(), other.data(), other.nbElems()*sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    stream << "CudaBuffer with " << other.nbBatches() << "elements";
    if constexpr(std::is_same<T, uint8_t>::value) {
        stream << " (raw measurement, contains: " << other.nbElems() << " uint8_t's total) elements";
    }
    stream << " each with size: " << other.bytesPerBatch();

    for(size_t j = 0; j < other.nbElems(); j++)
    {
        stream << std::setprecision(3) << hostPtr[j]  << ", ";
        if(j % 10 == 9)
            stream << std::endl;
    }
    return stream;
}
// For completeness, a HostBuffer overload:
template <typename T>
inline std::ostream& operator<<(std::ostream& stream, const HostBuffer<T>& other)
{
    stream << "HostBuffer with " << other.nbBatches() << "elements";
    if constexpr(std::is_same<T, uint8_t>::value) {
        stream << " (raw measurement, contains: " << other.nbElems() << " uint8_t's total) elements";
    }
    stream << " each with size: " << other.bytesPerBatch();

    for(size_t j = 0; j < other.nbElems(); j++)
    {
        stream << std::setprecision(3) << other.data()[j]  << ", ";
        if(j % 10 == 9)
            stream << std::endl;
    }

    return stream;
}


// Utility functions for copying
// Copy from host buffer to device buffer asynchronously, with optional speicified size.
template <class T>
void memcpyH2DAsync(CudaBuffer<T>& cudaBuffer, const HostBuffer<T>& hostBuffer, cudaStream_t stream, size_t size = 0)
{
    if (size == 0)
    {
        CHECK_EQ(cudaBuffer.nbElems(), hostBuffer.nbElems());
        size = cudaBuffer.nbBytes();
    }
    CHECK_EQ(cudaMemcpyAsync(cudaBuffer.data(), hostBuffer.data(), size, cudaMemcpyHostToDevice, stream), cudaSuccess);
}

template <class T>
void memcpyH2DAsync(CudaBuffer<T>& cudaBuffer, const T* hostBuffer, cudaStream_t stream, size_t size = 0)
{
    if (size == 0)
    {
        size = cudaBuffer.nbBytes();
    }
    // Check that we can actually move enough items to the cudabuffer as requested
    if (cudaBuffer.nbBytes() < size) {
        gLogInfo << "Buffer Overrun!" << std::endl;
        CHECK(false);
    }
    CHECK_EQ(cudaMemcpyAsync(cudaBuffer.data(), hostBuffer, size, cudaMemcpyHostToDevice, stream), cudaSuccess);
}

// Copy from device buffer to host buffer asynchronously, with optional speicified size.
template <class T>
void memcpyD2HAsync(HostBuffer<T>& hostBuffer, const CudaBuffer<T>& cudaBuffer, cudaStream_t stream, size_t size = 0)
{
    if (size == 0)
    {
        CHECK_EQ(cudaBuffer.nbElems(), hostBuffer.nbElems());
        size = cudaBuffer.nbBytes();
    }
    CHECK_EQ(cudaMemcpyAsync(hostBuffer.data(), cudaBuffer.data(), size, cudaMemcpyDeviceToHost, stream), cudaSuccess);
}
// Copy from device buffer to host buffer asynchronously, with optional speicified size.
template <class T>
void memcpyD2HAsync(T *hostBuffer, const CudaBuffer<T>& cudaBuffer, cudaStream_t stream, size_t size = 0)
{
    if (size == 0)
    {
        size = cudaBuffer.nbBytes();
    }
    if (cudaBuffer.nbBytes() < size) {
        gLogInfo << "Trying to read more data than in CudaBuffer!" << std::endl;
    }
    CHECK_EQ(cudaMemcpyAsync(hostBuffer.data(), cudaBuffer.data(), size, cudaMemcpyDeviceToHost, stream), cudaSuccess);
}


// Copy from device buffer to device buffer asynchronously, with optional speicified size.
template <class T>
void memcpyD2DAsync(CudaBuffer<T>& dstCudaBuffer, const CudaBuffer<T>& srcCudaBuffer, cudaStream_t stream, size_t size = 0)
{
    if (size == 0)
    {
        CHECK_EQ(dstCudaBuffer.nbElems(), srcCudaBuffer.nbElems());
        size = dstCudaBuffer.nbBytes();
    }
    CHECK_EQ(cudaMemcpyAsync(dstCudaBuffer.data(), srcCudaBuffer.data(), size, cudaMemcpyDeviceToDevice, stream), cudaSuccess);
}
