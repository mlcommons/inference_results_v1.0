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

void reportAssertion(const char* msg, const char* file, int line);

#define ASSERT(assertion)                                              \
    {                                                                  \
        if (!(assertion))                                              \
        {                                                              \
            reportAssertion(#assertion, __FILE__, __LINE__);           \
        }                                                              \
    }

#define CUDA_ASSERT(call)                                                                       \
    do                                                                                          \
    {                                                                                           \
        cudaError_t cudaStatus = call;                                                          \
        if (cudaStatus != cudaSuccess)                                                          \
        {                                                                                       \
            reportAssertion(cudaGetErrorString(cudaStatus), __FILE__, __LINE__);                \
            ASSERT(0);                                                                          \
        }                                                                                       \
    } while (0)

template <typename T>
void write(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
T read(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}
