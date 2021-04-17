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

#include <stdint.h>

#define DEVICE_FUNCTION static inline __device__

template< int N >
DEVICE_FUNCTION void zero(int (&dst)[N]) {
    #pragma unroll
    for( int i = 0; i < N; ++i ) {
        dst[i] = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
DEVICE_FUNCTION void ldg(int (&dst)[1], const T *gmem) {
    dst[0] = __ldg((const int*) gmem);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
DEVICE_FUNCTION void ldg(int8_t (&dst)[N], const int8_t *gmem) {
    int tmp[N/4];
    ldg(tmp, gmem);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
DEVICE_FUNCTION void stg(T *gmem, int (&src)[1]) {
    reinterpret_cast<int*>(gmem)[0] = src[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
DEVICE_FUNCTION void stg(int8_t *gmem, int8_t (&src)[N]) {
    int tmp[N/4];
    stg(gmem, tmp);
}

