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

#include <cuda_runtime_api.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int R_ = 2, int S_ = 2, int T_ = 2, typename Data_Type_ = int8_t, int THREADS_PER_CTA_ = 256, int THREADS_PER_PIXEL_ = 8, int C_ELEMENTS_PER_CTA_ = 32>
struct Pixel_shuffle_kernel_params {
    enum { THREADS_PER_CTA = THREADS_PER_CTA_ };
    enum { THREADS_PER_PIXEL = THREADS_PER_PIXEL_ }; 
    enum { R = R_ };
    enum { S = S_ };
    enum { T = T_ };

    typedef Data_Type_ Data_Type;

    enum { WARP_SIZE = 32 };
    enum { NUM_WARPS =  THREADS_PER_CTA_ / WARP_SIZE };

    enum { PIXELS_PER_THREAD = 2 };
    enum { C_ELEMENTS_PER_CTA = C_ELEMENTS_PER_CTA_ };
    enum { ELEMENTS_PER_LDG = C_ELEMENTS_PER_CTA / THREADS_PER_PIXEL };  // 4 default

    // Derived params.
    enum { PIXELS_PER_CTA = THREADS_PER_CTA/THREADS_PER_PIXEL * PIXELS_PER_THREAD };
};

struct PixelShuffleParams {
    // The input/output tensors.
    void *gmem_src, *gmem_dst;
    // The dimensions.
    int n, k, o, p, q;
    int r, s, t;

    int output_stride;

    float scale;
};

int pixel_shuffle_ncdhw32_to_ncdhw32_dispatch(PixelShuffleParams params, cudaStream_t stream);

int pixel_shuffle_ncdhw_to_ncdhw_dispatch(PixelShuffleParams params, cudaStream_t stream);


