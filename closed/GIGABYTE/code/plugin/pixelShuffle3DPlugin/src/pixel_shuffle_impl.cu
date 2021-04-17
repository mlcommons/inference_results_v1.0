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
#include <stdio.h>
#include <assert.h>

#include "pixel_shuffle.h"
#include "pixel_shuffle_common.h"

using kernel_params_ncdhw32_2x2x2_256 = Pixel_shuffle_kernel_params<2, 2, 2, int8_t, 256, 8, 32>;
using kernel_params_ncdhw32_2x2x2_128 = Pixel_shuffle_kernel_params<2, 2, 2, int8_t, 128, 8, 32>;
using kernel_params_ncdhw_fp32_2x2x2_256 = Pixel_shuffle_kernel_params<2, 2, 2, float, 256, 8, 32>;

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __host__ __device__ int div_up(int m, int n) 
{
    return (m + n - 1) / n;
}

template <typename Kernel_params, int Do_scales = 0>
__global__ void __launch_bounds__(Kernel_params::THREADS_PER_CTA, 1)
pixel_shuffle_ncdhw32_to_ncdhw32(PixelShuffleParams params)
{
    // blockIdx.x for O*P
    // blockIdx.y for k
    // blockIdx.z for batch
    const int tid = threadIdx.x;
    const int WARP_SIZE = Kernel_params::WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int R = Kernel_params::R;
    const int S = Kernel_params::S;
    const int T = Kernel_params::T;

    const int RST = R * S * T;

    int opq = params.o * params.p * params.q;
    int pq = params.p * params.q;
    int dhw = opq / RST;
    int hw = pq / (S * T);
    int o = blockIdx.x / params.p;
    int p = blockIdx.x % params.p;

    const int cta_k = blockIdx.y;
    const int k = cta_k * 32 + lane_id;

    const int is_valid_k = (k < params.k);

    if (!is_valid_k) return;

    int d = o / R;
    int h = p / S;

    // The base pointer to load from.
    const int8_t *gmem_src = &reinterpret_cast<const int8_t *>(params.gmem_src)[ blockIdx.z * params.k * opq
                                                                                + d * hw * 32
                                                                                + h * params.q / T * 32];

    int8_t *gmem_dst = &reinterpret_cast<int8_t *>(params.gmem_dst)[ blockIdx.z *  params.output_stride
                                                                                + cta_k * 32 * opq
                                                                                + blockIdx.x * params.q * 32
                                                                                //+ o * pq * 32 + p * params.q * 32
                                                                                + lane_id];
    int8_t nx[Kernel_params::ELEMENTS_PER_LDG] = {0};

    for (int iq = 0; iq < params.q; iq += Kernel_params::NUM_WARPS * Kernel_params::ELEMENTS_PER_LDG)
    {
        #pragma unroll
        for (int i = 0; i < Kernel_params::ELEMENTS_PER_LDG; i++)
        {
            int q = iq + warp_id * Kernel_params::ELEMENTS_PER_LDG + i;
            int is_valid_q = (q < params.q);
            int w = q / T;
            int c = k * RST + (o % R) * (S*T) + (p % S) * T + q % T;
            int is_valid_c = (c < params.k * RST);
    
            if (is_valid_c && is_valid_q)
            {
                nx[i] = gmem_src[ (c / 32) * dhw * 32 + w * 32 + c % 32];
                if (Do_scales)
                {
                    float x = __int2float_rn(nx[i]) * params.scale;
                    nx[i] = __float_as_int(min(max(x + 12582912.0F, 12582785.0F), 12583039.0F));
                    //nx[i] = __float2int_rn(fminf(fmaxf(x, INT8_MIN), INT8_MAX));
                }
            }
        }

        // vectorizing stores through "int" just a bit faster than below, since we need to transpose the warp in smem
        #pragma unroll
        for (int i = 0; i < Kernel_params::ELEMENTS_PER_LDG; i++)
        {
            int q = iq + warp_id * Kernel_params::ELEMENTS_PER_LDG + i;
            if (q >= params.q) continue;
            gmem_dst[q * 32] = nx[i];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_params>
__global__ void __launch_bounds__(Kernel_params::THREADS_PER_CTA, 1)
pixel_shuffle_ncdhw_to_ncdhw(PixelShuffleParams params)
{
    // blockIdx.x for O*P*Q / block_dim
    // blockIdx.y for k
    // blockIdx.z for batch

    const int R = Kernel_params::R;
    const int S = Kernel_params::S;
    const int T = Kernel_params::T;

    const int RST = R * S * T;

    int opq = params.o * params.p * params.q;
    int pq = params.p * params.q;
    int dhw = opq / RST;
    int hw = pq / (S * T);
    int opq_idx = blockIdx.x * Kernel_params::THREADS_PER_CTA + threadIdx.x;

    if (opq_idx >= opq) return;

    int pq_reminder = opq_idx % (params.p * params.q);
    int o = opq_idx / (params.p * params.q);
    int p = pq_reminder / params.q;
    int q = pq_reminder % params.q;

    const int cta_k = blockIdx.y;
    const int k = cta_k;

    int d = o / R;
    int h = p / S;
    int w = q / T;

    // The base pointer to load from.
    const typename Kernel_params::Data_Type *gmem_src = &reinterpret_cast<const typename Kernel_params::Data_Type *>(params.gmem_src)[ blockIdx.z * params.k * opq
                                                                                + d * hw 
                                                                                + h * params.q / T + w];

    typename Kernel_params::Data_Type *gmem_dst = &reinterpret_cast<typename Kernel_params::Data_Type *>(params.gmem_dst)[  blockIdx.z *  params.output_stride 
                                                                                + cta_k * opq
                                                                                + opq_idx];
    int c = k * RST + (o % R) * (S*T) + (p % S) * T + q % T;
    int is_valid_c = (c < params.k * RST);

    if (is_valid_c)
    {
        *gmem_dst = gmem_src[c * dhw];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int pixel_shuffle_ncdhw32_to_ncdhw32_dispatch(PixelShuffleParams params, cudaStream_t stream)
{
    using kernel_params = kernel_params_ncdhw32_2x2x2_128;
    assert(params.k >= 32);
    if (params.r == 2 && params.s == 2 && params.t == 2)
    {
        dim3 grid(params.o * params.p, div_up(params.k, 32), params.n);
        //CHECK_CUDA(cudaFuncSetCacheConfig(pixel_shuffle_ncdhw32_to_ncdhw32<T,R,S>, cudaFuncCachePreferL1));
        if (params.scale == 1.f)
        {
            pixel_shuffle_ncdhw32_to_ncdhw32<kernel_params, 0><<< grid 
            , kernel_params::THREADS_PER_CTA
            , 0, stream>>> (params);
        }
        else
        {
            pixel_shuffle_ncdhw32_to_ncdhw32<kernel_params, 1><<< grid 
            , kernel_params::THREADS_PER_CTA
            , 0, stream>>> (params);
        }

    }
    else
    {
        fprintf(stderr, "%d, %d, %d pixel shuffle is not supported\n", params.r, params.s, params.t);
        assert(0);
    }
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int pixel_shuffle_ncdhw_to_ncdhw_dispatch(PixelShuffleParams params, cudaStream_t stream)
{
    assert(params.k >= 32);
    if (params.r == 2 && params.s == 2 && params.t == 2)
    {
        dim3 grid(div_up(params.o * params.p * params.q, kernel_params_ncdhw_fp32_2x2x2_256::THREADS_PER_CTA), params.k, params.n);
        //CHECK_CUDA(cudaFuncSetCacheConfig(pixel_shuffle_ncdhw32_to_ncdhw32<T,R,S>, cudaFuncCachePreferL1));
        pixel_shuffle_ncdhw_to_ncdhw<kernel_params_ncdhw_fp32_2x2x2_256><<< grid 
                                                            , kernel_params_ncdhw_fp32_2x2x2_256::THREADS_PER_CTA
                                                            , 0, stream>>> (params);
    }
    else
    {
        fprintf(stderr, "%d, %d, %d pixel shuffle is not supported\n", params.r, params.s, params.t);
        assert(0);
    }
    return 0;
}