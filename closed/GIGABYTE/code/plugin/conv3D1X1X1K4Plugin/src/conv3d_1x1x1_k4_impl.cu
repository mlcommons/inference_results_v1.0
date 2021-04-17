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

#include <algorithm>

#include "conv3d_1x1x1_k4.h"

#include <mma.h>

using namespace nvcuda;

// <type, BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES>
// WARP_ROW_TILES is determined as `c / 16`, where c is input feature maps, BLOCK_ROW_WARPS must be 1
using kernel_params = Conv3d_1x1x1_k4_kernel_params<int8_t, __half, 1, 4, 32>; 

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int ELEMENTS_PER_WARP_LOAD>
using Copy_int8_t =
    typename std::conditional<ELEMENTS_PER_WARP_LOAD == 32, int8_t,
        typename std::conditional<ELEMENTS_PER_WARP_LOAD == 64, uint16_t,
            typename std::conditional<ELEMENTS_PER_WARP_LOAD == 128, int,
                typename std::conditional<ELEMENTS_PER_WARP_LOAD == 256, int2, int4
                >::type
            >::type
        >::type
    >::type;

template <typename T, int ELEMENTS_PER_WARP_LOAD>
using Copy_t = Copy_int8_t<sizeof(T) / sizeof(int8_t) * ELEMENTS_PER_WARP_LOAD>;

template<int ELEMENTS_PER_THREAD>
using copy_int8_t = Copy_t<int8_t, kernel_params::WARP_SIZE * ELEMENTS_PER_THREAD>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int ELEMENTS_PER_THREAD>
union Access_t
{
    Copy_t<T, kernel_params::WARP_SIZE * ELEMENTS_PER_THREAD> v;
    T x[ELEMENTS_PER_THREAD];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __host__ __device__ int div_up(int m, int n) 
{
    return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_params>
__global__ void __launch_bounds__(Kernel_params::THREADS_PER_BLOCK, 1)
conv3d_1x1x1_k4(Conv3d1x1x1k4Params params, int loops)
{

    const int M = Kernel_params::M;
    const int N = Kernel_params::N;
    const int K = Kernel_params::K;

    const int BLOCK_COL_TILES = Kernel_params::BLOCK_COL_TILES;
    const int CHUNK_K = Kernel_params::CHUNK_K;
    const int THREADS_PER_BLOCK = Kernel_params::THREADS_PER_BLOCK;

    typedef typename Kernel_params::Input_Data_Type Input_Data_Type;
    typedef typename Kernel_params::Output_Data_Type Output_Data_Type;

    const int warp_id = threadIdx.x / Kernel_params::WARP_SIZE;

    const int warp_col_tile = warp_id % BLOCK_COL_TILES;

#if __CUDA_ARCH__ >= 720

    const int SMEM_SIZE = BLOCK_COL_TILES * M * N;

    __shared__ int smem[SMEM_SIZE];

    wmma::fragment<wmma::accumulator, M, N, K, int> c;

    // a is row major, b is column major, c is row major
    Input_Data_Type *gmem_b = reinterpret_cast<Input_Data_Type *>(params.gmem_b); 
    Output_Data_Type *gmem_c = reinterpret_cast<Output_Data_Type *>(params.gmem_c);

    // load CHUNK_K tiles along row A and column B (common k dimension)
    wmma::fragment<wmma::matrix_a, M, N, K, Input_Data_Type, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, M, N, K, Input_Data_Type, wmma::col_major> b[CHUNK_K];

    for (int tid = threadIdx.x; tid < SMEM_SIZE; tid += THREADS_PER_BLOCK) {
        smem[tid] = 0;
    }

    __syncthreads();

    // load matrix b into smem with stride N
    Input_Data_Type* smem_b = reinterpret_cast<Input_Data_Type *>(smem);
    for (int tid = threadIdx.x; tid < params.k * params.n; tid += THREADS_PER_BLOCK) {
        smem_b[tid] = gmem_b[tid];
    }

    __syncthreads();

    // each warp preloads matrix b
    #pragma unroll
    for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::load_matrix_sync(b[k_step], smem_b + k_step * K, CHUNK_K * K);
    }

    int block_i_last = min((int)(blockIdx.x + 1) * loops * BLOCK_COL_TILES * M, params.m);

    for (int block_i = blockIdx.x * loops * BLOCK_COL_TILES * M ; block_i <  block_i_last; block_i+=BLOCK_COL_TILES * M) {

        Input_Data_Type *gmem_a = reinterpret_cast<Input_Data_Type *>(params.gmem_a) + warp_col_tile * M * params.k + block_i * params.k;

        // Initialize the output to zero
        wmma::fill_fragment(c, 0);

        #pragma unroll
        for (int k_step = 0; k_step < CHUNK_K; k_step++) {
            wmma::load_matrix_sync(a, gmem_a + k_step * K, CHUNK_K * K);
            wmma::mma_sync(c, a, b[k_step], c);
        }
    
        __syncthreads();

        wmma::store_matrix_sync(&smem[warp_col_tile * M * N], c, N, wmma::mem_row_major);
    
        __syncthreads();
        int m = block_i + threadIdx.x;
        if (m >= params.m ) return;
        int img_dhw = m % params.img_dhw;
        int img_n = m / params.img_dhw;
        #pragma unroll
        for ( int flt_k = 0; flt_k < params.n; flt_k++) {
            if (threadIdx.x < BLOCK_COL_TILES * M) {
                *(gmem_c + img_n * params.img_dhw * params.n + params.img_dhw * flt_k + img_dhw) = __float2half_rn(smem[threadIdx.x * N + flt_k] * params.scale);
            }
        }
    } // loop

#else
    // code for sm_70. Use HMMA since IMMA is not available

    const int lane_id = threadIdx.x % Kernel_params::WARP_SIZE;

    const int ELEMENTS_PER_THREAD = 16;
    typedef Access_t<int8_t, ELEMENTS_PER_THREAD> access_t;
    typedef copy_int8_t<ELEMENTS_PER_THREAD> copy_t;

    typedef float Accumulator_Data_Type;
    typedef  __half Matrix_AB_Data_Type;

    const int SMEM_SIZE_BYTES = (BLOCK_COL_TILES * M * K * CHUNK_K * sizeof(Matrix_AB_Data_Type) > BLOCK_COL_TILES * M * N * sizeof(Accumulator_Data_Type)) ?
                                BLOCK_COL_TILES * M * K * CHUNK_K * sizeof(Matrix_AB_Data_Type) :
                                BLOCK_COL_TILES * M * N * sizeof(Accumulator_Data_Type);

    __shared__ int8_t smem_buf[SMEM_SIZE_BYTES];
    auto smem_ab = reinterpret_cast<Matrix_AB_Data_Type *>(smem_buf);
    auto smem_c = reinterpret_cast<Accumulator_Data_Type *>(smem_buf);

    wmma::fragment<wmma::accumulator, M, N, K, Accumulator_Data_Type> c;

    // a is row major, b is column major, c is row major
    Input_Data_Type *gmem_b = reinterpret_cast<Input_Data_Type *>(params.gmem_b); 
    Output_Data_Type *gmem_c = reinterpret_cast<Output_Data_Type *>(params.gmem_c);

    // load CHUNK_K tiles along row A and column B (common k dimension)
    wmma::fragment<wmma::matrix_a, M, N, K, Matrix_AB_Data_Type, wmma::row_major> a[CHUNK_K];
    wmma::fragment<wmma::matrix_b, M, N, K, Matrix_AB_Data_Type, wmma::col_major> b[CHUNK_K];

    for (int tid = threadIdx.x; tid < N * K * CHUNK_K; tid += THREADS_PER_BLOCK) {
        smem_ab[tid] = 0;
    }

    __syncthreads();

    // load matrix b into smem with stride N
    Matrix_AB_Data_Type* smem_b = reinterpret_cast<Matrix_AB_Data_Type *>(smem_buf);
    for (int tid = threadIdx.x; tid < params.k * params.n; tid += THREADS_PER_BLOCK) {
        smem_ab[tid] = __int2half_rn(gmem_b[tid]);
    }

    __syncthreads();

    // each warp preloads matrix b
    #pragma unroll
    for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::load_matrix_sync(b[k_step], smem_b + k_step * K, CHUNK_K * K);
    }

    int block_i_last = min((int)(blockIdx.x + 1) * loops * BLOCK_COL_TILES * M, params.m);

    for (int block_i = blockIdx.x * loops * BLOCK_COL_TILES * M ; block_i <  block_i_last; block_i += BLOCK_COL_TILES * M) {

        Input_Data_Type *gmem_a = reinterpret_cast<Input_Data_Type *>(params.gmem_a) + warp_col_tile * M * params.k + block_i * params.k;
        
        auto smem_a = smem_ab + warp_col_tile * M * params.k;
        // Initialize the output to zero
        wmma::fill_fragment(c, 0);

        __syncthreads();
        #pragma unroll
        for (int lid = lane_id; lid < M * params.k / ELEMENTS_PER_THREAD; lid += Kernel_params::WARP_SIZE ) {
            access_t elem;
            elem.v = *(reinterpret_cast<copy_t *>(gmem_a) + lid);
            #pragma unroll
            for (int ii = 0; ii < ELEMENTS_PER_THREAD; ii++) {
                smem_a[lid *ELEMENTS_PER_THREAD + ii] = __int2half_rn(elem.x[ii]);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < CHUNK_K; k_step++) {
            wmma::load_matrix_sync(a[k_step], smem_a + k_step * K, CHUNK_K * K);
            wmma::mma_sync(c, a[k_step], b[k_step], c);
        }

        __syncthreads();

        wmma::store_matrix_sync(&smem_c[warp_col_tile * M * N], c, N, wmma::mem_row_major);

        __syncthreads();
        int m = block_i + threadIdx.x;
        if (m >= params.m ) return;
        int img_dhw = m % params.img_dhw;
        int img_n = m / params.img_dhw;
        #pragma unroll
        for ( int flt_k = 0; flt_k < params.n; flt_k++) {
            if (threadIdx.x < BLOCK_COL_TILES * M) {
                *(gmem_c + img_n * params.img_dhw * params.n + params.img_dhw * flt_k + img_dhw) = __float2half_rn(smem_c[threadIdx.x * N + flt_k] * params.scale);
            }
        }
    } // loop
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int conv3d_1x1x1_k4_dispatch(const Conv3d1x1x1k4Context& context, const Conv3d1x1x1k4Params& params, cudaStream_t stream)
{
    assert(params.img_c == 32);
    assert(params.flt_k <= 8);
    assert(kernel_params::BLOCK_ROW_WARPS == 1);
    assert(kernel_params::M == kernel_params::WARP_SIZE);
    assert(kernel_params::M >= kernel_params::K);

    const int block_sz = kernel_params::THREADS_PER_BLOCK;

    // since weights are kept in registers, the kernel benefits from multiple iterations along `m`
    const int occupancy_factor = 16;
    const int grid = occupancy_factor * context.sm_count;
    const int loops = div_up(div_up(params.m, block_sz), grid);

    conv3d_1x1x1_k4<kernel_params><<<grid, block_sz, 0, stream>>>(params, loops);

    return 0;
}
