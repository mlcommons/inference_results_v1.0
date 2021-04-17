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
#ifndef CONV_3D_1X1X1_K4_H
#define CONV_3D_1X1X1_K4_H
#include <stdint.h>

#include <cuda_runtime_api.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Input_Data_Type_,
          typename Output_Data_Type_,
          int BLOCK_ROW_WARPS_, // The number of warps in the block in the same row, in the "horizontal" (column) dimension
          int BLOCK_COL_WARPS_, // The number of warps in the block in the same column, in the "vertical" (row) dimension   
          int IMG_C>
struct Conv3d_1x1x1_k4_kernel_params {
    // each warp is doing a single (32,8,16) MMA tile
    enum { BLOCK_ROW_WARPS = BLOCK_ROW_WARPS_ };
    enum { BLOCK_COL_WARPS = BLOCK_COL_WARPS_ };

    enum { WARPS_PER_BLOCK =  BLOCK_ROW_WARPS * BLOCK_COL_WARPS};

    typedef Input_Data_Type_ Input_Data_Type;
    typedef Output_Data_Type_ Output_Data_Type;

    // MMA matrix tile dimensions.
    enum { M = 32 };
    enum { N = 8 };
    enum { K = 16 };

    // Implementation constants.
    enum { WARP_SIZE = 32 };
    enum { BLOCK_ROW_TILES =  BLOCK_ROW_WARPS }; 
    enum { BLOCK_COL_TILES =  BLOCK_COL_WARPS };
    enum { THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE };

    enum { CHUNK_K = IMG_C / K }; 
};

struct Conv3d1x1x1k4Context {
    Conv3d1x1x1k4Context() : sm_count(0), sm_shared_size(0), sm_version(0) {};
    int sm_count;
    int sm_shared_size;
    int sm_version;
};

struct GemmParams {
    // The dimensions of the problem.
    int m, n, k;
    // The A matrix.
    void *gmem_a;
    // The B matrix.
    void *gmem_b;
    // The C matrix.
    void *gmem_c;
    // The leading dimensions for A, B and C.
    int lda, ldb, ldc;
};

struct Conv3d1x1x1k4Params : public GemmParams {
    float scale;
    // The images.
    int img_n, img_c, img_d, img_h, img_w;
    int img_dhw;

    // The filter.
    int flt_k;
};

int conv3d_1x1x1_k4_dispatch(const Conv3d1x1x1k4Context& context, const Conv3d1x1x1k4Params& params, cudaStream_t stream);

#endif // CONV_3D_1X1X1_K4_H


