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
#include <cuda_fp16.h>

namespace instance_norm_impl
{

//typedef __half GMEM_SUMS_TYPE;
typedef float GMEM_SUMS_TYPE;

#define DISABLE_MEAN_VAR_OUTPUT 0

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename StorageType, int SM >
constexpr int get_pixels_per_thread_in_registers()
{
    switch (sizeof(StorageType)) {
        case 4: return 2; break;
        case 2: return 4; break;
        default: break;
    }

// case INT8 V100 - 8, A100 - 24, A10 - 16
    if ( SM < 800 ) {
        if ( SM == 750 ) {
            return 16;
        } else {
            return 8;
        }
    } else if ( SM == 860 ) {
        return 16;
    } else {
        return 24;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename StorageType, int SM >
constexpr int get_pixels_per_thread_in_smem()
{
    switch (sizeof(StorageType)) {
        case 4: return 4; break;
        case 2: return 8; break;
        default: break;
    }

// case INT8
    if ( SM < 800 ) {
        if ( SM == 750 ) {
            return 7;
        } else {
            return 8;
        }
    } else if ( SM == 860 ) {
        return 16;
    } else {
        return 24;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Input_Data_Type_ = uint16_t, typename Output_Data_Type_ = uint16_t, 
          typename StorageType_ = float, int THREADS_PER_CTA_ = 512,  
          int THREADS_PER_PIXEL_ = 16, int C_ELEMENTS_PER_CTA_ = 64, int SM_ = 700>
struct Instance_norm_kernel_params {
    enum { USE_ONLINE_APPROACH = 1 };
    enum { THREADS_PER_CTA = THREADS_PER_CTA_ };
    enum { THREADS_PER_PIXEL = THREADS_PER_PIXEL_ }; // 8 or 16
    enum { SM = SM_ };

    typedef Input_Data_Type_ Input_Data_Type;
    typedef Output_Data_Type_ Output_Data_Type;

    typedef StorageType_ StorageType;
    enum { PIXELS_PER_THREAD_IN_REGISTERS = get_pixels_per_thread_in_registers<StorageType, SM>() };
    enum { PIXELS_PER_THREAD_IN_SMEM = get_pixels_per_thread_in_smem<StorageType, SM>() };

    enum { C_ELEMENTS_PER_CTA = C_ELEMENTS_PER_CTA_ }; //64;
    enum { ELEMENTS_PER_LDG = C_ELEMENTS_PER_CTA / THREADS_PER_PIXEL };  // 4 default

    // Derived params.
    enum { PIXELS_PER_LDG = THREADS_PER_CTA/THREADS_PER_PIXEL };
    enum { MIN_PIXELS_PER_CTA = PIXELS_PER_LDG * PIXELS_PER_THREAD_IN_REGISTERS };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct InstanceNormFwdContext {
    InstanceNormFwdContext() : sm_count(0), sm_shared_size(0), sm_version(0) {};
    int sm_count;
    int sm_shared_size;
    int sm_version;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct InstanceNormFwdParams {
    // The input/output tensors.
    void *gmem_src, *gmem_dst;
    // The bias/scale.
    float *gmem_bias, *gmem_scale;
    // running mean/var (refer BN API from cudnn doc)
    float *gmem_running_mean, *gmem_running_var;
    // saved mean/var (refer BN API from cudnn doc)
    float *gmem_saved_mean, *gmem_saved_var;
    // The dimensions.
    int nhw, c, n;
    // The buffer to do the reduction for mean, stddev and count.
    GMEM_SUMS_TYPE *gmem_sums;
    // The buffer to count items in the different CTAs.
    int *gmem_counts;
    // The counters of retired CTAs.
    int *gmem_retired_ctas;
    // The epsilon to apply to the computation of the variance.
    float var_eps;
    // outer loop count
    int outer_loops;
    // exponential average factor
    float exp_avg_factor;
    // use relu as activation?
    bool use_relu;
    float relu_alpha;

    int c_blks;

    float in_scale;

    float out_scale;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void instance_norm_buffer_sizes_dispatch(const InstanceNormFwdContext& context,
                                const InstanceNormFwdParams& params, 
                                size_t &size_sums, size_t &size_counts, size_t &size_retired_ctas,
                                int input_data_type = 1, int output_data_type = 1);

int instance_norm_fwd_dispatch(const InstanceNormFwdContext& context, InstanceNormFwdParams& params,
                               cudaStream_t stream, int input_data_type = 1, 
                               int output_data_type = 1);

} // namespace instance_norm_impl