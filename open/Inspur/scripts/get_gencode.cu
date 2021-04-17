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

/* This script prints out the gencode of the specified GPU id. Defaults to GPU 0.
 */
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int device_id = 0;
    if (argc > 1) {
        device_id = atoi(argv[1]);
    }

    int device_count;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount() failed: %s\n", cudaGetErrorString(status));
        return -1;
    }

    if (device_id + 1 > device_count) {
        fprintf(stderr, "Invalid device index %d (Max index: %d)\n", device_id, device_count-1);
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    int gencode = prop.major * 10 + prop.minor;
    printf("%d\n", gencode);

    return 0;
}
