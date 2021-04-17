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

#include "dali/c_api.h"
#include "dali/operators.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>

#include <ctime>
#include <ratio>
#include <chrono>
#include <thread>
#include <string.h>

using namespace std;

std::string GetEncodedSample(const std::string& recording_path) {
    std::ifstream fin(recording_path, std::ios::binary);
    std::stringstream ss;
    ss << fin.rdbuf();
    return ss.str();
}

int main()
{
    daliInitialize();
    daliInitOperators();
    constexpr int batch_size = 16;
    int total_samples = 16;
    int device = 1; // 0 = CPU, 1 = GPU
    constexpr int num_threads = 2;
    constexpr int prefetch_queue_depth = 1;
    constexpr int device_id = 0;
    cudaStream_t copy_stream;
    std::string filename;
    std::ifstream *fin;
    
    if(device == 0)
    {
        fin = new std::ifstream("dali_pipeline_cpu.pth", std::ios::binary);
    }
    else
    {
        fin = new std::ifstream("dali_pipeline_gpu.pth", std::ios::binary);
    }

    std::stringstream ss;
    ss << fin->rdbuf();
    auto pipe = ss.str();
    daliPipelineHandle handle;
    cout << "Dali pipeline creating.." << endl;

    daliCreatePipeline(&handle,
                        pipe.c_str(),
                        pipe.length(),
                        batch_size,
                        num_threads,
                        device_id,
                        false,
                        prefetch_queue_depth,
                        prefetch_queue_depth,
                        prefetch_queue_depth, 0);
    cout << "Dali pipeline created" << endl;

    std::vector<std::string> recordings_paths;
    
    int total_batches = total_samples / batch_size;

    for(auto i=0; i < total_samples; i++){
        string filename = "wav_" + to_string(i) + ".npy";
        recordings_paths.push_back(filename);
    }

    std::vector<std::string> qsl;

    for(auto i=0; i < total_samples; i++){
        std::string sample;

        std::string &rec_path = recordings_paths[i];
        sample = GetEncodedSample(rec_path);       

        cout << "Sample size " << sample.size() << " Sample " << i <<  endl;
        qsl.push_back(sample);
    }


    using namespace std::chrono;

    auto non_blocking = 0;
    device_type_t dst_type;
    
    if(device == 0)
        dst_type = CPU;
    else
        dst_type = GPU;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    void *audio_samples;
    int64_t sample_lengths[batch_size];
    void *sample_lengths_ptr = (size_t *) sample_lengths;

    bool isPinned = 0;
    void *audio_samples_gpu;
    void *sample_lengths_gpu;

    size_t total_audio_samples_size = 0;

    size_t offset = 0;

    size_t sample_id = 0;
    for(auto i=0; i < total_batches; i++){
        for(auto j = 0; j < batch_size; j++)
        {
            auto sample_length = qsl[sample_id].length();
            cout << "Sample length " << sample_length << " Batch " << i <<  endl;
            sample_lengths[sample_id] = sample_length / 4;
            total_audio_samples_size += sample_length;
            sample_id++;
        }

        audio_samples = malloc(total_audio_samples_size);

        sample_id = 0;
        for(auto j = 0; j < batch_size; j++)
        {
            void *host_ptr = (int8_t*) audio_samples + offset;
            auto sample_length = qsl[sample_id].length();
            memcpy(host_ptr, qsl[sample_id].data(), sample_length);           
            offset += sample_length;
            sample_id++;
        }

        cudaMalloc(&audio_samples_gpu, total_audio_samples_size);
        cudaMalloc(&sample_lengths_gpu, batch_size * sizeof(size_t));

        cudaMemcpy(audio_samples_gpu, audio_samples, total_audio_samples_size, cudaMemcpyHostToDevice);
        cudaMemcpy(sample_lengths_gpu, sample_lengths_ptr, batch_size * sizeof(size_t), cudaMemcpyHostToDevice);

        //std::vector<std::string> qsl2;
        //for(auto sample : qsl) { qsl2.push_back(sample); }
        // Uncommenting above works for BS-256
        //daliSetExternalInput(&handle, "INPUT_0", device_type_t::GPU, audio_samples_gpu, DALI_FLOAT, (int64_t*)sample_lengths_gpu, 1, nullptr, 1); 
        if(device == CPU){
            daliSetExternalInput(&handle, "INPUT_0", device_type_t::CPU, audio_samples, DALI_FLOAT, (int64_t*)sample_lengths, 1, nullptr, 1); 
        }
        else{
            daliSetExternalInput(&handle, "INPUT_0", device_type_t::GPU, audio_samples_gpu, DALI_FLOAT, (int64_t*)sample_lengths_gpu, 1, nullptr, 1); 
        }

        std::cout << "In here" << std::endl;
        daliRun(&handle);
        daliOutput(&handle);
        auto num_outputs = daliGetNumOutput(&handle);
        cout << "Number of outputs " << num_outputs << endl;
        cudaStreamCreate(&copy_stream);
        void *dst[num_outputs];
        for(auto n=0; n < num_outputs; n++)
        {
            auto max_dim_tensor = daliMaxDimTensors(&handle, n);
            cout << "Max Dim Tensor at " << n << ": " << max_dim_tensor << endl;
            auto tensor_size = daliTensorSize(&handle, n);
            cout << "Tensor size at " << n << ": " << tensor_size << endl;
            auto num_tensors = daliNumTensors(&handle, n);
            cout << "Number of Tensors at " << n << ": " << num_tensors << endl;
            auto tensor_type = daliTypeAt(&handle,n);
            cout << "Tensor type at " << n << ": " << tensor_type << endl;
            auto non_blocking = 0;
            if(device == GPU)
                cudaMalloc(&dst[n], tensor_size);
            else    
                dst[n] = (int8_t*) malloc(tensor_size);
            daliCopyTensorListNTo(&handle, dst[n], n, dst_type, copy_stream, non_blocking);            
        }

        /* Print out tensor sizes */
        if(device == CPU){
            for(auto i = 0, j = 1 ; i < batch_size; i++, j+=2 )
            {
                std::cout << "Tensor size at [" << i << "] = " << *((size_t*)dst[1] + j) << std::endl;
            }
        }
        free(audio_samples);
      }



    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    std::cout << "It took " << time_span.count() << " seconds." << endl;
    std::cout << "Batch Size " << batch_size << " Total Batches = " << total_batches << " Num_threads " << num_threads << " Samples/Sec " << (batch_size * total_batches) / time_span.count() << endl;
    
    daliDeletePipeline(&handle);
    cout << "Dali pipeline deleted" << endl;
    return 0;
}
