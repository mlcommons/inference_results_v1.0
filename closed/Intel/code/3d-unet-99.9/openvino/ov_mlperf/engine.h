/*
 // Copyright (c) 2020 Intel Corporation
 //
 // Licensed under the Apache License, Version 2.0 (the "License");
 // you may not use this file except in compliance with the License.
 // You may obtain a copy of the License at
 //
 //      http://www.apache.org/licenses/LICENSE-2.0
 //
 // Unless required by applicable law or agreed to in writing, software
 // distributed under the License is distributed on an "AS IS" BASIS,
 // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 // See the License for the specific language governing permissions and
 // limitations under the License.
 */

#pragma once

#include <memory>
#include <inference_engine.hpp>

class Engine {
public:
    using Ptr = std::shared_ptr<Engine>;

    typedef struct Config {
        Config() {
            device = "CPU";

            batch_size = 1;

            streams = 1;
            threads = 0;
        }

        std::string model_xml;
        std::string model_bin;

        std::string device;

        size_t batch_size;

        size_t streams;
        size_t threads;
    } Config_t;

public:
    explicit Engine(Config config);
    virtual ~Engine();

public:
    size_t getNumRequests() {
        return _requests.size();
    }

    InferenceEngine::InferRequest::Ptr getRequest(size_t idx) {
        return _requests[idx];
    }

    void fillInput(size_t idx, float* src_ptr, size_t src_size);
    void setInput(size_t idx, float* src_ptr, size_t src_size);

    template<typename T>
    T* getOutput(size_t idx, size_t& size) {
        size = (getRequest(idx)->GetBlob(_output_name)->byteSize() / sizeof(T));
        return getRequest(idx)->GetBlob(_output_name)->buffer().as<T*>();
    }

    size_t inputSize();

    void startAsync(size_t idx);
    void waitAll();

private:
    void init();
    void setCoreConfig(InferenceEngine::Core& core);
    void validateConfig(Config& config);
    void findInputOutputName();
    
private:
    Config_t _config;
    InferenceEngine::ExecutableNetwork _exec_net;
    std::vector<InferenceEngine::InferRequest::Ptr> _requests;
    std::string _input_name;
    std::string _output_name;
    size_t _input_size;
};
