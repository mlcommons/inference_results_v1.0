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

#include <iostream>

#include "engine.h"

Engine::Engine(Config config) {
    validateConfig(config);
    _config = config;

    init();
}

Engine::~Engine() {
    if (_requests.size()) {
        waitAll();
        _requests.clear();
    }
}

void Engine::init() {
    InferenceEngine::Core core;

    std::map<std::string, std::string> config;
    if (_config.threads > 0) {
        config[InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(_config.threads);
    }
    config[InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] = std::to_string(_config.streams);
    core.SetConfig(config);

    InferenceEngine::CNNNetwork network = core.ReadNetwork(_config.model_xml, _config.model_bin);

    network.setBatchSize(_config.batch_size);

    _exec_net = core.LoadNetwork(network, _config.device);

    for (size_t i = 0; i < _config.streams; i++) {
        _requests.push_back(_exec_net.CreateInferRequestPtr());
    }

    findInputOutputName();

    InferenceEngine::TensorDesc src_td = _exec_net.GetInputsInfo().cbegin()->second->getTensorDesc();
    _input_size = 1;
    for (auto dim : src_td.getDims()) {
        _input_size *= dim;
    }
    _input_size *= src_td.getPrecision().size();
}

void Engine::setCoreConfig(InferenceEngine::Core& core) {
    std::map<std::string, std::string> config;
    if (_config.threads > 0) {
        config[InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(_config.threads);
    }
    config[InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] = std::to_string(_config.streams);
    core.SetConfig(config, _config.device);
}

void Engine::validateConfig(Config& config) {
    if (config.model_xml.empty() || config.model_bin.empty()) {
        throw "Please provide path to model xml and bin files.";
    }

    if (config.device.empty()) {
        throw "Please provide devoce name to use.";
    }

    if (config.streams == 0) {
        throw "Number of stream has to be more than 0.";
    }
}

void Engine::findInputOutputName() {
    _input_name = _exec_net.GetInputsInfo().begin()->first;

    size_t max_channels = 0;

    for (auto it : _exec_net.GetOutputsInfo()) {
        if (max_channels < it.second->getDims()[1]) {
            _output_name = it.first;
        }
    }
}

size_t Engine::inputSize() {
    return _input_size;
}

void Engine::fillInput(size_t idx, float* src_ptr, size_t src_size)  {
   char* blob = getRequest(idx)->GetBlob(_input_name)->buffer().as<char*>();
   if (inputSize() != src_size) {
       throw "Src size is not equal to buffer size.";
   }
   memcpy(blob, src_ptr, inputSize());
}

void Engine::setInput(size_t idx, float* src_ptr, size_t src_size) {
    InferenceEngine::Blob::Ptr src_blob = getRequest(idx)->GetBlob(_input_name);
    if (inputSize() != src_size) {
        throw "Src size is not equal to buffer size.";
    }
    InferenceEngine::TensorDesc src_td = src_blob->getTensorDesc();
    InferenceEngine::Blob::Ptr input_blob = InferenceEngine::make_shared_blob<float>(src_td, src_ptr);
    getRequest(idx)->SetBlob(_input_name, input_blob);
}

void Engine::startAsync(size_t idx) {
    getRequest(idx)->StartAsync();
}

void Engine::waitAll() {
    for (size_t i = 0; i < getNumRequests(); i++) {
        getRequest(i)->Wait(InferenceEngine::IInferRequest::RESULT_READY);
    }
}
