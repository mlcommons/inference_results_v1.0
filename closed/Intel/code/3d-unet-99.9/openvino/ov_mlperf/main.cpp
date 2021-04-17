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
#include <mutex>
#include <future>

#include <loadgen.h>
#include <query_sample.h>
#include <bindings/c_api.h>

#include "args.h"
#include "brats_qsl.h"
#include "engine.h"
#include "utils.h"

BratsQSL::Ptr _bratsQSL;
Engine::Ptr _engine;

void issueQueries(mlperf::c::ClientData client, const mlperf::QuerySample* samples, size_t num_samples) {
    size_t num_requests = _engine->getNumRequests();

    std::promise<void> done;
    std::atomic<std::size_t> num_finished{0};

    std::mutex sync_lock;
    size_t processed = 0;
    size_t finished = 0;
    std::atomic<std::size_t> next_sample{0};
    std::map<size_t, size_t> request_to_sample_id;

    using callback_t = std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>;

    if (num_requests > num_samples) {
        num_requests = num_samples;
    }

    for (size_t r = 0; r < num_requests; r++) {
        InferenceEngine::InferRequest::Ptr request = _engine->getRequest(r);

        callback_t callback =
            [r, &request_to_sample_id, samples, num_samples, num_requests, &next_sample, &processed, &finished, &done, &sync_lock]
        (InferenceEngine::IInferRequest::Ptr inferRequest, InferenceEngine::StatusCode code) {
            if (code != InferenceEngine::StatusCode::OK) {
                THROW_IE_EXCEPTION << "Infer request failed with code " << code;
            }

            size_t response_size;
            char* response_ptr = _engine->getOutput<char>(r, response_size);
            mlperf::QuerySampleResponse response { request_to_sample_id[r], reinterpret_cast<std::uintptr_t>(response_ptr), response_size };
            std::vector<mlperf::QuerySampleResponse> responses = { response };
            mlperf::QuerySamplesComplete(responses.data(), responses.size());

            bool run_next = false;
            size_t next_sample_id = 0;
            {
                std::lock_guard<std::mutex> lock(sync_lock);

                processed++;

                if (next_sample >= num_samples) {
                    finished++;

                    if (num_requests == finished) {
                        done.set_value();
                    }
                } else {
                    request_to_sample_id[r] = samples[next_sample].id;
                    next_sample_id = next_sample;
                    next_sample++;
                    run_next = true;
                }
            }

            if (run_next) {
                size_t sample_idx = samples[next_sample_id].index;

                printRow("Processing sample", sample_idx);

                _engine->setInput(r, _bratsQSL->getFeatures(sample_idx), _bratsQSL->getFeaturesSize(sample_idx));
                request_to_sample_id[r] = samples[next_sample_id].id;
                _engine->startAsync(r);
            }
            };

        request->SetCompletionCallback(callback);
    }

    for (size_t r = 0; r < num_requests; r++) {
        size_t sample_idx = samples[next_sample].index;

        printRow("Processing sample", sample_idx);

        _engine->setInput(r, _bratsQSL->getFeatures(sample_idx), _bratsQSL->getFeaturesSize(sample_idx));
        request_to_sample_id[r] = samples[next_sample++].id;
    }

    for (size_t r = 0; r < num_requests; r++) {
        _engine->startAsync(r);
    }

    done.get_future().wait();
}

void processLatencies(mlperf::c::ClientData client, const int64_t* latencies, size_t size) {
}

void flushQueries(void) {
}

void loadQuerySamples(mlperf::c::ClientData client, const mlperf::QuerySampleIndex* samples, size_t num_samples) {
    std::vector<size_t> ids;
    for (size_t i = 0; i < num_samples; i++) {
        ids.push_back(samples[i]);
    }
    _bratsQSL->loadQuerySamples(ids);
}

void unloadQuerySamples(mlperf::c::ClientData client, const mlperf::QuerySampleIndex* samples, size_t num_samples) {
    std::vector<size_t> ids;
    for (size_t i = 0; i < num_samples; i++) {
        ids.push_back(samples[i]);
    }
    _bratsQSL->unloadQuerySamples(ids);
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    
    if (Args::Help()) {
        Args::ShowUsage();
        return 1;
    }

    mlperf::TestSettings mlperf_settings;

    if (!Args::MlperfConfigPath().empty()) {
        if (!fileExists(Args::MlperfConfigPath())) {
            printRow("Config file does not exist", Args::MlperfConfigPath());
            return 1;
        }
        mlperf_settings.FromConfig(Args::MlperfConfigPath(), "3d-unet", Args::Scenario());
    }

    if (!Args::MlperfUserConfigPath().empty()) {
        if (!fileExists(Args::MlperfUserConfigPath())) {
            printRow("User config file does not exist", Args::MlperfUserConfigPath());
            return 1;
        }
        mlperf_settings.FromConfig(Args::MlperfUserConfigPath(), "3d-unet", Args::Scenario());
    }

    if (!stringToScenario(Args::Scenario(), mlperf_settings.scenario)) {
        printRow("Not supported scenario", Args::Scenario());
        return 1;
    }

    if (mlperf_settings.scenario != mlperf::TestScenario::Offline) {
        printRow("Requested scenarion not supported yet", Args::Scenario());
        return 1;
    }

    if (!stringToMode(Args::Mode(), mlperf_settings.mode)) {
        printRow("Not supported mode", Args::Mode());
        return 1;
    }

    printRow("Constructing QSL...");

    try {
        _bratsQSL = BratsQSL::Ptr(new BratsQSL(Args::DataPath()));
    }
    catch (const char* err) {
        printRow("Error while creating QSL");
        return 1;
    }

    if (_bratsQSL->size() == 0) {
        printRow("There is no samples in data path", Args::DataPath());
        return 1;
    } else {
        printRow("Found samples", _bratsQSL->size());
    }

    if (Args::SampleCount() > _bratsQSL->size()) {
        printRow("Sample count can not be greater than number of samples in data set.");
        return 1;
    }

    Engine::Config config;
    config.model_xml = Args::ModelXml();
    config.model_bin = Args::ModelBin();
    config.device = Args::Device();
    config.streams = Args::Streams();
    config.threads = Args::Threads();

    printRow("");
    printRow("Initializing OpenVino Engine...");
    printRow("Model", config.model_xml);
    printRow("Device", config.device);
    printRow("Streams", config.streams);
    printRow("Threads", config.threads);
    printRow("");

    try {
        _engine = Engine::Ptr(new Engine(config));
    }
    catch (const char* err) {
        printRow("Failed to load network", err);
        return 1;
    }

    printRow("OpenVino Engine is ready...");

    _bratsQSL->loadQuerySamples({ 0 });
    size_t expected_input_size = _engine->inputSize();
    size_t qsl_output_size = _bratsQSL->getFeaturesSize(0);
    if (expected_input_size != qsl_output_size) {
        printRow("Network input size and QSL sample size are not the same");
        printRow("Network input size", expected_input_size);
        printRow("QSL sample size", qsl_output_size);
        _engine = nullptr;
        _bratsQSL = nullptr;
        return 1;
    }
    _bratsQSL->unloadQuerySamples({ 0 });

    mlperf::c::ClientData client_data = 0;

    size_t sample_count = _bratsQSL->size();
    
    if (mlperf_settings.mode == mlperf::TestMode::PerformanceOnly) {
        sample_count = Args::SampleCount() == 0 ? _bratsQSL->size() : Args::SampleCount();

        if (mlperf_settings.performance_sample_count_override > 0) {
            sample_count = mlperf_settings.performance_sample_count_override;
        }
    }

    std::string sut_name = "OV_SUT";
    std::string qsl_name = "BratsQSL";

    void* sut = mlperf::c::ConstructSUT(client_data, sut_name.c_str(), sut_name.size(), issueQueries, flushQueries, processLatencies);
    void* qsl = mlperf::c::ConstructQSL(client_data, qsl_name.c_str(), qsl_name.size(), sample_count, Args::PerfSampleCount(), loadQuerySamples, unloadQuerySamples);

    printRow("Test settings");
    printRow("Scenario", Args::Scenario());
    printRow("Mode", Args::Mode());
    printRow("Samples Count", sample_count);
    printRow("Performance Samples Count", Args::PerfSampleCount());
    printRow("Expected QPS", mlperf_settings.offline_expected_qps);
    printRow("");

    printRow("Starting test...", Args::Scenario());

    mlperf::c::StartTest(sut, qsl, mlperf_settings);

    mlperf::c::DestroyQSL(qsl);
    mlperf::c::DestroySUT(sut);

    _engine = nullptr;
    _bratsQSL = nullptr;

    printRow("Done!", sample_count);

    dumpMlperfLogSummary();

    printRow("");

    return 0;
}
