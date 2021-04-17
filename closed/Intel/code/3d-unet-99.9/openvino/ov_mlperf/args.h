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

#include <string>
#include <gflags/gflags.h>

class Args {
public:
    static void ShowUsage();
    
    static std::string ModelXml();
    static std::string ModelBin();

    static std::string DataPath();

    static std::string Device();

    static std::string Scenario();
    static std::string Mode();

    static uint32_t SampleCount();
    static uint32_t PerfSampleCount();

    static std::string MlperfConfigPath();
    static std::string MlperfUserConfigPath();

    static uint32_t Streams();
    static uint32_t Threads();

    static bool Help();
};
