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
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <map>

#include "utils.h"

std::map<mlperf::TestScenario, std::string> _scenarios = {
    { mlperf::TestScenario::Offline, "Offline" },
    { mlperf::TestScenario::SingleStream, "SingleStream" },
    { mlperf::TestScenario::MultiStream, "MultiStream" },
    { mlperf::TestScenario::MultiStreamFree, "MultiStreamFree" },
    { mlperf::TestScenario::Server, "Server" }
};

std::map<mlperf::TestMode, std::string> _modes = {
    { mlperf::TestMode::AccuracyOnly, "Accuracy" },
    { mlperf::TestMode::PerformanceOnly, "Performance" },
    { mlperf::TestMode::SubmissionRun, "Submission" },
    { mlperf::TestMode::FindPeakPerformance, "FindPeakPerformance" }
};

void printRow(const std::string& col1, const std::string& col2) {
    std::cout << std::setw(30) << std::left << col1;

    if (!col2.empty()) {
        std::cout << " : " << col2;
    }

    std::cout << std::endl;
}

void printRow(const std::string& col1, int col2) {
    printRow(col1, std::to_string(col2));
}

bool fileExists(const std::string& path) {
    std::ifstream file(path);
    if (file.good()) {
        file.close();
        return true;
    }
    return false;
}

 bool stringToScenario(const std::string& str, mlperf::TestScenario& scenario) {
    for (auto scn : _scenarios) {
        if (scn.second == str) {
            scenario = scn.first;
            return true;
        }
    }
    return false;
}

 bool stringToMode(const std::string& str, mlperf::TestMode& mode) {
    for (auto mod : _modes) {
        if (mod.second == str) {
            mode = mod.first;
            return true;
        }
    }
    return false;
}

void dumpMlperfLogSummary() {
    std::ifstream log("mlperf_log_summary.txt");
    if (log.good()) {
        std::string line;
        while (std::getline(log, line)) {
            std::cout << line << std::endl;
        }
    }
}
