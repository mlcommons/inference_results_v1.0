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
#include <vector>
#include <map>
#include <tuple>
#include <utility>

#include "pickler/pickler.h"
#include "mmap.h"

class BratsQSL {
private:
    using Tuple = std::tuple<MmapAllocator::Ptr, float*, size_t>;
    using AllocatorMap = std::map<size_t, Tuple>;

public:
    using Ptr = std::shared_ptr<BratsQSL>;

public:
    BratsQSL(const std::string& list_path) {
        parseList(list_path);
    }

public:
    const void loadQuerySamples(const std::vector<size_t>& sample_list) {
        for (size_t idx : sample_list) {
            std::string file_path = _files[idx];

            size_t offset = 0;
            size_t size = 0;

            if (!pickler::Parser::dataOffsetAndSize(_files[idx], offset, size)) {
                std::string err = "Can not find data offset and size for file: " + _files[idx];
                throw err;
            }

            MmapAllocator::Ptr alloc = MmapAllocator::Ptr(new MmapAllocator(file_path));
            alloc->alloc();

            float* ptr = alloc->data<float>(offset);

            _allocators[idx] =std::make_tuple(alloc, (float*)ptr, size);

            std::string filename = file_path;
            const size_t last_slash_idx = filename.find_last_of("\\/");
            if (std::string::npos != last_slash_idx) {
                filename.erase(0, last_slash_idx + 1);
            }
        }
    }

    const void unloadQuerySamples(const std::vector<size_t>& sample_list) {
        for (size_t idx : sample_list) {
            AllocatorMap::iterator it = _allocators.find(idx);
            if (it != _allocators.end()) {
                _allocators.erase(it);
            }
        }
    }

    float* getFeatures(size_t sample_id) {
        checkAllocator(sample_id);
        return std::get<1>(_allocators[sample_id]);
    }

    size_t getFeaturesSize(size_t sample_id) {
        checkAllocator(sample_id);
        return std::get<2>(_allocators[sample_id]);
    }

    size_t size() {
        return _files.size();
    }

private:
    void checkAllocator(size_t sample_id) {
        if (_allocators.find(sample_id) == _allocators.end()) {
            std::string err = "Sample with id " + std::to_string(sample_id) + " not loaded";
            throw err;
        }
    }
    void parseList(const std::string& list_path) {
        std::string dir_path;

        const size_t sep = list_path.rfind(pathSeparator());
        if (sep != std::string::npos) {
            dir_path = list_path.substr(0, sep + 1);
        }

        std::vector<std::string> files = pickler::Parser::list(list_path);

        for (std::string path : files) {
            std::string full_path = dir_path + path + ".pkl";
            if (!isFileExist(full_path)) {
                std::string err = "File does not exist: " + full_path;
                throw err;
            }
            _files.push_back(full_path);
        }
    }

    bool isFileExist(const std::string& path) {
        std::ifstream infile(path);
        return infile.good();
    }

    char pathSeparator() {
#ifdef _WIN32
        return '\\';
#else
        return '/';
#endif
    }

private:
    AllocatorMap _allocators;
    std::vector<std::string> _files;
};
