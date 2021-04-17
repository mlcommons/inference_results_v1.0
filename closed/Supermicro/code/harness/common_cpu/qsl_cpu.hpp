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

#ifndef __QSL_CPU_HPP__
#define __QSL_CPU_HPP__

#include <chrono>
#include <deque>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <vector>

#include "numpy.hpp"
#include "query_sample_library.h"
#include "test_settings.h"

// QSL (Query Sample Library) is an implementation of the MLPerf Query Sample Library.  It's purpose
// is to:
// 1) Allow samples to be loaded and unloaded dynamically at runtime from Loadgen.
// 2) Support lookup of currently loaded tensor addresses in memory.

namespace qsl {
class LookupableQuerySampleLibrary : public mlperf::QuerySampleLibrary
{
  public:
    virtual void* GetSampleAddress(mlperf::QuerySampleIndex sample_index, size_t input_idx,
                                   size_t device_idx = 0) = 0;
    virtual size_t GetSampleSize(size_t input_idx) const = 0;
};
class SampleLibrary : public LookupableQuerySampleLibrary
{
  public:
    SampleLibrary(std::string name, std::string mapPath, std::vector<std::string> tensorPaths,
                  size_t perfSampleCount, size_t padding = 0, bool coalesced = false,
                  std::vector<bool> startFromDevice = std::vector<bool>(1, false))
        : m_Name(name), m_PerfSampleCount(perfSampleCount), m_PerfSamplePadding(padding),
          m_MapPath(mapPath), m_TensorPaths(tensorPaths), m_Coalesced(coalesced)
    {
        m_StartFromDevice.swap(startFromDevice);
        // Get input size and allocate memory
        m_NumInputs = m_TensorPaths.size();
        m_SampleSizes.resize(m_NumInputs);
        m_SampleMemory.resize(m_NumInputs);

        // Initialize npy file caches in coalescing mode
        if(m_Coalesced)
        {
            m_NpyFiles.resize(m_NumInputs);
            for(size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
            {
                m_NpyFiles[input_idx].reset(new npy::NpyFile(m_TensorPaths[input_idx]));
            }
        }

        // Get number of samples
        if(!m_Coalesced)
        {
            // load and read in the sample map
            std::ifstream fs(m_MapPath);
            CHECK(fs) << "Unable to open sample map file: " << m_MapPath;

            char s[1024];
            while(fs.getline(s, 1024))
            {
                std::istringstream iss(s);
                std::vector<std::string> r((std::istream_iterator<std::string>{iss}),
                                           std::istream_iterator<std::string>());

                m_FileLabelMap.insert(std::make_pair(
                    m_SampleCount, std::make_tuple(r[0], (r.size() > 1 ? std::stoi(r[1]) : 0))));
                m_SampleCount++;
            }
        }
        else
        {
            // In coalescing mode, the first dimension is number of samples
            m_SampleCount = m_NpyFiles[0]->getDims()[0];
        }

        // as a safety, don't allow the perfSampleCount to be larger than sampleCount.
        m_PerfSampleCount = std::min(m_PerfSampleCount, m_SampleCount);

        for(size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
        {
            if(!m_Coalesced)
            {
                std::string path =
                    m_TensorPaths[input_idx] + "/" + std::get<0>(m_FileLabelMap[0]) + ".npy";
                npy::NpyFile npy(path);
                m_SampleSizes[input_idx] = npy.getTensorSize();
            }
            else
            {
                m_SampleSizes[input_idx] = m_NpyFiles[input_idx]->getTensorSize() / m_SampleCount;
            }
            m_SampleMemory[input_idx].resize(1);
            m_SampleMemory[input_idx][0] =
                malloc((m_PerfSampleCount + m_PerfSamplePadding) * m_SampleSizes[input_idx]);
        }
    }

    ~SampleLibrary()
    {
        for(size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
        {
            free(m_SampleMemory[input_idx][0]);
        }
    }

    const std::string& Name() const override
    {
        return m_Name;
    }
    size_t TotalSampleCount() override
    {
        return m_SampleCount;
    }
    size_t PerformanceSampleCount() override
    {
        return m_PerfSampleCount;
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        // copy the samples into pinned memory
        if(!m_Coalesced)
        {
            for(size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++)
            {
                auto& sampleId = samples[sampleIndex];
                for(size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
                {
                    std::string path = m_TensorPaths[input_idx] + "/" +
                                       std::get<0>(m_FileLabelMap[sampleId]) + ".npy";
                    npy::NpyFile npy(path);
                    std::vector<char> data;
                    npy.loadAll(data);
                    auto sampleAddress = static_cast<int8_t*>(m_SampleMemory[input_idx][0]) +
                                         sampleIndex * m_SampleSizes[input_idx];
                    memcpy((char*)sampleAddress, data.data(), m_SampleSizes[input_idx]);
                }
            }
        }
        else
        {
            for(size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
            {
                std::vector<char> data;
                m_NpyFiles[input_idx]->loadSamples(data, samples);
                memcpy((char*)m_SampleMemory[input_idx][0], data.data(),
                       samples.size() * m_SampleSizes[input_idx]);
            }
        }

        // construct sample address map
        for(size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++)
        {
            auto& sampleId = samples[sampleIndex];
            m_SampleAddressMapHost[sampleId].push_back(std::vector<void*>(m_NumInputs, nullptr));
            for(size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
            {
                if(!m_StartFromDevice[input_idx])
                {
                    m_SampleAddressMapHost[sampleId].back()[input_idx] =
                        static_cast<int8_t*>(m_SampleMemory[input_idx][0]) +
                        sampleIndex * m_SampleSizes[input_idx];
                }
            }

            m_SampleAddressMapDevice[sampleId].push_back(
                std::vector<std::vector<void*>>(m_NumInputs));
            for(size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
            {
                if(m_StartFromDevice[input_idx])
                {
                    m_SampleAddressMapDevice[sampleId].back()[input_idx].resize(m_NumDevices,
                                                                                nullptr);
                    for(int device_idx = 0; device_idx < m_NumDevices; device_idx++)
                    {
                        m_SampleAddressMapDevice[sampleId].back()[input_idx][device_idx] =
                            static_cast<int8_t*>(m_SampleMemory[input_idx][device_idx]) +
                            sampleIndex * m_SampleSizes[input_idx];
                    }
                }
            }
        }
    }

    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        // due to the removal of freelisting this code is currently a check and not required for
        // functionality.
        for(auto& sampleId : samples)
        {
            {
                auto it = m_SampleAddressMapHost.find(sampleId);
                CHECK(it != m_SampleAddressMapHost.end())
                    << "Sample: " << sampleId << " not allocated properly";
                auto& sampleAddresses = it->second;
                CHECK(!sampleAddresses.empty()) << "Sample: " << sampleId << " not loaded";
                sampleAddresses.pop_back();
                if(sampleAddresses.empty())
                {
                    m_SampleAddressMapHost.erase(it);
                }
            }
            {
                auto it = m_SampleAddressMapDevice.find(sampleId);
                CHECK(it != m_SampleAddressMapDevice.end())
                    << "Sample: " << sampleId << " not allocated properly";
                auto& sampleAddresses = it->second;
                CHECK(!sampleAddresses.empty()) << "Sample: " << sampleId << " not loaded";
                sampleAddresses.pop_back();
                if(sampleAddresses.empty())
                {
                    m_SampleAddressMapDevice.erase(it);
                }
            }
        }

        CHECK(m_SampleAddressMapHost.empty() && m_SampleAddressMapDevice.empty())
            << "Unload did not remove all samples";
    }

    virtual void* GetSampleAddress(mlperf::QuerySampleIndex sample_index, size_t input_idx,
                                   size_t device_idx = 0)
    {
        if(!m_StartFromDevice[input_idx])
        {
            auto it = m_SampleAddressMapHost.find(sample_index);
            CHECK(it != m_SampleAddressMapHost.end())
                << "Sample: " << sample_index << " missing from RAM";
            CHECK(input_idx <= it->second.front().size()) << "invalid input_idx";
            return it->second.front()[input_idx];
        }

        auto it = m_SampleAddressMapDevice.find(sample_index);
        CHECK(it != m_SampleAddressMapDevice.end())
            << "Sample: " << sample_index << " missing from Device RAM";
        CHECK(input_idx <= it->second.front().size()) << "invalid input_idx";
        return it->second.front()[input_idx][device_idx];
    }

    virtual size_t GetSampleSize(size_t input_idx) const
    {
        return (m_SampleSizes.empty() ? 0 : m_SampleSizes[input_idx]);
    }

  protected:
    size_t m_NumInputs{0};
    int m_NumDevices{1};

  private:
    const std::string m_Name;
    size_t m_PerfSampleCount{0};
    size_t m_PerfSamplePadding{0};
    std::string m_MapPath;
    std::vector<std::string> m_TensorPaths;
    bool m_Coalesced;
    std::vector<bool> m_StartFromDevice;
    std::vector<size_t> m_SampleSizes;
    std::vector<std::vector<void*>> m_SampleMemory;
    std::vector<std::unique_ptr<npy::NpyFile>> m_NpyFiles;
    size_t m_SampleCount{0};
    // maps sampleId to <fileName, label>
    std::map<mlperf::QuerySampleIndex, std::tuple<std::string, size_t>> m_FileLabelMap;
    // maps sampleId to num_inputs of <address>
    std::map<mlperf::QuerySampleIndex, std::vector<std::vector<void*>>> m_SampleAddressMapHost;
    // maps sampleId to num_inputs of num_devices of <address>
    std::map<mlperf::QuerySampleIndex, std::vector<std::vector<std::vector<void*>>>>
        m_SampleAddressMapDevice;

    nvinfer1::DataType m_Precision{nvinfer1::DataType::kFLOAT};
};

typedef std::shared_ptr<qsl::SampleLibrary> SampleLibraryPtr_t;

class SampleLibraryEnsemble : public mlperf::QuerySampleLibrary
{
  public:
    SampleLibraryEnsemble(const std::vector<SampleLibraryPtr_t> qsls) : m_qsls(qsls){};
    const std::string& Name() const override
    {
        return m_qsls[0]->Name();
    }
    size_t TotalSampleCount() override
    {
        return m_qsls[0]->TotalSampleCount();
    }
    size_t PerformanceSampleCount() override
    {
        return m_qsls[0]->PerformanceSampleCount();
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        for(auto qsl : m_qsls)
        {
            qsl->LoadSamplesToRam(samples);
        }
    }
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        for(auto qsl : m_qsls)
        {
            qsl->UnloadSamplesFromRam(samples);
        }
    }

  private:
    std::vector<SampleLibraryPtr_t> m_qsls;
};

}; // namespace qsl

#endif // __QSL_HPP__
