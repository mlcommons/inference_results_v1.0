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

#ifndef __MULTI_QSL_HPP__
#define __MULTI_QSL_HPP__

#include "qsl_cpu.hpp"

// inherits qsl, while adding some callbacks in {Load|Unload}SampleLibrary calls
namespace qsl {
class MIGSampleLibrary : public SampleLibrary
{
  public:
    MIGSampleLibrary(std::string name, std::string mapPath, std::vector<std::string> tensorPaths,
                     size_t perfSampleCount, size_t padding = 0, bool coalesced = false,
                     std::vector<bool> startFromDevice = std::vector<bool>(1, false))
        : SampleLibrary(name, mapPath, tensorPaths, perfSampleCount, padding, coalesced,
                        startFromDevice)
    {
    }

    ~MIGSampleLibrary() {}

    virtual void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples)
    {
        m_load_cb(samples);
    }

    virtual void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples)
    {
        m_unload_cb(samples);
    }

    void registerLoadCB(std::function<void(std::vector<mlperf::QuerySampleIndex>)> cb)
    {
        m_load_cb = cb;
    }
    void registerUnloadCB(std::function<void(std::vector<mlperf::QuerySampleIndex>)> cb)
    {
        m_unload_cb = cb;
    }

  private:
    // callback for {Load|Unload}SamplesToRam
    std::function<void(std::vector<mlperf::QuerySampleIndex>)> m_load_cb;
    std::function<void(std::vector<mlperf::QuerySampleIndex>)> m_unload_cb;
};

typedef std::shared_ptr<qsl::MIGSampleLibrary> MIGSampleLibraryPtr_t;
}; // namespace qsl

#endif // __MULTI_QSL_HPP__
