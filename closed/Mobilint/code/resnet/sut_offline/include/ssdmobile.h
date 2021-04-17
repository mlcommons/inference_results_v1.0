#ifndef MLPERF_V_1_0_INCLUDE_SSDMOBILE_H_
#define MLPERF_V_1_0_INCLUDE_SSDMOBILE_H_
#include "maccel.h"
#include "loadgen.h"
#include "query_sample.h"
#include "bindings/c_api.h"

#include <vector>
#include <string>

namespace ssd_mobilenet {
    const int CLIENT_DATA = 0;
    const int TOTAL_SAMPLE_COUNT = 5000;
    const int QSL_SIZE = 256;
    const int IMAGE_SIZE = 300 * 320 * 3;
    const int MAX_WORKER_THREAD = 80;

    extern std::vector<uint8_t *> mSamples;
    extern std::string mDatasetPath;

    void worker(
        const mlperf::QuerySample* qs, int offset, size_t size, int acc_seq);
    void onComplete(uint64_t ticket);

    void issueQuery(mlperf::c::ClientData cd, 
        const mlperf::QuerySample* qs, size_t size);
    void flushQueries(void);
    void processLatencies(mlperf::c::ClientData cd, 
        const int64_t* data, size_t size);

    void loadCOCO(const mlperf::QuerySampleIndex* qsi, size_t size);
    void loadSamplesToRAM(mlperf::c::ClientData cd, 
        const mlperf::QuerySampleIndex* qsi, size_t size);
    void unloadSamplesFromRAM(mlperf::c::ClientData cd, 
        const mlperf::QuerySampleIndex* qsi, size_t size);

    void hw_worker();
    void issue_query(mlperf::c::ClientData cd, 
        const mlperf::QuerySample* qs, size_t size);
};

#endif