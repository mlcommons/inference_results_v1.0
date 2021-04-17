#include <iostream>
#include <thread>
#include <algorithm>
#include <fstream>
#include <unistd.h>
#include <string.h>

#include "include/benchmark.h"
#include "include/resnet50.h"
#include "include/bindings/c_api.h"

#include "ioutils.h"

using namespace std;
using std::thread;
using namespace mlperf;
using namespace mlperf::c;
using namespace maccel_type;
using namespace std::chrono_literals;

std::vector<uint8_t *> resnet50::mSamples;
std::string resnet50::mDatasetPath = "./resnet-dataset.txt";

#include <mutex>
#include <condition_variable>
#include <chrono>

/**
 * Worker Thread
 */
void resnet50::worker(const QuerySample* qs, int offset, size_t size, int acc_seq) {
    Model temp = ResNet;
    int max;

    for (size_t i = offset; i < offset+size; i++) {
        int8_t *result;
        int64_t requestId = -1;
        max = -9999999;

        mAccelerator[acc_seq]->allocateMemory({
            { (void **) &result, 1024 }
        });

        vector<Payload> vRB = {
            { 0, (uint8_t *) loadedSample[lookup[qs[i].index]], IMAGE_SIZE }
        };

        vector<Payload> vRBR = {
            { 372736, (uint8_t *) result, 1024 }
        };

        auto start = chrono::system_clock::now();
        requestId = mAccelerator[acc_seq]->request(temp, vRB, vRBR);
        mAccelerator[acc_seq]->receive(requestId);
        auto end = chrono::system_clock::now();
        auto duration = end - start;
        // cout << "Inference Time : " << duration.count() << endl;

        // int max = -9999999;
        int *res_bucket = (int *) malloc(sizeof(int));
        for (int i = 0; i < 1001; i++) {
            if (((int8_t) result[i]) > max) {
                max = (int8_t) result[i];
                *res_bucket = i - 1;
            }
        }

        QuerySampleResponse* resp = new QuerySampleResponse {
            qs[i].id, (uintptr_t) res_bucket, 4 };
        mlperf::c::QuerySamplesComplete(resp, 1);

        free(res_bucket);
        
        if ((uint64_t) qs[i].id == 0) {
            std::cout << "Error shown at " << i << endl;
            continue;
        }
    }
}

#define NUM_ACC 1

void resnet50::issueQuery(
    ClientData cd, const QuerySample* qs, size_t size) {
    cout << "SUT: Issue Query" << endl;

    const int MAX_WORKER_THREAD = 60;
    int num_worker = min((int) size, MAX_WORKER_THREAD);
    size_t samples_per_thread = (int) (size / num_worker);
    int sample_remain = size % num_worker;
    thread worker_thread[num_worker];

    int offset = 0;

    vector<vector<QuerySampleResponse>*> res;
    for (int i = 0; i < num_worker; i++) {
        int splitted = samples_per_thread;
        if (sample_remain-- > 0) {
            splitted++;
        }
        worker_thread[i] = thread(worker, qs, offset, splitted, 
            i % mAccelerator.size());
        worker_thread[i].detach();
        offset += splitted;
    }
}

void resnet50::flushQueries(void) {
    cout << "SUT: Flush Queries" << endl;
}

void resnet50::processLatencies(
    ClientData cd, const int64_t* data, size_t size) {
    cout << "Process Latencies" << endl;
}

void resnet50::loadImageNet(const QuerySampleIndex* qsi, size_t size) {
    cout << "Loading ImageNet Dataset from " << mDatasetPath << 
        ", dataset count is " << size << endl;

    string buf[50000];
    ifstream readFile;
    readFile.open(mDatasetPath);
    int k = 0;

    while (!readFile.eof()) {
        getline(readFile, buf[k++]);
    }

    readFile.close();

    int resolution = 224 * 256;

    for (unsigned int i = 0; i < size; i++) {
        uint8_t buf_img[resolution*3] = {0, };
        int dummy = 0;
        lookup[qsi[i]] = i;

        ReadFile(buf[(int) qsi[i]], buf_img, &dummy);
        
        for (int j=0; j <resolution; j++){
            loadedSample[i][3*j + 0] = buf_img[3*j + 0];
            loadedSample[i][3*j + 1] = buf_img[3*j + 1];
            loadedSample[i][3*j + 2] = buf_img[3*j + 2];
        }
    }

    cout << "ImageNet Loading End!" << endl;
}

void resnet50::loadSamplesToRAM(
    ClientData cd, const QuerySampleIndex* qsi, size_t size) {
    cout << "Load " << size << " samples to RAM." << endl;
    loadedSample = (uint8_t **) malloc(sizeof(uint8_t *) * (int) size);
    for (int i = 0; i < (int) size; i++) {
        posix_memalign((void **)&loadedSample[i], 4096, 
            IMAGE_SIZE + 4096);
    }
    loadImageNet(qsi, size);
}

void resnet50::unloadSamplesFromRAM(
    ClientData cd, const QuerySampleIndex* qsi, size_t size) {
    cout << "SUT: Unload " << size << " samples from RAM." << endl;
    for (auto& sample : mSamples) {
        free(sample);
    }
    mSamples.clear();
}