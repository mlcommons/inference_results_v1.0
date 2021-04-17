#include <iostream>
#include <thread>
#include <algorithm>
#include <fstream>
#include <unistd.h>
#include <string.h>

#include "include/benchmark.h"
#include "include/ssdmobile.h"
#include "include/bindings/c_api.h"
#include "ioutils.h"

#include <mutex>
#include <condition_variable>

#include <chrono>
#include <memory>
#include <vector>

using namespace std;
using std::thread;
using namespace mlperf;
using namespace mlperf::c;
using namespace maccel_type;
using namespace std::chrono_literals;

std::vector<uint8_t *> ssd_mobilenet::mSamples;
std::string ssd_mobilenet::mDatasetPath = "./ssd-mobilenet-dataset.txt";

/**
 * Worker Thread
 */
void ssd_mobilenet::worker(const QuerySample* qs, int offset, size_t size, int acc_seq) {
    // uint8_t* loadedSample;
    Model temp = SSDMobileNet;

    for (size_t i = offset; i < offset+size; i++) {
        uint64_t requestId;
        int8_t *total;
        int8_t *cls0, *cls1, *cls2, *cls3, *cls4, *cls5;
        int8_t *box0, *box1, *box2, *box3, *box4, *box5;
        vector<float> boxes, classes, scores;

        int ret;

        mAccelerator[acc_seq]->allocateMemory({
            { (void **)&total, 228608 }
            
        });

        box0 = total + 115520;
        box1 = total + 196224;
        box2 = total + 217024;
        box3 = total + 223808;
        box4 = total + 227712;
        box5 = total + 228544;
        cls0 = total +      0;
        cls1 = total + 138624;
        cls2 = total + 202624;
        cls3 = total + 218624;
        cls4 = total + 225408;
        cls5 = total + 227968;

        vector<Payload> vRB = {
            { 0, (uint8_t *) loadedSample[lookup[qs[i].index]], 300 * 320 * 3 }
        };

        vector<Payload> vRBR = {
            { 791616, (uint8_t *) total, 228608 }
        };

        auto start = chrono::system_clock::now();
        requestId = mAccelerator[acc_seq]->request(temp, vRB, vRBR);
        mAccelerator[acc_seq]->receive(requestId);
        auto end = chrono::system_clock::now();
        auto duration = end - start;
        // cout << "Inference Time : " << duration.count() << endl;
        
        auto start_1 = chrono::system_clock::now();
        requestId = mPostprocessor[acc_seq]->enqueue(
            cls0, cls1, cls2, cls3, cls4, cls5,
            box0, box1, box2, box3, box4, box5,
            boxes, classes, scores);
        mPostprocessor[acc_seq]->receive(requestId);
        auto end_1 = chrono::system_clock::now();
        auto duration_1 = end_1 - start_1;
        // cout << "Postprocessing Time : " << duration_1.count() << endl;

        unsigned char* alloc_dyn = (unsigned char *) malloc(28 * scores.size());
        unsigned char* offset = alloc_dyn;
        memset(alloc_dyn, 0, 28 * scores.size());

        for (int j = 0; j < scores.size(); j++) {
            float *t = (float *) malloc (sizeof(float));
            *t = qs[i].index;
            memcpy(offset, (unsigned char*) t, 4);
            offset += 4;
            free(t);

            memcpy(offset, (unsigned char*) &(boxes[4*j + 1]), 4);
            offset += 4;
            memcpy(offset, (unsigned char*) &(boxes[4*j + 0]), 4);
            offset += 4;
            memcpy(offset, (unsigned char*) &(boxes[4*j + 3]), 4);
            offset += 4;
            memcpy(offset, (unsigned char*) &(boxes[4*j + 2]), 4);
            offset += 4;

            memcpy(offset, (unsigned char*) &(scores[j]), 4);
            offset += 4;
            memcpy(offset, (unsigned char*) &(classes[j]), 4);
            offset += 4;
        }

        QuerySampleResponse* result = new QuerySampleResponse {
            qs[i].id, (uintptr_t) alloc_dyn, 28 * scores.size() };

        mlperf::c::QuerySamplesComplete(result, 1);
        
        free(alloc_dyn);
        free(total);

        if ((uint64_t) qs[i].id == 0) {
            cout << "Error shown at " << i << endl;
            continue;
        }
    }
}

void ssd_mobilenet::issueQuery(
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

void ssd_mobilenet::flushQueries(void) {
    cout << "SUT: Flush Queries" << endl;
}

void ssd_mobilenet::processLatencies(
    ClientData cd, const int64_t* data, size_t size) {
    cout << "Process Latencies" << endl;
}

void ssd_mobilenet::loadCOCO(const QuerySampleIndex* qsi, size_t size) {
    cout << "Loading COCO Dataset from " << mDatasetPath 
        << ", dataset count is " << size << endl;
        
    string buf[5000];
    ifstream readFile;

    int k = 0;
    readFile.open(mDatasetPath);
    while (!readFile.eof()) {
        getline(readFile, buf[k++]);
    }
    readFile.close();

    int resolution = 300 * 320;
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

    cout << "COCO Loading End!" << endl;
}

void ssd_mobilenet::loadSamplesToRAM(
    ClientData cd, const QuerySampleIndex* qsi, size_t size) {
    cout << "Load " << size << " samples to RAM." << endl;
    loadedSample = (uint8_t **) malloc(sizeof(uint8_t *) * (int) size);
    for (int i = 0; i < (int) size; i++) {
        posix_memalign((void **)&loadedSample[i], 4096, 
            IMAGE_SIZE + 4096);
    }

    loadCOCO(qsi, size);
}

void ssd_mobilenet::unloadSamplesFromRAM(
    ClientData cd, const QuerySampleIndex* qsi, size_t size) {
    cout << "SUT: Unload " << size << " samples from RAM." << endl;
    for (auto& sample : mSamples) {
        free(sample);
    }
    mSamples.clear();
}