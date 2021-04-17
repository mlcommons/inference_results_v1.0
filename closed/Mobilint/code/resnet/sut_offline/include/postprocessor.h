#include <queue>
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <array>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;

typedef struct SSDPostItem {
    uint64_t id;
    int8_t *cls0; int8_t *cls1; int8_t *cls2; int8_t *cls3; int8_t *cls4; int8_t *cls5; 
    int8_t *box0; int8_t *box1; int8_t *box2; int8_t *box3; int8_t *box4; int8_t *box5;
    vector<float>& boxes; vector<float>& classes; vector<float>& scores;
} SSDPostItem;

/**
 * Central Postprocessor for SSD-MobileNet.
 */
class PostprocessorManager {
private:
    thread mThread;
    uint8_t mPPType = 0;

    queue<SSDPostItem> mQueueIn;
    vector<uint64_t> mOut;
    uint64_t ticket = 0;
    
    mutex mMutexIn;
    mutex mMutexOut;
    condition_variable mCondIn;
    condition_variable mCondOut;

    bool destroyed = false;
    const int ch_depth_size = 64;

    float* prior_generation();
    float *priors_cpp = prior_generation();

    void generate_offset_array_cls(int offset_arr[][546]);
    void generate_offset_array_box(int offset_arr[][24]);

    int offset_array_cls[6][546];
    int offset_array_box[6][24];
    
    
    void worker();
    void reshape(int loop, int total_height, int total_channel, int8_t *dest, int8_t *src);
    void reshape_new(int grid_index, int is_cls, int total_height, int total_channel, int8_t *dest, int8_t *src);
    float area(float xmin, float ymin, float xmax, float ymax);
    float calculate_iou(array<float, 4> box1, array<float, 4> box2);
    float* decode(float* locations, float* priors);
    int filter_results(float* init_scores, float* init_boxes, vector<float> &boxes_final, vector<float> &classes_final, vector<float> &scores_final, float nms_threshold);
    void transpose_and_copy(float *boxes_float, float *clses_float, 
                        int8_t *clses0, int8_t *clses1, int8_t *clses2, int8_t *clses3, int8_t *clses4, int8_t *clses5,
                        int8_t *boxes0, int8_t *boxes1, int8_t *boxes2, int8_t *boxes3, int8_t *boxes4, int8_t *boxes5);
    int postprocessing(
        int8_t *clses0, int8_t *clses1, int8_t *clses2, int8_t *clses3, int8_t *clses4, int8_t *clses5, 
        int8_t *boxes0, int8_t *boxes1, int8_t *boxes2, int8_t *boxes3, int8_t *boxes4, int8_t *boxes5,
        vector<float> &boxes, vector<float> &classes, vector<float> &scores);

public:
    const int PP_SSD_MOBILENET = 1;
    const int PP_SSD_RESNET = 2;

    PostprocessorManager();
    ~PostprocessorManager();
    
    uint64_t enqueue(
        int8_t *cls0, int8_t *cls1, int8_t *cls2, int8_t *cls3, int8_t *cls4, int8_t *cls5, 
        int8_t *box0, int8_t *box1, int8_t *box2, int8_t *box3, int8_t *box4, int8_t *box5,
        vector<float> &boxes, vector<float> &classes, vector<float>& scores);
    
    void receive(uint64_t receipt_no);
};