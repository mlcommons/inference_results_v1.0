#include "include/postprocessor.h"
using namespace std;
/*
    multiply by scales and store into two array(boxes_float, clses_float)
*/
void PostprocessorManager::transpose_and_copy(float *boxes_float, float *clses_float, 
                        int8_t *clses0, int8_t *clses1, int8_t *clses2, int8_t *clses3, int8_t *clses4, int8_t *clses5,
                        int8_t *boxes0, int8_t *boxes1, int8_t *boxes2, int8_t *boxes3, int8_t *boxes4, int8_t *boxes5) {
    int lengths_boxes[6] = {4332, 2400, 600, 216, 96, 24};
    int lengths_clses[6] = {98553, 54600, 13650, 4914, 2184, 546};
    int indexes_boxes[7] = {0, 4332, 6732, 7332, 7548, 7644, 7668};
    int indexes_clses[7] = {0, 98553, 153153, 166803, 171717, 173901, 174447};
    float scales_boxes[6] = {0.13619571595680058, 0.0516454216063492, 0.036114073175144944, 0.02869759769890252, 0.023611947307436484, 0.02606797781516248};
    float scales_clses[6] = {0.26736167847640874, 0.1832739612248939, 0.1885884202371432, 0.1553749324768547, 0.13253335126741664, 0.13969923004390686};

    #pragma omp parallel for
    for (int i=0; i<lengths_boxes[0]; i++){
        boxes_float[indexes_boxes[0]+i] = boxes0[i] * scales_boxes[0];
    }
    #pragma omp parallel for
    for (int i=0; i<lengths_boxes[1]; i++){
        boxes_float[indexes_boxes[1]+i] = boxes1[i] * scales_boxes[1];
    }
    #pragma omp parallel for
    for (int i=0; i<lengths_boxes[2]; i++){
        boxes_float[indexes_boxes[2]+i] = boxes2[i] * scales_boxes[2];
    }
    #pragma omp parallel for
    for (int i=0; i<lengths_boxes[3]; i++){
        boxes_float[indexes_boxes[3]+i] = boxes3[i] * scales_boxes[3];
    }
    #pragma omp parallel for
    for (int i=0; i<lengths_boxes[4]; i++){
        boxes_float[indexes_boxes[4]+i] = boxes4[i] * scales_boxes[4];
    }
    #pragma omp parallel for
    for (int i=0; i<lengths_boxes[5]; i++){
        boxes_float[indexes_boxes[5]+i] = boxes5[i] * scales_boxes[5];
    }

    #pragma omp parallel for
    for (int i=0; i<lengths_clses[0]; i++){
        clses_float[indexes_clses[0]+i] = clses0[i] * scales_clses[0];
    }
    #pragma omp parallel for
    for (int i=0; i<lengths_clses[1]; i++){
        clses_float[indexes_clses[1]+i] = clses1[i] * scales_clses[1];
    }
    #pragma omp parallel for
    for (int i=0; i<lengths_clses[2]; i++){
        clses_float[indexes_clses[2]+i] = clses2[i] * scales_clses[2];
    }
    #pragma omp parallel for
    for (int i=0; i<lengths_clses[3]; i++){
        clses_float[indexes_clses[3]+i] = clses3[i] * scales_clses[3];
    }
    #pragma omp parallel for
    for (int i=0; i<lengths_clses[4]; i++){
        clses_float[indexes_clses[4]+i] = clses4[i] * scales_clses[4];
    }
    #pragma omp parallel for
    for (int i=0; i<lengths_clses[5]; i++){
        clses_float[indexes_clses[5]+i] = clses5[i] * scales_clses[5];
    }
}

/*
    Calculates area of a box, box coordinates are given as input
*/
float PostprocessorManager::area(float xmin, float ymin, float xmax, float ymax) {
    float width = xmax - xmin;
    float height = ymax - ymin;

    if(width < 0)
        return 0;

    if(height < 0)
        return 0;

    return width * height;
}


/*
    Calculates Intersection Over Union(IoU) between two boxes. 
    Coordinates of two boxes are given as input.
*/
float PostprocessorManager::calculate_iou(array<float, 4> box1, array<float, 4> box2) {
    float epsilon = 1 / 100000;

    // Coordinated of the overlapped region(intersection of two boxes)
    float overlap_xmin = max(box1[0], box2[0]);
    float overlap_ymin = max(box1[1], box2[1]);
    float overlap_xmax = min(box1[2], box2[2]);
    float overlap_ymax = min(box1[3], box2[3]);

    // Calculate areas
    float overlap_area = area(overlap_xmin, overlap_ymin, overlap_xmax, overlap_ymax);
    float area1 = area(box1[0], box1[1], box1[2], box1[3]);
    float area2 = area(box2[0], box2[1], box2[2], box2[3]);
    float iou = overlap_area / (area1 + area2 - overlap_area + epsilon);

    return iou;
}


/*
    Prior Generation.
*/
float* PostprocessorManager::prior_generation() {
    double start_priors = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now().time_since_epoch()).count();
    int num_layers = 6;
    // float fig_size[2] = {300, 300};
    float feat_size[num_layers][2] = {{19, 19}, {10, 10}, {5, 5}, {3, 3}, {2, 2}, {1, 1}};

    float min_scale = 0.2;
    float max_scale = 0.95;
    float scales[num_layers + 1];
    for(int i = 0; i < num_layers; i++){
        scales[i] = min_scale + (max_scale - min_scale) *i / (num_layers - 1);
    }
    scales[num_layers] = 1;
    // scales:  [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0]

    // -1 means that we only use the first number
    float aspect_ratio[6][2] = {{2, -1}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}};


    vector<float> priors;
    for(int i = 0; i < 6; i++){
        float sk1 = scales[i];
        float sk2 = scales[i+1];
        float sk3 = sqrt(sk1 * sk2);

        vector<pair<float, float> > all_sizes;
        if(i == 0)
            all_sizes.push_back(make_pair(0.1, 0.1));
        else
            all_sizes.push_back(make_pair(sk1, sk1));

        for(int j = 0; j < 2; j++){
            if(aspect_ratio[i][j] != -1){
                float w = sk1 * sqrt(aspect_ratio[i][j]);
                float h = sk1 / sqrt(aspect_ratio[i][j]);

                all_sizes.push_back(make_pair(w,h));
                all_sizes.push_back(make_pair(h,w));
            }
        }

        if(i != 0)
            all_sizes.push_back(make_pair(sk3, sk3));

    
        int range = (int) all_sizes.size();
        for(int k = 0; k < (int) feat_size[i][0]; k++){
            for(int l = 0; l < (int) feat_size[i][0]; l++){
                for(int j = 0; j < range; j++){
                    float cx = (l + 0.5) / feat_size[i][1];
                    float cy = (k + 0.5) / feat_size[i][0];
                    float w = all_sizes[j].first;
                    float h = all_sizes[j].second;

                    priors.push_back(cx);
                    priors.push_back(cy);
                    priors.push_back(w);
                    priors.push_back(h);
                }
            }
        }
    }

    int num_boxes = (int) priors.size();
    static float final_priors[7668];
    for(int i = 0; i < num_boxes; i++){
        final_priors[i] = priors[i];
    }

    double stop_priors = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now().time_since_epoch()).count();
    //cout << "Prior Generation Time: " << stop_priors - start_priors << endl;
    
    return final_priors;
}

/*
    generate offset array for cls reshape
*/
void PostprocessorManager::generate_offset_array_cls(int offset_arr[][546]){
    int offset;
    int index_ch_depth;
    int index_ch_offset;

    int grid_size[6] = {19, 10, 5, 3, 2, 1};

	for (int i = 0; i < 6; i++){
		for (int j = 0; j < 546; j++){
            index_ch_depth = (int) j / ch_depth_size;
            index_ch_offset = j  % ch_depth_size;
            offset = index_ch_depth * grid_size[i] * grid_size[i] * ch_depth_size + index_ch_offset;
                
            offset_arr[i][j] = offset;
		}
	}
}

/*
    generate offset array for box reshape
*/
void PostprocessorManager::generate_offset_array_box(int offset_arr[][24]){
    int offset;
    int index_ch_depth;
    int index_ch_offset;

    int grid_size[6] = {19, 10, 5, 3, 2, 1};

	for (int i = 0; i < 6; i++){
		for (int j = 0; j < 24; j++){
            index_ch_depth = (int) j / ch_depth_size;
            index_ch_offset = j  % ch_depth_size;
            offset = index_ch_depth * grid_size[i] * grid_size[i] * ch_depth_size + index_ch_offset;
                
            offset_arr[i][j] = offset;
		}
	}
}


/*
    Decode models regression outputs, change from offset(dy,dx,dh,dw) form to relative(0-1) coordinates in corner form(xmin, ymin, xmax, ymax)
*/
float* PostprocessorManager::decode(float* locations, float* priors) {
    float weights[4] = {0.1, 0.1, 0.2, 0.2};
    int num_boxes = 1917;

    for(int i=0; i < num_boxes; i++){
        int idx = 4*i;
        
        float pred_ctr_x = locations[idx+1] * weights[1] * priors[idx+2] + priors[idx];
        float pred_ctr_y = locations[idx] * weights[0] * priors[idx+3] + priors[idx+1];
        float pred_widths = exp(locations[idx+3] * weights[3]) * priors[idx+2] * 0.5;
        float pred_heights = exp(locations[idx+2] * weights[3]) * priors[idx+3] * 0.5;

        locations[idx] = pred_ctr_x - pred_widths;
        locations[idx+1] = pred_ctr_y - pred_heights;
        locations[idx+2] = pred_ctr_x + pred_widths;
        locations[idx+3] = pred_ctr_y + pred_heights;
    }

    return locations; 
}

/*
    This function decodes boxes, filters(mask) inference results with score_thresold = 0.3 and applies NMS(nms_threshold = 0.5) 
*/
int PostprocessorManager::filter_results(float* init_scores, float* init_boxes, vector<float> &boxes_final, vector<float> &classes_final, vector<float> &scores_final, float nms_threshold) {
    int length = 1917;
    int num_output = 0;
    
    init_boxes = decode(init_boxes, priors_cpp);
    
    // Do Filtering and NMS per category
    // We start with 1, 0 stands for BACKGROUND

    int NUM_THREADS = 20;
    vector<float > output_scores[NUM_THREADS];
    vector<float > output_labels[NUM_THREADS];
    vector<array<float, 4> > output_boxes[NUM_THREADS];

    #pragma omp parallel for default(none) shared(output_scores, output_labels, output_boxes, init_scores, init_boxes, nms_threshold, length)
    for(int i = 1; i < 91; i++){
        // Create new vector to store scores with indices
        pair<float, int> masked_scores[1917];
        int woha_indx = 0;
        
        for(int j = 0; j < length; j++){
            float temp_score = init_scores[j * 91 + i];
            if(temp_score > 0.3){
                // init_boxes and init_scores are model detection and classification outputs in one-dimensional array
                // in Python implementation they have following shapes - (1, 1917, 4) and (1, 1917, 91) respectively
                // thus, init_boxes[j * 4 + k] stands for j'th row and k'th column of Python boxes array
                // similarly, init_scores[j * 9 + i] stands for j'th row and i'th column of Python scores array 
                masked_scores[woha_indx] = make_pair(temp_score, j * 4);
                woha_indx = woha_indx + 1;
            }
        }

        // NMS Part
        // Sort Scores
        sort(masked_scores, masked_scores + woha_indx, greater <>());
        
        int _size = woha_indx;
        for(int idx = 0; idx < _size; idx++){
            // Checks whether we discarded this box(score) before or not
            if(masked_scores[idx].first > 0){
                int index_max_score = masked_scores[idx].second;

                // Add the box with highest score to arrays which contains final results
                int output_idx = omp_get_thread_num();
                output_labels[output_idx].push_back(i);
                output_scores[output_idx].push_back(masked_scores[idx].first);
                array<float, 4> max_box = {init_boxes[index_max_score + 0], init_boxes[index_max_score + 1], init_boxes[index_max_score + 2], init_boxes[index_max_score + 3]};
                output_boxes[output_idx].push_back(max_box);

                // Remove from list, discard
                // We will not face this box anymore
                masked_scores[idx].first = -99;
                
                for(int j = idx+1; j < _size; j++){
                    if(masked_scores[j].first > 0){
                        // index which corresponds to the following score
                        int index = masked_scores[j].second;
                        array<float, 4> temp_box = {init_boxes[index + 0], init_boxes[index + 1], init_boxes[index + 2], init_boxes[index + 3]};
                        
                        float iou = calculate_iou(max_box, temp_box);

                        if(iou > nms_threshold){
                            masked_scores[j].first = -99;
                        }
                    }
                }
            }
        }
    }

    for(int i = 0; i < NUM_THREADS; i++){
        for(int j = 0; j < (int) output_scores[i].size(); j++){
            scores_final.push_back(output_scores[i][j]);
            classes_final.push_back(output_labels[i][j]);
            boxes_final.push_back(output_boxes[i][j][0]);
            boxes_final.push_back(output_boxes[i][j][1]);
            boxes_final.push_back(output_boxes[i][j][2]);
            boxes_final.push_back(output_boxes[i][j][3]);

            num_output ++;
        }
    }
    return num_output;
}

int PostprocessorManager::postprocessing(int8_t *clses0, int8_t *clses1, int8_t *clses2, int8_t *clses3, int8_t *clses4, int8_t *clses5,
                int8_t *boxes0, int8_t *boxes1, int8_t *boxes2, int8_t *boxes3, int8_t *boxes4, int8_t *boxes5,
                vector<float> &boxes, vector<float> &classes, vector<float> &scores) {
    omp_set_dynamic(0);
    omp_set_num_threads(10);

    int num_output;

    int boxes_length = 7668;
    int clses_length = 174447;
    float boxes_float[boxes_length];
    float clses_float[clses_length];
    
    boxes.clear();
    classes.clear();
    scores.clear();

    transpose_and_copy(boxes_float, clses_float, 
                    clses0, clses1, clses2, clses3, clses4, clses5,
                    boxes0, boxes1, boxes2, boxes3, boxes4, boxes5);

    #pragma omp parallel for
    for (int i=0; i < clses_length; i++){
        clses_float[i] = 1 / (1+exp(-clses_float[i]));
    }
    
    num_output = filter_results(clses_float, boxes_float, boxes, classes, scores, 0.5);

    return num_output;
}

PostprocessorManager::PostprocessorManager() {
    mThread = thread(&PostprocessorManager::worker, this);

    generate_offset_array_cls(offset_array_cls);
    generate_offset_array_box(offset_array_box);
}

PostprocessorManager::~PostprocessorManager() {
    destroyed = true;
    mCondIn.notify_all();
    mCondOut.notify_all();

    mThread.join();
}

void PostprocessorManager::reshape(int loop, int total_height, int total_channel, int8_t *dest, int8_t *src) {
    int depth_scale = (total_height*total_height*ch_depth_size);

    #pragma omp parallel for
    for (int i = 0; i < loop; i++){
        int idx, depth, height, width, ch, a, b, c;
        // depth  = (int) i / (total_height*total_height*ch_depth_size);
        depth  = (int) i / depth_scale;
        // a      = i % (total_height*total_height*ch_depth_size);
        a      = i % depth_scale;
        height = (int) a / (total_height*ch_depth_size);
        b      = a % (total_height*ch_depth_size);
        width  = (int) b / ch_depth_size;
        c      = (int) b % ch_depth_size;
        ch     = depth*ch_depth_size + c;

        if (ch >= total_channel){
            continue;
        }

        idx = height*total_height*total_channel + width*total_channel + ch;
        dest[idx] = src[i];
    }
}

void PostprocessorManager::reshape_new(int grid_index, int is_cls, int total_height, int total_channel, int8_t *dest, int8_t *src) {
    if (is_cls){
        #pragma omp parallel for
        for (int i = 0; i < total_height*total_height*total_channel; i++){
            int src_idx, ch, wh_index;
            ch = i % total_channel;
            wh_index = (int) i / total_channel;
            src_idx = offset_array_cls[grid_index][ch] + ch_depth_size * wh_index;

            dest[i] = src[src_idx];
        }
    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < total_height*total_height*total_channel; i++){
            int src_idx, ch, wh_index;
            ch = i % total_channel;
            wh_index = (int) i / total_channel;
            src_idx = offset_array_box[grid_index][ch] + ch_depth_size * wh_index;

            dest[i] = src[src_idx];
        }
    }
}


void PostprocessorManager::worker() {
    omp_set_dynamic(0);
    omp_set_num_threads(10);

#ifdef DEBUG
    cout << "Postprocessor: Worker Thread Started" << endl;
#endif
    while (!destroyed) {
        unique_lock<mutex> lk(mMutexIn);
        if (mQueueIn.empty()) {
            mCondIn.wait(lk, [this] {
                return !mQueueIn.empty() || destroyed;
            });
        }

        if (destroyed) {
            break;
        }

        auto k = mQueueIn.front();
        mQueueIn.pop();
        lk.unlock();

        int8_t cls0_[19*19*273], cls1_[10*10*546], cls2_[5*5*546], cls3_[3*3*546], cls4_[2*2*546], cls5_[1*1*546];
        int8_t box0_[19*19* 12], box1_[10*10* 24], box2_[5*5* 24], box3_[3*3* 24], box4_[2*2* 24], box5_[1*1* 24];

        auto start = std::chrono::system_clock::now();
        // unlock on enqueue
        // reshape(115'520, 19, 273, cls0_, k.cls0);
        // reshape( 57'600, 10, 546, cls1_, k.cls1);
        // reshape( 14'400,  5, 546, cls2_, k.cls2);
        // reshape(  5'184,  3, 546, cls3_, k.cls3);
        // reshape(  2'304,  2, 546, cls4_, k.cls4);
        // reshape(    576,  1, 546, cls5_, k.cls5);

        // reshape( 23'104, 19,  12, box0_, k.box0);
        // reshape(  6'400, 10,  24, box1_, k.box1);
        // reshape(  1'600,  5,  24, box2_, k.box2);
        // reshape(    576,  3,  24, box3_, k.box3);
        // reshape(    256,  2,  24, box4_, k.box4);
        // reshape(     64,  1,  24, box5_, k.box5);

        reshape_new(0, 1, 19, 273, cls0_, k.cls0);
        reshape_new(1, 1, 10, 546, cls1_, k.cls1);
        reshape_new(2, 1,  5, 546, cls2_, k.cls2);
        reshape_new(3, 1,  3, 546, cls3_, k.cls3);
        reshape_new(4, 1,  2, 546, cls4_, k.cls4);
        reshape_new(5, 1,  1, 546, cls5_, k.cls5);

        reshape_new(0, 0, 19,  12, box0_, k.box0);
        reshape_new(1, 0, 10,  24, box1_, k.box1);
        reshape_new(2, 0,  5,  24, box2_, k.box2);
        reshape_new(3, 0,  3,  24, box3_, k.box3);
        reshape_new(4, 0,  2,  24, box4_, k.box4);
        reshape_new(5, 0,  1,  24, box5_, k.box5);
        
        auto end = chrono::system_clock::now();
        auto elapsed = end - start;
        // cout << "Postprocessor: Reshape done in " << elapsed.count() << "(ns)." << endl;

        int num_output;
        start = std::chrono::system_clock::now();
        num_output = postprocessing(
            cls0_, cls1_, cls2_, cls3_, cls4_, cls5_, 
            box0_, box1_, box2_, box3_, box4_, box5_, 
            k.boxes, k.classes, k.scores
        );
        end = chrono::system_clock::now();
        elapsed = end - start;
        // cout << "Postprocessor: Main postprocessing done in " << elapsed.count() << "(ns)." << endl;

#ifdef DEBUG
        cout << "Postprocessor: sizeof boxes " << k.boxes.size() << " classes " << k.classes.size() << endl;
#endif
        unique_lock<mutex> lk2(mMutexOut);
        mOut.push_back(k.id);
        lk2.unlock();

        unique_lock<mutex> lk_(mMutexOut); // JUST IN CASE
        mCondOut.notify_all();
        lk_.unlock(); // JUST IN CASE
    }
#ifdef DEBUG
    cout << "Postprocessor: Worker thread finished." << endl;
#endif
}

uint64_t PostprocessorManager::enqueue(
    int8_t *cls0, int8_t *cls1, int8_t *cls2, int8_t *cls3, int8_t *cls4, int8_t *cls5, 
    int8_t *box0, int8_t *box1, int8_t *box2, int8_t *box3, int8_t *box4, int8_t *box5,
    vector<float> &boxes, vector<float> &classes, vector<float> &scores) {
#ifdef DEBUG
        cout << "Postprocessor: Enqueue" << endl;
#endif  
    uint64_t ticket_save = 0;

    {
        lock_guard<mutex> lk(mMutexIn);
        mQueueIn.push({
            ++ticket, cls0, cls1, cls2, cls3, cls4, cls5, box0, box1, box2, box3, box4, box5, boxes, classes, scores
        });
        ticket_save = ticket;

        mCondIn.notify_all(); // JUST IN CASE
#ifdef DEBUG
        cout << "Postprocessor: Input Q size " << mQueueIn.size() << endl;
#endif
    }

    // mCondIn.notify_all(); RESTORE IT JUST IN CASE

    return ticket_save;
}

void PostprocessorManager::receive(uint64_t receipt_no) {
    while (!destroyed) {
#ifdef DEBUG
        cout << "Postprocessor: receive " << receipt_no << " wait.." << endl;
#endif
        unique_lock<mutex> lk(mMutexOut);
        
        if (mOut.empty()){
            mCondOut.wait(lk, [this] {
               return !(mOut.empty()) || destroyed;
            });
        }
        
#ifdef DEBUG
        cout << "Postprocessor: Received something! Recv Q Size " << mOut.size() << endl;
#endif
        if (destroyed) {
            break;
        }

        for (int i = 0; i < mOut.size(); i++) {
            if (mOut[i] == receipt_no) {
#ifdef DEBUG
                cout << "Postprocessor: I got my food " << receipt_no << mOut[i] << endl;
#endif
                mOut.erase(mOut.begin() + i);
                return;
            }
        }

        lk.unlock();
    }
}
