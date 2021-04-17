// Copyright 2020 Xilinx Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dirent.h>
#include <getopt.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <dirent.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"

/* mlperf headers */
#include <loadgen.h>
#include <query_sample_library.h>
#include <system_under_test.h>
#include <test_settings.h>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

/* header file for Runner APIs */
#include <dpu_runner.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>
//#include "FastMemcpy_Avx.h"
using namespace mlperf;
using namespace std;
using namespace cv;

class InputData {
public:
  InputData(std::string image, size_t size)
  : image_(image) {
    if (posix_memalign(&data_, getpagesize(), size*2))
      throw std::bad_alloc();
  }
  InputData(const InputData&) = delete;
  InputData& operator=(const InputData&&) = delete;
  InputData(InputData&&) = delete;
  InputData& operator=(InputData&&) = delete;
  ~InputData() {
    free(data_);
  }

  const std::string& image() { return image_; }
  void *data() { return data_; }

private:
  std::string image_;
  void *data_;
};

class ImageCache {
  public:  
    static ImageCache& getInst() {
      static ImageCache imageCache;  
      return imageCache;
    }

    void init(std::string imagePath, int numImages) {
      std::string imageListFile = imagePath + "/val_map.txt";
      imageList_.clear();
      imageList_.reserve(numImages);

      std::ifstream f;
      f.open(imageListFile.c_str());
      if (!f.is_open()) // check that image names file exists
      {
        std::cout << "Error: cannot open image list file " << imageListFile << std::endl;
        throw;
      }

      std::string imageFileName;
      unsigned int label;
      for (int i=0; i < numImages && !f.eof(); i++)
      {
        f >> imageFileName;
        f >> label;
        imageList_.push_back(std::make_pair(label, imagePath + "/" + imageFileName));
      }
    }

    void load(int width, int height, const std::vector<QuerySampleIndex> &samples)
    {
      const float mean[3] = {123.68, 116.78, 103.94};
      const float scaleIn = 0.5;
      const int inSize = width * height * 3;

      // populate input data
      inputData_.clear();
      inputData_.resize(samples.size());

      int outHeight = height;
      int outWidth  = width;
      for (auto index : samples) 
      {
        const std::string &filename = imageList_[index].second;
        inputData_[index].reset(new InputData(filename, sizeof(int8_t)*inSize));

        /* pre-process */
        cv::Mat orig = imread(filename);
        cv::cvtColor(orig, orig, cv::COLOR_BGR2RGB);
        float scale = 0.0;
        if (orig.rows > orig.cols)
          scale = 256.0/orig.cols;
        else
          scale = 256.0/orig.rows;
        int new_h = round(orig.rows*scale);
        int new_w = round(orig.cols*scale);

        cv::Mat resizedImage = cv::Mat(new_h, new_w, CV_8SC3);
        cv::resize(orig, resizedImage, cv::Size(new_w, new_h),0,0, cv::INTER_AREA);
        
        /// Center Crop Image
        const int offsetW = (new_w - outWidth) / 2;
        const int offsetH = (new_h - outHeight) / 2;

        const cv::Rect roi(offsetW, offsetH, outWidth, outHeight);
        resizedImage = resizedImage(roi).clone();

        /// Pre-Processing loop
        for (int c = 0; c < 3; c++)
          for (int h = 0; h < outHeight; h++)
            for (int w = 0; w < outWidth; w++) {
              int8_t *ptr = (int8_t*)(inputData_[index]->data());
              ptr[0
                + (3*h*outWidth)
                + (3*w) + c]
                = (resizedImage.at<cv::Vec3b>(h,w)[c]-mean[c])*scaleIn;
        }
      }

      std::cout << "Loaded samples: " << inputData_.size() << std::endl;
    }

    void unload() {
      inputData_.clear();
    }

    InputData& get(int idx) {
      assert(idx < inputData_.size());
      return *inputData_[idx];
    }

    int size() const {
      return inputData_.size(); 
    }

  private:
     std::vector<std::pair<int, std::string>> imageList_;
     std::vector<std::unique_ptr<InputData>> inputData_;
     std::mutex lock_;
};

/**
 * @brief calculate softmax
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void CPUCalcSoftmax(const int8_t *data, size_t size, float *result) {
  assert(data && result);
  const float scaleOut = 0.25;
  double sum = 0.0f;

  for (size_t i = 0; i < size; i++) {
    result[i] = exp(data[i]*scaleOut);
    sum += result[i];
  }
  for (size_t i = 0; i < size; i++) {
    result[i] /= sum;
  }
}

std::vector<float> TopK(const float *d, int size, int k) {
  assert(d && size > 0 && k > 0);
  priority_queue<pair<float, int>> q;

  for (auto i = 0; i < size; ++i) {
    q.push(pair<float, int>(d[i], i));
  }

  std::vector<float> topK;
  for (auto i = 0; i < k; ++i) {
    pair<float, int> ki = q.top();
    //printf("top[%d] prob=%-8f label=%d\n", i, d[ki.second], ki.second);
    topK.push_back(ki.second);
    q.pop();
  }

  return topK;
}

float Top1(const int8_t *d, int size) {
  assert(d && size > 0);
  priority_queue<pair<float, int>> q;

  for (auto i = 0; i < size; ++i) {
    q.push(pair<float, int>(d[i], i));
  }

  return q.top().second;
}

/*
 * Batcher collects samples to build a batch
 * & launches task when batch full (or timeout)
 */
class Batcher {
public:
  Batcher(unsigned batchSz, 
    std::function<void(std::vector<QuerySample>)> task,
    int maxWaitTimeUs=-1) 
    : batch_size_(batchSz), terminate_(false) {
    thread_ = std::thread(&Batcher::batcher, this, task, maxWaitTimeUs);
  }

  ~Batcher() {
    { 
      terminate_ = true;
      std::unique_lock<std::mutex> lock(mtx_);
      cvar_.notify_all();
    }

    thread_.join();
  }

  void add(QuerySample sample) {
    std::unique_lock<std::mutex> lock(mtx_);
    batch_.push_back(sample);
    cvar_.notify_one();
  }

private:
  void batcher(std::function<void(std::vector<QuerySample>)> task,
    int maxWaitTimeUs) {
    const auto timeout = (maxWaitTimeUs > 0) ? 
      std::chrono::microseconds(maxWaitTimeUs) : std::chrono::microseconds(10000);
    const unsigned checkIters = 10;
    const auto checkInterval = timeout / checkIters;

    while (1) {
      std::vector<QuerySample> batch;

      {
        std::unique_lock<std::mutex> lock(mtx_);
        bool timedOut = false;
        while (batch_.size() < batch_size_ && !terminate_ && !timedOut)
          if (cvar_.wait_for(lock, timeout) == std::cv_status::timeout)
            timedOut = true;

        if (batch_.size())
        {
          if (batch_.size() <= batch_size_)
          {
            batch = batch_;
            batch_.clear();
          }
          else
          {
            batch = { batch_.begin(), batch_.begin() + batch_size_ };
            batch_ = { batch_.begin() + batch_size_, batch_.end() };
          }
        }
        else if (terminate_)
          break;
      }

      if (batch.size())
        task(batch);
    }
  }

  unsigned batch_size_;
  std::mutex mtx_;
  std::condition_variable cvar_;
  std::vector<QuerySample> batch_;
  std::atomic<bool> terminate_;
  std::thread thread_;
};

/*
 * TaskRunner runs tasks in a background thread(s)
 */
class TaskRunner {
public:
  TaskRunner(unsigned numWorkers,
    std::function<void(std::vector<QuerySample>)> fn) 
    : terminate_(false) {
    for (unsigned n=0; n < numWorkers; n++)
      task_workers_.emplace_back(
        std::thread(&TaskRunner::taskWorker, this, fn));
  }
  ~TaskRunner() {
    { 
      terminate_ = true;
      std::unique_lock<std::mutex> lock(mtx_);
      cvar_.notify_all();
    }
    for (unsigned wi=0; wi < task_workers_.size(); wi++)
      task_workers_[wi].join();
  }

  void enqueue(std::vector<QuerySample> &batch) {
    std::unique_lock<std::mutex> lock(mtx_);
    task_queue_.push(batch);
    cvar_.notify_one();
  }

  unsigned get_num_workers() { return task_workers_.size(); }

private:
  void taskWorker(std::function<void(std::vector<QuerySample>)> fn) {
    while (1) {
      std::vector<QuerySample> task;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        while (task_queue_.empty() && !terminate_)
          cvar_.wait(lock);
        if (terminate_)
          break;

        task = task_queue_.front();
        task_queue_.pop();
      }

      fn(task);
    }
  }

  std::mutex mtx_;
  std::condition_variable cvar_;
  std::queue<std::vector<QuerySample>> task_queue_;
  std::vector<std::thread> task_workers_;
  std::atomic<bool> terminate_;
};

class XlnxSystemUnderTest : public SystemUnderTest {
public:
  XlnxSystemUnderTest(std::string runnerDir) 
  : runner_tbuf_idx_(0)
  {
    std::string model0 = runnerDir + "/dpu0.xmodel";
    std::string model1 = runnerDir + "/dpu1.xmodel";
    std::unique_ptr<xir::Graph> graph0 = xir::Graph::deserialize(model0);
    auto subgraph0 = get_dpu_subgraph(graph0.get());
    std::unique_ptr<xir::Graph> graph1 = xir::Graph::deserialize(model1);
    auto subgraph1 = get_dpu_subgraph(graph1.get());
    // create one Runner that we'll use from all threads
    runner_first_.reset(new vart::DpuRunner(subgraph0[0]));
    runner_.reset(new vart::DpuRunner(subgraph1[0]));
    input_tensors_ = runner_first_->get_input_tensors();
    output_tensors_ = runner_->get_output_tensors();
    const int batchSize = 16;

    // create worker threads to execute each batch and report results
    taskrunner_.reset(new TaskRunner(/*numWorkerThreads*/8, 
      [this](std::vector<QuerySample> batch) { execBatch(batch); }));

    // pre-allocate TensorBuffers and reuse for lifetime of program
    const unsigned tbufPoolSize  = 44;
    runner_respackets_.resize(tbufPoolSize);
    runner_resdata_.resize(tbufPoolSize);
    for (unsigned bi=0; bi < tbufPoolSize; bi++)
    {
      auto inputs = runner_first_->make_inputs(1);
      auto outputs0 = runner_first_->make_outputs(1,true);
      auto outputs = runner_->make_outputs(1);
      runner_tbufs_.push_back(std::make_tuple(inputs,outputs0, outputs));
      runner_respackets_[bi].resize(batchSize);
      runner_resdata_[bi].resize(batchSize);
    }

    // create Batcher to combine samples from multiple requests,
    // then enqueues combined batch for execution via TaskRunner
    batcher_.reset(new Batcher(batchSize, 
      [this](std::vector<QuerySample> batch) {
      /* 
       * execute a batch of images from Batcher.
       * Note: batch.size() may be less than requested batchSize
       */
      taskrunner_->enqueue(batch);
    }, /*timeoutUs*/5000));
  }

  void execBatch(std::vector<QuerySample> batch) {
    if (batch.empty())
      return;

    // select pre-allocated TensorBuffer from TensorBuffer pool
    const auto tbufIdx = runner_tbuf_idx_.fetch_add(1) % runner_tbufs_.size();
    auto &outTensors =std::get<2>( runner_tbufs_[tbufIdx]);
    auto &outTensors0 =std::get<1>( runner_tbufs_[tbufIdx]);
    auto &inTensors0 =std::get<0>( runner_tbufs_[tbufIdx]);
    QuerySampleResponse *qsr = runner_respackets_[tbufIdx].data();
    float *resData = runner_resdata_[tbufIdx].data();

    // collect input data from ImageCache
    ImageCache &imageCache = ImageCache::getInst();
    std::vector<std::unique_ptr<vart::CpuFlatTensorBuffer> > tbufs;
    std::vector<vart::TensorBuffer*> inTensors;
    
    int fr=batch.size()%2;
//    
    for (int bi=0; bi < batch.size(); bi++) 
    {
      int cnt=bi%2;
      auto &input = imageCache.get(batch[bi].index);
      if (cnt == 0) {
        int buffern=0;
        for (int idx=0;idx<bi;idx+=2) {
          if (batch[idx].index == batch[bi].index) {
            buffern=1;
            //break;
          }
        }
        if (buffern==0) {
          std::unique_ptr<vart::CpuFlatTensorBuffer> tbuf(
            new vart::CpuFlatTensorBuffer(input.data(), 
            (std::get<0>(runner_tbufs_[tbufIdx])[bi/2])->get_tensor()));
          inTensors.push_back(tbuf.get());
          tbufs.emplace_back(std::move(tbuf));
        } else {
          //memcpy_fast((int8_t*)((int8_t*)(inTensors0[bi/2]->data().first)), input.data(),150528);
          memcpy((int8_t*)((int8_t*)(inTensors0[bi/2]->data().first)), input.data(),150528);
          std::unique_ptr<vart::CpuFlatTensorBuffer> tbuf(
            new vart::CpuFlatTensorBuffer((void*)inTensors0[bi/2]->data().first, 
            (std::get<0>(runner_tbufs_[tbufIdx])[bi/2])->get_tensor()));
          inTensors.push_back(tbuf.get());
          tbufs.emplace_back(std::move(tbuf));
       }

      } else {

        //memcpy_fast((int8_t*)((int8_t*)(inTensors[(bi-1)/2]->data().first)+150528), input.data(),150528);
        memcpy((int8_t*)((int8_t*)(inTensors[(bi-1)/2]->data().first)+150528), input.data(),150528);
      }
    }
    for (int bi=(batch.size()+fr)/2; bi<8;bi++) 
    {
      std::unique_ptr<vart::CpuFlatTensorBuffer> tbuf(
        new vart::CpuFlatTensorBuffer((void*)inTensors0[bi]->data().first, 
        (std::get<0>(runner_tbufs_[tbufIdx])[bi])->get_tensor()));
      inTensors.push_back(tbuf.get());
      tbufs.emplace_back(std::move(tbuf));
    }
    auto job_id0 = runner_first_->execute_async(inTensors, outTensors0);
    runner_first_->wait(job_id0.first, -1); // wait for task to finish
    auto job_id = runner_->execute_async(outTensors0, outTensors);
    runner_->wait(job_id.first, -1); // wait for task to finish
    
    // report result
    //const unsigned outSize = output_tensors_[0]->get_element_num();
    for (int i=0; i < batch.size(); i++) 
    {
      std::uint64_t outData;
      size_t sz;
      int cnt=i%2;
      std::tie(outData, sz) = outTensors[(i-cnt)/2]->data();
      if (cnt) {
      const int8_t *fc = (const int8_t*)outData+1001;
      resData[i] = Top1(fc, 1001);
      resData[i] -= 1; // training labels are 1-indexed
      //cout <<sz << endl;
//      if (batch[i].index == 88) {
//        auto &input = imageCache.get(batch[i].index);
//        cout <<(int)((int8_t*)input.data())[0] << (int)((int8_t*)inTensors[(i-cnt)/2]->data().first)[150528] <<(int)fc[0] <<(int)fc[1]<< " " <<  resData[i]<< endl;
//      }
      const QuerySample &sample = batch[i];
      QuerySampleResponse &res = qsr[i];
      res.id = sample.id;
      res.data = (uintptr_t)(resData+i);
      res.size = sizeof(resData[i]);
      QuerySamplesComplete(&res, 1);
      } else {
      const int8_t *fc = (const int8_t*)outData;
      resData[i] = Top1(fc, 1001);
      resData[i] -= 1; // training labels are 1-indexed
      //if (batch[i].index == 88)
      //  cout << resData[i] << endl;

      const QuerySample &sample = batch[i];
      QuerySampleResponse &res = qsr[i];
      res.id = sample.id;
      res.data = (uintptr_t)(resData+i);
      res.size = sizeof(resData[i]);
      QuerySamplesComplete(&res, 1);

      }
    }
  }

  ~XlnxSystemUnderTest() { }

  const std::string &Name() const { return name; }

  void IssueQuery(const std::vector<QuerySample> &samples) {
    // send samples to batcher to exec when batch is full
    // (this function never blocks)
    for (int si=0; si < samples.size(); si++)
      batcher_->add(samples[si]);
  }

  int getWidth() {
    auto inputTensors = runner_first_->get_input_tensors();
    return inputTensors[0]->get_shape()[2]; // NHWC
  }
  int getHeight() {
    auto inputTensors = runner_first_->get_input_tensors();
    return inputTensors[0]->get_shape()[1]; // NHWC
  }

  void FlushQueries() {}

  void ReportLatencyResults(const std::vector<QuerySampleLatency> &/*latencies_ns*/) {}
  std::vector<const xir::Subgraph*> get_dpu_subgraph(
      const xir::Graph* graph) {
    auto root = graph->get_root_subgraph();
    auto children = root->children_topological_sort();
    auto ret = std::vector<const xir::Subgraph*>();
    for (auto c : children) {
      CHECK(c->has_attr("device"));
      auto device = c->get_attr<std::string>("device");
      if (device == "DPU") {
        ret.emplace_back(c);
      }
    }
    return ret;
  }


private:
  const std::string name = "XLNX_AI";
  std::unique_ptr<vart::DpuRunner> runner_;
  std::unique_ptr<vart::DpuRunner> runner_first_;
  std::vector<const xir::Tensor *> input_tensors_;
  std::vector<const xir::Tensor *> output_tensors_;

  // tensorbuffer pool for each Runner -- alloc once and reuse across queries
  std::vector<
    std::tuple<std::vector<vart::TensorBuffer*>, std::vector<vart::TensorBuffer*>,
              std::vector<vart::TensorBuffer*> > > runner_tbufs_;
  std::vector<std::vector<QuerySampleResponse> > runner_respackets_;
  std::vector<std::vector<float> > runner_resdata_;
  std::atomic<unsigned> runner_tbuf_idx_;

  std::unique_ptr<Batcher> batcher_;
  std::unique_ptr<TaskRunner> taskrunner_;
};

class XlnxQuerySampleLibrary : public QuerySampleLibrary {
private:
  const std::string name = "XLNX_AI";
  const int width_;
  const int height_;
  const int numSamples_;

public:
  XlnxQuerySampleLibrary(const std::string &path, int width, int height, int nSamples) 
  : width_(width), height_(height), numSamples_(nSamples) {
    ImageCache &imageCache = ImageCache::getInst();
    imageCache.init(path, nSamples);
  }

  ~XlnxQuerySampleLibrary() {}

  const std::string &Name() const { return name; }

  size_t TotalSampleCount() { 
    return numSamples_;
  }

  size_t PerformanceSampleCount() { 
    return std::min(50000, numSamples_);
  }

  void LoadSamplesToRam(const std::vector<QuerySampleIndex> &samples) override {
    ImageCache::getInst().load(width_, height_, samples);
  }

  void UnloadSamplesFromRam(const std::vector<QuerySampleIndex> &samples) override {
    ImageCache::getInst().unload();
  }
};

/* 
 * Usage: 
 * app.exe <options>
 */
int main(int argc, char **argv) {
  TestSettings testSettings = TestSettings();
  testSettings.scenario = TestScenario::SingleStream;
  testSettings.mode = TestMode::PerformanceOnly;
  testSettings.min_query_count = 270336;
  //testSettings.max_query_count = 10000;
  testSettings.min_duration_ms = 60000;
  testSettings.multi_stream_max_async_queries 
    = testSettings.server_max_async_queries 
    = 1;
  testSettings.multi_stream_target_qps = 20;
  testSettings.multi_stream_samples_per_query = 52;
  testSettings.server_target_latency_ns = 15000000;
  testSettings.server_target_qps = 100;
  testSettings.offline_expected_qps = 100;
  //testSettings.qsl_rng_seed = 12786827339337101903ULL;
  //testSettings.schedule_rng_seed = 3135815929913719677ULL;
  //testSettings.sample_index_rng_seed = 12640797754436136668ULL;
  testSettings.qsl_rng_seed = 7322528924094909334ULL;
  testSettings.schedule_rng_seed = 3507442325620259414ULL;
  testSettings.sample_index_rng_seed = 1570999273408051088ULL;
  
  LogSettings logSettings = LogSettings();
  logSettings.enable_trace = false;

  std::string dpuDir, imgDir;
  int numSamples = -1;
  for(;;) {
    struct option long_options[] = {
      //{"verbose", no_argument,       &verbose_flag, 1},
      {"imgdir", required_argument, 0, 'i'},
      {"dpudir", required_argument, 0, 'd'},
      {"logtrace", no_argument, 0, 'l'},
      {"num_queries", required_argument, 0, 'q'},
      {"num_samples", required_argument, 0, 's'},
      {"max_async_queries", required_argument, 0, 'a'},
      {"min_time", required_argument, 0, 't'},
      {"scenario", required_argument, 0, 'c'},
      {"mode", required_argument, 0, 'm'},
      {"qps", required_argument, 0,'r'},
      {0, 0, 0, 0}
    };
    /* getopt_long stores the option index here. */
    int option_index = 0;
    int c = getopt_long (argc, argv, "a:c:d:i:l:q:s:t:m:r:", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
      case 'a':
        testSettings.multi_stream_max_async_queries 
          = testSettings.server_max_async_queries 
          = atoi(optarg);
      case 'c':
        if (std::string(optarg) == "SingleStream")
          testSettings.scenario = TestScenario::SingleStream;
        else if (std::string(optarg) == "MultiStream")
          testSettings.scenario = TestScenario::MultiStream;
        else if (std::string(optarg) == "Server")
          testSettings.scenario = TestScenario::Server;
        else if (std::string(optarg) == "Offline")
          testSettings.scenario = TestScenario::Offline;
        break;

      case 'd':
        dpuDir = std::string(optarg);
        break;

      case 'i':
        imgDir = std::string(optarg);
        break;

      case 'l':
        logSettings.enable_trace = true;
        break;

      case 'q':
        //testSettings.min_query_count = testSettings.max_query_count = atoi(optarg);
        testSettings.min_query_count = atoi(optarg);
        break;

      case 'r':
        testSettings.server_target_qps = stoi(optarg);
        testSettings.offline_expected_qps = stoi(optarg);
        break;
 
      case 's':
        numSamples = atoi(optarg);
        break;

      case 't':
        testSettings.min_duration_ms = atoi(optarg);
        break;

      case 'm':
        if (std::string(optarg) == "SubmissionRun")
          testSettings.mode = TestMode::SubmissionRun;
        else if (std::string(optarg) == "AccuracyOnly")
          testSettings.mode = TestMode::AccuracyOnly;
        else if (std::string(optarg) == "PerformanceOnly")
          testSettings.mode = TestMode::PerformanceOnly;
    }
  }
  
  if (numSamples < 0)
    numSamples = testSettings.server_max_async_queries * 10;

  std::unique_ptr<XlnxSystemUnderTest> sut(new XlnxSystemUnderTest(
    dpuDir));
  std::unique_ptr<QuerySampleLibrary> qsl(new XlnxQuerySampleLibrary(
    imgDir, sut->getWidth(), sut->getHeight(), numSamples));

  cout << "Query count: " << testSettings.min_query_count << "\n";
  cout << "Sample count: " << numSamples << "\n";
  cout << "Target QPS: " << testSettings.server_target_qps << "\n";
  cout << "Start loadgen\n";
  StartTest(sut.get(), qsl.get(), testSettings, logSettings);

  return 0;
}
