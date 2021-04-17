#ifndef CLASSIFIER_H__
#define CLASSIFIER_H__
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <sstream>
#include <string>
#include <vector>


#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "net_config.h"

namespace tensorflow {

using namespace boost::interprocess;

template<typename T>
class Classifier {
  public:
    Classifier(T* input_data, const vector<int>& labels,
               const string& init_net_path, const string& predict_net_path,
               const int batch_size, const string& data_order, const bool use_accuracy_layer,
               const int thread_id, const string& net_conf,
               const bool quantized, const int log_level,
               const string& shared_memory_option, const string& numa_id,
               const int num_intra_threads, const int num_inter_threads,
               const int num_loaded_sample, const bool dummy_data);
    ~Classifier();
    void load_model(const string graph_file_name);
    void warmup(int warmup_times, const int total_samples, const bool contiguous_in_ram);
    void create_new_tensor();
    void prepare_batch(const int index, const bool contiguous_in_ram);
    void run(void);
    void postProcess(void);
    void getInfo(double* hd_seconds, float* top1, float* top5);
    void getInfo(double* run_seconds, double* hd_seconds, float* top1, float* top5);
    void AccuracyCompute();
    void accuracy(const int iteration, const bool random = false);
    std::vector<int> get_labels(int iteration = 0, const bool random = true);

  private:
    std::unique_ptr<NetConf> net_conf_;
    string device_type_;
    std::vector<int> labelsOut_;
    T* inputData_;
    managed_shared_memory managed_shm_;
    const vector<int>& labels_;
    const bool quantized_ = false;
    const bool accuracy_ = false;
    const bool dummy_data_ = false;
    const int log_level_ = 0;
    int thread_id_ = 0;
    int thread_step_ = 0;
    int batchSize_ = 1;
    int batchIdx_ = 0;
    int warmup_idx_ = 0;
    int random_count_ = 0;
    int num_intra_threads_ = 2;
    int num_inter_threads_ = 1;
    int num_loaded_sample_ = 1;
    unsigned long long  inputSize_ = 0;

    // Accuracy and performance Info
    float top1_ = 0;
    float top5_ = 0;
    double hd_seconds_ = 0;
    double runtime_seconds_ = 0;

    string input_node_;
    string output_node_;
    Tensor input_;
    std::vector<Tensor> outputs_;
    std::vector<int> expected_labels_;

    string dataOrder_ = "NCHW";
    string sharedMemory_;
    string numaId_;

    std::unique_ptr<tensorflow::Session> session_;
};

template<typename T>
Classifier<T>::Classifier(T* input_data, const vector<int>& labels,
               const string& init_net_path, const string& predict_net_path,
               const int batch_size, const string& data_order, const bool use_accuracy_layer,
               const int thread_id, const string& net_conf,
               const bool quantized, const int log_level,
               const string& shared_memory_option, const string& numa_id,
               const int num_intra_threads, const int num_inter_threads,
               const int num_loaded_sample, const bool dummy_data)
  : inputData_(input_data),
    labels_(labels),
    quantized_(quantized),
    accuracy_(use_accuracy_layer),
    num_loaded_sample_(num_loaded_sample),
    dummy_data_(dummy_data),
    log_level_(log_level),
    thread_id_(thread_id),
    batchSize_(batch_size),
    num_intra_threads_(num_intra_threads),
    num_inter_threads_(num_inter_threads),
    dataOrder_(data_order),
    sharedMemory_(shared_memory_option),
    numaId_(numa_id) {
  // Override intra threads from environment variable.
  const char* val = std::getenv("NUM_INTRA_THREADS");
  if (val)
    num_intra_threads_ = atoi(val);
  net_conf_ = get_net_conf(net_conf);
  LOG(INFO) << "net name is " << net_conf_->net_name;
  inputSize_ = net_conf_->channels * net_conf_->height * net_conf_->width;
  expected_labels_.resize(batchSize_);
  labelsOut_.resize(batchSize_, -1);

  input_node_ = net_conf_->input_node;
  output_node_ = net_conf_->output_node;

  load_model(init_net_path);

  create_new_tensor();
}

template<typename T>
Classifier<T>::~Classifier() {
}

template<typename T>
void Classifier<T>::load_model(const string graph_file_name) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    std::cout << "Failed to load graph at: " << graph_file_name;
  }

  tensorflow::SessionOptions options;
  tensorflow::ConfigProto & config = options.config;
  
  if (num_intra_threads_ != 1 || num_inter_threads_ != 1) {
    std::cout << "\n\n intra: " << num_intra_threads_ << "\t inter: " << num_inter_threads_ << "\n\n";
    config.set_intra_op_parallelism_threads(num_intra_threads_);
    config.set_inter_op_parallelism_threads(num_inter_threads_);
    config.set_use_per_session_threads(false);
  }

  session_.reset(tensorflow::NewSession(options));
  Status session_create_status = session_->Create(graph_def);
  if (!session_create_status.ok()) {
    std::cout << "Failed to create TF session";
  }
}

template<typename T>
void Classifier<T>::create_new_tensor() {
  std::vector<int> dims;

  if (dataOrder_ == "NHWC") {
    dims = {batchSize_, net_conf_->height, net_conf_->width, net_conf_->channels};
  } else if (dataOrder_ == "NCHW") {
    dims = {batchSize_, net_conf_->channels, net_conf_->height, net_conf_->width};
  } else {
    LOG(ERROR) << "unknown tensorflow format.";
    return;
  }

  Tensor image_tensor(DT_FLOAT, TensorShape(
      {dims[0], dims[1], dims[2], dims[3]}));

  input_ = image_tensor;
}

template<typename T>
void Classifier<T>::prepare_batch(const int index, const bool contiguous_in_ram) {
  // int inputSize_ = net_conf_->channels * net_conf_->height * net_conf_->width;
  
  float *imgTensorFlat = input_.flat<float>().data();
  float *imgTensorPtr;
  float *buffer_ptr;

  if (contiguous_in_ram) {
    // index is batch index
    int imgTensorOffset = 0;
    int remain = batchSize_;
    int start = (index * batchSize_) % num_loaded_sample_;
    int length = 0;
    int idx = 0;

    while (remain > 0) {
      if (start + remain > num_loaded_sample_) {
        length = num_loaded_sample_ - start;
      } else {
        length = remain;
      }

      for (int i = 0; i < length; i++) {
        expected_labels_[idx] = labels_[start + i] + net_conf_->label_offset;
        idx++;
      }

      // Memcpy from memory to Tensor
      imgTensorPtr = imgTensorFlat + imgTensorOffset * inputSize_;
      buffer_ptr = ((float *)inputData_) + start * inputSize_;

      memcpy(imgTensorPtr, buffer_ptr, length * inputSize_ * sizeof(float));

      start = (start + length) % num_loaded_sample_;
      imgTensorOffset += length;
      remain -= length;
    }
  } else {
    // Store query index for current batch
    expected_labels_[batchIdx_] = labels_[index] + net_conf_->label_offset;

    // Memcpy from memory to Tensor
    imgTensorPtr = imgTensorFlat + batchIdx_ * inputSize_;
    buffer_ptr = ((float *)inputData_) + index * inputSize_;

    memcpy(imgTensorPtr, buffer_ptr, inputSize_ * sizeof(float));

    batchIdx_++;
  }
}

template<typename T>
void Classifier<T>::run() {

#if 0
  # Use for debug
  int batch_size = input_.shape().dim_size(0);
  int h = input_.shape().dim_size(1);
  int w = input_.shape().dim_size(2);
  int c = input_.shape().dim_size(3);
  int dims = input_.shape().dims();
#endif

  Status run_status = session_->Run({{input_node_, input_}},
      {output_node_}, {}, &outputs_);

  batchIdx_ = 0;

  if (!run_status.ok()) {
    std::cout << "Running inference failed" << run_status;
  }
}

template<typename T>
void Classifier<T>::AccuracyCompute() {

  int dims = outputs_[0].shape().dims();

  if (dims == 1) {
    /* Top-1 accuracy*/
    int batch_size = outputs_[0].shape().dim_size(0);
    assert(batch_size == batchSize_);

    auto finalOutputTensor = outputs_[0].tensor<int64, 1>();

    for (int i = 0; i < batchSize_; i++) {
      labelsOut_[i] = finalOutputTensor(i) - net_conf_->label_offset;

      if (expected_labels_[i] == finalOutputTensor(i)) {
        top1_++;
      }
    }
  } else if (dims == 2) {
    /* TODO: Top-5 accuracy*/
  }
}

template<typename T>
void Classifier<T>::accuracy(const int iteration, const bool random) {
  AccuracyCompute();
}

template<typename T>
void Classifier<T>::warmup(int warmup_times, const int total_samples, const bool contiguous_in_ram) {
  LOG(INFO) << "Warmup ..." << warmup_times;
  for (int i = 0; i < warmup_times && i * batchSize_ < total_samples; i++) {
    if (contiguous_in_ram) {
      prepare_batch(i, contiguous_in_ram);
    } else {
      for (int j = warmup_idx_; j < warmup_idx_ + batchSize_; j++) {
        prepare_batch(j, contiguous_in_ram);
      }
      warmup_idx_ += batchSize_;
    }
    run();
  }

  warmup_idx_ = 0;
}

template<typename T>
std::vector<int> Classifier<T>::get_labels(int iteration, const bool random) {
  accuracy(iteration, random);

  return labelsOut_;
}

template<typename T>
void Classifier<T>::getInfo(double* hd_seconds, float* top1, float* top5) {
  *top1 = top1_;
  *top5 = top5_;
  *hd_seconds = hd_seconds_;
}

template<typename T>
void Classifier<T>::getInfo(double* run_seconds, double* hd_seconds, float* top1, float* top5) {
  *top1 = top1_;
  *top5 = top5_;
  *hd_seconds = hd_seconds_;
  *run_seconds = runtime_seconds_;
}
} // using namespace tensorflow
#endif // CLASSIFIER_H__
