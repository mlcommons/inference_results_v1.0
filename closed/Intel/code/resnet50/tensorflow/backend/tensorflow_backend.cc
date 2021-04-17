#include <iostream>
#include <string>
#include <vector>
#include "inferencer.h"

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


#include "data_provider.h"
#include "classifier.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;


// Parse flags
bool dummy_data = false;
bool use_accuracy_layer = false;
bool quantized = false;
bool random_multibatch = false;
int batch_size_ = 1;
int warmup_times_ = 0;
int log_level = 0;
int num_intra_threads = 2;
int num_inter_threads = 1;
std::string data_order("NHWC");
std::string net_conf("");
std::string images("");
std::string labels("");
std::string file_list("");
std::string init_net_path("");
std::string predict_net_path("");
std::string numa_id("0");
std::string shared_memory_option("USE_SHM");
std::string shared_weight("USE_SHM");


std::string datapath, model;

uint64_t constexpr mix(char m, uint64_t s) {
    return ((s<<7) + ~(s>>3)) + ~m;
}

uint64_t constexpr hash(const char * m) {
    return (*m) ? mix(*m,hash(m+1)) : 0;
}

class BackendTensorflow : public loadrun::Inferencer {
  public:
  BackendTensorflow() {
    std::cout << "BackendTensorflow Inferencer is opened\n";
  }

  ~BackendTensorflow() {
    data_provider_->clean_shared_memory(numa_id);
    std:: cout << "Inferencer cleaned shared mem\n";
    delete classifier_;
    delete data_provider_;
    std::cout << "BackendTensorflow Inferencer is closed\n";
  }

  void initialize(int argc, char **argv, bool contiguous_in_ram) {

    // Parse flag
    parse_flag(argc, argv);

    contiguous_in_ram_ = contiguous_in_ram;

    // data provider can read from image path and handle data transformation process
    // like preprocess and shared memory control.
    data_provider_ = new tensorflow::DataProvider<float>(file_list, images, labels,
                              batch_size_, data_order, dummy_data,
                              net_conf, shared_memory_option, numa_id,
                              total_samples_, contiguous_in_ram_);

    // should give available iterations to construct the classifer.
    classifier_ = new tensorflow::Classifier<float>(data_provider_->get_data(),
                              data_provider_->get_labels(), init_net_path,
                              predict_net_path, batch_size_,
                              data_order, use_accuracy_layer, 0, net_conf,
                              quantized, log_level, shared_weight, numa_id,
                              num_intra_threads, num_inter_threads,
                              total_samples_, dummy_data);

    classifier_->warmup(warmup_times_, total_samples_, contiguous_in_ram_);
    std::cout << "warmup done\n";
  }

  void prepare_batch(const int index){
    classifier_->prepare_batch(index, contiguous_in_ram_);
  }

  void run(int iteration, bool random){
    classifier_->run();
  }

  void accuracy(int iteration, bool random){
    classifier_->accuracy(iteration, random);
  }

  std::vector<int> get_labels(int iteration, bool random) {
    return classifier_->get_labels(iteration, random);
  }

  void load_sample(size_t* samples, size_t sample_size) {
    if (contiguous_in_ram_) {
      data_provider_->load_sample(samples, total_samples_);
    }
  }

  void getInfo(double* hd_seconds, float* top1, float* top5) {
    classifier_->getInfo(hd_seconds, top1, top5);
  }

  void getInfo(double* run_seconds, double* hd_seconds, float* top1, float* top5) {
    classifier_->getInfo(run_seconds, hd_seconds, top1, top5);
  }

  void parse_flag(int argc, char** argv) {
    for(int index = 0; index < argc; index++) {
      switch(hash(argv[index])){
        case hash("--dummy_data"):{
          std::string temp(argv[++index]);
          if (temp.compare("true") == 0){
            dummy_data = true;
          } else {
            dummy_data = false;
          }
          break;
        }
        case hash("--use_accuracy_layer"):{
          std::string temp(argv[++index]);
          if (temp.compare("true") == 0){
            use_accuracy_layer = true;
          } else {
            use_accuracy_layer = false;
          }
          break;
        }
        case hash("--quantized"):{
          std::string temp(argv[++index]);
          if (temp.compare("true") == 0){
            quantized = true;
          } else {
            quantized = false;
          }
          break;
        }
        case hash("--random_multibatch"):{
          std::string temp(argv[++index]);
          if (temp.compare("true") == 0){
            random_multibatch = true;
          } else {
            random_multibatch = false;
          }
          break;
        }
        case hash("--batch_size"):{
          batch_size_ = std::stoi(argv[++index]);
          break;
        }
        case hash("--w"):{
          warmup_times_ = std::stoi(argv[++index]);
          break;
        }
        case hash("--log_level"):{
          log_level = std::stoi(argv[++index]);
          break;
        }
        case hash("--data_order"):{
          data_order = std::string(argv[++index]);
          break;
        }
        case hash("--net_conf"):{
          net_conf = std::string(argv[++index]);
          break;
        }
        case hash("--images"):{
          images = std::string(argv[++index]);
          break;
        }
        case hash("--labels"):{
          labels = std::string(argv[++index]);
          break;
        }
        case hash("--file_list"):{
          file_list = std::string(argv[++index]);
          break;
        }
        case hash("--init_net_path"):{
          init_net_path = std::string(argv[++index]);
          break;
        }
        case hash("--predict_net_path"):{
          predict_net_path = std::string(argv[++index]);
          break;
        }
        case hash("--numa_id"):{
          numa_id = std::string(argv[++index]);
          break;
        }
        case hash("--shared_memory_option"):{
          shared_memory_option = std::string(argv[++index]);
          break;
        }
        case hash("--shared_weight"):{
          shared_weight = std::string(argv[++index]);
          break;
        }
        case hash("--total_samples"):{
          total_samples_ = std::stoi(argv[++index]);
          break;
        }
        case hash("--num_intra_threads"):{
          num_intra_threads = std::stoi(argv[++index]);
          break;
        }
        case hash("--num_inter_threads"):{
          num_inter_threads = std::stoi(argv[++index]);
          break;
        }
        default:
          break;
      }
    }
  }

  private:
  tensorflow::DataProvider<float> *data_provider_ = nullptr;
  tensorflow::Classifier<float> *classifier_ = nullptr;
  bool contiguous_in_ram_ = false;
  int total_samples_ = 1;
};

std::unique_ptr<loadrun::Inferencer> get_inferencer() {
  return std::unique_ptr<BackendTensorflow>(new BackendTensorflow());
}
