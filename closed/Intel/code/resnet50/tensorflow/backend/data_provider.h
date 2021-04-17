#ifndef DATAPROVIDER_H__
#define DATAPROVIDER_H__
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include "net_config.h"

namespace tensorflow {

using namespace boost::interprocess;

template<typename T>
class DataProvider {
  public:
    DataProvider(const string& file_list, const string& image_path,
                 const string& label_path, const int batch_size,
                 const string& data_order, const bool is_dummy_data,
                 const string& net_conf, const string& shared_memory_option,
                 const string& numa_id, const int total_samples,
                 const bool contiguous_in_ram);
    ~DataProvider();
    T* get_data() { return inputData_; }
    const vector<int>& get_labels() { return labels_; }
    void clean_shared_memory(const string& numa_id);
    void load_sample(size_t* sample, size_t sample_size);
  private:
    std::unique_ptr<NetConf> net_conf_;
    T* inputData_;
    managed_shared_memory managed_shm_;
    int batchSize_ = 1;
    int sample_offset_ = 0;
    int total_samples_ = 1;
    unsigned long long inputSize_ = 0;
    string dataOrder_ = "NCHW";
    string sharedMemory_;
    string numaId_;
    vector<int> input_shape_;
    vector<string> imgNames_;
    vector<int> labels_;
    vector<T> inputImgs_;
    // use for mlperf random index load sample
    const size_t IMAGENET_IMAGE_SIZE = 50000;
    bool contiguous_in_ram_ = false;
    vector<T> loadBuffer_;

    void ParseImageLabel(const string& file_list, const string& image_path,
                         const string& label_path, const size_t sample_size,
                         const bool is_dummy_data);
    // methods for preprocessing
    void SetMeanScale();
    void CenterCrop(cv::Mat* sample_resized, cv::Mat* sample_roi);
    void ResizeWithAspect(cv::Mat* sample, cv::Mat* sample_resized);
    void ResizeWithRescale(cv::Mat* sample, cv::Mat* sample_resized);
    void PreprocessSingleIteration(T* inputImgs,
                    const vector<string>& imgNames);
    void PreprocessUsingCVMethod(T* inputImgs,
                    const vector<string>& imgNames);
    void Preprocess(const bool dummy, T* inputImgs,
                    const vector<string>& imgNames);
    // methods for memory used
    void WrapInput(const bool is_dummy_data);
    void WrapSHMInput(const bool is_dummy_data);
    void CreateUseSharedMemory(const bool is_dummy_data);
    void DirectUseSharedMemory(const bool is_dummy_data);
    void WrapLocalInput(const bool is_dummy_data);
    void CleanSharedMemory(const string& numa_id);
};

template<typename T>
DataProvider<T>::DataProvider(const string& file_list, const string& image_path,
                 const string& label_path, const int batch_size,
                 const string& data_order, const bool is_dummy_data,
                 const string& net_conf, const string& shared_memory_option,
                 const string& numa_id, const int total_samples,
                 const bool contiguous_in_ram)
  : inputData_(nullptr),
    batchSize_(batch_size),
    dataOrder_(data_order),
    sharedMemory_(shared_memory_option),
    numaId_(numa_id),
    total_samples_(total_samples),
    contiguous_in_ram_(contiguous_in_ram) {
  net_conf_ = get_net_conf(net_conf);

  size_t parse_size = total_samples_;
  ParseImageLabel(file_list, image_path, label_path, parse_size, is_dummy_data);
  WrapInput(is_dummy_data);
}

template<typename T>
DataProvider<T>::~DataProvider() {

}

template<typename T>
void DataProvider<T>::load_sample(size_t* samples, size_t sample_size) {
  if (contiguous_in_ram_) {
    vector<int> sample_labels(sample_size, 0);
    if (sharedMemory_ == "CREATE_USE_SHM" || sharedMemory_ == "USE_LOCAL") {
#pragma omp parallel for
      for (size_t i = 0; i < sample_size; ++i) {
        std::memcpy(loadBuffer_.data() + i * inputSize_,
                    inputData_ + samples[i] * inputSize_,
                    inputSize_ * sizeof(T));
        sample_labels[i] = labels_[samples[i]];
      }
      std::memcpy(inputData_, loadBuffer_.data(), sample_size * inputSize_ * sizeof(T));
      labels_ = sample_labels;
      
      if (sharedMemory_ == "CREATE_USE_SHM") {
        *(managed_shm_.find<bool>(("SharedMemorySwap" + numaId_).c_str()).first) = true;
      }
    } else {
      int temp_status = 0;
      // check whether images has been preprocessed
#pragma omp parallel for
      for (size_t i = 0; i < sample_size; ++i) {
        sample_labels[i] = labels_[samples[i]];
      }
      labels_ = sample_labels;
      while (!(*(managed_shm_.find<bool>(("SharedMemorySwap" + numaId_).c_str()).first))) {
        if (temp_status == 0) {
          LOG(INFO) << "image swapping not ready, wait image memory swapping completed"; 
          temp_status++;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }
}

template<typename T>
void DataProvider<T>::clean_shared_memory(const string& numa_id) {
  CleanSharedMemory(numa_id);
}

template<typename T>
void DataProvider<T>::CleanSharedMemory(const string& numa_id) {
  shared_memory_object::remove(("SharedMemory" + numa_id).c_str());
}

template<typename T>
void DataProvider<T>::CreateUseSharedMemory(const bool is_dummy_data) {
  const size_t TOTAL_IMAGE_SIZE = 50001 * 8;
  managed_shm_ = managed_shared_memory(open_or_create,
    ("SharedMemory" + numaId_).c_str() , TOTAL_IMAGE_SIZE * inputSize_);
  // check whether shared memory has prepared target image data, if not, prepare target data.
  auto shared_image_size = managed_shm_.find_or_construct<int>(("SharedImageSize" + numaId_).c_str())(0);
  managed_shm_.find_or_construct<bool>(("SharedMemorySwap" + numaId_).c_str())(false);
  const allocator<T, managed_shared_memory::segment_manager>
    alloc_inst(managed_shm_.get_segment_manager());
  auto shared_input_images =
    managed_shm_.find_or_construct<vector<T,
      allocator<T,managed_shared_memory::segment_manager>>>(("SharedInputImgs" + numaId_).c_str())(alloc_inst);
  // do preprocess only when shared memory don't has enough images buffered.

  if (*shared_image_size != total_samples_) {
    size_t parse_size = total_samples_;
    if ((total_samples_ < IMAGENET_IMAGE_SIZE) && contiguous_in_ram_)
      parse_size = IMAGENET_IMAGE_SIZE;
    shared_input_images->resize(parse_size * inputSize_);
    inputData_ = shared_input_images->data();
    Preprocess(is_dummy_data, inputData_, imgNames_);
    *shared_image_size = total_samples_;
  } else {
    inputData_ = shared_input_images->data();
  }
}

template<typename T>
void DataProvider<T>::DirectUseSharedMemory(const bool is_dummy_data) {
  int temp_status = 0;
  while (temp_status == 0) {
    try {
      managed_shm_ = managed_shared_memory(open_only, ("SharedMemory" + numaId_).c_str());
      temp_status = 1;
    }
    catch(boost::interprocess::interprocess_exception) {
      LOG(INFO) << "check whether shared memory created, use CREATE_USE_SHM in command line"; 
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  temp_status = 0;
  // check whether images has been preprocessed
  while (*(managed_shm_.find<int>(("SharedImageSize" + numaId_).c_str()).first)
         != total_samples_) {
    if (temp_status == 0) {
      LOG(INFO) << "shared image size not satisfied, wait preprocess completed"; 
      temp_status++;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto shared_input_images = managed_shm_.find<vector<T, allocator<T,
       managed_shared_memory::segment_manager>>>(("SharedInputImgs" + numaId_).c_str());
  inputData_ = shared_input_images.first->data();
}

template<typename T>
void DataProvider<T>::WrapSHMInput(const bool is_dummy_data) {
  LOG(INFO) << "use shared memory";
  if (sharedMemory_ == "CREATE_USE_SHM") {
    CleanSharedMemory(numaId_);
    CreateUseSharedMemory(is_dummy_data);
  } else {
    DirectUseSharedMemory(is_dummy_data);
  }
}

template<typename T>
void DataProvider<T>::WrapLocalInput(const bool is_dummy_data) {
  LOG(INFO) << "use local memory";
  if (is_dummy_data) inputImgs_.resize(batchSize_ * inputSize_, 0);
  else inputImgs_.resize(total_samples_ * inputSize_, 0);
  inputData_ = inputImgs_.data();
  Preprocess(is_dummy_data, inputData_, imgNames_);
}

template<typename T>
void DataProvider<T>::WrapInput(const bool is_dummy_data) {
  inputSize_ = net_conf_->channels * net_conf_->height * net_conf_->width;
  if (sharedMemory_ == "USE_LOCAL") {
    WrapLocalInput(is_dummy_data);
  } else {
    WrapSHMInput(is_dummy_data);
  }
  if ((sharedMemory_ == "CREATE_USE_SHM" || sharedMemory_ == "USE_LOCAL")
        && contiguous_in_ram_)
    loadBuffer_.resize(total_samples_ * inputSize_, 0);
}

template<typename T>
void DataProvider<T>::SetMeanScale() {

}

template<typename T>
void DataProvider<T>::ResizeWithAspect(cv::Mat* sample, cv::Mat* sample_resized) {
  auto scale = net_conf_->aspect_scale;
  auto new_height = static_cast<int>(100. * net_conf_->height / scale);
  auto new_width = static_cast<int>(100. * net_conf_->width / scale);
  auto inter_pol = net_conf_->net_name == "resnet50" ? cv::INTER_AREA: cv::INTER_LINEAR;
  if ((*sample).rows > (*sample).cols) {
    auto res = static_cast<int>((*sample).rows * new_width / (*sample).cols);
    cv::resize((*sample), (*sample_resized), cv::Size(new_width, res), (0, 0), (0, 0), inter_pol);
  } else {
    auto res = static_cast<int>((*sample).cols * new_height / (*sample).rows);
    cv::resize((*sample), (*sample_resized), cv::Size(res, new_height), (0, 0), (0, 0), inter_pol);
  }
}

// resize image using rescale
template<typename T>
void DataProvider<T>::ResizeWithRescale(cv::Mat* sample, cv::Mat* sample_rescale) {
  auto aspect = static_cast<float>((*sample).cols) / (*sample).rows;
  if (aspect > 1) {
    auto res = static_cast<int>(net_conf_->rescale_size * aspect);
    cv::resize((*sample), (*sample_rescale), cv::Size(res, net_conf_->rescale_size));
  } else {
    auto res = static_cast<int>(net_conf_->rescale_size / aspect);
    cv::resize((*sample), (*sample_rescale), cv::Size(net_conf_->rescale_size, res));
  }
}

template<typename T>
void DataProvider<T>::CenterCrop(cv::Mat* sample_resized, cv::Mat* sample_roi) {
  int x = (*sample_resized).cols;
  int y = (*sample_resized).rows;
  int startx = static_cast<int>(std::floor(x * 0.5 - net_conf_->width * 0.5));
  int starty = static_cast<int>(std::floor(y * 0.5 - net_conf_->height * 0.5));
  cv::Rect roi(startx, starty, net_conf_->width, net_conf_->height);
  // roi image
  (*sample_roi) = (*sample_resized)(roi);
}

template<typename T>
void DataProvider<T>::PreprocessUsingCVMethod(T* inputImgs,
                            const vector<string>& imgNames) {
  // wrap and process image files.
  cv::Mat mean(net_conf_->width, net_conf_->height, CV_32FC3,
         cv::Scalar(net_conf_->mean_value[0], net_conf_->mean_value[1], net_conf_->mean_value[2]));
  cv::Mat scale(net_conf_->width, net_conf_->height, CV_32FC3,
                   cv::Scalar(net_conf_->scale, net_conf_->scale, net_conf_->scale));
  bool quantized_ = false;
  if (sizeof(T) == sizeof(char)) quantized_ = true;
  int converted_type;
  if (quantized_) {
    if (dataOrder_ == "NCHW") converted_type = CV_8SC1;
    else if (dataOrder_ == "NHWC") converted_type = CV_8SC3;
  } else {
    if (dataOrder_ == "NCHW") converted_type = CV_32FC1;
    else if (dataOrder_ == "NHWC") converted_type = CV_32FC3;
  }
#pragma omp parallel for
  for (size_t i = 0; i < imgNames.size(); ++i) {
    auto input_data = inputImgs + i * inputSize_;
    cv::Mat img = cv::imread(imgNames[i]);
    // convert the input image to the input image format of the network.
    cv::Mat sample;
    if (img.channels() == 3 && net_conf_->channels == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && net_conf_->channels == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && net_conf_->channels == 3)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && net_conf_->channels == 3)
      cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
      sample = img;

    cv::Mat sample_resized;
    cv::Mat sample_roi;
    if (net_conf_->preprocess_method == "ResizeWithAspect")
      ResizeWithAspect(&sample, &sample_resized);
    else
      ResizeWithRescale(&sample, &sample_resized);
    CenterCrop(&sample_resized, &sample_roi);

    cv::Mat sample_float;
    if (net_conf_->channels == 3) {
      sample_roi.convertTo(sample_float, CV_32FC3);
    } else sample_roi.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_subtract, sample_normalized;
    if (net_conf_->net_name=="resnet50") {
      cv::subtract(sample_float, mean, sample_subtract);
      cv::multiply(sample_subtract, scale, sample_normalized);
    } else if (net_conf_->net_name=="mobilenet") {
      cv::subtract(sample_float, mean, sample_subtract);
      cv::divide(sample_subtract, mean, sample_normalized);
    }
    std::vector<cv::Mat> input_channels;
    if (net_conf_->bgr2rgb) cv::cvtColor(sample_normalized, sample_normalized, cv::COLOR_RGB2BGR);
    if (quantized_) sample_normalized.convertTo(sample_normalized, converted_type, 1.0 / net_conf_->input_scale);
    if (dataOrder_ == "NCHW") {
      for (auto j = 0; j < net_conf_->channels; ++j) {
        cv::Mat channel(net_conf_->height, net_conf_->width, converted_type, input_data);
        input_channels.push_back(channel);
        input_data += net_conf_->width * net_conf_->height;
      }
      /* This operation will write the separate BGR planes directly to the
       * input layer of the network because it is wrapped by the cv::Mat
       * objects in input_channels. */
      cv::split(sample_normalized, input_channels);
    } else if (dataOrder_ == "NHWC") {
      cv::Mat channel(net_conf_->height, net_conf_->width, converted_type, input_data);
      sample_normalized.copyTo(channel);
    }
    // add zero_point 128 to u8 format
    auto u8_input_opt_option = getenv("U8_INPUT_OPT");
    if (quantized_ && (u8_input_opt_option != NULL) && (atoi(u8_input_opt_option) != 0))
      for(size_t i = 0; i < inputSize_; ++i) input_data[i] += 128;
  }
}

// preprocess the img
template<typename T>
void DataProvider<T>::Preprocess(const bool dummy, T* inputImgs,
                                 const vector<string>& imgNames) {
  if (!dummy) {
    LOG(INFO) << "this process will preprocess images";
    PreprocessUsingCVMethod(inputImgs, imgNames);
  } else {
    // only use one batch dummy
    for (int i = 0; i < batchSize_ * inputSize_; ++i)
      inputImgs[i] = static_cast<T>(std::rand()) / RAND_MAX;
  }
}

// preprocess the img according to batch size
template<typename T>
void DataProvider<T>::PreprocessSingleIteration(T* inputImgs,
                                       const vector<string>& imgNames) {
  // wrap and process image files.
  int img_size = net_conf_->channels * net_conf_->height * net_conf_->width;
  cv::Mat float_img;
  for (size_t i = 0; i < imgNames.size(); ++i) {
    cv::Mat raw_img = cv::imread(imgNames[i]);
    cv::Mat resized_img;
    cv::resize(raw_img,
               resized_img,
               cv::Size(net_conf_->width, net_conf_->height),
               0,
               0,
               cv::INTER_LINEAR);
    resized_img.convertTo(float_img, CV_32FC3);
    int tran_c = 0;
    int index = 0;
    for (int h = 0; h < net_conf_->height; ++h) {
      for (int w = 0; w < net_conf_->width; ++w) {
        for ( int c = 0; c < net_conf_->channels; ++c) {
          tran_c = net_conf_->bgr2rgb ? (2-c) : c;
          if (dataOrder_ == "NHWC") {
            index = img_size * i + h * net_conf_->width * net_conf_->channels + w * net_conf_->channels + c;
          } else if (dataOrder_ == "NCHW") {
            index = img_size * i + c * net_conf_->width * net_conf_->height + h * net_conf_->width + w;
          }
          inputImgs[index] =
            static_cast<T>((float_img.ptr<cv::Vec3f>(h)[w][tran_c] -
                            net_conf_->mean_value[c]) * net_conf_->scale);
        }
      }
    }
  }
}

template<typename T>
void DataProvider<T>::ParseImageLabel(const string& file_list, const string& image_path,
                                      const string& label_path, const size_t sample_size,
                                      const bool is_dummy_data) {
  if (is_dummy_data) {
    LOG(INFO) << "dummy data will not parse the image";
    labels_.resize(sample_size);
    return;
  }
  // std::vector<std::string> file_names;
  // cv::glob(image_path + "/*.JPEG", file_names);
  // for (auto name : file_names)
  //   imgNames_.push_back(name);
  // labels_.resize(imgNames_.size());
  
  string val;
  // wrap and process image files.
  if (!file_list.empty() || (!label_path.empty() && !image_path.empty())) {
    string file_name;
    if (!file_list.empty()) file_name = file_list;
    else file_name = label_path;
    std::ifstream image_file(file_name);
    while (std::getline(image_file, val)) {
      auto pos = val.find(" ");
      auto label = std::atoi(val.substr(pos + 1).c_str());
      labels_.push_back(label);
      if (!file_list.empty()) {
        imgNames_.push_back(val.substr(0, pos));
      } else {
        string image_val = val.substr(0, pos);
        imgNames_.push_back(image_path + "/" + image_val);
      }
      if (imgNames_.size() == sample_size) break;
    }
    image_file.close();
  } else if (image_path == "") {
    LOG(FATAL) << "image path should be given!";
  } else {
    if (label_path == "") {
      LOG(WARNING) << "label path not given, accuracy not caculated!";
      DIR* image_dir = opendir(image_path.c_str());
      if (image_dir == nullptr) LOG(FATAL) << "can't read image path!";
      struct dirent* file_name;
      int filter_dir = 0;
      while((file_name = readdir(image_dir)) != nullptr) {
        // linux dir will read dir "." and ".." as file_name, so filter that
        if ((filter_dir++) < 2) continue;
        imgNames_.push_back(image_path + file_name->d_name);
        if (imgNames_.size() == sample_size) break;
      }
    }
  }
  if (imgNames_.size() < sample_size) {
    LOG(ERROR) << "image size is " << imgNames_.size() << " sample size is " << sample_size;
    size_t append_count = sample_size - imgNames_.size();
    size_t real_image_size = imgNames_.size();
    for (size_t i = 0; i < append_count; ++i) {
      imgNames_.push_back(imgNames_[i % real_image_size]);
      labels_.push_back(labels_[i % real_image_size]);
    }
    return;
  }
}

} // using namespace tensorflow
#endif // DATAPROVIDER_H__
