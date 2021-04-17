#pragma once

#include <string>
#include <vector>
#include <queue>

#include "ncoresw/tf_delegate/ncore_delegate.h"
#include "ncoresw/tf_delegate/mlir/aic_delegate.h"
#include "absl/status/status.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/model.h"

#include "experimental/mlperf/inference/runner/options.h"

// #include "tensorflow/cc/saved_model/loader.h"
// #include "tensorflow/core/framework/tensor.h"
// #include "tensorflow/core/public/session.h"
// #include "tensorflow_serving/batching/batching_session.h"

namespace mlperf {

using int64 = long long int;

class Loader {
 public:
  Loader(const std::string aic_target, const std::string model_path,
         const int num_threads, const mlperf::BatchingOption batching_options,
         const bool latency_mode, bool init_aic, bool init_old_delegate);

  void UpdateQSL(const tensorflow::Tensor& qsl);

  void Predict(const tensorflow::Tensor& inputs, std::vector<tensorflow::Tensor>& outputs, const int thread_id=0);

  absl::Status MapStdThreadIdToThreadNum(std::thread::id thread_id, int thread_num);

 private:
  const int num_threads_;

  absl::Status LoadSavedModel();
  absl::Status AllocateTensors();
  absl::Status CreateBatchingSession();
  absl::Status InitializeAic();
  absl::Status ShutdownAic();
  absl::Status InitializeOldDelegate(const bool latency_mode);
  absl::Status ShutdownOldDelegate();

  std::string aic_target_;
  std::string model_path_;
  mlperf::BatchingOption batching_options_;

  // TODO(bryce): tensorflow batching support
  // tensorflow::SavedModelBundle saved_model_bundle_;
  // tensorflow::SignatureDef signature_def_;
  // std::unique_ptr<tensorflow::Session> batching_session_;
  // std::unique_ptr<tensorflow::Session> main_session_;
  std::vector<tensorflow::Tensor> qsl_queue_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_[NCORE_NUM_X86_THREADS];
  std::unique_ptr<TfLiteDelegate> aic_delegate_[NCORE_NUM_X86_THREADS];
  NcoreDelegate* old_ncore_delegate_[NCORE_NUM_X86_THREADS];

  std::string qsl_name_;
  std::string update_qsl_name_;
  std::string input_tensor_names_;
  std::vector<std::string> output_tensor_names_;
};
}  // namespace mlperf
