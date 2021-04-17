
#include <iterator>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <mutex>

#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

#include "ncoresw/common/logging.h"
#include "ncoresw/common/status_macros.h"
#include "ncoresw/compiler/mlir/ncore-runner-utils/include/profiler.h"

#include "experimental/mlperf/inference/runner/loader.h"

#include "ncoresw/tf_delegate/ncore_delegate.h"

namespace mlperf {

namespace {
// A GraphDef containing the ops required to initialize and shutdown a AIC.
// This proto was generated from the script oneoffs/generate_tpu_graph_def.py.
constexpr auto kAicOpsGraphDef = R"(
node {
  name: "ConfigureDistributedAIC"
  op: "ConfigureDistributedAIC"
  device: "/device:AIC_SYSTEM:0"
  attr {
    key: "embedding_config"
    value {
      s: ""
    }
  }
  attr {
    key: "is_global_init"
    value {
      b: false
    }
  }
  attr {
    key: "aic_embedding_config"
    value {
      s: ""
    }
  }
}
node {
  name: "ShutdownDistributedAIC"
  op: "ShutdownDistributedAIC"
  device: "/device:AIC_SYSTEM:0"
}
library {
}
)";
}  // namespace

// namespace {
  
// static GetPredictLock()

// }  // namespace

Loader::Loader(const std::string aic_target, const std::string model_path, const int num_threads,
               const mlperf::BatchingOption batching_options = mlperf::BatchingOption(),
               const bool latency_mode = true,
               bool init_aic = true, bool init_old_delegate = false)
    : aic_target_(aic_target), model_path_(model_path), batching_options_(batching_options), num_threads_(num_threads) {
  // The initialization order must be the following: (1) load the graph. (2)
  // initialize the AIC system, and (3) create a batching session.
  NCORESW_CHECK_OK(LoadSavedModel());
  NCORESW_CHECK_OK(AllocateTensors());

  if (init_aic) {
    NCORESW_CHECK_OK(InitializeAic());
  } else if (init_old_delegate) {
    NCORESW_CHECK_OK(InitializeOldDelegate(latency_mode));
  }

  // TODO(bryce): Create a batching session if we need it?
  // NCORESW_CHECK_OK(CreateBatchingSession());
}

absl::Status Loader::LoadSavedModel() {
  model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
  NCORESW_CHECK_NOTNULL(model_);
  NCORESW_LOG(INFO) << "Loaded TfLite Model...";
  
  for(int i = 0; i < num_threads_; i++) {
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model_, resolver)(&interpreter_[i]);
    NCORESW_CHECK_NOTNULL(interpreter_[i]);
    NCORESW_LOG(INFO) << "Built the interpreter...";
  }

  return absl::OkStatus();
}

absl::Status Loader::CreateBatchingSession() {
  CENTAUR_PROFILER_START;
  // Creates a batching session.
  // tensorflow::serving::BatchingSessionOptions batching_session_options;
  // for (int bs : batching_options_.batch_size) {
  //   batching_session_options.allowed_batch_sizes.push_back(bs);
  // }

  // tensorflow::serving::TensorSignature signature;
  // signature.input_tensors.insert(input_tensor_names_);
  // for (auto& name : output_tensor_names_) {
  //   signature.output_tensors.insert(name);
  // }
  // CHECK_LE(1, signature.input_tensors.size());
  // tensorflow::serving::BasicBatchScheduler<
  //     tensorflow::serving::BatchingSessionTask>::Options schedule_options;
  // schedule_options.thread_pool_name = batching_options_.thread_pool_name;
  // schedule_options.num_batch_threads = batching_options_.num_batch_threads;
  // schedule_options.batch_timeout_micros =
  //     batching_options_.batch_timeout_micros;
  // schedule_options.max_enqueued_batches =
  //     batching_options_.max_enqueued_batches;
  // schedule_options.max_batch_size = batching_options_.max_batch_size;

  // TF_CHECK_OK(CreateBasicBatchingSession(
  //     schedule_options, batching_session_options, signature,
  //     std::move(saved_model_bundle_.session), &batching_session_));
  CENTAUR_PROFILER_STOP;
  return absl::OkStatus();
}

absl::Status Loader::InitializeAic() {
  CENTAUR_PROFILER_START;
  NCORESW_LOG(INFO) << "Initializing AIC" << std::endl;

  auto options = TfLiteAICDelegateOptionsDefault();

  for(int i = 0; i < num_threads_; i++) {
    aic_delegate_[i].reset(TfLiteAICDelegateCreate(&options));
    NCORESW_CHECK_NOTNULL(aic_delegate_[i]);

    absl::Time compile_start = absl::Now();
    NCORESW_CHECK(interpreter_[i]->ModifyGraphWithDelegate(aic_delegate_[i].get()) == kTfLiteOk);
    absl::Time compile_stop = absl::Now();
    auto jit_compile_time = absl::ToDoubleSeconds(compile_stop - compile_start);
    NCORESW_LOG(INFO) << "JIT Compile time:" << jit_compile_time;
  }

  CENTAUR_PROFILER_STOP;
  return absl::OkStatus();
}

absl::Status Loader::InitializeOldDelegate(bool latency_mode/*=true*/) {
  CENTAUR_PROFILER_START;
  NCORESW_LOG(INFO) << "Initializing old delegate" << std::endl;

  for(int i = 0; i < num_threads_; i++) {
    old_ncore_delegate_[i] = new NcoreDelegate();
    old_ncore_delegate_[i]->SetInterpreter(interpreter_[i].get());
    old_ncore_delegate_[i]->delegate_options = NcoreDelegateOptions();
    if(latency_mode) {
      old_ncore_delegate_[i]->delegate_options.ncore_compile_options.performance_mode_type = NcoreCompileOptions::PerformanceModeType::kLatency;
    } else {
      old_ncore_delegate_[i]->delegate_options.ncore_compile_options.performance_mode_type = NcoreCompileOptions::PerformanceModeType::kThroughput;
    }

    absl::Time compile_start = absl::Now();
    NCORESW_CHECK(interpreter_[i]->ModifyGraphWithDelegate(old_ncore_delegate_[i]->tflite_delegate()) == kTfLiteOk);
    absl::Time compile_stop = absl::Now();
    auto jit_compile_time = absl::ToDoubleSeconds(compile_stop - compile_start);
    NCORESW_LOG(INFO) << "JIT Compile time:" << jit_compile_time;
  }

  CENTAUR_PROFILER_STOP;
  return absl::OkStatus();
}

absl::Status Loader::MapStdThreadIdToThreadNum(std::thread::id thread_id, int thread_num) {
  old_ncore_delegate_[thread_num]->ndev.set_thread_id_num_map(thread_id, thread_num);
  return absl::OkStatus();
}

absl::Status Loader::AllocateTensors() {
  CENTAUR_PROFILER_START;

  for(int i = 0; i < num_threads_; i++) {
    NCORESW_CHECK_NOTNULL(interpreter_[i]);
    NCORESW_CHECK(interpreter_[i]->AllocateTensors() == kTfLiteOk);
  }

  CENTAUR_PROFILER_STOP;
  return absl::OkStatus();
}

absl::Status Loader::ShutdownAic() {
  CENTAUR_PROFILER_START;
  NCORESW_LOG(INFO) << "Shutting down AIC" << std::endl;
  CENTAUR_PROFILER_STOP;
  return absl::OkStatus();
}

absl::Status Loader::ShutdownOldDelegate() {
  CENTAUR_PROFILER_START;
  NCORESW_LOG(INFO) << "Shutting down old delegate" << std::endl;
  CENTAUR_PROFILER_STOP;
  return absl::OkStatus();
}

void Loader::UpdateQSL(const tensorflow::Tensor& qsl) {
  qsl_queue_.clear();
  qsl_queue_.push_back(qsl);
}

// FIXME(bryce, mthomson): Using tensorflow::Tensor type out of convenience.
// Decide if it makes sense to switch over to using std::vector<TfLiteTensor>
// outputs or some simple, custom ncoresw::Tensor type?
void Loader::Predict(const tensorflow::Tensor& input, std::vector<tensorflow::Tensor>& outputs, const int thread_id/*=0*/) {
  CENTAUR_PROFILER_START;
  
  NCORESW_CHECK_EQ(qsl_queue_.size(), 1) << "Should only have one input tensorflow::Tensor at the moment";

  tensorflow::Tensor in = qsl_queue_.front().SubSlice(input.vec<int>()(0));

  const std::vector<int> interpreter_inputs = interpreter_[thread_id]->inputs();
  const std::vector<int> interpreter_outputs = interpreter_[thread_id]->outputs();
  NCORESW_CHECK_EQ(interpreter_inputs.size(), 1) << "MLPerf only supports one input tensor at the moment";

  const int tfl_input = interpreter_inputs.front();
  auto* tfl_input_tensor = interpreter_[thread_id]->tensor(tfl_input);
  
  // Convert tensorflow::Tensor to a TfLiteTensor by updating data ptr
  tfl_input_tensor->data.data = in.data();
  
  auto convertTfLiteIntArrayToVec = [](const TfLiteIntArray* arr) {
    std::vector<int64_t> result;
    for (int i = 0; i < arr->size; i++) {
      result.push_back(arr->data[i]);
    }
    return result;
  };
  
  auto convertStdInt64ToTfInt64 = [](const std::vector<int64_t>& dims) {
    std::vector<int64> tf_arr;
    for (auto& dim : dims) {
      tf_arr.push_back(dim);
    }
    return tf_arr;
  };
  
  auto convertTfLiteTypeToTfDataType = [](const TfLiteType type) {
    switch (type) {
      case kTfLiteNoType:
        return tensorflow::DT_INVALID;
      case kTfLiteFloat32:
        return tensorflow::DT_FLOAT;
      case kTfLiteInt16:
        return tensorflow::DT_INT16;
      case kTfLiteInt32:
        return tensorflow::DT_INT32;
      case kTfLiteUInt8:
        return tensorflow::DT_UINT8;
      case kTfLiteInt8:
        return tensorflow::DT_INT8;
      case kTfLiteInt64:
        return tensorflow::DT_INT64;
      case kTfLiteBool:
        return tensorflow::DT_BOOL;
      case kTfLiteComplex64:
        return tensorflow::DT_COMPLEX64;
      case kTfLiteString:
        return tensorflow::DT_STRING;
      case kTfLiteFloat16:
        return tensorflow::DT_BFLOAT16;
      default:
        NCORESW_LOG(FATAL) << "Unkown TfLiteType";
        return tensorflow::DT_INVALID;
    }
  };
  
  // Set the tflite interpreter to write outputs to underlying tensorflow::Tensor data ptr
  PROFILER_START_TIME(tp_out_ptrs);
  for (size_t i = 0; i < interpreter_outputs.size(); i++) {
    const auto& tfl_output_tensor_id = interpreter_outputs[i];
    auto* tfl_output_tensor = interpreter_[thread_id]->tensor(tfl_output_tensor_id);
    
    auto output_shape = convertTfLiteIntArrayToVec(tfl_output_tensor->dims);

    auto output =
        tensorflow::Tensor(tensorflow::DataType(convertTfLiteTypeToTfDataType(tfl_output_tensor->type)),
                           tensorflow::TensorShape(convertStdInt64ToTfInt64(output_shape)));
    tfl_output_tensor->data.data = output.data();

    outputs.push_back(std::move(output));
  }
  PROFILER_END_RECORD_THREAD(tp_out_ptrs, "Set output tensor ptrs", thread_id);
  
  // Invoke
  interpreter_[thread_id]->Invoke();

  CENTAUR_PROFILER_STOP_THREAD(thread_id);
}

}  // namespace mlperf
