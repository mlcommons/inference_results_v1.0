#include "ncoresw/compiler/mlir/ncore-runner-utils/include/profiler.h"
#include "experimental/mlperf/inference/runner/runner.h"

#include <cstdlib>
#include <random>

#include "experimental/mlperf/inference/loadgen/loadgen.h"
#include "experimental/mlperf/inference/loadgen/query_sample.h"
#include "experimental/mlperf/inference/runner/loader.h"
#include "experimental/mlperf/inference/runner/options.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace mlperf {

Runner::Runner(const mlperf::Option& option,
               bool standalone, bool latency_mode)
    : option_(option),
      batching_option_(option.batching_option),
      num_threads_(option.num_worker_threads),
      export_model_path_(option.export_model_path),
      standalone_(standalone) {

  // Parse the tpu_target seprated by ","
  std::string delimiter = ",";
  size_t pos = 0;
  std::string token;
  std::string aic_target_string = option_.aic_target;
  while ((pos = aic_target_string.find(delimiter)) != std::string::npos) {
    token = aic_target_string.substr(0, pos);
    aic_target_.push_back(token);
    std::cout << token << std::endl;
    aic_target_string.erase(0, pos + delimiter.length());
  }
  aic_target_.push_back(aic_target_string);

  num_aics_ = std::max(static_cast<int>(aic_target_.size()), 1);
  loader_.resize(num_aics_);

  // Initialized loaders.
  for (int i = 0; i < num_aics_; i++) {
    loader_[i] = absl::make_unique<mlperf::Loader>(
        /*aic_target_=*/aic_target_[i],
        /*saved_model_path=*/export_model_path_, num_threads_, batching_option_,
        latency_mode, option_.init_aic, option_.init_old_delegate);
  }

  // Prepare response vectors, input tensors, and output tensors.
  responses_.resize(num_threads_);
  inputs_.resize(num_threads_);
  inputs_max_batch_size_.resize(num_threads_);
  outputs_.resize(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    responses_[i] = new QuerySampleResponse[batching_option_.max_batch_size];
    inputs_[i] = tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
    inputs_max_batch_size_[i] = tensorflow::Tensor(
        tensorflow::DT_INT32, tensorflow::TensorShape({batching_option_.max_batch_size}));
    // Zero-initialize input tensors.
    inputs_[i].vec<int>()(0) = 0;
    for (int j = 0; j < batching_option_.max_batch_size; ++j) {
      inputs_max_batch_size_[i].vec<int>()(j) = 0;
    }
    workers_.emplace_back(std::thread(&Runner::HandleTask, this, i));
  }
}

void Runner::Enqueue(const QuerySamples& sample) { queue_.put(sample); }

void Runner::HandleTask(int thread_id) {
  loader_[thread_id % num_aics_]->MapStdThreadIdToThreadNum(std::this_thread::get_id(), thread_id);

  // Post-processing function pointer, dependent on model
  uintptr_t (*post_process_fn)(std::vector<tensorflow::Tensor>& outputs, QuerySamples& queries, uintptr_t post_out_ptr, size_t& calc_out_size) = nullptr;
  uintptr_t post_scratch_buf = 0;
  if(option_.model_name == "resnet50") {
    post_process_fn = ResNet50_PostProcess;
  } else if(option_.model_name == "ssd-mobilenet") {
    post_process_fn = SSDMobileNetV1_PostProcess;
    post_scratch_buf = (uintptr_t)malloc(70*sizeof(float));
  }

  CENTAUR_PROFILER_START;
  while (true) {
    auto queries = queue_.get();
    if (queries.empty()) {
      break;
    }
    tensorflow::profiler::TraceMe trace_me([&] { return "Predict"; },
                                           /*level=*/2);
    outputs_[thread_id].clear();

    while (!queries.empty() && queries.back().id == 0) {
      queries.pop_back();
    }
    NCORESW_CHECK_LE(queries.size(), batching_option_.max_batch_size);
    NCORESW_CHECK_LE(thread_id, responses_.size());

    PROFILER_START_TIME(tp_idlookup);
    for (int64_t i = 0; i < queries.size(); ++i) {
      inputs_max_batch_size_[thread_id].vec<int>()(i) =
          sample_id_to_qsl_index_map_[queries[i].index];
    }
    PROFILER_END_RECORD_THREAD(tp_idlookup, "sample_id_to_qsl_index", thread_id);

    PROFILER_START_TIME(tp_predict);
    loader_[thread_id % num_aics_]->Predict(inputs_max_batch_size_[thread_id],
                                            outputs_[thread_id], thread_id);

    uintptr_t raw_data;
    size_t out_num_bytes = 0;
    if(post_process_fn) {
      PROFILER_START_TIME(tp_post);
      raw_data = post_process_fn(outputs_[thread_id], queries, post_scratch_buf, out_num_bytes);
      PROFILER_END_RECORD_THREAD(tp_post, "post_process_fn", thread_id);
    } else {
      raw_data = reinterpret_cast<uintptr_t>(outputs_[thread_id][0].tensor_data().data());
      for(auto out : outputs_[thread_id]) {
        out_num_bytes += out.TotalBytes();
      }
    }
    PROFILER_END_RECORD_THREAD(tp_predict, "Predict", thread_id);

    PROFILER_START_TIME(tp_get_response);
    for (int64_t i = 0; i < queries.size(); ++i) {
      responses_[thread_id][i].id = queries[i].id;
      responses_[thread_id][i].data = raw_data + out_num_bytes * i;
      responses_[thread_id][i].size = out_num_bytes;
    }
    PROFILER_END_RECORD_THREAD(tp_get_response, "get_predict_response", thread_id);

    if (!standalone_) {
      PROFILER_START_TIME(tp_complete);
      QuerySamplesComplete(responses_[thread_id], queries.size());
      PROFILER_END_RECORD_THREAD(tp_complete, "QuerySampleComplete", thread_id);
    }
  }

  if(post_scratch_buf) {
    free((void*)post_scratch_buf);
  }

  CENTAUR_PROFILER_STOP_THREAD(thread_id);
}

void Runner::UpdateQslIndexMap(
    std::unordered_map<QuerySampleIndex, QuerySampleIndex> sample_id_to_qsl_index_map) {
  CENTAUR_PROFILER_START;
  sample_id_to_qsl_index_map_ = std::move(sample_id_to_qsl_index_map);
  CENTAUR_PROFILER_STOP;
}

void Runner::UpdateQSL(
    const tensorflow::Tensor& qsl,
    std::unordered_map<QuerySampleIndex, QuerySampleIndex> sample_id_to_qsl_index_map) {
  CENTAUR_PROFILER_START;
  for (int i = 0; i < num_aics_; i++) {
    loader_[i]->UpdateQSL(qsl);
  }
  UpdateQslIndexMap(sample_id_to_qsl_index_map);
  CENTAUR_PROFILER_STOP;
}

void Runner::WarmUp() {
  CENTAUR_PROFILER_START;
  NCORESW_LOG(INFO) << "Warming up the system.";
  for (auto bs : batching_option_.batch_size) {
    NCORESW_LOG(INFO) << "  batch size: " << bs << ", warmup started.";
    tensorflow::Tensor inputs =
        tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({bs}));
    for (int idx = 0; idx < bs; ++idx) {
      inputs.vec<int>()(idx) = 0;
    }
    for (int j = 0; j < num_aics_; j++) {
      for (int64_t i = 0; i < 1; ++i) {
        loader_[j]->Predict(inputs, outputs_[0]);
      }
    }
    NCORESW_LOG(INFO) << "  batch size: " << bs << ", warmup done.";
  }
  CENTAUR_PROFILER_STOP;
}

void Runner::Finish() {
  CENTAUR_PROFILER_START;
  for (int i = 0; i < num_threads_; i++) {
    QuerySamples empty_queries;
    queue_.put(empty_queries);
  }

  for (int i = 0; i < num_threads_; i++) {
    if (workers_[i].joinable()) {
      workers_[i].join();
    }
  }
  profiling::Profiler::GetProfiler()->Write();
  CENTAUR_PROFILER_STOP;
}

}  // namespace mlperf
