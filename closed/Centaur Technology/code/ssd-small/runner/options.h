#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "ncoresw/common/logging.h"
#include "experimental/mlperf/inference/loadgen/test_settings.h"

namespace mlperf {

struct BatchingOption {
  std::string thread_pool_name;
  int num_batch_threads;
  int batch_timeout_micros;
  int max_enqueued_batches;
  std::vector<int> batch_size;
  // max_batch_size is derived from batch_size.
  int max_batch_size;
  BatchingOption(
      std::string thread_pool_name = std::string("shared_batch_scheduler"),
      const int num_batch_threads = 1, const int batch_timeout_micros = 1000,
      const int max_enqueued_batches = 1 << 16,
      std::vector<int> batch_size_list = std::vector<int>({4}))
      : thread_pool_name(std::move(thread_pool_name)),
        num_batch_threads(num_batch_threads),
        batch_timeout_micros(batch_timeout_micros),
        max_enqueued_batches(max_enqueued_batches),
        batch_size(std::move(batch_size_list)) {
    max_batch_size = *std::max_element(batch_size.begin(),  batch_size.end());
  }
};

struct Option {
  int num_worker_threads;
  mlperf::TestScenario scenario;
  BatchingOption batching_option;
  int total_sample_count;
  int performance_sample_count;
  int qps;
  int time;
  int max_latency;
  std::string outdir;
  std::string model_name;
  std::string export_model_path;
  std::string aic_target;
  std::string mlperf_conf;
  std::string user_conf;
  bool init_aic;
  bool init_old_delegate;
  Option(const int num_worker_threads = 4,
         const BatchingOption batching_option = BatchingOption(),
         const int total_sample_count = 50000, const int performance_sample_count = 1024,
         std::string outdir = "/tmp", std::string export_model_path = "", std::string model_name = "resnet50",
         std::string aic_target = "", std::string test_scenario = "SingleStream",
         std::string mlperf_conf = "mlperf.conf", std::string user_conf = "user.conf",
         bool init_aic = false, bool init_old_delegate = true)
      : num_worker_threads(num_worker_threads),
        batching_option(batching_option),
        total_sample_count(total_sample_count),
        performance_sample_count(performance_sample_count),
        outdir(std::move(outdir)),
        model_name(std::move(model_name)),
        export_model_path(std::move(export_model_path)),
        aic_target(std::move(aic_target)),
        mlperf_conf(std::move(mlperf_conf)),
        user_conf(std::move(user_conf)),
        init_aic(init_aic),
        init_old_delegate(init_old_delegate) {
    if (test_scenario == "SingleStream" || test_scenario == "Single Stream") {
      scenario = mlperf::TestScenario::SingleStream;
    } else if (test_scenario == "Offline") {
      scenario = mlperf::TestScenario::Offline;
    } else {
      NCORESW_LOG(FATAL) << "Unsupported test_scenario: " << test_scenario;
    }
  }
  
};
}  // namespace mlperf
