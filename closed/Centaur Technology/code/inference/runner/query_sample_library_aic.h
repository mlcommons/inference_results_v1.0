#pragma once

#include <string>

#include "experimental/mlperf/inference/loadgen/query_sample_library.h"
#include "experimental/mlperf/inference/runner/options.h"
#include "experimental/mlperf/inference/runner/dataset.h"
#include "experimental/mlperf/inference/runner/runner.h"

namespace mlperf {

class QuerySampleLibraryAic : public QuerySampleLibrary {
 public:
  QuerySampleLibraryAic(std::string name, mlperf::Runner* runner, std::string dataset_path,
                        const std::vector<int64>& shape, tensorflow::DataType datatype,
                        const mlperf::Option& option)
      : name_(std::move(name)),
        runner_(runner),
        dataset_path_(dataset_path),
        shape_(shape),
        datatype_(datatype),
        total_sample_count_(option.total_sample_count),
        performance_sample_count_(std::min(option.total_sample_count, option.performance_sample_count)),
        option_(option) {}
  ~QuerySampleLibraryAic() override = default;

  const std::string& Name() const override { return name_; }
  size_t TotalSampleCount() { return total_sample_count_; }
  size_t PerformanceSampleCount() { return performance_sample_count_; }

  void LoadSamplesToRam(const std::vector<QuerySampleIndex>& samples) override {
    auto qsl = mlperf::CreateQSLOrDie(dataset_path_, samples, shape_, datatype_, option_);
    auto sample_id_to_qsl_index_map = mlperf::CreateSampleIdToQSLIndexMap(samples);
    runner_->UpdateQSL(qsl, sample_id_to_qsl_index_map);
  }
  void UnloadSamplesFromRam(const std::vector<QuerySampleIndex>& samples) override {}

 private:
  std::string name_;
  mlperf::Runner* runner_;  // Not owned
  std::string dataset_path_;
  std::vector<int64> shape_;
  tensorflow::DataType datatype_;
  size_t total_sample_count_;
  size_t performance_sample_count_;
  mlperf::Option option_;
};

}  // namespace mlperf
