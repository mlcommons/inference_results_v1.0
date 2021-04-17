#include <sys/stat.h>
#include <sys/types.h>

#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"

#include "ncoresw/common/logging.h"
#include "ncoresw/common/status_macros.h"

#include "experimental/mlperf/inference/loadgen/loadgen.h"
#include "experimental/mlperf/inference/loadgen/test_settings.h"
#include "experimental/mlperf/inference/runner/options.h"
#include "experimental/mlperf/inference/runner/query_sample_library_aic.h"
#include "experimental/mlperf/inference/runner/runner.h"
#include "experimental/mlperf/inference/runner/system_under_test_aic.h"

using int64 = long long int;

ABSL_FLAG(std::string, aic_name, "", "The name of the AIC instance.");
ABSL_FLAG(int, num_worker_threads, std::thread::hardware_concurrency() * 2,
          "Number of threads to use.");
ABSL_FLAG(std::string, model_name, "resnet50", "Name of the mlperf model, ie. resnet50");
ABSL_FLAG(std::string, export_model_path, "", "The directory that includes the frozen model.");
ABSL_FLAG(std::string, preprocessed_dataset_path, "",
          "The directory that includes the preprocessed dataset.");
ABSL_FLAG(std::string, mlperf_conf, "mlperf.conf", "MLPerf rules config");
ABSL_FLAG(std::string, user_conf, "user.conf", "user config for user LoadGen settings such as target QPS");
// Loadgen test settings.
ABSL_FLAG(bool, accuracy_mode, false, "Running in accuracy model.");
ABSL_FLAG(bool, init_aic, false, "Initialize AIC.");
ABSL_FLAG(bool, init_old_delegate, true, "Initialize old delegate.");
ABSL_FLAG(std::string, scenario, std::string("Single Stream"), "Test scenario.");
ABSL_FLAG(int, total_sample_count, 0, "Number of samples to test.");
ABSL_FLAG(int, performance_sample_count, 1024, "Number of samples to test.");
ABSL_FLAG(std::string, outdir, "/tmp/mlperf", "Output directory for Loadgen.");
// Batching settings.
ABSL_FLAG(std::string, batch_size, std::string("1"),
          "comma-separated list that specifies allowed batch sizes.");
ABSL_FLAG(std::string, batching_thread_pool_name, std::string("shared_batch_scheduler"),
          "The name of the thread pool for batching.");
ABSL_FLAG(int, batching_num_batch_threads, 1, "The number of thread for batching.");
ABSL_FLAG(int, batching_batch_timeout_micros, 2000, "The timeout in microseconds for batching.");
ABSL_FLAG(int, batching_max_enqueued_batches, 1 << 21, "The maximum batches in the queue.");
// Performance settings.
//ABSL_FLAG(int, space_to_depth_block_size, 0, "conv0 space-to-depth block size");

namespace {
 
absl::Status ParseFlagBatchSize(absl::string_view text, std::vector<int>& batch_size) {
  for (absl::string_view part : absl::StrSplit(text, ',', absl::SkipEmpty())) {
    // Let flag module parse the element type for us.
    int element;
    std::string error;
    if (!absl::ParseFlag(std::string(part), &element, &error)) {
      return absl::InvalidArgumentError("invalid");
    }
    batch_size.emplace_back(element);
  }
  return absl::OkStatus();
}

std::unique_ptr<mlperf::QuerySampleLibraryAic> ConstructQsl(const mlperf::Option& option,
                                                            mlperf::Runner* runner) {
  int image_h = 224;
  int image_w = 224;
  int image_c = 3;
  tensorflow::DataType dtype = tensorflow::DT_FLOAT;
  if(option.model_name == "mobilenet") {
    image_h = 224;
    image_w = 224;
    image_c = 3;
    dtype = tensorflow::DT_UINT8;
  } else if(option.model_name == "resnet50") {
    image_h = 224;
    image_w = 224;
    image_c = 3;
    dtype = tensorflow::DT_FLOAT;
  } else if(option.model_name == "ssd-mobilenet") {
    image_h = 300;
    image_w = 300;
    image_c = 3;
    dtype = tensorflow::DT_UINT8;
  }
  std::vector<int64> unbatched_shape_nhwc({image_h, image_w, image_c});
  std::unique_ptr<mlperf::QuerySampleLibraryAic> qsl =
      absl::make_unique<mlperf::QuerySampleLibraryAic>(
          "qsl", runner, absl::GetFlag(FLAGS_preprocessed_dataset_path), unbatched_shape_nhwc,
          dtype, option);
  return qsl;
}

void RunInference(const mlperf::Option& option) {
  mlperf::TestSettings requested_settings;
  std::string scenario = absl::GetFlag(FLAGS_scenario);
  bool latency_mode = true;
  if(scenario == "Offline") {
    latency_mode = false;
  }

  // Set up runner.
  std::unique_ptr<mlperf::Runner> runner = absl::make_unique<mlperf::Runner>(
      option, /*standalone*/ false, latency_mode);

  NCORESW_LOG(INFO) << "Runner was created.";

  // Offline mode: use the max batch size; server mode: use 1.
  int issue_batch_size = 1;
  if (option.scenario == mlperf::TestScenario::Offline) {
    issue_batch_size = *std::max_element(option.batching_option.batch_size.begin(),
                                         option.batching_option.batch_size.end());
  }
  NCORESW_LOG(INFO) << "issue_batch_size: " << issue_batch_size << std::endl;
  // Set up sut.
  std::unique_ptr<mlperf::SystemUnderTestAic> sut =
      absl::make_unique<mlperf::SystemUnderTestAic>("sut", issue_batch_size, runner.get());
  // Set up qsl.
  std::unique_ptr<mlperf::QuerySampleLibraryAic> qsl = ConstructQsl(option, runner.get());

  // Set up the loadgen.
  if(!option.mlperf_conf.empty()) {
      requested_settings.FromConfig(option.mlperf_conf, option.model_name, scenario);
  }
  if(!option.user_conf.empty()) {
      requested_settings.FromConfig(option.user_conf, option.model_name, scenario);
  }

  requested_settings.scenario = option.scenario;
  if (absl::GetFlag(FLAGS_accuracy_mode)) {
    requested_settings.mode = mlperf::TestMode::AccuracyOnly;
  } else {
    requested_settings.mode = mlperf::TestMode::PerformanceOnly;
  }

  // Override target latency when it needs to be less than 1ms
  if(option.model_name == "mobilenet") {
      requested_settings.single_stream_expected_latency_ns =  200000;
  } else if(option.model_name == "resnet50") {
      requested_settings.single_stream_expected_latency_ns =  900000;
  } else if(option.model_name == "ssd-mobilenet") {
      requested_settings.single_stream_expected_latency_ns = 900000;
  }

  if(absl::GetFlag(FLAGS_total_sample_count) != 0) {
    requested_settings.min_query_count = option.total_sample_count;
    requested_settings.max_query_count = option.total_sample_count;
  }

  mlperf::LogSettings log_settings;
  log_settings.log_output.outdir = option.outdir;
  log_settings.log_output.copy_detail_to_stdout = true;
  log_settings.log_output.copy_summary_to_stdout = true;
  log_settings.enable_trace = false;

  // Warm up the system.
  qsl->LoadSamplesToRam({0});
  runner->WarmUp();

  // After warmup, give the system a moment to quiesce before putting it under load.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Start test.
  mlperf::StartTest(sut.get(), qsl.get(), requested_settings, log_settings);
  runner->Finish();

  exit(0);
}
}  // namespace

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Run the MLPerf Benchmark for the ImageNet model.");
  absl::ParseCommandLine(argc, argv);
  std::vector<int> batch_size;
  NCORESW_CHECK_OK(ParseFlagBatchSize(absl::GetFlag(FLAGS_batch_size), batch_size));
  mlperf::BatchingOption batching_option(absl::GetFlag(FLAGS_batching_thread_pool_name),
                                         absl::GetFlag(FLAGS_batching_num_batch_threads),
                                         absl::GetFlag(FLAGS_batching_batch_timeout_micros),
                                         absl::GetFlag(FLAGS_batching_max_enqueued_batches),
                                         batch_size);
  mlperf::Option option(
      absl::GetFlag(FLAGS_num_worker_threads), batching_option,
      absl::GetFlag(FLAGS_total_sample_count), absl::GetFlag(FLAGS_performance_sample_count),
      absl::GetFlag(FLAGS_outdir), absl::GetFlag(FLAGS_export_model_path), absl::GetFlag(FLAGS_model_name),
      absl::GetFlag(FLAGS_aic_name), absl::GetFlag(FLAGS_scenario),
      absl::GetFlag(FLAGS_mlperf_conf), absl::GetFlag(FLAGS_user_conf),
      absl::GetFlag(FLAGS_init_aic), absl::GetFlag(FLAGS_init_old_delegate));

  system(std::string("mkdir -p -m 0777 " + option.outdir).c_str());
  if(option.total_sample_count == 0) {
    if(option.model_name == "mobilenet" || option.model_name == "resnet50") {
      option.total_sample_count = 50000;
    } else if(option.model_name == "ssd-mobilenet") {
      option.total_sample_count = 5000;
    }
  }
  RunInference(option);
  return 0;
}
