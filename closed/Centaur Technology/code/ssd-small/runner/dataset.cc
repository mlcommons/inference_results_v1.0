#include "experimental/mlperf/inference/runner/dataset.h"

#include "absl/strings/str_join.h"
#include "ncoresw/common/logging.h"
#include "tensorflow/core/platform/env.h"

namespace mlperf {
namespace {

class DataSetFile {
 public:
  DataSetFile(const std::string& filename) : offset_(0) {
    tensorflow::Env* env = tensorflow::Env::Default();
    TF_CHECK_OK(env->NewRandomAccessFile(filename, &file_));
  }

  ~DataSetFile() = default;

  void ReadToBufferOrDie(char* content, int64 length, int64* bytes_read) {
    absl::string_view result;
    TF_CHECK_OK(file_->Read(offset_, length, &result, content));
    *bytes_read = result.size();
    offset_ += result.size();
  }

  void SeekOrDie(int64 position) { offset_ = position; }

 private:
  size_t offset_;
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
};

void ParseNpyHeader(DataSetFile* fp, tensorflow::DataType* element_type,
                    std::vector<int64>* shape) {
  // The header consists of a prolog (11 bytes) and a body (<256 bytes).
  constexpr int kPrologSize = 11;
  constexpr int kBodySize = 256;
  constexpr int kHeaderSize = kPrologSize + kBodySize;
  std::array<char, kHeaderSize> buffer;
  int64 bytes_read;
  fp->ReadToBufferOrDie(buffer.begin(), kHeaderSize, &bytes_read);
  std::string header(buffer.begin(), buffer.end());

  auto body_begin = header.begin();
  body_begin += kPrologSize;

  // Seek to just after the first '\n'
  auto body_end = std::find(body_begin, header.end(), '\n');
  NCORESW_CHECK(body_end != header.end()) << "Cannot find header end.";

  fp->SeekOrDie(body_end - header.begin() + 1);

  auto shape_begin = std::find(body_begin, body_end, '(');
  NCORESW_CHECK(shape_begin != body_end) << "Cannot find shape start.";

  auto shape_end = std::find(shape_begin, body_end, ')');
  NCORESW_CHECK(shape_end != body_end) << "Cannot find shape end.";

  for (auto it = shape_begin; it != shape_end;) {
    ++it;
    auto next = std::find(it, shape_end, ',');
    std::string dim = header.substr(it - header.begin(), next - it);
    shape->push_back(std::stoi(dim));
    it = next;
  }

  if (header.find("\'descr\': \'<i4\'", body_begin - header.begin()) != std::string::npos) {
    *element_type = tensorflow::DT_INT32;
  } else if (header.find("\'descr\': \'<f4\'", body_begin - header.begin()) != std::string::npos) {
    *element_type = tensorflow::DT_FLOAT;
  } else if (header.find("\'descr\': \'|i1\'", body_begin - header.begin()) != std::string::npos) {
    *element_type = tensorflow::DT_INT8;
  } else if (header.find("\'descr\': \'|u1\'", body_begin - header.begin()) != std::string::npos) {
    *element_type = tensorflow::DT_UINT8;
  } else {
    NCORESW_LOG(FATAL) << "ParseNpyHeader: Unknown element type.";
  }
}

void ReadNpyFileToBuffer(const std::string& filename, const std::vector<int64>& shape,
                         tensorflow::DataType type, int64 bytes_per_sample, char* buf) {
  DataSetFile fp(filename);

  tensorflow::DataType element_type;
  std::vector<int64> element_shape;
  ParseNpyHeader(&fp, &element_type, &element_shape);
  NCORESW_CHECK_EQ(element_type, type) << "Data type mismatch";
  //NCORESW_LOG(INFO) << "element_shape: " << absl::StrJoin(element_shape, ", ");
  //NCORESW_LOG(INFO) << "shape: " << absl::StrJoin(shape, ", ");
  NCORESW_CHECK(element_shape == shape) << "Shape mismatch";

  int64 bytes_read;
  fp.ReadToBufferOrDie(buf, bytes_per_sample, &bytes_read);
  NCORESW_CHECK_EQ(bytes_read, bytes_per_sample) << "Too few bytes in file.";
}

}  // namespace

tensorflow::Tensor CreateQSLOrDie(const std::string& dataset_path,
                                  const std::vector<QuerySampleIndex>& sample_ids,
                                  const std::vector<int64>& shape, tensorflow::DataType type,
                                  const mlperf::Option& option) {
  if (sample_ids.empty()) {
    return tensorflow::Tensor();
  }

  // Encode the batch dimension by looking at how many samples we are given by loadgen to process.
  // Then add the rest of the dimensions
  auto batch_dim = sample_ids.size();
  std::vector<int64> result_shape;
  result_shape.push_back(batch_dim);
  for (auto dim_size : shape) {
    result_shape.push_back(dim_size);
  }
  //NCORESW_LOG(INFO) << "result_shape: " << absl::StrJoin(result_shape, ", ");
  tensorflow::Tensor result_tensor(type, tensorflow::TensorShape(result_shape));

  char* buf;
  //NCORESW_LOG(INFO) << "create qsl, dtype: " << type;
  int element_size;
  if (type == tensorflow::DT_FLOAT) {
    buf = reinterpret_cast<char*>(result_tensor.flat<float>().data());
    element_size = sizeof(float);
  } else if (type == tensorflow::DT_INT32) {
    buf = reinterpret_cast<char*>(result_tensor.flat<int>().data());
    element_size = sizeof(int);
  } else if (type == tensorflow::DT_INT8) {
    buf = reinterpret_cast<char*>(result_tensor.flat<int8_t>().data());
    element_size = sizeof(int8_t);
  } else if (type == tensorflow::DT_UINT8) {
    buf = reinterpret_cast<char*>(result_tensor.flat<uint8_t>().data());
    element_size = sizeof(uint8_t);
  } else {
    NCORESW_LOG(FATAL) << "Unknown element type.";
  }

  int64 bytes_per_sample =
      std::accumulate(shape.begin(), shape.end(), element_size, std::multiplies<int64>());
  for (const auto& sample_id : sample_ids) {

    std::string filename;
    filename = dataset_path + GetPreprocessedImageFilename(sample_id, option);

    ReadNpyFileToBuffer(filename, shape, type, bytes_per_sample, buf);
    buf += bytes_per_sample;
  }

  return result_tensor;
}

std::unordered_map<QuerySampleIndex, QuerySampleIndex> CreateSampleIdToQSLIndexMap(
    const std::vector<QuerySampleIndex>& sample_ids) {
  std::unordered_map<QuerySampleIndex, QuerySampleIndex> sample_id_to_qsl_idx_map;
  for (int i = 0; i < sample_ids.size(); ++i) {
    sample_id_to_qsl_idx_map[sample_ids[i]] = i;
  }
  return sample_id_to_qsl_idx_map;
}

std::string GetPreprocessedImageFilename(const uint64_t sample_id, const mlperf::Option& option) {
  std::string filename = "";
  if(option.model_name == "mobilenet" || option.model_name == "resnet50") {
    std::ostringstream ss;
    ss << std::setw(8) << std::setfill('0') << (sample_id + 1);
    filename = "ILSVRC2012_val_" + ss.str() + ".JPEG.npy";
  } else if(option.model_name == "ssd-mobilenet") {
    filename = CocoPreprocessedImageFilename(sample_id);
  }
  return filename;
}

}  // namespace mlperf
