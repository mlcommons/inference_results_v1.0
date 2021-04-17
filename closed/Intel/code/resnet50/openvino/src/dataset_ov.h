#ifndef DATASET_H__
#define DATASET_H__

#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <map>

// loadgen
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"

#include <inference_engine.hpp>
#include <ie_blob.h>

#include "item_ov.h"

using namespace InferenceEngine;
using namespace std;
using namespace cv;


class Dataset {
public:
    Dataset() {
    }

    ~Dataset() {
    }

    int getItemCount() {
        return image_list_.size();
    }

    virtual void loadQuerySamples(const mlperf::QuerySampleIndex* samples,
            size_t num_samples) {
        if (image_list_inmemory_.size() == 0) {
            if ((this->settings_.scenario == mlperf::TestScenario::SingleStream)
                    || (this->settings_.scenario == mlperf::TestScenario::Server)) {
                this->image_list_inmemory_.resize(total_count_);
            } else if (this->settings_.scenario
                    == mlperf::TestScenario::Offline) {
                image_list_inmemory_.resize(num_samples);
            } else if (this->settings_.scenario
                    == mlperf::TestScenario::MultiStream) {
                image_list_inmemory_.resize(num_samples);
                sample_list_inmemory_.resize(num_samples);
            }
        }

        mlperf::QuerySampleIndex sample;

        handle = (new float[num_samples * num_channels_ * image_height_
                * image_width_]);
	
        for (uint i = 0; i < num_samples; ++i) {
            cv::Mat processed_image;
            sample = (*samples);
            samples++;

            std::string image_path;

            if (dataset_.compare("imagenet") == 0) {
                image_path = this->datapath_ + "/" + image_list_[sample];
            }
            else if (dataset_.compare("coco") == 0) {
                image_path = this->datapath_ + "/val2017/" + image_list_[sample];
            }

            auto image = cv::imread(image_path);

            if (image.empty()) {
                throw std::logic_error("Invalid image at path: " + i);
            }

            if (this->workload_.compare("resnet50") == 0) {
                preprocessVGG(&image, &processed_image);
            } else if (this->workload_.compare("mobilenet") == 0) {
                preprocessMobilenet(&image, &processed_image);
            } else if (this->workload_.compare("ssd-mobilenet") == 0) {
                preprocessSSDMobilenet(&image, &processed_image);
            } else if (this->workload_.compare("ssd-resnet34") == 0) {
                preprocessSSDResnet(&image, &processed_image);
            } else if (this->workload_.compare("ssd-mobilenet-v2") == 0) {
		preprocessSSDMobilenet(&image, &processed_image);
            } else if (this->workload_.compare("mobilenet-edge") == 0) {
		preprocessMobilenet(&image, &processed_image);
            }

            processed_image.copyTo(image);
            size_t image_size = (image_height_ * image_width_);

            InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32,
                    { 1, num_channels_, image_height_, image_width_ },
                    InferenceEngine::Layout::NCHW);

            auto input = make_shared_blob < PrecisionTrait < Precision::FP32
                    > ::value_type
                    > (tDesc, ((((float *) handle)
                            + (i * image_size * num_channels_))), image_size
                            * num_channels_);

            for (size_t image_id = 0; image_id < 1; ++image_id) {

                for (size_t pid = 0; pid < image_size; pid++) {

                    for (size_t ch = 0; ch < num_channels_; ++ch) {
                        input->data()[image_id * image_size * num_channels_
                                + ch * image_size + pid] = image.at < cv::Vec3f
                                > (pid)[ch];
                    }
                }
            }

            if (settings_.scenario == mlperf::TestScenario::Offline) {
                image_list_inmemory_[i] = input;
            } else if ((settings_.scenario == mlperf::TestScenario::SingleStream)
                    || (settings_.scenario == mlperf::TestScenario::Server)) {
                image_list_inmemory_[sample] = input;
            } else if (this->settings_.scenario
                    == mlperf::TestScenario::MultiStream) {
                image_list_inmemory_[i] = input;
                sample_list_inmemory_[i] = sample;
            }
        }
    }

    virtual void unloadQuerySamples(const mlperf::QuerySampleIndex* samples,
            size_t num_samples) {

        image_list_inmemory_.clear();
        delete [] handle;
    }

    void getSamplesSquad(const mlperf::QuerySampleIndex* samples, std::vector<Blob::Ptr>* data) 
    {
        data->push_back(input_ids_inmemory_[samples[0]]);
        data->push_back(input_mask_inmemory_[samples[0]]);
        data->push_back(segment_ids_inmemory_[samples[0]]);
    }

    void getSamples(const mlperf::QuerySampleIndex* samples, Blob::Ptr* data,
            int* label) {
        *data = image_list_inmemory_[samples[0]];
        if (dataset_.compare("imagenet") == 0) {
            *label = label_list_[samples[0]];
        }
    }

    virtual void getSample(const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> queryIds, size_t bs,
            int num_batches, Item *item) {

        std::vector< Blob::Ptr > input { image_list_inmemory_[samples[0]] };
	*item = Item(input, queryIds, samples); 
	
    }

    virtual void getSamplesBatched(const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> queryIds, size_t bs,
            int num_batches, std::vector<Item> &items) {

        size_t image_size = (image_height_ * image_width_);
        InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, {
                (bs), num_channels_, image_height_, image_width_ },
                InferenceEngine::Layout::NCHW);
        TBlob<PrecisionTrait<Precision::FP32>::value_type>::Ptr input;
        
	for (int i = 0; i < num_batches; ++i) {
            auto start = (i * bs) % perf_count_;

	    std::vector< Blob::Ptr > inputs { make_shared_blob < PrecisionTrait < Precision::FP32 > ::value_type >
                (tDesc, image_list_inmemory_[start]->buffer(), (bs * image_size * num_channels_)) };
	    
        std::vector < mlperf::QuerySampleIndex > idxs;
        std::vector < mlperf::ResponseId > ids;
        for (size_t j = (i * bs); j < (unsigned) ((i * bs) + bs); ++j) {
            ids.push_back(queryIds[j]);
            idxs.push_back(samples[j]);
        }
        items.push_back(Item(inputs, ids, idxs));//, label));
    }
    }

    void getSamplesBatched(const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> queryIds, size_t bs,
            int num_batches, std::vector<Item> &items, size_t num_input_layers) {
        size_t image_size = (image_height_ * image_width_);
        InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, {
                (bs), num_channels_, image_height_, image_width_ },
                InferenceEngine::Layout::NCHW);
        TBlob<PrecisionTrait<Precision::FP32>::value_type>::Ptr input;
        for (int i = 0; i < num_batches; ++i) {
            auto start = (i * bs) % perf_count_;

            input = make_shared_blob < PrecisionTrait < Precision::FP32
                    > ::value_type
                    > (tDesc, image_list_inmemory_[start]->buffer(), (bs
                            * image_size * num_channels_));

            std::vector < mlperf::QuerySampleIndex > idxs;
            std::vector < mlperf::ResponseId > ids;
            for (size_t j = (i * bs); j < (unsigned) ((i * bs) + bs); ++j) {
                ids.push_back(queryIds[j]);
                idxs.push_back(samples[j]);
            }
            std::vector<Blob::Ptr> inputs{ input };
            items.push_back(Item(inputs, ids, idxs));
        }
    }

    void getSamplesBatchedServer(
            const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> queryIds, size_t bs,
            int num_batches, std::vector<Item> &items) {

        size_t image_size = (image_height_ * image_width_);
        InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, {
                (bs), num_channels_, image_height_, image_width_ },
                InferenceEngine::Layout::NCHW);

        for (int i = 0; i < num_batches; ++i) {
            auto input = make_shared_blob < PrecisionTrait < Precision::FP32
                    > ::value_type > (tDesc);
            input->allocate();

            for (size_t k = 0; k < bs; ++k) {
                auto start = samples[k];

                std::memcpy((input->data() + (k * image_size * num_channels_)),
                        image_list_inmemory_[start]->buffer().as<
                                PrecisionTrait<Precision::FP32>::value_type *>(),
                        (image_size * num_channels_ * sizeof(float)));
            }
            std::vector < mlperf::QuerySampleIndex > idxs;
            std::vector < mlperf::ResponseId > ids;
            for (size_t j = 0; j < (unsigned) bs; ++j) {
                ids.push_back(queryIds[j]);
                idxs.push_back(samples[j]);
            }

            items.push_back(Item(input, ids, idxs));
        }
    }

    virtual void getSamplesBatchedMultiStream(
            const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> queryIds, size_t bs,
            int num_batches, std::vector<Item> &items) {
        size_t image_size = (image_height_ * image_width_);
        InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, {
                (bs), num_channels_, image_height_, image_width_ },
                InferenceEngine::Layout::NCHW);

        // find sample
        std::vector<mlperf::QuerySampleIndex>::iterator it = std::find(
                sample_list_inmemory_.begin(), sample_list_inmemory_.end(),
                samples[0]);
        if (!(it != std::end(sample_list_inmemory_))) {
            std::cout << "    [ERROR] No sample found\n ";
        }

        int index = std::distance(sample_list_inmemory_.begin(), it);
        auto start = index;
        auto query_start = 0;
        for (int i = 0; i < num_batches; ++i) {
            TBlob<PrecisionTrait<Precision::FP32>::value_type>::Ptr input;

            input = make_shared_blob < PrecisionTrait < Precision::FP32
                    > ::value_type
                    > (tDesc, image_list_inmemory_[start]->buffer(), (bs
                            * image_size * num_channels_));

            std::vector < mlperf::QuerySampleIndex > idxs;
            std::vector < mlperf::ResponseId > ids;
            for (size_t j = (query_start); j < (unsigned) (query_start + bs);
                    ++j) {
                ids.push_back(queryIds[j]);
                idxs.push_back(samples[j]);
            }

            items.push_back(Item(input, ids, idxs));

            start = start + bs;
            query_start = query_start + bs;
        }
    }

    // Preprocessing routines
    void centerCrop(cv::Mat* image, int out_height, int out_width,
            cv::Mat* cropped_image) {
        int width = (*image).cols;
        int height = (*image).rows;
        int left = int((width - out_width) / 2);
        int top = int((height - out_height) / 2);
        cv::Rect customROI(left, top, out_width, out_height);

        (*cropped_image) = (*image)(customROI);
    }

    void resizeWithAspectratio(cv::Mat* image, cv::Mat* resized_image,
            int out_height, int out_width, int interpol, float scale = 87.5) {
        int width = (*image).cols;
        int height = (*image).rows;
        int new_height = int(100. * out_height / scale);
        int new_width = int(100. * out_width / scale);

        int h, w = 0;
        if (height > width) {
            w = new_width;
            h = int(new_height * height / width);
        } else {
            h = new_height;
            w = int(new_width * width / height);
        }

        cv::resize((*image), (*resized_image), cv::Size(w, h), interpol);
    }

    void preprocessSSDMobilenet(cv::Mat* image, cv::Mat* processed_image) {
        cv::Mat img, resized_image, float_image, u8_image;

        image->convertTo(float_image, CV_32F);
        if (num_channels_ < 3) {
            cv::cvtColor(float_image, img, cv::COLOR_GRAY2RGB);
        } else {
            cv::cvtColor(float_image, img, cv::COLOR_BGR2RGB);
        }

        cv::resize((img), (resized_image),
                cv::Size(image_width_, image_height_), cv::INTER_LINEAR);

        resized_image.copyTo(*processed_image);
    }

    void preprocessSSDResnet(cv::Mat* image, cv::Mat* processed_image) {
        cv::Mat img, resized_image, sub_image, float_image, div_image,
                std_image;

        image->convertTo(float_image, CV_32F);
        if (num_channels_ < 3) {
            cv::cvtColor(float_image, img, cv::COLOR_GRAY2RGB);
        } else {
            cv::cvtColor(float_image, img, cv::COLOR_BGR2RGB);
        }

        cv::resize((img), (resized_image),
                cv::Size(image_width_, image_height_), cv::INTER_LINEAR);

        resized_image.copyTo(*processed_image);
    }

    void preprocessVGG(cv::Mat* image, cv::Mat* processed_image) {
        cv::Mat img, resized_image, cropped_image, float_image, norm_image,
                sub_image;
        cv::cvtColor(*image, img, cv::COLOR_BGR2RGB);

        resizeWithAspectratio(&img, &resized_image, image_height_, image_width_,
                cv::INTER_AREA);

        centerCrop(&resized_image, image_height_, image_width_, &cropped_image);

        cropped_image.convertTo(*processed_image, CV_32FC3);
    }

    void preprocessMobilenet(cv::Mat* image, cv::Mat* processed_image) {
        cv::Mat img, resized_image, cropped_image, float_image, norm_image,
                sub_image;
        cv::cvtColor(*image, img, cv::COLOR_BGR2RGB);

        resizeWithAspectratio(&img, &resized_image, image_height_, image_width_,
                cv::INTER_LINEAR);

        centerCrop(&resized_image, image_height_, image_width_, &cropped_image);

        cropped_image.convertTo(float_image, CV_32FC3);
        float_image.copyTo(*processed_image);
    }

public:
    std::vector<string> image_list_;
    std::vector<int> label_list_;
    std::vector<Blob::Ptr> image_list_inmemory_;
    std::vector<std::pair<Blob::Ptr, Blob::Ptr>> data_list_inmemory_;
    std::vector<mlperf::QuerySampleIndex> sample_list_inmemory_;
    size_t image_width_;
    size_t image_height_;
    size_t num_channels_;
    string datapath_;
    string image_format_;
    size_t total_count_;
    size_t perf_count_;
    bool need_transpose_ = false;
    string workload_ = "resnet50";
    mlperf::TestSettings settings_;
    string dataset_;
    float * handle;
    std::vector<std::vector<int>> squad_input_ids_;
    std::vector<std::vector<int> > squad_input_mask_;
    std::vector<std::vector<int>> squad_segment_ids_;
    std::vector<std::vector<string>> squad_tokens_;
    std::vector<Blob::Ptr> input_ids_inmemory_;
    std::vector<Blob::Ptr> input_mask_inmemory_;
    std::vector<Blob::Ptr> segment_ids_inmemory_;

}
;

class Imagenet: public Dataset {
public:
    Imagenet(mlperf::TestSettings settings, int image_width, int image_height,
            int num_channels, string datapath, string image_format,
            int total_count, int perf_count, string workload, string dataset) {
        this->image_width_ = image_width;
        this->image_height_ = image_height;
        this->num_channels_ = num_channels;
        this->datapath_ = datapath;
        this->image_format_ = image_format;
        this->total_count_ = total_count;
        this->perf_count_ = perf_count;
        this->settings_ = settings;

        this->workload_ = workload;
        this->dataset_ = dataset;

        this->need_transpose_ = image_format == "NHWC" ? false : true;

        string image_list_file = datapath + "/val_map.txt";

		
        boost::filesystem::path p( image_list_file );
        if (!(boost::filesystem::exists( p ) ) ) {
            std::cout <<" Imagenet validation list file '" + image_list_file + "' not found. Pleasemake sure data_path contains 'val_map.txt'\n";
            throw;
        }
		

        std::ifstream imglistfile;
        imglistfile.open(image_list_file, std::ios::binary);

        std::string line, image_name, label;
        if (imglistfile.is_open()) {
            while (getline(imglistfile, line)) {
                std::regex ws_re("\\s+");
                std::vector < std::string
                        > img_label { std::sregex_token_iterator(line.begin(),
                                line.end(), ws_re, -1), { } };
                label = img_label.back();
                image_name = img_label.front();
                img_label.clear();

                this->image_list_.push_back(image_name);
                this->label_list_.push_back(stoi(label));

                // limit dataset
                if (total_count_
                        && (image_list_.size() >= (uint) total_count_)) {
                    break;
                }
            }
        }

        imglistfile.close();

        if (!image_list_.size()) {
            std::cout << "No images in image list found";
        }
    }

    ~Imagenet() {
    }
};

class Coco: public Dataset {
public:
    Coco(mlperf::TestSettings settings, int image_width, int image_height,
            int num_channels, string datapath, string image_format,
            int total_count, int perf_count, string workload, string dataset) {
        this->image_width_ = image_width;
        this->image_height_ = image_height;
        this->num_channels_ = num_channels;
        this->datapath_ = datapath;
        this->image_format_ = image_format;
        this->total_count_ = total_count;
        this->perf_count_ = perf_count;
        this->settings_ = settings;

        this->workload_ = workload;
        this->dataset_ = dataset;
        this->need_transpose_ = image_format == "NHWC" ? false : true;

        string image_list_file = datapath + "/annotations/instances_val2017.json";

        boost::filesystem::path p( image_list_file );
        if (!(boost::filesystem::exists( p ) ) ) {
            std::cout << "[ERROR] COCO annotation file '" + image_list_file + "' not found. Pleasemake sure --data_path contains 'annotations/instances_val2017.json'\n";
            throw;
        }

        std::ifstream imglistfile(image_list_file);
        if (imglistfile) {
            std::stringstream buffer;

            buffer << imglistfile.rdbuf();
            boost::property_tree::ptree pt;
            boost::property_tree::read_json(buffer, pt);

            // Parse JSON for filenames
            for (auto& img : pt.get_child("images")) {

                for (auto& prop : img.second) {
                    if (prop.first == "file_name") {
                        this->image_list_.push_back(
                                prop.second.get_value<std::string>());
                    }
                    // limit dataset
                    if (total_count_
                            && (image_list_.size() >= (uint) total_count_)) {
                        break;
                    }
                }
            }
        }

        if (!image_list_.size()) {
            std::cout << "No images in image list found";
        }
    }

    ~Coco() {
    }
};

class ADE20K : public Dataset {
public:
    ADE20K(mlperf::TestSettings settings, int image_width, int image_height,
        int num_channels, string datapath, string image_format,
        int total_count, int perf_count, string workload, string dataset) {
        this->image_width_ = image_width;
        this->image_height_ = image_height;
        this->num_channels_ = num_channels;
        this->datapath_ = datapath;
        this->image_format_ = image_format;
        this->total_count_ = total_count;
        this->perf_count_ = perf_count;
        this->settings_ = settings;
        this->workload_ = workload;
        this->dataset_ = dataset;
        this->need_transpose_ = image_format == "NHWC" ? false : true;

        string image_list_file = datapath + "/val.txt";  

        boost::filesystem::path p( image_list_file );
        if (!(boost::filesystem::exists( p ) ) ) {
            std::cout << " ADE20K Validation image list file '" + image_list_file + "' not found. Please make sure data_path contains 'val.txt'\n";
            throw;
        }
        
        std::ifstream imglistfile;
        imglistfile.open(image_list_file);

        std::string line, image_name, label;
        if (imglistfile.is_open()) {
            while (getline(imglistfile, line)) {
                
                image_name = line;
                this->image_list_.push_back(image_name);
                if (total_count_  && (image_list_.size() >= (uint)total_count_)) {
                    break;
                }
            }
        }
        else {
            std::cout << "[ERROR] Unable to read image list file\n";
            throw std::logic_error("Aborted");
        }

        imglistfile.close();

        if (!image_list_.size()) {
            std::cout << "No images in image list found";
        }
    }

    void loadQuerySamples(const mlperf::QuerySampleIndex* samples,
        size_t num_samples) {
        if (image_list_inmemory_.size() == 0) {
            if ((this->settings_.scenario == mlperf::TestScenario::SingleStream)
                || (this->settings_.scenario == mlperf::TestScenario::Server)) {
                this->image_list_inmemory_.resize(total_count_);
            }
            else if (this->settings_.scenario
                == mlperf::TestScenario::Offline) {
                image_list_inmemory_.resize(num_samples);
            }
            else if (this->settings_.scenario
                == mlperf::TestScenario::MultiStream) {
                image_list_inmemory_.resize(num_samples);
                sample_list_inmemory_.resize(num_samples);
            }
        }

        mlperf::QuerySampleIndex sample;
	size_t image_size = (image_height_ * image_width_);
        InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, { 1, num_channels_, image_height_, image_width_ }, InferenceEngine::Layout::NCHW);
        handle = (new float[num_samples * num_channels_ * image_size]);

        for (uint i = 0; i < num_samples; ++i) {
            cv::Mat processed_image;
            sample = (*samples);
            samples++;

            std::string image_path = this->datapath_ + "/" + image_list_[sample];
            auto image = cv::imread(image_path);

            if (image.empty()) {
                throw std::logic_error("Invalid image at path: " + i);
            }

            if (DEBUG_PREPROCESSING)
            {
                std::cout << " == Sample image: " << image_list_[sample] << " ==" << std::endl;
            }
            
            preprocessDeepLabv3(image, &processed_image);
            
            auto input = make_shared_blob < PrecisionTrait < Precision::FP32 > ::value_type > 
                (tDesc, ((((float*)handle) + (i * image_size * num_channels_))), image_size * num_channels_);

            for (size_t image_id = 0; image_id < 1; ++image_id) {

                for (size_t pid = 0; pid < image_size; pid++) {

                    for (size_t ch = 0; ch < num_channels_; ++ch) {
                        input->data()[image_id * image_size * num_channels_ + ch * image_size + pid] = processed_image.at < cv::Vec3f >(pid)[ch];
                    }
                }
            }

            if (settings_.scenario == mlperf::TestScenario::Offline) {
                image_list_inmemory_[i] = input;
            }
            else if ((settings_.scenario == mlperf::TestScenario::SingleStream)
                || (settings_.scenario == mlperf::TestScenario::Server)) {
                image_list_inmemory_[sample] = input;
            }
            else if (this->settings_.scenario
                == mlperf::TestScenario::MultiStream) {
                image_list_inmemory_[i] = input;
                sample_list_inmemory_[i] = sample;
            }
        }
    }
     void resize_to_range(cv::Mat inp_image, cv::Mat* out_image) {
        int max_size = max_resize_value_ - (max_resize_value_ - 1) % resize_factor_;
        float min_size = static_cast<float>(min_resize_value_);

        float img_height = static_cast<float>(inp_image.size().height);
        float img_width = static_cast<float>(inp_image.size().width);
        
        float img_min_size = std::min({ img_width, img_height });
        float large_scale_factor = min_size / img_min_size;
        int large_height = static_cast<int>(img_height * large_scale_factor);
        int large_width = static_cast<int>(img_width * large_scale_factor);

        int new_height = large_height;
        int new_width = large_width;

        float img_max_size = std::max({ img_width, img_height });
        float small_scale_factor = ((float)max_size) / img_max_size;
        int small_height = static_cast<int>(img_height * small_scale_factor);
        int small_width = static_cast<int>(img_width * small_scale_factor);

        if (std::max({ large_height, large_width }) > max_size)
        {
            new_height = small_height;
            new_width = small_width;
        }

        new_height += (resize_factor_ - (new_height - 1) % resize_factor_) % resize_factor_;
        new_width += (resize_factor_ - (new_width - 1) % resize_factor_) % resize_factor_;

        cv::resize(inp_image, *out_image, cv::Size(new_width, new_height), cv::INTER_LINEAR);
        if (DEBUG_PREPROCESSING)
        {
            std::cout << " Resized: " << out_image->size() << " \n";
        }
        
    }

    void pad_to_bounding_box(cv::Mat inp_image, cv::Mat* out_image, int offset_height, int offset_width, int target_height, int target_width) {
        float img_height = static_cast<float>(inp_image.size().height);
        float img_width = static_cast<float>(inp_image.size().width);

        int after_padding_width = target_width - offset_width - img_width;
        int after_padding_height = target_height - offset_height - img_height;

        int borderType = cv::BORDER_CONSTANT;
        cv::Scalar Pad0(127.5, 127.5, 127.5); // From original implementation


        if (DEBUG_PREPROCESSING)
        {
            std::cout << " Padding: " << offset_height << "," << after_padding_height << "," << offset_width << "," << after_padding_width << " \n";
        }
       
        copyMakeBorder(inp_image, *out_image, offset_height, after_padding_height, offset_width, after_padding_width, borderType, Pad0);
        
        
    }


    void preprocessDeepLabv3(cv::Mat image, cv::Mat* processed_image) {

        cv::Mat res_image, temp_im;

        if (num_channels_ < 3) {
            cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
        }
        else {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        }

        image.convertTo(image, CV_32F);
        if (DEBUG_PREPROCESSING)
        {
            std::cout << " Original Size: " << image.size() << "\n";
        }
	    if(use_tf_preproc_) {
            // Use preprocessing implemented by tensorflow
            if (DEBUG_PREPROCESSING)
            {
                std::cout << " Using TF implementation of preprocessing\n";
            }
            
            resize_to_range(image, &res_image);

            int img_height = res_image.size().height;
            int img_width = res_image.size().width;
            int offset_height = 0;
            int offset_width = 0;

            int target_height = img_height + std::max({ crop_height_ - img_height - offset_height, 0 });
            int target_width = img_width + std::max({ crop_width_ - img_width - offset_width, 0 });

            pad_to_bounding_box(res_image, processed_image, offset_height, offset_width, target_height, target_width);
        }
        else {
            // Just use simple resize
            if (DEBUG_PREPROCESSING)
            {
                std::cout << " Using simple resizing\n";
            }
            
            cv::resize(image, *processed_image, cv::Size(image_width_, image_height_), cv::INTER_LINEAR);
        }

    }

    ~ADE20K() {
    }
    bool DEBUG_PREPROCESSING = false; // true;
    int min_resize_value_ = 512;
    int max_resize_value_ = 512;
    int crop_height_ = 512;
    int crop_width_ = 512;
    int resize_factor_ = 1;

    bool use_tf_preproc_ = true;

};

class Squad : public Dataset {

public:
	Squad(mlperf::TestSettings settings, int max_seq_length, int max_query_length,
            int doc_stride, string datapath, int total_count, int perf_count, string workload, string dataset) {
        this->datapath_ = datapath;
        this->total_count_ = total_count;
        this->perf_count_ = perf_count;
        this->settings_ = settings;

        this->workload_ = workload;
        this->dataset_ = dataset;

        string vocab_file = datapath  + "/vocab.txt";
	    string data_json = datapath + "/dev-v1.1.json";
        string output_dir = "samples_cache";
        
	boost::filesystem::path p_vocab( vocab_file );
        if (!(boost::filesystem::exists( p_vocab ) ) ) {
            throw std::logic_error(" SQUAD vocab file '" + vocab_file + "' not found. Please make sure --data_path contains 'vocab.txt'");
        }
       

        boost::filesystem::path p_json( data_json );
        if (!(boost::filesystem::exists( p_json ) ) ) {
            throw std::logic_error(" SQUAD data file '" + data_json + "' not found. Please makesure --data_path contains 'dev-v1.1.json'");
        }

        string output_json = output_dir + "/squad_examples.json";
	boost::filesystem::path o_json( output_json );

	if ( !(boost::filesystem::exists( o_json ) ) ){
        std::cout << "    [INFO] Preprocessing SQuAD samples\n";
	        const std::string cmd = "python3 py-bindings/convert.py --vocab_file " + vocab_file + " --output_dir " + output_dir + " --test_file " + data_json;

	        int ret_val = system(cmd.c_str());
	}	
        
        if (!(boost::filesystem::exists( o_json ) ) ) {
            throw std::logic_error(" SQUAD data preprocessed file '" + output_json + "' not found.");
        }
        else
        {
           cout<< "    [INFO] Reading SQuAD data preprocessed file at: "<< output_json <<"\n";
        } 
      
        std::ifstream squadjsonfile(output_json);
        if (squadjsonfile) {
            std::stringstream buffer;

            buffer << squadjsonfile.rdbuf();
            boost::property_tree::ptree pt;
            boost::property_tree::read_json(buffer, pt);
            for (auto& sample : pt.get_child("samples")) {
                for (auto& prop : sample.second) {

                        if (prop.first == "segment_ids") {
                        boost::property_tree::ptree subtree = (boost::property_tree::ptree) prop.second ;
                        BOOST_FOREACH(boost::property_tree::ptree::value_type &vs,
                            subtree)
                        {
                            sample_segment_ids.push_back(std::stoi(vs.second.data()));
                        }
                        this->squad_segment_ids_.push_back(
                                sample_segment_ids);
                        sample_segment_ids.clear();
                    }
                    if (prop.first == "input_mask") {
                        boost::property_tree::ptree subtree = (boost::property_tree::ptree) prop.second ;
                        BOOST_FOREACH(boost::property_tree::ptree::value_type &vs,
                            subtree)
                        {
                            sample_input_mask.push_back(std::stoi(vs.second.data()));
                        }
                        this->squad_input_mask_.push_back(
                                sample_input_mask);
                        sample_input_mask.clear();
                     } 
                        if (total_count_
                            && (squad_input_ids_.size() >= (uint) total_count_)) {
                        break;
                    }
                }
            }           

            for (auto& sample : pt.get_child("samples")) {
                for (auto& prop : sample.second) {
                    if (prop.first == "input_ids") {
                        boost::property_tree::ptree subtree = (boost::property_tree::ptree) prop.second ;
                        BOOST_FOREACH(boost::property_tree::ptree::value_type &vs,
                            subtree)
                        {
                            sample_input_ids.push_back(std::stoi(vs.second.data()));
                        }
                        this->squad_input_ids_.push_back(
                                sample_input_ids);
                        sample_input_ids.clear();
                    }
                        if (total_count_
                            && (squad_input_ids_.size() >= (uint) total_count_)) {
                        break;
                    }
                }
            }
        }
           
    }

    void loadQuerySamples(const mlperf::QuerySampleIndex* samples,
            size_t num_samples) {
        if (input_ids_inmemory_.size() == 0) {
            if ((this->settings_.scenario == mlperf::TestScenario::SingleStream)
                    || (this->settings_.scenario == mlperf::TestScenario::Server)) {
                this->input_ids_inmemory_.resize(total_count_);
                this->input_mask_inmemory_.resize(total_count_);
                this->segment_ids_inmemory_.resize(total_count_);
            } else if (this->settings_.scenario
                    == mlperf::TestScenario::Offline) {
                input_ids_inmemory_.resize(num_samples);
                input_mask_inmemory_.resize(num_samples);
                segment_ids_inmemory_.resize(num_samples);
            } else if (this->settings_.scenario
                    == mlperf::TestScenario::MultiStream) {
                input_ids_inmemory_.resize(num_samples);
                input_mask_inmemory_.resize(num_samples);
                segment_ids_inmemory_.resize(num_samples);
                sample_list_inmemory_.resize(num_samples); 
            }
        }

        mlperf::QuerySampleIndex sample;

        for (uint i = 0; i < num_samples; ++i) {
            sample = (*samples);
            samples++;

            InferenceEngine::TensorDesc desc(InferenceEngine::Precision::I32, {1, max_seq_length_}, InferenceEngine::Layout::NC );
            auto m_inp0 = make_shared_blob < PrecisionTrait < Precision::I32 >::value_type > (desc);
            auto m_inp1 = make_shared_blob < PrecisionTrait < Precision::I32 >::value_type > (desc);
            auto m_inp2 = make_shared_blob < PrecisionTrait < Precision::I32 >::value_type > (desc);

            m_inp0->allocate();
            m_inp1->allocate();
            m_inp2->allocate();

            for (size_t j = 0; j < max_seq_length_; j++){
                    m_inp0->data()[j] = static_cast<int32_t>( squad_input_ids_.at(sample).at(j));
                    m_inp1->data()[j] = static_cast<int32_t>( squad_input_mask_.at(sample).at(j));
                    m_inp2->data()[j] = static_cast<int32_t>( squad_segment_ids_.at(sample).at(j));
            }


            if (settings_.scenario == mlperf::TestScenario::Offline) {
                input_ids_inmemory_[i] = m_inp0;
                input_mask_inmemory_[i] = m_inp1;
		segment_ids_inmemory_[i] = m_inp2;
            } else if ((settings_.scenario == mlperf::TestScenario::SingleStream)
                    || (settings_.scenario == mlperf::TestScenario::Server)) {
                input_ids_inmemory_[sample] = m_inp0;
                input_mask_inmemory_[sample] = m_inp1;
                segment_ids_inmemory_[sample] = m_inp2;
            } else if (this->settings_.scenario
                    == mlperf::TestScenario::MultiStream) {
                input_ids_inmemory_[i] = m_inp0;
                input_mask_inmemory_[i] = m_inp1;
                segment_ids_inmemory_[i] = m_inp2;
                sample_list_inmemory_[i] = sample; 
            }
        }
    }

    void unloadQuerySamples(const mlperf::QuerySampleIndex* samples,
            size_t num_samples) {

	this->segment_ids_inmemory_.clear();
	this->input_ids_inmemory_.clear();
	this->input_mask_inmemory_.clear();
    }


     void getSample(const std::vector<mlperf::QuerySampleIndex> samples,
             std::vector<mlperf::ResponseId> queryIds, size_t bs,
             int num_batches, Item *item) {
         std::vector<Blob::Ptr> input;
         input.push_back(input_ids_inmemory_[samples[0]]);
         input.push_back(input_mask_inmemory_[samples[0]]);
         input.push_back(segment_ids_inmemory_[samples[0]]);
             *item = Item(input, queryIds, samples);
     }

    void getSamplesBatched(const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> queryIds, size_t bs,
            int num_batches, std::vector<Item> &items) {

        InferenceEngine::TensorDesc desc(InferenceEngine::Precision::I32, {bs, max_seq_length_}, InferenceEngine::Layout::NC );

        for (int i = 0; i < num_batches; ++i) {
            auto start = (i * bs) % perf_count_;

            Blob::Ptr input_ids = make_shared_blob< int32_t > (desc, input_ids_inmemory_[start]->buffer(), (bs * max_seq_length_));
            Blob::Ptr input_mask = make_shared_blob< int32_t > (desc, input_mask_inmemory_[start]->buffer(), (bs * max_seq_length_));
            Blob::Ptr segment_ids = make_shared_blob< int32_t > (desc, segment_ids_inmemory_[start]->buffer(), (bs * max_seq_length_));


            std::vector< Blob::Ptr > inputs {input_ids, input_mask, segment_ids};

            std::vector < mlperf::QuerySampleIndex > idxs;
            std::vector < mlperf::ResponseId > ids;

            for (size_t j = (i * bs); j < (unsigned) ((i * bs) + bs); ++j) {
                ids.push_back(queryIds[j]);
                idxs.push_back(samples[j]);
            }

            items.push_back(Item(inputs, ids, idxs));
        }
    }

    void getSamplesBatchedMultiStream(
            const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> queryIds, size_t bs,
            int num_batches, std::vector<Item> &items) {
        InferenceEngine::TensorDesc desc(InferenceEngine::Precision::I32, {bs, max_seq_length_}, InferenceEngine::Layout::NC );

        std::vector<mlperf::QuerySampleIndex>::iterator it = std::find(
                sample_list_inmemory_.begin(), sample_list_inmemory_.end(),
                samples[0]);
        if (!(it != std::end(sample_list_inmemory_))) {
            std::cout
                    << "getSamplesBatchedMultiStream: ERROR: No sample found\n ";
        }

        int index = std::distance(sample_list_inmemory_.begin(), it);
        auto start = index;
        auto query_start = 0;
        for (int i = 0; i < num_batches; ++i) {
            TBlob<PrecisionTrait<Precision::FP32>::value_type>::Ptr input;

            Blob::Ptr input_ids = make_shared_blob< int32_t > (desc, input_ids_inmemory_[start]->buffer(), (bs * max_seq_length_));
            Blob::Ptr input_mask = make_shared_blob< int32_t > (desc, input_mask_inmemory_[start]->buffer(), (bs * max_seq_length_));
            Blob::Ptr segment_ids = make_shared_blob< int32_t > (desc, segment_ids_inmemory_[start]->buffer(), (bs * max_seq_length_));


            std::vector< Blob::Ptr > inputs {input_ids, input_mask, segment_ids};


            std::vector < mlperf::QuerySampleIndex > idxs;
            std::vector < mlperf::ResponseId > ids;
            for (size_t j = (query_start); j < (unsigned) (query_start + bs);
                    ++j) {
                ids.push_back(queryIds[j]);
                idxs.push_back(samples[j]);
            }

            items.push_back(Item(inputs, ids, idxs));

            start = start + bs;
            query_start = query_start + bs;
        }
    }


	~Squad(){};

private:
    size_t max_seq_length_ = 384;
    int max_query_length_ = 64;
    int doc_stride_ = 128;
    std::vector<int> sample_input_ids;
    std::vector<int> sample_input_mask;
    std::vector<int> sample_segment_ids;
    std::vector<string> sample_tokens;

};

#endif
