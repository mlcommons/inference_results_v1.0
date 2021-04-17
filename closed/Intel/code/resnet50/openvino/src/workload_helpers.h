#ifndef OV_SETTINGS_H_
#define OV_SETTINGS_H_

#include<string>
#include<vector>
#include<map>

namespace OV {
	enum class WorkloadName {
		ResNet50,
		MobileNet,
		SSDMobilenet,
		SSDResNet34,
		Bert,
		Default
	};


	class WorkloadBase {
	public:
                std::string getDataset() { return dataset_; };
                std::vector<std::string> getInputBlobs() { return input_blob_names_; };
                std::vector<std::string> getOutputBlobs() { return output_blob_names_; };
	
		std::vector<std::string> input_blob_names_, output_blob_names_;
		std::string dataset_;

	};

	class ResNet50 : public WorkloadBase {
	public:
		ResNet50(){
			this->input_blob_names_.push_back("input_tensor");
			this->output_blob_names_.push_back("softmax_tensor");//"ArgMax/Squeeze");
			this->dataset_ = "imagenet";
		}

		~ResNet50(){};
		void postProcess(){};
	};

	class MobileNet : public WorkloadBase {
        public:
		MobileNet(){
			this->input_blob_names_.push_back("input");
			this->output_blob_names_.push_back("Mobilenetv1/Predictions/Softmax");
			this->dataset_ = "imagenet";
		}

		~MobileNet(){};

        void postProcess(){};
	};

	class MobileNetEdge : public WorkloadBase {
	public:
		MobileNetEdge() {
			this->input_blob_names_.push_back("images");
			this->output_blob_names_.push_back("Softmax");
			this->dataset_ = "imagenet";
		}

		~MobileNetEdge() {};
		void postProcess() {};
	};
    class SSDMobileNet : public WorkloadBase {
    public:
            SSDMobileNet(){
                    this->input_blob_names_.push_back("image_tensor");
                    this->output_blob_names_.push_back("DetectionOutput");
                    this->dataset_ = "coco";
            }

            ~SSDMobileNet(){};

            void postProcess(){};
    };

	class SSDResNet : public WorkloadBase {
	public:
		SSDResNet() {
			this->input_blob_names_.push_back("image");

			this->output_blob_names_.push_back("Unsqueeze_bboxes/Unsqueeze");
			this->output_blob_names_.push_back("Unsqueeze_scores/Unsqueeze");
			this->output_blob_names_.push_back("Add_labels");
			this->dataset_ = "coco";
		}

		~SSDResNet() {};

		void postProcess() {};
	};

	class DeepLabv3 : public WorkloadBase {
	public:
		DeepLabv3() {
			this->input_blob_names_.push_back("ImageTensor");

			this->output_blob_names_.push_back("ArgMax/Squeeze");
			this->dataset_ = "ade20k";
		}

		~DeepLabv3() {};

		void postProcess() {};
	};

	class Bert : public WorkloadBase {
    public:
            Bert(){
                    this->input_blob_names_.push_back("input_ids");
                    this->input_blob_names_.push_back("attention_mask");
	                	this->input_blob_names_.push_back("token_type_ids");
	
                    this->output_blob_names_.push_back("start_logits");
					this->output_blob_names_.push_back("end_logits");
                    this->dataset_ = "squad";
            }

            ~Bert(){}; 

            void postProcess(){};
    };

	class MobileBert : public WorkloadBase {
    public:
            MobileBert(){
                    this->input_blob_names_.push_back("result.1");
                    this->input_blob_names_.push_back("result.2");
	                	this->input_blob_names_.push_back("result.3");		

                    this->output_blob_names_.push_back("Squeeze_7015");
					this->output_blob_names_.push_back("Squeeze_7016");
                    this->dataset_ = "squad";
            }

            ~MobileBert(){}; 

            void postProcess(){};
    };
}; // namespace OV

#endif
