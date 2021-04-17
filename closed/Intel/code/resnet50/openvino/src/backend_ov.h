#ifndef BACKENDOV_H__
#define BACKENDOV_H__

#include <inference_engine.hpp>
#include<vector>

#include <vpu/vpu_plugin_config.hpp>
#include <cldnn/cldnn_config.hpp>

#include "utils.h"
#include "infer_request_wrap.h"
#include "item_ov.h"

using namespace InferenceEngine;

class OVBackendBase {

public:
	OVBackendBase(mlperf::TestSettings settings, unsigned batch_size, unsigned nireq, std::string nstreams,
		int nthreads, std::string device, std::string cpu_extension, std::string workload, std::vector<std::string> input_blob_names, std::vector<std::string> output_blob_names, std::string input_model, PPFunction post_processor, uint seq, uint seq_step, bool enforcebf16) :
		settings_(settings), batch_size_(batch_size), nireq_(nireq), nstreams_(nstreams), nthreads_(nthreads), device_(device), cpu_extension_(cpu_extension), workload_(workload), input_blob_names_(input_blob_names), output_blob_names_(output_blob_names), input_model_(input_model), post_processor_(post_processor), 
                seq_(seq),  seq_step_(seq_step), enforcebf16_(enforcebf16)
	{

	}

	~OVBackendBase() {

	}

	std::string version() {
		return "";
	}

	std::string name() {
		return "openvino";
	}

	std::string image_format() {
		// Find right format
		return "NCHW";
	}

	std::string fileNameNoExt(const std::string &filepath) {
		auto pos = filepath.rfind('.');
		if (pos == std::string::npos)
			return filepath;
		return filepath.substr(0, pos);
	}

	void setBatchSize() {
		InputsDataMap inputInfo(network_.getInputsInfo());
		InferenceEngine::ICNNNetwork::InputShapes shapes = network_.getInputShapes();

		bool reshape = false;
		reshape |= adjustShapesBatch(shapes, batch_size_, inputInfo);

		if (reshape) network_.reshape(shapes);
		//std::cout << "Reshaping network: " << getShapesString(shapes) << std::endl;
		//network_.setBatchSize(batch_size_);

	}

	void setOutputInfo() {
		OutputsDataMap outputInfo( network_.getOutputsInfo());
		for (auto& item : outputInfo){
			item.second->setPrecision( Precision::FP32 );
		}
	}

	void setInputInfo( Precision precision, Layout layout){
		
		InputsDataMap inputInfo(network_.getInputsInfo());
		for (auto& item : inputInfo){
			item.second->setPrecision( precision );
			//item.second->setLayout( layout );
		}
	}

	void setInferRequest() {
		inferRequest_ = exe_network_.CreateInferRequest();
	}

	void createRequests() {
		inferRequestsQueue_ = new InferRequestsQueue(exe_network_, nireq_, input_blob_names_, output_blob_names_, settings_, workload_, batch_size_, post_processor_);
	}


	void createServerRequests() {
		inferRequestsQueueServer_ = new InferRequestsQueueServer(exe_network_, nireq_, input_blob_names_, output_blob_names_, settings_, workload_, batch_size_, post_processor_);
	}

	void warmUp(Item input){
		for (size_t j = 0; j < input_blob_names_.size(); j++){
			inferRequest_.SetBlob(input_blob_names_[ j ], input.blobs_[ j ]);
		}
		inferRequest_.Infer();
		std::vector<float> results;
		std::vector<mlperf::ResponseId> response_ids;
		std::vector<unsigned> counts;
		post_processor_(input, inferRequest_, results, response_ids, 1, counts);
	}

	void reset() {
		if (settings_.scenario == mlperf::TestScenario::Offline || 
				settings_.scenario == mlperf::TestScenario::MultiStream){
			inferRequestsQueue_->reset();
		} else if (settings_.scenario == mlperf::TestScenario::Server){
			inferRequestsQueueServer_->waitAll();
			inferRequestsQueueServer_->reset();
			inferRequestsQueueServer_->setWarmup(false);
		}
	}

	void setServerWarmup(bool warmup){
		inferRequestsQueueServer_->setWarmup(warmup);
	}


	size_t getTruePositives(){
		return inferRequestsQueue_->true_positives_;
	}

	std::vector < mlperf::QuerySampleResponse > getQuerySampleResponses(){
		return inferRequestsQueue_->getQuerySampleResponses();
	}

	// Progress tracker method
	void progress(size_t iter, size_t num_batches, size_t progress_bar_length) {

		float p_bar_frac = (float)iter / (float)num_batches;
		int p_length = static_cast<int>(progress_bar_length * p_bar_frac);
		std::string progress_str(p_length, '.');
		std::string remainder_str(progress_bar_length - p_length, ' ');
		progress_str = "    [BENCHMARK PROGRESS] [" + progress_str + remainder_str + "] " + std::to_string((int)(p_bar_frac * 100)) + "%";
		std::cout << progress_str << "\r" << std::flush;
	}

	void predictSingleStream(std::vector<Blob::Ptr> inputs, mlperf::QuerySample *sample, std::vector< mlperf::QuerySampleResponse> &responses){

		for (size_t j = 0; j < input_blob_names_.size(); j++){
			inferRequest_.SetBlob(input_blob_names_[ j ], inputs[ j ]);
		}
		inferRequest_.Infer();

		std::vector<unsigned> result;
		Blob::Ptr output_blob = inferRequest_.GetBlob(output_blob_names_[0]);
		TopResults(1, *output_blob, result);
		mlperf::QuerySampleResponse response{ sample[0].id, reinterpret_cast<std::uintptr_t>(&result[0]), sizeof(float) };
		responses.push_back(response);
	}

	void predict( Item input, std::vector<float> &results, std::vector<mlperf::ResponseId> &response_ids, std::vector<unsigned> &counts) {
		for (size_t j = 0; j < input_blob_names_.size(); j++){
                        inferRequest_.SetBlob(input_blob_names_[ j ], input.blobs_[ j ]);
                }

		inferRequest_.Infer();
		post_processor_(input, inferRequest_, results, response_ids, 1, counts);
	}

	void predictAsync(std::vector<Item> input_items) {

		for (size_t j = 0; j < input_items.size(); ++j){
			auto inferRequest = inferRequestsQueue_->getIdleRequest();
			inferRequest->setInputs(input_items[j]);//, this->input_blob_names_[0]);
			inferRequest->startAsync();
		}

		inferRequestsQueue_->waitAll();
	}

	void predictAsyncServer(Item input_item) {

		auto inferRequest = inferRequestsQueueServer_->getIdleRequest();
		
		inferRequest->setInputs(input_item); //, this->input_blob_names_[0]);
		inferRequest->startAsync();
		return;
	}

	void load() {

		// std::cout << " == Configuring network for device ==\n";

		Core ie_;

		// Load model to device
		auto devices = parseDevices(this->device_);
		//std::map<std::string, uint32_t> device_nstreams = parseValuePerDevice(devices, this->nstreams_);
		
		std::map<std::string, uint32_t> device_nstreams = parseNStreamsValuePerDevice(devices, this->nstreams_);
		for (auto& pair : device_nstreams) {
			auto key = std::string(pair.first + "_THROUGHPUT_STREAMS");
			std::vector<std::string> supported_config_keys = ie_.GetMetric(pair.first, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
			if (std::find(supported_config_keys.begin(), supported_config_keys.end(), key) == supported_config_keys.end()) {
				throw std::logic_error("Device " + pair.first + " doesn't support config key '" + key + "'! " +
					"Please specify -nstreams for correct devices in format  <dev1>:<nstreams1>,<dev2>:<nstreams2>");
			}
		}

		for (auto& device : devices) {
			if (device == "CPU") {
			    
				if (enforcebf16_){
					ie_.SetConfig( { {CONFIG_KEY(ENFORCE_BF16), CONFIG_VALUE(YES) } }, device );
				}

				if (!cpu_extension_.empty()) {
					const auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(this->cpu_extension_);
					ie_.AddExtension(extension_ptr, "CPU");
				}

				if (settings_.scenario == mlperf::TestScenario::SingleStream) {
					ie_.SetConfig(	{ { CONFIG_KEY(CPU_THREADS_NUM), std::to_string(this->nthreads_) } }, device);
				}

				if (settings_.scenario != mlperf::TestScenario::SingleStream) {
					ie_.SetConfig({ { CONFIG_KEY(CPU_THREADS_NUM), std::to_string(	this->nthreads_) } }, device);
					ie_.SetConfig({	{ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(YES) } },	device);
					//ie_.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS), this->nstreams_ } }, device);
					ie_.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),	(device_nstreams.count(device) > 0 ? std::to_string(device_nstreams.at(device)) : "CPU_THROUGHPUT_AUTO") } }, device);
					if ((device_.find("MULTI") != std::string::npos) &&
						((device_.find("GPU") != std::string::npos) || (device_.find("HDDL") != std::string::npos))) {
						ie_.SetConfig({ { CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) } }, device);

					}
				}
			}

			if ((device == "MYRIAD") || (device == "HDDL")) {
				ie_.SetConfig({ { CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE) },
							  { VPU_CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING) } }, device);
			}

			if (device == "GPU") {
				if (settings_.scenario != mlperf::TestScenario::SingleStream) {
					//ie_.SetConfig({ { CONFIG_KEY(GPU_THROUGHPUT_STREAMS), this->nstreams_ } }, device);
					ie_.SetConfig({ { CONFIG_KEY(GPU_THROUGHPUT_STREAMS),(device_nstreams.count(device) > 0 ? std::to_string(device_nstreams.at(device)) : "GPU_THROUGHPUT_AUTO") } }, device);

					if ((device_.find("MULTI") != std::string::npos) &&
						(device_.find("CPU") != std::string::npos)) {
						ie_.SetConfig({ { CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" } }, "GPU");
					}
				}
			}
		}

		network_ = ie_.ReadNetwork(input_model_);


		if (workload_.find("bert") != std::string::npos){
			setInputInfo(Precision::I32, Layout::NC);
		} else {
			setInputInfo(Precision::FP32, Layout::NCHW);
		}

		setOutputInfo();

		setBatchSize();

		std::cout << "    [INFO] Loading network to device... \n";		
		exe_network_ = ie_.LoadNetwork(network_, device_);
		std::cout << "    [INFO] Network loaded to device \n";

		std::cout << "    [INFO] Creating inference request(s) \n";
		if (settings_.scenario == mlperf::TestScenario::SingleStream) {
			setInferRequest();
		} else if (settings_.scenario == mlperf::TestScenario::Offline || 
				settings_.scenario == mlperf::TestScenario::MultiStream) {
			createRequests();
		} else if (settings_.scenario == mlperf::TestScenario::Server){
			createServerRequests();
		}

	}

public:
	CNNNetwork network_;
	ExecutableNetwork exe_network_;
	InferRequestsQueue* inferRequestsQueue_;
	InferRequestsQueueServer* inferRequestsQueueServer_;
	InferRequest inferRequest_;
	std::string input_model_;
	std::vector<std::string> output_blob_names_;
	std::vector<std::string> input_blob_names_;
	std::string device_ = "CPU";
	std::string cpu_extension_ = "";
	std::string nstreams_ = "1";
	unsigned seq_ = 0;
	unsigned seq_step_ = 0;
	unsigned nireq_ = 1;
	int nthreads_ = 56;
	unsigned batch_size_ = 1;

	mlperf::TestSettings settings_;
	std::string workload_;
	int object_size_;

	PPFunction post_processor_;
	bool enforcebf16_;
};

class OVBackendAsync : public OVBackendBase {
public:
	OVBackendAsync(mlperf::TestSettings settings, unsigned batch_size, unsigned nireq, std::string nstreams, int nthreads, std::string device, std::string cpu_extension, std::string workload, std::vector<std::string> input_blob_names, std::vector<std::string> output_blob_names, std::string input_model, PPFunction post_processor, uint seq = 0, uint seq_step = 0, bool enforcebf16 = false) :	
	OVBackendBase(settings, batch_size, nireq, nstreams, nthreads, device, cpu_extension, workload, input_blob_names, output_blob_names, input_model, post_processor, seq, seq_step, enforcebf16) {
		//std::cout << " == Initializing OVBackendOffline == " << std::endl;

	}


};

class OVBackendServer : public OVBackendBase {
public:
	OVBackendServer(mlperf::TestSettings settings, unsigned batch_size, unsigned nireq, std::string nstreams, int nthreads, std::string device, std::string cpu_extension, std::string workload, std::vector<std::string> input_blob_names, std::vector<std::string> output_blob_names, std::string input_model,PPFunction post_processor, uint seq = 0, uint seq_step = 0, bool enforcebf16 = false) :
	OVBackendBase(settings, batch_size, nireq, nstreams, nthreads, device, cpu_extension, workload, input_blob_names, output_blob_names, input_model, post_processor, seq, seq_step, enforcebf16) {
		//std::cout << " == Initializing OVBackendOffline == " << std::endl;

	}


};

#endif
