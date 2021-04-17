#pragma once

#include <queue>
#include <condition_variable>
#include <mutex>

#include <inference_engine.hpp>
#include <ie_blob.h>

// loadgen
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"

#include "item_ov.h"

extern std::unique_ptr<Dataset> ds;

/// Post processor function type
typedef std::function<void(Item qitem, InferenceEngine::InferRequest req, std::vector<float> &results, std::vector<mlperf::ResponseId>&response_ids, unsigned batch_size, std::vector<unsigned> &counts )> PPFunction;

typedef std::function<
        void(size_t id, InferenceEngine::InferRequest req, Item input, std::vector<float> &results, std::vector< mlperf::ResponseId > &response_ids, std::vector<unsigned> &counts) 
	> QueueCallbackFunction;

typedef std::function<
        void(size_t id, InferenceEngine::InferRequest req, Item input, std::vector<float> &results, std::vector< mlperf::ResponseId > &response_ids, std::vector<unsigned> &counts, std::vector< mlperf::QuerySampleResponse > &respns)
        > ServerQueueCallbackFunction;

/// @brief Wrapper class for InferenceEngine::InferRequest. Handles asynchronous callbacks .
class InferReqWrap final {
public:
    using Ptr = std::shared_ptr<InferReqWrap>;

    ~InferReqWrap() = default;

    InferReqWrap(InferenceEngine::ExecutableNetwork& net, size_t id, std::vector<std::string> input_blob_names, std::vector<std::string> output_blob_names, mlperf::TestSettings settings,
            std::string workload, QueueCallbackFunction callback_queue) :
            request_(net.CreateInferRequest()), id_(id), input_blob_names_(input_blob_names), output_blob_names_(output_blob_names), settings_(settings), workload_(workload), callback_queue_(
                    callback_queue) {

           request_.SetCompletionCallback([&]() {
                
                callback_queue_(id_, request_, input_, results_, response_ids_, counts_);

        });
    }

    InferReqWrap(InferenceEngine::ExecutableNetwork& net, size_t id, std::vector<std::string> input_blob_names, std::vector<std::string> output_blob_names, mlperf::TestSettings settings,
            std::string workload, ServerQueueCallbackFunction callback_queue, bool server) :
            request_(net.CreateInferRequest()), id_(id), input_blob_names_(input_blob_names), output_blob_names_(output_blob_names), settings_(settings), workload_(workload), callback_queue_server_(
                    callback_queue) {

           request_.SetCompletionCallback([&]() {

                callback_queue_server_(id_, request_, input_, results_, response_ids_, counts_, respns_);

        });
    }
    

    void inferReqCallback(){
	 getOutputBlobs();
	 callback_queue_(id_, request_, outputs_, results_, response_ids_, counts_);
    }

    void startAsync() {
        request_.StartAsync();
    }

    void infer() {
        request_.Infer();
    }

    InferenceEngine::Blob::Ptr getBlob(const std::string &name) {
        return request_.GetBlob(name);
    }

    Item getOutputBlobs() {
            Item outputs;

	    for (size_t i = 0; i < output_blob_names_.size(); ++i){
                outputs.blobs_.push_back( request_.GetBlob(output_blob_names_[i]) );
	    }
	    return outputs;
    }


    void setInputs(Item input, std::string name) {
        input_ = input;
        request_.SetBlob(name, input_.blob_);
    }

    void setInputs(Item input) {
        input_ = input;
        for (size_t i= 0; i < input_blob_names_.size(); ++i){
            request_.SetBlob(input_blob_names_[i], input_.blobs_[i]);
        }
    }

    unsigned getBatchSize(){
        return input_.sample_idxs_.size();
    }

    void setIsWarmup(bool warmup) {
        is_warm_up = warmup;
    }

    size_t getRequestID(){
	 return id_;
    }

    void reset(){
	response_ids_.clear();
	counts_.clear();
	results_.clear();
	respns_.clear();
	true_pos_ = 0;

    }

public:
    std::vector < mlperf::ResponseId > response_ids_;
    std::vector<unsigned> counts_;
    std::vector<float> results_;
    std::vector< mlperf::QuerySampleResponse > respns_;
    size_t true_pos_ = 0;
    
private:
    InferenceEngine::InferRequest request_;
    size_t id_;
    std::vector<std::string> input_blob_names_, output_blob_names_;
    mlperf::TestSettings settings_;
    std::string workload_;
    QueueCallbackFunction callback_queue_;
    ServerQueueCallbackFunction callback_queue_server_;

    Item input_;
    Item outputs_;
    bool is_warm_up = false;
    
};

using namespace InferenceEngine;

class InferRequestsQueue final {
public:

    InferRequestsQueue(InferenceEngine::ExecutableNetwork& net, size_t nireq,
            std::vector<std::string> input_blob_names, std::vector<std::string> output_blob_names, mlperf::TestSettings settings,
            std::string workload, unsigned batch_size, PPFunction post_processor) {
        for (size_t id = 0; id < nireq; id++) {
            requests.push_back(
                    std::make_shared < InferReqWrap
                            > (net, id, input_blob_names, output_blob_names, settings, workload, std::bind(
                                    &InferRequestsQueue::putIdleRequest, this,
                                    std::placeholders::_1,
                                    std::placeholders::_2,
				                    std::placeholders::_3,
				                    std::placeholders::_4,
				                    std::placeholders::_5,
				                    std::placeholders::_6)));
            idle_ids_.push(id);
        }

        settings_ = settings;
        workload_ = workload;
        batch_size_ = batch_size;
        post_processor_ = post_processor;

    }

    ~InferRequestsQueue() = default;


    void putIdleRequest(size_t id, InferenceEngine::InferRequest req, Item qitem, std::vector<float> &results, std::vector< mlperf::ResponseId > &response_ids, std::vector<unsigned> &counts ){

        

        post_processor_(qitem, req, results, response_ids, batch_size_, counts);
        
    	std::unique_lock < std::mutex > lock(mutex_);
        idle_ids_.push(id);
        cv_.notify_one();

    }


    InferReqWrap::Ptr getIdleRequest() {
        std::unique_lock < std::mutex > lock(mutex_);
        cv_.wait(lock, [this] {return idle_ids_.size() > 0;});
        auto request = requests.at(idle_ids_.front());
        idle_ids_.pop();
        return request;
    }

    void waitAll() {
        std::unique_lock < std::mutex > lock(mutex_);
        cv_.wait(lock, [this] {return idle_ids_.size() == requests.size();});

	    size_t j = 0;
	    for (auto &req : requests ){
                size_t idx = 0;
                for (size_t i = 0; i < req->counts_.size(); ++i) {
                    mlperf::QuerySampleResponse response { req->response_ids_[i],
                        reinterpret_cast<std::uintptr_t>(&(req->results_[idx])),
                        (sizeof(float) * req->counts_[i]) };
                    responses_.push_back(response);
                    idx = idx + req->counts_[i];
                }
        	    
	    }
    }


    std::vector<Item> getOutputs() {
        return outputs_;
    }

    std::vector < mlperf::QuerySampleResponse > getQuerySampleResponses(){
        return responses_;
    }

    void reset() {
        outputs_.clear();
	responses_.clear();
	true_positives_ = 0;
	for (auto &req : requests){
	    req->reset();
	}
    }


    std::vector<InferReqWrap::Ptr> requests;
    size_t true_positives_ = 0;

private:
    std::queue<size_t> idle_ids_;
    std::mutex mutex_;
    std::condition_variable cv_;

    mlperf::TestSettings settings_;
    std::string out_name_;
    std::string workload_;
    std::vector<Item> outputs_;
    std::vector < mlperf::QuerySampleResponse > responses_;
    unsigned batch_size_, num_batches_;
    PPFunction post_processor_;
};



/**================================== Server Queue Runner ================================**/

class InferRequestsQueueServer {
public:

        InferRequestsQueueServer(InferenceEngine::ExecutableNetwork& net, size_t nireq,
            std::vector<std::string> input_blob_names, std::vector<std::string> output_blob_names, mlperf::TestSettings settings,
            std::string workload, unsigned batch_size, PPFunction post_processor) {
	
	for (size_t id = 0; id < nireq; id++) {
            requests.push_back(
                    std::make_shared < InferReqWrap
                            > (net, id, input_blob_names, output_blob_names, settings, workload, std::bind(
                                    &InferRequestsQueueServer::putIdleRequest, this,
                                    std::placeholders::_1,
                                    std::placeholders::_2,
                                    std::placeholders::_3,
                                    std::placeholders::_4,
                                    std::placeholders::_5,
                                    std::placeholders::_6,
				                    std::placeholders::_7),
				    true));
            idle_ids_.push(id);
        }
	

        settings_ = settings;
        workload_ = workload;
        batch_size_ = batch_size;
        post_processor_ = post_processor;
        setWarmup(false);

    }
    
    ~InferRequestsQueueServer() = default;

    
    void putIdleRequest(size_t id, InferenceEngine::InferRequest req, Item qitem, std::vector<float> &results, std::vector< mlperf::ResponseId > &response_ids, std::vector<unsigned> &counts, std::vector<mlperf::QuerySampleResponse > &respns){

        post_processor_(qitem, req, results, response_ids, batch_size_, counts);
	
        size_t idx = 0;
	
        for (size_t i = 0; i < counts.size(); ++i) {
            mlperf::QuerySampleResponse response { response_ids[i],
                reinterpret_cast<std::uintptr_t>(&(results[idx])),
                (sizeof(float) * counts[i]) };
            respns.push_back(response);
            idx = idx + counts[i];
        }
	
	if (!(isWarmup_)){
            mlperf::QuerySamplesComplete( respns.data(), respns.size() );
	} 


	results.clear();
	counts.clear();
	response_ids.clear();
	respns.clear();

	
        std::unique_lock < std::mutex > lock(mutex_);
        idle_ids_.push(id);
        cv_.notify_one();
	

    }
    


    void setWarmup(bool warmup){
	isWarmup_ = warmup;
    }

    // Maybe different post-completion tasks for warmup
    void warmupPutIdleReq(size_t id){
	std::unique_lock <std::mutex> lock(mutex_);
	idle_ids_.push(id);
	cv_.notify_one();
    }


    InferReqWrap::Ptr getIdleRequest() {
        std::unique_lock < std::mutex > lock(mutex_);
        cv_.wait(lock, [this] {return idle_ids_.size() > 0;});
        auto request = requests.at(idle_ids_.front());
        idle_ids_.pop();
        return request;
    }

    void waitAll() {

        std::unique_lock < std::mutex > lock(mutex_);
        cv_.wait(lock, [this] {return idle_ids_.size() == requests.size();});

    }


    std::vector<Item> getOutputs() {
        return outputs_;
    }

    std::vector < mlperf::QuerySampleResponse > getQuerySampleResponses(){
        return responses_;
    }

    void reset() {
        outputs_.clear();
        responses_.clear();
        true_positives_ = 0;
        for (auto &req : requests){
            req->reset();
        }
    }


    std::vector<InferReqWrap::Ptr> requests;
    size_t true_positives_ = 0;

private:
    std::queue<size_t> idle_ids_;
    std::mutex mutex_;
    std::condition_variable cv_;

    mlperf::TestSettings settings_;
    std::string out_name_;
    std::string workload_;
    std::vector<Item> outputs_;
    std::vector < mlperf::QuerySampleResponse > responses_;
    unsigned batch_size_, num_batches_;
    PPFunction post_processor_;
    bool isWarmup_;
};

