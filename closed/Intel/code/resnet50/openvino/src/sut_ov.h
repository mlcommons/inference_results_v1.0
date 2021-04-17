#ifndef SUT_H__
#define SUT_H__

#include <list>
#include <condition_variable>
#include <mutex>

// loadgen
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"

#include "backend_ov.h"
#include "item_ov.h"

extern std::unique_ptr<Dataset> ds;
using namespace InferenceEngine;

class SUTBase {

public:
	SUTBase(){};

    SUTBase(mlperf::TestSettings settings, unsigned nireq, std::string nstreams,
        int nthreads, int batch_size,
        std::string dataset, std::string workload, std::string device, std::string cpu_extension, std::vector<std::string> input_blob_names, std::vector<std::string> output_blob_names, std::string input_model, PPFunction post_processor, uint seq, uint seq_step, bool enforcebf16) :
        settings_(settings), batch_size_(batch_size), nstreams_(nstreams), nireq_(nireq), workload_(workload),
        backend_ov_(std::unique_ptr < OVBackendBase	>(new OVBackendBase(settings, batch_size, nireq, nstreams, nthreads, device, cpu_extension, workload, input_blob_names, output_blob_names, input_model, post_processor, seq, seq_step, enforcebf16))) {

        backend_ov_->load();
        comm_count_ = 1;
    }

        SUTBase(mlperf::TestSettings settings, unsigned nireq, std::string nstreams, int nthreads, int batch_size, std::string dataset, std::string workload, std::string device, std::string cpu_extension, std::vector<std::string> input_blob_names, std::vector<std::string> output_blob_names, std::string input_model, PPFunction post_processor, bool async, uint seq = 0, uint seq_step = 0, bool enforcebf16 = false) :
                settings_(settings), batch_size_(batch_size), nstreams_(nstreams), nireq_(nireq), workload_(workload), backend_ov_async_(std::unique_ptr < OVBackendAsync >(new OVBackendAsync(settings, batch_size, nireq, nstreams, nthreads, device, cpu_extension, workload, input_blob_names, output_blob_names, input_model, post_processor, seq, seq_step, enforcebf16))) {
    		backend_ov_async_->load();
        }
	

	virtual void warmUp(size_t nwarmup_iters) {
		std::vector < mlperf::QuerySampleIndex > samples;

		std::vector < mlperf::ResponseId > response_ids;
		for (size_t i = 0; i < batch_size_; ++i) {
			samples.push_back(0);
			response_ids.push_back(1);
		}

		ds->loadQuerySamples(samples.data(), batch_size_);
        std::vector<float> results;
        std::vector<unsigned> counts;

		for (size_t i = 0; i < nwarmup_iters; ++i) {
            ds->getSample(samples, response_ids, 1, 1, &qitem_);
			backend_ov_->warmUp(qitem_);
		}
		ds->unloadQuerySamples(samples.data(),0);
	}

	void processLatencies(const int64_t* latencies, size_t size) {
		return;
	}

	void flushQueries(void) {
		return;
	}

	void runOneItem(std::vector<mlperf::ResponseId> response_id) {
		std::vector<float> result;
		std::vector<unsigned> counts;

		std::vector< mlperf::QuerySampleResponse > responses;
		backend_ov_->predict(qitem_, result, response_id, counts );
		
		mlperf::QuerySampleResponse response{ response_id[0], reinterpret_cast<std::uintptr_t>(&result[0]), sizeof(float)*counts[0] };
		responses.push_back(response);

		mlperf::QuerySamplesComplete(responses.data(), responses.size());
	}

	void enqueue(const mlperf::QuerySample* samples, size_t size) {
            std::vector < mlperf::QuerySampleResponse > responses;
            std::vector < mlperf::QuerySampleIndex > sample_idxs{ (samples[0].index)};
            std::vector < mlperf::ResponseId > response_ids{ (samples[0].id)};

            int num_batches = size / batch_size_;
            ds->getSample(sample_idxs, response_ids, 1, num_batches, &qitem_);


            runOneItem(response_ids);
	}

	void issueQueries(const mlperf::QuerySample* samples, size_t size, bool accuracy) {
		enqueue(samples, size);
	}
public:
	mlperf::TestSettings settings_;
	int batch_size_ = 1;
	std::string nstreams_ = "1";
	int nthreads_;
	unsigned nireq_ = 1;
	std::string workload_;
	std::unique_ptr<OVBackendBase> backend_ov_;
	std::unique_ptr<OVBackendAsync> backend_ov_async_;
        std::unique_ptr<OVBackendServer> backend_ov_server_;
	Item qitem_;
	std::vector<Item> qitems_;
	int comm_count_;
};


class SUTOffline : public SUTBase {
public:
	SUTOffline(mlperf::TestSettings settings, unsigned nireq, std::string nstreams,
                int nthreads, int batch_size, std::string dataset, std::string workload, std::string device, std::string cpu_extension, std::vector<std::string> input_blob_names, std::vector<std::string> output_blob_names, std::string input_model, PPFunction post_processor
                , uint seq = 0, uint seq_step = 0, bool enforcebf16 = false) : 
       SUTBase(settings, nireq, nstreams, nthreads, batch_size, dataset, workload, device, cpu_extension, input_blob_names, output_blob_names, input_model, post_processor, true, seq, seq_step, enforcebf16) {

	}


	void warmupOffline(size_t nwarmup_iters){
	        std::vector < mlperf::QuerySampleIndex > samples;
        	std::vector < mlperf::ResponseId > queryIds;

	        for (size_t i = 0; i < batch_size_*nireq_; ++i) {
        	    samples.push_back(0);
	            queryIds.push_back(1);
        	}

		ds->loadQuerySamples(samples.data(), nireq_ * batch_size_);

		for (size_t i = 0; i < nwarmup_iters; ++i){

	                std::vector < mlperf::ResponseId > results_ids;
                	std::vector < mlperf::QuerySampleIndex > sample_idxs;

        	        std::vector<Item> items;
	                ds->getSamplesBatched(samples, queryIds, batch_size_, nireq_, items);
                    backend_ov_async_->predictAsync(items);

		}
        ds->unloadQuerySamples(samples.data(),0);

		backend_ov_async_->reset();
	}

        void runOneItem(std::vector<mlperf::ResponseId> response_id) {

                backend_ov_async_->predictAsync(qitems_ );

                std::vector < mlperf::QuerySampleResponse > responses = backend_ov_async_->getQuerySampleResponses();

                mlperf::QuerySamplesComplete(responses.data(), responses.size());
		backend_ov_async_->reset();
		qitems_.clear();
        }

        void enqueue(const mlperf::QuerySample* samples, size_t size) {
            std::vector < mlperf::QuerySampleResponse > responses;
            std::vector < mlperf::QuerySampleIndex > sample_idxs;
            std::vector < mlperf::ResponseId > response_ids;
            
	    for (size_t i = 0; i < size; ++i) {
                auto sample = (*samples);
                samples++;
                sample_idxs.push_back(sample.index);
                response_ids.push_back(sample.id);
            }

            int num_batches = size / batch_size_;
            ds->getSamplesBatched(sample_idxs, response_ids, batch_size_, num_batches, qitems_);


            runOneItem(response_ids);
        }

        void issueQueriesOffline(const mlperf::QuerySample* samples, size_t size, bool accuracy) {
                enqueue(samples, size);
        }


};

/** ============================= MultiStream hooks =====================*/

class SUTMultistream : public SUTBase {
public:
	SUTMultistream(mlperf::TestSettings settings, unsigned nireq, std::string nstreams,
                int nthreads, int batch_size, std::string dataset, std::string workload, std::string device, std::string cpu_extension, std::vector<std::string> input_blob_names, std::vector<std::string> output_blob_names, std::string input_model, PPFunction post_processor) : 
       SUTBase(settings, nireq, nstreams, nthreads, batch_size, dataset, workload, device, cpu_extension, input_blob_names, output_blob_names, input_model, post_processor, true) {

	}


	void warmupMultistream(size_t nwarmup_iters){
	        std::vector < mlperf::QuerySampleIndex > samples;
        	std::vector < mlperf::ResponseId > queryIds;

	        for (size_t i = 0; i < batch_size_*nireq_; ++i) {
        	    samples.push_back(0);
	            queryIds.push_back(1);
        	}

		ds->loadQuerySamples(samples.data(), nireq_);
		std::cout << " == Starting Warmup ==\n";
		for (size_t i = 0; i < nwarmup_iters; ++i){

	                std::vector < mlperf::ResponseId > results_ids;
                	std::vector < mlperf::QuerySampleIndex > sample_idxs;

        	        std::vector<Item> items;
	                ds->getSamplesBatchedMultiStream(samples, queryIds, batch_size_, nireq_, items);
                        backend_ov_async_->predictAsync(items);

		}
                ds->unloadQuerySamples(samples.data(),0);
		backend_ov_async_->reset();
		std::cout << " == Warmup Completed ==\n";
	}

        void runOneItem(std::vector<mlperf::ResponseId> response_id) {

                backend_ov_async_->predictAsync(qitems_ );

                std::vector < mlperf::QuerySampleResponse > responses = backend_ov_async_->getQuerySampleResponses();

                mlperf::QuerySamplesComplete(responses.data(), responses.size());
		backend_ov_async_->reset();
		qitems_.clear();
        }

        void enqueue(const mlperf::QuerySample* samples, size_t size) {
            std::vector < mlperf::QuerySampleResponse > responses;
            std::vector < mlperf::QuerySampleIndex > sample_idxs;
            std::vector < mlperf::ResponseId > response_ids;
            
	    for (size_t i = 0; i < size; ++i) {
                auto sample = (*samples);
                samples++;
                sample_idxs.push_back(sample.index);
                response_ids.push_back(sample.id);
            }

            int num_batches = size / batch_size_;
            ds->getSamplesBatchedMultiStream(sample_idxs, response_ids, batch_size_, num_batches, qitems_);
            runOneItem(response_ids);
        }

        void issueQueriesMultistream(const mlperf::QuerySample* samples, size_t size, bool accuracy) {
                enqueue(samples, size);
        }

};

/** ============================= Server hooks =====================*/

class SUTServer : public SUTBase {
public:
        SUTServer(mlperf::TestSettings settings, unsigned nireq, std::string nstreams,
                int nthreads, int batch_size, std::string dataset, std::string workload, std::string device, std::string cpu_extension, std::vector<std::string> input_blob_names, std::vector<std::string> output_blob_names, std::string input_model, PPFunction post_processor, uint seq = 0, uint seq_step = 0, bool enforcebf16 = false) :
       SUTBase(settings, nireq, nstreams, nthreads, batch_size, dataset, workload, device, cpu_extension, input_blob_names, output_blob_names, input_model, post_processor, true, seq, seq_step, enforcebf16) {

        }

        void warmupServer(size_t nwarmup_iters){
		backend_ov_async_->setServerWarmup(true);
               

                std::vector < mlperf::QuerySampleIndex > samples;
                std::vector < mlperf::ResponseId > response_ids;
		
                for (size_t i = 0; i < batch_size_; ++i) {
                    samples.push_back(0);
                    response_ids.push_back(1);
                }

                ds->loadQuerySamples(samples.data(), batch_size_);

		std::vector<Blob::Ptr> data;
		Item item;

                ds->getSample(samples, response_ids, 1, 1, &item);
		
                for (size_t i = 0; i < nwarmup_iters; ++i){

                        backend_ov_async_->predictAsyncServer(item);
                }

                ds->unloadQuerySamples(samples.data(),0);
                backend_ov_async_->reset();
        }

	void runOneItem(std::vector<mlperf::ResponseId> response_id, Item item) {
                backend_ov_async_->predictAsyncServer( item );
        }

        void enqueue(const mlperf::QuerySample* samples, size_t size) {
            std::vector < mlperf::QuerySampleResponse > responses;
            std::vector < mlperf::QuerySampleIndex > sample_idxs;
            std::vector < mlperf::ResponseId > response_ids;
	     
	    for (size_t i = 0; i < size; ++i) {
                auto sample = (*samples);
                samples++;
                sample_idxs.push_back(sample.index);
                response_ids.push_back(sample.id);
	    }

	    Item item;
            ds->getSample(sample_idxs, response_ids, batch_size_, 1, &item);
            runOneItem(response_ids, item);
        }

        void issueQueriesServer(const mlperf::QuerySample* samples, size_t size, bool accuracy) {
                enqueue(samples, size);
        }
private:
	int qid = 0;
	std::mutex mutex_;
	std::condition_variable cv_;
	size_t online_bs_;
};

#endif
