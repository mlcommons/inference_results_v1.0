#include <string>
#include <functional>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <regex>

#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"
#include "dataset_ov.h"
#include "sut_ov.h"

#include "input_flags.h"
#include "workload_helpers.h"
#include "post_processors.h"

#define NANO_SEC 1e9
#define MILLI_SEC 1000
#define MILLI_TO_NANO 1000000

std::unique_ptr<Dataset> ds;
std::unique_ptr<SUTBase> sut;
std::unique_ptr<SUTOffline> sut_offline;
std::unique_ptr<SUTMultistream> sut_multistream;
std::unique_ptr<SUTServer> sut_server;
mlperf::TestSettings settings;
mlperf::LogSettings log_settings;

PPFunction post_processor;

std::unique_ptr<OV::WorkloadBase> workload;


void issueQueries(mlperf::c::ClientData client,
        const mlperf::QuerySample* samples, size_t size) {
    sut->issueQueries(samples, size,
            ((settings.mode == mlperf::TestMode::AccuracyOnly) ? 1 : 0));
}

void issueQueriesOffline(mlperf::c::ClientData client,
        const mlperf::QuerySample* samples, size_t size) {
    sut_offline->issueQueriesOffline(samples, size,
            ((settings.mode == mlperf::TestMode::AccuracyOnly) ? 1 : 0));
}

void issueQueriesMultistream(mlperf::c::ClientData client,
        const mlperf::QuerySample* samples, size_t size) {
    sut_multistream->issueQueriesMultistream(samples, size,
            ((settings.mode == mlperf::TestMode::AccuracyOnly) ? 1 : 0));
}


void issueQueriesServer(mlperf::c::ClientData client,
        const mlperf::QuerySample* samples, size_t size) {
    sut_server->issueQueriesServer(samples, size,
            ((settings.mode == mlperf::TestMode::AccuracyOnly) ? 1 : 0));
}



void processLatencies(mlperf::c::ClientData client, const int64_t* latencies,
        size_t size) {
    sut->processLatencies(latencies, size);
}

void flushQueries(void) {
    sut->flushQueries();
}

void loadQuerySamples(mlperf::c::ClientData client,
        const mlperf::QuerySampleIndex* samples, size_t num_samples) {
    ds->loadQuerySamples(samples, num_samples);
}

void unloadQuerySamples(mlperf::c::ClientData client,
        const mlperf::QuerySampleIndex* samples, size_t num_samples) {
    ds->unloadQuerySamples(samples, num_samples);
}

int main(int argc, char **argv) {

    // Parse Command flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    settings.mode = mlperf::TestMode::PerformanceOnly;
    settings.scenario = mlperf::TestScenario::SingleStream;
    int image_width = 224, image_height = 224, num_channels = 3;
    int max_seq_length = 384, max_query_length = 64, doc_stride = 128;
    std::string image_format = "NCHW", multi_device_streams = "";


    if (!(FLAGS_mlperf_conf.compare("") == 0)) {
        settings.FromConfig(FLAGS_mlperf_conf, FLAGS_model_name, FLAGS_scenario);
    }
    if (!(FLAGS_user_conf.compare("") == 0)) {
        settings.FromConfig(FLAGS_user_conf, FLAGS_model_name, FLAGS_scenario);
    }

    if (FLAGS_mode.compare("Accuracy") == 0) {
        settings.mode = mlperf::TestMode::AccuracyOnly;
    }
    if (FLAGS_mode.compare("Performance") == 0) {
        settings.mode = mlperf::TestMode::PerformanceOnly;
    }
    if (FLAGS_mode.compare("Submission") == 0) {
        settings.mode = mlperf::TestMode::SubmissionRun;
    }
    if (FLAGS_mode.compare("FindPeakPerformance") == 0) {
        settings.mode = mlperf::TestMode::FindPeakPerformance;
    }

    
    if (FLAGS_scenario.compare("SingleStream") == 0) {
        settings.scenario = mlperf::TestScenario::SingleStream;
    }
    if (FLAGS_scenario.compare("Server") == 0) {
        settings.scenario = mlperf::TestScenario::Server;
    }
    if (FLAGS_scenario.compare("Offline") == 0) {
        settings.scenario = mlperf::TestScenario::Offline;
    }
    if (FLAGS_scenario.compare("MultiStream") == 0) {
        settings.scenario = mlperf::TestScenario::MultiStream;
    }


    log_settings.enable_trace = false;
    
    if (settings.performance_sample_count_override > 0 ){
	FLAGS_perf_sample_count = settings.performance_sample_count_override;
    }

    if (FLAGS_model_name.compare("resnet50") == 0) {
	post_processor = std::bind( &Processors::postProcessResnet50, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);

	workload = std::unique_ptr<OV::ResNet50> (new OV::ResNet50());
        image_format = "NCHW";
        image_height = 224;
        image_width = 224;
        num_channels = 3;
    } else if (FLAGS_model_name.compare("mobilenet") == 0) {
        post_processor = std::bind( &Processors::postProcessMobilenet, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);

	workload = std::unique_ptr<OV::MobileNet> (new OV::MobileNet());
        image_format = "NCHW";
        image_height = 224;
        image_width = 224;
        num_channels = 3;
    } else if (FLAGS_model_name.compare("mobilenet-edge") == 0) {
        post_processor = std::bind( &Processors::postProcessMobilenetEdge, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);

        workload = std::unique_ptr<OV::MobileNetEdge> (new OV::MobileNetEdge());
        image_format = "NCHW";
        image_height = 224;
        image_width = 224;
        num_channels = 3;
    } else if (FLAGS_model_name.compare("ssd-mobilenet") == 0) {
        post_processor = std::bind( &Processors::postProcessSSDMobilenet, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);

	workload = std::unique_ptr<OV::SSDMobileNet> (new OV::SSDMobileNet());
        image_format = "NCHW";
        image_height = 300;
        image_width = 300;
        num_channels = 3;
    } else if (FLAGS_model_name.compare("ssd-mobilenet-v2") == 0) {
        post_processor = std::bind( &Processors::postProcessSSDMobilenet, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);

        workload = std::unique_ptr<OV::SSDMobileNet> (new OV::SSDMobileNet());
        image_format = "NCHW";
        image_height = 300;
        image_width = 300;
        num_channels = 3;
    } else if (FLAGS_model_name.compare("ssd-resnet34") == 0) {
        post_processor = std::bind( &Processors::postProcessSSDResnet, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
        workload = std::unique_ptr<OV::SSDResNet>(new OV::SSDResNet());

        image_format = "NCHW";
        image_height = 1200;
        image_width = 1200;
        num_channels = 3;
    }else if (FLAGS_model_name.compare("deeplabv3") == 0) {
        post_processor = std::bind(&Processors::postProcessDeepLabv3, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
        workload = std::unique_ptr<OV::DeepLabv3>(new OV::DeepLabv3());

        image_format = "NCHW";
        image_height = 512;
        image_width = 512;
        num_channels = 3;

    } else if (FLAGS_model_name.compare("bert") == 0) {
        post_processor = std::bind( &Processors::postProcessBert, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);

	workload = std::unique_ptr<OV::Bert> (new OV::Bert());
        max_seq_length = 384; 
        max_query_length = 64;
        doc_stride = 128;
    } else if (FLAGS_model_name.compare("mobilebert") == 0) {
        post_processor = std::bind( &Processors::postProcessMobileBert, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);

        workload = std::unique_ptr<OV::MobileBert> (new OV::MobileBert());
        max_seq_length = 384;
        max_query_length = 64;
        doc_stride = 128;
    }


	// Set Workload Attributes
	std::vector<std::string> in_blobs, out_blobs;
	std::string dataset = workload->getDataset();
	in_blobs = workload->getInputBlobs();
	out_blobs = workload->getOutputBlobs();

	std::map< std::string, std::string > DATASET;
	DATASET["resnet50"] = "imagenet";
	DATASET["mobilenet"] = "imagenet";
	DATASET["ssd-mobilenet"] = "coco";
	DATASET["ssd-resnet34"] = "coco";
	DATASET["deeplabv3"] = "ADE20K";
	DATASET["bert"] = "squad";
	DATASET["mobilebert"] = "squad";
	DATASET["mobilenet-edge"] = "imagenet";
	DATASET["ssd-mobilenet-v2"] = "coco";


    if (dataset.compare("imagenet")==0){
        ds = std::unique_ptr < Imagenet
                > (new Imagenet(settings, image_width, image_height,
                        num_channels, FLAGS_data_path, image_format,
                        FLAGS_total_sample_count, FLAGS_perf_sample_count, FLAGS_model_name,
                        dataset));
    } 
    else if (dataset.compare("coco")==0){ 
        ds = std::unique_ptr < Coco
                > (new Coco(settings, image_width, image_height, num_channels,
                        FLAGS_data_path, image_format, FLAGS_total_sample_count,
                        FLAGS_perf_sample_count, FLAGS_model_name, dataset));
    }
    else if (dataset.compare("ade20k") == 0) { 
        ds = std::unique_ptr < ADE20K >(new ADE20K(settings, image_width, image_height, num_channels,
            FLAGS_data_path, image_format, FLAGS_total_sample_count,
            FLAGS_perf_sample_count, FLAGS_model_name, dataset));
    }
    else if (dataset.compare("squad")==0){
        ds = std::unique_ptr < Squad
                > (new Squad(settings, max_seq_length, max_query_length,
                        doc_stride, FLAGS_data_path, FLAGS_total_sample_count,
                        FLAGS_perf_sample_count, FLAGS_model_name, dataset));
                         
    }

    std::string trail_space(45, ' ');
    std::string app_info = "";
    if (FLAGS_mode.compare("Performance") == 0) {
        app_info = "    [INFO] Starting Performance Benchmark\n";
    }
    else {
        app_info = "    [INFO] Starting Accuracy Benchmark\n";
    }

    // Init SUT
    if (settings.scenario == mlperf::TestScenario::SingleStream){ 
    sut = std::unique_ptr < SUTBase
            > (new SUTBase(settings, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads, FLAGS_batch_size,
                    FLAGS_dataset, FLAGS_model_name, FLAGS_device, FLAGS_cpu_extension, in_blobs, out_blobs, FLAGS_model_path, post_processor, FLAGS_nseq, FLAGS_nseq_step, FLAGS_enforcebf16));

        if (FLAGS_warmup_iters > 0) {
            std::cout << "    [INFO] Warming up \n";
            sut->warmUp(FLAGS_warmup_iters);
        }

	    void* sut = mlperf::c::ConstructSUT(0, "SUT", 3, issueQueries, flushQueries, processLatencies);
	    void* qsl = mlperf::c::ConstructQSL(0, "QSL", 3, FLAGS_total_sample_count, FLAGS_perf_sample_count, loadQuerySamples, unloadQuerySamples);

        std::cout << app_info;
	    mlperf::c::StartTest(sut, qsl, settings);
        std::cout << "    [INFO] Benchmark Completed" << trail_space << "\n";

	    mlperf::c::DestroyQSL(qsl);
	    mlperf::c::DestroySUT(sut);
        


     } else if (settings.scenario == mlperf::TestScenario::Offline) {
    	sut_offline = std::unique_ptr < SUTOffline > (new SUTOffline(settings, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads, FLAGS_batch_size,  FLAGS_dataset, FLAGS_model_name, FLAGS_device, FLAGS_cpu_extension, in_blobs, out_blobs, FLAGS_model_path, post_processor, FLAGS_nseq, FLAGS_nseq_step, FLAGS_enforcebf16));

	    if (FLAGS_warmup_iters > 0) {
            std::cout << "    [INFO] Warming up \n";
        	sut_offline->warmupOffline(FLAGS_warmup_iters);
    	}

    	void* sut = mlperf::c::ConstructSUT(0, "SUT", 3, issueQueriesOffline, flushQueries, processLatencies);
    	void* qsl = mlperf::c::ConstructQSL(0, "QSL", 3, FLAGS_total_sample_count, FLAGS_perf_sample_count, loadQuerySamples, unloadQuerySamples);
        
        
        std::cout << app_info;
        mlperf::c::StartTest(sut, qsl, settings);
        std::cout << "    [INFO] Benchmark Completed" << trail_space << "\n";

	    mlperf::c::DestroyQSL(qsl);
    	mlperf::c::DestroySUT(sut);

     } else if (settings.scenario == mlperf::TestScenario::MultiStream) {
    	sut_multistream = std::unique_ptr < SUTMultistream > (new SUTMultistream(settings, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads, FLAGS_batch_size,  FLAGS_dataset, FLAGS_model_name, FLAGS_device, FLAGS_cpu_extension, in_blobs, out_blobs, FLAGS_model_path, post_processor));

	    if (FLAGS_warmup_iters > 0) {
            std::cout << "    [INFO] Warming up \n";
        	sut_multistream->warmupMultistream(FLAGS_warmup_iters);
    	}

    	void* sut = mlperf::c::ConstructSUT(0, "SUT", 3, issueQueriesMultistream, flushQueries,  processLatencies);
    	void* qsl = mlperf::c::ConstructQSL(0, "QSL", 3, FLAGS_total_sample_count, FLAGS_perf_sample_count, loadQuerySamples, unloadQuerySamples);
        
        std::cout << app_info;
        mlperf::c::StartTest(sut, qsl, settings);
        std::cout << "    [INFO] Benchmark Completed" << trail_space << "\n";

        mlperf::c::DestroyQSL(qsl);
    	mlperf::c::DestroySUT(sut);

     } else if (settings.scenario == mlperf::TestScenario::Server) {
        sut_server = std::unique_ptr < SUTServer > (new SUTServer(settings, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads, FLAGS_batch_size,  FLAGS_dataset, FLAGS_model_name, FLAGS_device, FLAGS_cpu_extension, in_blobs, out_blobs, FLAGS_model_path, post_processor, FLAGS_nseq, FLAGS_nseq_step, FLAGS_enforcebf16));

        if (FLAGS_warmup_iters > 0) {
            std::cout << "    [INFO] Warming up \n";
            sut_server->warmupServer(FLAGS_warmup_iters);
        }

        void* sut = mlperf::c::ConstructSUT(0, "SUT", 3, issueQueriesServer, flushQueries, processLatencies);
        void* qsl = mlperf::c::ConstructQSL(0, "QSL", 3, FLAGS_total_sample_count, FLAGS_perf_sample_count, loadQuerySamples, unloadQuerySamples);
        
        std::cout << app_info;
        mlperf::c::StartTest(sut, qsl, settings);
        std::cout << "    [INFO] Benchmark Completed" << trail_space << "\n";
        
        mlperf::c::DestroyQSL(qsl);
        mlperf::c::DestroySUT(sut);

     }


    return 0;
}


