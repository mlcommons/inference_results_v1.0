#include <iostream>
#include <vector>
#include <memory>
#include <getopt.h>
#include <string>
#include <string.h>
#include <experimental/filesystem>

#include "include/benchmark.h"
#include "include/ssdmobile.h"
#include "include/resnet50.h"
#include "include/maccel.h"
#include "include/type.h"
#include "include/bindings/c_api.h"
#include "include/loadgen.h"
#include "include/system_under_test.h"
#include "include/query_sample_library.h"
#include "include/test_settings.h"

#include <thread>

using std::thread;

using namespace std;
using namespace mlperf;
using namespace mlperf::c;
using namespace mobilint;
using namespace maccel_type;

vector<unique_ptr<Accelerator>> mAccelerator;
vector<unique_ptr<PostprocessorManager>> mPostprocessor;
Model SSDMobileNet, ResNet, SSDResNet;

uint8_t** loadedSample = nullptr;
uint64_t* lookup = nullptr;

int main(int argc, char **argv) {
    const string SUT_NAME = "MOBILINT";

    const string SCENARIO_SINGLE_STREAM = "SingleStream";
    const string SCENARIO_MULTI_STREAM = "MultiStream";
    const string SCENARIO_OFFLINE = "Offline";
    const string SCENARIO_SERVER = "Server";
    const string MODE_ACCURACY_ONLY = "AccuracyOnly";
    const string MODE_PERFORMANCE_ONLY = "PerformanceOnly";
    const string MODEL_SSD_MOBILENET_V1 = "SSDMobileNetV1";
    const string MODEL_SSD_RESNET = "SSDResNet";
    const string MODEL_RESNET_50 = "ResNet50";

    string config_path = "mlperf.conf";
    string config_user_path = "user.conf";
    string dataset_path = "./dataset";
    string scenario = SCENARIO_SINGLE_STREAM;
    string mode = MODE_ACCURACY_ONLY;
    string model = MODEL_SSD_MOBILENET_V1;

    int cmd_opt;
    bool config_fail = false;
        
    static struct option const long_opts[] = {
        {"config",          required_argument, NULL, 'c'},
        {"config-user",     required_argument, NULL, 'u'},
        {"dataset-path",    required_argument, NULL, 'd'},
        {"scenario",        required_argument, NULL, 's'},
        {"mode",            required_argument, NULL, 'm'},
        {"model",           required_argument, NULL, 'l'},
        {0, 0, 0, 0}
    };

    while ((cmd_opt = getopt_long(argc, argv, "c:s:m:l", 
        long_opts, NULL)) != -1) {
		switch (cmd_opt) {
		case 0:
			break;
		case 'c':
			config_path = strdup(optarg);
            if (!experimental::filesystem::exists(config_path)) {
                config_fail = true;
                cerr << "Config file does not exist." << endl;
            }
			break;
        case 'u':
			config_user_path = strdup(optarg);
            if (!experimental::filesystem::exists(config_user_path)) {
                config_fail = true;
                cerr << "User config file does not exist." << endl;
            }
			break;
        case 'd':
			dataset_path = strdup(optarg);
            if (!experimental::filesystem::exists(dataset_path)) {
                config_fail = true;
                cerr << "Dataset directory does not exist." << endl;
            }
			break;
        case 's':
            scenario = strdup(optarg);
            if (scenario != SCENARIO_SINGLE_STREAM && 
                scenario != SCENARIO_MULTI_STREAM &&
                scenario != SCENARIO_OFFLINE &&
                scenario != SCENARIO_SERVER) {
                config_fail = true;
                cerr << "Wrong Scenario." << endl;
            }
            break;
        case 'm':
            mode = strdup(optarg);
            if (mode != MODE_ACCURACY_ONLY && mode != MODE_PERFORMANCE_ONLY) {
                config_fail = true;
                cerr << "Wrong Mode." << endl;
            }
            break;
        case 'l':
            model = strdup(optarg);
            if (model != MODEL_SSD_MOBILENET_V1 && 
                model != MODEL_SSD_RESNET && 
                model != MODEL_RESNET_50) {
                config_fail = true;
                cerr << "Wrong Model." << endl;
            }
            break;
		default:
            config_path = true;
            cerr << "Wrong argument provided." << endl;
			break;
		}
	}

    if (config_fail) {
        cerr << "Config by argument failed, abort benchmark." << endl;
        return 1;
    }

    TestSettings settings;
    settings.FromConfig(config_path, model, scenario);
    settings.FromConfig(config_user_path, model, scenario);

    if (scenario == SCENARIO_SINGLE_STREAM) {
        settings.scenario = TestScenario::SingleStream;
    } else if (scenario == SCENARIO_MULTI_STREAM) {
        settings.scenario = TestScenario::MultiStream;
    } else if (scenario == SCENARIO_OFFLINE) {
        settings.scenario = TestScenario::Offline;
    } else if (scenario == SCENARIO_SERVER) {
        settings.scenario = TestScenario::Server;
    } else {
        cerr << 
            "Unknown error, TestSetting scenario set failed. abort benchmark." 
            << endl;
        return 1;
    }

    if (mode == MODE_ACCURACY_ONLY) {
        settings.mode = TestMode::AccuracyOnly;
    } else if (mode == MODE_PERFORMANCE_ONLY) {
        settings.mode = TestMode::PerformanceOnly;
    } else {
        cerr << 
            "Unknown error, TestSetting mode set failed. abort benchmark." 
            << endl;
        return 1;
    }

    /* Instantiate the Accelerators (Multiple Accelerator if needed) */
    mAccelerator.push_back(make_unique<Accelerator>(0x02, 0, true));
    mPostprocessor.push_back(make_unique<PostprocessorManager>());

    void *sut, *qsl = nullptr;

    if (model == MODEL_SSD_MOBILENET_V1) {
        lookup = (uint64_t *) malloc(sizeof(uint64_t) * 5000);

        vector<V1BinConfig*> group_efficientdet = {
            new V1BinConfig(CoreName::Core1,
                "ssdmobilenet/imem.bin.core1",
                "ssdmobilenet/lmem.bin",
                "ssdmobilenet/dmem.bin",
                "ssdmobilenet/ddr.bin",
                3'500'000, 5'000'000),
            new V1BinConfig(CoreName::Core2,
                "ssdmobilenet/imem.bin.core2",
                "ssdmobilenet/lmem.bin",
                "ssdmobilenet/dmem.bin",
                "ssdmobilenet/ddr.bin",
                3'500'000, 5'000'000),
            new V1BinConfig(CoreName::Core3,
                "ssdmobilenet/imem.bin.core3",
                "ssdmobilenet/lmem.bin",
                "ssdmobilenet/dmem.bin",
                "ssdmobilenet/ddr.bin",
                3'500'000, 5'000'000)
        };

        SSDMobileNet = ModelBuilder()
            .setNickname("SSD-MobileNet Core")
            .setBinConfig(group_efficientdet)
            .setCollaborationModel(CollaborationModel::Unified)
            .build();

        for (int i = 0; i < mAccelerator.size(); i++) {
            mAccelerator[i]->setModel(SSDMobileNet);
        }

        /* Prepare SUT, QSL */
        sut = ConstructSUT(
            ssd_mobilenet::CLIENT_DATA, SUT_NAME.c_str(), SUT_NAME.length(), 
            ssd_mobilenet::issueQuery, ssd_mobilenet::flushQueries, 
            ssd_mobilenet::processLatencies);
        qsl = ConstructQSL(
            ssd_mobilenet::CLIENT_DATA, SUT_NAME.c_str(), SUT_NAME.length(), 
            ssd_mobilenet::TOTAL_SAMPLE_COUNT, ssd_mobilenet::QSL_SIZE, 
            ssd_mobilenet::loadSamplesToRAM, 
            ssd_mobilenet::unloadSamplesFromRAM);
    } else if (model == MODEL_RESNET_50) {
        lookup = (uint64_t *) malloc(sizeof(uint64_t) * 50000);

        vector<V1BinConfig*> group = {
            new V1BinConfig(CoreName::Core1,
                "resnet50/imem.bin.core1",
                "resnet50/lmem.bin",
                "resnet50/dmem.bin",
                "resnet50/ddr.bin",
                13'000'000, 15'000'000),
            new V1BinConfig(CoreName::Core2,
                "resnet50/imem.bin.core2",
                "resnet50/lmem.bin",
                "resnet50/dlmem.bin",
                "resnet50/ddr.bin",
                13'000'000, 15'000'000),
            new V1BinConfig(CoreName::Core3,
                "resnet50/imem.bin.core3",
                "resnet50/lmem.bin",
                "resnet50/dlmem.bin",
                "resnet50/ddr.bin",
                13'000'000, 15'000'000)
        };

        ResNet = ModelBuilder()
            .setNickname("ResNet Core")
            .setBinConfig(group)
            .setCollaborationModel(CollaborationModel::Unified)
            .build();

        for (int i = 0; i < mAccelerator.size(); i++) {
            mAccelerator[i]->setModel(ResNet);
        }

        /* Prepare SUT, QSL */
        sut = ConstructSUT(
            resnet50::CLIENT_DATA, SUT_NAME.c_str(), SUT_NAME.length(), 
            resnet50::issueQuery, resnet50::flushQueries, 
            resnet50::processLatencies);
        qsl = ConstructQSL(
            resnet50::CLIENT_DATA, SUT_NAME.c_str(), SUT_NAME.length(), 
            resnet50::TOTAL_SAMPLE_COUNT, resnet50::QSL_SIZE, 
            resnet50::loadSamplesToRAM, 
            resnet50::unloadSamplesFromRAM);
    }

    if (sut != nullptr && qsl != nullptr) {
        StartTest(sut, qsl, settings);
    } else {
        cerr << 
            "Unknown error, StartTest failed. abort benchmark." 
            << endl;
        return 1;
    }

    DestroyQSL(qsl);
    DestroySUT(sut);

    return 0;
}
