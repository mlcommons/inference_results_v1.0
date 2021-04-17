#ifndef ITEM_H__
#define ITEM_H__

#include <vector>
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


using namespace InferenceEngine;

class Item {
public:
    Item(Blob::Ptr blob, std::vector<mlperf::ResponseId> response_ids,
            std::vector<mlperf::QuerySampleIndex> sample_idxs) :
            blob_(blob), response_ids_(response_ids), sample_idxs_(sample_idxs) {
    }

    Item(Blob::Ptr blob, std::vector<mlperf::ResponseId> response_ids,
            std::vector<mlperf::QuerySampleIndex> sample_idxs, int label) :
            blob_(blob), response_ids_(response_ids), sample_idxs_(sample_idxs), label_(label) {
    }

    Item(Blob::Ptr blob, std::vector<mlperf::QuerySampleIndex> sample_idxs) :
            blob_(blob),
            sample_idxs_(sample_idxs) {
    }

    Item(std::vector<Blob::Ptr> blobs, std::vector<mlperf::ResponseId> response_ids, std::vector<mlperf::QuerySampleIndex> sample_idxs) :
            blobs_(blobs), response_ids_(response_ids), sample_idxs_(sample_idxs) {
    }


        Item(std::map<std::string, Blob::Ptr> blobs_map, std::vector<mlperf::ResponseId> response_ids, std::vector<mlperf::QuerySampleIndex> sample_idxs) :
            blobs_map_(blobs_map), response_ids_(response_ids), sample_idxs_(sample_idxs) {
    }


    Item() {
    }

    void setBlob(Blob::Ptr blob){
	    std::cout << " Setting blob\n";
	    blob_ = blob;
    }

    void setResponseIds(std::vector<mlperf::ResponseId> res_ids){
	    std::cout << " Setting response ids\n";
	    response_ids_ = res_ids;
    }

    void setSampleIndex(std::vector<mlperf::QuerySampleIndex> samp_idx){
	    std::cout << " Setting sample index\n";
	    sample_idxs_ = samp_idx;
    }

    void setLabel(int label){
	    std::cout << " Setting label\n";
	    label_ = label;
    }

public:
    Blob::Ptr blob_;
    std::vector<mlperf::ResponseId> response_ids_;
    std::vector<mlperf::QuerySampleIndex> sample_idxs_;
    Blob::Ptr blob1_;
    Blob::Ptr blob2_;
    std::vector<Blob::Ptr> blobs_;
    std::map<std::string, Blob::Ptr> blobs_map_;
    int label_;

};

#endif
