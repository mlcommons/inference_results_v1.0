#ifndef POST_PROCESSORS_H_
#define POST_PROCESSORS_H_

#include "utils.h"

namespace Processors {

void postProcessSSDMobilenet(Item qitem, InferenceEngine::InferRequest req, std::vector<float> &results, std::vector<mlperf::ResponseId> &response_ids, unsigned batch_size,  std::vector<unsigned> &counts) {

    Blob::Ptr out = req.GetBlob("DetectionOutput");
    unsigned count = 0;
    int image_id = 0, prev_image_id = 0;
    size_t j = 0;
    int object_size_ = 7;
    int max_proposal_count_ = 100;
    
    //Blob::Ptr out = outputs.blobs_[0]; //outputs.blob_;
    std::vector < mlperf::QuerySampleIndex > sample_idxs =  qitem.sample_idxs_;
    std::vector < mlperf::QuerySampleIndex > resp_ids =     qitem.response_ids_;

    const float* detection =
            static_cast<PrecisionTrait<Precision::FP32>::value_type*>(out->buffer());

    for (int curProposal = 0; curProposal < max_proposal_count_;
            curProposal++) {
        image_id = static_cast<int>(detection[curProposal * object_size_ + 0]);

        if (image_id != prev_image_id) {
            counts.push_back(count * 7);
            response_ids.push_back(resp_ids[j]);
            ++j;
            count = 0;
            prev_image_id = prev_image_id + 1;
            if (image_id > 0) {
                while (image_id != prev_image_id) {
                    counts.push_back(count * 7);
                    response_ids.push_back(resp_ids[j]);
                    ++j;
                    count = 0;
                    prev_image_id = prev_image_id + 1;
                }
            } else {
                while (prev_image_id < (int) batch_size) {
                    counts.push_back(count * 7);
                    response_ids.push_back(resp_ids[j]);
                    ++j;
                    count = 0;
                    prev_image_id = prev_image_id + 1;
                }
            }
        }
        if (image_id < 0) {
            break;
        }

        float confidence = detection[curProposal * object_size_ + 2];
        float label = static_cast<float>(detection[curProposal * object_size_ + 1]);
        float xmin = static_cast<float>(detection[curProposal * object_size_ + 3]);
        float ymin = static_cast<float>(detection[curProposal * object_size_ + 4]);
        float xmax = static_cast<float>(detection[curProposal * object_size_ + 5]);
        float ymax = static_cast<float>(detection[curProposal * object_size_ + 6]);

        if (confidence > 0.05) {
            /** Add only objects with >95% probability **/
            results.push_back(float(sample_idxs[j]));
            results.push_back(ymin);
            results.push_back(xmin);
            results.push_back(ymax);
            results.push_back(xmax);
            results.push_back(confidence);
            results.push_back(label);

            ++count;
        }

        if (curProposal == (max_proposal_count_ - 1)) {
            counts.push_back(count * 7);
            response_ids.push_back(resp_ids[j]);
            ++j;
            count = 0;
            prev_image_id = prev_image_id + 1;
            while (prev_image_id < (int) batch_size) {
                counts.push_back(count * 7);
                response_ids.push_back(resp_ids[j]);
                ++j;
                count = 0;
                prev_image_id = prev_image_id + 1;
            }
        }
    }
}


void postProcessSSDResnet(Item qitem, InferenceEngine::InferRequest req,
        std::vector<float> &result, std::vector<mlperf::ResponseId> &response_ids,
        unsigned batch_size, std::vector<unsigned> &counts) {
    unsigned count = 0;
    
    Blob::Ptr bbox_blob = req.GetBlob("Unsqueeze_bboxes/Unsqueeze");// "Unsqueeze_bboxes777");//outputs.blob_;
    Blob::Ptr scores_blob = req.GetBlob("Unsqueeze_scores/Unsqueeze");// "Unsqueeze_scores835");//outputs.blob1_;
    Blob::Ptr labels_blob = req.GetBlob("Add_labels");//outputs.blob2_;
    

    /*
    Blob::Ptr bbox_blob = outputs.blobs_[0];
    Blob::Ptr scores_blob = outputs.blobs_[1];
    Blob::Ptr labels_blob = outputs.blobs_[2];
    */

    int object_size_ = 4;
    int max_proposal_count_ = 200;

    std::vector < mlperf::QuerySampleIndex > sample_idxs = qitem.sample_idxs_;

    const float* BoundingBoxes =
            static_cast<float*>(bbox_blob->buffer());
    const float* Confidence = static_cast<float*>(scores_blob->buffer());
    const float* Labels = static_cast<float*>(labels_blob->buffer());

    for (size_t j = 0; j < batch_size; ++j) {
        auto cur_item = (j * max_proposal_count_);
        auto cur_bbox = (j * max_proposal_count_ * object_size_);

        count = 0;
        for (int curProposal = 0; curProposal < max_proposal_count_;
                curProposal++) {
            float confidence = Confidence[cur_item + curProposal];
            float label =
                    static_cast<int>(Labels[cur_item + curProposal]);
            float xmin = static_cast<float>(BoundingBoxes[cur_bbox
                    + curProposal * object_size_ + 0]);
            float ymin = static_cast<float>(BoundingBoxes[cur_bbox
                    + curProposal * object_size_ + 1]);
            float xmax = static_cast<float>(BoundingBoxes[cur_bbox
                    + curProposal * object_size_ + 2]);
            float ymax = static_cast<float>(BoundingBoxes[cur_bbox
                    + curProposal * object_size_ + 3]);

            if (confidence > 0.05) {
                /** Add only objects with > 0.05 probability **/
                result.push_back(float(sample_idxs[j]));
                result.push_back(ymin);
                result.push_back(xmin);
                result.push_back(ymax);
                result.push_back(xmax);
                result.push_back(confidence);
                result.push_back(label);

                ++count;
            }
        }

        counts.push_back(count * 7);
        response_ids.push_back(qitem.response_ids_[j]);
    }
}

void postProcessDeepLabv3(Item qitem, InferenceEngine::InferRequest req,
    std::vector<float>& result, std::vector<mlperf::ResponseId>& response_ids,
    unsigned batch_size, std::vector<unsigned>& counts) {

    unsigned count = 0;
    size_t blob_size = 512 * 512;

    Blob::Ptr segmask_blob = req.GetBlob("ArgMax/Squeeze");
    const float* seg_data = static_cast<float*>(segmask_blob->buffer());
    //std::cout << " segmask size: " << segmask_blob->size() << " \n";

    std::vector < mlperf::QuerySampleIndex > sample_idxs = qitem.sample_idxs_;

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t j = 0; j < blob_size; j++) {
            result.push_back(seg_data[j]);
        }
        response_ids.push_back(qitem.response_ids_[b]);
        counts.push_back( blob_size );
    }

}

void postProcessMobilenet(Item qitem, InferenceEngine::InferRequest req,
        std::vector<float> &results, std::vector<mlperf::ResponseId> &response_ids, unsigned batch_size, std::vector<unsigned> &counts) {

    std::vector<unsigned> res;
    //std::cout << " -- Fetching output blob --\n";
    Blob::Ptr out = req.GetBlob("MobilenetV1/Predictions/Softmax");

    //std::cout << " -- Computing top results --\n";
    TopResults(1, *out, res);

    for (size_t j = 0; j < res.size(); ++j) {
        results.push_back(static_cast<float>(res[j] - 1));
        response_ids.push_back(qitem.response_ids_[j]);
	    counts.push_back(1);
    }
}

void postProcessMobilenetEdge(Item qitem, InferenceEngine::InferRequest req,
    std::vector<float>& results, std::vector<mlperf::ResponseId>& response_ids, unsigned batch_size, std::vector<unsigned>& counts) {

    std::vector<unsigned> res;
    Blob::Ptr out = req.GetBlob("Softmax");
    TopResults(1, *out, res); 

    for (size_t j = 0; j < res.size(); ++j) {
        results.push_back(static_cast<float>(res[j] - 1));
        response_ids.push_back(qitem.response_ids_[j]);
        counts.push_back(1);
    }
}

void postProcessResnet50(Item qitem, InferenceEngine::InferRequest req,
        std::vector<float> &results, std::vector<mlperf::ResponseId> &response_ids, unsigned batch_size, std::vector<unsigned> &counts) {
    std::vector<unsigned> res;
    Blob::Ptr out = req.GetBlob("softmax_tensor");
    TopResults(1, *out, res);
    
    for (size_t j = 0; j < res.size(); ++j) {
        results.push_back(static_cast<float>(res[j] - 1));
        response_ids.push_back(qitem.response_ids_[j]);
	counts.push_back(1);
    }
}


void postProcessBert(Item qitem, InferenceEngine::InferRequest req,
        std::vector<float> &results, std::vector<mlperf::ResponseId> &response_ids, unsigned batch_size, std::vector<unsigned> &counts){



	Blob::Ptr out_0 = req.GetBlob("start_logits");
	Blob::Ptr out_1 = req.GetBlob("end_logits");
	
	size_t offset = out_0->size() / batch_size;
	const float* out_0_data = static_cast<float*> (out_0->buffer());
	const float* out_1_data = static_cast<float*> (out_1->buffer());

    size_t n0 = results.size();
    results.resize(n0 + 2 * out_0->size());

    for (size_t j = 0; j < batch_size; j++) {

        response_ids.push_back(qitem.response_ids_[j]);
        for (size_t i = 0, k = 0; i < offset; i++, k += 2) {

            results[n0 + j * offset + k] = out_0_data[i];
            results[n0 + j * offset + k + 1] = out_1_data[i];
        }
		counts.push_back( offset * 2 );

		// Next sample
		out_0_data += offset;
		out_1_data += offset;
	}	
}        

void postProcessMobileBert(Item qitem, InferenceEngine::InferRequest req,
    std::vector<float>& results, std::vector<mlperf::ResponseId>& response_ids, unsigned batch_size, std::vector<unsigned>& counts) {



    Blob::Ptr out_0 = req.GetBlob("Squeeze_7015");
    Blob::Ptr out_1 = req.GetBlob("Squeeze_7016");

    size_t offset = out_0->size() / batch_size;
    const float* out_0_data = static_cast<float*> (out_0->buffer());
    const float* out_1_data = static_cast<float*> (out_1->buffer());

    size_t n0 = results.size();
    results.resize(n0 + 2 * out_0->size());

    for (size_t j = 0; j < batch_size; j++) {

        response_ids.push_back(qitem.response_ids_[j]);
        for (size_t i = 0, k = 0; i < offset; i++, k += 2) {

            results[n0 + j * offset + k] = out_0_data[i];
            results[n0 + j * offset + k + 1] = out_1_data[i];
        }
        counts.push_back(offset * 2);

        // Next sample
        out_0_data += offset;
        out_1_data += offset;
    }

}
};
#endif
