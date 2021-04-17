// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <algorithm>
#include <thread>
#include <utility>
#include <vector>
#include <map>

#include <ie_blob.h>

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

std::vector<std::string> parseDevices(const std::string& device_string) {
    std::string comma_separated_devices = device_string;
    if (comma_separated_devices.find(":") != std::string::npos) {
        comma_separated_devices = comma_separated_devices.substr(comma_separated_devices.find(":") + 1);
    }
    auto devices = split(comma_separated_devices, ',');
    for (auto& device : devices)
        device = device.substr(0, device.find("("));
    return devices;
}

std::map<std::string, uint32_t> parseValuePerDevice(const std::vector<std::string>& devices,
                                                    const std::string& values_string) {
    //  Format: <device1>:<value1>,<device2>:<value2> or just <value>
    auto values_string_upper = values_string;
    std::transform(values_string_upper.begin(),
                   values_string_upper.end(),
                   values_string_upper.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    std::map<std::string, uint32_t> result;
    auto device_value_strings = split(values_string_upper, ',');
    for (auto& device_value_string : device_value_strings) {
        auto device_value_vec =  split(device_value_string, ':');
        if (device_value_vec.size() == 2) {
            auto it = std::find(devices.begin(), devices.end(), device_value_vec.at(0));
            if (it != devices.end()) {
                result[device_value_vec.at(0)] = std::stoi(device_value_vec.at(1));
            }
        } else if (device_value_vec.size() == 1) {
            uint32_t value = std::stoi(device_value_vec.at(0));
            for (auto& device : devices) {
                result[device] = value;
            }
        } else if (device_value_vec.size() != 0) {
            throw std::runtime_error("Unknown string format: " + values_string);
        }
    }
    return result;
}

std::map<std::string, uint32_t> parseNStreamsValuePerDevice(const std::vector<std::string>& devices,
	const std::string& values_string) {
	//  Format: <device1>:<value1>,<device2>:<value2> or just <value>
	auto values_string_upper = values_string;
	std::map<std::string, uint32_t> result;
	auto device_value_strings = split(values_string_upper, ',');
	for (auto& device_value_string : device_value_strings) {
		auto device_value_vec = split(device_value_string, ':');
		if (device_value_vec.size() == 2) {
			auto device_name = device_value_vec.at(0);
			auto nstreams = device_value_vec.at(1);
			auto it = std::find(devices.begin(), devices.end(), device_name);
			if (it != devices.end()) {
				result[device_name] = std::stoi(nstreams);
			}
			else {
				throw std::logic_error("Can't set nstreams value " + std::string(nstreams) +
					" for device '" + device_name + "'! Incorrect device name!");
			}
		}
		else if (device_value_vec.size() == 1) {
			uint32_t value = std::stoi(device_value_vec.at(0));
			for (auto& device : devices) {
				result[device] = value;
			}
		}
		else if (device_value_vec.size() != 0) {
			throw std::runtime_error("Unknown string format: " + values_string);
		}
	}
	return result;
}


bool adjustShapesBatch(InferenceEngine::ICNNNetwork::InputShapes& shapes,
    const size_t batch_size, const InferenceEngine::InputsDataMap& input_info) {
    bool updated = false;
    for (auto& item : input_info) {
        auto layout = item.second->getTensorDesc().getLayout();

        int batch_index = -1;
        if ((layout == InferenceEngine::Layout::NCHW) || (layout == InferenceEngine::Layout::NCDHW) ||
            (layout == InferenceEngine::Layout::NHWC) || (layout == InferenceEngine::Layout::NDHWC) ||
            (layout == InferenceEngine::Layout::NC)) {
            batch_index = 0;
        }
        else if (layout == InferenceEngine::Layout::CN) {
            batch_index = 1;
        }
        if ((batch_index != -1) && (shapes.at(item.first).at(batch_index) != batch_size)) {
            shapes[item.first][batch_index] = batch_size;
            updated = true;
        }
    }
    return updated;
}

std::string getShapesString(const InferenceEngine::ICNNNetwork::InputShapes& shapes) {
    std::stringstream ss;
    for (auto& shape : shapes) {
        if (!ss.str().empty()) ss << ", ";
        ss << "\'" << shape.first << "': [";
        for (size_t i = 0; i < shape.second.size(); i++) {
            if (i > 0) ss << ", ";
            ss << shape.second.at(i);
        }
        ss << "]";
    }
    return ss.str();
}

template <class T>
void TopResults(unsigned int n, InferenceEngine::TBlob<T>& input, std::vector<unsigned>& output) {
        InferenceEngine::SizeVector dims = input.getTensorDesc().getDims();
        size_t input_rank = dims.size();
        if (!input_rank || !dims[0]) THROW_IE_EXCEPTION << "Input blob has incorrect dimensions!";
        size_t batchSize = dims[0];
        std::vector<unsigned> indexes(input.size() / batchSize);

        n = static_cast<unsigned>(std::min<size_t>((size_t)n, input.size()));

        output.resize(n * batchSize);

        for (size_t i = 0; i < batchSize; i++) {
            size_t offset = i * (input.size() / batchSize);
            T* batchData = input.data();
            batchData += offset;

            std::iota(std::begin(indexes), std::end(indexes), 0);
            std::partial_sort(std::begin(indexes), std::begin(indexes) + n, std::end(indexes),
                              [&batchData](unsigned l, unsigned r) {
                                  return batchData[l] > batchData[r];
                              });
            for (unsigned j = 0; j < n; j++) {
                output.at(i * n + j) = indexes.at(j);
            }
        }
    }
	

    /**
     * @brief Gets the top n results from a blob
     *
     * @param n Top n count
     * @param input 1D blob that contains probabilities
     * @param output Vector of indexes for the top n places
     */
	
    void TopResults(unsigned int n, InferenceEngine::Blob& input, std::vector<unsigned>& output) {
    #define TBLOB_TOP_RESULT(precision)                                                                            \
        case InferenceEngine::Precision::precision: {                                                              \
            using myBlobType = InferenceEngine::PrecisionTrait<InferenceEngine::Precision::precision>::value_type;  \
            InferenceEngine::TBlob<myBlobType>& tblob = dynamic_cast<InferenceEngine::TBlob<myBlobType>&>(input);   \
            TopResults(n, tblob, output);                                                                          \
            break;                                                                                                 \
        }

        switch (input.getTensorDesc().getPrecision()) {
            TBLOB_TOP_RESULT(FP32);
            TBLOB_TOP_RESULT(FP16);
            TBLOB_TOP_RESULT(Q78);
            TBLOB_TOP_RESULT(I16);
            TBLOB_TOP_RESULT(U8);
            TBLOB_TOP_RESULT(I8);
            TBLOB_TOP_RESULT(U16);
            TBLOB_TOP_RESULT(I32);
            TBLOB_TOP_RESULT(U64);
            TBLOB_TOP_RESULT(I64);
        default:
            THROW_IE_EXCEPTION << "cannot locate blob for precision: " << input.getTensorDesc().getPrecision();
        }

        #undef TBLOB_TOP_RESULT
}


