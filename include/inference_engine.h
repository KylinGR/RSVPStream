#pragma once
#include <vector>
#include <string>
#include <memory>
#include "data_types.h"
#include "ensemble_model.h"

class InferenceEngine {
private:
    std::string rknn_model_path_;
    std::unique_ptr<ENSEMBLEModel> model_;

public:
    explicit InferenceEngine(const std::string& rknn_model_path);
    ~InferenceEngine() = default;
    
    void initialize();
    InferenceResult run_inference(const ProcessedData& data);
    
private:
    std::vector<float> prepare_input_data(
        const std::vector<std::vector<float>>& x_local,
        const std::vector<std::vector<float>>& x_global
    );
};
