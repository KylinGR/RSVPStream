#include "inference_engine.h"
#include <stdexcept>

InferenceEngine::InferenceEngine(const std::string& rknn_model_path)
    : rknn_model_path_(rknn_model_path) {
}

void InferenceEngine::initialize() {
    model_ = std::make_unique<ENSEMBLEModel>(rknn_model_path_);
}

InferenceResult InferenceEngine::run_inference(const ProcessedData& data) {
    const auto& input1_dims = model_->get_input1_dims();
    const auto& input2_dims = model_->get_input2_dims();
    
    std::vector<float> input1(input1_dims.batch * input1_dims.height * input1_dims.width * input1_dims.channel);
    std::vector<float> input2(input2_dims.batch * input2_dims.height * input2_dims.width * input2_dims.channel);

    // 准备第一个输入
    size_t index = 0;
    for (size_t k = 0; k < data.x_local[0].size(); ++k) {
        for (const auto& outer : data.x_local) {
            input1[index++] = outer[k];
        }
    }

    // 准备第二个输入
    index = 0;
    for (size_t k = 0; k < data.x_global[0].size(); ++k) {
        for (const auto& outer : data.x_global) {
            input2[index++] = outer[k];
        }
    }

    // 执行推理
    if (!model_->infer(input1.data(), input2.data())) {
        throw std::runtime_error("Inference failed for sample: " + data.file_path);
    }

    // 获取输出结果
    const auto& output = model_->get_output();
    
    return {output[0], data.file_path, data.sample_id};
}
