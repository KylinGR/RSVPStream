#pragma once
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include "data_types.h"
#include "rknn_api.h"
#include "utils.h"

class InferenceEngine {
public:
    // 输入维度结构体
    struct InputDims {
        uint32_t batch;
        uint32_t height;
        uint32_t width;
        uint32_t channel;
    };

private:
    std::string rknn_model_path_;
    int npu_core_id_;  // NPU核心ID，-1表示自动选择
    
    // RKNN相关
    rknn_context ctx_ = 0;
    InputDims input1_dims_;
    InputDims input2_dims_;
    rknn_tensor_attr* input_attrs_ = nullptr;
    rknn_tensor_attr* output_attrs_ = nullptr;
    std::vector<float*> input_buffers_;
    std::vector<std::vector<float>> outputs_;

public:
    explicit InferenceEngine(const std::string& rknn_model_path, int npu_core_id = -1);
    ~InferenceEngine();
    
    void initialize();
    InferenceResult run_inference(const ProcessedData& data);
    
    // 获取输入维度信息
    const InputDims& get_input1_dims() const { return input1_dims_; }
    const InputDims& get_input2_dims() const { return input2_dims_; }
    
private:
    bool _initialize_model();
    bool _setup_io_buffers();
    bool _infer(const float* input1, const float* input2);
    const std::vector<float>& _get_output(size_t index = 0) const;
    std::vector<float> _prepare_input_data(
        const std::vector<std::vector<float>>& x_local,
        const std::vector<std::vector<float>>& x_global
    );
};
