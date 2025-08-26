#ifndef ENSEMBLE_MODEL_H
#define ENSEMBLE_MODEL_H

#include <vector>
#include <string>
#include <stdexcept>
#include "rknn_api.h"
#include "utils.h"

class ENSEMBLEModel {
public:
    // 输入维度结构体
    struct InputDims {
        uint32_t batch;
        uint32_t height;
        uint32_t width;
        uint32_t channel;
    };

    explicit ENSEMBLEModel(const std::string& model_path);
    ~ENSEMBLEModel();

    // 推理接口（支持双输入）
    bool infer(const float* input1, const float* input2);

    // 获取输出结果
    const std::vector<float>& get_output(size_t index = 0) const;

    // 获取输入维度信息
    const InputDims& get_input1_dims() const { return input1_dims_; }
    const InputDims& get_input2_dims() const { return input2_dims_; }

private:
    bool initialize();
    bool setup_io_buffers();

    rknn_context ctx_ = 0;
    std::string model_path_;
    
    // 输入输出维度信息
    InputDims input1_dims_;
    InputDims input2_dims_;
    rknn_tensor_attr* input_attrs_ = nullptr;
    rknn_tensor_attr* output_attrs_ = nullptr;

    // 输入输出缓冲区
    std::vector<float*> input_buffers_;
    std::vector<std::vector<float>> outputs_;
};
#endif