#include "inference_engine.h"
#include <stdexcept>
#include <iostream>
#include <cstring>

InferenceEngine::InferenceEngine(const std::string& rknn_model_path, int npu_core_id)
    : rknn_model_path_(rknn_model_path), npu_core_id_(npu_core_id) {
}

InferenceEngine::~InferenceEngine() {
    if (ctx_) {
        rknn_destroy(ctx_);
    }
    if (input_attrs_) {
        delete[] input_attrs_;
    }
    if (output_attrs_) {
        delete[] output_attrs_;
    }
    for (auto* buffer : input_buffers_) {
        delete[] buffer;
    }
}

void InferenceEngine::initialize() {
    if (!_initialize_model() || !_setup_io_buffers()) {
        throw std::runtime_error("Model initialization failed");
    }
}

bool InferenceEngine::_initialize_model() {
    // 加载模型
    int ret = rknn_init(&ctx_, (void*)rknn_model_path_.c_str(), 0, 0, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_init failed: " << ret << std::endl;
        return false;
    }

    // 设置NPU核心绑定
    if (npu_core_id_ >= 0) {
        rknn_core_mask core_mask;
        switch (npu_core_id_) {
            case 0:
                core_mask = RKNN_NPU_CORE_0;
                break;
            case 1:
                core_mask = RKNN_NPU_CORE_1;
                break;
            case 2:
                core_mask = RKNN_NPU_CORE_2;
                break;
            default:
                core_mask = RKNN_NPU_CORE_AUTO;
                break;
        }
        
        ret = rknn_set_core_mask(ctx_, core_mask);
        if (ret != RKNN_SUCC) {
            std::cerr << "Warning: rknn_set_core_mask failed for core " << npu_core_id_ 
                      << ", error: " << ret << std::endl;
            // 不要返回失败，继续使用自动模式
        } else {
            std::cout << "Successfully bound inference engine to NPU core " << npu_core_id_ << std::endl;
        }
    }

    // 查询IO信息
    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_query IO num failed: " << ret << std::endl;
        return false;
    }

    // 获取输入属性
    input_attrs_ = new rknn_tensor_attr[io_num.n_input];
    for (int i = 0; i < io_num.n_input; ++i) {
        input_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            std::cerr << "rknn_query input attr failed: " << ret << std::endl;
            return false;
        }
        // 记录输入维度
        if (i == 0) {
            input1_dims_ = {input_attrs_[i].dims[0], input_attrs_[i].dims[1], input_attrs_[i].dims[2], input_attrs_[i].dims[3]};
        } else {
            input2_dims_ = {input_attrs_[i].dims[0], input_attrs_[i].dims[1], input_attrs_[i].dims[2], input_attrs_[i].dims[3]};
        }
    }

    // 获取输出属性
    output_attrs_ = new rknn_tensor_attr[io_num.n_output];
    for (int i = 0; i < io_num.n_output; ++i) {
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            std::cerr << "rknn_query output attr failed: " << ret << std::endl;
            return false;
        }
    }

    return true;
}

bool InferenceEngine::_setup_io_buffers() {
    // 初始化输入缓冲区
    input_buffers_.resize(2);
    input_buffers_[0] = new float[input1_dims_.batch * input1_dims_.height * input1_dims_.width * input1_dims_.channel];
    input_buffers_[1] = new float[input2_dims_.batch * input2_dims_.height * input2_dims_.width * input2_dims_.channel];
    
    // 初始化输出容器
    outputs_.resize(1);
    return true;
}

bool InferenceEngine::_infer(const float* input1, const float* input2) {
    // 设置输入
    rknn_input inputs[2];
    memset(inputs, 0, sizeof(inputs));

    // 输入1配置
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = const_cast<float*>(input1);
    inputs[0].size = input1_dims_.batch * input1_dims_.height * input1_dims_.width * input1_dims_.channel * sizeof(float);

    // 输入2配置
    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    inputs[1].fmt = RKNN_TENSOR_NHWC;
    inputs[1].buf = const_cast<float*>(input2);
    inputs[1].size = input2_dims_.batch * input2_dims_.height * input2_dims_.width * input2_dims_.channel * sizeof(float);

    if (rknn_inputs_set(ctx_, 2, inputs) != RKNN_SUCC) {
        return false;
    }

    // 执行推理
    if (rknn_run(ctx_, nullptr) != RKNN_SUCC) {
        return false;
    }

    // 获取输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    if (rknn_outputs_get(ctx_, 1, outputs, nullptr) != RKNN_SUCC) {
        return false;
    }

    // 存储输出
    float* out_data = static_cast<float*>(outputs[0].buf);
    size_t out_size = outputs[0].size / sizeof(float);
    outputs_[0].assign(out_data, out_data + out_size);

    rknn_outputs_release(ctx_, 1, outputs);
    return true;
}

const std::vector<float>& InferenceEngine::_get_output(size_t index) const {
    if (index >= outputs_.size()) {
        throw std::out_of_range("Output index out of range");
    }
    return outputs_[index];
}

InferenceResult InferenceEngine::run_inference(const ProcessedData& data) {
    float* input1_buf = input_buffers_[0];
    float* input2_buf = input_buffers_[1];
    
    const auto& x_local = data.x_local;
    const size_t outer_size = x_local.size();
    const size_t inner_size = x_local[0].size();
    
    size_t idx = 0;
    for (size_t k = 0; k < inner_size; ++k) {
        for (size_t i = 0; i < outer_size; ++i) {
            input1_buf[idx++] = x_local[i][k];
        }
    }

    const auto& x_global = data.x_global;
    const size_t outer_size2 = x_global.size();
    const size_t inner_size2 = x_global[0].size();
    
    idx = 0;
    for (size_t k = 0; k < inner_size2; ++k) {
        for (size_t i = 0; i < outer_size2; ++i) {
            input2_buf[idx++] = x_global[i][k];
        }
    }

    if (!_infer(input1_buf, input2_buf)) {
        throw std::runtime_error("Inference failed for sample: " + data.file_path);
    }

    const auto& output = _get_output();
    
    return {output[0], data.file_path, data.sample_id};
}
