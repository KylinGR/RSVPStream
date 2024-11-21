#ifndef RKNN_MODEL_H
#define RKNN_MODEL_H

#include <vector>
#include <iostream>
#include <cstring>
#include "rknn_api.h"

class RKNNModel {
private:
    const char* model_path;  // 模型路径
    rknn_context context;    // RKNN 上下文
    rknn_input_output_num io_num;  // 输入输出数量信息
    int batch_size;          // 批处理大小
    int dim1;                // 输入维度 1
    int dim2;                // 输入维度 2

    // 动态存储推理结果，支持多输出
    std::vector<std::vector<float>> outputs;

public:
    // 构造函数和析构函数
    RKNNModel(const char* model_path, int dim1, int dim2, int batch_size = 1);
    ~RKNNModel();

    // 运行推理测试
    bool test(const float* input_data);

    // 获取指定索引的推理结果
    const std::vector<float>& getOutput(size_t index) const;

private:
    // 初始化模型
    bool initialize();

    // 设置输入
    bool setInput(const float* input_data);

    // 运行推理
    bool runInference();
};

#endif  // RKNN_MODEL_H
