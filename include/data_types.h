#pragma once
#include <vector>
#include <string>

// 数据结构定义
struct ProcessedData {
    std::vector<std::vector<float>> x_local;
    std::vector<std::vector<float>> x_global;
    std::string file_path;
    int sample_id;
};

struct InferenceResult {
    float score;
    std::string file_path;
    int sample_id;
};

struct EvaluationConfig {
    int n_positive = 61;
    int n_negative = 1096;
    float threshold = 0.5;
};

struct ModelConfig {
    std::string model_order_path;  // 独立的model_order.npy文件路径
    std::string rknn_model_path;
    int win_len;
    int chan_xlen;
    int chan_ylen;
    int step_x;
    int step_y;
    int max_N_model;
    float gstf_weight;
    int N_local_model = 299;
};
