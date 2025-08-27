#pragma once
#include <vector>
#include <string>
#include <memory>
#include "data_types.h"

class DataPreprocessor {
private:
    ModelConfig config_;
    std::vector<std::vector<int>> channel_loc_;
    std::vector<std::vector<int>> channel_conv_;
    std::vector<int> channel_;
    std::vector<int> window_st_;
    std::vector<int> window_ov_;
    std::vector<int64_t> model_order_;
    
    int chan_len_;
    int T_local_;
    int N_win_;
    int N_chanwin_;
    int N_conv_;
    int N_model_;

public:
    explicit DataPreprocessor(const ModelConfig& config);
    
    void initialize();
    ProcessedData process_file(const std::string& file_path, int sample_id);
    
private:
    void setup_3d_convolution();
    void load_model_order();
    std::vector<std::vector<float>> read_data(const std::string& data_src);
    void preprocess_data(std::vector<std::vector<float>>& data);
    std::vector<std::vector<float>> get_3d_cuboids(const std::vector<std::vector<float>>& data);
    void reorder_local_data(std::vector<std::vector<float>>& x_local);
};
