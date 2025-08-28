#include "data_preprocessor.h"
#include "utils.h"
#include <algorithm>
#include <stdexcept>

DataPreprocessor::DataPreprocessor(const ModelConfig& config) 
    : config_(config) {
    // 初始化通道位置
    channel_loc_ = {{1, 2, 3, 4, 5, 6, 7, 8, 9},
                    {10, 11, 12, 13, 14, 15, 16, 17, 18},
                    {19, 20, 21, 22, 23, 24, 25, 26, 27},
                    {28, 29, 30, 31, 32, 33, 34, 35, 36},
                    {37, 38, 39, 40, 41, 42, 43, 44, 45},
                    {46, 47, 48, 49, 50, 51, 52, 53, 54}};
    
    channel_ = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 58, 54, 60, 55, 56, 57};
    
    chan_len_ = config_.chan_xlen * config_.chan_ylen;
    T_local_ = config_.win_len * chan_len_;
}

void DataPreprocessor::initialize() {
    _load_model_order();
    _setup_3d_convolution();
}

void DataPreprocessor::_setup_3d_convolution() {
    int Nx_channel = channel_loc_[0].size();
    std::vector<int> channel_xst;
    for (int i = 0; i <= Nx_channel - config_.chan_xlen; i += config_.step_x) {
        channel_xst.push_back(i);
    }

    int Ny_channel = channel_loc_.size();
    std::vector<int> channel_yst;
    for (int i = 0; i <= Ny_channel - config_.chan_ylen; i += config_.step_y) {
        channel_yst.push_back(i);
    }

    for (int idx_y : channel_yst) {
        for (int idx_x : channel_xst) {
            std::vector<int> cup;
            for (int j = 0; j < config_.chan_xlen; ++j) {
                for (int i = 0; i < config_.chan_ylen; ++i) {
                    cup.push_back(channel_loc_[idx_y + i][idx_x + j]);
                }
            }
            channel_conv_.push_back(cup);
        }
    }

    int step_win = config_.win_len / 2;
    for (int i = 1; i <= 250 - config_.win_len + 1; i += step_win) {
        window_st_.push_back(i);
    }
    window_ov_.resize(window_st_.size());
    for (size_t i = 0; i < window_st_.size(); ++i) {
        window_ov_[i] = window_st_[i] + config_.win_len - 1;
    }

    N_win_ = window_st_.size();
    N_chanwin_ = channel_conv_.size();
    N_conv_ = N_win_ * N_chanwin_;
    N_model_ = std::min(N_conv_, config_.max_N_model);
}

void DataPreprocessor::_load_model_order() {
    model_order_ = load_npz_array_int(config_.model_path, "model_order");
}

ProcessedData DataPreprocessor::process_file(const std::string& file_path, int sample_id) {
    // 使用优化的数据处理流程
    std::vector<float> data_flat;
    size_t rows, cols;
    _read_data(file_path, data_flat, rows, cols);
    _preprocess_data(data_flat, rows, cols);
    
    // 直接从连续内存格式生成3D cuboids，避免格式转换
    auto x_local = _get_3d_cuboids(data_flat, rows, cols);
    _reorder_local_data(x_local);
    
    // 转换x_global为兼容格式（仅在需要时）
    std::vector<std::vector<float>> x_global(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            x_global[i][j] = data_flat[i * cols + j];
        }
    }
    
    return {x_local, x_global, file_path, sample_id};
}

void DataPreprocessor::_read_data(const std::string& data_src, std::vector<float>& data, size_t& rows, size_t& cols) {
    data = load_npz_2d_array_float(data_src, "data", rows, cols);
}

void DataPreprocessor::_preprocess_data(std::vector<float>& data, size_t rows, size_t cols) {
    // 对每一行进行均值中心化
    for (size_t i = 0; i < rows; ++i) {
        const size_t row_start = i * cols;
        const size_t row_end = row_start + cols;
        
        // 计算均值 - 使用更高效的累积方式
        float sum = 0.0f;
        for (size_t j = row_start; j < row_end; ++j) {
            sum += data[j];
        }
        const float mean = sum / cols;
        
        // 原地减去均值
        for (size_t j = row_start; j < row_end; ++j) {
            data[j] -= mean;
        }
    }
}

std::vector<std::vector<float>> DataPreprocessor::_get_3d_cuboids(
    const std::vector<float>& data, size_t rows, size_t cols) {
    std::vector<std::vector<float>> Tset(T_local_, std::vector<float>(N_chanwin_ * N_win_, 0.0f));
    
    int idx_conv = -1;
    for (int idx_chan = 0; idx_chan < N_chanwin_; ++idx_chan) {
        for (int idx_win = 0; idx_win < N_win_; ++idx_win) {
            ++idx_conv;
            const auto& chan_indices = channel_conv_[idx_chan];
            int start = window_st_[idx_win] - 1;
            int end = window_ov_[idx_win];
            int cup_index = 0;

            for (int i = start; i < end; ++i) {
                for (int chan : chan_indices) {
                    int chan_idx = channel_[chan - 1] - 1;
                    if (cup_index < T_local_ && chan_idx < static_cast<int>(rows) && i < static_cast<int>(cols)) {
                        // 直接访问连续内存：data[row * cols + col]
                        Tset[cup_index][idx_conv] = data[chan_idx * cols + i];
                        ++cup_index;
                    }
                }
            }
        }
    }
    return Tset;
}

void DataPreprocessor::_reorder_local_data(std::vector<std::vector<float>>& x_local) {
    for (auto& row : x_local) {
        std::vector<float> new_row;
        for (int k = 0; k < config_.N_local_model; ++k) {
            new_row.push_back(row[model_order_[k]]);
        }
        row = new_row;
    }
}
