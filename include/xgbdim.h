#pragma once
#include <vector>
#include <string>
#include <tuple>

class XGBDIM {
public:
    XGBDIM(const std::string& model_path, const std::string& rknn_model_path,
           int win_len, int chan_xlen, int chan_ylen, int step_x, int step_y,
           int max_N_model, float gstf_weight);
    void get_3Dconv();
    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>> get_data(std::string data_src);
    std::tuple<float, float, float, float, float> test();

private:
    std::string model_path;
    std::string rknn_model_path;
    int win_len;
    int chan_xlen;
    int chan_ylen;
    int step_x;
    int step_y;
    int max_N_model;
    float gstf_weight;
    int chan_len;
    int T_local;

    std::vector<std::vector<int>> channel_loc;
    std::vector<std::vector<int>> channel_conv;
    std::vector<int> channel;
    std::vector<int> window_st;
    std::vector<int> window_ov;
    int N_win;
    int N_chanwin;
    int N_conv;
    int N_model;

    std::vector<std::vector<float>> Tset_test_global; 
    std::vector<std::vector<float>> NTset_test_global; 
    int K1t;
    int K2t;
    
    std::vector<int64_t> Model_order;

    void load_model();
    std::vector<std::vector<float>> read_data(std::string data_src); 
    std::vector<std::vector<float>> get_3D_cuboids(const std::vector<std::vector<float>>& data);
    void preprocess(std::vector<std::vector<float>>& data);
    std::vector<float> rknn_runner(const std::vector<std::vector<float>>& x_local, const std::vector<std::vector<float>>& x_global, const std::string& rknn_model_path);
};
