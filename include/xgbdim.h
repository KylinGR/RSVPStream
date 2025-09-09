#pragma once
#include <vector>
#include <string>
#include <tuple>

class XGBDIM {
public:
    XGBDIM(int sub_idx, const std::string& model_path,
           int n_cutpoint, int win_len, int chan_xlen, int chan_ylen, int step_x, int step_y,
           int max_N_model, float gstf_weight);
    void get_3Dconv();
    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>> get_data(std::string data_src);
    std::tuple<float, float, float, float, float> test(std::string data_dir);

private:
    std::string model_path;
    int sub_idx;
    int win_len;
    int chan_xlen;
    int chan_ylen;
    int step_x;
    int step_y;
    int max_N_model;
    float gstf_weight;
    int n_cutpoint;
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
};
