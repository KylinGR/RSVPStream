#pragma once
#include <vector>
#include <string>
#include <tuple>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <filesystem>
#include <optional>

// 线程安全的脑电样本队列，支持阻塞读取
class EEGSampleQueue {
public:
    void push(std::vector<std::vector<float>> sample);
    std::vector<std::vector<float>> pop_blocking();
    size_t size() const;

private:
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<std::vector<std::vector<float>>> queue_;
};

class XGBDIM {
public:
        XGBDIM(int sub_idx,
            int n_cutpoint, int win_len, int chan_xlen, int chan_ylen, int step_x, int step_y,
            int max_N_model, float gstf_weight);
    void get_3Dconv();
    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>> get_data(std::string data_src);
    std::tuple<float, float, float, float, float> test(
        std::string data_dir,
        bool use_queue = false,
        size_t queue_samples = 0,
        EEGSampleQueue* queue = nullptr,
        const std::string& coeff_dir = "",
        const std::string& scale_file = "",
        const std::string& update_flag_path = "");

    // 监听套接字接收推送的脑电信号流，推入队列，推送终止信号后由消费端退出
    std::thread start_socket_receiver(EEGSampleQueue& queue,
                                      int listen_port,
                                      size_t max_samples = 0);

private:
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

    std::string coeff_dir_;
    std::string scale_file_;
    std::string update_flag_path_;
    std::filesystem::file_time_type update_flag_mtime_{};
    bool enable_updates_ = false;

    std::vector<std::vector<float>> Tset_test_global; 
    std::vector<std::vector<float>> NTset_test_global; 
    int K1t;
    int K2t;
    
    std::vector<std::vector<float>> read_data(std::string data_src); 
    std::vector<std::vector<float>> get_3D_cuboids(const std::vector<std::vector<float>>& data);
    void preprocess(std::vector<std::vector<float>>& data);
    std::optional<std::pair<std::string, std::string>> poll_update_flag();
};
