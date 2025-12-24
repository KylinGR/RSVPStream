#include <cmath>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <map>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "xgbdim.h"
#include "utils.h"
#include "fpga_runner.h"

// ---------------- EEGSampleQueue ----------------
void EEGSampleQueue::push(std::vector<std::vector<float>> sample) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push(std::move(sample));
    }
    cv_.notify_one();
}

std::vector<std::vector<float>> EEGSampleQueue::pop_blocking() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return !queue_.empty(); });
    auto sample = std::move(queue_.front());
    queue_.pop();
    return sample;
}

size_t EEGSampleQueue::size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.size();
}

namespace {
bool recv_all(int fd, void* buffer, size_t len) {
    char* buf = static_cast<char*>(buffer);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = ::recv(fd, buf, remaining, 0);
        if (n <= 0) {
            return false;
        }
        buf += n;
        remaining -= static_cast<size_t>(n);
    }
    return true;
}
}

std::thread XGBDIM::start_socket_receiver(EEGSampleQueue& queue,
                                          int listen_port,
                                          size_t max_samples) {
    return std::thread([&queue, listen_port, max_samples]() {
        int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            std::cerr << "Failed to create socket" << std::endl;
            return;
        }

        int opt = 1;
        ::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(listen_port));

        if (::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::cerr << "Failed to bind socket" << std::endl;
            ::close(server_fd);
            return;
        }
        if (::listen(server_fd, 1) < 0) {
            std::cerr << "Failed to listen on socket" << std::endl;
            ::close(server_fd);
            return;
        }

        std::cout << "Listening on port " << listen_port << " for EEG stream..." << std::endl;
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = ::accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            std::cerr << "Failed to accept connection" << std::endl;
            ::close(server_fd);
            return;
        }

        size_t received = 0;
        while (true) {
            int32_t rows_net = 0;
            int32_t cols_net = 0;
            if (!recv_all(client_fd, &rows_net, sizeof(rows_net))) break;
            if (!recv_all(client_fd, &cols_net, sizeof(cols_net))) break;

            int32_t rows = ntohl(rows_net);
            int32_t cols = ntohl(cols_net);

            if (rows <= 0 || cols <= 0) {
                std::cout << "Received termination signal." << std::endl;
                break;
            }

            std::vector<float> flat(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
            if (!recv_all(client_fd, flat.data(), flat.size() * sizeof(float))) {
                std::cerr << "Failed to receive sample payload" << std::endl;
                break;
            }

            std::vector<std::vector<float>> sample(rows, std::vector<float>(cols, 0.0f));
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    sample[r][c] = flat[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)];
                }
            }

            queue.push(std::move(sample));
            ++received;
            if (max_samples > 0 && received >= max_samples) {
                std::cout << "Reached max_samples limit on receiver, stopping." << std::endl;
                break;
            }
        }

        // 终止信号：推送空样本到队列
        queue.push({});

        ::close(client_fd);
        ::close(server_fd);
    });
}

XGBDIM::XGBDIM(int sub_idx,
                             int n_cutpoint, int win_len, int chan_xlen, int chan_ylen, int step_x, int step_y,
                             int max_N_model, float gstf_weight)
        : sub_idx(sub_idx), n_cutpoint(n_cutpoint), win_len(win_len), chan_xlen(chan_xlen), chan_ylen(chan_ylen), step_x(step_x), step_y(step_y),
            max_N_model(max_N_model), gstf_weight(gstf_weight) {
    channel_loc = {{1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {10, 11, 12, 13, 14, 15, 16, 17, 18},
                   {19, 20, 21, 22, 23, 24, 25, 26, 27},
                   {28, 29, 30, 31, 32, 33, 34, 35, 36},
                   {37, 38, 39, 40, 41, 42, 43, 44, 45},
                   {46, 47, 48, 49, 50, 51, 52, 53, 54}};   //6*9
    channel = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 58, 54, 60, 55, 56, 57};
    chan_len = chan_xlen * chan_ylen;
    T_local = win_len * chan_len;
}

void XGBDIM::get_3Dconv() {
    int Nx_channel = channel_loc[0].size();
    std::vector<int> channel_xst;
    for (int i = 0; i <= Nx_channel - chan_xlen; i += step_x) {
        channel_xst.push_back(i);
    }

    int Ny_channel = channel_loc.size();
    std::vector<int> channel_yst;
    for (int i = 0; i <= Ny_channel - chan_ylen; i += step_y) {
        channel_yst.push_back(i);
    }

    for (int idx_y : channel_yst) {
        for (int idx_x : channel_xst) {
            std::vector<int> cup;
            for (int j = 0; j < chan_xlen; ++j) {
                for (int i = 0; i < chan_ylen; ++i) {
                    cup.push_back(channel_loc[idx_y + i][idx_x + j]);
                }
            }
            channel_conv.push_back(cup);
        }
    }

    int step_win = win_len / 2;
    for (int i = 1; i <= 250 - win_len + 1; i += step_win) {
        window_st.push_back(i);
    }
    window_ov.resize(window_st.size());
    for (size_t i = 0; i < window_st.size(); ++i) {
        window_ov[i] = window_st[i] + win_len - 1;
    }

    N_win = window_st.size();
    N_chanwin = channel_conv.size();
    N_conv = N_win * N_chanwin;
    N_model = std::min(N_conv, max_N_model);
}

std::vector<std::vector<float>> XGBDIM::read_data(std::string data_src) {
    return load_npz_2d_array_float(data_src, "data");
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>> XGBDIM::get_data(std::string data_src) {
    auto x_global = read_data(data_src);
    preprocess(x_global);
    auto x_local = get_3D_cuboids(x_global);
    return {x_local, x_global};
}

void XGBDIM::preprocess(std::vector<std::vector<float>>& data) {
    for (size_t i = 0; i < data.size(); ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < data[i].size(); ++j) {
            sum += data[i][j];
        }
        float mean = sum / data[i].size();
        for (size_t j = 0; j < data[i].size(); ++j) {
            data[i][j] -= mean;
        }
    }
}

std::vector<std::vector<float>> XGBDIM::get_3D_cuboids(const std::vector<std::vector<float>>& data) {
    std::vector<std::vector<float>> Tset(T_local, std::vector<float>(N_chanwin * N_win, 0.0f));
    int idx_conv = -1;
    for (int idx_chan = 0; idx_chan < N_chanwin; ++idx_chan) {
        for (int idx_win = 0; idx_win < N_win; ++idx_win) {
            ++idx_conv;
            const auto& chan_indices = channel_conv[idx_chan];
            int start = window_st[idx_win] - 1;
            int end = window_ov[idx_win];
            int cup_index = 0;

            for (int i = start; i < end; ++i) {
                for (int chan : chan_indices) {
                    int chan_idx = channel[chan - 1] - 1;
                    if (cup_index < T_local) {
                        Tset[cup_index][idx_conv] = data[chan_idx][i];
                        ++cup_index;
                    }
                }
            }
        }
    }
    return Tset;
}

std::tuple<float, float, float, float, float> XGBDIM::test(std::string data_dir,
                                                          bool use_queue,
                                                          size_t queue_samples,
                                                          EEGSampleQueue* queue) {
    get_3Dconv();
    FPGAProcessor processor;

    std::vector<float> s_all;
    std::map<long long, int> loop_cost_hist;

    auto process_sample = [&](const std::vector<std::vector<float>>& x_global) {
        auto loop_start = std::chrono::high_resolution_clock::now();

        auto x_global_copy = x_global;
        preprocess(x_global_copy);
        auto x_local = get_3D_cuboids(x_global_copy);
        auto get_data_end = std::chrono::high_resolution_clock::now();
        auto get_data_duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_data_end - loop_start).count();
        std::cout << "Get data cost " << get_data_duration << " ms" << std::endl;

        float s_mean = processor.fpga_runner(x_local, x_global_copy);
        s_all.push_back(s_mean);
        auto loop_end = std::chrono::high_resolution_clock::now();
        auto loop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
        ++loop_cost_hist[loop_duration];

        std::cout << "Output Val: [" << s_mean << "]" << std::endl;
        std::cout << "FPGA loop cost " << loop_duration << " ms" << std::endl;
        std::cout << std::endl;
    };

    if (use_queue) {
        if (queue == nullptr) {
            throw std::invalid_argument("test: queue mode requires a valid EEGSampleQueue pointer");
        }
        std::cout << "模型初始化完毕，开始等待接收数据..." << std::endl;
        size_t consumed = 0;
        while (true) {
            auto sample = queue->pop_blocking();
            if (sample.empty()) { // 终止信号
                break;
            }
            process_sample(sample);
            ++consumed;
            if (queue_samples > 0 && consumed >= queue_samples) {
                break;
            }
        }
        // 在线推理不做最终指标评估，直接返回占位值
        return {0.f, 0.f, 0.f, 0.f, 0.f};
    } else {
        std::vector<std::string> file_list;
        for (auto &entry : std::filesystem::directory_iterator(data_dir)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                file_list.push_back(file_path);
            }
        }
        std::sort(file_list.begin(), file_list.end());
        for (const auto& file_path : file_list) {
            auto x_global = read_data(file_path);
            process_sample(x_global);
        }
    }

    std::cout << "===== FPGA循环耗时统计 =====" << std::endl;
    std::cout << "总样本数: " << s_all.size() << std::endl;
    std::cout << "耗时分布(单位ms):" << std::endl;
    for (const auto& [cost, cnt] : loop_cost_hist) {
        std::cout << "  " << cost << " ms: " << cnt << " 个" << std::endl;
    }

    if (s_all.empty()) {
        return {0.f, 0.f, 0.f, 0.f, 0.f};
    }

    size_t Ns = s_all.size();
    size_t n_positive = std::min<size_t>(61, Ns);
    size_t n_negative = Ns - n_positive;
    std::vector<int> label_test(Ns, 1);
    if (n_positive < Ns) {
        std::fill(label_test.begin() + static_cast<long>(n_positive), label_test.end(), 0);
    }

    std::vector<int> y_predicted_final(Ns, 0);
    for (size_t n = 0; n < Ns; ++n) {
        y_predicted_final[n] = (s_all[n] >= 0.5f);
    }

    size_t n_tp = 0;
    size_t n_fp = 0;
    for (size_t n = 0; n < Ns; ++n) {
        if (y_predicted_final[n] == 1 && label_test[n] == 1) n_tp++;
        if (y_predicted_final[n] == 1 && label_test[n] == 0) n_fp++;
    }

    float tpr = n_positive ? static_cast<float>(n_tp) / static_cast<float>(n_positive) : 0.f;
    float fpr = n_negative ? static_cast<float>(n_fp) / static_cast<float>(n_negative) : 0.f;
    float ba = (tpr + (1 - fpr)) / 2;
    float acc = static_cast<float>(n_tp + (n_negative - n_fp)) / static_cast<float>(Ns);

    std::vector<float> fpr_1, tpr_1;
    for (float threshold = 0.0f; threshold <= 1.0f; threshold += 0.01f) {
        size_t tp = 0, fp = 0;
        for (size_t n = 0; n < Ns; ++n) {
            if (s_all[n] >= threshold && label_test[n] == 1) tp++;
            if (s_all[n] >= threshold && label_test[n] == 0) fp++;
        }
        fpr_1.push_back(n_negative ? static_cast<float>(fp) / static_cast<float>(n_negative) : 0.f);
        tpr_1.push_back(n_positive ? static_cast<float>(tp) / static_cast<float>(n_positive) : 0.f);
    }
    float auc = 0.0f;
    for (size_t i = 1; i < fpr_1.size(); ++i) {
        auc += (fpr_1[i] - fpr_1[i - 1]) * (tpr_1[i] + tpr_1[i - 1]) / 2;
    }
    auc = std::abs(auc);

    return {ba, acc, tpr, fpr, auc};
}
