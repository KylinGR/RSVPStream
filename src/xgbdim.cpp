#include <cmath>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include "xgbdim.h"
#include "utils.h"
#include "fpga_runner.h"

XGBDIM::XGBDIM( int sub_idx, const std::string& model_path,
               int n_cutpoint, int win_len, int chan_xlen, int chan_ylen, int step_x, int step_y,
               int max_N_model, float gstf_weight)
    : sub_idx(sub_idx), model_path(model_path), n_cutpoint(n_cutpoint), win_len(win_len), chan_xlen(chan_xlen), chan_ylen(chan_ylen), step_x(step_x), step_y(step_y),
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
    int Nx_channel = channel_loc[0].size(); //9
    std::vector<int> channel_xst;
    for (int i = 0; i <= Nx_channel - chan_xlen; i += step_x) {
        channel_xst.push_back(i);
    } //0,3,6

    int Ny_channel = channel_loc.size(); //6
    std::vector<int> channel_yst;
    for (int i = 0; i <= Ny_channel - chan_ylen; i += step_y) {
        channel_yst.push_back(i);
    }//0,3

    for (int idx_y : channel_yst) {
        for (int idx_x : channel_xst) {
            std::vector<int> cup;
            // 提取子矩阵并转置
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

    N_win = window_st.size(); //时间维度上的滑动次数
    N_chanwin = channel_conv.size(); //电极维度上的滑动次数
    N_conv = N_win * N_chanwin;
    N_model = std::min(N_conv, max_N_model);
}

//read_data函数读取数据
std::vector<std::vector<float>> XGBDIM::read_data(std::string data_src) {
    std::vector<std::vector<float>> data = load_npz_2d_array_float(data_src, "data");  // data shape: 60x250

    return data;
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
        // 计算第 i 行的总和
        for (size_t j = 0; j < data[i].size(); ++j) {
            sum += data[i][j];
        }
        float mean = sum / data[i].size();
        // 减去均值
        for (size_t j = 0; j < data[i].size(); ++j) {
            data[i][j] -= mean;
        }
    }
}

// 获取3D卷积的立方体数据
std::vector<std::vector<float>> XGBDIM::get_3D_cuboids(
           const std::vector<std::vector<float>>& data) {
    // 初始化 Tset 和 NTset
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

//X_test和X_test_global是模型输入，对应原python文件中的两个同名三维向量
// std::vector<float> XGBDIM::rknn_runner(const std::vector<std::vector<float>>& x_local, const std::vector<std::vector<float>>& x_global, const std::string& rknn_model_path) {
//     // 初始化模型
//     ENSEMBLEModel model(rknn_model_path);

//     // 准备输入数据
//     const auto& input1_dims = model.get_input1_dims();
//     const auto& input2_dims = model.get_input2_dims();
//     // std::cout << "input processed !" << std::endl;
//     std::vector<float> input1(input1_dims.batch * input1_dims.height * input1_dims.width * input1_dims.channel);
//     std::vector<float> input2(input2_dims.batch * input2_dims.height * input2_dims.width * input2_dims.channel);


//     size_t index = 0;
//     for (size_t k = 0; k < x_local[0].size(); ++k) { // 遍历最内层
//         for (const auto& outer : x_local) { // 遍历最外层
//             input1[index++] = outer[k];
//             }
//         }

//     index = 0;
//     for (size_t k = 0; k < x_global[0].size(); ++k) { // 遍历最内层
//         for (const auto& outer : x_global) { // 遍历最外层
//             input2[index++] = outer[k];
//             }
//         }

//     // 执行推理
//     // auto start = std::chrono::high_resolution_clock::now();
//     if (!model.infer(input1.data(), input2.data())) {
//         throw std::runtime_error("Inference failed");
//     }
//     // auto end = std::chrono::high_resolution_clock::now();

//     // // 输出推理时间
//     // std::cout << "Inference time: " 
//     //             << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
//     //             << " ms\n";

//     // 获取输出结果
//     const auto& output = model.get_output();
//     // std::cout << "output : " << output[1156]<<' '<< output[1155] <<' '<< output[1154] << std::endl;
//     return output;
// }

void XGBDIM::load_model() {
    Model_order = load_npz_array_int(model_path, "model_order");
}
std::tuple<float, float, float, float, float> XGBDIM::test(std::string data_dir) {
    get_3Dconv();
    FPGAProcessor processor;

    std::vector<std::string> file_list;
    for (auto &entry : std::filesystem::directory_iterator(data_dir)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            file_list.push_back(file_path);
        }
    }
    std::sort(file_list.begin(), file_list.end()); // 对文件名进行排序
    std::vector<float> s_all;
    for (auto file_path : file_list) {
        // auto start = std::chrono::high_resolution_clock::now();
        auto [x_local, x_global] = get_data(file_path);
        
        // int N_local_model = 299;//299

        // // 根据 Model_order 对 x_local 进行索引操作
        // for (auto& row : x_local) {
        //     std::vector<float> new_row;
        //     for (int k = 0; k < N_local_model; ++k) {
        //         new_row.push_back(row[Model_order[k]]);
        //     }
        //     // 用新的行替换原来的行
        //     row = new_row;
        // }


        // auto end = std::chrono::high_resolution_clock::now();
        // std::cout << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000 << " seconds taken for reordering." << std::endl;

        float s_mean = processor.fpga_runner(x_local, x_global);
        s_all.push_back(s_mean); 
        // 输出调试信息
        std::cout << "Output Val: [" << s_mean << "]" << std::endl;

    }
    int n_positive = 61;//61
    int n_negative = 1096;//1096
    int Ns= n_positive + n_negative;
    std::vector<int> label_test(Ns, 1);
    std::fill(label_test.begin() + n_positive, label_test.end(), 0);

    // 计算预测结果
    std::vector<int> y_predicted_final(Ns, 0);
    for (int n = 0; n < Ns; ++n) {
        y_predicted_final[n] = (s_all[n] >= 0.5);
    }

    // 计算准确率、TPR、FPR、BA
    int n_tp = 0;
    int n_fp = 0;
    for (int n = 0; n < Ns; ++n) {
        if (y_predicted_final[n] == 1 && label_test[n] == 1) n_tp++;
        if (y_predicted_final[n] == 1 && label_test[n] == 0) n_fp++;
    }

    float tpr = static_cast<float>(n_tp) / n_positive;
    float fpr = static_cast<float>(n_fp) / n_negative;
    float ba = (tpr + (1 - fpr)) / 2;
    float acc = static_cast<float>(n_tp + (n_negative - n_fp)) / Ns;

    // 计算AUC
    std::vector<float> fpr_1, tpr_1;
    for (float threshold = 0.0; threshold <= 1.0; threshold += 0.01) {
        int tp = 0, fp = 0;
        for (int n = 0; n < Ns; ++n) {
            if (s_all[n] >= threshold && label_test[n] == 1) tp++;
            if (s_all[n] >= threshold && label_test[n] == 0) fp++;
        }
        fpr_1.push_back(static_cast<float>(fp) / n_negative);
        tpr_1.push_back(static_cast<float>(tp) / n_positive);
    }
    float auc = 0.0;
    for (size_t i = 1; i < fpr_1.size(); ++i) {
        auc += (fpr_1[i] - fpr_1[i - 1]) * (tpr_1[i] + tpr_1[i - 1]) / 2;
    }
    auc = std::abs(auc);

    return {ba, acc, tpr, fpr, auc};

}


