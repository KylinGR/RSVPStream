#include "xgbdim.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <atomic>
#include <cstdlib>

namespace {
void print_usage() {
    std::cout << "Usage: ./rsvp_rknn [--mode file|queue] [--data_dir <dir>]"
              << " [--dat_dir <coeff_dir>] [--scale_file <scale_hex>]"
              << " [--max-samples <N>] [--listen-port <port>]"
              << " [--update-flag <flag_path>]" << std::endl;
}
}

int main(int argc, char** argv) {
    std::string mode = "file";              // file | queue
    std::string data_dir = "/home/hzhy/RSVPStream/data/egg_data";
    std::string coeff_dir = "/home/hzhy/RSVPStream/data/dat";
    std::string scale_file;
    std::string update_flag_path; // queue-only
    size_t max_samples = 10;
    int listen_port = 5001;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) {
            if (i + 1 >= argc) {
                std::cerr << name << " requires a value" << std::endl;
                print_usage();
                std::exit(1);
            }
        };

        if (arg == "--mode") {
            need_value("--mode");
            mode = argv[++i];
        } else if (arg == "--data_dir") {
            need_value("--data_dir");
            data_dir = argv[++i];
        } else if (arg == "--dat_dir") {
            need_value("--dat_dir");
            coeff_dir = argv[++i];
        } else if (arg == "--scale_file") {
            need_value("--scale_file");
            scale_file = argv[++i];
        } else if (arg == "--update-flag") {
            need_value("--update-flag");
            update_flag_path = argv[++i];
        } else if (arg == "--max-samples") {    // 设为 0 则表示不限制，直到发送端发出终止信号才退出
            need_value("--max-samples");
            max_samples = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--listen-port") {
            need_value("--listen-port");
            listen_port = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else {
            std::cerr << "Unknown arg: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }

    ::setenv("RSVP_DAT_DIR", coeff_dir.c_str(), 1);

    XGBDIM xgb(1, 50, 6, 3, 3, 3, 3, 299, 0.3);
    std::cout << "Starting program..." << std::endl;

    std::tuple<float, float, float, float, float> metrics;

    if (mode == "queue") {
        EEGSampleQueue queue;
        // 后续修改接收方式请替换start_socket_receiver函数，这里用socket本机通信模拟
        auto receiver = xgb.start_socket_receiver(queue, listen_port, max_samples);
        metrics = xgb.test(data_dir, true, max_samples, &queue, coeff_dir, scale_file, update_flag_path);
        receiver.join();
    } else if (mode == "file") {
        if (!update_flag_path.empty()) {
            std::cerr << "--update-flag is ignored in file mode." << std::endl;
        }
        metrics = xgb.test(data_dir, false, 0, nullptr, coeff_dir, scale_file, "");
    } else {
        std::cerr << "Unsupported mode: " << mode << ", expected file or queue" << std::endl;
        return 1;
    }

    auto [ba, acc, tpr, fpr, auc] = metrics;
    std::cout << std::setprecision(4) << std::fixed
              << "BA: " << ba << ", ACC: " << acc << ", TPR: " << tpr
              << ", FPR: " << fpr << ", AUC: " << auc << std::endl;

    return 0;
}

