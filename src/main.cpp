#include "async_pipeline.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_order_path> <rknn_model_path> <data_directory>" << std::endl;
    std::cout << "  model_order_path: Path to the model order .npy file" << std::endl;
    std::cout << "  rknn_model_path:  Path to the RKNN model .rknn file" << std::endl;
    std::cout << "  data_directory:   Path to the directory containing EEG data files" << std::endl;
}

bool file_exists(const std::string& path) {
    namespace fs = std::filesystem;
    return fs::exists(path) && fs::is_regular_file(path);
}

bool directory_exists(const std::string& path) {
    namespace fs = std::filesystem;
    return fs::exists(path) && fs::is_directory(path);
}

int count_files_in_directory(const std::string& path) {
    namespace fs = std::filesystem;
    int count = 0;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            count++;
        }
    }
    return count;
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 4) {
            std::cerr << "Error: Invalid number of arguments." << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        
        std::string model_order_path = argv[1];
        std::string rknn_model_path = argv[2];
        std::string data_directory = argv[3];
        
        // 检查文件和目录是否存在
        if (!file_exists(model_order_path)) {
            std::cerr << "Error: Model order file does not exist: " << model_order_path << std::endl;
            return 1;
        }
        
        if (!file_exists(rknn_model_path)) {
            std::cerr << "Error: RKNN model file does not exist: " << rknn_model_path << std::endl;
            return 1;
        }
        
        if (!directory_exists(data_directory)) {
            std::cerr << "Error: Data directory does not exist or is not a directory: " << data_directory << std::endl;
            return 1;
        }
        
        // 配置参数
        ModelConfig model_config = {
            .model_order_path = model_order_path,
            .rknn_model_path = rknn_model_path,
            .win_len = 6,
            .chan_xlen = 3,
            .chan_ylen = 3,
            .step_x = 3,
            .step_y = 3,
            .max_N_model = 299,
            .gstf_weight = 0.3,
            .N_local_model = 299
        };
        
        EvaluationConfig eval_config = {
            .n_positive = 61,
            .n_negative = 1096,
            .threshold = 0.5
        };
        
        // 统计文件数量
        int file_count = count_files_in_directory(data_directory);
        std::cout << "Total files to process: " << file_count << std::endl;
        
        // 创建异步管道
        AsyncPipeline pipeline(model_config, eval_config, data_directory);
        
        std::cout << "Initializing async pipeline..." << std::endl;
        auto init_start = std::chrono::high_resolution_clock::now();
        pipeline.initialize();
        auto init_end = std::chrono::high_resolution_clock::now();
        
        std::cout << "Initialization completed in " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count()
                  << " ms" << std::endl;
        
        std::cout << "Starting evaluation..." << std::endl;
        auto eval_start = std::chrono::high_resolution_clock::now();
        auto [ba, acc, tpr, fpr, auc] = pipeline.run_evaluation();
        auto eval_end = std::chrono::high_resolution_clock::now();
        
        auto eval_duration = std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start);
        double total_seconds = static_cast<double>(eval_duration.count()) / 1000.0;
        double avg_time_per_sample = total_seconds / file_count;
        
        // 输出结果
        std::cout << std::setprecision(4) << std::fixed 
                  << "Results - BA: " << ba 
                  << ", ACC: " << acc 
                  << ", TPR: " << tpr 
                  << ", FPR: " << fpr 
                  << ", AUC: " << auc << std::endl;
        
        std::cout << "=== Multi-threaded Performance Statistics ===" << std::endl;
        std::cout << "Total evaluation time: " << total_seconds << " seconds" << std::endl;
        std::cout << "Processed samples: " << file_count << std::endl;
        std::cout << "Average time per sample: " << std::setprecision(3) << std::fixed 
                  << avg_time_per_sample << " seconds" << std::endl;
        std::cout << "Throughput: " << std::setprecision(2) << std::fixed 
                  << (1.0 / avg_time_per_sample) << " samples/second" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
