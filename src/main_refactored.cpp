#include "async_pipeline.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>

int main() {
    try {
        // 配置参数
        ModelConfig model_config = {
            .model_path = "/home/hzhy/workspace/csk/RSVPStream/data/model/model.npz",
            .rknn_model_path = "/home/hzhy/workspace/csk/RSVPStream/data/model/optimized_model_v3_1.rknn",
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
        
        std::string data_directory = "/home/hzhy/workspace/csk/RSVPStream/data/egg_data";
        
        // 统计文件数量
        int file_count = 0;
        for (auto& entry : std::filesystem::directory_iterator(data_directory)) {
            if (entry.is_regular_file()) {
                file_count++;
            }
        }
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
