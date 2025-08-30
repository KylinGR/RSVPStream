#include "async_pipeline.h"
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

AsyncPipeline::AsyncPipeline(const ModelConfig& config,
                           const EvaluationConfig& eval_config,
                           const std::string& data_directory)
    : config_(config), eval_config_(eval_config), data_directory_(data_directory) {
}

AsyncPipeline::~AsyncPipeline() {
    for (auto& thread : preprocess_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    for (auto& thread : inference_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void AsyncPipeline::initialize() {
    preprocessor_ = std::make_unique<DataPreprocessor>(config_);
    preprocessor_->initialize();
    
    int num_fpga_devices = _detect_fpga_devices();
    std::cout << "Detected " << num_fpga_devices << " FPGA devices" << std::endl;
    
    fpga_engines_.reserve(num_fpga_devices);
    for (int i = 0; i < num_fpga_devices; ++i) {
        auto engine = std::make_unique<FpgaInferenceEngine>(config_.fpga_config);
        engine->initialize();
        fpga_engines_.push_back(std::move(engine));
        std::cout << "Initialized FPGA inference engine " << (i + 1) << std::endl;
    }
    
    evaluator_ = std::make_unique<PerformanceEvaluator>(eval_config_);
}

std::tuple<float, float, float, float, float> AsyncPipeline::run_evaluation() {
    processed_count_ = 0;
    inference_count_ = 0;
    stop_preprocessing_ = false;
    stop_inference_ = false;
    
    // 设置文件队列
    _setup_file_queue();
    
    // 启动预处理线程（可以根据CPU核数调整）
    int num_preprocess_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2));
    for (int i = 0; i < num_preprocess_threads; ++i) {
        preprocess_threads_.emplace_back(&AsyncPipeline::_preprocess_worker, this);
    }
    
    // 启动FPGA推理线程
    int num_inference_threads = static_cast<int>(fpga_engines_.size());
    std::cout << "Starting " << num_inference_threads << " FPGA inference threads..." << std::endl;
    for (int i = 0; i < num_inference_threads; ++i) {
        inference_threads_.emplace_back(&AsyncPipeline::_fpga_inference_worker, this, i);
    }
    
    // 等待所有预处理完成
    for (auto& thread : preprocess_threads_) {
        thread.join();
    }
    preprocess_threads_.clear();
    
    std::cout << "All preprocessing completed. Total processed: " << processed_count_ << std::endl;
    
    // 标记预处理完成
    processed_queue_.set_finished();
    
    // 等待所有推理完成
    for (auto& thread : inference_threads_) {
        thread.join();
    }
    inference_threads_.clear();
    
    std::cout << "All inference completed. Total inferences: " << inference_count_ << std::endl;
    
    // 收集结果
    std::vector<InferenceResult> results;
    InferenceResult result;
    while (result_queue_.try_pop(result)) {
        results.push_back(result);
    }
    
    std::cout << "Processed " << results.size() << " samples" << std::endl;
    
    // 评估性能
    return evaluator_->evaluate(results);
}

void AsyncPipeline::_preprocess_worker() {
    std::string file_path;
    
    while (!stop_preprocessing_) {
        if (file_queue_.try_pop(file_path)) {
            try {
                int current_sample_id = processed_count_.load();  // 使用原子计数器
                auto processed_data = preprocessor_->process_file(file_path, current_sample_id);
                processed_queue_.push(processed_data);
                processed_count_++;
                
                if (processed_count_ % 100 == 0) {
                    std::cout << "Preprocessed: " << processed_count_ << " files" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error processing file " << file_path << ": " << e.what() << std::endl;
            }
        } else if (file_queue_.is_finished()) {
            break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void AsyncPipeline::_fpga_inference_worker(int worker_id) {
    ProcessedData data;
    
    std::cout << "FPGA inference worker " << worker_id << " started" << std::endl;
    
    while (!stop_inference_) {
        if (processed_queue_.wait_and_pop(data)) {
            try {
                // 使用对应的FPGA推理引擎实例
                auto result = fpga_engines_[worker_id]->run_inference(data);
                result_queue_.push(result);
                inference_count_++;
                
                if (inference_count_ % 100 == 0) {
                    std::cout << "FPGA inference completed: " << inference_count_ 
                              << " samples (worker " << worker_id << ")" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in FPGA inference worker " << worker_id 
                          << " for " << data.file_path << ": " << e.what() << std::endl;
            }
        } else {
            // 队列已完成且为空，退出循环
            break;
        }
    }
    
    std::cout << "FPGA inference worker " << worker_id << " finished" << std::endl;
}

void AsyncPipeline::_setup_file_queue() {
    // 获取文件列表
    std::vector<std::string> file_list;
    for (auto& entry : std::filesystem::directory_iterator(data_directory_)) {
        if (entry.is_regular_file()) {
            file_list.push_back(entry.path().string());
        }
    }
    
    // 排序文件列表以确保处理顺序一致
    std::sort(file_list.begin(), file_list.end());
    
    std::cout << "Processing " << file_list.size() << " files..." << std::endl;
    
    // 填充文件队列
    for (const auto& file_path : file_list) {
        file_queue_.push(file_path);
    }
    
    // 标记文件队列已完成填充
    file_queue_.set_finished();
}

int AsyncPipeline::_detect_fpga_devices() {
    // 方法1: 检查XDMA设备文件
    std::vector<std::string> xdma_device_paths = {
        "/dev/xdma0_h2c_0",  // 主机到FPGA
        "/dev/xdma0_c2h_0",  // FPGA到主机
        "/dev/xdma0_user",   // 用户空间设备
        "/dev/xdma0_control" // 控制设备
    };
    
    int fpga_devices_found = 0;
    bool xdma_detected = false;
    
    for (const auto& path : xdma_device_paths) {
        if (std::filesystem::exists(path)) {
            xdma_detected = true;
            std::cout << "XDMA device detected: " << path << std::endl;
        }
    }
    
    if (xdma_detected) {
        fpga_devices_found = 1;  // 假设有一个FPGA设备
        
        // 方法2: 检查是否有多个FPGA设备
        for (int i = 1; i < 4; ++i) {  // 检查最多4个设备
            std::string h2c_path = "/dev/xdma" + std::to_string(i) + "_h2c_0";
            std::string c2h_path = "/dev/xdma" + std::to_string(i) + "_c2h_0";
            
            if (std::filesystem::exists(h2c_path) && std::filesystem::exists(c2h_path)) {
                std::cout << "Additional FPGA device detected: xdma" << i << std::endl;
                fpga_devices_found++;
            }
        }
    }
    
    // 方法3: 检查PCIe设备信息
    if (!xdma_detected) {
        std::ifstream lspci_check("/proc/bus/pci/devices");
        if (lspci_check.is_open()) {
            std::string line;
            while (std::getline(lspci_check, line)) {
                // 查找Xilinx设备ID (简化检查)
                if (line.find("10ee") != std::string::npos) {  // Xilinx vendor ID
                    std::cout << "Xilinx PCIe device detected in /proc/bus/pci/devices" << std::endl;
                    fpga_devices_found = 1;
                    break;
                }
            }
            lspci_check.close();
        }
    }
    
    if (fpga_devices_found > 0) {
        std::cout << "Total FPGA devices detected: " << fpga_devices_found << std::endl;
        return fpga_devices_found;
    }
    
    // 默认情况：如果检测失败，使用单个FPGA设备
    std::cout << "Could not detect FPGA devices automatically, defaulting to 1 device" << std::endl;
    std::cout << "Note: FPGA simulation mode will be used" << std::endl;
    
    return 1;  // 默认使用1个FPGA设备
}
