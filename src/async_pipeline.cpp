#include "async_pipeline.h"
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <pthread.h>
#include <sched.h>

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
    
    int num_npu_cores = _detect_npu_cores();
    std::cout << "Detected " << num_npu_cores << " NPU cores" << std::endl;
    
    inference_engines_.reserve(num_npu_cores);
    for (int i = 0; i < num_npu_cores; ++i) {
        auto engine = std::make_unique<InferenceEngine>(config_.rknn_model_path, i);
        engine->initialize();
        inference_engines_.push_back(std::move(engine));
        std::cout << "Initialized inference engine " << (i + 1) << " for NPU core " << i << std::endl;
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
    
    // 启动预处理线程（使用A55小核，适合CPU密集型预处理任务）
    int num_preprocess_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2));
    std::cout << "Starting " << num_preprocess_threads << " preprocessing threads..." << std::endl;
    for (int i = 0; i < num_preprocess_threads; ++i) {
        preprocess_threads_.emplace_back(&AsyncPipeline::_preprocess_worker, this, i);
    }
    
    // 启动推理线程（使用检测到的NPU核心数量）
    int num_inference_threads = static_cast<int>(inference_engines_.size());  // 使用已初始化的推理引擎数量
    std::cout << "Starting " << num_inference_threads << " inference threads for NPU cores..." << std::endl;
    for (int i = 0; i < num_inference_threads; ++i) {
        inference_threads_.emplace_back(&AsyncPipeline::_inference_worker, this, i);
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

void AsyncPipeline::_preprocess_worker(int worker_id) {
    // 绑定到A55小核 (CPU 0-3)，适合CPU密集型预处理任务
    int cpu_core = worker_id % 4;  // 使用CPU 0-3
    _set_cpu_affinity(cpu_core);
    
    std::string file_path;
    
    std::cout << "Preprocess worker " << worker_id << " started on CPU " << cpu_core << std::endl;
    
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
    
    std::cout << "Preprocess worker " << worker_id << " finished" << std::endl;
}

void AsyncPipeline::_inference_worker(int worker_id) {
    // 绑定到A76大核 (CPU 4-7)，适合推理任务
    int cpu_core = 4 + (worker_id % 4);  // 使用CPU 4-7
    _set_cpu_affinity(cpu_core);
    
    // 为推理线程设置较高优先级
    // _set_thread_priority(10);  // 可选：设置更高优先级
    
    ProcessedData data;
    
    std::cout << "Inference worker " << worker_id << " started on CPU " << cpu_core << std::endl;
    
    while (!stop_inference_) {
        if (processed_queue_.wait_and_pop(data)) {
            try {
                // 使用对应的推理引擎实例
                auto result = inference_engines_[worker_id]->run_inference(data);
                result_queue_.push(result);
                inference_count_++;
                
                if (inference_count_ % 100 == 0) {
                    std::cout << "Inference completed: " << inference_count_ 
                              << " samples (worker " << worker_id << ")" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in inference worker " << worker_id 
                          << " for " << data.file_path << ": " << e.what() << std::endl;
            }
        } else {
            // 队列已完成且为空，退出循环
            break;
        }
    }
    
    std::cout << "Inference worker " << worker_id << " finished" << std::endl;
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

int AsyncPipeline::_detect_npu_cores() {
    // 方法1: 尝试读取RK3588 NPU设备信息
    std::vector<std::string> npu_device_paths = {
        "/sys/class/devfreq/fdab0000.npu/device",
        "/sys/devices/platform/fdab0000.npu",
        "/dev/rknpu_mem",
        "/proc/device-tree/npu",
        "/sys/kernel/debug/rknpu"
    };
    
    // 检查NPU设备是否存在
    bool npu_detected = false;
    for (const auto& path : npu_device_paths) {
        if (std::filesystem::exists(path)) {
            npu_detected = true;
            std::cout << "NPU device detected at: " << path << std::endl;
            break;
        }
    }
    
    if (npu_detected) {
        // 方法2: 检查CPU信息确认是RK3588
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        bool is_rk3588 = false;
        
        if (cpuinfo.is_open()) {
            while (std::getline(cpuinfo, line)) {
                if (line.find("rk3588") != std::string::npos || 
                    line.find("RK3588") != std::string::npos) {
                    is_rk3588 = true;
                    break;
                }
            }
            cpuinfo.close();
        }
        
        // 方法3: 检查设备树信息
        if (!is_rk3588) {
            std::ifstream devicetree("/proc/device-tree/compatible");
            if (devicetree.is_open()) {
                std::string content;
                std::getline(devicetree, content);
                if (content.find("rk3588") != std::string::npos) {
                    is_rk3588 = true;
                }
                devicetree.close();
            }
        }
        
        if (is_rk3588) {
            std::cout << "RK3588 platform detected, using 3 NPU cores" << std::endl;
            return 3;  // RK3588有3个NPU核心
        }
    }
    
    // 方法4: 尝试通过RKNN运行时查询（如果可能）
    // 这里可以添加RKNN API查询逻辑，但需要包含相应的头文件
    
    // 默认情况：如果检测失败，使用保守的单核心设置
    std::cout << "Could not detect NPU cores automatically, defaulting to 1 core" << std::endl;
    std::cout << "You can manually set the number of cores if needed" << std::endl;
    
    return 1;  // 默认使用1个核心
}

void AsyncPipeline::_set_cpu_affinity(int cpu_core_id) {
    if (cpu_core_id < 0 || cpu_core_id >= 8) {
        std::cerr << "Warning: Invalid CPU core ID " << cpu_core_id << std::endl;
        return;
    }
    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core_id, &cpuset);
    
    pthread_t current_thread = pthread_self();
    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    
    if (result == 0) {
        std::cout << "Thread bound to CPU core " << cpu_core_id << std::endl;
    } else {
        std::cerr << "Warning: Failed to set CPU affinity to core " << cpu_core_id 
                  << ", error: " << result << std::endl;
    }
}

void AsyncPipeline::_set_thread_priority(int priority) {
    struct sched_param param;
    param.sched_priority = priority;
    
    int result = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    if (result == 0) {
        std::cout << "Thread priority set to " << priority << std::endl;
    } else {
        std::cerr << "Warning: Failed to set thread priority to " << priority 
                  << ", error: " << result << std::endl;
    }
}
