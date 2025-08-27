#include "async_pipeline.h"
#include <filesystem>
#include <algorithm>
#include <iostream>

AsyncPipeline::AsyncPipeline(const ModelConfig& config,
                           const EvaluationConfig& eval_config,
                           const std::string& data_directory)
    : config_(config), eval_config_(eval_config), data_directory_(data_directory) {
}

AsyncPipeline::~AsyncPipeline() {
    // 确保所有线程都已完成
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
    
    // 为每个NPU核心创建独立的推理引擎实例
    int num_npu_cores = 3;
    inference_engines_.reserve(num_npu_cores);
    for (int i = 0; i < num_npu_cores; ++i) {
        auto engine = std::make_unique<InferenceEngine>(config_.rknn_model_path);
        engine->initialize();
        inference_engines_.push_back(std::move(engine));
        std::cout << "Initialized inference engine " << (i + 1) << " for NPU core " << i << std::endl;
    }
    
    evaluator_ = std::make_unique<PerformanceEvaluator>(eval_config_);
}

std::tuple<float, float, float, float, float> AsyncPipeline::run_evaluation() {
    auto file_list = get_file_list();
    std::cout << "Processing " << file_list.size() << " files..." << std::endl;
    
    // 重置计数器
    processed_count_ = 0;
    inference_count_ = 0;
    stop_preprocessing_ = false;
    stop_inference_ = false;
    
    // 填充文件队列
    populate_file_queue();
    
    // 启动预处理线程（可以根据CPU核数调整）
    int num_preprocess_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2));
    for (int i = 0; i < num_preprocess_threads; ++i) {
        preprocess_threads_.emplace_back(&AsyncPipeline::preprocess_worker, this);
    }
    
    // 启动推理线程（针对RK3588的3个NPU核心优化）
    int num_inference_threads = 3;  // 使用3个线程对应3个NPU核心
    std::cout << "Starting " << num_inference_threads << " inference threads for NPU cores..." << std::endl;
    for (int i = 0; i < num_inference_threads; ++i) {
        inference_threads_.emplace_back(&AsyncPipeline::inference_worker, this, i);
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

void AsyncPipeline::preprocess_worker() {
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

void AsyncPipeline::inference_worker(int worker_id) {
    ProcessedData data;
    
    std::cout << "Inference worker " << worker_id << " started" << std::endl;
    
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

std::vector<std::string> AsyncPipeline::get_file_list() {
    std::vector<std::string> file_list;
    
    for (auto& entry : std::filesystem::directory_iterator(data_directory_)) {
        if (entry.is_regular_file()) {
            file_list.push_back(entry.path().string());
        }
    }
    
    std::sort(file_list.begin(), file_list.end());
    return file_list;
}

void AsyncPipeline::populate_file_queue() {
    auto file_list = get_file_list();
    
    std::cout << "Found " << file_list.size() << " files to process" << std::endl;
    
    for (size_t i = 0; i < file_list.size(); ++i) {
        file_queue_.push(file_list[i]);
    }
    
    file_queue_.set_finished();
}
