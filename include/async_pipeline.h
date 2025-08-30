#pragma once
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <memory>
#include "data_types.h"
#include "thread_safe_queue.h"
#include "data_preprocessor.h"
#include "fpga_inference_engine.h"
#include "performance_evaluator.h"

class AsyncPipeline {
private:
    ModelConfig config_;
    EvaluationConfig eval_config_;
    std::string data_directory_;
    
    // 线程安全队列
    ThreadSafeQueue<std::string> file_queue_;
    ThreadSafeQueue<ProcessedData> processed_queue_;
    ThreadSafeQueue<InferenceResult> result_queue_;
    
    // 工作线程
    std::vector<std::thread> preprocess_threads_;
    std::vector<std::thread> inference_threads_;
    
    // 处理器
    std::unique_ptr<DataPreprocessor> preprocessor_;
    std::vector<std::unique_ptr<FpgaInferenceEngine>> fpga_engines_;  // FPGA推理引擎
    std::unique_ptr<PerformanceEvaluator> evaluator_;
    std::mutex fpga_engine_mutex_;  // 保护FPGA引擎访问的互斥锁
    
    // 控制变量
    std::atomic<bool> stop_preprocessing_{false};
    std::atomic<bool> stop_inference_{false};
    std::atomic<int> processed_count_{0};
    std::atomic<int> inference_count_{0};

public:
    AsyncPipeline(const ModelConfig& config, 
                  const EvaluationConfig& eval_config,
                  const std::string& data_directory);
    
    ~AsyncPipeline();
    
    void initialize();
    std::tuple<float, float, float, float, float> run_evaluation();
    
private:
    void _preprocess_worker();
    void _fpga_inference_worker(int worker_id);  // FPGA推理工作线程
    void _setup_file_queue();
    int _detect_fpga_devices();  // 检测FPGA设备数量
};
