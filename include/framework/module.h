#pragma once

#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

#include "opencv2/opencv.hpp"
#include "utils/common.h"
#include "utils/module_logger.h"
#include "utils/module_profiler.h"

// 禁止拷贝和移动
class NonCopyable {
 protected:
  NonCopyable() = default;
  ~NonCopyable() = default;

 private:
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable(NonCopyable&&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
  NonCopyable& operator=(NonCopyable&&) = delete;
};

// 模块基类
template <class T1>
class Module : public NonCopyable {
 protected:
  int cpu_id_{-1};  // 默认不绑定到特定 CPU
  int npu_id_{-1};  // 默认不绑定到特定 NPU

  int pre_module_nums_{0};  // 前置模块数量
  size_t cnt_{0};           // 处理计数器

  int* input_flag_{nullptr};   // 输入标志
  int* output_flag_{nullptr};  // 输出标志

  std::mutex* input_mutex_{nullptr};
  std::mutex* output_mutex_{nullptr};

  std::condition_variable* input_cv_{nullptr};
  std::condition_variable* output_cv_{nullptr};

  std::queue<T1>* input_ptr_{nullptr};
  std::queue<T1>* output_ptr_{nullptr};

  ModuleProfiler profiler_;  // 性能分析器

 public:
  // 构造函数
  Module() = default;

  explicit Module(int pre_module_nums, bool enable_profiler = false, int cpu_id = -1, int npu_id = -1)
      : cpu_id_(cpu_id),
        npu_id_(npu_id),
        pre_module_nums_(pre_module_nums),
        profiler_(enable_profiler) {}

  virtual ~Module() = default;

  // 纯虚函数，处理主逻辑
  virtual void run() override = 0;

  // 子类需要实现的处理逻辑
  virtual bool process(Package* package) override = 0;

  // 设置输入/输出标志
  void set_input_flag(int* input_flag) { input_flag_ = input_flag; }
  void set_output_flag(int* output_flag) { output_flag_ = output_flag; }

  // 设置输入/输出互斥锁
  void set_input_mutex(std::mutex* input_mutex) { input_mutex_ = input_mutex; }
  void set_output_mutex(std::mutex* output_mutex) { output_mutex_ = output_mutex; }

  // 设置输入/输出条件变量
  void set_input_cv(std::condition_variable* input_cv) { input_cv_ = input_cv; }
  void set_output_cv(std::condition_variable* output_cv) { output_cv_ = output_cv; }

  // 设置输入/输出队列
  void set_input_ptr(std::queue<T1>* input_ptr) { input_ptr_ = input_ptr; }
  void set_output_ptr(std::queue<T1>* output_ptr) { output_ptr_ = output_ptr; }

  // 获取模块的 CPU 和 NPU ID
  int get_cpu_id() const { return cpu_id_; }
  int get_npu_id() const { return npu_id_; }

  // 获取处理计数
  size_t get_cnt() const { return cnt_; }

  // 获取性能分析结果
  double get_profile() { return profiler_.get_profile(); }

  // 设置线程 CPU 亲和性
  inline void set_cpu_affinity(const char* process_name) {
    if (cpu_id_ < 0) {
      MLOG_DEBUG("%s not bound to a specific CPU", process_name);
      return;
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id_, &mask);

    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) != 0) {
      MLOG_ERROR("Set thread affinity failed for %s", process_name);
    } else {
      MLOG_DEBUG("Bind %s process to CPU %d", process_name, cpu_id_);
    }
  }

  // 简化的线程安全队列操作
  std::optional<T1> pop_input() {
    std::unique_lock<std::mutex> lock(*input_mutex_);
    input_cv_->wait(lock, [this]() { return !input_ptr_->empty(); });
    if (input_ptr_->empty()) return std::nullopt;  // 如果队列仍为空，返回空对象
    T1 data = std::move(input_ptr_->front());
    input_ptr_->pop();
    return data;
  }

  void push_output(const T1& data) {
    {
      std::lock_guard<std::mutex> lock(*output_mutex_);
      output_ptr_->push(data);
    }
    output_cv_->notify_one();
  }
};
