#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>

#include "framework/module.h"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

class Source : public Module<Package *> {
 public:
  // 构造函数
  explicit Source(int max_queue_length, bool enable_profiler, int cpu_id, int npu_id)
      : Module<Package *>(0, enable_profiler, cpu_id, npu_id), 
        max_queue_length_(max_queue_length), 
        exit_flag_(false) {}

  // 析构函数
  virtual ~Source() = default;

  // 禁止拷贝和移动
  Source(const Source &) = delete;
  Source(Source &&) = delete;
  Source &operator=(const Source &) = delete;
  Source &operator=(Source &&) = delete;

  // 运行函数
  void run() final {
    set_cpu_affinity("Source");  // 设置线程的 CPU 亲和性

    while (!exit_flag_) {
      try {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 检查队列长度是否超限
        if (is_queue_full()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 等待队列有空闲空间
          continue;
        }

        // 创建一个新的数据包
        std::shared_ptr<Package> package = std::make_shared<Package>();

        // 调用子类实现的具体处理逻辑
        if (!process(package.get())) {
          MLOG_ERROR("Source failed to process package");
          continue;
        }

        // 将数据推入输出队列
        push_output(package);

        // 性能分析
        if (profiler_.is_enabled()) {
          auto end_time = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
          profiler_.add_profile(duration);
        }
      } catch (const std::exception &e) {
        MLOG_ERROR("Exception in Source: %s", e.what());
      }
    }

    MLOG_INFO("Source has exited.");
  }

  // 子类需要实现的核心处理逻辑
  virtual bool process(Package *package) = 0;

  // 安全退出函数
  void exit() { exit_flag_ = true; }

 private:
  int max_queue_length_;         // 队列的最大长度
  std::atomic<bool> exit_flag_;  // 退出标志

  // 检查队列是否已满
  bool is_queue_full() {
    std::lock_guard<std::mutex> lock(*output_mutex_);
    return output_ptr_ && output_ptr_->size() >= static_cast<size_t>(max_queue_length_);
  }
};
