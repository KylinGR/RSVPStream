#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>

#include "framework/module.h"

class Sink : public Module<Package *> {
 public:
  // 构造函数
  explicit Sink(int pre_module_nums, bool enable_profiler, int cpu_id, int npu_id)
      : Module<Package *>(pre_module_nums, enable_profiler, cpu_id, npu_id), exit_flag_(false) {}

  // 析构函数
  virtual ~Sink() = default;

  // 禁止拷贝和移动
  Sink(const Sink &) = delete;
  Sink(Sink &&) = delete;
  Sink &operator=(const Sink &) = delete;
  Sink &operator=(Sink &&) = delete;

  // 运行函数
  void run() final {
    set_cpu_affinity("Sink");  // 设置 CPU 亲和性

    while (!exit_flag_) {
      try {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 从输入队列中获取数据
        auto input_package = pop_input();
        if (!input_package) {
          continue;  // 如果输入为空，继续等待
        }

        // 处理数据并输出结果
        if (!process(input_package.get())) {
          MLOG_ERROR("Sink failed to process package");
          continue;
        }

        // 性能分析
        if (profiler_.is_enabled()) {
          auto end_time = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
          profiler_.add_profile(duration);
        }
      } catch (const std::exception &e) {
        MLOG_ERROR("Exception in Sink: %s", e.what());
      }
    }

    MLOG_INFO("Sink has exited.");
  }

  // 子类需要实现的核心处理逻辑
  virtual bool process(Package *package) = 0;

  // 安全退出函数
  void exit() { exit_flag_ = true; }

 private:
  std::atomic<bool> exit_flag_;  // 退出标志
};
