#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <exception>

#include "framework/module.h"

class Preprocessor : public Module<Package *> {
 public:
  // 构造函数
  explicit Preprocessor(int pre_module_nums, bool enable_profiler, int cpu_id, int npu_id)
      : Module<Package *>(pre_module_nums, enable_profiler, cpu_id, npu_id), exit_flag_(false) {}

  // 析构函数
  virtual ~Preprocessor() = default;

  // 禁止拷贝和移动
  Preprocessor(const Preprocessor &) = delete;
  Preprocessor(Preprocessor &&) = delete;
  Preprocessor &operator=(const Preprocessor &) = delete;
  Preprocessor &operator=(Preprocessor &&) = delete;

  // 运行函数
  void run() final {
    set_cpu_affinity("Preprocessor");  // 设置 CPU 亲和性

    while (!exit_flag_) {
      try {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 从输入队列中获取数据
        auto input_package = pop_input();
        if (!input_package) {
          continue;  // 如果输入为空，继续等待
        }

        // 调用子类实现的处理逻辑
        if (!process(input_package.get())) {
          MLOG_ERROR("Preprocessor failed to process package");
          continue;
        }

        // 将处理结果推入输出队列
        push_output(input_package);

        // 性能分析
        if (profiler_.is_enabled()) {
          auto end_time = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
          profiler_.add_profile(duration);
        }
      } catch (const std::exception &e) {
        MLOG_ERROR("Exception in Preprocessor: %s", e.what());
      }
    }

    MLOG_INFO("Preprocessor has exited.");
  }

  // 子类必须实现的处理逻辑
  virtual bool process(Package *package) = 0;

  // 安全退出函数
  void exit() { exit_flag_ = true; }

 private:
  std::atomic<bool> exit_flag_;  // 退出标志
};

