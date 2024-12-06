#pragma once

#include <chrono>
#include <string>

class ModuleProfiler {
 private:
  bool enabled_;  // 是否启用性能分析
  std::chrono::steady_clock::time_point start_time_;  // 起始时间
  std::chrono::steady_clock::time_point end_time_;    // 结束时间
  double total_time_{0.0};  // 累积时间（毫秒）
  size_t run_count_{0};     // 运行次数

 public:
  // 默认构造函数，不启用性能分析
  ModuleProfiler(bool enabled = false) : enabled_(enabled) {}

  // 重置性能统计数据
  void reset() {
    total_time_ = 0.0;
    run_count_ = 0;
  }

  // 开始计时
  void start() {
    if (enabled_) {
      start_time_ = std::chrono::steady_clock::now();
    }
  }

  // 停止计时并更新统计数据
  void stop() {
    if (enabled_) {
      end_time_ = std::chrono::steady_clock::now();
      double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_).count();
      total_time_ += elapsed_time;
      ++run_count_;
    }
  }

  // 获取平均处理时间（毫秒）
  double get_average_time() const {
    if (run_count_ == 0) return 0.0;
    return total_time_ / run_count_;
  }

  // 获取总运行时间（毫秒）
  double get_total_time() const {
    return total_time_;
  }

  // 获取运行次数
  size_t get_run_count() const {
    return run_count_;
  }

  // 获取是否启用性能分析
  bool is_enabled() const {
    return enabled_;
  }

  // 打印性能分析结果
  std::string report(const std::string& module_name = "Module") const {
    if (!enabled_) {
      return module_name + " Profiler: Disabled";
    }

    char buffer[256];
    snprintf(buffer, sizeof(buffer),
             "%s Profiler: Total Time = %.2f ms, Average Time = %.2f ms, Run Count = %zu",
             module_name.c_str(), total_time_, get_average_time(), run_count_);
    return std::string(buffer);
  }
};
