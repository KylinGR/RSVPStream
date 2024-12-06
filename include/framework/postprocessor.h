#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <memory>

#include "framework/module.h"

class Postprocessor : public Module<Package *> {
 public:
  // 构造函数
  explicit Postprocessor(int pre_module_nums, bool enable_profiler, int cpu_id, int npu_id)
      : Module<Package *>(pre_module_nums, enable_profiler, cpu_id, npu_id), exit_flag_(false) {}

  // 析构函数
  virtual ~Postprocessor() { clear_buffer(); }

  // 禁止拷贝和移动
  Postprocessor(const Postprocessor &) = delete;
  Postprocessor(Postprocessor &&) = delete;
  Postprocessor &operator=(const Postprocessor &) = delete;
  Postprocessor &operator=(Postprocessor &&) = delete;

  // 运行函数
  void run() final {
    set_cpu_affinity("Postprocessor");  // 设置 CPU 亲和性
    while (!exit_flag_) {
      try {
        // 从输入队列中获取数据
        auto input_package = pop_input();
        if (!input_package) {
          continue;  // 如果输入为空，继续等待
        }

        // 调用子类实现的处理逻辑
        if (!process(input_package.get())) {
          MLOG_ERROR("Postprocessor failed to process package");
          continue;
        }

        // 将处理结果存入缓冲区
        add_to_buffer(cur_idx_, std::move(input_package));
        cur_idx_++;
      } catch (const std::exception &e) {
        MLOG_ERROR("Exception in Postprocessor: %s", e.what());
      }
    }
    MLOG_INFO("Postprocessor has exited.");
  }

  // 子类必须实现的处理逻辑
  virtual bool process(Package *package) = 0;

  // 安全退出函数
  void exit() { exit_flag_ = true; }

 private:
  std::unordered_map<int, std::shared_ptr<Package>> buffer_;  // 缓冲区
  std::mutex buffer_mutex_;                                  // 缓冲区互斥锁
  std::atomic<int> cur_idx_{0};                              // 当前缓冲区索引
  std::atomic<bool> exit_flag_;                              // 退出标志

  // 清空缓冲区
  void clear_buffer() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    buffer_.clear();
  }

 protected:
  // 添加数据到缓冲区
  void add_to_buffer(int index, std::shared_ptr<Package> package) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    buffer_[index] = std::move(package);
  }

  // 获取缓冲区中的数据（可扩展函数）
  std::shared_ptr<Package> get_from_buffer(int index) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    auto it = buffer_.find(index);
    if (it != buffer_.end()) {
      return it->second;
    }
    return nullptr;
  }
};
