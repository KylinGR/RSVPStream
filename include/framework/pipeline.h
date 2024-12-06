#pragma once

#include <numeric>
#include <memory>
#include <mutex>         // NOLINT
#include <queue>
#include <thread>        // NOLINT
#include <vector>
#include <condition_variable>

#include "framework/module.h"
#include "framework/postprocessor.h"
#include "framework/preprocessor.h"
#include "framework/runner.h"
#include "framework/sink.h"
#include "framework/source.h"
#include "framework/tracker.h"

// Pipeline 类
class Pipeline {
 protected:
  // 资源管理：使用智能指针替代裸指针
  std::vector<std::shared_ptr<int>> flags_;                                 // 阶段标志
  std::vector<std::shared_ptr<std::mutex>> mutexes_;                        // 每个阶段的互斥锁
  std::vector<std::shared_ptr<std::condition_variable>> cvs_;               // 每个阶段的条件变量
  std::vector<std::shared_ptr<std::queue<std::shared_ptr<Package>>>> buffers_;  // 每个阶段的队列
  int stage_num_;                                                           // 阶段数量

 public:
  // 构造函数
  explicit Pipeline(int stage_num) : stage_num_(stage_num) {
    initialize_resources(stage_num_);
  }

  // 析构函数：智能指针会自动管理资源，无需手动释放
  ~Pipeline() = default;

  // 运行函数：支持并行运行模式
  void run(const std::vector<std::vector<Module<Package*>*>>& modules, bool enable_profile);

  // 顺序运行函数
  void seq_run(const std::vector<std::vector<Module<Package*>*>>& modules, bool enable_profile);

 private:
  // 初始化资源
  void initialize_resources(int stage_num) {
    for (int i = 0; i < stage_num - 1; ++i) {
      flags_.emplace_back(std::make_shared<int>(0));                          // 标志初始化为0
      mutexes_.emplace_back(std::make_shared<std::mutex>());                  // 每个阶段一个互斥锁
      cvs_.emplace_back(std::make_shared<std::condition_variable>());         // 每个阶段一个条件变量
      buffers_.emplace_back(std::make_shared<std::queue<std::shared_ptr<Package>>>());  // 每个阶段一个队列
    }
  }

  // 辅助函数：模块运行逻辑
  void run_module(Module<Package*>* module, int stage_index, bool enable_profile);

  // 辅助函数：处理单个模块的输入输出
  std::shared_ptr<Package> pop_from_buffer(int stage_index);
  void push_to_buffer(int stage_index, std::shared_ptr<Package> package);
};
