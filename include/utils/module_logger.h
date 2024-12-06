#pragma once

#include <iostream>
#include <fstream>
#include <mutex>
#include <string>
#include <sstream>
#include <ctime>
#include <iomanip>

enum class LogLevel {
  DEBUG,
  INFO,
  WARN,
  ERROR
};

class ModuleLogger {
 private:
  LogLevel log_level_;              // 当前日志级别
  std::ostream* output_stream_;     // 日志输出流（默认是 std::cout）
  std::ofstream file_stream_;       // 如果日志写入文件，则使用此流
  std::mutex log_mutex_;            // 用于线程安全的日志操作

  // 获取当前时间的字符串
  std::string get_current_time() const {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time = *std::localtime(&now_time);

    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S");
    return oss.str();
  }

  // 将日志级别转换为字符串
  std::string log_level_to_string(LogLevel level) const {
    switch (level) {
      case LogLevel::DEBUG: return "DEBUG";
      case LogLevel::INFO: return "INFO";
      case LogLevel::WARN: return "WARN";
      case LogLevel::ERROR: return "ERROR";
      default: return "UNKNOWN";
    }
  }

 public:
  // 构造函数，默认输出到标准输出，日志级别为 INFO
  explicit ModuleLogger(LogLevel level = LogLevel::INFO, const std::string& file_path = "")
      : log_level_(level), output_stream_(&std::cout) {
    if (!file_path.empty()) {
      file_stream_.open(file_path, std::ios::out | std::ios::app);
      if (file_stream_.is_open()) {
        output_stream_ = &file_stream_;
      } else {
        std::cerr << "Failed to open log file: " << file_path << std::endl;
      }
    }
  }

  // 析构函数，关闭文件流
  ~ModuleLogger() {
    if (file_stream_.is_open()) {
      file_stream_.close();
    }
  }

  // 设置日志级别
  void set_log_level(LogLevel level) {
    log_level_ = level;
  }

  // 获取当前日志级别
  LogLevel get_log_level() const {
    return log_level_;
  }

  // 写日志（通用模板）
  template <typename... Args>
  void log(LogLevel level, const std::string& module_name, Args... args) {
    if (level < log_level_) {
      return;  // 如果日志级别低于当前设置的日志级别，忽略
    }

    std::lock_guard<std::mutex> lock(log_mutex_);

    // 生成日志内容
    std::ostringstream oss;
    oss << "[" << get_current_time() << "] "
        << "[" << log_level_to_string(level) << "] "
        << "[" << module_name << "] ";
    (oss << ... << args);  // 使用 C++17 的折叠表达式处理变参
    oss << std::endl;

    // 写入日志到输出流
    (*output_stream_) << oss.str();
    output_stream_->flush();
  }

  // 便捷的日志接口
  template <typename... Args>
  void debug(const std::string& module_name, Args... args) {
    log(LogLevel::DEBUG, module_name, args...);
  }

  template <typename... Args>
  void info(const std::string& module_name, Args... args) {
    log(LogLevel::INFO, module_name, args...);
  }

  template <typename... Args>
  void warn(const std::string& module_name, Args... args) {
    log(LogLevel::WARN, module_name, args...);
  }

  template <typename... Args>
  void error(const std::string& module_name, Args... args) {
    log(LogLevel::ERROR, module_name, args...);
  }
};
