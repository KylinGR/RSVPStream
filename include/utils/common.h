#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>
#include "opencv2/opencv.hpp"  // 如果涉及图像数据传输，可以使用 OpenCV 类型





// ==================== 全局宏定义 ====================

// 禁用拷贝和赋值的宏
#define DISABLE_COPY_AND_ASSIGN(ClassName) \
  ClassName(const ClassName&) = delete;    \
  ClassName& operator=(const ClassName&) = delete;

// 打印调试信息的宏（仅在 DEBUG 模式下生效）
#ifdef DEBUG
#define DEBUG_LOG(msg) std::cout << "[DEBUG] " << msg << std::endl
#else
#define DEBUG_LOG(msg) ((void)0)
#endif

// 获取数组大小的宏
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

// 安全删除指针
#define SAFE_DELETE(ptr) \
  {                      \
    delete ptr;          \
    ptr = nullptr;       \
  }

// ==================== 全局常量定义 ====================

constexpr int DEFAULT_BUFFER_SIZE = 1024;  // 默认缓冲区大小
constexpr int DEFAULT_TIMEOUT_MS = 5000;  // 默认超时时间 (ms)

// ==================== 全局类型定义 ====================

/**
 * 时间戳类型，使用 std::chrono 封装
 */
using Timestamp = std::chrono::time_point<std::chrono::steady_clock>;

// ==================== 工具函数声明 ====================

/**
 * 获取当前时间的字符串，格式为 "YYYY-MM-DD HH:MM:SS"
 * @return 当前时间的字符串
 */
std::string get_current_time_str();

/**
 * 将数字转换为带固定宽度的字符串（左侧填充 0）
 * @param num 要转换的数字
 * @param width 宽度
 * @return 转换后的字符串
 */
template <typename T>
std::string to_fixed_width_string(T num, int width);

/**
 * 字符串是否以指定前缀开头
 * @param str 输入字符串
 * @param prefix 前缀
 * @return 是否以前缀开头
 */
bool starts_with(const std::string& str, const std::string& prefix);

/**
 * 字符串是否以指定后缀结尾
 * @param str 输入字符串
 * @param suffix 后缀
 * @return 是否以后缀结尾
 */
bool ends_with(const std::string& str, const std::string& suffix);

/**
 * 将字符串转换为小写
 * @param str 输入字符串
 * @return 转换后的字符串
 */
std::string to_lower(const std::string& str);

/**
 * 将字符串转换为大写
 * @param str 输入字符串
 * @return 转换后的字符串
 */
std::string to_upper(const std::string& str);

/**
 * @brief 定义 Package 数据包，作为模块间的统一传输载体
 */
class Package {
 public:
  // 用于存储多种数据类型的通用类型定义
  using DataType = std::variant<int, float, double, std::string, cv::Mat, std::vector<uint8_t>>;

 private:
  // 包的唯一标识 ID（可用于追踪数据流）
  std::string package_id_;

  // 数据存储容器（键值对形式，支持多种类型）
  std::unordered_map<std::string, DataType> data_;

 public:
  // 默认构造函数
  Package() = default;

  // 构造函数，支持指定包 ID
  explicit Package(const std::string& id) : package_id_(id) {}

  // 获取包的唯一 ID
  const std::string& get_id() const { return package_id_; }

  // 设置包的唯一 ID
  void set_id(const std::string& id) { package_id_ = id; }

  // 添加数据到包中
  template <typename T>
  void add_data(const std::string& key, const T& value) {
    data_[key] = value;
  }

  // 检查是否存在某个键
  bool has_key(const std::string& key) const {
    return data_.find(key) != data_.end();
  }

  // 获取数据（带类型检查）
  template <typename T>
  T get_data(const std::string& key) const {
    auto it = data_.find(key);
    if (it == data_.end()) {
      throw std::runtime_error("Key not found in package: " + key);
    }
    return std::get<T>(it->second);
  }

  // 获取数据（无异常处理，返回 nullptr）
  template <typename T>
  std::optional<T> try_get_data(const std::string& key) const {
    auto it = data_.find(key);
    if (it != data_.end()) {
      if (auto value = std::get_if<T>(&it->second)) {
        return *value;
      }
    }
    return std::nullopt;
  }

  // 移除某个键值对
  void remove_data(const std::string& key) {
    data_.erase(key);
  }

  // 清空包中的所有数据
  void clear() {
    data_.clear();
  }

  // 打印包内容（调试用）
  void debug_print() const {
    std::cout << "Package ID: " << package_id_ << std::endl;
    for (const auto& [key, value] : data_) {
      std::cout << "  Key: " << key << ", Value Type: " << value.index() << std::endl;
    }
  }
};