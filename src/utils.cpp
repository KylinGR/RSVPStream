#include "utils.h"
#include <cnpy.h>
#include <stdexcept>
#include <cstdint>  // 包含 int64_t 类型
#include <limits>
#include <cmath>    // 包含 std::round
#include <utility>  // 包含 std::pair
#include <algorithm> // 包含 std::clamp

//动态量化函数
std::pair<std::vector<std::vector<int16_t>>, float> dynamic_quantize_tensor(const std::vector<std::vector<float>>& tensor) {
    constexpr int16_t Q_MIN = std::numeric_limits<int16_t>::min();  // -32768
    constexpr int16_t Q_MAX = std::numeric_limits<int16_t>::max();  // 32767
    constexpr float MIN_CLAMP = 1e-8f;  // 防止除以0
    
    if (tensor.empty()) {
        return {std::vector<std::vector<int16_t>>(), 0.0f};
    }
    
    // 找到所有元素中绝对值的最大值
    float max_val = 0.0f;
    for (const auto& row : tensor) {
        for (const float& val : row) {
            max_val = std::max(max_val, std::abs(val));
        }
    }
    
    // 计算缩放因子，防止除以0
    float scale = static_cast<float>(Q_MAX) / std::max(max_val, MIN_CLAMP);
    
    // 量化数据
    std::vector<std::vector<int16_t>> quantized;
    quantized.reserve(tensor.size());
    
    for (const auto& row : tensor) {
        std::vector<int16_t> quantized_row;
        quantized_row.reserve(row.size());
        
        for (const float& val : row) {
            float scaled = val * scale;
            int32_t rounded = static_cast<int32_t>(std::round(scaled));
            int16_t clamped = static_cast<int16_t>(
                std::clamp(rounded, static_cast<int32_t>(Q_MIN), static_cast<int32_t>(Q_MAX))
            );
            quantized_row.push_back(clamped);
        }
        
        quantized.push_back(std::move(quantized_row));
    }
    
    return {quantized, scale};
}

//动态量化函数(返回置换后的矩阵)
std::pair<std::vector<std::vector<int16_t>>, float> dynamic_quantize_tensor_T(const std::vector<std::vector<float>>& tensor) {
    constexpr int16_t Q_MIN = std::numeric_limits<int16_t>::min();  // -32768
    constexpr int16_t Q_MAX = std::numeric_limits<int16_t>::max();  // 32767
    constexpr float MIN_CLAMP = 1e-8f;  // 防止除以0
    
    if (tensor.empty()) {
        return {std::vector<std::vector<int16_t>>(), 0.0f};
    }
    
    // 找到所有元素中绝对值的最大值
    float max_val = 0.0f;
    for (const auto& row : tensor) {
        for (const float& val : row) {
            max_val = std::max(max_val, std::abs(val));
        }
    }
    
    // 计算缩放因子，防止除以0
    float scale = static_cast<float>(Q_MAX) / std::max(max_val, MIN_CLAMP);
    
    // 量化数据并转置维度
    if (tensor.empty() || tensor[0].empty()) {
        return {std::vector<std::vector<int16_t>>(), scale};
    }
    
    size_t rows = tensor.size();      // 原始行数
    size_t cols = tensor[0].size();   // 原始列数
    
    // 创建转置后的量化矩阵 (cols x rows)
    std::vector<std::vector<int16_t>> quantized(cols, std::vector<int16_t>(rows));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float scaled = tensor[i][j] * scale;
            int32_t rounded = static_cast<int32_t>(std::round(scaled));
            int16_t clamped = static_cast<int16_t>(
                std::clamp(rounded, static_cast<int32_t>(Q_MIN), static_cast<int32_t>(Q_MAX))
            );
            quantized[j][i] = clamped;  // 转置：原来的[i][j]变成[j][i]
        }
    }
    
    return {quantized, scale};
}

std::vector<int64_t> load_npz_array_int(const std::string& file_path, const std::string& array_name) {
    cnpy::npz_t npz = cnpy::npz_load(file_path);
    if (npz.find(array_name) == npz.end()) {
        throw std::runtime_error("Array " + array_name + " not found in " + file_path);
    }
    auto arr = npz[array_name];
    std::vector<int64_t> result(arr.shape[0]);
    for (size_t i = 0; i < arr.shape[0]; ++i) {
        result[i] = arr.data<int64_t>()[i];
    }
    return result;
}

std::vector<std::vector<std::vector<float>>> load_npz_3d_array_float(const std::string& file_path, const std::string& array_name) {
    // 加载 .npz 文件
    cnpy::npz_t npz = cnpy::npz_load(file_path);
    
    // 检查数组是否存在
    if (npz.find(array_name) == npz.end()) {
        throw std::runtime_error("Array " + array_name + " not found in " + file_path);
    }
    
    // 获取数组
    auto arr = npz[array_name];
    
    // 创建结果容器
    std::vector<std::vector<std::vector<float>>> result(
        arr.shape[0], 
        std::vector<std::vector<float>>(
            arr.shape[1], 
            std::vector<float>(arr.shape[2], 0.0f)
        )
    );
    
    // 根据数据类型读取并转换为 float
    if (arr.word_size == sizeof(float)) {
        // 数据是 float32，直接读取为 float
        const float* data = arr.data<float>();
        for (size_t i = 0; i < arr.shape[0]; ++i) {
            for (size_t j = 0; j < arr.shape[1]; ++j) {
                for (size_t k = 0; k < arr.shape[2]; ++k) {
                    result[i][j][k] = data[i * arr.shape[1] * arr.shape[2] + j * arr.shape[2] + k];
                }
            }
        }
    } else if (arr.word_size == sizeof(double)) {
        // 数据是 float64，读取为 double 并转换为 float
        const double* data = arr.data<double>();
        for (size_t i = 0; i < arr.shape[0]; ++i) {
            for (size_t j = 0; j < arr.shape[1]; ++j) {
                for (size_t k = 0; k < arr.shape[2]; ++k) {
                    result[i][j][k] = static_cast<float>(data[i * arr.shape[1] * arr.shape[2] + j * arr.shape[2] + k]);
                }
            }
        }
    } else {
        // 数据类型不支持
        throw std::runtime_error("Unsupported data type for array " + array_name);
    }
    
    return result;
}

std::vector<std::vector<float>> load_npz_2d_array_float(const std::string& file_path, const std::string& array_name) {
    cnpy::npz_t npz = cnpy::npz_load(file_path);
    if (npz.find(array_name) == npz.end()) {
        throw std::runtime_error("Array " + array_name + " not found in " + file_path);
    }
    auto arr = npz[array_name];

    std::vector<std::vector<float>> result(
        arr.shape[0],
        std::vector<float>(arr.shape[1], 0.0f)
    );

    if (arr.word_size == sizeof(float)) {
        const float* data = arr.data<float>();
        for (size_t i = 0; i < arr.shape[0]; ++i) {
            for (size_t j = 0; j < arr.shape[1]; ++j) {
                result[i][j] = data[i * arr.shape[1] + j];
            }
        }
    } else if (arr.word_size == sizeof(double)) {
        const double* data = arr.data<double>();
        for (size_t i = 0; i < arr.shape[0]; ++i) {
            for (size_t j = 0; j < arr.shape[1]; ++j) {
                result[i][j] = static_cast<float>(data[i * arr.shape[1] + j]);
            }
        }
    } else {
        throw std::runtime_error("Unsupported data type for array " + array_name);
    }
    return result;
}