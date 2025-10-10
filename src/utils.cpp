#include "utils.h"
#include <cnpy.h>
#include <stdexcept>
#include <cstdint>  // 包含 int64_t 类型
#include <limits>
#include <cmath>    // 包含 std::round
#include <utility>  // 包含 std::pair
#include <algorithm> // 包含 std::clamp
#include <cstring>

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

// 将二维浮点向量展平为一维浮点向量
std::vector<float> flatten_2d_vector(const std::vector<std::vector<float>>& tensor) {
    if (tensor.empty() || tensor[0].empty()) {
        return std::vector<float>();
    }
    
    size_t rows = tensor.size();
    size_t cols = tensor[0].size();
    std::vector<float> flat_data(rows * cols);
    
    // 方法1: 逐行复制（推荐，最快）
    for (size_t i = 0; i < rows; ++i) {
        std::memcpy(&flat_data[i * cols], 
                    tensor[i].data(), 
                    cols * sizeof(float));
    }
    
    return flat_data;
}

// ===== 纯标量版本（无SIMD，任何平台都能用） =====
std::pair<std::vector<int16_t>, float> dynamic_quantize_tensor_T_flatten(
    const std::vector<float>& tensor_data, size_t rows, size_t cols) {
    
    constexpr int16_t Q_MIN = std::numeric_limits<int16_t>::min();
    constexpr int16_t Q_MAX = std::numeric_limits<int16_t>::max();
    constexpr float MIN_CLAMP = 1e-8f;
    
    // if (!tensor_data || rows == 0 || cols == 0) {
    //     return {std::vector<int16_t>(), 0.0f};
    // }
    
    size_t data_size = rows * cols;
    
    // 1. 找最大绝对值（标量版本）
    float max_val = 0.0f;
    for (size_t i = 0; i < data_size; ++i) {
        float abs_val = std::abs(tensor_data[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    
    // 2. 计算scale
    float scale = static_cast<float>(Q_MAX) / std::max(max_val, MIN_CLAMP);
    
    // 3. 计算填充后的维度
    size_t padded_rows = (rows + 7) & ~7;  // 向上对齐到8的倍数
    size_t single_size = padded_rows * cols;
    
    // 4. 量化到临时数组（只量化一次）
    std::vector<int16_t> single_quantized(single_size, 0);
    
    for (size_t i = 0; i < data_size; ++i) {
        float scaled = tensor_data[i] * scale;
        int32_t rounded = static_cast<int32_t>(std::round(scaled));
        single_quantized[i] = static_cast<int16_t>(
            std::clamp(rounded, 
                      static_cast<int32_t>(Q_MIN), 
                      static_cast<int32_t>(Q_MAX))
        );
    }
    // padding区域已经通过初始化为0处理
    
    // 5. 复制17份（使用memcpy批量复制）
    size_t total_size = single_size * 17;
    std::vector<int16_t> quantized(total_size);
    
    for (int k = 0; k < 17; ++k) {
        std::memcpy(&quantized[k * single_size], 
                    single_quantized.data(), 
                    single_size * sizeof(int16_t));
    }
    
    return {std::move(quantized), scale};
}

// RK3588 支持 ARM NEON
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
// ===== NEON加速版本（仅在ARM平台使用） =====
std::pair<std::vector<int16_t>, float> dynamic_quantize_tensor_T_flatten_neon(
    const std::vector<float>& tensor_data, size_t rows, size_t cols) {
    
    constexpr int16_t Q_MIN = std::numeric_limits<int16_t>::min();
    constexpr int16_t Q_MAX = std::numeric_limits<int16_t>::max();
    constexpr float MIN_CLAMP = 1e-8f;
    
    // if (!tensor_data || rows == 0 || cols == 0) {
    //     return {std::vector<int16_t>(), 0.0f};
    // }
    
    size_t data_size = rows * cols;
    float max_val = 0.0f;
    
#ifdef __ARM_NEON
    float32x4_t max_vec = vdupq_n_f32(0.0f);
    size_t i = 0;
    
    // NEON处理，一次4个
    for (; i + 4 <= data_size; i += 4) {
        float32x4_t data = vld1q_f32(&tensor_data[i]);
        float32x4_t abs_data = vabsq_f32(data);
        max_vec = vmaxq_f32(max_vec, abs_data);
    }
    
    // 归约
    float32x2_t max_pair = vmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
    float32x2_t max_final = vpmax_f32(max_pair, max_pair);
    float neon_max = vget_lane_f32(max_final, 0);
    if (neon_max > max_val) max_val = neon_max;
    
    // 处理剩余
    for (; i < data_size; ++i) {
        float abs_val = std::abs(tensor_data[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
#else
    for (size_t i = 0; i < data_size; ++i) {
        float abs_val = std::abs(tensor_data[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
#endif
    
    float scale = static_cast<float>(Q_MAX) / std::max(max_val, MIN_CLAMP);
    
    size_t padded_rows = (rows + 7) & ~7;
    size_t single_size = padded_rows * cols;
    std::vector<int16_t> single_quantized(single_size, 0);
    
#ifdef __ARM_NEON
    float32x4_t scale_vec = vdupq_n_f32(scale);
    float32x4_t q_min_f = vdupq_n_f32(static_cast<float>(Q_MIN));
    float32x4_t q_max_f = vdupq_n_f32(static_cast<float>(Q_MAX));
    
    i = 0;
    for (; i + 4 <= data_size; i += 4) {
        float32x4_t data = vld1q_f32(&tensor_data[i]);
        float32x4_t scaled = vmulq_f32(data, scale_vec);    // 缩放
        float32x4_t rounded = vrndnq_f32(scaled);           // 四舍五入
        rounded = vmaxq_f32(rounded, q_min_f);              // 下限截断
        rounded = vminq_f32(rounded, q_max_f);              // 上限截断
        
        int32x4_t i32_vec = vcvtq_s32_f32(rounded);         // 转int32
        int16x4_t i16_vec = vmovn_s32(i32_vec);             // 窄化成int16
        vst1_s16(&single_quantized[i], i16_vec);            // 写回结果
    }
    
    for (; i < data_size; ++i) {
        float scaled = tensor_data[i] * scale;
        int32_t rounded = static_cast<int32_t>(std::round(scaled));
        single_quantized[i] = static_cast<int16_t>(
            std::clamp(rounded, static_cast<int32_t>(Q_MIN), 
                      static_cast<int32_t>(Q_MAX))
        );
    }
#else
    for (size_t i = 0; i < data_size; ++i) {
        float scaled = tensor_data[i] * scale;
        int32_t rounded = static_cast<int32_t>(std::round(scaled));
        single_quantized[i] = static_cast<int16_t>(
            std::clamp(rounded, static_cast<int32_t>(Q_MIN), 
                      static_cast<int32_t>(Q_MAX))
        );
    }
#endif
    
    // 复制17份
    size_t total_size = single_size * 17;
    std::vector<int16_t> quantized(total_size);
    
    for (int k = 0; k < 17; ++k) {
        std::memcpy(&quantized[k * single_size], 
                    single_quantized.data(), 
                    single_size * sizeof(int16_t));
    }
    
    return {std::move(quantized), scale};
}
