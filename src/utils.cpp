#include "utils.h"
#include <cnpy.h>
#include <stdexcept>
#include <cstdint>

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

std::vector<float> load_npz_2d_array_float(const std::string& file_path, const std::string& array_name, 
                                                    size_t& rows, size_t& cols) {
    cnpy::npz_t npz = cnpy::npz_load(file_path);
    if (npz.find(array_name) == npz.end()) {
        throw std::runtime_error("Array " + array_name + " not found in " + file_path);
    }
    auto arr = npz[array_name];
    
    rows = arr.shape[0];
    cols = arr.shape[1];
    const size_t total_size = rows * cols;
    
    // 使用单一连续内存块，避免碎片化
    std::vector<float> result;
    result.reserve(total_size);
    
    if (arr.word_size == sizeof(float)) {
        // 直接内存拷贝，最高效
        const float* data = arr.data<float>();
        result.assign(data, data + total_size);
    } else if (arr.word_size == sizeof(double)) {
        // 批量类型转换，减少函数调用开销
        const double* data = arr.data<double>();
        result.resize(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            result[i] = static_cast<float>(data[i]);
        }
    } else {
        throw std::runtime_error("Unsupported data type for array " + array_name);
    }
    
    return result;
}