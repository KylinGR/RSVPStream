#pragma once
#include <vector>
#include <string>

std::vector<int64_t> load_npz_array_int(const std::string& file_path, const std::string& array_name);
std::vector<std::vector<std::vector<float>>> load_npz_3d_array_float(const std::string& file_path, const std::string& array_name);
std::vector<std::vector<float>> load_npz_2d_array_float(const std::string& file_path, const std::string& array_name); 
std::pair<std::vector<int16_t>, float> dynamic_quantize_tensor_T_flatten(const std::vector<float>& tensor_data, size_t rows, size_t cols);
std::pair<std::vector<int16_t>, float> dynamic_quantize_tensor_T_flatten_neon(const std::vector<float>& tensor_data, size_t rows, size_t cols);
std::vector<float> flatten_2d_vector(const std::vector<std::vector<float>>& tensor);
