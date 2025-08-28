#pragma once
#include <vector>
#include <string>

std::vector<int64_t> load_npz_array_int(const std::string& file_path, const std::string& array_name);
std::vector<float> load_npz_2d_array_float(const std::string& file_path, const std::string& array_name, 
                                                    size_t& rows, size_t& cols);