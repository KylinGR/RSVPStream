#include "fpga_inference_engine.h"
#include <algorithm>
#include <cstring>
#include <random>
#include <chrono>
#include <iostream>
#include <thread>
#include <cmath>
#include <unistd.h>

FpgaInferenceEngine::FpgaInferenceEngine(const FpgaConfig& config) 
    : config_(config) {
    // 初始化PCIe传输配置
    pcie_config_.dev_h2c = config.dev_h2c;
    pcie_config_.dev_c2h = config.dev_c2h;
    pcie_config_.base_addr = config.base_addr;
    pcie_config_.mat1_offset = config.x_local_offset;
    pcie_config_.mat2_offset = config.x_global_offset;
    pcie_config_.out_offset = config.result_offset;
    pcie_config_.read_delay_us = config.processing_timeout_us;  // 使用超时时间作为兼容延时
}

FpgaInferenceEngine::~FpgaInferenceEngine() {
    // 析构函数
}

void FpgaInferenceEngine::initialize() {
    // 预分配结果缓冲区
    fpga_result_buffer_.resize(config_.result_size);
    
    std::cout << "FPGA inference engine initialized with PCIe transfer" << std::endl;
    std::cout << "  H2C device: " << config_.dev_h2c << std::endl;
    std::cout << "  C2H device: " << config_.dev_c2h << std::endl;
    std::cout << "  DDR base address: 0x" << std::hex << config_.base_addr << std::dec << std::endl;
}

InferenceResult FpgaInferenceEngine::run_inference(const ProcessedData& data) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 1. 补零到8的倍数
    auto x_local_padded = _pad_to_multiple_of_8(data.x_local);
    auto x_global_padded = _pad_to_multiple_of_8(data.x_global);
    
    // 2. 动态量化
    QuantizationInfo local_quant_info, global_quant_info;
    auto x_local_quantized = _dynamic_quantize_to_int16(x_local_padded, local_quant_info);
    auto x_global_quantized = _dynamic_quantize_to_int16(x_global_padded, global_quant_info);
    
    auto quant_time = std::chrono::high_resolution_clock::now();
    
    // 3. 通过PCIe传输到FPGA DDR
    if (!_transfer_to_fpga_ddr(x_local_quantized, x_global_quantized)) {
        std::cerr << "Failed to transfer data to FPGA DDR" << std::endl;
        InferenceResult result;
        result.score = 0.0f;
        result.file_path = data.file_path;
        result.sample_id = data.sample_id;
        return result;
    }
    
    auto transfer_time = std::chrono::high_resolution_clock::now();
    
    // 4. 通知FPGA数据已就绪，并等待处理完成
    if (config_.use_hardware_sync) {
        // 通知FPGA数据已就绪
        if (!_notify_fpga_data_ready()) {
            std::cerr << "Failed to notify FPGA data ready" << std::endl;
            InferenceResult result;
            result.score = 0.0f;
            result.file_path = data.file_path;
            result.sample_id = data.sample_id;
            return result;
        }
        
        // 等待FPGA处理完成
        if (!_wait_for_fpga_processing_complete()) {
            std::cerr << "FPGA processing timeout or failed" << std::endl;
            InferenceResult result;
            result.score = 0.0f;
            result.file_path = data.file_path;
            result.sample_id = data.sample_id;
            return result;
        }
    } else {
        // 兼容模式：使用固定延时
        std::this_thread::sleep_for(std::chrono::microseconds(config_.processing_timeout_us));
    }
    
    auto process_time = std::chrono::high_resolution_clock::now();
    
    // 5. 从FPGA DDR读取结果
    if (!_read_from_fpga_ddr(fpga_result_buffer_, config_.result_size)) {
        std::cerr << "Failed to read result from FPGA DDR" << std::endl;
        InferenceResult result;
        result.score = 0.0f;
        result.file_path = data.file_path;
        result.sample_id = data.sample_id;
        return result;
    }
    
    auto read_time = std::chrono::high_resolution_clock::now();
    
    // 6. 反量化（假设输出使用相同的量化参数）
    auto result_float = _dequantize_from_int16(fpga_result_buffer_, local_quant_info);
    
    // 7. 转换为最终分数
    float score = _convert_result_to_score(result_float);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // 输出详细的时间统计
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    auto quant_us = std::chrono::duration_cast<std::chrono::microseconds>(quant_time - start_time).count();
    auto transfer_us = std::chrono::duration_cast<std::chrono::microseconds>(transfer_time - quant_time).count();
    auto process_us = std::chrono::duration_cast<std::chrono::microseconds>(process_time - transfer_time).count();
    auto read_us = std::chrono::duration_cast<std::chrono::microseconds>(read_time - process_time).count();
    
    if (data.sample_id % 100 == 0) {  // 每100个样本打印一次详细信息
        std::cout << "Sample " << data.sample_id << " timing: "
                  << "Quant=" << quant_us << "us, "
                  << "Transfer=" << transfer_us << "us, "
                  << "Process=" << process_us << "us, "
                  << "Read=" << read_us << "us, "
                  << "Total=" << total_us << "us" << std::endl;
    }
    
    InferenceResult result;
    result.score = score;
    result.file_path = data.file_path;
    result.sample_id = data.sample_id;
    
    return result;
}

std::vector<std::vector<float>> FpgaInferenceEngine::_pad_to_multiple_of_8(
    const std::vector<std::vector<float>>& input) {
    
    std::vector<std::vector<float>> padded = input;
    
    for (auto& row : padded) {
        size_t current_size = row.size();
        size_t padded_size = ((current_size + 7) / 8) * 8;  // 向上取整到8的倍数
        
        if (padded_size > current_size) {
            row.resize(padded_size, 0.0f);  // 用0填充
        }
    }
    
    return padded;
}

std::vector<int16_t> FpgaInferenceEngine::_dynamic_quantize_to_int16(
    const std::vector<std::vector<float>>& input,
    QuantizationInfo& quant_info) {
    
    // 对称量化：找到绝对值的最大值
    float abs_max = 0.0f;
    
    for (const auto& row : input) {
        for (float val : row) {
            abs_max = std::max(abs_max, std::abs(val));
        }
    }
    
    // 计算对称量化的缩放因子
    // int16范围: -32768 to 32767，对称使用 -32767 to 32767
    quant_info.scale = abs_max / 32767.0f;
    
    // 防止除零
    if (quant_info.scale == 0.0f) {
        quant_info.scale = 1.0f;
    }
    
    // 对称量化
    std::vector<int16_t> quantized;
    quantized.reserve(input.size() * (input.empty() ? 0 : input[0].size()));
    
    for (const auto& row : input) {
        for (float val : row) {
            int32_t quantized_val = static_cast<int32_t>(std::round(val / quant_info.scale));
            quantized_val = std::max(-32767, std::min(32767, quantized_val));
            quantized.push_back(static_cast<int16_t>(quantized_val));
        }
    }
    
    return quantized;
}

std::vector<float> FpgaInferenceEngine::_dequantize_from_int16(
    const std::vector<int16_t>& input,
    const QuantizationInfo& quant_info) {
    
    std::vector<float> dequantized;
    dequantized.reserve(input.size());
    
    // 对称反量化：直接乘以scale
    for (int16_t val : input) {
        float dequant_val = static_cast<float>(val) * quant_info.scale;
        dequantized.push_back(dequant_val);
    }
    
    return dequantized;
}

bool FpgaInferenceEngine::_transfer_to_fpga_ddr(
    const std::vector<int16_t>& x_local_quantized,
    const std::vector<int16_t>& x_global_quantized) {
    
    try {
        // 计算数据大小
        size_t local_rows = x_local_quantized.size() / ((x_local_quantized.size() > 0) ? 1 : 1);
        size_t local_cols = 1;  // 扁平化数据
        size_t global_rows = x_global_quantized.size() / ((x_global_quantized.size() > 0) ? 1 : 1);
        size_t global_cols = 1;  // 扁平化数据
        
        // 重新计算矩阵维度（根据实际数据结构）
        // 假设x_local是299x56，x_global是250x64，但都已扁平化
        if (!x_local_quantized.empty()) {
            local_rows = 299;  // 根据实际配置调整
            local_cols = x_local_quantized.size() / local_rows;
        }
        if (!x_global_quantized.empty()) {
            global_rows = 250;  // 根据实际配置调整  
            global_cols = x_global_quantized.size() / global_rows;
        }
        
        // 使用PCIe写入两个矩阵到FPGA DDR
        int write_result = pcie_write_matrix_pair(
            pcie_config_,
            x_local_quantized.data(), local_rows, local_cols,
            x_global_quantized.data(), global_rows, global_cols
        );
        
        if (write_result < 0) {
            std::cerr << "PCIe write failed with error code: " << write_result << std::endl;
            return false;
        }
        
        std::cout << "Successfully transferred " << write_result 
                  << " bytes to FPGA DDR via PCIe" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during PCIe transfer: " << e.what() << std::endl;
        return false;
    }
}

bool FpgaInferenceEngine::_read_from_fpga_ddr(std::vector<int16_t>& result, size_t expected_size) {
    try {
        // 确保结果缓冲区大小正确
        if (result.size() != expected_size) {
            result.resize(expected_size);
        }
        
        // 计算结果矩阵维度
        size_t result_rows = expected_size;  // 简化为一维
        size_t result_cols = 1;
        
        // 从FPGA DDR读取结果
        int read_result = pcie_read_result_matrix(
            pcie_config_,
            result.data(), result_rows, result_cols
        );
        
        if (read_result < 0) {
            std::cerr << "PCIe read failed with error code: " << read_result << std::endl;
            return false;
        }
        
        std::cout << "Successfully read " << read_result 
                  << " bytes from FPGA DDR via PCIe" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during PCIe read: " << e.what() << std::endl;
        return false;
    }
}

float FpgaInferenceEngine::_convert_result_to_score(const std::vector<float>& result) {
    if (result.empty()) {
        return 0.0f;
    }
    
    // 简单的分数计算：取平均值并应用sigmoid函数
    float sum = 0.0f;
    for (float val : result) {
        sum += val;
    }
    float mean = sum / result.size();
    
    // 应用sigmoid函数将结果映射到[0,1]范围
    float score = 1.0f / (1.0f + std::exp(-mean));
    
    return score;
}

// FPGA硬件同步控制函数实现

bool FpgaInferenceEngine::_notify_fpga_data_ready() {
    try {
        // 设置数据就绪标志
        _set_data_ready_flag(true);
        
        // 向控制寄存器写入启动信号
        _write_control_register(0x00000001);  // 启动处理
        
        std::cout << "Notified FPGA: data ready for processing" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error notifying FPGA data ready: " << e.what() << std::endl;
        return false;
    }
}

bool FpgaInferenceEngine::_wait_for_fpga_processing_complete() {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto timeout = std::chrono::microseconds(config_.processing_timeout_us);
    
    while (true) {
        // 检查是否超时
        auto current_time = std::chrono::high_resolution_clock::now();
        if (current_time - start_time > timeout) {
            std::cerr << "FPGA processing timeout after " 
                      << config_.processing_timeout_us << " microseconds" << std::endl;
            return false;
        }
        
        // 检查FPGA状态
        if (_check_fpga_status()) {
            // 检查结果是否就绪
            if (_get_result_ready_flag()) {
                std::cout << "FPGA processing completed successfully" << std::endl;
                return true;
            }
        }
        
        // 短暂休眠后继续轮询
        std::this_thread::sleep_for(std::chrono::microseconds(config_.polling_interval_us));
    }
}

bool FpgaInferenceEngine::_check_fpga_status() {
    try {
        uint32_t status = _read_status_register();
        
        // 检查状态寄存器的各个位
        bool is_busy = (status & 0x00000001) != 0;      // bit 0: 忙碌标志
        bool has_error = (status & 0x00000002) != 0;    // bit 1: 错误标志
        bool is_ready = (status & 0x00000004) != 0;     // bit 2: 就绪标志
        
        if (has_error) {
            std::cerr << "FPGA reported error status: 0x" << std::hex << status << std::dec << std::endl;
            return false;
        }
        
        // FPGA空闲且就绪表示可以继续
        return !is_busy && is_ready;
        
    } catch (const std::exception& e) {
        std::cerr << "Error checking FPGA status: " << e.what() << std::endl;
        return false;
    }
}

void FpgaInferenceEngine::_write_control_register(uint32_t value) {
    try {
        // 使用PCIe写入控制寄存器
        int result = pcie_write_register(pcie_config_, config_.ctrl_reg_offset, value);
        
        if (result < 0) {
            throw std::runtime_error("PCIe register write failed with code: " + std::to_string(result));
        }
        
        std::cout << "Successfully wrote control register: 0x" << std::hex << value << std::dec << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error writing control register: " << e.what() << std::endl;
        throw;
    }
}

uint32_t FpgaInferenceEngine::_read_status_register() {
    try {
        // 使用PCIe读取状态寄存器
        uint32_t status = pcie_read_register(pcie_config_, config_.status_reg_offset);
        
        std::cout << "Read status register: 0x" << std::hex << status << std::dec << std::endl;
        return status;
        
    } catch (const std::exception& e) {
        std::cerr << "Error reading status register: " << e.what() << std::endl;
        // 返回错误状态（busy + error）
        return 0x00000003;
    }
}

void FpgaInferenceEngine::_set_data_ready_flag(bool ready) {
    try {
        uint32_t flag_value = ready ? 0x00000001 : 0x00000000;
        
        int result = pcie_write_register(pcie_config_, config_.data_ready_offset, flag_value);
        
        if (result < 0) {
            throw std::runtime_error("PCIe data ready flag write failed with code: " + std::to_string(result));
        }
        
        std::cout << "Set data ready flag to " << (ready ? "TRUE" : "FALSE") << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error setting data ready flag: " << e.what() << std::endl;
        throw;
    }
}

bool FpgaInferenceEngine::_get_result_ready_flag() {
    try {
        // 读取结果就绪标志
        uint32_t flag_value = pcie_read_register(pcie_config_, config_.result_ready_offset);
        
        bool result_ready = (flag_value & 0x00000001) != 0;
        
        if (result_ready) {
            std::cout << "Result ready flag is TRUE" << std::endl;
        }
        
        return result_ready;
        
    } catch (const std::exception& e) {
        std::cerr << "Error reading result ready flag: " << e.what() << std::endl;
        return false;
    }
}
