#pragma once
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include "data_types.h"
#include "pcie/PcieMatrixTransfer.h"

class FpgaInferenceEngine {
public:
    // 对称量化信息结构
    struct QuantizationInfo {
        float scale;        // 缩放因子（对称量化只需要scale）
    };

private:
    FpgaConfig config_;
    PcieMatrixTransferConfig pcie_config_;  // PCIe传输配置
    
    // 量化缓冲区
    std::vector<int16_t> x_local_quantized_;
    std::vector<int16_t> x_global_quantized_;
    std::vector<int16_t> fpga_result_buffer_;  // FPGA结果缓冲区

public:
    explicit FpgaInferenceEngine(const FpgaConfig& config);
    ~FpgaInferenceEngine();
    
    void initialize();
    InferenceResult run_inference(const ProcessedData& data);

private:
    // 数据预处理：补零到8的倍数
    std::vector<std::vector<float>> _pad_to_multiple_of_8(
        const std::vector<std::vector<float>>& input);
    
    // 动态量化：float -> int16
    std::vector<int16_t> _dynamic_quantize_to_int16(
        const std::vector<std::vector<float>>& input,
        QuantizationInfo& quant_info);
    
    // 反量化：int16 -> float 
    std::vector<float> _dequantize_from_int16(
        const std::vector<int16_t>& input,
        const QuantizationInfo& quant_info);
    
    // 真实PCIe传输到FPGA DDR
    bool _transfer_to_fpga_ddr(const std::vector<int16_t>& x_local_quantized,
                               const std::vector<int16_t>& x_global_quantized);
    
    // 从FPGA DDR读取结果
    bool _read_from_fpga_ddr(std::vector<int16_t>& result, size_t expected_size);
    
    // FPGA硬件同步控制
    bool _notify_fpga_data_ready();                // 通知FPGA数据已就绪
    bool _wait_for_fpga_processing_complete();     // 等待FPGA处理完成
    bool _check_fpga_status();                     // 检查FPGA状态
    void _write_control_register(uint32_t value);  // 写控制寄存器
    uint32_t _read_status_register();              // 读状态寄存器
    void _set_data_ready_flag(bool ready);         // 设置数据就绪标志
    bool _get_result_ready_flag();                 // 获取结果就绪标志
    
    // 结果转换为分数
    float _convert_result_to_score(const std::vector<float>& result);
};
