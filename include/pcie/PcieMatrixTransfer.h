#ifndef PCIE_MATRIX_TRANSFER_H_
#define PCIE_MATRIX_TRANSFER_H_

#include <cstdint>
#include <cstddef>
#include <string>

// 配置结构，描述一次矩阵对/结果的传输参数
struct PcieMatrixTransferConfig {
    const char* dev_h2c = "/dev/xdma0_h2c_0"; // 主机-->FPGA
    const char* dev_c2h = "/dev/xdma0_c2h_0"; // FPGA-->主机
    uint64_t base_addr   = 0x01000000ULL;      // 基址 (AXI DDR基地址偏移)
    uint64_t mat1_offset = 0x00000000ULL;      // 矩阵1写入偏移
    uint64_t mat2_offset = 0x00100000ULL;      // 矩阵2写入偏移
    uint64_t out_offset  = 0x00200000ULL;      // 结果读取偏移（示例）
    uint32_t read_delay_us = 5000;             // 触发写入后等待FPGA处理的延时
};

// 返回码：>=0 成功字节数；<0 失败
int pcie_write_matrix_pair(const PcieMatrixTransferConfig& cfg,
                           const int16_t* mat1, size_t mat1_rows, size_t mat1_cols,
                           const int16_t* mat2, size_t mat2_rows, size_t mat2_cols);

// 从FPGA读取结果矩阵（结果尺寸由调用方指定）
int pcie_read_result_matrix(const PcieMatrixTransferConfig& cfg,
                            int16_t* out, size_t rows, size_t cols);

// 组合操作：写入两矩阵 -> 等待 -> 读取结果
int pcie_process_matrix_pair(const PcieMatrixTransferConfig& cfg,
                             const int16_t* mat1, size_t mat1_rows, size_t mat1_cols,
                             const int16_t* mat2, size_t mat2_rows, size_t mat2_cols,
                             int16_t* result, size_t res_rows, size_t res_cols);

// 寄存器读写操作（用于FPGA同步控制）
int pcie_write_register(const PcieMatrixTransferConfig& cfg, uint64_t reg_offset, uint32_t value);
uint32_t pcie_read_register(const PcieMatrixTransferConfig& cfg, uint64_t reg_offset);

// 批量寄存器操作
int pcie_write_registers(const PcieMatrixTransferConfig& cfg, uint64_t start_offset, 
                         const uint32_t* values, size_t count);
int pcie_read_registers(const PcieMatrixTransferConfig& cfg, uint64_t start_offset, 
                        uint32_t* values, size_t count);

#endif // PCIE_MATRIX_TRANSFER_H_
