#pragma once
#include <vector>
#include <string>

// 数据结构定义
struct ProcessedData {
    std::vector<std::vector<float>> x_local;
    std::vector<std::vector<float>> x_global;
    std::string file_path;
    int sample_id;
};

struct InferenceResult {
    float score;
    std::string file_path;
    int sample_id;
};

struct EvaluationConfig {
    int n_positive = 61;
    int n_negative = 1096;
    float threshold = 0.5;
};

// FPGA配置结构
struct FpgaConfig {
    // PCIe设备路径
    const char* dev_h2c = "/dev/xdma0_h2c_0";      // 主机到FPGA
    const char* dev_c2h = "/dev/xdma0_c2h_0";      // FPGA到主机
    
    // DDR内存地址配置
    uint64_t base_addr = 0x01000000ULL;            // DDR基地址
    uint64_t x_local_offset = 0x00000000ULL;       // x_local数据偏移
    uint64_t x_global_offset = 0x00100000ULL;      // x_global数据偏移  
    uint64_t result_offset = 0x00200000ULL;        // 结果数据偏移
    
    // 控制和状态寄存器地址
    uint64_t ctrl_reg_offset = 0x00300000ULL;      // 控制寄存器偏移
    uint64_t status_reg_offset = 0x00300004ULL;    // 状态寄存器偏移
    uint64_t data_ready_offset = 0x00300008ULL;    // 数据就绪标志偏移
    uint64_t result_ready_offset = 0x0030000CULL;  // 结果就绪标志偏移
    
    // 处理配置
    uint32_t processing_timeout_us = 50000;        // FPGA处理超时时间
    uint32_t polling_interval_us = 100;            // 状态轮询间隔
    bool use_dynamic_quantization = true;          // 使用动态量化
    bool use_hardware_sync = true;                 // 使用硬件同步
    
    // 结果配置
    size_t result_size = 100;                      // 期望的结果向量大小
};

struct ModelConfig {
    std::string model_order_path;  // 独立的model_order.npy文件路径
    FpgaConfig fpga_config;        // FPGA配置
    int win_len;
    int chan_xlen;
    int chan_ylen;
    int step_x;
    int step_y;
    int max_N_model;
    float gstf_weight;
    int N_local_model = 299;
};
