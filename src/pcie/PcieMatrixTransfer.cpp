#include "PcieMatrixTransfer.h"
#include "SystemGlobal.h"
#include "dma_utils.h"
#include "HzhyGadget.h"

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <cstdio>

static int open_dev(const char* path) {
    int fd = open(path, O_RDWR);
    if (fd < 0) {
        perror(path);
    }
    return fd;
}

static int write_block(const char* dev_name, int fd, const void* buf, size_t bytes, uint64_t addr) {
    ssize_t rc = write_from_buffer((char*)dev_name, fd, (char*)buf, bytes, addr);
    if (rc != (ssize_t)bytes) {
        fprintf(stderr, "write_block mismatch rc=%ld expect=%zu addr=0x%lx\n", (long)rc, bytes, (unsigned long)addr);
        return -1;
    }
    return 0;
}

static int read_block(const char* dev_name, int fd, void* buf, size_t bytes, uint64_t addr) {
    ssize_t rc = read_to_buffer((char*)dev_name, fd, (char*)buf, bytes, addr);
    if (rc != (ssize_t)bytes) {
        fprintf(stderr, "read_block mismatch rc=%ld expect=%zu addr=0x%lx\n", (long)rc, bytes, (unsigned long)addr);
        return -1;
    }
    return 0;
}

int pcie_write_matrix_pair(const PcieMatrixTransferConfig& cfg,
                           const int16_t* mat1, size_t mat1_rows, size_t mat1_cols,
                           const int16_t* mat2, size_t mat2_rows, size_t mat2_cols) {
    if (!mat1 || !mat2) return -1;
    size_t bytes1 = mat1_rows * mat1_cols * sizeof(int16_t);
    size_t bytes2 = mat2_rows * mat2_cols * sizeof(int16_t);

    int fdH2C = open_dev(cfg.dev_h2c);
    if (fdH2C < 0) return -1;

    uint64_t t0 = HzhyGetNowUs();
    if (write_block(cfg.dev_h2c, fdH2C, mat1, bytes1, cfg.base_addr + cfg.mat1_offset) < 0) { close(fdH2C); return -1; }
    if (write_block(cfg.dev_h2c, fdH2C, mat2, bytes2, cfg.base_addr + cfg.mat2_offset) < 0) { close(fdH2C); return -1; }
    uint64_t t1 = HzhyGetNowUs();
    printf("pcie_write_matrix_pair: total bytes=%zu time=%llu us\n", bytes1 + bytes2, (unsigned long long)(t1 - t0));

    close(fdH2C);
    return (int)(bytes1 + bytes2);
}

int pcie_read_result_matrix(const PcieMatrixTransferConfig& cfg,
                            int16_t* out, size_t rows, size_t cols) {
    if (!out) return -1;
    size_t bytes = rows * cols * sizeof(int16_t);
    int fdC2H = open_dev(cfg.dev_c2h);
    if (fdC2H < 0) return -1;

    uint64_t t0 = HzhyGetNowUs();
    if (read_block(cfg.dev_c2h, fdC2H, out, bytes, cfg.base_addr + cfg.out_offset) < 0) { close(fdC2H); return -1; }
    uint64_t t1 = HzhyGetNowUs();
    printf("pcie_read_result_matrix: bytes=%zu time=%llu us\n", bytes, (unsigned long long)(t1 - t0));

    close(fdC2H);
    return (int)bytes;
}

int pcie_process_matrix_pair(const PcieMatrixTransferConfig& cfg,
                             const int16_t* mat1, size_t mat1_rows, size_t mat1_cols,
                             const int16_t* mat2, size_t mat2_rows, size_t mat2_cols,
                             int16_t* result, size_t res_rows, size_t res_cols) {
    int w = pcie_write_matrix_pair(cfg, mat1, mat1_rows, mat1_cols, mat2, mat2_rows, mat2_cols);
    if (w < 0) return -1;

    // 等待 FPGA 处理
    usleep(cfg.read_delay_us);

    int r = pcie_read_result_matrix(cfg, result, res_rows, res_cols);
    if (r < 0) return -1;

    return 0;
}

// 寄存器读写操作实现

int pcie_write_register(const PcieMatrixTransferConfig& cfg, uint64_t reg_offset, uint32_t value) {
    int fdH2C = open_dev(cfg.dev_h2c);
    if (fdH2C < 0) {
        fprintf(stderr, "Failed to open H2C device for register write: %s\n", cfg.dev_h2c);
        return -1;
    }
    
    uint64_t reg_addr = cfg.base_addr + reg_offset;
    
    // 写入32位寄存器值
    int result = write_block(cfg.dev_h2c, fdH2C, &value, sizeof(uint32_t), reg_addr);
    
    close(fdH2C);
    
    if (result < 0) {
        fprintf(stderr, "Register write failed at offset 0x%lx\n", (unsigned long)reg_offset);
        return -1;
    }
    
    printf("pcie_write_register: offset=0x%lx value=0x%08x\n", 
           (unsigned long)reg_offset, (unsigned)value);
    
    return 0;  // 成功
}

uint32_t pcie_read_register(const PcieMatrixTransferConfig& cfg, uint64_t reg_offset) {
    int fdC2H = open_dev(cfg.dev_c2h);
    if (fdC2H < 0) {
        fprintf(stderr, "Failed to open C2H device for register read: %s\n", cfg.dev_c2h);
        return 0xFFFFFFFF;  // 错误标识
    }
    
    uint64_t reg_addr = cfg.base_addr + reg_offset;
    uint32_t value = 0;
    
    // 读取32位寄存器值
    int result = read_block(cfg.dev_c2h, fdC2H, &value, sizeof(uint32_t), reg_addr);
    
    close(fdC2H);
    
    if (result < 0) {
        fprintf(stderr, "Register read failed at offset 0x%lx\n", (unsigned long)reg_offset);
        return 0xFFFFFFFF;  // 错误标识
    }
    
    printf("pcie_read_register: offset=0x%lx value=0x%08x\n", 
           (unsigned long)reg_offset, (unsigned)value);
    
    return value;
}

int pcie_write_registers(const PcieMatrixTransferConfig& cfg, uint64_t start_offset, 
                         const uint32_t* values, size_t count) {
    if (!values || count == 0) return -1;
    
    int fdH2C = open_dev(cfg.dev_h2c);
    if (fdH2C < 0) {
        fprintf(stderr, "Failed to open H2C device for bulk register write: %s\n", cfg.dev_h2c);
        return -1;
    }
    
    uint64_t reg_addr = cfg.base_addr + start_offset;
    size_t total_bytes = count * sizeof(uint32_t);
    
    // 批量写入寄存器
    int result = write_block(cfg.dev_h2c, fdH2C, values, total_bytes, reg_addr);
    
    close(fdH2C);
    
    if (result < 0) {
        fprintf(stderr, "Bulk register write failed at offset 0x%lx, count=%zu\n", 
                (unsigned long)start_offset, count);
        return -1;
    }
    
    printf("pcie_write_registers: offset=0x%lx count=%zu bytes=%zu\n", 
           (unsigned long)start_offset, count, total_bytes);
    
    return 0;  // 成功
}

int pcie_read_registers(const PcieMatrixTransferConfig& cfg, uint64_t start_offset, 
                        uint32_t* values, size_t count) {
    if (!values || count == 0) return -1;
    
    int fdC2H = open_dev(cfg.dev_c2h);
    if (fdC2H < 0) {
        fprintf(stderr, "Failed to open C2H device for bulk register read: %s\n", cfg.dev_c2h);
        return -1;
    }
    
    uint64_t reg_addr = cfg.base_addr + start_offset;
    size_t total_bytes = count * sizeof(uint32_t);
    
    // 批量读取寄存器
    int result = read_block(cfg.dev_c2h, fdC2H, values, total_bytes, reg_addr);
    
    close(fdC2H);
    
    if (result < 0) {
        fprintf(stderr, "Bulk register read failed at offset 0x%lx, count=%zu\n", 
                (unsigned long)start_offset, count);
        return -1;
    }
    
    printf("pcie_read_registers: offset=0x%lx count=%zu bytes=%zu\n", 
           (unsigned long)start_offset, count, total_bytes);
    
    return 0;  // 成功
}
