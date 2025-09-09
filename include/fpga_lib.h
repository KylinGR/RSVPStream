#ifndef FPGA_LIB_H
#define FPGA_LIB_H

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <thread>
#include <string>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include "SystemGlobal.h"
#include "file_base.h"

using namespace std;

// 设备节点定义
#define  FPGA_C2H_NODE  "/dev/xdma0_c2h_0"
#define  FPGA_H2C_NODE  "/dev/xdma0_h2c_0"

// 缓冲区和地址定义
#define HZHY_PS_BUF_LEN    (0x200000) //1M字节
#define HZHY_FPGA_ADDR_BASE  (0x0000000) //偏移16M的位置
#define RW_MAX_SIZE	0x7ffff000

// DDR地址定义
#define XLOCAL_ADR        0x10000  
#define LOCAL_COEFF_ADR   0x100000
#define BLOCAL_COEFF_ADR  0x400000
#define GLOBAL_COEFF_ADR  0x500000
#define QGLOB_COEFF_ADR   0x800000
#define XGLOB_ADR         0x900000

// 寄存器地址定义
#define        ADR_LOGIC_RST                      (0x0 <<2) 
#define        ADR_ALG_START                      (0x1 <<2) 
#define        ADR_WR_X_LOCAL_SCALE               (0x10<<2) 
#define        ADR_WR_M_LOCAL_SCALE               (0x11<<2) 
#define        ADR_GAMMA_LOCAL_SCALE              (0x12<<2) 
#define        ADR_BETA_LOCAL_SCALE               (0x13<<2) 
#define        ADR_B_LOCAL_SCALE                  (0x14<<2) 
#define        ADR_W_LOCAL_SCALE                  (0x15<<2) 
#define        ADR_LR_MODEL_SCALE                 (0x16<<2) 
#define        ADR_GAMMA_GLOBSCALE                (0x17<<2) 
#define        ADR_XGLOBAL_SCALE                  (0x18<<2) 
#define        ADR_MGLOBAL_SCALE                  (0x19<<2) 
#define        ADR_BETAGLOB_SCALE                 (0x1a<<2)   
#define        ADR_WGLOBAL_SCALE                  (0x1b<<2)   
#define        ADR_QGLOB_SCALE                    (0x1c<<2)   
#define        ADR_BGLOBAL_SCALE                  (0x1d<<2)   
#define        ADR_GSTF_WEIGHT_SCALE              (0x1e<<2)   
#define        ADR_HTOT_SCALE                     (0x1f<<2)   

#define        ADR_ALGOUT_DAT0                    (0x20<<2)
#define        ADR_ALGOUT_DAT1                    (0x21<<2)
#define        ADR_ALGOUT_DAT2                    (0x22<<2)
#define        ADR_ALGOUT_DAT3                    (0x23<<2)
#define        ADR_ALGOUT_DAT4                    (0x24<<2)
#define        ADR_ALGOUT_DAT5                    (0x25<<2)
#define        ADR_ALGOUT_DAT6                    (0x26<<2)
#define        ADR_ALGOUT_DAT7                    (0x27<<2)
#define        ADR_ALGOUT_DAT8                    (0x28<<2)
#define        ADR_ALGOUT_DAT9                    (0x29<<2)
#define        ADR_ALGOUT_DAT10                   (0x2a<<2)
#define        ADR_ALGOUT_DAT11                   (0x2b<<2)
#define        ADR_ALGOUT_DAT12                   (0x2c<<2)
#define        ADR_ALGOUT_DAT13                   (0x2d<<2)
#define        ADR_ALGOUT_DAT14                   (0x2e<<2)
#define        ADR_ALGOUT_DAT15                   (0x2f<<2)
#define        ADR_ALGOUT_DAT16                   (0x30<<2)   
#define        ADR_TIM                            (0x31<<2)
#define        ADR_HW_STATUS                      (0x80<<2) 

// 延时宏定义
#define DELAY_S(n) std::this_thread::sleep_for(std::chrono::seconds(n))
#define DELAY_MS(n) std::this_thread::sleep_for(std::chrono::milliseconds(n))
#define DELAY_US(n) std::this_thread::sleep_for(std::chrono::microseconds(n))

// 全局变量声明
extern int verbose;
extern int fdC2H, fdH2C;
extern int fdReg;
extern char *allocated;
extern uint8_t* map;
extern float X_global_scale[100];
extern float X_local_scale[100];
extern float htot_gold[17*100];

// 函数声明
template<typename ... Args>
std::string strFormat(const std::string& format, Args ... args) {
    size_t size = 1 + snprintf(nullptr, 0, format.c_str(), args ...);  // Extra space for \0
    char bytes[size];
    snprintf(bytes, size, format.c_str(), args ...);
    return std::string(bytes);
}

float hexToFloat_ptr(uint32_t hex);
uint32_t floatToHex(float f);
float sigmoid(float x);

ssize_t write_from_buffer(char *fname, int fd, char *buffer, uint64_t size, uint64_t base);

void RegWr(uint32_t u32Adr, uint32_t u32Dat);
uint32_t RegRd(uint32_t u32Adr);
void Vec2DDR(const std::vector<int16_t>& data, uint32_t ddr_sta_adr);
void File2DDR(string& strFile, uint32_t ddr_sta_adr);
void LoadGoldDat();
void NetRegInit();
void Start();
void Reset();
void LoadScale();

// 初始化和清理函数
int InitFPGA();
void CleanupFPGA();

#endif // FPGA_LIB_H
