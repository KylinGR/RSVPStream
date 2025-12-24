#include "fpga_runner.h"
#include "fpga_lib.h"
#include "utils.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>

FPGAProcessor::FPGAProcessor() : s_all(0), s_data(0), bErr(false) {
    std::cout << "Hello,World!" << __DATE__ << "," << __TIME__ << std::endl;
    
    uint64_t newTime, oldTime, loopNum = 0;
    int s32Ret = 0;
    uint8_t temp;
    bool flag;

    if (InitFPGA() != 0) {
        throw std::runtime_error("FPGA初始化失败");
    }

    Reset();
    load_coeff();
    
    std::string strFile;
    strFile = "/home/hzhy/RSVPStream/data/dat/local_coef.dat";
    File2DDR(strFile, LOCAL_COEFF_ADR);
    
    strFile = "/home/hzhy/RSVPStream/data/dat/b_local.dat";
    File2DDR(strFile, BLOCAL_COEFF_ADR);
    
    strFile = "/home/hzhy/RSVPStream/data/dat/global_coef.dat";
    File2DDR(strFile, GLOBAL_COEFF_ADR);
    
    strFile = "/home/hzhy/RSVPStream/data/dat/Q_global.dat";
    File2DDR(strFile, QGLOB_COEFF_ADR);
    
    NetRegInit();

    int32_t dat = RegRd(0xc000);
	printf("FPGA VERSION =0x%x\n",dat);

    dat = RegRd(ADR_ALG_START);
    printf("ADR_ALG_START=%x\n", dat);

    dat = RegRd(ADR_HW_STATUS);
    printf("ADR_HW_STATUS=%x\n", dat);

    // LoadScale();
}

float FPGAProcessor::fpga_runner(const std::vector<std::vector<float>>& x_local_data, const std::vector<std::vector<float>>& x_global_data) {

    auto validate_matrix = [](const auto& mat, const char* name) -> std::pair<size_t, size_t> {
        if (mat.empty() || mat[0].empty()) {
            throw std::invalid_argument(std::string(name) + " is empty");
        }
        const size_t cols = mat[0].size();
        for (const auto& row : mat) {
            if (row.size() != cols) {
                throw std::invalid_argument(std::string(name) + " has inconsistent row sizes");
            }
        }
        return {mat.size(), cols};
    };

    auto [local_rows, local_cols] = validate_matrix(x_local_data, "x_local_data");
    auto [global_rows, global_cols] = validate_matrix(x_global_data, "x_global_data");

    auto x_local_data_flatten = flatten_2d_vector(x_local_data);
    auto x_global_data_flatten = flatten_2d_vector(x_global_data);
    auto start = std::chrono::high_resolution_clock::now();
    auto [x_local, local_scale] = dynamic_quantize_tensor_T_flatten_neon(x_local_data_flatten, local_rows, local_cols);
    auto [x_global, global_scale] = dynamic_quantize_tensor_T_flatten_neon(x_global_data_flatten, global_rows, global_cols);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Quantize cost " << duration << " ms" << std::endl;

    bErr = false;
    RegWr(ADR_WR_X_LOCAL_SCALE, floatToHex(local_scale));
    RegWr(ADR_XGLOBAL_SCALE, floatToHex(global_scale));
    
    uint32_t dat = RegRd(ADR_WR_X_LOCAL_SCALE);
    float dat_fp = hexToFloat_ptr(dat);

    Vec2DDR(x_global, XGLOB_ADR);
    Vec2DDR(x_local,  XLOCAL_ADR);

    Start();
    DELAY_MS(1);

    for (int i = 0; i < 17; i++) {
        dat = RegRd((0x20 + i) << 2);
        dat_fp = hexToFloat_ptr(dat);
        s_data += sigmoid(dat_fp);
    }

    s_all = s_data / 17.0f;
    s_data = 0;

    dat = RegRd(ADR_TIM);
    printf("R[ADR_TIM]=%f us\n", dat * 4.1666 / 1000.0);
    // if (bErr == true) {
    //     printf("Cmp ERROR!\n");
    // }
    return s_all;
}