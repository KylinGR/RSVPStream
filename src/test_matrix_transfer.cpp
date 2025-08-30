#include "SystemGlobal.h"
#include "PcieMatrixTransfer.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

#define MAT1_R 250
#define MAT1_C 64
#define MAT2_R 299
#define MAT2_C 56
#define RES_R  MAT1_R   // 示例：假设结果矩阵大小，可按需求修改
#define RES_C  MAT2_C

static void fill_matrix_int16(int16_t* data, int rows, int cols, int seed)
{
    for(int r = 0; r < rows; ++r)
        for(int c = 0; c < cols; ++c)
            data[r * cols + c] = (int16_t)((seed + r * cols + c) & 0x7FFF);
}

int main()
{
    size_t mat1_bytes = (size_t)MAT1_R * MAT1_C * sizeof(int16_t);
    size_t mat2_bytes = (size_t)MAT2_R * MAT2_C * sizeof(int16_t);
    size_t res_bytes  = (size_t)RES_R  * RES_C  * sizeof(int16_t);

    int16_t *m1 = nullptr, *m2 = nullptr, *res = nullptr;
    if(posix_memalign((void**)&m1, 4096, mat1_bytes) ||
       posix_memalign((void**)&m2, 4096, mat2_bytes) ||
       posix_memalign((void**)&res,4096, res_bytes)) {
        fprintf(stderr, "alloc fail\n");
        return -1;
    }
    fill_matrix_int16(m1, MAT1_R, MAT1_C, 1000);
    fill_matrix_int16(m2, MAT2_R, MAT2_C, 2000);
    memset(res, 0, res_bytes);

    PcieMatrixTransferConfig cfg; // 使用默认设备与地址，可根据需要修改
    printf("Send two matrices...\n");
    if(pcie_write_matrix_pair(cfg, m1, MAT1_R, MAT1_C, m2, MAT2_R, MAT2_C) < 0) {
        fprintf(stderr, "write pair failed\n");
        return -1;
    }
    printf("Wait FPGA processing %u us...\n", cfg.read_delay_us);
    usleep(cfg.read_delay_us);

    if(pcie_read_result_matrix(cfg, res, RES_R, RES_C) < 0) {
        fprintf(stderr, "read result failed\n");
        return -1;
    }
    // 这里未做校验逻辑（结果由FPGA生成），可根据协议增加CRC/首尾标记等
    printf("Result first 8 elements: ");
    int show = (RES_R*RES_C < 8) ? (RES_R*RES_C) : 8;
    for(int i=0;i<show;i++) printf("%d ", res[i]);
    printf("\nDone.\n");

    free(m1); free(m2); free(res);
    return 0;
}
