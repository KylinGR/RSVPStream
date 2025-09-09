#include <iostream>
#include <rknn_api.h>

int main() {
    rknn_context ctx = 0;
    
    // 使用现有的模型文件初始化一个临时context
    std::string model_path = "/home/hzhy/csk/RSVPStream/data/model/optimized_model_v3_1.rknn";
    
    int ret = rknn_init(&ctx, (void*)model_path.c_str(), 0, 0, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "Failed to initialize RKNN context: " << ret << std::endl;
        return 1;
    }
    
    // 查询SDK版本
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
    if (ret == RKNN_SUCC) {
        std::cout << "=== RKNN Version Information ===" << std::endl;
        std::cout << "API Version: " << version.api_version << std::endl;
        std::cout << "Driver Version: " << version.drv_version << std::endl;
    } else {
        std::cerr << "Failed to query RKNN version: " << ret << std::endl;
    }
    
    // 清理
    rknn_destroy(ctx);
    
    return 0;
}
