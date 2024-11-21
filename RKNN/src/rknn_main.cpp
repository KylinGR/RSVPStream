#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "rknn_model.h"

void generateRandomInput(std::vector<float>& data, int size) {
    srand((unsigned int)time(NULL));
    data.resize(size);
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void printOutput(const std::vector<float>& output)
{
    std::cout << "Output size: " << output.size() << std::endl;
    std::cout << "Output (first 10 values): ";
    for (size_t i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <gstf_model_path>" << " <local_model_path>" << std::endl;
        return -1;
    }

    const char* gstf_model_path = argv[1];
    const char* local_model_path = argv[2];

    // 生成随机输入数据
    std::vector<float> gstf_input_data;
    std::vector<float> local_input_data;
    generateRandomInput(gstf_input_data, 32 * 250 * 60);
    generateRandomInput(local_input_data, 32 * 15000);

    RKNNModel gstf(gstf_model_path, 250, 60, 32);
    RKNNModel local(local_model_path, 32, 15000);

    if (!gstf.test(gstf_input_data.data())) {
        std::cout << "gstf test failed" << std::endl;
        return -1;
    }
    if (!local.test(local_input_data.data())) {
        std::cout << "local test failed" << std::endl;
        return -1;
    }

    // 获取输出数据
    const std::vector<float>& s = gstf.getOutput(0);
    const std::vector<float>& h = gstf.getOutput(1);
    const std::vector<float>& f = local.getOutput(0);
    
    // 输出推理结果
    std::cout << "--Output s information--" << std::endl;
    printOutput(s);
    std::cout << "--Output h information--" << std::endl;
    printOutput(h);
    std::cout << "--Output f information--" << std::endl;
    printOutput(f);

    return 0;
}
