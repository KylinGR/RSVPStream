#include "xgbdim.h"
#include <iostream>
#include <iomanip>

int main() {
    std::string model_path = "../model.npz"; //模型参数
    std::string rknn_model_path = "../rknn_model/optimized_model_v3_1.rknn";
    XGBDIM xgb(model_path, rknn_model_path, 6, 3, 3, 3, 3, 299, 0.3);
    std::cout << "Starting program..." << std::endl;
    auto [ba, acc, tpr, fpr, auc] = xgb.test();
    std::cout << std::setprecision(4) << std::fixed << "BA: " << ba << ", ACC: " << acc << ", TPR: " << tpr << ", FPR: " << fpr << ", AUC: " << auc << std::endl;

    return 0;
}

