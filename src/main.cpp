#include "xgbdim.h"
#include <iostream>
#include <iomanip>

int main() {
    std::string model_path = "../model.npz"; //model_order
    std::string data_dir = "/home/hzhy/csk/RSVPStream/data/egg_data";
    XGBDIM xgb(1, model_path, 50, 6, 3, 3, 3, 3, 299, 0.3);
    std::cout << "Starting program..." << std::endl;
    auto [ba, acc, tpr, fpr, auc] = xgb.test(data_dir);
    std::cout << std::setprecision(4) << std::fixed << "BA: " << ba << ", ACC: " << acc << ", TPR: " << tpr << ", FPR: " << fpr << ", AUC: " << auc << std::endl;

    return 0;
}

