#include <iostream>
#include <vector>
#include <string>

class FPGAProcessor {
private:
    float s_all;
    float s_data;
    bool bErr;

public:
    FPGAProcessor();
    float fpga_runner(const std::vector<std::vector<float>>& x_local_data, const std::vector<std::vector<float>>& x_global_data);
};