#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <mutex>
#include <optional>

class FPGAProcessor {
private:
    float s_all;
    float s_data;
    bool bErr;
    std::filesystem::path dat_dir_;
    std::filesystem::path scale_file_;
    std::mutex reload_mtx_;

public:
    FPGAProcessor(const std::filesystem::path& dat_dir,
                  const std::filesystem::path& scale_file = {});
    float fpga_runner(const std::vector<std::vector<float>>& x_local_data, const std::vector<std::vector<float>>& x_global_data);
    bool reload_parameters(const std::filesystem::path& dat_dir,
                           const std::filesystem::path& scale_file);
};