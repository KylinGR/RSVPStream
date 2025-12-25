#include "fpga_runner.h"
#include "fpga_lib.h"
#include "utils.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <filesystem>
#include <optional>
#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace {
std::filesystem::path default_dat_dir() {
    const char* env = std::getenv("RSVP_DAT_DIR");
    if (env && *env) return std::filesystem::path(env);
    return std::filesystem::path("/home/hzhy/RSVPStream/data/dat");
}

struct ScaleSet {
    float w_global = 0.f;
    float q_global = 0.f;
    float b_global = 0.f;
    float beta_global = 0.f;
    float beta_local = 0.f;
    float w_local = 0.f;
    float b_local = 0.f;
    float lr_model = 0.f;
    float gstf_weight = 0.f;
    float m_global = 0.f;
    float m_local = 0.f;
    float merge_g = 0.f; // merge_isSg_Gamma_global_scale
    float merge_l = 0.f; // merge_isSl_gamma_local_scale
};

std::optional<float> parse_hex_float(const std::string& token) {
    errno = 0;
    char* end = nullptr;
    const unsigned long v = std::strtoul(token.c_str(), &end, 16);
    if (errno != 0 || end == token.c_str()) return std::nullopt;
    const uint32_t val = static_cast<uint32_t>(v);
    float f = 0.f;
    std::memcpy(&f, &val, sizeof(float));
    return f;
}

std::optional<ScaleSet> load_scales_file(const std::filesystem::path& scale_file) {
    if (scale_file.empty()) return std::nullopt;
    FILE* fp = std::fopen(scale_file.c_str(), "r");
    if (!fp) return std::nullopt;

    std::unordered_map<std::string, float> kv;
    auto trim_inplace = [](std::string& s) {
        const auto first = s.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            s.clear();
            return;
        }
        const auto last = s.find_last_not_of(" \t\r\n");
        s = s.substr(first, last - first + 1);
    };

    char buf[512];
    while (std::fgets(buf, sizeof(buf), fp) != nullptr) {
        std::string line(buf);
        auto pos = line.find(':');
        if (pos == std::string::npos) continue;
        std::string key = line.substr(0, pos);
        std::string val = line.substr(pos + 1);
        trim_inplace(key);
        trim_inplace(val);
        auto parsed = parse_hex_float(val);
        if (parsed) kv[key] = *parsed;
    }
    std::fclose(fp);

    ScaleSet s;

    const char* required_keys[] = {
        "W_global_scale",
        "Q_global_scale",
        "b_global_scale",
        "Beta_global_scale",
        "beta_local_scale",
        "w_local_scale",
        "b_local_scale",
        "lr_model_scale",
        "gstf_weight_scale",
        "M_global_scale",
        "M_local_scale",
        "merge_isSg_Gamma_global_scale",
        "merge_isSl_gamma_local_scale",
    };
    for (const char* k : required_keys) {
        if (!kv.count(k)) {
            std::fprintf(stderr, "scale file missing key: %s\n", k);
            return std::nullopt;
        }
    }

    s.w_global = kv["W_global_scale"];
    s.q_global = kv["Q_global_scale"];
    s.b_global = kv["b_global_scale"];
    s.beta_global = kv["Beta_global_scale"];
    s.beta_local = kv["beta_local_scale"];
    s.w_local = kv["w_local_scale"];
    s.b_local = kv["b_local_scale"];
    s.lr_model = kv["lr_model_scale"];
    s.gstf_weight = kv["gstf_weight_scale"];
    s.m_global = kv["M_global_scale"];
    s.m_local = kv["M_local_scale"];
    s.merge_g = kv["merge_isSg_Gamma_global_scale"];
    s.merge_l = kv["merge_isSl_gamma_local_scale"];

    return s;
}

void apply_scales(const ScaleSet& s) {
    // Always overwrite NetRegInit constants.
    RegWr(ADR_WR_M_LOCAL_SCALE, floatToHex(s.m_local));
    RegWr(ADR_GAMMA_LOCAL_SCALE, floatToHex(s.merge_l));
    RegWr(ADR_BETA_LOCAL_SCALE, floatToHex(s.beta_local));
    RegWr(ADR_B_LOCAL_SCALE, floatToHex(s.b_local));
    RegWr(ADR_W_LOCAL_SCALE, floatToHex(s.w_local));
    RegWr(ADR_LR_MODEL_SCALE, floatToHex(s.lr_model));

    RegWr(ADR_GAMMA_GLOBSCALE, floatToHex(s.merge_g));
    RegWr(ADR_MGLOBAL_SCALE, floatToHex(s.m_global));
    RegWr(ADR_BETAGLOB_SCALE, floatToHex(s.beta_global));
    RegWr(ADR_WGLOBAL_SCALE, floatToHex(s.w_global));
    RegWr(ADR_QGLOB_SCALE, floatToHex(s.q_global));
    RegWr(ADR_BGLOBAL_SCALE, floatToHex(s.b_global));
    RegWr(ADR_GSTF_WEIGHT_SCALE, floatToHex(s.gstf_weight));
}
}

FPGAProcessor::FPGAProcessor(const std::filesystem::path& dat_dir,
                             const std::filesystem::path& scale_file)
    : s_all(0), s_data(0), bErr(false), dat_dir_(dat_dir.empty() ? default_dat_dir() : dat_dir),
      scale_file_(scale_file) {
    std::cout << "Hello,World!" << __DATE__ << "," << __TIME__ << std::endl;
    if (!scale_file_.empty() && !std::filesystem::exists(scale_file_)) {
        scale_file_.clear();
    }

    if (InitFPGA() != 0) {
        throw std::runtime_error("FPGA初始化失败");
    }

    set_dat_dir(dat_dir_);
    Reset();
    load_coeff();

    auto load_ddr = [&](const std::string& filename, uint32_t addr) {
        std::string path = (dat_dir_ / filename).string();
        File2DDR(path, addr);
    };

    load_ddr("local_coef.dat", LOCAL_COEFF_ADR);
    load_ddr("b_local.dat", BLOCAL_COEFF_ADR);
    load_ddr("global_coef.dat", GLOBAL_COEFF_ADR);
    load_ddr("Q_global.dat", QGLOB_COEFF_ADR);

    NetRegInit();
    if (!scale_file_.empty()) {
        if (auto scales = load_scales_file(scale_file_)) {
            apply_scales(*scales);
        }
    }

    int32_t dat = RegRd(0xc000);
    printf("FPGA VERSION =0x%x\n",dat);

    dat = RegRd(ADR_ALG_START);
    printf("ADR_ALG_START=%x\n", dat);

    dat = RegRd(ADR_HW_STATUS);
    printf("ADR_HW_STATUS=%x\n", dat);
}

bool FPGAProcessor::reload_parameters(const std::filesystem::path& dat_dir,
                                      const std::filesystem::path& scale_file) {
    std::lock_guard<std::mutex> lock(reload_mtx_);
    std::filesystem::path new_dat = dat_dir.empty() ? dat_dir_ : dat_dir;
    std::filesystem::path new_scale = scale_file.empty() ? scale_file_ : scale_file;

    set_dat_dir(new_dat);
    try {
        load_coeff();
        auto load_ddr = [&](const std::string& filename, uint32_t addr) {
            std::string path = (new_dat / filename).string();
            File2DDR(path, addr);
        };
        load_ddr("local_coef.dat", LOCAL_COEFF_ADR);
        load_ddr("b_local.dat", BLOCAL_COEFF_ADR);
        load_ddr("global_coef.dat", GLOBAL_COEFF_ADR);
        load_ddr("Q_global.dat", QGLOB_COEFF_ADR);

        // scales
        if (!new_scale.empty() && std::filesystem::exists(new_scale)) {
            if (auto scales = load_scales_file(new_scale)) {
                apply_scales(*scales);
            }
        }

        dat_dir_ = new_dat;
        scale_file_ = new_scale;
        std::cout << "Parameters reloaded from " << new_dat << std::endl;
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "reload_parameters failed: " << ex.what() << std::endl;
        return false;
    }
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