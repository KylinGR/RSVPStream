#include "rknn_model.h"

// 构造函数
RKNNModel::RKNNModel(const char* model_path, int dim1, int dim2, int batch_size)
    : model_path(model_path), context(0), dim1(dim1), dim2(dim2), batch_size(batch_size) {
    if (!initialize()) {
        throw std::runtime_error("Failed to initialize model.");
    }
}

// 析构函数
RKNNModel::~RKNNModel() {
    if (context) {
        rknn_destroy(context);
    }
}

// 初始化模型
bool RKNNModel::initialize() {
    if (rknn_init(&context, (void*)model_path, 0, 0, NULL) != RKNN_SUCC) {
        std::cerr << "Failed to initialize RKNN model." << std::endl;
        return false;
    }

    if (rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) != RKNN_SUCC) {
        std::cerr << "Failed to query RKNN model IO info." << std::endl;
        return false;
    }

    std::cout << "Model input num: " << io_num.n_input
              << ", output num: " << io_num.n_output << std::endl;

    outputs.resize(io_num.n_output);  // 初始化输出容器
    return true;
}

// 设置输入
bool RKNNModel::setInput(const float* input_data) {
    rknn_input input[io_num.n_input];
    memset(input, 0, sizeof(input));

    input[0].index = 0;
    input[0].buf = const_cast<float*>(input_data);
    input[0].size = batch_size * dim1 * dim2 * sizeof(float);
    input[0].pass_through = 0;
    input[0].type = RKNN_TENSOR_FLOAT32;
    input[0].fmt = RKNN_TENSOR_UNDEFINED;

    if (rknn_inputs_set(context, io_num.n_input, input) != RKNN_SUCC) {
        std::cerr << "Failed to set input." << std::endl;
        return false;
    }
    return true;
}

// 运行推理
bool RKNNModel::runInference() {
    if (rknn_run(context, NULL) != RKNN_SUCC) {
        std::cerr << "Failed to run inference." << std::endl;
        return false;
    }

    rknn_output output[io_num.n_output];
    memset(output, 0, sizeof(output));

    for (int i = 0; i < io_num.n_output; ++i) {
        output[i].index = i;
        output[i].is_prealloc = 0;
        output[i].want_float = 1;
    }

    if (rknn_outputs_get(context, io_num.n_output, output, NULL) != RKNN_SUCC) {
        std::cerr << "Failed to get outputs." << std::endl;
        return false;
    }

    // 清空旧的输出数据
    for (auto& out : outputs) {
        out.clear();
    }

    // 提取输出数据
    for (int i = 0; i < io_num.n_output; ++i) {
        float* buf = (float*)output[i].buf;
        size_t size = output[i].size / sizeof(float);
        outputs[i].assign(buf, buf + size);
    }

    rknn_outputs_release(context, io_num.n_output, output);
    return true;
}

// 推理测试
bool RKNNModel::test(const float* input_data) {
    return setInput(input_data) && runInference();
}

// 获取指定索引的推理结果
const std::vector<float>& RKNNModel::getOutput(size_t index) const {
    if (index >= outputs.size()) {
        throw std::out_of_range("Output index out of range.");
    }
    return outputs[index];
}
