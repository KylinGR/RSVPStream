# RSVPStream `src/` 目录说明

本目录存放 RK3588 CPU 与 FPGA 协同加速的 RSVPStream 推理链路核心源码。系统当前以离线 `.npz` 脑电数据为输入（后续可替换为实时脑电帽采集），经 CPU 预处理、NEON 量化后，通过 PCIe 发送到 FPGA 并行推理，再将结果取回并统计指标，单样本端到端耗时可控在 30 ms 以内。

## ⚙️ 已实现的关键优化

- **滑窗 3D 立体特征构造**：`xgbdim.cpp` 内的 `get_3Dconv()` 和 `get_3D_cuboids()` 根据设定的空间/时间步长生成局部立体块，压缩原始 60×250 通道数据的搜索空间。
- **均值消除 + 浮点压缩**：`preprocess()` 对每个通道减均值，随后利用 `utils.cpp` 中的 `flatten_2d_vector()` 将二维矩阵展平成紧凑连续内存，降低缓存 miss。
- **NEON 动态量化**：`utils.cpp` 的 `dynamic_quantize_tensor_T_flatten_neon()` 在 ARM 平台启用 NEON 指令一次处理 4 个样本，输出 `int16_t` 张量和尺度因子，兼容非 NEON 平台自动回退标量实现。
- **零拷贝 PCIe 传输**：`fpga_runner.cpp` 通过 `Vec2DDR()` 将量化后向量直接写入 FPGA DDR，避免中间缓冲复制；尺度因子同步写入控制寄存器。
- **FPGA 并行推理内核**：`fpga_runner.cpp::Start()` 触发硬件流水线，读取 17 个神经元累加值后经 Sigmoid 归一化得到样本概率。
- **细粒度性能计时**：量化阶段、FPGA 总循环 (`xgbdim.cpp`) 以及 FPGA 内部运行时间寄存器 (`ADR_TIM`) 均输出至日志，便于定位瓶颈。
- **稳健的错误处理**：FPGA 初始化失败、安全寄存器校验等情况会抛出异常或打印调试信息，方便上电阶段诊断。

## 🗂️ 目录结构与作用

| 文件 | 主要职责 |
|------|----------|
| `main.cpp` | 命令行入口，配置 XGBDIM 参数、触发 `test()` 并打印 BA/ACC/TPR/FPR/AUC。 |
| `xgbdim.cpp` | 核心算法层：数据读取（`load_npz_2d_array_float`）、均值消除、3D 立体块生成、批量遍历样本并调用 FPGA。 |
| `utils.cpp` | 通用辅助：NPZ 载入、向量展平、NEON/标量量化、Sigmoid、计时工具等。 |
| `fpga_runner.cpp` | FPGA 交互层：初始化、寄存器访问、数据量化后写入 DDR、触发运算及结果汇总。 |
| `fpga_lib.cpp` / `file_base.cpp` | 底层驱动封装和文件到 DDR 的搬运实现。 |

> 相关头文件位于 `../include/`。运行时需要的 `.dat` 系数文件默认路径在 `fpga_runner.cpp` 中写死，可按需调整。

## 🚀 部署与运行

1. **准备数据与模型**
  - 在仓库根目录创建 `data/`，将脑电 `.npz` 数据放置于 `data/egg_data/`；`main.cpp` 默认模型路径为 `../model.npz`，可视情况修改。

2. **上电与驱动加载**
  - 先给板卡整体上电并等待 FPGA 初始化完成（耗时略长）。
  - 通过串口/SSH 登录 RK3588，执行：

    ```bash
    sudo reboot
    ```

  - 重启后插入 PCIe 驱动：

    ```bash
    sudo insmod /path/to/xdma.ko
    ```

  - 查看 `/dev/xdma*` 是否生成对应设备节点，确认驱动加载成功。

3. **编译工程**

  ```bash
  cd /home/hzhy/csk/RSVPStream
  mkdir -p build
  cd build
  cmake ..
  make -j8
  ```

4. **执行推理与评估**
  - 切换至构建目录，运行示例：

    ```bash
    ./rsvp_rknn_async
    ```

  - 程序会遍历 `data/egg_data` 下的所有样本，输出每个样本的概率、量化耗时以及整体统计指标。

## ✅ 调试与验证建议

- 若 FPGA 初始化失败，检查 `fpga_runner.cpp` 中的系数文件路径以及 `InitFPGA()` 返回值。
- 观察日志中的 “Quantize cost X ms” 与 “R[ADR_TIM]=Y us” 以评估 CPU 与 FPGA 两阶段负载。
- 单样本耗时超过 30 ms 时，可优先检查数据量化是否退化为标量实现（确认编译器定义了 `__ARM_NEON`）。
- 需要更换模型或窗口参数时，请同时更新 `main.cpp` 中的构造参数与 `xgbdim.cpp` 内部的通道映射。

更多系统级优化（如实时采集、异步调度、批推理策略）可参考仓库根目录的 `PERFORMANCE_OPTIMIZATION.md`，并据此迭代本目录代码。