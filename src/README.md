# RSVPStream `src/` 目录说明

本目录存放 RK3588 CPU 与 FPGA 协同加速的 RSVPStream 推理链路核心源码。系统当前以离线 `.npz` 脑电数据为输入（后续可替换为实时脑电帽采集），经 CPU 预处理、NEON 量化后，通过 PCIe 发送到 FPGA 并行推理，再将结果取回并统计指标。

## ⚙️ 已实现的关键优化
- 滑窗 3D 立体特征构造：`xgbdim.cpp` 的 `get_3Dconv()`/`get_3D_cuboids()` 按空间/时间步长生成立体块。
- 均值消除 + 浮点压缩：`preprocess()` 减均值，`flatten_2d_vector()` 展平矩阵，提升缓存亲和。
- NEON 动态量化：`dynamic_quantize_tensor_T_flatten_neon()` 在 ARM 平台启用 NEON，一次处理 4 个样本，非 NEON 自动回退标量实现。
- 零拷贝 PCIe 传输：`Vec2DDR()` 直接写 FPGA DDR，尺度因子写控制寄存器。
- FPGA 并行推理：`fpga_runner.cpp::Start()` 触发流水，读取 17 个神经元累加值经 Sigmoid 得到概率。
- 性能计时：量化耗时、FPGA 循环耗时 (`xgbdim.cpp`)、FPGA 内部运行时间寄存器 (`ADR_TIM`) 均输出日志。

## 🗂️ 目录结构与作用
| 文件 | 主要职责 |
| --- | --- |
| `main.cpp` | CLI 入口，配置 XGBDIM，选择模式并打印指标。 |
| `xgbdim.cpp` | 数据读取/预处理/3D 立体块生成，批量遍历样本并调用 FPGA。 |
| `utils.cpp` | NPZ 载入、展平、量化（NEON/标量）、工具函数。 |
| `fpga_runner.cpp` | FPGA 交互：初始化、寄存器访问、DDR 传输、启动与汇总。 |
| `fpga_lib.cpp` / `file_base.cpp` | 底层驱动封装与文件到 DDR 搬运。 |

> 头文件位于 `../include/`，FPGA 系数文件路径在 `fpga_runner.cpp` 中写死，可按需调整。

## 准备数据与模型
- 数据：将脑电 `.npz` 放在 `/home/hzhy/RSVPStream/data/egg_data`（主程序默认）。
- FPGA 系数：`/home/hzhy/RSVPStream/data/dat/*.dat`，随上电加载。

## 上电与驱动加载
1) 板卡上电，等待 FPGA 完成自举。 
2) 在 RK3588 上加载驱动：`sudo insmod /path/to/xdma.ko`。
3) 确认 `/dev/xdma*` 设备节点存在。 

## 编译（CMake）
```bash
cmake -S . -B build
cmake --build build
```
生成：`./build/rsvp_rknn`（主程序）、`./build/eeg_sender`（数据推送模拟器）。

## 运行

### file 模式（离线批处理）
```bash
./build/rsvp_rknn --mode file --data_dir /home/hzhy/RSVPStream/data/egg_data
```
- 按字典序遍历 npz，逐条处理，末尾输出 FPGA 耗时直方图与 BA/ACC/TPR/FPR/AUC。

### queue 模式（在线推理）
需同时启动接收端（主程序）与发送端（模拟器）。

接收端：
```bash
./build/rsvp_rknn --mode queue --listen-port 5001 --max-samples 0
```
- `--listen-port`：当前 socket 监听端口（若未来更换通信方式，可忽略此参数）。
- `--max-samples`：最大处理样本数；0 表示不设上限，直到收到终止信号（空样本 / rows=0, cols=0）。
- 模型/FPGA 初始化完成后会打印“模型初始化完毕，开始等待接收数据...”，随后阻塞等待队列数据。
- 队列模式只做实时逐条输出，不计算末尾指标（占位 0）。

发送端（当前为 socket 推送）：
```bash
./build/eeg_sender --host 127.0.0.1 --port 5001 \
  --data_dir /home/hzhy/RSVPStream/data/egg_data \
  --max-samples 0 --interval-ms 1000
```
- `--max-samples`：发送上限；0 表示不限（由文件数或终止信号决定）。
- `--interval-ms`：发送间隔（毫秒）。
- 文件顺序：目录遍历后 `std::sort` 字典序，零填充编号 `..._001.npz` 等会按递增发送。
- 终止信号：发送端发送 rows=0, cols=0；接收端收到后向队列 `push({})`，消费循环退出。

## 替换通信方式指南
- 保持消费端接口不变：`EEGSampleQueue::push(sample)` 推样本，空样本 `{}` 终止；`XGBDIM::test(...use_queue=true)` 阻塞 `pop_blocking()`。
- 若改串口/共享内存/消息队列等，只需实现新的生产者，将解码后的 `std::vector<std::vector<float>>` 按上述约定推入队列，并发送终止空样本。
- 主程序消费逻辑与指标行为无需调整。

## 常用命令速查
```bash
# 构建
cmake -S . -B build && cmake --build build

# 文件模式
./build/rsvp_rknn --mode file --data_dir /home/hzhy/RSVPStream/data/egg_data

# 队列模式（监听 5001，无上限）
./build/rsvp_rknn --mode queue --listen-port 5001 --max-samples 0

# 推送端（本机 5001，1s 间隔，全量发送）
./build/eeg_sender --host 127.0.0.1 --port 5001 \
  --data_dir /home/hzhy/RSVPStream/data/egg_data --max-samples 0 --interval-ms 1000
```