# RSVPStream

本仓库提供 RK3588 CPU 与 FPGA 协同加速的 RSVPStream 推理链路：CPU 侧完成预处理与量化，FPGA 侧完成并行推理。

## 运行模式
- **file 模式**：启动时一次性加载参数并离线遍历 `.npz`；运行过程中不热更新。
- **queue 模式**：通过 socket 接收流式样本；每条样本处理完可检查更新标记并热更新参数。

## 准备数据与模型
- 数据：将脑电 `.npz` 放在 `/home/hzhy/RSVPStream/data/egg_data`（主程序默认）。
- FPGA 系数目录：`/home/hzhy/RSVPStream/data/dat/`（或版本化目录 `data/dat_vN/`）。

## 上电与驱动加载
1) 板卡上电，等待 FPGA 完成自举。
2) 上电后建议先在 RK3588 上执行一次 `sudo reboot`。
3) 重启后在仓库根目录加载驱动：`sudo insmod ./xdma.ko`。
4) 确认 `/dev/xdma*` 设备节点存在（例如 `/dev/xdma0_user`、`/dev/xdma0_h2c_0`）。

## 编译（CMake）
```bash
cmake -S . -B build
cmake --build build
```

生成：
- `./build/rsvp_fpga`（主程序）
- `./build/eeg_sender`（数据推送模拟器）

## 运行

### file 模式（离线批处理）
```bash
./build/rsvp_fpga --mode file \
    --data_dir /home/hzhy/RSVPStream/data/egg_data \
    --dat_dir  /home/hzhy/RSVPStream/data/dat \
    --scale_file /home/hzhy/RSVPStream/data/dat/quantized_scales_hex.txt
```
- 按字典序遍历 npz，逐条处理；末尾输出 BA/ACC/TPR/FPR/AUC。
- file 模式不会检查任何控制标记文件，不会触发 reload。

### queue 模式（在线推理）
需同时启动接收端（主程序）与发送端（模拟器）。

接收端：
```bash
./build/rsvp_fpga --mode queue \
    --listen-port 5001 --max-samples 0 \
    --dat_dir  /home/hzhy/RSVPStream/data/dat \
    --scale_file /home/hzhy/RSVPStream/data/dat/quantized_scales_hex.txt \
    --update-flag /tmp/rsvp_param_update.json
```
- `--max-samples 0` 表示不设上限，直到发送端发终止信号。
- `--update-flag` 仅 queue 模式启用；file 模式会忽略该参数。

发送端（socket 推送）：
```bash
./build/eeg_sender --host 127.0.0.1 --port 5001 \
    --data_dir /home/hzhy/RSVPStream/data/egg_data \
    --max-samples 0 --interval-ms 1000
```

## 热更新（仅 queue 模式）
queue 模式会在“每处理完一条样本”后检查 `--update-flag` 指向的标记文件是否更新；若更新则触发参数热重载：
1) `load_coeff()` 重新加载寄存器系数；
2) 将 4 个 DDR 系数文件写入 FPGA DDR；
3) 读取 scale 文件并覆盖写入寄存器（替换 `NetRegInit()` 的默认常量）。

### 标记文件格式（JSON）
建议路径：`/tmp/rsvp_param_update.json`

内容示例：
```json
{"dat_dir":"/home/hzhy/RSVPStream/data/dat_v2","scale_file":"/home/hzhy/RSVPStream/data/dat_v2/quantized_scales_hex.txt","version":"v2"}
```

### Python 生成参数并原子通知
```bash
python3 /home/hzhy/RSVPStream/python/gen_dat_from_pt.py \
    --checkpoint /home/hzhy/RSVPStream/data/model/Model_1_MG_50.pt \
    --out-dir /home/hzhy/RSVPStream/data/dat \
    --version v2 \
    --update-flag /tmp/rsvp_param_update.json
```
这会生成目录 `/home/hzhy/RSVPStream/data/dat_v2/` 并触发接收端在样本间隙热更新。

也可以用一键脚本（更方便）：
```bash
chmod +x /home/hzhy/RSVPStream/scripts/gen_dat_and_notify.sh
/home/hzhy/RSVPStream/scripts/gen_dat_and_notify.sh v2
```

## 常见问题
- 启动时报错找不到 `/dev/xdma*`：先确认已 `insmod xdma.ko` 且节点存在。
- `--update-flag` 路径建议用绝对路径（例如 `/tmp/rsvp_param_update.json`）。
- 出现 `betaglobal.dat size ... smaller than expected ...`：表示文件比历史“期望长度”短，但仍会写入已读取的元素；如需严格对齐可再调整导出脚本与读取长度。

## 目录速览
- `src/`：C++ 主程序与推理/热更新逻辑。
- `include/`：头文件。
- `python/`：参数导出脚本（生成 `data/dat_vN/` 与标记文件）。
- `scripts/`：一键生成并通知脚本。
- `data/`：样本与导出的 FPGA 参数目录。