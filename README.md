# RSVPStream

本仓库提供 RK3588 CPU 与 FPGA 协同加速的 RSVPStream 推理链路：CPU 侧完成预处理与量化，FPGA 侧完成并行推理。

## 运行模式
- **file 模式**：启动时一次性加载参数并离线遍历 `.npz`；运行过程中不热更新。
- **queue 模式**：通过 socket 接收流式样本；每条样本处理完可检查更新标记并热更新参数。

## 准备数据与模型
- 数据：将脑电 `.npz` 放在 `data/eeg_data`（主程序默认）。
- FPGA 系数目录：`data/dat/`（或版本化目录 `data/dat_vN/`）。

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

下面给出 C++/Python 的常用运行命令，并对每个参数的作用、默认值、以及如何设置进行说明。

> 重要提示：
> - 默认路径已改为仓库相对路径（例如 `data/eeg_data`、`data/dat`）。因此建议在“仓库根目录”下运行，或显式传入绝对路径。
> - 所有相对路径都是相对于“启动命令时的当前工作目录（cwd）”解析的；如果你在其它目录执行 `./build/...`，请把 `--data_dir/--dat_dir/--scale_file` 改成绝对路径或正确的相对路径。
> - `--scale_file` 为空时，会自动使用 `(<dat_dir>/quantized_scales_hex.txt)`。

### file 模式（离线批处理）
```bash
./build/rsvp_fpga --mode file \
    --data_dir data/eeg_data \
    --dat_dir  data/dat \
    --scale_file data/dat/quantized_scales_hex.txt
```

参数说明（`./build/rsvp_fpga`）：
- `--mode`：运行模式。
    - `file`：离线遍历目录内的 `.npz` 文件逐条推理。
    - `queue`：在线模式，从 socket 接收流式样本。
    - 默认值：`file`。
- `--data_dir`：样本目录（目录内的 `.npz` 会被按字典序遍历）。
    - 默认值：`data/eeg_data`（相对仓库根目录）。
    - 如何设置：建议设为仓库内的 `data/eeg_data`（或你自己的样本目录）。每个 `.npz` 需要包含名为 `data` 的二维数组。
- `--dat_dir`：FPGA 系数（`.dat`）目录。
    - 默认值：`data/dat`（相对仓库根目录），且会通过环境变量 `RSVP_DAT_DIR` 影响底层加载逻辑。
    - 如何设置：指向包含如下文件的目录（至少需要这些基础文件，实际以程序加载为准）：
        - `ptrim.dat`、`bglobal.dat`、`betaglobal.dat`、`gstf_weight.dat`、`lr_model.dat` 以及 DDR 所需的若干 `.dat`。
- `--scale_file`：量化 scale 文件路径（十六进制 float 文本）。
    - 默认值：若不传，则自动使用 `(<dat_dir>/quantized_scales_hex.txt)`。
    - 如何设置：通常就放在 `dat_dir` 内，对应你当前版本的导出结果。
- `--update-flag`：热更新标记文件路径。
    - 在 `file` 模式下会被忽略（程序会提示 `--update-flag is ignored in file mode.`）。

运行行为：
- file 模式按字典序遍历 `.npz`，逐条处理；末尾输出 BA/ACC/TPR/FPR/AUC。
- file 模式不会检查任何控制标记文件，不会触发参数热更新（reload）。

### queue 模式（在线推理）
需同时启动接收端（主程序）与发送端（模拟器）。

接收端：
```bash
./build/rsvp_fpga --mode queue \
    --listen-port 5001 --max-samples 0 \
    --dat_dir  data/dat \
    --scale_file data/dat/quantized_scales_hex.txt \
    --update-flag /tmp/rsvp_param_update.json
```

参数说明（queue 模式特有/常用项）：
- `--listen-port`：接收端监听端口（TCP）。
    - 默认值：`5001`。
    - 如何设置：发送端的 `--port` 必须与之保持一致；若端口被占用请改成空闲端口。
- `--max-samples`：queue 模式下最多消费多少条样本。
    - 默认值：`10`。
    - 如何设置：
        - 设为 `0` 表示不限制，直到发送端发“终止信号”（发送 0 行/0 列）或你手动停止程序。
        - 设为正整数 `N` 则消费 `N` 条后退出。
- `--update-flag`：热更新标记文件路径（仅 queue 模式生效）。
    - 默认值：空（不开启热更新）。
    - 如何设置：建议使用绝对路径，例如 `/tmp/rsvp_param_update.json`；该文件内容应为 JSON，至少包含 `dat_dir` 字段。

运行行为：
- queue 模式只做在线推理与打印输出，不做最终 BA/ACC/TPR/FPR/AUC 评估（返回占位值）。
- 每处理完一条样本，会检查 `--update-flag` 是否更新；如更新则尝试热加载新参数。

发送端（socket 推送）：
```bash
./build/eeg_sender --host 127.0.0.1 --port 5001 \
    --data_dir data/eeg_data \
    --max-samples 0 --interval-ms 1000
```

参数说明（`./build/eeg_sender`）：
- `--host`：接收端 IP。
    - 默认值：`127.0.0.1`。
    - 如何设置：
        - 同机模拟：用 `127.0.0.1`。
        - 跨机推送：填接收端机器的可达 IP。
- `--port`：接收端端口。
    - 默认值：`5001`。
    - 如何设置：必须与接收端 `--listen-port` 一致。
- `--data_dir`：待推送的 `.npz` 目录。
    - 默认值：`data/eeg_data`（相对仓库根目录）。
    - 如何设置：建议显式传 `data/eeg_data` 或自定义目录；程序会按字典序遍历目录下文件，并从每个 `.npz` 中读取名为 `data` 的二维数组。
- `--max-samples`：最多推送多少条。
    - 默认值：`10`。
    - 如何设置：设为 `0` 表示推送目录内全部样本（直到遍历结束），最后会自动发送终止信号。
- `--interval-ms`：两条样本之间的发送间隔（毫秒）。
    - 默认值：`1000`。
    - 如何设置：根据实时性/吞吐需求调整；过小可能导致接收端压力上升。

## 热更新（仅 queue 模式）
queue 模式会在“每处理完一条样本”后检查 `--update-flag` 指向的标记文件是否更新；若更新则触发参数热重载：
1) `load_coeff()` 重新加载寄存器系数；
2) 将 4 个 DDR 系数文件写入 FPGA DDR；
3) 读取 scale 文件并覆盖写入寄存器（替换 `NetRegInit()` 的默认常量）。

### 标记文件格式（JSON）
建议路径：`/tmp/rsvp_param_update.json`

内容示例：
```json
{"dat_dir":"data/dat_v2","scale_file":"data/dat_v2/quantized_scales_hex.txt","version":"v2"}
```

字段说明：
- `dat_dir`（必填）：新的参数目录。建议写绝对路径；相对路径会以接收端当前工作目录为基准。
- `scale_file`（可选）：新的 scale 文件路径。
    - 若省略，接收端会自动取 `(<dat_dir>/quantized_scales_hex.txt)`。
- `version`（可选）：仅用于记录/调试，接收端不会使用该字段。

### Python 生成参数并原子通知
```bash
python3 python/gen_dat_from_pt.py \
    --checkpoint data/model/Model_1_MG_50.pt \
    --out-dir data/dat \
    --version v2 \
    --update-flag /tmp/rsvp_param_update.json
```

参数说明（`python/gen_dat_from_pt.py`）：
- `--checkpoint`/`-c`：原始 PyTorch checkpoint 路径。
    - 默认值：`data/model/Model_1_MG_50.pt`。
    - 如何设置：指向你要导出的模型；若不想从 checkpoint 导出，可改用 `--params`。
- `--params`/`-p`：已转换好的 `converted_parameters.pt` 路径（仅当不使用 `--checkpoint` 时生效）。
    - 默认值：`data/model/converted_parameters.pt`。
- `--unsafe-load`：强制使用 `torch.load(weights_only=False)`（用于需要自定义类的 checkpoint）。
    - 默认值：关闭。
    - 如何设置：只有在你确认 checkpoint 来源可信且安全加载失败时才启用。
- `--out-dir`/`-o`：输出系数的“基础目录”。
    - 默认值：`data/dat`。
    - 如何设置：建议与 C++ 侧 `--dat_dir` 的“基础前缀”一致。
- `--version`：输出版本号。
    - 默认值：不设置（直接写入 `--out-dir` 目录）。
    - 如何设置：例如 `v2`，则输出目录会变为 `data/dat_v2/`（规则：`<out_dir>_<version>`）。
- `--update-flag`：原子写入 JSON 标记文件，用于通知 queue 模式接收端热更新。
    - 默认值：不设置（只生成文件，不通知）。
    - 如何设置：建议使用 `/tmp/rsvp_param_update.json`；脚本会写入绝对路径的 `dat_dir/scale_file`。

这会生成目录 `data/dat_v2/` 并触发接收端在样本间隙热更新。

也可以用一键脚本（更方便）：
```bash
chmod +x scripts/gen_dat_and_notify.sh
scripts/gen_dat_and_notify.sh v2
```

脚本参数说明（`scripts/gen_dat_and_notify.sh`）：
- 第 1 个参数：`version`（默认 `v2`）。
- 第 2 个参数：`checkpoint`（默认 `data/model/Model_1_MG_50.pt`）。
- 第 3 个参数：`out_dir_base`（默认 `data/dat`）。
- 第 4 个参数：`update_flag`（默认 `/tmp/rsvp_param_update.json`）。

示例：
```bash
scripts/gen_dat_and_notify.sh v3 data/model/Model_1_MG_50.pt data/dat /tmp/rsvp_param_update.json
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