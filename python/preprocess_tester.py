import numpy as np
import matplotlib.pyplot as plt
from eeg_preprocess import EEGPreprocess

# 创建一个64x256的随机EEG数据（64个通道，256个采样点）
fs = 1000  # 采样频率 1000 Hz
num_channels = 64
num_samples = 256

# 模拟EEG信号，每个通道一个随机信号
np.random.seed(42)  # 设置随机种子，确保结果可重现
eeg_data = np.random.randn(num_channels, num_samples)  # 生成随机数据

# 初始化EEGPreprocess对象
eeg_preprocess = EEGPreprocess(fs=fs, low_cut=0.5, high_cut=49, filter_order=4)

# 使用forward方法对每个通道的EEG信号进行滤波
filtered_eeg_data = eeg_preprocess.forward(eeg_data)

# 绘制某个通道的原始信号与滤波后的信号
channel_idx = 0  # 选择第一个通道进行展示

plt.figure(figsize=(10, 6))

# 绘制原始信号
plt.subplot(2, 1, 1)
plt.plot(eeg_data[channel_idx], label="Original Signal", color='gray')
plt.title(f"Original EEG Signal (Channel {channel_idx + 1})")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()

# 绘制滤波后的信号
plt.subplot(2, 1, 2)
plt.plot(filtered_eeg_data[channel_idx], label="Filtered Signal", color='b')
plt.title(f"Filtered EEG Signal (Channel {channel_idx + 1})")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()
