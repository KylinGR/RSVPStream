import numpy as np
from scipy.signal import butter, filtfilt

class EEGPreprocess():
    def __init__(self, fs=1000, low_cut=0.5, high_cut=49, filter_order=4):
        """
        初始化EEGPreprocess类。
        :param fs: 采样频率（Hz）
        :param low_cut: 带通滤波器下限（Hz）
        :param high_cut: 带通滤波器上限（Hz）

        """
        self.fs = fs
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.order = filter_order

    # 初始化滤波器
    def butter_bandpass(self, low_cut, high_cut, fs, order=4):
        nyquist = 0.5 * fs  # 奈奎斯特频率
        low = low_cut / nyquist
        high = high_cut / nyquist
        b, a = butter(order, [low, high], btype='band')  # 设计带通滤波器
        return b, a

    def filtering(self, x):
        b, a = self.butter_bandpass(self.low_cut, self.high_cut, self.fs, self.order)
        y = filtfilt(b, a, x)
        return y

    # z-score
    def zscore(self, x):
        # 去均值
        x = x - np.mean(x, axis=-1, keepdims=True)
        # 对每个通道进行归一化（按标准差）
        std_dev = np.std(x, axis=-1, keepdims=True)
        # 防止除以零的情况，如果标准差为零，则不做归一化
        std_dev[std_dev == 0] = 1
        x = x / std_dev
        return x

    def forward(self, x):
        # remove
        x_new = np.delete(x, [32, 42, 59, 63], axis=0)
        # 滤波
        x = self.filtering(x)
        # 归一化
        x = self.zscore(x)
        return x

