import numpy as np
import matplotlib.pyplot as plt
import os  # 导入 os 模块

# 定义信号
t = np.linspace(0, 1, 1000)
f = 5  # 信号频率 5 Hz
x = np.sin(2 * np.pi * f * t)

# 计算傅里叶变换
X_f = np.fft.fft(x)
frequencies = np.fft.fftfreq(len(t), t[1] - t[0])

# 归一化幅值
X_f_abs = np.abs(X_f) / len(t)

# 只取正频率部分
pos_mask = frequencies >= 0
frequencies_pos = frequencies[pos_mask]
X_f_abs_pos = X_f_abs[pos_mask]

# 绘制时域与频域对比图
fig = plt.figure(figsize=(12, 5))  # 获取 figure 对象

# 时域图
plt.subplot(1, 2, 1)
plt.plot(t, x)
plt.title('Time Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# 频域图
plt.subplot(1, 2, 2)
plt.stem(frequencies_pos, X_f_abs_pos)
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Amplitude')
plt.xlim(0, 20)  # 合理缩放，只显示0~20Hz
plt.grid(True)

plt.tight_layout()

# 创建输出目录
script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, "Fourier_Transform - output")
os.makedirs(output_dir, exist_ok=True)

# 保存图像
output_filepath = os.path.join(output_dir, "time_frequency_comparison.png")
fig.savefig(output_filepath)  # 使用 figure 对象保存

plt.show()