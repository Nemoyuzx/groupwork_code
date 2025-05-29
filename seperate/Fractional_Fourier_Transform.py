import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
import os

# --- DFrFT 实现 (基于 FFT 的 chirp 卷积方法) ---
def dfrft(x, alpha):
    """
    计算离散分数傅里叶变换 (DFrFT)。
    参数:
        x (numpy.ndarray): 输入信号 (1D).
        alpha (float): 变换的阶数.
    返回:
        numpy.ndarray: 信号 x 的 DFrFT.
    """
    N = len(x)
    if N == 0:
        return np.array([])
    if alpha == 0:
        return x.copy()
    if alpha == 1:
        return fftshift(fft(fftshift(x))) / np.sqrt(N)
    if alpha == 2:
        return x[::-1] # 时域反转

    # 避免 alpha 是 2 的整数倍时 sin(a) 为 0
    a = alpha * np.pi / 2
    if np.sin(a) == 0:
         # 对于 alpha = 2k, 结果是 x 或 x[::-1]
        if (alpha / 2) % 2 == 0: # alpha = 4k
            return x.copy()
        else: # alpha = 2(2k+1)
            return x[::-1]

    # 使用基于 FFT 的 chirp 卷积算法
    n = np.arange(N)
    k = n.copy()

    # Chirp 信号
    chirp = np.exp(-1j * np.pi * np.tan(a / 2) * (n - N // 2)**2 / N)
    x_chirped = x * chirp

    # 使用 FFT 进行卷积
    kernel = np.exp(1j * np.pi / np.sin(a) * (n - N // 2)**2 / N)
    X_conv = fft(fftshift(x_chirped)) * fft(fftshift(kernel))
    X_intermediate = ifft(X_conv)

    # 最终的 chirp 乘积和归一化因子
    result = np.sqrt(N) * np.exp(-1j * (np.pi / 2 * (1 - alpha) - a / 2)) / np.sqrt(2 * np.pi * N * abs(np.sin(a))) \
             * chirp * fftshift(X_intermediate)

    return result

# --- 主程序 ---
# 创建输出目录
output_dir = "Fractional_Fourier_Transform - output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义信号 (线性调频信号)
N = 1000  # 采样点数
t = np.linspace(0, 1, N, endpoint=False)
f0 = 2      # 起始频率
f1 = 20     # 终止频率
k = (f1 - f0) # 调频斜率
x = np.sin(2 * np.pi * (f0 * t + k * t**2 / 2)) # Chirp 信号

# 1. 绘制并保存原始信号
plt.figure(figsize=(10, 4))
plt.plot(t, x)
plt.title('Original Chirp Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
original_signal_path = os.path.join(output_dir, 'original_signal.png')
plt.savefig(original_signal_path)
print(f"原始信号图像已保存到: {original_signal_path}")
# plt.show() # 暂时不显示，最后一起显示

# 2. 计算分数傅里叶变换
alpha = 0.5  # 分数傅里叶变换的阶数
X_frac = dfrft(x, alpha)

# 3. 绘制并保存 DFrFT 结果
# 分数域的 "频率" 轴 （注意：这不是严格的物理频率）
u = np.arange(N) # 或者使用 fftshift 后的频率轴

plt.figure(figsize=(10, 4))
# 绘制幅度谱，使用 fftshift 将零频移到中心
plt.plot(u - N // 2, np.abs(fftshift(X_frac)))
plt.title(f'Magnitude of DFrFT (α={alpha})')
plt.xlabel('Fractional Domain (u)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.xlim([-N//4, N//4]) # 缩放横轴以突出主要成分
frft_signal_path = os.path.join(output_dir, f'frft_alpha_{alpha}.png')
plt.savefig(frft_signal_path)
print(f"分数傅里叶变换结果图像已保存到: {frft_signal_path}")

# 4. 显示所有图像
plt.show()

# 讨论:
# 原始信号是一个线性调频信号，其频率随时间线性增加。
# 标准傅里叶变换 (alpha=1) 会将这个信号的能量分散在整个频带上，因为它在整个持续时间内频率都在变化。
# 分数傅里叶变换 (例如 alpha=0.5) 可以看作是在时频平面上的旋转。对于 chirp 信号，存在一个特定的旋转角度 (alpha 值)，
# 可以使得 chirp 信号在对应的分数域中变得更加集中，类似于一个脉冲。
# 图中 DFrFT 的幅度谱显示能量在分数域的某个区域相对集中，这与标准 FFT 的结果不同。
# 通过改变 alpha 值，我们可以观察到能量在不同分数域中的分布变化，这体现了 FrFT 在时频分析中的能力。
# 对于这个特定的 chirp 信号和 alpha=0.5，能量在分数域 u 的某个范围内达到峰值。