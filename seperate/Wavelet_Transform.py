import pywt
import matplotlib.pyplot as plt
import numpy as np
import os  # 导入 os 模块

# 原始信号
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.5, 1000)

# --- 第一次小波变换去噪 ---
coeffs = pywt.wavedec(signal, 'db1', level=5)
threshold = 0.3
coeffs_thresh = coeffs[:] # 创建副本以保留原始系数
coeffs_thresh[1:] = [pywt.threshold(i, threshold, mode='soft') for i in coeffs[1:]]
denoised_signal = pywt.waverec(coeffs_thresh, 'db1')

# --- 第二次小波变换去噪 (对第一次去噪结果进行) ---
coeffs_2nd = pywt.wavedec(denoised_signal, 'db1', level=5)
# 可以选择使用相同的或不同的阈值，这里使用相同的
coeffs_2nd_thresh = coeffs_2nd[:]
coeffs_2nd_thresh[1:] = [pywt.threshold(i, threshold, mode='soft') for i in coeffs_2nd[1:]]
denoised_signal_2nd = pywt.waverec(coeffs_2nd_thresh, 'db1')


# --- 准备输出目录 ---
# 获取当前脚本的文件名（不含扩展名）
script_filename = os.path.splitext(os.path.basename(__file__))[0]
# 定义输出目录名
output_dir = f"{script_filename} - output"
# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)


# --- 绘制并保存包含第二次去噪结果的对比图 ---
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, signal)
plt.title('Original Noisy Signal')

plt.subplot(3, 1, 2)
plt.plot(t[:len(denoised_signal)], denoised_signal)
plt.title('Denoised Signal (1st Pass)')

plt.subplot(3, 1, 3)
# 确保 denoised_signal_2nd 长度与 t 匹配
plt.plot(t[:len(denoised_signal_2nd)], denoised_signal_2nd)
plt.title('Denoised Signal (2nd Pass)')

plt.tight_layout()
# 定义第二次去噪的输出文件名
output_filename_2nd = os.path.join(output_dir, "denoising_comparison_2nd_pass.png")
# 保存图像
plt.savefig(output_filename_2nd)
print(f"第二次去噪对比图已保存到: {output_filename_2nd}")

# 显示所有图形 (如果之前注释了 plt.show())
plt.show()