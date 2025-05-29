import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os

# --- 1. 定义离散信号 ---
# 选择信号 x[n] = a^n * u[n]
a = 0.5
n = np.arange(0, 10)  # 选择绘制的点数
x_n = a**n

# --- Z变换 ---
# X(z) = z / (z - a)
# 分子系数 (对应 z)
num = [1, 0]
# 分母系数 (对应 z - a)
den = [1, -a]

# --- 创建输出目录 ---
output_dir = "Z_transform - output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. 绘制变换前（原始信号）的效果图 ---
plt.figure(figsize=(8, 5)) # 合理缩放图像
plt.stem(n, x_n, basefmt=" ")
plt.title(f'Discrete Signal $x[n] = ({a})^n u[n]$')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid(True)
# 保存图像
signal_plot_path = os.path.join(output_dir, 'original_signal.png')
plt.savefig(signal_plot_path)
print(f"原始信号图像已保存至: {signal_plot_path}")
# 显示图像
plt.show()


# --- 3. 计算并绘制变换后（零极点图）的效果图 ---
# 计算极点零点
zeros, poles, _ = signal.tf2zpk(num, den)

# 绘制极点零点图
plt.figure(figsize=(6, 6)) # 合理缩放图像，保持圆形
ax = plt.gca()

# 绘制单位圆
unit_circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
ax.add_patch(unit_circle)

# 绘制零点和极点
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', s=100, facecolors='none', edgecolors='blue', label='Zeros')
plt.scatter(np.real(poles), np.imag(poles), marker='x', s=100, color='red', label='Poles')

plt.title('Pole-Zero Plot of $X(z) = z / (z - 0.5)$')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.legend()
plt.grid(True)
plt.axis('equal') # 保持横纵轴比例一致
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])

# 保存图像
pz_plot_path = os.path.join(output_dir, 'pole_zero_plot.png')
plt.savefig(pz_plot_path)
print(f"零极点图已保存至: {pz_plot_path}")
# 显示图像
plt.show()