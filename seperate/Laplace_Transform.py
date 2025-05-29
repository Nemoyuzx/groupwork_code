import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 定义符号和输入信号 ---
t, s = sp.symbols('t s')
# 使用单位阶跃函数作为输入信号 x(t) = u(t)
x_t = sp.Heaviside(t)
print(f"输入信号 x(t): {x_t}")

# 计算输入信号的拉普拉斯变换 X(s)
X_s = sp.laplace_transform(x_t, t, s, noconds=True) # noconds=True 简化输出
print(f"输入信号的拉普拉斯变换 X(s): {X_s}")

# --- 2. 定义线性系统 ---
# 考虑一个简单的一阶系统，例如 RC 低通滤波器或阻尼系统
# 其微分方程为: dy/dt + a*y(t) = x(t)
# 假设 a = 2，则传递函数 H(s) = 1 / (s + a) = 1 / (s + 2)
a = 2
H_s = 1 / (s + a)
print(f"系统传递函数 H(s): {H_s}")

# --- 3. 计算系统响应 ---
# 输出信号的拉普拉斯变换 Y(s) = H(s) * X(s)
Y_s = H_s * X_s
print(f"输出信号的拉普拉斯变换 Y(s): {Y_s}")

# 计算输出信号的时域表示 y(t) (Y(s) 的拉普拉斯反变换)
y_t = sp.inverse_laplace_transform(Y_s, s, t)
print(f"输出信号 y(t): {y_t}")

# 讨论：
# 拉普拉斯变换将时域的微分方程转换为了频域（复频域 s）的代数方程。
# 这使得求解系统响应变得更加容易：
# 时域卷积运算 y(t) = h(t) * x(t) 变成了频域乘积运算 Y(s) = H(s)X(s)。
# 通过计算 Y(s) 并进行拉普拉斯反变换，我们可以得到系统在给定输入下的时域响应 y(t)。
# 对于这个例子，输入是单位阶跃信号，系统是一阶低通系统。
# 输出 y(t) = (1/2 - exp(-2*t)/2)*Heaviside(t) 显示了系统对阶跃输入的响应：
# 初始值为 0，然后逐渐指数上升，最终稳定在 1/a = 1/2。这符合一阶系统的特性。

# --- 4. 绘图 ---
# 将符号表达式转换为数值函数
x_t_func = sp.lambdify(t, x_t, 'numpy')
y_t_func = sp.lambdify(t, y_t, 'numpy')

# 生成时间点
t_vals = np.linspace(0, 5, 500) # 从 t=0 开始，因为使用了 Heaviside 函数

# 计算对应的 x(t) 和 y(t) 值
x_vals = x_t_func(t_vals)
y_vals = y_t_func(t_vals)

# 创建图像
plt.figure(figsize=(10, 6))
plt.plot(t_vals, x_vals, label='Input Signal $x(t) = u(t)$')
plt.plot(t_vals, y_vals, label=f'Output Signal $y(t) = {sp.latex(y_t)}$') # 使用 LaTeX 格式化标签

# 添加图例、标题和标签
plt.title('System Response to Unit Step Input')
plt.xlabel('Time $t$')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.ylim(-0.1, 1.2) # 调整 Y 轴范围以获得更好的视觉效果

# --- 5. 保存图像 ---
# 创建输出目录
output_dir = "Laplace_Transform - output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存图像
output_filename = os.path.join(output_dir, "system_response.png")
plt.savefig(output_filename)
print(f"图像已保存到: {output_filename}")

# 显示图像
plt.show()