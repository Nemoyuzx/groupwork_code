import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import numpy.fft as npfft
from scipy.fft import fftshift
import pywt
import cv2
import sympy as sp
import scipy.signal as signal
from django.http import JsonResponse

def plot_to_base64():
    """将matplotlib图像转换为base64字符串"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    plt.close()
    return graphic

# 继续views.py的其他处理函数

def handle_laplace_transform(params):
    """处理拉普拉斯变换"""
    try:
        system_param = float(params.get('system_param', 2))
        input_signal = params.get('input_signal', 'step')
        signal_frequency = float(params.get('signal_frequency', 2))
        
        # 根据输入信号类型计算时域响应
        if input_signal == 'step':
            # 单位阶跃函数 u(t)
            t_vals = np.linspace(0, 5, 1000)
            x_vals = np.ones_like(t_vals)
            # 一阶系统对阶跃输入的响应: y(t) = (1/a) * (1 - e^(-at)) * u(t)
            y_vals = (1/system_param) * (1 - np.exp(-system_param * t_vals))
            signal_desc = 'u(t)'
            
        elif input_signal == 'impulse':
            # 单位冲激函数 δ(t)
            t_vals = np.linspace(0, 5, 1000)
            x_vals = np.zeros_like(t_vals)
            x_vals[0] = 10  # 在t=0处显示冲激
            # 一阶系统对冲激输入的响应: y(t) = e^(-at) * u(t)
            y_vals = np.exp(-system_param * t_vals)
            signal_desc = 'δ(t)'
            
        elif input_signal == 'ramp':
            # 单位斜坡函数 t*u(t)
            t_vals = np.linspace(0, 5, 1000)
            x_vals = t_vals
            # 一阶系统对斜坡输入的响应: y(t) = (1/a²) * (at - 1 + e^(-at)) * u(t)
            y_vals = (1/system_param**2) * (system_param * t_vals - 1 + np.exp(-system_param * t_vals))
            signal_desc = 't·u(t)'
            
        elif input_signal == 'exponential':
            # 指数函数 e^(-at)*u(t)
            t_vals = np.linspace(0, 5, 1000)
            x_vals = np.exp(-system_param * t_vals)
            # 对于相同指数参数，系统响应为: y(t) = t * e^(-at) * u(t)
            y_vals = t_vals * np.exp(-system_param * t_vals)
            signal_desc = f'e^(-{system_param}t)·u(t)'
            
        elif input_signal == 'sine':
            # 正弦函数 sin(ωt)*u(t)
            t_vals = np.linspace(0, 4*np.pi/signal_frequency, 1000)
            x_vals = np.sin(signal_frequency * t_vals)
            # 系统对正弦输入的稳态响应
            omega = signal_frequency
            magnitude = 1 / np.sqrt(system_param**2 + omega**2)
            phase = -np.arctan(omega / system_param)
            # 考虑暂态和稳态响应
            transient = np.exp(-system_param * t_vals)
            steady_state = magnitude * np.sin(omega * t_vals + phase)
            y_vals = steady_state + transient * (x_vals[0] - steady_state[0])
            signal_desc = f'sin({signal_frequency}t)·u(t)'
            
        elif input_signal == 'cosine':
            # 余弦函数 cos(ωt)*u(t)
            t_vals = np.linspace(0, 4*np.pi/signal_frequency, 1000)
            x_vals = np.cos(signal_frequency * t_vals)
            # 系统对余弦输入的稳态响应
            omega = signal_frequency
            magnitude = 1 / np.sqrt(system_param**2 + omega**2)
            phase = -np.arctan(omega / system_param)
            # 考虑暂态和稳态响应
            transient = np.exp(-system_param * t_vals)
            steady_state = magnitude * np.cos(omega * t_vals + phase)
            y_vals = steady_state + transient * (x_vals[0] - steady_state[0])
            signal_desc = f'cos({signal_frequency}t)·u(t)'
            
        else:
            # 默认为阶跃函数
            t_vals = np.linspace(0, 5, 1000)
            x_vals = np.ones_like(t_vals)
            y_vals = (1/system_param) * (1 - np.exp(-system_param * t_vals))
            signal_desc = 'u(t)'
        
        # 绘制图形
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(t_vals, x_vals, 'b-', linewidth=2, label=f'Input: {signal_desc}')
        plt.title('Input Signal')
        plt.xlabel('Time t')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(t_vals, y_vals, 'r-', linewidth=2, label='Output Signal y(t)')
        plt.title('System Response')
        plt.xlabel('Time t')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        image_data = plot_to_base64()
        
        return JsonResponse({
            'success': True,
            'image': image_data,
            'info': f'Input: {signal_desc}, System parameter: {system_param}'
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'计算错误: {str(e)}',
            'success': False
        }, status=500)

def handle_wavelet_transform(params):
    """处理小波变换"""
    noise_level = params.get('noise_level', 0.5)
    threshold = params.get('threshold', 0.3)
    
    # 生成信号
    t = np.linspace(0, 1, 1000)
    signal_clean = np.sin(2 * np.pi * 5 * t)
    signal_noisy = signal_clean + np.random.normal(0, noise_level, 1000)
    
    # 小波去噪
    coeffs = pywt.wavedec(signal_noisy, 'db1', level=5)
    coeffs_thresh = coeffs[:]
    coeffs_thresh[1:] = [pywt.threshold(i, threshold, mode='soft') for i in coeffs[1:]]
    denoised_signal = pywt.waverec(coeffs_thresh, 'db1')
    
    # 绘图
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, signal_noisy)
    plt.title('Original Noisy Signal')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(t[:len(denoised_signal)], denoised_signal)
    plt.title('Denoised Signal')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(t, signal_clean)
    plt.title('Original Clean Signal')
    plt.grid(True)
    
    plt.tight_layout()
    image_data = plot_to_base64()
    
    return JsonResponse({
        'success': True,
        'image': image_data,
        'info': f'Noise level: {noise_level}, Threshold: {threshold}'
    })

def handle_hough_transform(params):
    """处理霍夫变换 - 支持多种几何形状检测"""
    shape_type = params.get('shape_type', 'circles')
    
    if shape_type == 'circles':
        return _detect_circles(params)
    elif shape_type == 'lines':
        return _detect_lines(params)
    elif shape_type == 'rectangles':
        return _detect_rectangles(params)
    else:
        return JsonResponse({
            'error': 'Unknown shape type',
            'success': False
        }, status=400)

def _detect_circles(params):
    """检测圆形"""
    num_circles = int(params.get('num_circles', 3))
    
    # 创建图像
    image = np.zeros((300, 400), dtype=np.uint8)
    
    # 根据参数绘制圆
    centers = [(100, 150), (250, 100), (280, 220)]
    radii = [40, 30, 50]
    
    for i in range(min(num_circles, len(centers))):
        cv2.circle(image, centers[i], radii[i], 255, 2)
    
    # 霍夫圆变换
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                              param1=100, param2=15, minRadius=25, maxRadius=60)
    
    # 创建结果图像
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    detected_count = 0
    if circles is not None:
        circles = np.round(circles).astype(np.uint16)
        detected_count = len(circles[0])
        for circle in circles[0]:
            cv2.circle(output_image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.circle(output_image, (circle[0], circle[1]), 2, (0, 0, 255), 3)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image with Circles")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Circles using Hough Transform")
    plt.axis('off')
    
    plt.tight_layout()
    image_data = plot_to_base64()
    
    return JsonResponse({
        'success': True,
        'image': image_data,
        'info': f'Circles drawn: {num_circles}, Detected: {detected_count}'
    })

def _detect_lines(params):
    """检测直线"""
    num_lines = int(params.get('num_lines', 3))
    
    # 创建图像
    image = np.zeros((300, 400), dtype=np.uint8)
    
    # 绘制直线
    lines_to_draw = [
        [(50, 50), (350, 100)],   # 斜线1
        [(80, 250), (320, 200)],  # 斜线2
        [(150, 50), (150, 250)],  # 垂直线
        [(50, 150), (350, 150)]   # 水平线
    ]
    
    for i in range(min(num_lines, len(lines_to_draw))):
        cv2.line(image, lines_to_draw[i][0], lines_to_draw[i][1], 255, 3)
    
    # 霍夫直线变换
    lines = cv2.HoughLines(image, 1, np.pi/180, threshold=100)
    
    # 创建结果图像
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    detected_count = 0
    if lines is not None:
        detected_count = len(lines)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image with Lines")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Lines using Hough Transform")
    plt.axis('off')
    
    plt.tight_layout()
    image_data = plot_to_base64()
    
    return JsonResponse({
        'success': True,
        'image': image_data,
        'info': f'Lines drawn: {num_lines}, Detected: {detected_count}'
    })

def _detect_rectangles(params):
    """检测矩形（通过轮廓检测）"""
    num_rectangles = int(params.get('num_rectangles', 2))
    
    # 创建图像
    image = np.zeros((300, 400), dtype=np.uint8)
    
    # 绘制矩形
    rectangles_to_draw = [
        [(50, 50), (150, 120)],   # 矩形1
        [(200, 100), (320, 180)], # 矩形2
        [(80, 200), (180, 270)]   # 矩形3
    ]
    
    for i in range(min(num_rectangles, len(rectangles_to_draw))):
        cv2.rectangle(image, rectangles_to_draw[i][0], rectangles_to_draw[i][1], 255, 2)
    
    # 查找轮廓来检测矩形
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建结果图像
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    detected_count = 0
    for contour in contours:
        # 使用轮廓近似来检测矩形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果近似轮廓有4个顶点，认为是矩形
        if len(approx) == 4:
            detected_count += 1
            cv2.drawContours(output_image, [approx], -1, (0, 255, 0), 2)
            # 标记矩形的顶点
            for point in approx:
                cv2.circle(output_image, tuple(point[0]), 5, (0, 0, 255), -1)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image with Rectangles")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Rectangles using Contour Detection")
    plt.axis('off')
    
    plt.tight_layout()
    image_data = plot_to_base64()
    
    return JsonResponse({
        'success': True,
        'image': image_data,
        'info': f'Rectangles drawn: {num_rectangles}, Detected: {detected_count}'
    })

def handle_z_transform(params):
    """处理Z变换"""
    a = params.get('a', 0.5)
    n_points = params.get('n_points', 10)
    
    # 定义离散信号
    n = np.arange(0, n_points)
    x_n = a**n
    
    # Z变换的零极点
    zeros = np.array([0])
    poles = np.array([a])
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 原始信号
    plt.subplot(1, 2, 1)
    plt.stem(n, x_n, basefmt=" ")
    plt.title(f'Discrete Signal x[n] = ({a})^n u[n]')
    plt.xlabel('n')
    plt.ylabel('x[n]')
    plt.grid(True)
    
    # 零极点图
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    from matplotlib.patches import Circle
    unit_circle = Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(unit_circle)
    
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', s=100, 
               facecolors='none', edgecolors='blue', label='Zeros')
    plt.scatter(np.real(poles), np.imag(poles), marker='x', s=100, 
               color='red', label='Poles')
    
    plt.title(f'Pole-Zero Plot of X(z) = z / (z - {a})')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    
    plt.tight_layout()
    image_data = plot_to_base64()
    
    return JsonResponse({
        'success': True,
        'image': image_data,
        'info': f'Parameter a: {a}, Points: {n_points}'
    })

def dfrft(x, alpha):
    """计算离散分数傅里叶变换"""
    N = len(x)
    if N == 0:
        return np.array([])
    if alpha == 0:
        return x.copy()
    if alpha == 1:
        return fftshift(npfft.fft(fftshift(x))) / np.sqrt(N)
    if alpha == 2:
        return x[::-1]

    a = alpha * np.pi / 2
    if np.sin(a) == 0:
        if (alpha / 2) % 2 == 0:
            return x.copy()
        else:
            return x[::-1]

    n = np.arange(N)
    chirp = np.exp(-1j * np.pi * np.tan(a / 2) * (n - N // 2)**2 / N)
    x_chirped = x * chirp
    
    kernel = np.exp(1j * np.pi / np.sin(a) * (n - N // 2)**2 / N)
    X_fft1 = npfft.fft(fftshift(x_chirped))
    X_fft2 = npfft.fft(fftshift(kernel))
    X_conv = X_fft1 * X_fft2
    X_intermediate = npfft.ifft(X_conv)
    
    result = np.sqrt(N) * np.exp(-1j * (np.pi / 2 * (1 - alpha) - a / 2)) / \
             np.sqrt(2 * np.pi * N * abs(np.sin(a))) * chirp * fftshift(X_intermediate)
    
    return result

def handle_fractional_fourier_transform(params):
    """处理分数傅里叶变换"""
    alpha = params.get('alpha', 0.5)
    f0 = params.get('f0', 2)
    f1 = params.get('f1', 20)
    
    # 生成线性调频信号
    N = 1000
    t = np.linspace(0, 1, N, endpoint=False)
    k = (f1 - f0)
    x = np.sin(2 * np.pi * (f0 * t + k * t**2 / 2))
    
    # 计算分数傅里叶变换
    X_frac = dfrft(x, alpha)
    u = np.arange(N)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t, x)
    plt.title('Original Chirp Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(u - N // 2, np.abs(fftshift(X_frac)))
    plt.title(f'Magnitude of DFrFT (α={alpha})')
    plt.xlabel('Fractional Domain (u)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim([-N//4, N//4])
    
    plt.tight_layout()
    image_data = plot_to_base64()
    
    return JsonResponse({
        'success': True,
        'image': image_data,
        'info': f'Alpha: {alpha}, Frequency range: {f0}-{f1} Hz'
    })
