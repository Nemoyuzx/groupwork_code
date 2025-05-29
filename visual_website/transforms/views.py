from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import numpy.fft as npfft
import pywt
import cv2
import sympy as sp
import scipy.signal as signal
import os
from django.conf import settings

def index(request):
    """主页视图"""
    return render(request, 'transforms/index.html')

def fourier_transform(request):
    """傅里叶变换页面"""
    return render(request, 'transforms/fourier.html')

def laplace_transform(request):
    """拉普拉斯变换页面"""
    return render(request, 'transforms/laplace.html')

def wavelet_transform(request):
    """小波变换页面"""
    return render(request, 'transforms/wavelet.html')

def hough_transform(request):
    """霍夫变换页面"""
    return render(request, 'transforms/hough.html')

def z_transform(request):
    """Z变换页面"""
    return render(request, 'transforms/z_transform.html')

def fractional_fourier_transform(request):
    """分数傅里叶变换页面"""
    return render(request, 'transforms/fractional_fourier.html')

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

@csrf_exempt
def api_transform(request):
    """API端点处理变换计算"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        transform_type = data.get('transform_type')
        params = data
        
        if transform_type == 'fourier':
            return handle_fourier_transform(params)
        elif transform_type == 'laplace':
            return handle_laplace_transform(params)
        elif transform_type == 'wavelet':
            return handle_wavelet_transform(params)
        elif transform_type == 'hough':
            return handle_hough_transform(params)
        elif transform_type == 'z_transform':
            return handle_z_transform(params)
        elif transform_type == 'fractional_fourier':
            return handle_fractional_fourier_transform(params)
        else:
            return JsonResponse({'error': 'Unknown transform type'}, status=400)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def handle_fourier_transform(params):
    """处理傅里叶变换"""
    try:
        # 参数验证和默认值
        frequency = float(params.get('frequency', 5))
        sample_rate = int(params.get('sample_rate', 1000))
        
        # 参数范围验证
        if frequency <= 0 or frequency > 500:
            return JsonResponse({
                'error': '频率必须在0-500Hz之间',
                'success': False
            }, status=400)
            
        if sample_rate < 100 or sample_rate > 10000:
            return JsonResponse({
                'error': '采样率必须在100-10000之间',
                'success': False
            }, status=400)
    
        # 生成信号
        t = np.linspace(0, 1, sample_rate)
        x = np.sin(2 * np.pi * frequency * t)
        
        # 计算傅里叶变换
        X_f = np.fft.fft(x)
        frequencies = np.fft.fftfreq(len(t), t[1] - t[0])
        X_f_abs = np.abs(X_f) / len(t)
        
        # 只取正频率部分
        pos_mask = frequencies >= 0
        frequencies_pos = frequencies[pos_mask]
        X_f_abs_pos = X_f_abs[pos_mask]
        
        # 绘制时域信号
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(t, x)
        plt.title('Time Domain Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # 绘制频域信号
        plt.subplot(1, 2, 2)
        plt.stem(frequencies_pos, X_f_abs_pos)
        plt.title('Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Normalized Amplitude')
        plt.xlim(0, 20)
        plt.grid(True)
        
        plt.tight_layout()
        image_data = plot_to_base64()
        
        return JsonResponse({
            'success': True,
            'image': image_data,
            'info': f'Signal frequency: {frequency} Hz, Sample rate: {sample_rate}'
        })
        
    except ValueError as e:
        return JsonResponse({
            'error': f'参数错误: {str(e)}',
            'success': False
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': f'计算错误: {str(e)}',
            'success': False
        }, status=500)

# 导入其他处理函数
from .transform_handlers import (
    handle_laplace_transform,
    handle_wavelet_transform, 
    handle_hough_transform,
    handle_z_transform,
    handle_fractional_fourier_transform
)
