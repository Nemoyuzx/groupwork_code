# 数学变换可视化工具

一个基于Django的交互式数学变换可视化Web应用程序，支持6种不同的数学变换的实时可视化和参数调节。

## 功能特性

### 支持的变换类型

1. **傅里叶变换 (Fourier Transform)**
   - 时域信号到频域的转换
   - 可调节信号频率和采样率
   - 实时显示时域和频域对比图

2. **拉普拉斯变换 (Laplace Transform)** 
   - 线性时不变系统分析
   - 系统响应特性可视化
   - 可调节系统参数

3. **小波变换 (Wavelet Transform)**
   - 时频分析和信号去噪
   - 支持多种小波基函数
   - 噪声水平可调节

4. **霍夫变换 (Hough Transform)**
   - 图像中圆形检测
   - 可调节检测参数
   - 支持原始图像和检测结果对比

5. **Z变换 (Z-Transform)**
   - 离散时间系统分析
   - 极点零点图显示
   - 可调节极点和零点位置

6. **分数傅里叶变换 (Fractional Fourier Transform)**
   - 时频域旋转变换
   - Chirp信号分析
   - 可调节变换阶数

### 界面特性

- **现代化UI设计**: 采用玻璃拟态设计风格，响应式布局
- **实时参数调节**: 滑块和输入框支持实时参数修改
- **交互式可视化**: 高质量matplotlib图表，自动更新
- **移动端适配**: 支持各种屏幕尺寸的设备

## 技术栈

- **后端**: Django 4.2.9
- **前端**: Bootstrap 5.1.3, jQuery
- **科学计算**: NumPy, SciPy, SymPy
- **可视化**: Matplotlib
- **图像处理**: OpenCV, scikit-image
- **小波分析**: PyWavelets

## 安装和运行

### 环境要求

- Python 3.8+
- pip包管理器

### 安装步骤

1. **克隆或下载项目**
   ```bash
   cd /path/to/project
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行Django开发服务器**
   ```bash
   cd visual_website
   python manage.py runserver
   ```

4. **访问应用**
   在浏览器中打开: `http://127.0.0.1:8000`

## 项目结构

```
概率论/groupwork/groupwork_code/
├── visual_website/                 # Django项目根目录
│   ├── manage.py                  # Django管理脚本
│   ├── transform_visualizer/      # 主项目配置
│   │   ├── settings.py           # Django设置
│   │   ├── urls.py               # 主URL配置
│   │   └── wsgi.py               # WSGI配置
│   └── transforms/                # 变换应用
│       ├── views.py              # 视图函数
│       ├── transform_handlers.py  # 变换处理逻辑
│       ├── urls.py               # 应用URL配置
│       └── templates/            # HTML模板
├── requirements.txt               # Python依赖包
├── Fourier_Transform.py          # 独立的傅里叶变换脚本
├── Laplace_Transform.py          # 独立的拉普拉斯变换脚本
├── Wavelet_Transform.py          # 独立的小波变换脚本
├── Hough_Transform.py            # 独立的霍夫变换脚本
├── Z_transform.py                # 独立的Z变换脚本
├── Fractional_Fourier_Transform.py # 独立的分数傅里叶变换脚本
└── */output/                     # 各变换的输出图像目录
```

## API接口

### 变换计算API

**端点**: `POST /api/transform/`

**请求格式**:
```json
{
  "transform_type": "fourier|laplace|wavelet|hough|z_transform|fractional_fourier",
  "参数名": "参数值",
  ...
}
```

**响应格式**:
```json
{
  "success": true,
  "image": "base64编码的图像数据",
  "info": "变换信息描述"
}
```

### 参数说明

#### 傅里叶变换
- `frequency`: 信号频率 (Hz)
- `sample_rate`: 采样率

#### 拉普拉斯变换  
- `system_param`: 系统参数

#### 小波变换
- `noise_level`: 噪声水平
- `wavelet`: 小波基函数

#### 霍夫变换
- `min_radius`: 最小检测半径
- `max_radius`: 最大检测半径

#### Z变换
- `pole_real`: 极点实部
- `pole_imag`: 极点虚部

#### 分数傅里叶变换
- `alpha`: 变换阶数

## 使用说明

1. **选择变换类型**: 在主页点击相应的变换卡片
2. **调节参数**: 使用滑块或输入框修改参数
3. **查看结果**: 图像会自动更新显示变换结果
4. **参数实验**: 尝试不同参数组合观察效果变化

## 开发说明

### 添加新的变换

1. 在`transform_handlers.py`中添加处理函数
2. 在`views.py`的`api_transform`中添加路由
3. 创建对应的HTML模板
4. 在`urls.py`中添加URL路由

### 自定义样式

项目使用Bootstrap 5和自定义CSS，主要样式定义在`base.html`中。

### 性能优化

- 使用matplotlib的`Agg`后端避免GUI依赖
- 图像压缩为PNG格式减少传输大小  
- 使用base64编码实现无文件传输

## 许可证

本项目仅用于教育和学习目的。

## 贡献

欢迎提交问题和改进建议！

## 更新日志

### v1.0.0 (2025-05-29)
- 初始版本发布
- 支持6种数学变换
- 现代化Web界面
- 实时参数调节功能
