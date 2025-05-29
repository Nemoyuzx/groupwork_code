import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# --- 1. 创建具有径向对称性的信号（三个圆） ---
# 创建一个简单的黑色图像，尺寸稍大以容纳更多圆
image = np.zeros((300, 400), dtype=np.uint8)
# 在图像中绘制三个白色圆
center1 = (100, 150)
radius1 = 40
center2 = (250, 100)
radius2 = 30
center3 = (280, 220)
radius3 = 50
color = 255
thickness = 2
cv2.circle(image, center1, radius1, color, thickness)
cv2.circle(image, center2, radius2, color, thickness)
cv2.circle(image, center3, radius3, color, thickness)

# --- 准备输出目录 ---
output_dir = "Hough_Transform - output"
# 创建输出目录（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 保存并显示原始图像 ---
original_filename = os.path.join(output_dir, "original_circles.png")
cv2.imwrite(original_filename, image)
print(f"Original image saved to: {original_filename}")

plt.figure(figsize=(6, 4.5)) # 调整尺寸以匹配新图像比例
plt.imshow(image, cmap='gray')
plt.title("Original Image with Circles")
plt.axis('off')
plt.show()

# --- 2. 计算其霍夫变换 (Hough Circle Transform) ---
# 应用霍夫圆变换
# 调整 minDist 以允许检测更近的圆
circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=100, param2=15, minRadius=25, maxRadius=60) # 调整半径范围

# --- 3 & 4. 绘制检测到的圆并准备保存 ---
# 创建一个彩色图像副本用于绘制检测结果
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# 如果检测到圆，绘制它们
if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f"Detected {len(circles[0, :])} circles.") # 打印检测到的圆的数量
    for i in circles[0, :]:
        # 绘制外圆
        cv2.circle(output_image, (i[0], i[1]), i[2], (0, 255, 0), 2) # 绿色圆圈
        # 绘制圆心
        cv2.circle(output_image, (i[0], i[1]), 2, (0, 0, 255), 3) # 红色圆心
else:
    print("No circles were detected.")

# --- 5. 保存/显示结果图像 ---
# 定义结果图像文件名
output_filename = os.path.join(output_dir, "detected_circles.png")

# 保存结果图像
cv2.imwrite(output_filename, output_image)
print(f"Result image saved to: {output_filename}")

# 显示带有检测结果的图像
plt.figure(figsize=(6, 4.5)) # 调整尺寸以匹配新图像比例
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)) # Matplotlib 需要 RGB 格式
plt.title("Detected Circles using Hough Transform")
plt.axis('off')
plt.show()