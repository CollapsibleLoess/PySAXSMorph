import numpy as np
import matplotlib.pyplot as plt

# 随机生成多个波矢量和相位
from g_export.export2d import export_2d_images

num_waves = 10
k_values = np.random.randn(num_waves, 2) * 1
phi_values = np.random.rand(num_waves) * 2 * np.pi

# 定义网格
x = np.linspace(0, 10, 1000)
y = np.linspace(0, 10, 1000)
X, Y = np.meshgrid(x, y)

#绘制每个单个波
for i in range(num_waves):
    Z_single = np.cos(k_values[i, 0] * X + k_values[i, 1] * Y + phi_values[i])
    #调用 export_2d_images 函数，传入 X, Y, D 矩阵
    export_2d_images(X, Y, Z_single)


# 计算随机场
Z = np.zeros_like(X)
for i in range(num_waves):
    Z += np.cos(k_values[i, 0] * X + k_values[i, 1] * Y + phi_values[i])


# 调用 export_2d_images 函数，传入 X, Y, D 矩阵
export_2d_images(X, Y, Z)
# 定义阈值 alpha
alpha = 0
# 进行裁剪
Z_clipped = Z > alpha
export_2d_images(X, Y, Z_clipped)
