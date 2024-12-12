import numpy as np
import matplotlib.pyplot as plt

# 随机生成多个波矢量和相位
num_waves = 10
k_values = np.random.randn(num_waves, 2)
phi_values = np.random.rand(num_waves) * 2 * np.pi

# 定义网格
x = np.linspace(0, 10, 50)
y = np.linspace(0, 10, 50)
X, Y = np.meshgrid(x, y)

# 计算随机场
Z = np.zeros_like(X)
for i in range(num_waves):
    Z += np.cos(k_values[i, 0] * X + k_values[i, 1] * Y + phi_values[i])

# 定义阈值 alpha
alpha = 0

# 进行裁剪
Z_clipped = Z > alpha

# 绘制裁剪后的结果
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z_clipped, cmap='viridis')
plt.colorbar(label='Phase')
plt.title('Clipped Random Field')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
