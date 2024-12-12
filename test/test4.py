import numpy as np
import matplotlib.pyplot as plt

# 随机生成多个波矢量和相位
num_waves = 10
k_values = np.random.randn(num_waves, 2)
phi_values = np.random.rand(num_waves) * 2 * np.pi

# 定义网格
x = np.linspace(0, 10, 500)
y = np.linspace(0, 10, 500)
X, Y = np.meshgrid(x, y)

# 计算随机场
Z = np.zeros_like(X)
for i in range(num_waves):
    Z += np.cos(k_values[i, 0] * X + k_values[i, 1] * Y + phi_values[i])

# 绘制随机场
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(label='Random Field Amplitude')
plt.title('Random Field from Multiple Waves')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
