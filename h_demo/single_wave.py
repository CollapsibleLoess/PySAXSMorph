import numpy as np
import matplotlib.pyplot as plt

# 定义波矢量和相位
k = np.array([2, 10])  # 波矢量
phi = 1  # 相位

# 定义网格
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)

# 计算波形
Z = np.cos(k[0] * X + k[1] * Y + phi)

# 绘制波形
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(label='Wave Amplitude')
plt.title('Single Wave in 2D')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
