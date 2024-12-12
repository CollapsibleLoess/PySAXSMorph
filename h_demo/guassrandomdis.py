import numpy as np
import matplotlib.pyplot as plt

# 参数设置
mean = 0
std_dev = 1
size = 1000

# 生成正态分布数据
x = np.random.normal(mean, std_dev, size)

# 对 x 进行排序
x_sorted = np.sort(x)

# 计算高斯密度
gaussian_density = np.exp(-(x_sorted - mean)**2 / (2 * std_dev**2)) / (std_dev * np.sqrt(2 * np.pi))

# 绘制曲线图
plt.figure(figsize=(10, 6))
plt.plot(x_sorted, gaussian_density)
plt.title('Gaussian Density Function')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True)
plt.axvline(x=mean, color='r', linestyle='--', label='Mean')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.legend()
plt.show()
