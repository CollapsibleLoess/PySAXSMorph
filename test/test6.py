import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 定义目标分布 f(y)
def f(y):
    return np.exp(-0.5 * (y - 3) ** 2)

# 计算归一化常数 D
Z, _ = quad(f, -np.inf, np.inf)

# 归一化目标分布 f(y)
def f_norm(y):
    return f(y) / Z

# 定义包络分布 g(y)
def g(y):
    return np.ones_like(y) / 2.0  # 均匀分布在 [0, 6] 上的常数值

# 定义包络常数 c
y_test = np.linspace(0, 6, 1000)
c = np.max(f_norm(y_test) / g(y_test))

# 从 g(y) 中采集y值的函数
def g_sampler():
    return np.random.uniform(0, 6)

# 接受-拒绝采样
def rejection_sampling(f, g, c, g_sampler, iterations):
    samples = []
    for _ in range(iterations):
        while True:
            y = g_sampler()
            u = np.random.uniform(0, 1)
            if u <= f(y) / (c * g(y)):
                samples.append(y)
                break
    return samples

# 生成样本
iterations = 100000  # 增加采样次数
samples = rejection_sampling(f_norm, g, c, g_sampler, iterations)

# 绘制结果
y_values = np.linspace(0, 6, 1000)
f_values = f_norm(y_values)
g_values = g(y_values)

plt.plot(y_values, f_values, label='Normalized Target Distribution f(y)')
plt.plot(y_values, g_values, label='Envelope Distribution g(y)', linestyle='-.')
plt.plot(y_values, g_values * c, label='Scaled Envelope Distribution c * g(y)', linestyle='--')
plt.axhline(y=c, color='r', linestyle=':', label='Envelope Constant c')
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g', label='Generated Samples')
plt.legend()
plt.xlabel('y')
plt.ylabel('Density')
plt.title('Rejection Sampling with Normalized f(y)')
plt.show()
