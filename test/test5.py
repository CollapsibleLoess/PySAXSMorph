import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 假设你的离散点如下
k_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
f_values = np.array([0.5, 0.6, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35])

# 使用插值来估计任意点的概率密度
target_pdf = interp1d(k_values, f_values, kind='linear', fill_value="extrapolate")

# 提议分布的采样器，这里使用正态分布
def proposal_sampler(x, sigma=0.1):
    return np.random.normal(x, sigma)

# Metropolis-Hastings算法
def metropolis_hastings(target_pdf, proposal_sampler, initial_sample, iterations):
    samples = []
    current_sample = initial_sample

    for _ in range(iterations):
        proposed_sample = proposal_sampler(current_sample)
        if proposed_sample < min(k_values) or proposed_sample > max(k_values):
            continue  # 如果候选样本超出范围则拒绝
        acceptance_ratio = target_pdf(proposed_sample) / target_pdf(current_sample)

        if np.random.rand() < acceptance_ratio:
            current_sample = proposed_sample

        samples.append(current_sample)

    return samples

# 参数设置
initial_sample = 0.5
iterations = 10000

# 生成样本
samples = metropolis_hastings(target_pdf, proposal_sampler, initial_sample, iterations)

# 绘制生成的样本直方图
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Generated Samples')

# 绘制目标分布的PDF
x = np.linspace(0.1, 1.0, 1000)
pdf = target_pdf(x)
plt.plot(x, pdf, 'r-', lw=2, label='Original PDF')

# 绘制原始的离散点
plt.scatter(k_values, f_values, color='b', zorder=5, label='Original Discrete Points')

plt.title("Metropolis-Hastings Sampling with Discrete Target PDF")
plt.xlabel('k')
plt.ylabel('Density')
plt.legend()
plt.show()
