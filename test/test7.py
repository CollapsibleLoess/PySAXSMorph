import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom

# 定义参数
n = 100  # 试验次数
theta_true = 0.7  # 真正的成功概率
alpha_prior = 2  # Beta分布先验的alpha参数
beta_prior = 2  # Beta分布先验的beta参数

# 模拟伯努利试验
# np.random.seed(100)  # 固定随机种子以确保结果可重复
successes = np.random.binomial(1, theta_true, n)
k = np.sum(successes)  # 成功次数

# 更新后的后验参数
alpha_posterior = alpha_prior + k
beta_posterior = beta_prior + n - k

# 定义 theta 范围
theta_range = np.linspace(0, 1, 100)

# 计算各个分布的概率密度函数
prior_pdf = beta.pdf(theta_range, alpha_prior, beta_prior)
posterior_pdf = beta.pdf(theta_range, alpha_posterior, beta_posterior)
binom_pmf = binom.pmf(np.arange(n + 1), n, theta_true)

# 绘制所有曲线在一张图上
plt.figure(figsize=(10, 6))

# 先验分布
plt.plot(theta_range, prior_pdf, label=f'1. Prior Beta({alpha_prior}, {beta_prior})', linestyle='--')

# 后验分布
plt.plot(theta_range, posterior_pdf, label=f'2. Posterior Beta({alpha_posterior}, {beta_posterior})')

# 二项分布 PMF
for i in range(n + 1):
    plt.vlines(x=i/n, ymin=0, ymax=binom_pmf[i], color='green', alpha=0.6, label='3. Binomial PMF' if i == 0 else "")

# 添加真正的 theta 值
plt.axvline(theta_true, color='red', linestyle=':', label='True $\\theta$')

# 设置图形属性
plt.xlabel('$\\theta$')
plt.ylabel('Density/Probability')
plt.title('Prior, Posterior, and Binomial Distributions')
plt.legend()
plt.show()
