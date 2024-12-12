import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

def alpha_function(gamma_alpha_0):
    return np.sqrt(2) * erfinv(1 - 2*gamma_alpha_0)

# 创建 Γ^α(0) 的值范围
gamma_values = np.linspace(0, 1, 100000)

# 计算对应的 α 值
alpha_values = alpha_function(gamma_values)

# 绘制图像
# 设置全局字体大小
plt.rcParams.update({'font.size': 16})  # 你可以根据需要调整这个数值

plt.figure(figsize=(10, 6))
plt.plot(gamma_values, alpha_values)
plt.xlabel('Porosity', fontsize=20)  # 也可以在这里单独设置字体大小
plt.ylabel('α', fontsize=20)         # 同样可以在这里单独设置
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--')
plt.axvline(x=0.5, color='r', linestyle='--')
plt.show()
