import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.colors import LogNorm

# 定义试函数
def qtest_function(x):
    epsilon = 1e-10
    return np.where(np.abs(x) < 1, np.exp(-1 / (1 - x ** 2 + epsilon)), 0)

# 定义传热方程
def heat_equation(u, t, alpha, L):
    d2u = np.zeros_like(u)
    d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (L/(len(u)-1))**2
    return alpha * d2u

# 设置参数

nx = 101
nt = 1000
t_max = 0.5

L = 2.0
alpha = 0.01

x = np.linspace(-L/2, L/2, nx)
t = np.linspace(0, t_max, nt)

# 初始条件
u0 = qtest_function(x)

# 求解传热方程
solution = odeint(heat_equation, u0, t, args=(alpha, L))

# 计算能量
def calculate_energy(u):
    return np.sum(u**2) * (L / (nx - 1))

energy = np.array([calculate_energy(u) for u in solution])
energy_loss_rate = np.diff(energy) / np.diff(t)



# 创建一个大图
fig = plt.figure(figsize=(10, 8))

# 绘制初始条件（试函数）
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(x, u0)
ax1.set_title('Initial Condition (Test Function)')
ax1.set_xlabel('Position')
ax1.set_ylabel('Temperature')

# 绘制温度随时间的演化
ax2 = fig.add_subplot(3, 2, 2)
im = ax2.imshow(solution.T, aspect='auto', extent=[-L/2, L/2, t_max, 0],
                norm=LogNorm(), cmap='viridis')
fig.colorbar(im, ax=ax2, label='Temperature (log scale)')
ax2.set_title('Heat Equation Solution')
ax2.set_xlabel('Position')
ax2.set_ylabel('Time')

# 添加温度变化的探针点
probe_positions = [-0.5, 0, 0.5]
for pos in probe_positions:
    idx = np.argmin(np.abs(x - pos))
    ax2.plot([pos] * len(t), t, 'r.', markersize=2)
    ax2.text(pos, t_max*1.05, f'x={pos}', ha='center', va='bottom')

# 绘制能量随时间的变化
ax3 = fig.add_subplot(3, 2, (3, 4))
ax3.semilogy(t, energy)
ax3.set_title('Energy over Time')
ax3.set_xlabel('Time')
ax3.set_ylabel('Energy (log scale)')
ax3.grid(True)

# 绘制能量损失率随时间的变化
ax4 = fig.add_subplot(3, 2, (5, 6))
ax4.semilogy(t[1:], np.abs(energy_loss_rate))
ax4.set_title('Absolute Energy Loss Rate over Time')
ax4.set_xlabel('Time')
ax4.set_ylabel('|Energy Loss Rate| (log scale)')
ax4.grid(True)

# 绘制探针点的温度变化
ax5 = fig.add_subplot(3, 2, 3)
for pos in probe_positions:
    idx = np.argmin(np.abs(x - pos))
    ax5.semilogy(t, solution[:, idx], label=f'x={pos}')
ax5.set_title('Temperature at Probe Points')
ax5.set_xlabel('Time')
ax5.set_ylabel('Temperature (log scale)')
ax5.legend()
ax5.grid(True)

plt.tight_layout()
plt.show()