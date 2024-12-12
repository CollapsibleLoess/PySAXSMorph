import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义参数
L = 1.0  # 空间长度
N = 5  # 使用的试函数数量
alpha = 0.1  # 热扩散系数


# 定义试函数（这里使用正弦函数）
def phi(x, n):
    return np.sin(n * np.pi * x / L)


# 定义试函数的导数
def dphi(x, n):
    return n * np.pi / L * np.cos(n * np.pi * x / L)


# 计算质量矩阵和刚度矩阵
def compute_matrices():
    M = np.zeros((N, N))
    K = np.zeros((N, N))
    x = np.linspace(0, L, 1000)
    dx = x[1] - x[0]
    for i in range(N):
        for j in range(N):
            M[i, j] = np.sum(phi(x, i + 1) * phi(x, j + 1)) * dx
            K[i, j] = alpha * np.sum(dphi(x, i + 1) * dphi(x, j + 1)) * dx
    return M, K


# 定义ODE系统
def ode_system(a, t, M, K):
    return np.linalg.solve(M, -np.dot(K, a))


# 初始条件（假设初始温度分布为 sin(pi*x/L)）
def initial_condition(x):
    return np.sin(np.pi * x / L)


# 主程序
def main():
    M, K = compute_matrices()

    # 计算初始系数
    x = np.linspace(0, L, 1000)
    a0 = np.zeros(N)
    for i in range(N):
        a0[i] = np.sum(initial_condition(x) * phi(x, i + 1)) * (x[1] - x[0])

    # 求解ODE
    t = np.linspace(0, 1, 100)  # 时间从0到1
    solution = odeint(ode_system, a0, t, args=(M, K))

    # 绘制结果
    X, T = np.meshgrid(x, t)
    U = np.zeros_like(X)
    for i in range(N):
        U += solution[:, i][:, np.newaxis] * phi(X, i + 1)

    plt.figure(figsize=(10, 8))
    plt.contourf(X, T, U, levels=20, cmap='hot')
    plt.colorbar(label='Temperature')
    plt.xlabel('Position (x)')
    plt.ylabel('Time (t)')
    plt.title('Heat Equation Solution using Variational Method')
    plt.show()


if __name__ == "__main__":
    main()
