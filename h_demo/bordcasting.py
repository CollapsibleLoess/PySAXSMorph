import numpy as np
import matplotlib.pyplot as plt

# 网格分辨率
boxres = 5

# 生成一维网格
x = np.linspace(0, 1, boxres)
y = np.linspace(0, 1, boxres)
z = np.linspace(0, 1, boxres)

# 生成三维网格
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 样本参数
samples = [
    {'K': np.array([1, 0, 0]), 'phi': 0},
    {'K': np.array([0, 1, 0]), 'phi': np.pi / 4}
]

# 计算相位矩阵
phase_matrices = []
for sample in samples:
    K = sample['K']
    phi = sample['phi']

    # 扩展维度
    Kx, Ky, Kz = K[:, np.newaxis, np.newaxis, np.newaxis]

    # 计算相位矩阵
    phase_matrix = Kx * X + Ky * Y + Kz * Z + phi
    phase_matrices.append(phase_matrix)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for i, phase_matrix in enumerate(phase_matrices):
    ax = axes[i]
    c = ax.contourf(X[:, :, 0], Y[:, :, 0], phase_matrix[:, :, 0], cmap='viridis')
    ax.set_title(f'Sample {i + 1}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(c, ax=ax)

plt.show()
