import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义网格
x = np.array([0, 1])
y = np.array([2, 3, 4])
z = np.array([8, 9])
X, Y, Z = np.meshgrid(x, y, z)
print(X)
print(".........")
print(Y)
print(".........")
print(Z)

# 可视化网格点
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制网格点
ax.scatter(X, Y, Z, color='b')

# 给每个点加标签，包含索引和实际坐标
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            idx_label = f'({i},{j},{k})'
            coord_label = f'({X[i, j, k]:.1f},{Y[i, j, k]:.1f},{Z[i, j, k]:.1f})'
            ax.text(X[i, j, k], Y[i, j, k], Z[i, j, k], f'{idx_label}\n{coord_label}', color='red')


# 设置轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 设置标题
ax.set_title('3D Grid Points with Indices')

# 显示图形
plt.show()
