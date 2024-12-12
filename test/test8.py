import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义网格
x = np.linspace(0, 5, 6)
y = np.linspace(0, 4, 5)
z = np.linspace(0, 3, 4)
X, Y, Z = np.meshgrid(x, y, z)

# 创建一个新的图形窗口
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制网格点
ax.scatter(X, Y, Z, marker='o', color='b')

# 设置轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 设置标题
ax.set_title('3D Grid Points')

# 显示图形
plt.show()
