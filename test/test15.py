import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建示例数据
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [2, 4, 6, 8, 10],
    'C': [5, 4, 3, 2, 1],
    'D': [1, 3, 5, 7, 9]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 计算相关系数矩阵
corr_matrix = df.corr()

# 绘制热力图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

# 显示图像
plt.title('Pearson Correlation Heatmap')
plt.show()
