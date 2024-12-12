import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 假设 globalvars 是一个已经定义的模块，其中包含 output_folder 和 plot_counter
import globalvars
All = globalvars.AllParams()

def export_2d_grayimages(X, Y, D):
    exporter = GrayImageExporter(X, Y, D)
    exporter.plot()

class GrayImageExporter:
    def __init__(self, X, Y, D):
        self.X = X
        self.Y = Y
        self.D = D
        self.new_folder = os.path.join(globalvars.output_path, "restructure_gray")
        os.makedirs(self.new_folder, exist_ok=True)

    def plot(self):
        # 二值化处理：大于0的部分设为1（白色），其他部分为0（黑色）
        binary_image = np.where(self.D > 0, 1, 0).astype(np.uint8)  # 转换为 uint8 类型

        plt.figure(figsize=(All.boxsize, All.boxsize), dpi=1)  # Allboxsize个像素
        # 设置插值为 'none'，以防止模糊效果
        plt.imshow(binary_image, cmap='gray', extent=[self.X.min(), self.X.max(), self.Y.min(), self.Y.max()], interpolation='none')
        plt.axis('off')  # 隐藏坐标轴
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除边缘白色空间

        globalvars.plot_counter += 1
        plt.savefig(os.path.join(self.new_folder, f"{globalvars.plot_counter}.png"), dpi=1, bbox_inches='tight', pad_inches=0, pil_kwargs={'mode': '1'})
        plt.close()

