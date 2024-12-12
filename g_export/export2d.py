import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 假设 globalvars 是一个已经定义的模块，其中包含 output_folder 和 plot_counter
import globalvars

def export_2d_images(X, Y, D, x_label='x', y_label='y', d_label='Amplitude', clear_image=True):
    exporter = ImageExporter(X, Y, D, x_label, y_label, d_label, clear_image)
    exporter.plot()

class ImageExporter:
    def __init__(self, X, Y, D, x_label, y_label, d_label, clear_image):
        self.X = X
        self.Y = Y
        self.D = D
        self.x_label = x_label
        self.y_label = y_label
        self.d_label = d_label
        self.save_plot = clear_image
        self.title = f"{self.d_label} over {self.x_label} and {self.y_label}"
        self.new_folder = os.path.join(globalvars.output_path, "restructure")
        os.makedirs(self.new_folder, exist_ok=True)

    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.contourf(self.X, self.Y, self.D, cmap='viridis')
        plt.colorbar(label=self.d_label)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        if self.save_plot:
            globalvars.plot_counter += 1
            plt.savefig(os.path.join(self.new_folder, f"{globalvars.plot_counter}_{self.title}.png"), dpi=600)
            plt.close()