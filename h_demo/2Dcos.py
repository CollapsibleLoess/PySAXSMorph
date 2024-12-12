import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from mpl_toolkits.mplot3d import Axes3D


def plot_wave(kx, ky, phi):
    x = np.linspace(1, 100, 100)
    y = np.linspace(1, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(kx * X + ky * Y + phi)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    fig.colorbar(surf)
    ax.set_title(f'3D Cosine Wave: kx={kx:.4f}, ky={ky:.4f}, phi={phi:.2f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


interact(plot_wave,
         kx=FloatSlider(min=0, max=1, step=np.pi * 0.01, value=0, description='kx'),
         ky=FloatSlider(min=0, max=1, step=np.pi * 0.01, value=0, description='ky'),
         phi=FloatSlider(min=-2 * np.pi, max=2 * np.pi, step=0.01, value=0, description='phi'))
