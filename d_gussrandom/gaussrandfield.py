import math
import time
from numba import njit, prange
import globalvars
from g_export.export1d import export_datas
from g_export.export2d import export_2d_images
from g_export.export2dgray import export_2d_grayimages
from joblib import Parallel, delayed
import numpy as np
import numba as nb
from numba import prange


def generate_matrix(k_fk, boxres, boxsize, alpha, num_waves, kconst):
    def generate_Kn_and_phin(k_fk, num_samples):
        minfk = np.min(k_fk['fK'])
        maxfk = np.max(k_fk['fK'])
        Kn = np.zeros((num_samples, 3), dtype=np.float32)
        phin = np.random.uniform(0, 2 * math.pi, num_waves).astype(np.float32)
        for j in range(num_samples):
            while True:
                # 从 k_fk['K'] 中随机抽取一个 randk
                rand_idx = np.random.randint(0, len(k_fk['K']))
                randk = k_fk['K'][rand_idx]
                randfk = np.random.uniform(minfk, maxfk)

                # 获取对应的 fK 值
                corresponding_fk = k_fk['fK'][rand_idx]

                if randfk <= corresponding_fk:
                    kvec = np.random.uniform(-1, 1, 3)
                    kvecnorm = np.linalg.norm(kvec)
                    Kn[j] = (kvec / kvecnorm * randk * kconst).astype(np.float32)
                    break
        return Kn, phin


    def process_and_export_density_field(gaussian_density_field, bin_width):
        """
        将二维高斯密度场展平成一维数组，并统计直方图数据，最后调用 export_datas 函数导出数据。

        参数:
        gaussian_density_field (numpy.ndarray): 二维高斯密度场。
        bin_width (float): 直方图的区间间隔。
        export_datas (function): 用于导出数据的函数。
        """
        # 将二维数组展平成一维数组
        flattened_density_field = gaussian_density_field.flatten()

        # 使用 numpy.histogram 进行统计
        bins = np.arange(flattened_density_field.min(), flattened_density_field.max() + bin_width, bin_width)
        hist, bin_edges = np.histogram(flattened_density_field, bins=bins)

        # 准备数据字典
        flattened_density = {
            'intensity': bin_edges[:-1],  # 使用 bin_edges[:-1] 作为强度值
            'count': hist  # 使用 hist 作为数量
        }
        return flattened_density


    def compute_stack_density(num_waves, k_fk, boxsize, boxres):
        """
        计算并返回三维密度场stack_density。

        参数:
        - num_waves: int, 波的数量。
        - k_fk: 生成Kn和phin的参数。
        - boxsize: float, 网格的尺寸。
        - boxres: int, 网格的分辨率。

        返回:
        - stack_density: numpy.ndarray, 计算得到的三维密度场。
        """
        # 初始化参数

        Kn, phin = generate_Kn_and_phin(k_fk, num_waves)
        print(f"计算kn...耗时：{time.time() - globalvars.start_time}秒。")

        # 生成一维网格
        x = np.linspace(1, boxsize, boxres, dtype=np.float32)
        y = np.linspace(1, boxsize, boxres, dtype=np.float32)
        z = np.linspace(1, boxsize, boxres, dtype=np.float32)

        # 初始化phases数组
        phases = np.zeros((boxres, boxres, boxres), dtype=np.float32)

        # 预计算x和y的网格
        X, Y = np.meshgrid(x, y, indexing='ij')

        # 逐层计算Z切片
        for k, zi in enumerate(z):
            # 计算当前Z层的相位
            phase = (Kn[:, 0][:, np.newaxis, np.newaxis] * X +
                     Kn[:, 1][:, np.newaxis, np.newaxis] * Y +
                     Kn[:, 2][:, np.newaxis, np.newaxis] * zi +
                     phin[:, np.newaxis, np.newaxis])

            # 计算余弦之和并存储在phases数组中
            phases[:, :, k] = np.sum(np.cos(phase), axis=0)

        # 计算stack_density
        stack_density = np.sqrt(2 / num_waves) * phases
        return X, Y, stack_density



    @nb.njit(parallel=True, fastmath=True)
    def compute_stack_density_optimized(num_waves, Kn, phin, boxsize, boxres):
        """
        计算并返回三维密度场stack_density。

        参数:
        - num_waves: int, 波的数量。
        - Kn: numpy.ndarray, 波矢数组。
        - phin: numpy.ndarray, 相位数组。
        - boxsize: float, 网格的尺寸。
        - boxres: int, 网格的分辨率。

        返回:
        - stack_density: numpy.ndarray, 计算得到的三维密度场。
        """
        # 生成一维网格
        x = np.arange(1, boxsize + 1, (boxsize - 1) / (boxres - 1), dtype=np.float32)

        # 初始化stack_density数组
        stack_density = np.empty((boxres, boxres, boxres), dtype=np.float32)

        # 计算常数因子
        factor = np.sqrt(2 / num_waves)

        # 并行计算每个Z切片
        for k in prange(boxres):
            for i in range(boxres):
                for j in range(boxres):
                    phase = 0.0
                    for n in range(num_waves):
                        phase += np.cos(Kn[n, 0] * x[i] + Kn[n, 1] * x[j] + Kn[n, 2] * x[k] + phin[n])
                    stack_density[i, j, k] = factor * phase

        return stack_density


    Kn, phin = generate_Kn_and_phin(k_fk, num_waves)
    stack_density = compute_stack_density_optimized(num_waves, Kn, phin, boxsize, boxres)


    # X, Y, stack_density = compute_stack_density(num_waves, k_fk, boxsize, boxres)
    # print(f"计算stack_density...耗时：{time.time() - globalvars.start_time}秒。")

    # 重新生成 X 和 Y，并导出每个Z平面上的2D切片
    x = np.linspace(1, boxsize, boxres, dtype=np.float32)
    y = np.linspace(1, boxsize, boxres, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 导出展平的数据
    flattened_density = process_and_export_density_field(stack_density, 0.1)
    # 调用 export_datas 函数
    export_datas(flattened_density, 'intensity', 'count', log_scale=False, plot_type='line', clear_image=True)

    # 导出每个Z平面上的2D切片
    for i in range(boxres):
        Z_slice = stack_density[:, :, i]  # 获取当前Z值的二维切片
        # 导出每个Z平面上的2D切片
        #export_2d_images(X, Y, Z_slice, x_label='x', y_label='y', d_label=f'Amplitude')
        Z_clipped = Z_slice > alpha
        # 导出每个Z平面上的2D切片
        #export_2d_images(X, Y, Z_clipped, x_label='x', y_label='y', d_label=f'Amplitude')
        export_2d_grayimages(X, Y, Z_clipped)