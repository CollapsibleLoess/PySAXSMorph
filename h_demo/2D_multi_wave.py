import numpy as np
from g_export.export1d import export_datas
from g_export.export2d import export_2d_images

# 参数设置
grid_size = 100  # 网格大小
x_range = (0, 1000)  # x 方向的范围
y_range = (0, 1000)  # y 方向的范围
threshold = 2
num_waves = 1000  # 波的数量
kmin = 0.00318  # 图片周期内实现循环（过小无法实现周期内多次波动）
kmax = 0.141  # （过大，格子内部叠加多次，无法在分辨率内识别长程关联结构）
phi_range = (0, 2 * np.pi)  # 相位的范围


# 随机生成多个波矢量和相位
def generate_log_scale_wave_vectors(kmin, kmax, num_waves):
    """
    在对数坐标下生成波矢量。

    参数:
    kmin (float): 波矢量分量的最小值。
    kmax (float): 波矢量分量的最大值。
    num_waves (int): 生成的波矢量数量。

    返回:
    np.ndarray: 生成的波矢量数组，形状为 (num_waves, 2)。
    """
    # 在对数尺度上生成均匀分布的随机数
    log_kmin = np.log10(kmin)
    log_kmax = np.log10(kmax)
    log_k_values = np.random.uniform(log_kmin, log_kmax, (num_waves, 2))

    # 将对数尺度上的随机数转换回线性尺度
    k_values = 10 ** log_k_values

    return k_values


k_values = generate_log_scale_wave_vectors(kmin, kmax, num_waves)  # 二维波矢量
phi_values = np.random.uniform(phi_range[0], phi_range[1], num_waves)

# 定义二维网格
x = np.linspace(x_range[0], x_range[1], grid_size)
y = np.linspace(y_range[0], y_range[1], grid_size)
X, Y = np.meshgrid(x, y, indexing='ij')

# 计算二维随机场
density_field = np.zeros_like(X)
for i in range(num_waves):
    density_field += np.sqrt(2 / num_waves) * np.cos(k_values[i, 0] * X + k_values[i, 1] * Y + phi_values[i])

gaussian_density_field = density_field
# 将二维数组展平成一维数组
flattened_density_field = gaussian_density_field.flatten()

# 使用 numpy.histogram 进行统计，指定区间的间隔为0.1
bin_width = 0.01
bins = np.arange(flattened_density_field.min(), flattened_density_field.max() + bin_width, bin_width)
hist, bin_edges = np.histogram(flattened_density_field, bins=bins)

# 准备数据字典
data_dict = {
    'intensity': bin_edges[:-1],  # 使用 bin_edges[:-1] 作为强度值
    'count': hist  # 使用 hist 作为数量
}

# 调用 export_datas 函数
export_datas(data_dict, 'intensity', 'count', log_scale=False, plot_type='line', clear_image=True)

# 裁剪高斯密度场
clipped_density_field = np.where(gaussian_density_field > threshold, 0, 1)

# 导出二维图像
export_2d_images(X, Y, gaussian_density_field, x_label='x', y_label='y', d_label='Summed Amplitude')
export_2d_images(X, Y, clipped_density_field, x_label='x', y_label='y', d_label='Clipped Amplitude')
