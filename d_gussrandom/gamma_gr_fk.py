import math
import numpy as np
from scipy.integrate import romberg, quad
from scipy.special import erfinv
from g_export.export1d import export_datas
from scipy.integrate import quad
from scipy.optimize import fsolve

# 示例调用
# 注意：你需要提供 extrap 数组和其他参数的实际值
# calc_gamma_of_r(rpts, epts, rmax, rmin, extrap, porosity)
# rpts: 整数，表示 gamma 计算的点数。
# epts: 整数，表示用于计算的数据点数。
# rmax 和 rmin: 浮点数，表示 r 的最大值和最小值。
# extrap: 二维浮点数组，存储用于计算的数据。
# porosity: 浮点数，表示孔隙度。
# outputfilepath 和 outputfilestring: 字符串，表示输出文件的路径和文件名。


def calcgammar(rpts, rmax, rmin, extrap, epts, porosity):
    # 创建一个新字典来存储计算后的数据
    r_gammar = {'R': np.zeros(rpts), 'gammaR': np.zeros(rpts), 'raw_gammaR': np.zeros(rpts)}      # 创建一个有rpts行和2列的数组
    raw_gamma_r = np.zeros(rpts)       # 长度为rpts的一维数组
    gamma_intergal_q = np.zeros(epts)          # 长度为epts的一维数组
    gamma_r_max = 0.0

    for i in range(rpts):
        r_gammar['R'][i] = i * (rmax - rmin) / rpts + rmin      # 计算了从rmin到rmax范围内的一个等间距的值
        trapz = 0.0                                         # 累加梯形法则计算的面积
        for j in range(epts):
            qr = r_gammar['R'][i] * extrap['Q'][j]
            sinqr = math.sin(qr)
            gamma_intergal_q[j] = (4 * math.pi * extrap['Q'][j]**2 * extrap['ExtraIQ'][j] * sinqr) / qr
            if j >0:
                trapz += (extrap['Q'][j] - extrap['Q'][j-1]) * \
                         (gamma_intergal_q[j] + gamma_intergal_q[j - 1]) / 2.0
        raw_gamma_r[i] = trapz
        if raw_gamma_r[i] > gamma_r_max:
            gamma_r_max = raw_gamma_r[i]
    # if abs(gamma_r[-1]) > 0.001:
    #     print("Gamma(r) not converged...")
    for i in range(rpts):
        r_gammar['raw_gammaR'][i] = raw_gamma_r[i]
    # 导出r_gammar
    export_datas(r_gammar, 'R', 'raw_gammaR', log_scale=False, clear_image=True, plot_type="line")

    for i in range(rpts):
        r_gammar['gammaR'][i] = (porosity - porosity**2) * raw_gamma_r[i] / gamma_r_max + porosity**2
    # 导出归一化r_gammar
    export_datas(r_gammar, 'R', 'gammaR', log_scale=False, clear_image=True, plot_type="line")

    return r_gammar


def calcgr(rpts, r_gammar, porosity):

    def target_function(g, alpha, target):

        def integrand(x, alpha):
            return 2 * np.exp(-alpha ** 2 / (np.sqrt(2.0 - x ** 2) * x + 1.0)) / np.sqrt(2.0 - x ** 2)
        result = quad(integrand, g, 1, args=(alpha,))

        return result[0] - target

    def calc_gr(alpha, porosity, gammaR):
        target_value = 2 * np.pi * (porosity - gammaR)
        g_r_solution = fsolve(target_function, 1, args=(alpha, target_value))
        return g_r_solution

    r_gr = {'R': np.zeros(rpts), 'gR': np.zeros(rpts)}
    alpha = np.sqrt(2.0) * erfinv(1.0 - porosity * 2.0)

    for i in range(rpts):
        r_gr['R'][i] = r_gammar['R'][i]
        r_gr['gR'][i] = calc_gr(alpha, porosity, r_gammar['gammaR'][i])

    # 导出r_gr
    export_datas(r_gr, 'R', 'gR', log_scale=False, clear_image=True, plot_type="line")
    return r_gr, alpha



def calc_spectral_function(kpts, kmax, kmin, rpts, r_gr):
    # 创建一个新字典来存储计算后的数据
    k_fk = {'K': np.zeros(kpts), 'fK': np.zeros(kpts)}      # 创建一个有rpts行和2列的数组
    specfunint = np.zeros(rpts)
    # 生成对数分布的数组
    log_array = np.logspace(np.log10(kmin), np.log10(kmax), kpts)
    for i in range(kpts):
        k_fk['K'][i] = log_array[i]
        trap = 0.0
        for j in range(rpts):
            kr = k_fk['K'][i] * r_gr['R'][j]
            specfunint[j] = 4 * math.pi * (r_gr['R'][j] ** 2) * r_gr['gR'][j] * math.sin(kr) / (kr)
            if j > 0:
                trap += (specfunint[j] + specfunint[j - 1]) * (r_gr['R'][j] - r_gr['R'][j-1]) / 2.0
        k_fk['fK'][i] = abs(trap)
    # 导出r_gr
    export_datas(k_fk, 'K', 'fK', log_scale=True, clear_image=True, plot_type="line")
    return k_fk
