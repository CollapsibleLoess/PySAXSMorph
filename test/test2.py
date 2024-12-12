import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

# 定义孔隙度范围
porosity = np.linspace(0, 1, 500)

# 计算alpha值
alpha = np.sqrt(2) * erfinv(1 - 2 * porosity)

# 创建图形
plt.figure(figsize=(10, 6))
plt.plot(porosity, alpha, label=r'$\alpha = \sqrt{2} \cdot \mathrm{erfinv}(1 - 2 \cdot \mathrm{porosity})$', color='blue')
plt.xlabel('Porosity')
plt.ylabel(r'$\alpha$')
plt.title(r'Relationship between Porosity and $\alpha$')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()


def calcgr(rpts, r_gammar, porosity):
    def gr_fun(test_x, alpha, porosity, gammaR):
        # 实现了龙贝格积分（RombergIntegration）的函数
        # decdigs 的值决定了这个迭代过程进行的深度，即进行了多少轮迭代和外推。
        def rombInt(bottom, top, decdigs, alpha):
            # 定义被积分的函数
            def integrand(x, alpha):
                return 2 * np.exp(-alpha ** 2 / (np.sqrt(2.0 - x ** 2) * x + 1.0)) / np.sqrt(2.0 - x ** 2)

            # 计算积分
            result = romberg(integrand, bottom, top, args=(alpha,), divmax=decdigs - 1, tol=10 ** (-decdigs))
            return result

        value = rombInt(test_x, 1.0, 5, alpha)
        value /= 2 * np.pi
        result = np.abs(porosity - gammaR - value)
        return result

    def calc_gr(alpha, porosity, gammaR):
        test_x = 0.5
        steps = [0.01, 0.001, 1e-4]
        for step in steps:
            current = gr_fun(test_x, alpha, porosity, gammaR)
            above = gr_fun(test_x + step, alpha, porosity, gammaR)
            below = gr_fun(test_x - step, alpha, porosity, gammaR)
            # 简化逻辑，直接使用循环调整 test_x 的值
            while (above < current or below < current):
                if above < current:
                    test_x += step
                else:
                    test_x -= step
                current = gr_fun(test_x, alpha, porosity, gammaR)
                above = gr_fun(test_x + step, alpha, porosity, gammaR)
                below = gr_fun(test_x - step, alpha, porosity, gammaR)
        # 返回计算结果
        return test_x * np.sqrt(2.0 - test_x * test_x)

    # 创建一个新字典来存储计算后的数据
    r_gr = {'R': np.zeros(rpts), 'gR': np.zeros(rpts)}  # 创建一个有rpts行和2列的数组
    alpha = np.sqrt(2.0) * erfinv(1.0 - porosity * 2.0)
    for i in range(rpts):
        r_gr['R'][i] = r_gammar['R'][i]
        r_gr['gR'][i] = calc_gr(alpha, porosity, r_gammar['gammaR'][i])
    # 导出r_gr
    export_datas(r_gr, 'R', 'gR', log_scale=False, clear_image=True, plot_type="line")
    return r_gr, alpha