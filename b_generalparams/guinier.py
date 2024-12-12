import numpy as np
from c_publicfunctions import linearfitting as lf, generatelogspace as gf
from g_export.export1d import export_datas
import globalvars
AP = globalvars.AllParams()


def guinier_fit_extra(cor_q_iq):
    # 创建一个新字典来存储计算后的数据
    q2_lniq = {'Q2': cor_q_iq['Q'] ** 2, 'LN(IQ)': np.log(cor_q_iq['CorIQ'])}
    # 求取斜率、截距、相关系数
    slope, intercept, r_value = lf.linear_fitting(q2_lniq, 'Q2', 'LN(IQ)', start_index=0, end_index=0.03)
    # 线性拟合参数
    globalvars.paras_dict['GuinierAppr(LnI(0))'] = intercept
    globalvars.paras_dict['GuinierAppr(-1/3RG2)'] = slope
    globalvars.paras_dict['GuinierApprR'] = r_value

    # 线性拟合结果
    fit_q2_lniq = {'Q2': q2_lniq['Q2'], 'LN(IQ)': (q2_lniq['Q2'] * slope + intercept)}

    # 导出guinier近似拟合数据***********************************************************************
    export_datas(q2_lniq, 'Q2', 'LN(IQ)', log_scale=True, clear_image=False)
    export_datas(fit_q2_lniq, 'Q2', 'LN(IQ)', log_scale=True, plot_type="line")

    # 获得外推q2
    extra_porod_q2 = gf.generate_logspace(AP.minguinier, min(cor_q_iq['Q']), AP.numguinier) ** 2

    # 计算外推结果
    extra_q2_lniqq4 = {'Q2': extra_porod_q2, 'LN(IQ)': (extra_porod_q2 * slope + intercept)}
    Guinier_q_iq = {'Q': np.sqrt(extra_q2_lniqq4['Q2']), 'GuinierExtraIQ': np.exp(extra_q2_lniqq4['LN(IQ)'])}
    return Guinier_q_iq