import numpy as np
from c_publicfunctions import linearfitting as lf, generatelogspace as gf
from g_export.export1d import export_datas
import globalvars
AP = globalvars.AllParams()

def cal_porod_paras(sub_q_iq):
    # 创建一个新字典来存储计算后的数据
    q2_lniqq4 = {'Q2': sub_q_iq['Q'] ** 2, 'LN(IQQ4)': np.log(sub_q_iq['SubIQ'] * (sub_q_iq['Q'] ** 4))}
    # 求取斜率、截距、相关系数
    slope, intercept, r_value = lf.linear_fitting(q2_lniqq4, 'Q2', 'LN(IQQ4)', start_index=0.9, end_index=1)
    # 线性拟合参数
    globalvars.paras_dict['PorodCorDeviBgLnK'] = intercept
    globalvars.paras_dict['PorodCorDeviBgb'] = slope
    globalvars.paras_dict['PorodCorDeviBgR'] = r_value
    # 线性拟合结果
    fit_q2_lniqq4 = {'Q2': q2_lniqq4['Q2'], 'LN(IQQ4)': (q2_lniqq4['Q2'] * slope + intercept)}

    # porod校正数据
    cor_q_iq = {'Q': sub_q_iq['Q'], 'CorIQ': np.exp(-slope * np.square(sub_q_iq['Q'])) * sub_q_iq['SubIQ']}

    # 导出porod校正前后数据*************************************************************************
    export_datas(sub_q_iq, 'Q', 'SubIQ', clear_image=False)
    export_datas(cor_q_iq, 'Q', 'CorIQ')

    # 导出porod拟合数据***************************************************************************
    export_datas(q2_lniqq4, 'Q2', 'LN(IQQ4)', log_scale=False, clear_image=False)
    export_datas(fit_q2_lniqq4, 'Q2', 'LN(IQQ4)', log_scale=False, clear_image=False, plot_type="line")

    return cor_q_iq


def porod_fit_extra(cor_q_iq):
    # 创建一个新字典来存储计算后的数据
    q2_lniqq4 = {'Q2': cor_q_iq['Q'] ** 2, 'LN(IQQ4)': np.log(cor_q_iq['CorIQ'] * (cor_q_iq['Q'] ** 4))}
    # 求取斜率、截距、相关系数
    slope, intercept, r_value = lf.linear_fitting(q2_lniqq4, 'Q2', 'LN(IQQ4)', start_index=0.9, end_index=1)
    # 线性拟合参数
    globalvars.paras_dict['PorodFitLnK'] = intercept
    globalvars.paras_dict['PorodFitb'] = slope
    globalvars.paras_dict['PorodFitR'] = r_value
    # 线性拟合结果
    fit_q2_lniqq4 = {'Q2': q2_lniqq4['Q2'], 'LN(IQQ4)': (q2_lniqq4['Q2'] * slope + intercept)}

    # 导出porod校正后拟合数据***********************************************************************
    export_datas(q2_lniqq4, 'Q2', 'LN(IQQ4)', log_scale=False, clear_image=False)
    export_datas(fit_q2_lniqq4, 'Q2', 'LN(IQQ4)', log_scale=False, plot_type="line")

    # 获得外推q2
    extra_porod_q2 = gf.generate_logspace(max(cor_q_iq['Q']), AP.maxporod, AP.numporod) ** 2

    # 计算外推结果
    extra_q2_lniqq4 = {'Q2': extra_porod_q2, 'LN(IQQ4)': (extra_porod_q2 * slope + intercept)}
    Porod_q_iq = {'Q': np.sqrt(extra_q2_lniqq4['Q2']), 'PorodExtraIQ': np.exp(extra_q2_lniqq4['LN(IQQ4)']) / (np.sqrt(extra_q2_lniqq4['Q2']) ** 4)}

    return Porod_q_iq