import numpy as np
from c_publicfunctions import linearfitting as lf, generatelogspace as gf
from g_export.export1d import export_datas
import globalvars
from globalvars import density_p


def sub_incoh_bg(raw_q_iq):
    # 创建一个新字典来存储计算后的数据
    q4_iqq4 = {'Q4': raw_q_iq['Q'] ** 4, 'IQQ4': raw_q_iq['IQ'] * (raw_q_iq['Q'] ** 4)}
    # 线性拟合求取斜率、截距、相关系数
    slope, intercept, r_value = lf.linear_fitting(q4_iqq4, "Q4", "IQQ4", start_index=0, end_index=1)
    globalvars.paras_dict['IncoherentScatBg'] = slope
    globalvars.paras_dict['IncoherentScatBgFitR'] = r_value
    # 线性拟合结果
    fit_q4_iqq4 = {'Q4': q4_iqq4['Q4'], 'FitIQQ4': (q4_iqq4['Q4'] * slope + intercept)}

    # 导出非相干散射背景拟合数据*********************************************************************
    export_datas(q4_iqq4, "Q4", "IQQ4", log_scale=False, clear_image=False)
    export_datas(fit_q4_iqq4, "Q4", "FitIQQ4", log_scale=False, plot_type="line")

    # 导出扣除背底*******************************************************************************
    sub_q_iq = {'Q': raw_q_iq['Q'], 'SubIQ': (raw_q_iq['IQ'] - slope * 0.5)}
    export_datas(sub_q_iq, "Q", "SubIQ")

    return sub_q_iq


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
    globalvars.paras_dict['PorodFit(LnK)'] = intercept
    globalvars.paras_dict['PorodFit(b)'] = slope
    globalvars.paras_dict['PorodFitR'] = r_value
    # 线性拟合结果
    fit_q2_lniqq4 = {'Q2': q2_lniqq4['Q2'], 'LN(IQQ4)': (q2_lniqq4['Q2'] * slope + intercept)}

    # 导出porod校正后拟合数据***********************************************************************
    export_datas(q2_lniqq4, 'Q2', 'LN(IQQ4)', log_scale=False, clear_image=False)
    export_datas(fit_q2_lniqq4, 'Q2', 'LN(IQQ4)', log_scale=False, plot_type="line")

    # 获得外推q2
    extra_porod_q2 = gf.generate_logspace(max(cor_q_iq['Q']), 0.581, 2) ** 2

    # 计算外推结果
    extra_q2_lniqq4 = {'Q2': extra_porod_q2, 'LN(IQQ4)': (extra_porod_q2 * slope + intercept)}
    Porod_q_iq = {'Q': np.sqrt(extra_q2_lniqq4['Q2']), 'PorodExtraIQ': np.exp(extra_q2_lniqq4['LN(IQQ4)']) / (np.sqrt(extra_q2_lniqq4['Q2']) ** 4)}

    return Porod_q_iq


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
    extra_porod_q2 = gf.generate_logspace(0.00515, min(cor_q_iq['Q']), 2) ** 2

    # 计算外推结果
    extra_q2_lniqq4 = {'Q2': extra_porod_q2, 'LN(IQ)': (extra_porod_q2 * slope + intercept)}
    Guinier_q_iq = {'Q': np.sqrt(extra_q2_lniqq4['Q2']), 'GuinierExtraIQ': np.exp(extra_q2_lniqq4['LN(IQ)'])}
    return Guinier_q_iq


def merg_dict(Guinier_q_iq, cor_q_iq, Porod_q_iq):
    merged_dict = {'Q': np.concatenate([Guinier_q_iq['Q'], cor_q_iq['Q'], Porod_q_iq['Q']]),
                   'ExtraIQ': np.concatenate([Guinier_q_iq.get('GuinierExtraIQ', []),
                                              cor_q_iq.get('CorIQ', []),
                                              Porod_q_iq.get('PorodExtraIQ', [])])}
    # 输出外推结果**************************************************************************************
    export_datas(merged_dict, 'Q', 'ExtraIQ')
    return merged_dict


def invariant_integral(extrap_q_iq):
    q_q2iq = {'Q': extrap_q_iq['Q'], 'Q2IQ': (extrap_q_iq['Q']**2 * extrap_q_iq['ExtraIQ'])}
    x = q_q2iq['Q']
    y = q_q2iq['Q2IQ']
    print(x, y)
    invariant_q_iq = np.trapz(y, x)
    invariant_a = invariant_q_iq/(2*((np.pi*density_p) ** 2))
    phi_1 = (1 - (1 - 4 * invariant_a) ** 0.5) / 2
    phi_2 = (1 + (1 - 4 * invariant_a) ** 0.5) / 2

    # invariant参数
    globalvars.paras_dict['Invariant(A)'] = invariant_a
    globalvars.paras_dict['Invariant(phi_1)'] = phi_1
    globalvars.paras_dict['Invariant(phi_2)'] = phi_2

    return invariant_q_iq, invariant_a, phi_1, phi_2
