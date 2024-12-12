from c_publicfunctions import linearfitting as lf, generatelogspace as gf
from g_export.export1d import export_datas
import globalvars


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