import numpy as np
from g_export.export1d import export_datas


def merg_dict(Guinier_q_iq, cor_q_iq, Porod_q_iq):
    merged_dict = {'Q': np.concatenate([Guinier_q_iq['Q'], cor_q_iq['Q'], Porod_q_iq['Q']]),
                   'ExtraIQ': np.concatenate([Guinier_q_iq.get('GuinierExtraIQ', []),
                                              cor_q_iq.get('CorIQ', []),
                                              Porod_q_iq.get('PorodExtraIQ', [])])}
    # 输出外推结果**************************************************************************************
    export_datas(merged_dict, 'Q', 'ExtraIQ')
    return merged_dict
