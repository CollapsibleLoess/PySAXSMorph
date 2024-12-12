import os
from g_export.export1d import export_datas
import numpy as np


def import_data_as_dict(file_path):
    # 获取文件扩展名***********************************************************************
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() in ['.txt']:
        raw_q_iq = read_data_from_file(file_path, ' ')
        # 导出原始数据*********************************************************************
        export_datas(raw_q_iq, "Q", "IQ")  # Fig1
    else:
        print(f"Unsupported file type: {file_extension}")
    return raw_q_iq


def read_data_from_file(filename, delimiter):
    npraw_q_iq = np.genfromtxt(filename, delimiter=delimiter)
    raw_q_iq = {'Q': npraw_q_iq[:, 0], 'IQ': npraw_q_iq[:, 1]}
    return raw_q_iq
