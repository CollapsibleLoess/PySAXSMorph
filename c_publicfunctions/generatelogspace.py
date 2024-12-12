import numpy as np


def generate_logspace(start, stop, num_points):
    # 将起始和结束点在对数坐标上取对数
    start_log = np.log10(start)
    stop_log = np.log10(stop)

    # 在对数坐标上生成等距的数列
    logspace = np.linspace(start_log, stop_log, num_points)
    result = 10 ** logspace

    return result
