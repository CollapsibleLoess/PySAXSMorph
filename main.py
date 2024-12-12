import os
import time
from datetime import datetime
import numpy as np
import globalvars
from b_generalparams import subbg, porod, guinier, extrapolate, invariant
from d_gussrandom.gamma_gr_fk import calcgammar, calcgr, calc_spectral_function
from a_preprocess.getdata import import_data_as_dict
from d_gussrandom.gaussrandfield import generate_matrix
from g_export.export1d import export_datas
from g_export.export_dict2excel import dictexport

All = globalvars.AllParams()


def execution_stream(input_folder):
    start_time = time.time()
    globalvars.start_time = start_time

    file_list = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]
    print(f"获取文件名列表...耗时：{time.time() - start_time}秒。")

    #list_num_waves = np.round(np.logspace(1, 4, 25, base=10)).astype(int)

    for file_path in file_list:
        #for num_waves in list_num_waves:
            #All.num_waves = num_waves
            # Get the current date and time
            now = datetime.now()
            # Format the date and time
            date_time = now.strftime("%Y%m%d%H%M%S")
            # Create a new folder based on date_time in output_folder
            globalvars.output_path = os.path.join(globalvars.output_folder, date_time)

            raw_q_iq = import_data_as_dict(file_path)
            print(f"获取原始数据...耗时：{time.time() - start_time}秒。")

            sub_q_iq = subbg.sub_incoh_bg(raw_q_iq)
            print(f"扣除背景...耗时：{time.time() - start_time}秒。")

            cor_q_iq = porod.cal_porod_paras(sub_q_iq)
            print(f"porod正偏离校正...耗时：{time.time() - start_time}秒。")

            porod_q_iq = porod.porod_fit_extra(cor_q_iq)
            print(f"porod拟合...耗时：{time.time() - start_time}秒。")

            guinier_q_iq = guinier.guinier_fit_extra(cor_q_iq)
            print(f"Guinier近似...耗时：{time.time() - start_time}秒。")

            extrap_q_iq = extrapolate.merg_dict(guinier_q_iq, cor_q_iq, porod_q_iq)
            print(f"外推...耗时：{time.time() - start_time}秒。")

            phi_1 = invariant.integral(extrap_q_iq)
            print(f"计算不变量...耗时：{time.time() - start_time}秒。")
            #phi_1 = 0.3

            r_gammar = calcgammar(All.rpts, All.rmax, All.rmin, extrap_q_iq, len(extrap_q_iq['Q']), phi_1)
            print(f"计算r_gammar...耗时：{time.time() - start_time}秒。")

            r_gr, alpha = calcgr(All.rpts, r_gammar, phi_1)
            print(f"计算r_gr...耗时：{time.time() - start_time}秒。")
            print(alpha)

            k_fk = calc_spectral_function(All.kpts, All.kmax, All.kmin, All.rpts, r_gr)
            print(f"计算k_fk耗时：{time.time() - start_time}秒。")

            #generate_matrix(k_fk, All.boxres, All.boxsize, alpha, All.num_waves, All.kconst)
            print(f"计算高斯场...耗时：{time.time() - start_time}秒。")

            dictexport(globalvars.paras_dict)


if __name__ == "__main__":
    input_folder = 'InputFiles'
    globalvars.output_folder = 'OutputFiles'
    execution_stream(input_folder)
