import numpy as np
import globalvars
AP = globalvars.AllParams()

def integral(extrap_q_iq):
    q_q2iq = {'Q': extrap_q_iq['Q'], 'Q2IQ': (extrap_q_iq['Q']**2 * extrap_q_iq['ExtraIQ'])}
    x = q_q2iq['Q']
    y = q_q2iq['Q2IQ']
    invariant_q_iq = np.trapz(y, x)
    invariant_a = 1.e-8 * invariant_q_iq / (2 * ((np.pi * AP.density_p) ** 2))
    phi_1 = (1 - (1 - 4 * invariant_a) ** 0.5) / 2
    phi_2 = (1 + (1 - 4 * invariant_a) ** 0.5) / 2

    # invariant参数
    globalvars.paras_dict['Invariant(A)'] = invariant_a
    globalvars.paras_dict['Invariant(phi_1)'] = phi_1
    globalvars.paras_dict['Invariant(phi_2)'] = phi_2

    return phi_1