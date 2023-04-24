import numpy as np
from scipy import constants as sp
import itertools

def pump(t, param):
    return np.array(param['pp'][i])*np.exp(-((t*param['dt']-param['pdel'])**2)/2/param['psig']**2)

def tempdyn(te, tp, freqs, param, coeffs,t, dqes_long):
    Ce=np.array(itertools.chain.from_iterable(list([param['ce_arr'][i](te[coeffs[i][0]:coeffs[i][1]]) for i in range(len(param['sam']))])))
    Cp=np.array(itertools.chain.from_iterable(list([param['cp_arr'][i](tp[coeffs[i][0]:coeffs[i][1]]) for i in range(len(param['sam']))])))
    gep=np.array(itertools.chain.from_iterable(list([param['gep_arr'][i](te[coeffs[i][0]:coeffs[i][1]]) for i in range(len(param['sam']))])))
    ke=param['kappae_arr']*np.divide(te, tp)
    kp=param['kappap_arr']
    pulse=pump(t, param)

    te_hat=np.fft.fft(te)
    tp_hat=np.fft.fft(tp)
    ke_hat=np.fft.fft(ke)

    term_e_pulse=np.divide(pulse, Ce)
    term_e_gep=np.multiply(np.divide(gep,Ce),(te-tp))
    term_e_dqes=np.divide(dqes_long, Ce)

    term_e_pulse_hat=np.fft.fft(term_e_pulse)
    term_e_gep_hat=np.fft.fft(term_e_gep)
    term_e_diff_hat=-np.power(freqs,2)*np.convolve(te_hat, ke_hat, mode='same')

    term_p_gep_hat=-term_e_gep_hat


