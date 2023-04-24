### In this file the magnetization dynamics in the M3TM for arbitrary spin are calculated. For detailes see
#Beens et al., PRB 100, 220409 (2019)

import numpy as np
from scipy import constants as sp


def magdyn_s(ss, mz, fs, sup, sdn, te, tp, param, i):
    sam=param['sam'][i]
    mzhere=mz[str(i)]
    mzlast=np.roll(mzhere,1)
    mzlast[0]=0 if i==0 else mz[str(i-1)][-1]*param['sam'][i+1].muat/sam.muat
    mznext=np.roll(mzhere, -1)
    mznext[-1]=0 if i==len(param['sam'])-1 else mz[str(i+1)][0]*sam.muat/param['sam'][i+1].muat

    hmfaz = np.array(param['Jhere'][i])*mzhere+np.array(param['Jnext'][i]*mznext)+np.array(param['Jlast'][i])*mzlast
    const_ijk=sam.arbsc*tp*abs(mzhere)/4/sam.spin/np.sinh(hmfaz/2/sam.spin/sp.k/te)*sam.gepfit(te)*hmfaz
    fsup=sup*fs
    fsdn=sdn*fs
    wupwegprep=const_ijk*np.exp(-hmfaz/2/sam.spin/sp.k/te)
    wdnwegprep=const_ijk*np.exp(hmfaz/2/sam.spin/sp.k/te)
    wupweg=wupwegprep[...,np.newaxis]*fsup
    wdnweg=wdnwegprep[...,np.newaxis]*fsdn
    wuphin=np.roll(wupweg,1)
    wdnhin=np.roll(wdnweg,-1)
    dfs=(-wupweg-wdnweg+wuphin+wdnhin)*param['dt']
    return(dfs)
