import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as sp

import magdyn


def itmag(te, tpo, mloc, mp, param, i):
    sam = param['sam'][i]
    mzhere=mloc[str(i)]
    mphere=mp[str(i)]

    hmfaz = sam.Jpd*mzhere  # exchange coupling is only dependent on (on site) localized magnetization!

    eta = hmfaz/sp.k/te                     # ratio of exchange energy and thermal energy of electron bath
    mmag = magdyn.brillouin(eta,sam.spin)   # mean field mean magnetization at te
    #mmag = np.tanh(sam.tc/te*mzhere)       # for S=1/2 the Brillouin function has an easy analytical expression, this can also be called
    gamma = tpo/sam.tc*sam.R*sam.gepfit(te) # prefactor in the M3TM magnetization dynamics. tpo is the temperature of optical phonons
    rate = (1-mphere/mmag)*gamma            # rate of M3TM magnetization dynamics
    dmp=rate*mphere*param['dt']             # alltogether, this is the M3TM mag dynamics. Note that it scales linear in the itinerant magnetization mphere
    return(dmp)                             # return the magnetization increment


def locmag(mz, tpa, mp, fs, sup, sdn, param, i):
    sam= param['sam'][i]

    mzhere=mz[str(i)]
    mznext=np.roll(mzhere, -1)
    mzlast=np.roll(mzhere, 1)

    mznext[-1]=0 if i==len(param['sam'])-1 else mz[str(i+1)][0]
    mzlast[0]=0 if i==0 else mz[str(i-1)][-1]

    hmfaz=np.array(param['Jlochere'][i])*mzhere+np.array(param['Jlocnext'][i]*mznext)+np.array(param['Jloclast'][i])*mzlast+param['hex']

    mmag=magdyn.brillouin(hmfaz/sp.k/tpa, sam.locspin)

    dmloc=1/sam.tmp*(mmag-mzhere)*param['dt']
    return(dmloc)