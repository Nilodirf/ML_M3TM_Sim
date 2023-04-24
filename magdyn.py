### In this file the magnetization dynamicd of the M3TM are calculated. For details see
# Koopmans et al., Nat. Mat. 9 (2010)

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import constants as sp



#this file computes the magnetization daynamics for each time step

def magdyn(ss, mz, te, tp, param, i):

    #compute M3TM magnetization dynamics for z component of magnetization
    sam=param['sam'][i]
    mzhere=mz[str(i)]
    mzlast=np.roll(mzhere,1)
    mzlast[0]=0 if i==0 else mz[str(i-1)][-1]
    mznext=np.roll(mzhere, -1)
    mznext[-1]=0 if i==len(param['sam'])-1 else mz[str(i+1)][0]

    hmfaz = param['Jhere'][i]*mzhere+param['Jnext'][i]*mznext+param['Jlast'][i]*mzlast+param['hex']

    eta = hmfaz/sp.k/te
    mmag = brillouin(eta,sam.spin)
    #mmag = np.tanh(sam.tc/te*mzhere)
    gamma = tp/sam.tc*sam.R*sam.gepfit(te)
    rate = (1-mzhere/mmag)*gamma
    dmz=rate*mzhere*param['dt']

    return(dmz)




def brillouin(x,spin):
    #Compute equilibrium magnetization via Brillouin function
    c1=(2*spin+1)/(2*spin)
    c2=1/(2*spin)
    fb=c1/np.tanh(c1*x)-c2/np.tanh(c2*x)
    return(fb)

def dbrillouin(x,spin):
    # Derivative of the Brollouin function, not used in code here
    c1=(2*spin+1)/(2*spin)
    c2=1/(2*spin)
    dfb=c2**2/(np.sinh(c2*x))**2-c1**2/(np.sinh(c1*x))**2
    return(dfb)