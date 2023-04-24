#### In this file the magnetization dynamics od the s-d-model is calculated. For details see
#Beens et al., PRB 102, 054442 (2020)

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as sp


def itmag(dmloc, mus, param, i):
    #This function computes the dynamics of spin polarization in the itinerant electron system
    sam = param['sam'][i]
    dz = sam.dz
    sam=param['sam'][i]
    dz=sam.dz
    mushere=mus[str(i)]
    muslast = np.roll(mushere, 1)
    musnext = np.roll(mushere, -1)
    muslast[0] = 0 if i == 0 else mus[str(i - 1)][-1]
    musnext[-1] = 0 if i == len(param['sam']) - 1 else mus[str(i + 1)][0]
    laplace = (musnext + muslast - 2 * mushere)

    #compute diffusion of spin polarization
    if i == 0:
        #first layer
        #no spin diffusion at outer boundary:
        laplace[0]+=mushere[0]
        #bulk diffusion:
        musdiff=list([l*sam.musdiff/dz**2 for l in laplace[:-1]])
        #diffusion at inner boundary:
        musdiff+=list([(musnext[-1]-mushere[-1])*param['musint'][0]/2/dz**2+(muslast[-1]-mushere[-1])*sam.musdiff/dz**2])
        #gradient of diffusion coefficient at inner boundary:
        #musdiff[-1]+=(param['musint'][0]-sam.musdiff)*(musnext[-1]-mushere[-1])/2/dz**2
    elif i==len(param['sam'])-1 and len(param['sam'])>1:
        #last layer
        #no spin diffusion at outer boundary:
        laplace[-1]+=mushere[-1]
        #diffusion at inner boundary:
        musdiff=list([(muslast[0]-mushere[0])*param['musint'][-1]/2/dz/param['sam'][i-1].dz+(musnext[0]-mushere[0])*sam.musdiff/dz**2])
        #bulk diffusion:
        musdiff+=list([l*sam.musdiff/dz**2 for l in laplace[1:]])
        #gradient of diffusion at inner boundary:
        #musdiff[0]+=(param['musint'][-1]-sam.musdiff)*(muslast[0]-mushere[0])/2/param['sam'][i-1].dz**2
    else:
        #in-between layers
        #diffusion at the boundary to previous layer:
        musdiff=list([(muslast[0]-mushere[0])*param['musint'][i-1]/2/dz/param['sam'][i-1].dz+(musnext[0]-mushere[0])*sam.musdiff/dz**2])
        #bulk diffusion:
        musdiff += list([l*sam.musdiff/dz**2 for l in laplace[1:]])
        #diffusion at boundary to next layer
        musdiff+=list([(musnext[-1]-mushere[-1])*param['musint'][i]/2/dz**2+(muslast[-1]-mushere[-1])*sam.musdiff/dz**2])
        #gradient of diffusion constant at boundary to previous layer:
        #musdiff[0]+=(param['musint'][i-1]-sam.musdiff)*(muslast[0] - mushere[0])/2/param['sam'][i-1].dz**2
        #gradient of diffusion constant at boundary to next layer:
        #musdiff[-1]+=(param['musint'][i]-sam.musdiff)*(musnext[-1]-mushere[-1])/2/dz**2

    dmus=sam.rhosd*dmloc-(mushere/sam.tsl+musdiff)*param['dt']
    return(dmus)

def locmag(ss, mz, te, mus, fs, sup, sdn, param, i):
    # This function computes the magnetization of the localized electron system
    sam = param['sam'][i]
    mzhere = mz[str(i)]
    mzlast = np.roll(mzhere, 1)
    mzlast[0] = 0 if i == 0 else mz[str(i - 1)][-1] * param['sam'][i - 1].muat / sam.locmom
    mznext = np.roll(mzhere, -1)
    mznext[-1] = 0 if i == len(param['sam']) - 1 else mz[str(i + 1)][0] * param['sam'][i + 1].muat / sam.locmom

    hmfaz = param['Jlochere'][i] * mzhere + param['Jlocnext'][i] * mznext + param['Jloclast'][i] * mzlast + param['hex']

    const_ijk=1/sam.tsd*(mzhere-mus[str(i)]/param['Jlochere'][i])/np.sinh((hmfaz-mus[str(i)])/2/sam.locspin/sp.k/te)
    fsup = sup * fs
    fsdn = sdn * fs
    wupwegprep = const_ijk * np.exp(-(hmfaz - mus[str(i)]) / 2 / sam.locspin / sp.k / te)
    wdnwegprep = const_ijk * np.exp((hmfaz - mus[str(i)]) / 2 / sam.locspin / sp.k / te)
    wupweg = wupwegprep[..., np.newaxis] * fsup
    wdnweg = wdnwegprep[..., np.newaxis] * fsdn
    wuphin = np.roll(wupweg, 1)
    wdnhin = np.roll(wdnweg, -1)
    dfs = (-wupweg - wdnweg + wuphin + wdnhin) * param['dt']
    return (dfs)