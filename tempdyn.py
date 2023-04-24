#### In this file the two temperature model is calculated. If param['qes']==True, the energetic coupling of electron and spin systems is included

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as sp


def tempdyn(ts, tel, tph, tpha, dqes, param, i):
    sam=param['sam'][i]
    te=tel[str(i)]
    tp=tph[str(i)]

    cv_el=sam.celfit(te)
    cv_ph=sam.cphfit(tp)

    ############add magnon specific heat if T<Tc#############
    #cv_mag=np.zeros(param['nj'])
    #masktc = tp<param['tc']
    #cv_mag[masktc] = np.sum(param['mp']*(tp[masktc,np.newaxis]**np.arange(param['mp'].size)),axis=-1)
    #cv_ph+=cv_mag

    ###########compute pump power density:##########

    #compute power density of pump pulse:
    dpump=np.array(param['pp'][i])*np.exp(-((ts*param['dt']-param['pdel'])**2)/2/param['psig']**2)

    #term computing energy exchange of electron and lattice systems:
    teq=sam.gepfit(te)*(te-tp)

    #add pump energy to electron system:
    dtel=dpump-teq
    dtph=(teq)

    #add energy coupling of spin- and electron-systems:
    dtel+=+param['qes']*dqes[str(i)]

    #add electron temperature diffusion:
    dtel=dtel+tediffusion(tel, tph, param, i)
    dtph=dtph+tpdiffusion(tph, param, i)

    #add electron temperature diffusion according to the code of Andreas Donges' LLG implementation (only for crosschecking):
    #dtel=dtel+diffdonges(tel, tph, param, i)

    #compute the temperature change of electron and lattice systems:
    dte=param['dt']*np.true_divide(dtel,cv_el)
    dtp=param['dt']*np.true_divide(dtph,cv_ph)
    dtpa=np.array([0])

    #return the temperature changes to add in the output file:
    return(dte, dtp, dtpa)
        
    

#The following fucntions compute the Debye integral for lattice specific heat. This is not used in the code but can be called if desired:

def debint(b):
    grains=2**8
    delta=np.true_divide(b,grains)
    x=delta
    sum=np.zeros(param['nj'],dtype=np.float64)
    for gr in range(1,grains):
        sum=sum+np.multiply(integrand(x),delta)
        x+=delta
    cp=0.5*3*2.55e6*((b)**(-3))+sum
    return(cp)

def integrand(y):
    num=np.power(y,4)*np.exp(y)
    rdenom=np.exp(y)-1
    denom=rdenom**2
    quot=np.true_divide(num,denom)  
    return(quot)
    
    
def tediffusion(tel, tph, param, i):

    ###This function computes the diffusion of electron temperature with kappa=kappa_0*T_e/T_p###
    
    keln=param['keln'][i]
    kell=param['kell'][i]

    te=tel[str(i)]
    tp=tph[str(i)]

    telast=np.roll(te,1)
    tenext=np.roll(te,-1)
    tplast=np.roll(tp,1)
    tpnext=np.roll(tp,-1)

    telast[0]=0 if i==0 else tel[str(i-1)][-1]
    tenext[-1]=0 if i==len(param['sam'])-1 else tel[str(i+1)][0]
    tplast[0]=1 if i==0 else tph[str(i-1)][-1]
    tpnext[-1]=1 if i==len(param['sam'])-1 else tph[str(i+1)][0]
        
    te_grad_next=tenext-te
    te_grad_last=telast-te
    
    if i==len(param['sam'])-1:
        te_grad_next[-1]+=te[-1] 
    elif i==0:
        te_grad_last[0]+=te[0]
    
    diff=te_grad_next*keln*te/tp+te_grad_last*kell*te/tp
    
    gradkappa=0.5*kell[1:-1]*(tenext[1:-1]/tpnext[1:-1]-telast[1:-1]/tplast[1:-1])
    
    diff[1:-1]+=gradkappa
    
    return diff
    

def tpdiffusion(tph, param, i):
    kphn=param['kphn'][i]
    kphl=param['kphl'][i]
    
    tp=tph[str(i)]
    
    tplast=np.roll(tp,1)
    tpnext=np.roll(tp,-1)
    
    tplast[0]=0 if i==0 else tph[str(i-1)][-1]
    tpnext[-1]=0 if i==len(param['sam'])-1 else tph[str(i+1)][0]
    
    tp_grad_next=tpnext-tp
    tp_grad_last=tplast-tp
    
    if i==len(param['sam'])-1:
        tp_grad_next[-1]+=tp[-1] 
    elif i==0:
        tp_grad_last[0]+=tp[0] 
    
    diff=tp_grad_next*kphn+tp_grad_last*kphl
    
    return diff
    

def optaccdyn(ts, tel, tpo, tpa, dqes, dqps, param, i):
    sam=param['sam'][i]
    te=tel[str(i)]
    topt=tpo[str(i)]
    tacc=tpa[str(i)]

    cv_el=sam.celfit(te)
    cv_po=sam.cphfit(topt)
    cv_pa=sam.cp2(tacc)

    #compute power density of pump pulse:
    dpump=np.array(param['pp'][i])*np.exp(-((ts*param['dt']-param['pdel'])**2)/2/param['psig']**2)

    #term computing energy exchange of electron and lattice systems:
    teq=sam.gepfit(te)*(te-topt)
    teqp=sam.gpp(topt)*(topt-tacc)

    #add pump energy to electron system:
    dtel=dpump-teq
    dtpho=(teq-teqp)+param['qpos']*dqps[str(i)]
    dtpha=teqp+param['qpas']*dqps[str(i)]

    #add energy coupling of spin- and electron-systems:
    dtel+=+param['qes']*dqes[str(i)]

    #add electron temperature diffusion:
    dtel=dtel+tediffusion(tel, tpo, param, i)

    #add electron temperature diffusion according to the code of Andreas Donges' LLG implementation (only for crosschecking):
    #dtel=dtel+diffdonges(tel, tph, param, i)

    #compute the temperature change of electron and lattice systems:
    dte=param['dt']*np.true_divide(dtel,cv_el)
    dtpo=param['dt']*np.true_divide(dtpho,cv_po)
    dtpa=param['dt']*np.true_divide(dtpha, cv_pa)

    #return the temperature changes to add in the output file:
    return(dte, dtpo, dtpa)