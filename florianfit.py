### In this file, the parameters for g_ep, c_e and c_p are defined

from scipy import interpolate as ipl
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy import constants as sp
import math
import os
import sys

def interpol(file, func, tein):
    if type(file)== str :
        #Files with ab initio data are called here
        dat=open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '3TM_Data/Ab-initio-parameters2/' + str(file)),'r')
        lines=dat.readlines()[2:]
        #vals=[line for line in lines if not line.startswith('M') or line.startswith('T')]
        t=np.array([float(i.split()[0]) for i in lines])
        y=np.array([float(line.split()[1]) for line in lines])
        if np.amin(t)>290:
            t[0]=290.
        fit=ipl.interp1d(t,y)
        return(fit)
    elif isinstance(file, float):
        # If no ab initio parameters are available, constant g_ep, linear c_e and Einstein-model c_p can be created
        if func=='const':
            #use for gep
            t=np.arange(0,10000, 250)
            y=np.array([file for i in t])
            fit = ipl.interp1d(t, y)
        elif func=='lin':
            #use for c_e
            t=np.arange(0,10000,10)
            y=np.array([file*float(i) for i in t])
            fit= ipl.interp1d(t,y)
        elif func=='einstein':
            #use for c_p
            t=np.arange(1,2500, 0.001)
            y=file*(tein/t)**2*np.exp(tein/t)/(np.exp(tein/t)-1)**2
            fit=ipl.interp1d(t,y)
    return(fit)


def fitcp(T, Tein):
    return(3*sp.k*Tein**2/T**2/3.524/3.524/3.524*1e30*np.exp(Tein/T)/np.power((np.exp(Tein/T)-1),2)*4)

def fitce(T, gamma):
    return(gamma*T)

def fitcplow(T,Tdeb):
    return(12/5*np.power(math.pi,4)*np.power(T/Tdeb,3)*sp.k/np.power(3.524e-10,3)*4)


###Irrelevant:
def plot():

    fig=plt.figure(figsize=(4,2.5))

    ax1=fig.add_axes([0.1, 0.1, 0.85, 0.85])
    #ax2=fig.add_axes([0.4, 0.63, 0.3 , 0.3])

    temp=np.arange(14,2500,0.1)
    temp2=np.arange(0,2500,0.1)
    ax1.plot(temp, 1e-18*interpol('Ab-initio-Fe/Fe_g_ep.txt')(temp), color=(0.8,0,0), linewidth=2.0)
    ax1.plot(temp, 1e-18*interpol('Ab-initio-Co/Co_g_ep.txt')(temp), color=(0,0.4,0.6), linewidth=2.0)
    ax1.plot(temp, 1e-18*interpol('Ab-initio-Ni/Ni_g_ep.txt')(temp), color=(0.6, 0.8, 0), linewidth=2.0)
    ax1.plot(temp2, [2.025 for t in temp2], color=(0.6, 0.8, 0), linewidth=2.0, linestyle='dashed')
    ax1.plot(temp2, [2.025 for t in temp2], color=(0.6, 0.8, 0), linewidth=2.0, linestyle='dashed')
    ###plt.plot(temp, [4.05e18 for i in temp], color=(0.6, 0.8, 0), linestyle='dashed')
    ###plt.plot(temp, [4.05e18 for i in temp], color=(0,0.4,0.6), linestyle='dashed')


    ###popt, pcov= curve_fit(fitcplow, temp, interpol('Ab-initio-Ni/Ni_c_p.txt')(temp))
    ###popt, pcov= curve_fit(fitcp, temp, interpol('Ab-initio-Co/Co_c_e.txt')(temp), color=(0,51, 102))
    ###popt, pcov= curve_fit(fitcp, temp, interpol('Ab-initio-Ni/Ni_c_e.txt')(temp), color=(153, 204, 0))
    ###plt.plot(temp, fitcp(temp,*popt), 'r--')
    #plt.plot(temp, [4.05e18 for t in temp], color=(0.6, 0.8, 0), linestyle='dashed')
    #plt.plot(temp, [4.05e18 for t in temp], color=(0,0.4,0.6), linestyle='dashed')
    ##print(popt)


    ax1.set_xlabel(r'$T_e$ [K]', fontsize=22)
    ax1.set_ylabel(r'$g_{ep} [10^{18}$W/($\rm{m}^3$K)]', fontsize=22)
    ax1.set_title('Electron phonon coupling', fontsize=25)
    ax1.legend([r'Iron', r'Cobalt', r'Nickel', r'Ni/Co Koopmans*0.5'], fontsize=18, loc='lower right')
    ax1.set_xlim(0,2500)
    ax1.tick_params(axis='both', which='major', labelsize=16)


    ##ce plot###
    ax2.set_ylabel(r'$\gamma$ [J/$\rm{m}^3\rm{K}^2$]', fontsize=18)
    ax2.set_xlabel(r'$T_e$ [K]', fontsize=18)
    ax2.plot(temp2, [5435 for t in temp2], color=(0.6, 0.8, 0), linewidth=2.0, linestyle='dashed')
    ax2.plot(temp2, [5533 for t in temp2], color=(0, 0.4, 0.6), linewidth=2.0, linestyle='dashed')
    ax2.set_ylim((5300,5700))
    ax2.set_yticks([5434,5533])
    ax2.set_xticks([0,2500])
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlim((0,2500))
    ax1.hlines(0.75,136, 300, linewidth=1.5, color='black')
    ax1.annotate(r'$c_e(T_e)=\gamma T_e$',(310, 0.725), fontsize=18)
    ax1.set_ylim((-0.1,2))

    ##cp plot###
    ax1.vlines(460, -0.4, 3.62, linewidth=1.5, colors=(0,0.4,0.6))
    ax1.vlines(477, -0.4, 3.3835, linewidth=1.5, colors=(0.8,0,0))
    ax1.vlines(477, 3.3835, 3.6718, linewidth=1.5, colors=(0.6, 0.8,0))
    ax1.hlines(1.25, 477, 577, linewidth=1.0, colors='black')
    ax1.hlines(1.25, 360, 460, linewidth=1.0, colors='black')
    ax1.annotate(r'$T_D$(Fe, Ni)=$477$ K', (585, 1.2), fontsize=18)
    ax1.annotate(r'$T_D$(Co)=$460$ K', (130, 1.2), fontsize=18)

    ax1.set_ylim((-0.05, 4))
    ax1.set_xlim((0,1500))

    plt.show()