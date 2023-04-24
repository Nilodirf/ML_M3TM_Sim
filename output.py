### This file defines the initial parameters in a usable form and calls the function(s) to compute the dynamics

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as sp
import os
import sys
from datetime import datetime
from npy_append_array import NpyAppendArray
import copy
import diffusion
from scipy import fftpack

#import other files
import writer
date=datetime.now().replace(microsecond=0)

#This is the main function that runs for the length of simlen. pl denotes the ParameterList defined in readinput.readinput()
def output(pl):
    #create file and call function to document input data (file=open(filepath, 'w+'))
    
    simpath=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '3TM_results/CGT/'+ str(pl['filename']))

    if not os.path.exists(simpath):
        makefolder=os.makedirs(simpath)
    for file in os.listdir(simpath):
        if str(file).endswith('.npy'):
            os.remove(os.path.join(simpath, file))
    
    paramfile=open(os.path.join(simpath, 'params.dat'), 'w+')
    tefile=NpyAppendArray(simpath+'/tes.npy')
    tpfile=NpyAppendArray(simpath+'/tps.npy')
    tp2file=NpyAppendArray(simpath+'/tp2s.npy')
    mfile=NpyAppendArray(simpath+'/ms.npy')
    delayfile=NpyAppendArray(simpath+'/delays.npy')
    datmagfile= open(os.path.join(simpath, 'magdat.dat'), 'w+')
    dattefile= open(os.path.join(simpath, 'tedat.dat'), 'w+')
    dattpfile= open(os.path.join(simpath, 'tpdat.dat'), 'w+')
    dattp2file= open(os.path.join(simpath, 'tp2dat.dat'), 'w+')
    
    writer.iniwrite(paramfile, date, pl)
    paramfile.close()

    ###initialize dictionaries to hold dynamical parameters for any model, constituent i is then later accessed by dict[str(i)]###
    tefiles={}
    tpfiles={}
    tp2files={}
    mfiles={}
    
    mzs={} # magnetization of the localized spin system
    fss={} # for arbspin and s-d-model: Occupation of spin-z levels. If other model is chosen, this will automatically always be [0]
    muss={} # magnetic information of itinerant spin system. For s-d model this is spin accumulation, for m5tm, this is the magnetization of S=1/2 itinerant system. For other models this is always [0]

    tes={} # electron temperatures
    tps={} # lattice temperatures (for m5tm this is optical phonon temperature)
    tpas={} # accoustic phonon temperatures, for all models but m5tm this will always be [0]

    dqes={} # increments of mean field magnetic energy of spin system coupled to electron bath
    dqps={} # increments of mean field magnetic energy of spin system coupled to accoustic phonon bath in m5tm
    dmagz={} # increments of localized magnetization change
    dfss={} # increments of change of occupation of spin-z levels. If not arbspin-/ or s-d- model is chosen, this will automatically always be [0]
    dmuss={} #increments of itinerant spin system. For s-d model this is spin accumulation, for m5tm, this is the magnetization of S=1/2 itinerant system. For other models this is always [0]


    ## Because different sample consituents may have different models to compute the dynamics, we call every consituent in pl['sam'] individually:
    for i, sam in enumerate(pl['sam']):
        ### Initiate the dictionaries that carry numpy arrays with relevant dynamical data for each time step. For each constituent i this creates something like dict['i'] starting with i=0
        mzs['{0}'.format(i)], fss['{0}'.format(i)], muss['{0}'.format(i)] = sam.initmag(pl['ss'][i])

        tes['{0}'.format(i)], tps['{0}'.format(i)], tpas['{0}'.format(i)] = sam.inittemp((pl['ss'][i]), pl['initemp'])

        dqes['{0}'.format(i)] = np.zeros(pl['ss'][i])
        dqps['{0}'.format(i)] = np.zeros(pl['ss'][i])

        dmagz['{0}'.format(i)], dfss['{0}'.format(i)], muss['{0}'.format(i)] = sam.initmag(pl['ss'][i])
        
        tefiles['{0}'.format(i)]=NpyAppendArray(simpath+'/tes' + str(sam.name) + '.npy')
        tpfiles['{0}'.format(i)]=NpyAppendArray(simpath+'/tps' + str(sam.name) + '.npy')
        tp2files['{0}'.format(i)]=NpyAppendArray(simpath+'/tp2s' + str(sam.name) + '.npy')
        mfiles['{0}'.format(i)]=NpyAppendArray(simpath+'/ms' + str(sam.name) + '.npy')

    te_long=np.array([pl['initemp'] for _ in range(sum([pl['ss'][i]*pl['sam'][i].dz*1e10 for i in range(len(pl['sam']))]))])
    tp_long=te_long.copy()

    freqs=2 * np.pi * fftpack.fftfreq(len(te_long), d=1e-10)

    ## Start the dynamical simulation:
    for t in range(pl['simlen']):
        writedelay=True
        for i, sam in enumerate(pl['sam']):
            if t%int(1e-14/pl['dt'])==0:
                if writedelay:
                    delayfile.append(np.array([t*pl['dt']]))
                    datmagfile.write(str(t*pl['dt']) + '\t')
                    dattefile.write(str(t*pl['dt']) + '\t')
                    dattpfile.write(str(t*pl['dt']) + '\t')
                    dattp2file.write(str(t*pl['dt']) + '\t')
                    writedelay=False
                #Call fucntion to write data on file every 10 fs simulation time
                tefiles[str(i)].append(np.array([tes[str(i)]]))
                tpfiles[str(i)].append(np.array([tps[str(i)]]))
                tp2files[str(i)].append(np.array([tpas[str(i)]]))
                mfiles[str(i)].append(np.array([mzs[str(i)]]))
                if i==0:
                    for j in range  (len(mzs[str(i)][:-1])):
                        datmagfile.write(str(mzs[str(i)][j]) + '\t')
                        dattefile.write(str(tes[str(i)][j]) + '\t')
                        dattpfile.write(str(tps[str(i)][j]) + '\t')
                        dattp2file.write(str(tpas[str(i)][j]) + '\t')
                    datmagfile.write(str(mzs[str(i)][-1]) + '\n')
                    dattefile.write(str(tes[str(i)][-1]) + '\n')
                    dattpfile.write(str(tps[str(i)][-1]) + '\n')
                    dattp2file.write(str(tpas[str(i)][-1]) + '\n')
            #Call functions to compute magnetization dynamics in Heun method. Becasue of Heun method in the dmag() functions, the increments are only added in the next loop.
            mznew=mzs
            dmz, df, dmu= sam.dmag(pl['ss'][i], tes[str(i)], tps[str(i)], tpas[str(i)], copy.deepcopy(mzs), copy.deepcopy(fss[str(i)]), copy.deepcopy(muss), pl, i)            
            dmagz[str(i)] = dmz
            dfss[str(i)] = df
            dmuss[str(i)] = dmu
            
        for i, sam in enumerate(pl['sam']):
            # add magnetization increments
            mznew = mzs[str(i)]+dmagz[str(i)]
            fsnew = fss[str(i)]+dfss[str(i)]
            musnew = muss[str(i)]+dmuss[str(i)]
            mzs[str(i)]=mznew
            fss[str(i)]=fsnew
            muss[str(i)]=musnew
            

            if t > pl['pdel'] / pl['dt'] - 1e4:
                # compute the mean field energy cost of spin-flip after pump pulse sets in (if not for the if-clause, electron temperature would rise in the initial equilibration process)
                dqes[str(i)] = sam.dqes(mzs[str(i)], muss[str(i)], dmagz[str(i)], dmuss[str(i)], pl)
                dqps[str(i)] = sam.dqps(mzs[str(i)], muss[str(i)], dmagz[str(i)], dmuss[str(i)], pl)

    newte, newtp=diffusion.tempdyn(te_long, tp_long, freqs, pl)

    datmagfile.close()
    return
