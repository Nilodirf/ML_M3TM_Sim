#In this file data is written on the file defined at the top of output.output()

import numpy as np

def iniwrite(file, date, pl):
    ### Initial documentation of input parameters:
    file.write('# time of execution: ' + str(date) + '\n')
    file.write('### simulation parameters ###' + '\n')
    file.write('# timestep dt=' + str(pl['dt']) + '[s]' + '\n')
    file.write('# initial temperature:' + str(pl['initemp']) + '[K]' + '\n')
    file.write('# initial magnetization [x,y,z]:' + str([str(i.inimag) for i in pl['sam']]) + '\n')
    file.write('### sample paramters ###' + '\n')
    file.write('# Sample: ' + str([i.name for i in pl['sam']]) + '\n')
    file.write('# S=' + str([i.spin for i in pl['sam']]) + '\n')
    file.write('# mu_at=' + str([i.muat for i in pl['sam']]) + '[mu_B]' + '\n')
    file.write('# nz=' + str(pl['nj']) + '\n')
    file.write('# dx=' + str([i.dx for i in pl['sam']]) + '\t' + 'dy=' + str([i.dy for i in pl['sam']]) + '\t' + 'dz=' + str([i.dz for i in pl['sam']]) + '\t' + '[m]' '\n')
    file.write('# atoms per unit cell:' + str([i.apc for i in pl['sam']]) + '\n')
    file.write('# asf=' + str([i.asf for i in pl['sam']]) + '\n')
    file.write('# Jint=' + str(pl['jint']) + '[J]' + '\n')
    file.write('# kappa_el_int=' + str(pl['kint']) + '[W/mK]' + '\n')
    file.write('# kappa_ph_int' + str(pl['kphint']) + '[W/mK]' + '\n' )
    file.write('# Hex=' + str(pl['hex']) + '[J]' +'\n')
    file.write('# gep=' + str([str(i.gepfit(pl['initemp'])) for i in pl['sam']]) + '[W/m^3/K]' + '\n')
    file.write('# factor cv_el=' + str([str(i.celfit(100.)/100.) for i in pl['sam']]) + '[J/m^3/K^2]' + '\n')
    file.write('# cv_ph_max=' + str([str(i.cphfit(2400.)) for i in pl['sam']]) + '[J/m^3/K]' + '\n')
    file.write('# kappa_el='  + str([str(i.kappa) for i in pl['sam']]) + '[W/mK]' + '\n')
    file.write('# kappa_ph='  + str([str(i.kappaph) for i in pl['sam']]) + '[W/mK]' + '\n')
    file.write('# Tc=' + str([i.tc for i in pl['sam']]) + '[K]' + '\n')
    file.write('# T_Deb=' + str([i.tdeb for i in pl['sam']]) + '[K]' + '\n')
    file.write('# kappa=' + str([i.kappa for i in pl['sam']]) + '[W/m/K]' + '\n')
    file.write('# D=' + str([i.musdiff for i in pl['sam']]) + '[W/m^2]' + '\n')
    file.write('### pulse paramters ###' + '\n')
    file.write('# peak power:' + str(pl['pp']) + '[W/m^3]' + '\n')
    file.write('# fluence input:' + str(pl['power_input']) + '[W/m^2]' + '\n')
    file.write('# absolute absorbed fluence: ' + str(0.1*pl['abs_flu']) + '[mJ/cm^2]' + '\n')
    file.write('# sigma=' + str(pl['psig']) + '[s]' + '\n')
    file.write('# delay=' + str(pl['pdel']) + '[s]' + '\n')
    file.write('#############################' + '\n')
    file.write('# time [ps]' + '\t'+'magnetization' + '\t'+'T_e [K]' + '\t'+'T_p [K]' + '\t' + 'T_p(acc) [K]' + '\t' + 'itinerant mag' + '\n')


def dwrite(file, t, mzs, tes, tps, tpas, mus, samples, pl):
    #Documenting the dynamical process:
    file.write(str(round(t*pl['dt']*1e12,2)) + '\t' + str([list(mzs[str(i)]) for i in range(len(samples))]) + '\t' + str([list(tes[str(i)]) for i in range(len(samples))])
               + '\t' + str([list(tps[str(i)]) for i in range(len(samples))]) + '\t' + str([list(tpas[str(i)]) for i in range(len(samples))]) + '\t' + str([list(mus[str(i)]) for i in range(len(samples))]) + '\n')



