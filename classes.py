import numpy as np
import math
from scipy import interpolate as ipl
import os
from scipy import constants as sp

import florianfit
import magdyn
import magdyn_s
import sd_mag
import pd_mag
import tempdyn
import copy


class sample:
    ### The sample class defines all necessary parameters to compute temperature and magnetization dynamics
    def __init__(self, name, gepfit, celfit, cphfit, spin, tc, tdeb, muat, dx, dy, dz, apc, asf, inimag, kappa, kappaph, locmom, locspin, coupl, musdiff):
        self.name=name  #sample name just for documentation
        self.gepfit=gepfit #electron phonon coupling(function, use (value,'const.', None) as parameters for easy temperature independent implementation) [W/(m^3K)]
        self.celfit=celfit #electronic heat capacity (function, use (slope, 'lin', None) as parameters for approximate implementation) [J/(m^3K^2)]
        self.cphfit=cphfit #lattice heat capacity (function, use (value, 'const', None) as parameters for easy implementation, for Einstein model see florianfit.py) [J/(m^3K)]
        self.spin=spin #effective spin of itinerant electrons
        self.tc=tc  #critical temperature [K]
        self.tdeb=tdeb #debye temperature [K]
        self.muat=muat #atomic magnetic moment in units of mu_b [J/T]
        self.dx=dx #lattice constant transversal x [m]
        self.dy=dy #lattice constant transversal y [m]
        self.dz=dz #lattice constant in depth direction [m]
        self.apc=apc #atoms per unit cell
        self.asf=asf #spin flip probability
        self.inimag=inimag #initial magnetization [0,0,from -1 to 1]
        self.kappa=kappa #electronic heat diffusion constant [W/(mK)]
        self.kappaph=kappaph #phononic heat diffusion constant [W/(mK)]
        self.locmom=locmom #magnetic moment of localized spins in unit of mu_b [J/T]
        self.locspin=locspin #effective spin of localized spin system
        self.coupl=coupl #coupling to nearest neighbours in [last, this, next] layer
        self.musdiff=musdiff #spin diffusion constant [m^2/s]
        if self.spin>0:
            self.J=float(3*sp.k*self.tc*((self.spin-self.locspin)**2)/((self.spin-self.locspin)*((self.spin-self.locspin)+1)))/sum(self.coupl)
        else:
            self.J=0
        if self.locspin>0:
            self.Jloc = float(3 * sp.k * self.tc * (self.locspin ** 2) / (self.locspin * (self.locspin + 1)))/sum(self.coupl)
            self.ms = (np.arange(2 * self.locspin + 1) + np.array([-self.locspin for i in range(int(2 * self.locspin) + 1)]))
            self.sup = -np.power(self.ms, 2) - self.ms + self.locspin ** 2 + self.locspin
            self.sdn = -np.power(self.ms, 2) + self.ms + self.locspin ** 2 + self.locspin
            self.J=float(3*sp.k*self.tc*(self.locspin-self.spin) ** 2) / ((self.spin-self.locspin) * (self.spin-self.locspin+1))/sum(self.coupl)
        else:
            self.Jloc=0
            self.ms = (np.arange(2 * self.spin + 1) + np.array([-self.spin for i in range(int(2 * self.spin) + 1)]))
            self.sup = -np.power(self.ms, 2) - self.ms + self.spin ** 2 + self.spin
            self.sdn = -np.power(self.ms, 2) + self.ms + self.spin ** 2 + self.spin
        if self.spin>0:
            self.R =8*self.asf*self.dx*self.dy*self.dz/self.apc*self.tc**2/self.tdeb**2/sp.k/(self.muat-self.locmom)
            self.arbsc=self.R/self.tc**2/sp.k
        else:
            self.R=0
            self.arbsc=0

    def inittemp(self, z, initemp):
        ### This function initiates arrays of the initial temperatures for all layers. tpa corresponds to a second set for lattice temperatures, that are only
        # needed in pd_dynamics.
        te = np.array([initemp for _ in range(z)])
        tpo = np.array([initemp for _ in range(z)])
        tpa = np.array([0])
        return(te, tpo, tpa)


    def dtem(self,t, tes, tps, tpas, dqes, dqps, pl, i):
        dte, dtp, dtpa=tempdyn.tempdyn(t, tes, tps, tpas, dqes, pl, i)
        return(dte, dtp, dtpa)

    def dqes(self, mz, mus, dmz, dmus, pl):
        return dmz/pl['dt'] * self.J * mz / self.dx / self.dy / self.dz * self.apc

    def dqps(self, mz, mus, dmz, dmus, pl):
        return np.array([0 for _ in range(len(mz))])

### The following subclasses inherit all the parameters defined above and are distinguished by the model to compute magnetization dynamics and possibly some additional parameters. Each model is commented at the beginning.
#(All magnetization dynamics are computed with the Heun method):

class m3tm(sample):
    ### This class computes magnetization dynamics with the M3TM (see magdyn.py)
    def initmag(self,z):
        mz = np.array([self.inimag for i in range(z)])
        return mz, np.array([0]), np.array([0])

    def dmag(self, z, te, tp, tpa, mz, fs, mus, param,  i):
        dmz = magdyn.magdyn(z, mz, te, tp, param, i)
        mz[str(i)]+=dmz
        dmz2= magdyn.magdyn(z, mz, te, tp, param, i)
        dmagz=(dmz+dmz2)/2
        return dmagz, np.array([0]), np.array([0])

class arbspin(sample):
    ### This class computes magnetization dynamics with the M3TM with arbitrary spin (see magdyn_s.py)
    def initmag(self, z):
        fs0 = np.zeros(int(2 * self.spin) + 1)
        fs0[0] = 1
        fs = np.array([fs0 for i in range(z)])
        magz = -np.sum(self.ms * fs, axis=-1) / self.spin
        return magz, fs, np.array([0])

    def dmag(self, z, te, tp, tpa, mz, fs, mus, param, i):
        df=magdyn_s.magdyn_s(z, mz, fs, self.sup, self.sdn, te, tp, param, i)
        dmz = -np.sum(self.ms * df, axis=-1) / self.spin
        mz[str(i)]+=dmz
        fs+=df
        df2=magdyn_s.magdyn_s(z, mz, fs, self.sup, self.sdn, te, tp, param, i)
        dmz2= -np.sum(self.ms * df2, axis=-1)/self.spin
        dfs=(df+df2)/2
        dmagz=(dmz+dmz2)/2
        return dmagz, dfs, np.array([0])

class sd(sample):
    ### This class computes magnetization dynamics with an interaction of localized magnetic moments and an itinerant spin polarized electron system [s-d-model] (see sd_mag.py)
    #The initial conditions for fs (named fs0) are for Gd at 100K, change to [0,...,1] of length (2S+1) for m_0=1.
    def __init__(self, name, gepfit, celfit, cphfit, spin, tc, tdeb, muat, dx, dy, dz, apc, asf, inimag, kappa, kappaph, locmom, locspin, coupl, musdiff, rhosd, tsd, tsl):
        super().__init__(name, gepfit, celfit, cphfit, spin, tc, tdeb, muat, dx, dy, dz, apc, asf, inimag, kappa, kappaph, locmom, locspin, coupl, musdiff)
        self.rhosd=rhosd
        self.tsd=tsd
        self.tsl=tsl

    def dtem(self, t, tes, tps, tpas, dqes, dqps, param, i):
        dte, dtpo, dtpa= tempdyn.optaccdyn(t, tes, tps, tpas, dqes, param, i)
        return(dte, dtpo, dtpa)

    def inittemp(self, z, initemp):
        te=np.array([initemp for _ in range(z)])
        tpo=np.array([initemp for _ in range(z)])
        tpa=np.array([initemp for _ in range(z)])
        return(te, tpo, tpa)

    def initmag(self,z):
        fs0=np.array([8.60047597e-05, 3.92842882e-04, 1.79438360e-03, 8.19618386e-03, 3.74376081e-02, 1.71003302e-01, 7.81089676e-01])
        fs = np.array([fs0 for i in range(z)])
        fs[1]=np.array([3.58125532e-06, 2.83077694e-05, 2.23756682e-04, 1.76866824e-03, 1.39803081e-02, 1.10506317e-01, 8.73489061e-01])
        fs[-2]=fs[1]
        for i in range(2,len(fs)-2):
          fs[i]=np.array([3.05360135e-06, 2.48028016e-05, 2.01460143e-04, 1.63635504e-03, 1.32912534e-02, 1.07957878e-01, 8.76885198e-01])
        magz = -np.sum(self.ms * fs, axis=-1) / self.locspin
        mus = np.array([0. for i in range(z)])
        return magz, fs, mus

    def dmag(self, z, te, tp, tpa, magz, fs, mus, param, i):
        df=sd_mag.locmag(z, magz, te, mus, fs, self.sup, self.sdn, param, i)
        dmz = -np.sum(self.ms * df, axis=-1) / self.locspin
        dmu=sd_mag.itmag(dmz, mus, param, i)
        magz[str(i)]+=dmz
        fs+=df
        mus[str(i)]+=dmu
        df2=sd_mag.locmag(z, magz, te, mus, fs, self.sup, self.sdn, param, i)
        dmz2=-np.sum(self.ms * df2, axis=-1) / self.locspin
        dmu2=sd_mag.itmag(dmz2, mus, param, i)
        dfs=(df+df2)/2
        dmagz=(dmz+dmz2)/2
        dmus=(dmu+dmu2)/2
        return dmagz, dfs, dmus

class m4tm(sample):
    def __init__(self, name, gepfit, celfit, cphfit, spin, tc, tdeb, muat, dx, dy, dz, apc, asf, inimag, kappa, kappaph, locmom, locspin, coupl, musdiff, cp2, gpp):
        super().__init__(name, gepfit, celfit, cphfit, spin, tc, tdeb, muat, dx, dy, dz, apc, asf, inimag, kappa, kappaph, locmom, locspin, coupl, musdiff)
        self.cp2=cp2
        self.gpp=gpp

    def inittemp(self, z, initemp):
        te=np.array([initemp for _ in range(z)])
        tpo=np.array([initemp for _ in range(z)])
        tpa=np.array([initemp for _ in range(z)])
        return(te, tpo, tpa)

    def initmag(self, z):
        fs0 = np.zeros(int(2 * self.spin) + 1)
        fs0[0] = 1
        fs = np.array([fs0 for i in range(z)])
        magz = -np.sum(self.ms * fs, axis=-1) / self.spin
        return(magz, fs, np.array([0]))

    def dtem(self, t, tes, tpos, tpas, dqes, dqps, param, i):
        dte, dtpo, dtpa= tempdyn.optaccdyn(t, tes, tpos, tpas, dqes, dqps, param, i)
        return(dte, dtpo, dtpa)

    def dmag(self, z, te, tpo, tpa, mz, fs, mus, param, i):
        df=magdyn_s.magdyn_s(z, mz, fs, self.sup, self.sdn, te, tpo, param, i)
        dmz = -np.sum(self.ms * df, axis=-1) / self.spin
        mz[str(i)]+=dmz
        fs+=df
        df2=magdyn_s.magdyn_s(z, mz, fs, self.sup, self.sdn, te, tpo, param, i)
        dmz2= -np.sum(self.ms * df2, axis=-1)/self.spin
        dfs=(df+df2)/2
        dmagz=(dmz+dmz2)/2
        return dmagz, dfs, np.array([0])

        
class m5tm(sample):
    ###This class defines additional parameters and call additional functions needed to compute magnetization dynamics in the picture of localized magnetic moments
    # coupled to optical phonons (coupled to accoustic phonons) and itinerant spins connected to electron bath that undergo sf-scattering (M3TM)
    def __init__(self, name, gepfit, celfit, cphfit, spin, tc, tdeb, muat, dx, dy, dz, apc, asf, inimag, kappa, kappaph, locmom, locspin, coupl, musdiff, cp2, Jpd, tmp, gpp):
        super().__init__(name, gepfit, celfit, cphfit, spin, tc, tdeb, muat, dx, dy, dz, apc, asf, inimag, kappa, kappaph, locmom, locspin, coupl, musdiff)
        self.cp2=cp2
        self.Jpd=Jpd
        self.tmp=tmp
        self.gpp=gpp
        self.J=self.Jpd

    def inittemp(self, z, initemp):
        te=np.array([initemp for _ in range(z)])
        tpo=np.array([initemp for _ in range(z)])
        tpa=np.array([initemp for _ in range(z)])
        return(te, tpo, tpa)

    def initmag(self, z):
        magz = np.array([self.inimag for _ in range(z)])
        magp= np.array([self.inimag for _ in range(z)])
        return(magz, np.array([0]), magp)

    def dtem(self, t, tes, tpos, tpas, dqes, dqps, param, i):
        dte, dtpo, dtpa= tempdyn.optaccdyn(t, tes, tpos, tpas, dqes, dqps, param, i)
        return(dte, dtpo, dtpa)

    def dmag(self, z, te, tpo, tpa, magz, fs, mus, param, i):
        dmz =pd_mag.locmag(magz, tpa, mus, fs, self.sup, self.sdn, param, i)
        dmu=pd_mag.itmag(te, tpo, magz, mus, param, i)
        magz[str(i)]+=dmz
        mus[str(i)]+=dmu
        dmz2=pd_mag.locmag(magz, tpa, mus, fs, self.sup, self.sdn, param, i)
        dmu2=pd_mag.itmag(te, tpo, magz, mus, param, i)
        dmagz=(dmz+dmz2)/2
        dmus=(dmu+dmu2)/2
        return dmagz, np.array([0]), dmus

    def dqes(self, mz, mus, dmz, dmus, pl):
        return self.J*mz*dmus*self.dx*self.dy*self.dz/self.apc

    def dqps(self, mz, mus, dmz, dmus, pl):
        return self.Jloc*mz*dmz/pl['dt']*self.dx*self.dy*self.dz/self.apc

class normalmetal(sample):
    ### This class can compute diffusive spin transport in normal metals
    def __init__(self, name, gepfit, celfit, cphfit, spin, tc, tdeb, muat, dx, dy, dz, apc, asf, inimag, kappa, kappaph, locmom, locspin, coupl, musdiff, rhosd, tsd, tsl):
        super().__init__(name, gepfit, celfit, cphfit, spin, tc, tdeb, muat, dx, dy, dz, apc, asf, inimag, kappa, kappaph, locmom, locspin, coupl, musdiff)
        self.rhosd=rhosd
        self.tsd=tsd
        self.tsl=tsl

    def initmag(self,z):
        fs=np.zeros(1)
        magz=np.zeros(1)
        mus=np.zeros(1)
        return magz, fs, mus

    def dmag(self, z, te, tp, tpa, magz, fs, mus, param, i):
        #dmu=sd_mag.itmag(0, mus, param, i)
        #mus[str(i)]+=dmu
        #dmu2=sd_mag.itmag(0, mus, param, i)
        #dmus=(dmu+dmu2)/2
        return np.zeros(1), np.zeros(1), np.zeros(1) #dmus
        
class insulator(sample):
    def __init__(self, name, gepfit, celfit, cphfit, spin, tc, tdeb, muat, dx, dy, dz, apc, asf, inimag, kappa, kappaph, locmom, locspin, coupl, musdiff):
        super().__init__(name, gepfit, celfit, cphfit, spin, tc, tdeb, muat, dx, dy, dz, apc, asf, inimag, kappa, kappaph, locmom, locspin, coupl, musdiff)
        self.spin=0
        self.tc=0
        self.muat=1. #this is for computational reasons as at some point one has division by this factor, it is however multiplied by 0 anyways..
        self.asf=0
        self.inimag=0
        self.locmom=0
        self.locspin=0
        
    def dmag(self, z, te, tp, tpa, magz, fs, mus, param, i):
        return np.zeros(1), np.zeros(1), np.zeros(1)
        
    def dtem(self, t, tes, tpos, tpas, dqes, dqps, param, i):
        dtp=param['dt']*tempdyn.tpdiffusion(tpos, param, i)/param['sam'][i].cphfit(tpos[str(i)])
        return dtp, dtp, np.array([0])
        
    def initmag(self,z):
        fs=np.zeros(1)
        magz=np.zeros(1)
        mus=np.zeros(1)
        return magz, fs, mus
        

def samplechoice(sam):
    ### Here you can define your samples: Just copy and insert the dummy to save it and change the parameters and rename it: For units check the explanation in the sample class

    ##For M3TM and arbspin model the initialization is identical!
    # if sam=='Dummy-arbspin':
    #     mat=arbspin('put_name_here',
    #                  florianfit.interpol(gep constant, 'const', None),
    #                  florianfit.interpol(Sommerfeld constant, 'lin', None),
    #                  florianfit.interpol(maximum lattice heat capacity, 'einstein', put einstein temperature here),
    #                  effective spin,
    #                  Curie temperature,
    #                  Debye temperature,
    #                  atomic magnetic moment,
    #                  dx,
    #                  dy,
    #                  dz (sample depth direction),
    #                  atoms per grain cell,
    #                  spin-flip-probability,
    #                  initial magnetization (in z-direction)
    #                  electronic heat diffusion constant,
    #                  phononic heat diffusion constant,
    #                  localized magnetic moment,
    #                  localized effective spin,
    #                  coupling to [last layer (closer to laser), this layer, next layer (further from laser)],
    #                  spin diffusion constant)

    ## For s-d model three additional parameters must be defined (last three). Check paper mentioned at beginning of sd_mag.py
    # if sam=='Dummy-sd':
    #   mat=sd('put_name_here',
    #                  florianfit.interpol(gep
    #                  constant, 'const', None),
    #                  florianfit.interpol(Sommerfeld constant, 'lin', None),
    #                  florianfit.interpol(maximum lattice heat capacity, 'einstein', put einstein temperature here),
    #                  effective spin,
    #                  Curie temperature,
    #                  Debye temperature,
    #                  atomic magnetic moment,
    #                  dx,
    #                  dy,
    #                  dz (sample depth direction),
    #                  atoms per grain cell,
    #                  spin-flip-probability,
    #                  initial magnetization (in z-direction)
    #                  electronic heat diffusion constant,
    #                  phononic heat diffusion constant
    #                  localized magnetic moment,
    #                  localized effective spin,
    #                  coupling to [last layer (closer to laser), this layer, next layer (further from laser)],
    #                  spin diffusion constant,
    #                  rhosd,
    #                  tsd, (timescale of s-d equilibration)
    #                  tsl) (timescale of s-lattice equilibration)

    ## For the M5TM four additional parameters mustbe defined (last four)
    # if sam=='Dummy-arbspin':
    #     mat=arbspin('put_name_here',
    #                  florianfit.interpol(gep constant, 'const', None),
    #                  florianfit.interpol(Sommerfeld constant, 'lin', None),
    #                  florianfit.interpol(heat capacity OF OPTICAL PHONONS, 'const.', None),
    #                  effective spin, (%0.5, effective spin=localized spin+0.5)
    #                  Curie temperature,
    #                  Debye temperature,
    #                  atomic magnetic moment, (total, localized+itinerant)
    #                  dx,
    #                  dy,
    #                  dz (sample depth direction),
    #                  atoms per grain cell,
    #                  spin-flip-probability,
    #                  initial magnetization (in z-direction)
    #                  electronic heat diffusion constant,
    #                  phononic heat diffusion constant
    #                  localized magnetic moment, #(vast majority of atomic moment total)
    #                  localized effective spin,  #(effective spin-0.5)
    #                  coupling to [last layer (closer to laser), this layer, next layer (further from laser)],
    #                  spin diffusion constant,
    #                  florianfit.interpol(maximum heat capacity of ACCOUSTIC PHONONS, 'const.', None),
    #                  Jpd, #(exchange coupling of p- and d- magnetic momenta)
    #                  tmp, #(timescale of equilibration of accoustic phonons and localized magnetic moments (slow!))
    #                  florianfit.interpol(coupling constant of optical and accoustic phonons, 'const', None)
    
    
    if sam=='max_sub':
        mat=insulator('substrate',
                    florianfit.interpol(0., 'const', None),
                    florianfit.interpol(1., 'const', None),
                    florianfit.interpol(1.8e6, 'const', None),
                    0,
                    0,
                    100,
                    0,
                    2.859e-10,
                    2.859e-10,
                    2.859e-10,
                    1,  #apc
                    0,  #asf
                    0,  #inimag
                    0.,  #kappa
                    2., #kph
                    0,
                    0,
                    [1,1,1],
                    0)

    elif sam=='m5tmtest':
        mat=m5tm('testmat',
                   florianfit.interpol(1e16, 'const', None),
                   florianfit.interpol(1e3, 'lin', None),
                   florianfit.interpol(1e6, 'const', None),
                   2,
                   400,
                   360,
                   2.,
                   1e-10,
                   1e-10,
                   1e-10,
                   1,
                   0.01,
                   1.,
                   0.,
                   0.,
                   1.95,
                   1.5,
                   [2,4,2],
                   0.,
                   florianfit.interpol(1e6, 'const', None),
                   3e-21,
                   1e-9,
                   florianfit.interpol(1e17, 'const', None))

    elif sam=='Nickel':
        mat=m3tm('Nickel',
                      florianfit.interpol('Ab-initio-Ni/Ni_G_ep.txt', None, None),  #gep
                      florianfit.interpol('Ab-initio-Ni/Ni_c_e.txt', None, None),   #ce
                      florianfit.interpol('Ab-initio-Ni/Ni_c_p.txt', None, None),   #cp
                      0.5, #spin
                      633, #tc
                      360, #tdeb
                      0.616, #muat
                      3.524e-10, 3.524e-10, 3.524e-10, #dxdydz
                      4, #apc
                      0.063,  #asf
                      0.96925, #inimag at 295K
                      81., #kappa
                      9.6, #kappaph
                      0., #locmom
                      0., #locspin
                      [0,1,0], #layercoupling
                      0.) #spin diffusion constant

    elif sam=='Cobalt':
        mat = arbspin('Cobalt',
                        florianfit.interpol('Ab-initio-Co/Co_G_ep.txt', None, None),
                        florianfit.interpol('Ab-initio-Co/Co_c_e.txt', None, None),
                        florianfit.interpol('Ab-initio-Co/Co_c_p.txt', None, None),
                        1.5,  # spin
                        1423,  # tc
                        342.5,  # tdeb
                        1.8,  # muat
                        3.54e-10, 3.54e-10, 3.54e-10,  # dxdydz
                        4,  # apc
                        0.045,  # asf
                        0.99793,  # inimag at 295K
                        90.7,  # kappa
                        0., #kappaph
                        0.,
                        0.,
                        [3,6,3],
                        0)

    elif sam=='Iron':
        mat=arbspin('Iron',
                    florianfit.interpol('Ab-initio-Fe/Fe_G_ep.txt', None, None),
                    florianfit.interpol('Ab-initio-Fe/Fe_c_e.txt', None, None),
                    florianfit.interpol('Ab-initio-Fe/Fe_c_p.txt', None, None),
                    2, #spin
                    1041, #tc
                    396, #tdeb
                    2.2, #muat
                    2.856e-10,2.856e-10, 2.856e-10, #dxdydz
                    2, #apc
                    0.035, #asf
                    0.9863, #inimag at 295K
                    0., #kappa
                    0., #kappaph
                    0., #locmom
                    0., #locspin
                    [2,4,2], #layer coupling
                    0.) #spin diffusion constant

    elif sam=='Gadolinium':
        mat = sd('Gadolinium',
                            florianfit.interpol(2.5e17, 'const', None), #gep
                            florianfit.interpol(225., 'lin', None), #ce
                            florianfit.interpol(1.51e6, 'einstein', 120), #cp
                            3.5, #spin
                            293, #Tc
                            160, #tdeb
                            7.5, #muat
                            3.6e-10, #dx
                            3.6e-10, #dy
                            5.8e-10, #dz
                            6, #atoms per unit cell
                            0.12, #asf
                            1., #inimag
                            10.5, #kappa
                            0., #kappaph
                            7., #spin diffusion constant
                            3.,
                            [3,6,3],
                            0*250e-6,
                            1e-19,
                            1.5e-10,
                            1e-13)

    elif sam=='Gold':
        mat=normalmetal('Gold',
                        florianfit.interpol(25e15, 'const', None),
                        florianfit.interpol(67.6, 'lin', None),
                        florianfit.interpol(2.5e6, 'einstein', 0.75*178),
                        0,
                        0,
                        178.,
                        0,
                        2.354e-10,
                        2.354e-10,
                        2.354e-10,
                        4,
                        0,
                        0, #inimag
                        313.,
                        5., #kappaph
                        0.,
                        0.,
                        [1,1,1],
                        0*9500e-6,
                        1.,
                        1.,
                        1.)

    elif sam=='Tantalum':
        mat = normalmetal('Tantalum',
                          florianfit.interpol(100e15, 'const', None),
                          florianfit.interpol(384.3, 'lin', None),
                          florianfit.interpol(2.32e6, 'einstein', 0.75*225),
                          0.,
                          0.,
                          225,
                          0,
                          2.859e-10,
                          2.859e-10,
                          2.859e-10,
                          2,
                          0, #inimag
                          0,
                          52.,
                          5., #kappaph
                          0.,
                          0.,
                          [1,1,1],
                          0*9500e-6,
                          1.,
                          1.,
                          1.)

    elif sam=='Tungsten':
        mat = normalmetal('Thungsten',
                          florianfit.interpol(2.5e17,'const', None), #gep
                          florianfit.interpol(0.14e3,'lin', None), #ce
                          florianfit.interpol(1.3e6,'const', None), #cp
                          0, #spin
                          0, #tc
                          100., #td
                          0, #muat
                          3.2e-10,
                          3.2e-10,
                          3.2e-10,
                          8, #apc
                          0, #asf
                          0, #inimag
                          100., #kappa
                          0., #kappaph
                          0., #locmom
                          0., #locspin
                          [4,0,4], #nearest neighbours
                          0*9500e-6, #spin diff const
                          1e-19,
                          1.5e-10,
                          1e-13)
    elif sam=='FGT':
        mat = m4tm('FGT',
                          florianfit.interpol(4.4e17, 'const', None), #gep
                          florianfit.interpol(1.5e3, 'lin', None), #ce
                          florianfit.interpol(0.5*4e6, 'einstein', 0.75*220), #cp
                          2., #spin
                          191, #tc
                          220, #tdeb
                          2., #muat
                          3.9e-10, #dx
                          3.9e-10, #dy
                          16.3e-10, #dz
                          10, #apc
                          0.035, #asf
                          0.8,#inimag
                          0,  #kappa
                          0., #kappaph
                          0., #locmom
                          0., #locspin
                          [0,1,0], #nearest neighbours
                          0,    #spin diffusion constant
                          florianfit.interpol(0.5*4e6, 'einstein', 0.75*220), #cp2
                          florianfit.interpol(1e17, 'const', None)) #gpp
    elif sam=='CGT':
        mat = arbspin('CGT',
                          florianfit.interpol(15e16, 'const', None), # gep (electron-phonon coupling) [W/(K*m^3)]. Value not found in the literature (value Theo)
                          florianfit.interpol(736.87, 'lin', None), # Sommerfeld constant [units?]. Value not found in the literature (https://arxiv.org/abs/2202.03251) (value Theo)
                          florianfit.interpol(8.9e6, 'const', None), # cp (maximum lattice heat capacity) [J/(K*m^3)]. Value not found in the literature. This phononic bath is the one coupled to the electronic one! (value Theo)
                          1.5, # Effective spin. Literature: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.100.224427, https://pubs.rsc.org/en/content/articlelanding/2022/nr/d1nr05821e, 
                          65., # Tc (Curie temperature) [K]. Literature: 65 K (https://onlinelibrary.wiley.com/doi/full/10.1002/adma.202110583, https://onlinelibrary.wiley.com/doi/abs/10.1002/adma.202008586), 61 K (https://www.sciencedirect.com/science/article/abs/pii/S0927025622004013), and 30 K in bilayer and 68 K in bulk (https://iopscience.iop.org/article/10.1088/1361-6528/ac17fd)
                          175, # ThetaD (Debye temperature) [K]. Literature: 198 K (https://onlinelibrary.wiley.com/doi/abs/10.1002/adfm.202105111), 174.5 or 181.9 K (https://arxiv.org/abs/2202.03251), and monolayer: 243 K (Suppl. Inf.: https://pubs.rsc.org/en/content/articlelanding/2020/cp/d0cp03884a)
                          4., # muat (atomic magnetic moment) [in units of mu_B [J/T]]. Literature: approximatelly (local) 3*muB per Cr atom (https://www.sciencedirect.com/science/article/abs/pii/S0927025622004013), monolayer: 3.297*muB (https://pubs.rsc.org/en/content/articlelanding/2022/nr/d1nr05821e), and 1.9 or 2.2-2.9*muB per Cr atom (https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.013139)
                          6.826e-10, # dx (lattice constant along x-th direction) [m]. Literature: 6.826 AA (https://onlinelibrary.wiley.com/doi/full/10.1002/adma.202110583), 6.857 AA (https://www.sciencedirect.com/science/article/abs/pii/S0927025622004013), and 6.82 AA (https://pubs.rsc.org/en/content/articlelanding/2022/nr/d1nr05821e)
                          6.856e-10, # dy (lattice constant along y-th direction) [m]. Literature: 6.857 AA (https://www.sciencedirect.com/science/article/abs/pii/S0927025622004013)
                          20e-10, # dz (lattice constant along z-th direction) [m]. Literature: 20.531 AA (https://onlinelibrary.wiley.com/doi/full/10.1002/adma.202110583)
                          10, # apc ((magnetic) atoms per unit cell). Literature: 2 Cr atoms per unit cell (https://onlinelibrary.wiley.com/doi/abs/10.1002/adfm.202108953)
                          0.1, # asf (spin flip probability). Value not found in the literature
                          1, # initial magnetization (in z-th direction). Literature: in principle, for bulk, "It exhibits a strong magnetic anisotropy with an out-of-plane easy axis..." (https://www.nature.com/articles/s41928-020-0427-7)
                          0.0016, # kappa (electronic heat diffusion constant) [W/(m*K)]. (https://pubs.acs.org/doi/pdf/10.1021/acs.chemmater.5b04895)
                          5.34, # kappaph (phononic heat diffusion constant) [W/(m*K)]. (https://pubs.acs.org/doi/pdf/10.1021/acs.chemmater.5b04895)
                          0, # locmom (localized magnetic moment (vast majority of the total magnetic moment)) [J/T]. Literature: (local) 3.297*muB (https://pubs.rsc.org/en/content/articlelanding/2022/nr/d1nr05821e)
                          0, # locspin (localized effective spin) [dimensionless?]. Literature: (effective spin magnetic moment) of 1.9*muB per Cr atom (https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.013139)
                          [0,1,0], # coupl (coupling to [last layer (closer to the laser), this layer, next layer (further from laser)]. For monolayer, [0,1,0])
                          0) # musdiff (spin diffusion constant) [m^2/s]. Value not found in literature

    elif sam=='SiO2':
        mat = insulator('SiO2',
                          florianfit.interpol(0., 'const', None), # gep (electron-phonon coupling) [W/(K*m^3)]. Value not found in the literature. Apparently, there is a strong coupling between the electronic transition and a collective vibration in the SiO2 nanoparticle due to a sudden relaxation mechanism involving charges (https://pubs.acs.org/doi/pdf/10.1021/nl901509k). In an insulator the interaction between a charge carrier and the lattice vibrations or phonons is often very strong (Electron-Phonon Interactions in an Insulator) (I took the value of max_sub)
                          florianfit.interpol(1., 'lin', None), # celfit (electronic heat capacity) [W/(K*m^3)]. Value not found in the literature (I took the value of max_sub)
                          florianfit.interpol(1e4, 'const', None), # cphfit (lattice heat capacity) [J/(K*m^3)]. Value not found in the literature. This phononic bath is the one coupled to the electronic one! (I took the value of max_sub)
                          0, # Effective spin. Non-magnetic material: 0
                          0, # Tc (Curie temperature) [K]. Non-magnetic material: 0 K
                          403, # ThetaD (Debye temperature) [K]. Literature: 403 K (https://arxiv.org/pdf/1501.03176.pdf)
                          0, # muat (atomic magnetic moment) [in units of mu_B [J/T]]. Non-magnetic material: 0
                          4.92e-10, # dx (lattice constant along x-th direction) [m]. (https://materialsproject.org/materials/mp-6930/)
                          4.92e-10, # dy (lattice constant along y-th direction) [m]. (https://materialsproject.org/materials/mp-6930/)
                          5e-10, # dz (lattice constant along z-th direction) [m]. (https://materialsproject.org/materials/mp-6930/)
                          9, # apc (number of atoms per unit cell) (https://materialsproject.org/materials/mp-6930/)
                          0, # asf (spin flip probability). Value not found in the literature
                          0, # inimag (initial magnetization)
                          1.4, # kappa (electronic heat diffusion constant) [W/(m*K)]. Literature: heat conductivity 1.4 W/(m*K) (https://aip.scitation.org/doi/full/10.1063/1.2399441?casa_token=HuRB25Yj-iEAAAAA%3AzvsBpRMumamNbEkksIZ3FIgW4pb8Ye6YKZM1h7Z_bDJf-SOqdGg7L_D1J84xspc3hCZ_CwYOybE), and thermal conductivity 1.38 W/(m*K) (https://www.sciencedirect.com/science/article/pii/S1876610215022365). Important parameter in our case, both for sample and substrate!
                          1.4, # kph (phononic heat diffusion constant) [W/(m*K)]
                          0, # locmom (localized magnetic moment (vast majority of the total magnetic moment)) [J/T]. Non-magnetic material, 0?
                          0, # locspin (localized effective spin) [dimensionless?]. Non-magnetic material, 0?
                          [0,1,0], # coupl (coupling to [last layer (closer to the laser), this layer, next layer (further from laser)]. For monolayer, [0,1,0])
                          0) # musdiff (spin diffusion constant) [m^2/s]
    
    elif sam=='hBN':
        mat = insulator('hBN', # Apparently, this material is a promising 2D insulator (https://www.nature.com/articles/s41928-020-00529-x)
                          florianfit.interpol(0., 'const', None), # gep (electron-phonon coupling) [W/(K*m^3)]. Apparently, "electron-phonon couplings are the main reason for the formation of the double peak" (with XANES) (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.235205). Value not found in literature! (I take the one given in max_sub, but it should not be zero)
                          florianfit.interpol(1., 'lin', None), # celfit (electronic heat capacity) [W/(K*m^3)]. Value not found in the literature (I took the value of max_sub)
                          florianfit.interpol(1e4, 'const', None), # cphfit (lattice heat capacity) [J/(K*m^3)]. (https://aip.scitation.org/doi/pdf/10.1063/1.4991715)
                          0, # Effective spin. Apparently, in the abscence of vacancies, non-magnetic material: 0 (https://iopscience.iop.org/article/10.1088/1674-1056/27/1/016301/meta, http://cpb.iphy.ac.cn/article/2018/1919/cpb_27_1_016301.html)
                          0, # Tc (Curie temperature) [K]. Apparently, in the abscence of vacancies, non-magnetic material: 0 K (https://iopscience.iop.org/article/10.1088/1674-1056/27/1/016301/meta, http://cpb.iphy.ac.cn/article/2018/1919/cpb_27_1_016301.html)
                          400, # ThetaD (Debye temperature) [K]. Literature: 400 K (https://www.ioffe.ru/SVA/NSM/Semicond/BN/thermal.html), 410 or 598 K (https://materials.springer.com/googlecdn/assets/sm_lbs/834/sm_lbs_978-3-540-31356-4_576/sm_lbs_978-3-540-31356-4_576.pdf?trackRequired=true&originUrl=/lb/docs/sm_lbs_978-3-540-31356-4_576&componentId=Download%20Chapter), and 600 K? (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.73.064304)
                          0, # muat (atomic magnetic moment) [in units of mu_B [J/T]]. Non-magnetic material: 0
                          2.51e-10, # dx (lattice constant along x-th direction) [m]. (http://www.ioffe.ru/SVA/NSM/Semicond/BN/basic.html)
                          2.51e-10, # dy (lattice constant along y-th direction) [m]. (http://www.ioffe.ru/SVA/NSM/Semicond/BN/basic.html)
                          8e-10, # dz (lattice constant along z-th direction) [m]. (http://www.ioffe.ru/SVA/NSM/Semicond/BN/basic.html)
                          4, # apc (2 B atoms and 2 N atoms? All non-magnetic. Correct phase?). Literature: .cif file from Springer webpage (bulk case, because the layers in experiments are of 10-20 nm along the thickness of the unit cell) (https://materialsproject.org/materials/mp-984#summary)
                          0, # asf (spin flip probability). Value not found in the literature
                          0, # inimag (initial magnetization)                          
                          5., # kappa (electronic heat diffusion constant) [W/(m*K)]. (https://aip.scitation.org/doi/pdf/10.1063/1.4991715)
                          360., # kph (phononic heat diffusion constant) [W/(m*K)]. Value not found in literature (I took the value from max_sub)
                          0, # locmom (localized magnetic moment (vast majority of the total magnetic moment)) [J/T]. Non-magnetic material: 0?
                          0, # locspin (localized effective spin) [dimensionless?]. Non-magnetic material: 0?
                          [0,1,0], # coupl (coupling to [last layer (closer to the laser), this layer, next layer (further from laser)]. For monolayer, [0,1,0])
                          0) # musdiff (spin diffusion constant) [m^2/s]
                          
    return(mat)
