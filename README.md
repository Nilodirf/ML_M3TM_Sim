# Multilayer Microscopic Three Temperature Model Simulations

Welcome! This code solves the M3TM equations in a magnetic material in a multilayer of an arbitrary amount of cap and substrate layers.
This code is not yet an official python package, so to use it you must download the code and save it locally.
This is a quick guide to set up a simulation, e.g. how to write a Scriptfile, specifying and shortly explaining the input parameters.
The last version control has been done for Python 3.11.

## Necessary packages
The following packages need to be installed to run the code:

- numpy
- scipy
- os
- time
- datetime
- matplotlib

## Creating the Script file
Create a script within the code/Scripts folder. All the contents of this file are listed below.

### Import the Source files 
These are relative imports of all the source files needed for the simulation:

```
from ..Source.mats import SimMaterials
from ..Source.sample import SimSample
from ..Source.pulse import SimPulse
from ..Source.mainsim import SimDynamics
```

### Create materials

Materials are defined merely by their parameters. Here is a list of all the available parameters:

- ***name*** (string).    _Name of the material_
- ***tdeb*** (float).     _Debye temperature of the material_
- ***cp_max*** (float).   _Maximal phononic heat capacity in W/m**3/K. Temperature dependence is computed with Einstein model_
- ***kappap*** (float).   _Phononic heat diffusion constant in W/m/K_
- ***kappae*** (float).   _Electronic heat diffusion constant in W/m/K_
- ***ce_gamma*** (float). _Sommerfeld constant of electronic heat capacity in J/m**3/K_
- ***gep*** (float).      _Electron-phonon coupling constant in W/m**3/K_
- ***spin*** (float).     _Effective spin of the material
- ***tc*** (float).       _Curie temperature of the material_
- ***muat*** (float).     _Atomic magnetic moment in unit of \mu_Bohr_
- ***asf*** (float).      _Electron-phonon-scattering induced spin flip probability of the material_
- ***vat*** (float).      _Magnetic atomic volume in m^3._


_For an insulating material, you only need to define the parameters for a phononic system:_
```
my_insulator = SimMaterials(name='Gummydummy', tdeb=400, cp_max=3e6, kappap=30.)
```

_For a conducting (or semimetallic materials whose electrons can be excited above the bandgap by the used pulse), you also need to define thermal parameters for the electronic system and their interaction with the phonons via electron-phonon-counpling:_
```
my_conductor = SimMaterials(name='Blitzydummy', tdeb=300, cp_max=2.5e6, kappap=20., ce_gamma=1000, kappae=100., gep=1e18)
```

_For a magnetic material, whose spin dynamics you want to model with the M3TM, you need to define additional parameters within the model:_
```
my_magnet = SimMaterials(name='Spinnydummy', tdeb=200, cp_max=2e6, kappap=10., ce_gamma=750, kappae=150., gep=0.8e17, spin=2.5, vat=1e-28., tc=600., muat=5., asf=0.06)
```

### Create a sample structure

With the materials you created before you can now build a sample.

```
my_sample = SimSample()
```

So far, our sampleholder is completely empty. Let's quickly grow a sample with a 5 nm _Blitzydummy_ cap layer, a 15 nm _Spinnydummy_ magnet and a 300 nm _Gummydummy_ substrate.

You need to watch out for three things:
1. If you interface two materials, you always need to spcify boundary conditions for the interfacial phononic heat diffusion.
2. If you interface two materials __with itinerant electrons__, you also need to specify boundary conditions for the interfacial electronic heat diffusion.
3. You need two specify either a penetration depth for the laser pulse or a complex refractive index to simulate the penetration of the pump pulse into your sample structure(I will add both because I can!). For insulating materials introduce a penetration depth of exactly an integer 1!

```
my_sample.add_layers(material=my_insulator, dz= 1nm, layers=5, pen_dep=7.5e-9, n_comp=2.8+8.45j)
my_sample.add_layers(material=my_magnet, dz=1nm, layers=15, pen_dep=34e-9, n_comp=2.21+2.73j, kappae_int='max', kappap_int=1.)
my_sample.add_layers(material=my_conductor, dz=1nm, layers=300, pen_dep=1, n_comp=1.97+0j, kappap_int='av')
```

### Create a Pulse

With the sample created we can now compute already how a pump pulse interacts with it.

```
my_pulse_Abeles = SimPulse(sample=my_sample, method='Abeles', pulse_width=20e-15, fluence=5., delay=0.5e-12, photon_energy_eV=1.55, theta=1/4, phi=1/3)
my_pulse_Lambert_Beer = SimPulse(sample=my_sample, method='LB', pulse_width=20e-15, fluence=5., delay=0.5e-12)
```

### Define Simulation Parameters

All the physical parameters are defined now, we just need to run the simulation now. Therefor we need to define some computational parameters. These may influence if the simulation runs fast, slow, wonky or exact. In the end, you might need to play around a little until you have found a configuration that yields solid and fast results.

```
my_simulation = SimDynamics(sample=my_sample, pulse=my_pulse_Abeles, ini_temp=300., end_time=200e-12, solver='RK23', max_step=1e-14)
```

Let's just run it and see if we did okay here. To look at our data later we also need to save it:
```
my_results = my_simulation.get_t_m_maps()
my_simulation.save_data(my_results, save_file='my_result_files')
```





