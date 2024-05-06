# Multilayer Microscopic Three Temperature Model Simulations (_ML_M3TM_Sim_)

Welcome! This code solves the M3TM equations in a magnetic material in a multilayer of an arbitrary amount of cap and substrate layers.
This code is not yet an official python package, so to use it you must download the code and save it locally.
This is a quick guide to set up a simulation, e.g. how to write a Scriptfile, specifying and shortly explaining the input parameters.
The last version control has been done for Python 3.11.

## Necessary packages
The following packages need to be installed to run the code:

- ***numpy***
- ***scipy***
- ***os***
- ***time***
- ***datetime***
- ***matplotlib***

## Creating the Script file

Create a script within the code/Scripts folder. For here, we shall call our file **tutorial_input.py**. All the contents of this file are listed below.

### Source code imports

<details>
<summary>How to import the source code</summary>
  
These are relative imports of all the source files needed for the simulation:

```python
from ..Source.mats import SimMaterials
from ..Source.sample import SimSample
from ..Source.pulse import SimPulse
from ..Source.mainsim import SimDynamics
```

</details>
  
### Create materials

<details>
<summary>How to create a material</summary>

Materials are defined merely by their parameters. Here is a list of all the available parameters, where  [**_optional_1_**] denotes that these parameters need to be defined for materials with an electronic subsystem and  [**_optional_2_**] parameters need to be introduced for materials with a spin subsystem:

- ***name*** (string).    _Name of the material_
- ***tdeb*** (float).     _Debye temperature of the material_
- ***cp_max*** (float).   _Maximal phononic heat capacity in W/m**3/K. Temperature dependence is computed with Einstein model_
- ***kappap*** (float).   _Phononic heat diffusion constant in W/m/K_
- ***kappae*** [**_optional_1_**] (float).   _Electronic heat diffusion constant in W/m/K_
- ***ce_gamma*** [**_optional_1_**] (float). _Sommerfeld constant of electronic heat capacity in J/m**3/K_
- ***gep*** [**_optional_1_**] (float).      _Electron-phonon coupling constant in W/m**3/K_
- ***spin*** [**_optional_2_**] (float).     _Effective spin of the material_
- ***tc*** [**_optional_2_**] (float).       _Curie temperature of the material_
- ***muat*** [**_optional_2_**] (float).     _Atomic magnetic moment in unit of \mu_Bohr_
- ***asf*** [**_optional_2_**] (float).      _Electron-phonon-scattering induced spin flip probability of the material_
- ***vat*** [**_optional_2_**] (float).      _Magnetic atomic volume in m^3._


_For an insulating material, you only need to define the parameters for a phononic system:_
```python
my_insulator = SimMaterials(name='Gummydummy', tdeb=400, cp_max=3e6, kappap=30.)
```

_For a conducting (or semimetallic materials whose electrons can be excited above the bandgap by the used pulse), you also need to define thermal parameters for the electronic system and their interaction with the phonons via electron-phonon-counpling:_
```python
my_conductor = SimMaterials(name='Blitzydummy', tdeb=300, cp_max=2.5e6, kappap=20., ce_gamma=100, kappae=100., gep=1e18)
```

_For a magnetic material, whose spin dynamics you want to model with the M3TM, you need to define additional parameters within the model:_
```python
my_magnet = SimMaterials(name='Spinnydummy', tdeb=200, cp_max=2e6, kappap=10., ce_gamma=75, kappae=150., gep=0.8e18, spin=2.5, vat=1e-28, tc=600., muat=5., asf=0.06)
```

</details>

### Create a sample structure

<details>
<summary>How to build a sample</summary>

With the materials you created before you can now build a sample.

```python
my_sample = SimSample()
```

So far, our sampleholder is completely empty. Let's quickly grow a sample with a 5 nm _Blitzydummy_ cap layer, a 15 nm _Spinnydummy_ magnet and a 300 nm _Gummydummy_ substrate.

You need to watch out for three things:
1. If you interface two materials, you always need to spcify boundary conditions for the interfacial phononic heat diffusion.
2. If you interface two materials __with itinerant electrons__, you also need to specify boundary conditions for the interfacial electronic heat diffusion.
3. You need two specify either a penetration depth for the laser pulse or a complex refractive index to simulate the penetration of the pump pulse into your sample structure(I will add both because I can!). For insulating materials introduce a penetration depth of exactly an integer 1!

Here is a list of the parameters to chose when adding layers to your sample:
- ***material*** (object). _A material previously defined with the materials class_
- ***dz*** (float). _Layer thickness of the material in m. Important only for resolution of heat diffusion_
- ***layers*** (int). _Number of layers with depth material.dz to be added to the sample_
- ***kappap_int*** [**_see 1._**] (float/string). _Phononic interface heat conductivity to the last block of the sample. Either in W/m/K or 'av', 'min', 'max' of the constants of the two interfaced materials_
- ***kappae_int*** [**_see 2._**] (float/string). _Electronic interface heat conductivity to the last block of the sample. Either in W/m/K or 'av', 'min', 'max' of the constants of the two interfaced materials_
- ***pen_dep*** [**_see 3._**] (float). _Penetration depth of the laser pulse in m if to be computed with Lambert-Beer absorption profile_
- ***n_comp***[**_see 3._**]  (complex float). _Complex refractive index of the material. Use syntax 'n_r'+'n_i'j to initiate_

```python
my_sample.add_layers(material=my_conductor, dz= 1e-9, layers=5, pen_dep=7.5e-9, n_comp=2.8+8.45j)
my_sample.add_layers(material=my_magnet, dz=1e-9, layers=15, pen_dep=34e-9, n_comp=2.21+2.73j, kappae_int='max', kappap_int=1.)
my_sample.add_layers(material=my_insulator, dz=1e-9, layers=300, pen_dep=1, n_comp=1.97+0j, kappap_int='av')
```

</details>

### Create a Pulse

<details>
<summary>How to create a Pulse</summary>

With the sample created we can now compute already how a pump pulse interacts with it.

To define the pulse you can/must introduce the following parameters:
- ***sample*** (object). _Sample in use_
- ***pulse_width*** (float). _Sigma of gaussian pulse shape in s_
- ***fluence*** (float). _Fluence of the laser pulse in mJ/cm^2_
- ***delay*** (float). _Time-delay of the pulse peak after simulation start in s_
- ***method*** (String). _Method to calculate the pulse excitation map. Either 'LB' for Lambert-Beer or 'Abeles' for the matrix formulation calculating the profile via the Fresnel equations_
- ***photon_energy_ev***  [**_only_Abeles_**] (float). _Energy of the optical laser pulse in eV. Only necessary for method 'Abeles'_
- ***theta*** [**_only_Abeles_**] (float). _Angle of incidence of the pump pulse in respect to the sample plane normal in units of pi, so between 0 and 1/2. Only necessary for method 'Abeles'_
- ***phi*** [**_only_Abeles_**] (float). _Angle of polarized E-field of optical pulse in respect to incidence plane in units of pi, so between 0 and 1/2. Only necessary for method 'Abeles'_

```python
my_pulse_Abeles = SimPulse(sample=my_sample, method='Abeles', pulse_width=20e-15, fluence=5., delay=0.5e-12, photon_energy_ev=1.55, theta=1/4, phi=1/3)
my_pulse_Lambert_Beer = SimPulse(sample=my_sample, method='LB', pulse_width=20e-15, fluence=5., delay=0.5e-12)
```

</details>

### Define Simulation Parameters

<details>
<summary>How to set up the simulation</summary>

All the physical parameters are defined now, we just need to run the simulation now. Therefor we need to define some computational parameters. These may influence if the simulation runs fast, slow, wonky or exact. In the end, you might need to play around a little until you have found a configuration that yields solid and fast results.

The parameters to define are:
- ***sample*** (object). _The sample in use_
- ***pulse*** (object). _The pulse excitation in use_
- ***end_time*** (float). _Final time of simulation (including pulse delay) in s_
- ***ini_temp*** (float). _Initial temperature of electron and phonon baths in the whole sample in K_
- ***solver*** (String). _The solver used to evaluate the differential equation. See_ [documentation of scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
- ***max_step*** (float). _Maximum step size in s of the solver for the whole simulation_
- ***atol*** [**_optional_**] (float). _Absolute tolerance of solve_ivp solver. Default is 1e-6 as the default of the solver_
- ***rtol*** [**_optional_**] (float). _Relative tolerance of solve_ivp solver. Default is 1e-3 as the default of the solver_

```python
my_simulation = SimDynamics(sample=my_sample, pulse=my_pulse_Abeles, ini_temp=300., end_time=20e-12, solver='Radau', max_step=1e-13)
```

Let's just run it and see if we did okay here. To look at our data later we also need to save it on the harddrive. Automatically, in the package's regisrtry, a folder 'Results' will be created, where the simulation output files will be stored within a folder denoted by _save_file_. The actual data is stored in .npy format:
```python
my_results = my_simulation.get_t_m_maps()
my_simulation.save_data(my_results, save_file='my_result_files')
```

</details>

## Run the simulation

To run the simulation you have set up, open the terminal and navigate to _ML_M3TM_Sim_.
Run the command
```
py -m code.Scripts.tutorial_input
```

The command prompt will show the setup of your sample, the computed pulse profile, initial conditions and briefly inform about the state of the simulation.

## Check the output
Within the plot module there are several methods to visualize your output data. Here is a quick guide to look at your data in some ways:
