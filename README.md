# Multilayer Microscopic Three Temperature Model Simulations

Welcome! This code solves the M3TM equations in a magnetic material in a multilayer of an arbitrary amount of cap and substrate layers.
This code is not yet an official python package, so to use it you must download the code and save it locally.
This is a quick guide to set up a simulation, e.g. how to write a Scriptfile, specifying and shortly explaining the input parameters.
The last version control has been done for Python 3.11.

## Necessary packages
The following packages need to be installed to run the code:

-numpy
-scipy
-os
-time
-datetime
-matplotlib

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

_For an insulating material, you only need to define the parameters for a phononic system._
```
my_insulator = SimMaterials(name='Gummydummy', tdeb=400, cp_max=3e6, kappap=30.)
```

_For a conducting (or semimetallic materials whose electrons can be excited above the bandgap by the used pulse), you also need to define thermal parameters for the electronic system and their interaction with the phonons via electron-phonon-counpling._
```
my_conductor = SimMaterials(name='Blitzydummy', tdeb=300, cp_max=2.5e6, kappap=20., ce_gamma=1000, kappae=100., gep=1e18)
```

_For a magnetic material, whose spin dynamics you want to model with the M3TM, you need to define additional parameters within the model._
```
my_magnet = SimMaterials(name='Spinnydummy', tdeb=200, cp_max=2e6, kappap=10., ce_gamma=750, kappae=150., gep=0.8e17, spin=2.5, vat=1e-28., tc=600., muat=5., asf=0.06)
```
