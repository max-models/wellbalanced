---
title: Quickstart
description: A global model and a 1D discharge with wellbalanced.
---

First, ensure that `wellbalanced` is [installed](/wellbalanced/getting-started/installation/).

## A global model

Species, reactions with rates from cross sections (averaged over a Maxwellian by
[plasmacoll](https://github.com/max-models/plasmacoll)), a chamber, and the power the
electrons absorb:

```python
from plasmacoll import CrossSection
from plasmacoll.constants import ELECTRON_MASS

from wellbalanced import ELECTRON, Chamber, GlobalModel, Reaction, Species

argon, argon_ion = Species.atomic("Ar", 39.948), Species.atomic("Ar+", 39.948, 1)
ionization = Reaction.from_cross_section(
    ("e", "Ar"),
    ("e", "e", "Ar+"),
    CrossSection.constant(2.0e-20, threshold=15.76),
    ELECTRON_MASS,
    electron_energy_loss=15.76,
)
model = GlobalModel(
    species=[ELECTRON, argon, argon_ion],
    reactions=[ionization],
    chamber=Chamber(volume=1.0e-3, wall_area=0.06, h_factor=0.5),
    power=100.0,  # W
    fixed={"Ar": 1.0e20},  # m^-3
)
state = model.steady_state()
print(state.electron_temperature, state.electron_density)
print(state.power)  # every term of the electron energy balance, in W
```

`model.scan("power", [50.0, 100.0, 200.0])` returns the steady states for several values,
each starting from the last.

Every result is also a labelled xarray dataset, `state.to_dataset()`, and a scan one
dataset along its parameter:

```python
import plasma_plots  # registers the .plasma accessor

from wellbalanced import stack

powers = [50.0, 100.0, 200.0]
scan = stack(model.scan("power", powers), "power", powers, units="W")
scan.density.sel(species="e").plasma.plot.lineout()  # with plasma-plots
```

## A 1D discharge

The same reaction in a slab between two grounded walls, sustained by uniform heating:

```python
import numpy as np

from wellbalanced import Fluid1D, FluidSpecies

model = Fluid1D(
    species=[
        FluidSpecies(ELECTRON, mobility=1.0e3),
        FluidSpecies(argon_ion, mobility=10.0),
    ],
    reactions=[ionization],
    length=0.05,
    cells=200,
    fixed={"Ar": 1.0e21},
    heating=2.0e4,  # W/m^3
)
guess = 1.0e15 * np.sin(np.pi * model.x / 0.05) + 1.0e12
state = model.steady_state(model.pack({"e": guess, "Ar+": guess}, 3.0, 0.0))
print(state.densities["e"].max(), model.energy_balance(state))
```

Boundaries are set per species and end, and can be changed between steps:
`model.set_boundary("left", "e", Boundary.flux(1e16))`. Resolve the Debye length with the
cells: with an unresolved sheath Newton's method cannot take long implicit steps.

## A neutralizer

A fast beam's charge states on the gas and on the plasma it makes, with that plasma as a
global model of the channel:

```python
from wellbalanced.neutralizer import illustrative_deuterium_neutralizer

model = illustrative_deuterium_neutralizer()  # placeholder numbers: it warns
result = model.solve()
print(result.neutral_fraction, result.plasma.ionization_degree(["D2", "D"]))
```

Real data go into a `Neutralizer` with a `BeamData` (the beam's cross sections on the gas
and the plasma at its energy) or a `BeamCrossSections` (the same as functions of the beam
energy, e.g. plasmacoll `CrossSection`s, and the gas ionization as an energy distribution of
its secondary electrons such as `plasmacoll.RuddIonization`), and the plasma's species and
reactions. `model.beam_sources(flux, fractions, energy)` gives the beam's sources where the
beam has that flux, those charge-state fractions and that energy: numbers for the global
model, or arrays on the cells of a `Fluid1D` (its `volume_sources`).

See the [API reference](/wellbalanced/api/wellbalanced/) for every class and function.
