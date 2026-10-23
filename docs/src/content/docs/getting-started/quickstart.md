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

`FluidSpecies(argon_ion, mobility=10.0, momentum=True)` gives the ions their momentum
equation instead of drift-diffusion (the friction on the gas from the mobility), so they
accelerate through a presheath to the Bohm speed and on through the sheath; with a large
mobility the ions are collisionless.

Boundaries are set per species and end, and can be changed between steps:
`model.set_boundary("left", "e", Boundary.flux(1e16))`. Resolve the Debye length with the
cells: with an unresolved sheath Newton's method cannot take long implicit steps.

## A 2D driver plasma

`Fluid2D` is the driver plasma on the cells of the inductive field's grid: electrons, ions
with inertia, the electron energy and a quasi-neutral potential with sheath walls. Here
an infinitely long driver (a plasma column in its solenoid, periodic along `z`), heated by
its field at 25 kW absorbed:

```python
import numpy as np
from simgeometry import Rectangle

from wellbalanced.driver import illustrative_hydrogen_driver
from wellbalanced.fluid1d import FluidSpecies
from wellbalanced.spatial import (
    CellGrid,
    DriverCoupling,
    Fluid2D,
    InductiveField2D,
    Winding,
    fluid_plasma,
)

driver = illustrative_hydrogen_driver()  # placeholder rates: it warns
R, L = driver.radius, driver.length
grid = CellGrid(
    np.r_[np.linspace(0, R, 25), np.linspace(R, 0.2, 12)[1:]],
    [0.0, L],
    geometry="cylindrical",
    boundaries=("neumann", "periodic"),
)
ions = [s for s in driver.species if s.charge > 0]
fluid = Fluid2D(
    grid,
    [FluidSpecies(driver.species[0], 1.0)] + [FluidSpecies(s, 500.0) for s in ions],
    driver.reactions,
    region=grid.xx < R,
    fixed=driver.fixed,
    gas_temperature=0.08,
    momentum_transfer=driver.momentum_transfer,  # nu of the electrons, also the field's
    ion_mean_free_path=driver.ion_mean_free_path,
)
coil = Winding(Rectangle((driver.coil_radius, L / 2), (0.004, 1.0)), sign=6)


def field(profile):
    return InductiveField2D(
        grid=grid, frequency=1e6, windings=[coil], **profile.settings()
    )


start = fluid.pack({"H+": 5e17, "H2+": 1e18, "H3+": 5e17}, 5.0, 25.0)
state = DriverCoupling(
    field=field,
    plasma=fluid_plasma(fluid, field, start, lorentz=False),
    network_resistance=0.6,
).at_absorbed_power(25e3)
print(state.coil_current, state.efficiency, fluid.energy_balance(state.plasma.state))
```

`lorentz=True` (the default) adds the RF Lorentz force, which pushes the electrons away
from the coil; `ramp` and `relaxation` take a strong force in gradually. At tens of kW in
a driver the force exceeds the plasma pressure at the edge, and the fixed point with it
does not converge yet. Conservative transport on refined patches waits for patchwork's
refluxing: `Fluid2D` runs on a tensor `CellGrid` or a single-level `PatchGrid`. The full example is
[the BUG driver in r–z](/wellbalanced/full-examples/rf-driver-fluid2d/).

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

The gas need not be uniform along the channel: a `GasProfile` gives its density for a feed,
from Knudsen diffusion or from a kinetic run (picnic's `NeutralParticleTransport`, as arrays):

```python
from plasmacoll.constants import ATOMIC_MASS
from wellbalanced.neutralizer import GasProfile

gas = GasProfile.knudsen(3.0, 0.1, 1.7, 4.028 * ATOMIC_MASS, inflow=2e21, temperature=0.03)
result = illustrative_deuterium_neutralizer(gas_profile=gas).solve()
print(result.line_density, result.neutral_fraction, result.plasma_potential)
```

See the [API reference](/wellbalanced/api/wellbalanced/) for every class and function.
