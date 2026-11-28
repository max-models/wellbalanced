"""Global (0D) and 1D fluid models of low-temperature plasmas.

A global model balances, averaged over a chamber, the density of every species
and the electron energy (and optionally the gas temperature): the reactions,
the losses to the walls, the feed and pumping of the gases and external
sources such as a fast beam. Reactions take their rates from plasmacoll's
cross sections, the same data its Monte Carlo collisions use, so a fluid and a
kinetic model of the same plasma differ in their physics, not their data.

`Fluid1D` resolves the same plasma along one dimension: drift-diffusion of
every species, the electron energy and Poisson's equation, implicit, with
boundaries that can be changed between steps.

`GasDynamics` is the neutral gas of a sputtering magnetron, argon and the sputtered
metal as one compressible gas in one or two dimensions (its boundaries and sources are in
`carafe.gasdynamics`).

`RFDriver` is a 0D RF driver: a global model of its plasma and a transformer
model of the coil's coupling, giving the coil current and the RF power transfer efficiency
(`carafe.driver`). The spatially resolved models are in
`carafe.spatial`: `InductiveField2D`, the inductive RF field of coil
windings, conductors and a prescribed plasma in a 2D cut or in r–z, with cold or viscous
electrons, `DriverCoupling`, that field with a power-dependent plasma, and
`InductiveField3D`, the field in 3D on the edges of a Cartesian grid.
`Fluid2D`, the driver plasma as a multi-species fluid heated by that field.

`HallThruster1D` is the 1D axial model of a Hall thruster: neutrals, ions with
momentum and quasineutral electrons with an anomalous collision frequency and an
energy equation with wall losses, the discharge current from the voltage
(`carafe.hall`).

`MHD2D` is ideal and resistive MHD in the plane (three velocity and field components) on the patches
of a patchwork hierarchy, the in-plane field from a nodal vector potential so that its
divergence is zero to round-off, refined patches included (`carafe.mhd`). `MHD3D` is the same
in three dimensions, the vector potential on the cell edges (`carafe.mhd3d`).

The solves, time steps and runs carry scope-profiler regions tagged ``"carafe"``,
an opt-in tag: they record nothing unless an application enables it, e.g. with
``scope_profiler.setup(enable_tags=["carafe"])``.
"""

import scope_profiler

from carafe._profiling import _PROFILE_TAGS
from carafe.datasets import stack
from carafe.driftfluid2d import (
    DriftFluid2D,
    DriftFluid2DState,
    SpokeReport,
    spoke_report,
)
from carafe.driver import Coil, DriverState, RFDriver
from carafe.fluid1d import (
    Boundary,
    Fluid1D,
    FluidSpecies,
    FluidState,
    PotentialBoundary,
)
from carafe.gas_coupling import GasCoupledDriftFluid
from carafe.gasdynamics import GasDynamics, GasState
from carafe.global_model import (
    Chamber,
    Cusp,
    GasTemperature,
    GlobalModel,
    GlobalState,
    VolumeSource,
    WallReaction,
)
from carafe.hall import HallState, HallThruster1D
from carafe.irm import IonizationRegion
from carafe.magnetron import MATERIALS, Magnetron, TargetMaterial
from carafe.mhd import MHD2D
from carafe.mhd3d import MHD3D
from carafe.reactions import Reaction
from carafe.spatial import (
    CellGrid,
    Conductor,
    DriverCoupling,
    Fluid2D,
    InductiveField2D,
    InductiveField3D,
    InductiveSolution,
    PlasmaRegion,
    Winding,
)
from carafe.species import ELECTRON, Species

scope_profiler.register_opt_in_tag(*_PROFILE_TAGS)

__version__ = "0.1.0"

__all__ = [
    "ELECTRON",
    "Boundary",
    "CellGrid",
    "Chamber",
    "Coil",
    "Conductor",
    "Cusp",
    "DriftFluid2D",
    "DriftFluid2DState",
    "DriverCoupling",
    "DriverState",
    "Fluid1D",
    "Fluid2D",
    "FluidSpecies",
    "FluidState",
    "GasCoupledDriftFluid",
    "GasDynamics",
    "GasState",
    "GasTemperature",
    "GlobalModel",
    "GlobalState",
    "HallState",
    "HallThruster1D",
    "InductiveField2D",
    "InductiveField3D",
    "InductiveSolution",
    "IonizationRegion",
    "MHD2D",
    "MHD3D",
    "MATERIALS",
    "Magnetron",
    "PlasmaRegion",
    "PotentialBoundary",
    "RFDriver",
    "Reaction",
    "Species",
    "SpokeReport",
    "TargetMaterial",
    "VolumeSource",
    "WallReaction",
    "Winding",
    "spoke_report",
    "stack",
]
