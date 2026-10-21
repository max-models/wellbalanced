"""Global (0D) and 1D fluid models of low-temperature plasmas.

A global model balances, averaged over a chamber, the density of every species
and the electron energy (and optionally the gas temperature): the reactions,
the losses to the walls, the feed and pumping of the gases and external
sources such as a fast beam. Reactions take their rates from plasmacoll's
cross sections, the same data its Monte Carlo collisions use, so a fluid and a
kinetic model of the same plasma differ in their physics, not their data.

:class:`Fluid1D` resolves the same plasma along one dimension: drift-diffusion of
every species, the electron energy and Poisson's equation, implicit, with
boundaries that can be changed between steps.

:class:`GasDynamics` is the neutral gas of a sputtering magnetron, argon and the sputtered
metal as one compressible gas in one or two dimensions (its boundaries and sources are in
:mod:`wellbalanced.gasdynamics`).

The solves, time steps and runs carry scope-profiler regions tagged ``"wellbalanced"``,
an opt-in tag: they record nothing unless an application enables it, e.g. with
``scope_profiler.setup(enable_tags=["wellbalanced"])``.
"""

import scope_profiler

from wellbalanced._profiling import _PROFILE_TAGS
from wellbalanced.datasets import stack
from wellbalanced.driftfluid2d import (
    DriftFluid2D,
    DriftFluid2DState,
    SpokeReport,
    spoke_report,
)
from wellbalanced.fluid1d import (
    Boundary,
    Fluid1D,
    FluidSpecies,
    FluidState,
    PotentialBoundary,
)
from wellbalanced.gasdynamics import GasDynamics, GasState
from wellbalanced.global_model import (
    Chamber,
    Cusp,
    GasTemperature,
    GlobalModel,
    GlobalState,
    VolumeSource,
    WallReaction,
)
from wellbalanced.magnetron import MATERIALS, Magnetron, TargetMaterial
from wellbalanced.reactions import Reaction
from wellbalanced.species import ELECTRON, Species

scope_profiler.register_opt_in_tag(*_PROFILE_TAGS)

__version__ = "0.1.0"

__all__ = [
    "ELECTRON",
    "Boundary",
    "Chamber",
    "Cusp",
    "DriftFluid2D",
    "DriftFluid2DState",
    "Fluid1D",
    "FluidSpecies",
    "FluidState",
    "GasDynamics",
    "GasState",
    "GasTemperature",
    "GlobalModel",
    "GlobalState",
    "MATERIALS",
    "Magnetron",
    "PotentialBoundary",
    "Reaction",
    "Species",
    "SpokeReport",
    "TargetMaterial",
    "VolumeSource",
    "WallReaction",
    "spoke_report",
    "stack",
]
