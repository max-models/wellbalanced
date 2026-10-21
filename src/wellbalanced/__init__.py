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

:class:`InductiveField2D` is the inductive RF field of coil windings, conductors and a
prescribed plasma in a 2D cut (:mod:`wellbalanced.inductive`).
"""

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
from wellbalanced.inductive import (
    Conductor,
    InductiveField2D,
    InductiveSolution,
    Winding,
)
from wellbalanced.magnetron import MATERIALS, Magnetron, TargetMaterial
from wellbalanced.reactions import Reaction
from wellbalanced.species import ELECTRON, Species

__version__ = "0.1.0"

__all__ = [
    "ELECTRON",
    "Boundary",
    "Chamber",
    "Conductor",
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
    "InductiveField2D",
    "InductiveSolution",
    "MATERIALS",
    "Magnetron",
    "PotentialBoundary",
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
