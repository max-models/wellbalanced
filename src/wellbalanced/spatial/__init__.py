"""Spatially resolved (2D) models: the inductive RF field of a driver, in a cut or in r–z.

Everything that resolves the plane lives here, on one cell grid
(a tensor :class:`~wellbalanced.spatial.grid.CellGrid`, or the refined patches of a
:class:`~wellbalanced.spatial.patches.PatchGrid`; the only place the geometry and its
metric enter): the magnetoquasistatic field of coil windings, conductors and a
prescribed plasma (:mod:`~wellbalanced.spatial.inductive`), the plasma region
(:mod:`~wellbalanced.spatial.regions`) and the coupling of the field to a
power-dependent plasma (:mod:`~wellbalanced.spatial.coupling`). ``Fluid2D`` and the
neutral fluid of the RF-driver plan (step R3) will join them, and the subpackage is the
unit a later split of wellbalanced moves as a whole (the magnetron's
:class:`~wellbalanced.DriftFluid2D` is a candidate to join it then).

Shapes are anything with ``contains_points`` (simgeometry shapes, an optional
dependency: ``pip install "wellbalanced[spatial]"``); this subpackage does not import
simgeometry. patchwork, for :class:`~wellbalanced.spatial.patches.PatchGrid`, is in the
same extra and imported when a patch grid is made.
"""

from wellbalanced.spatial.coupling import (
    CoupledState,
    DriverCoupling,
    PlasmaProfile,
    global_model_plasma,
)
from wellbalanced.spatial.grid import CellGrid, stretched_edges
from wellbalanced.spatial.inductive import (
    Conductor,
    InductiveField2D,
    InductiveSolution,
    InductiveState,
    InductiveTimeStepper,
    Winding,
)
from wellbalanced.spatial.patches import PatchGrid
from wellbalanced.spatial.regions import PlasmaRegion

__all__ = [
    "CellGrid",
    "Conductor",
    "CoupledState",
    "DriverCoupling",
    "InductiveField2D",
    "InductiveSolution",
    "InductiveState",
    "InductiveTimeStepper",
    "PatchGrid",
    "PlasmaProfile",
    "PlasmaRegion",
    "Winding",
    "global_model_plasma",
    "stretched_edges",
]
