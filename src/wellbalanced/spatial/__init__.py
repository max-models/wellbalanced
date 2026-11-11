"""Spatially resolved models: the inductive RF field of a driver, in a cut, in r–z or in 3D.

Everything that resolves the plane lives here, on one cell grid
(a tensor `wellbalanced.spatial.grid.CellGrid`, or the refined patches of a
`wellbalanced.spatial.patches.PatchGrid`; the only place the geometry and its
metric enter): the magnetoquasistatic field of coil windings, conductors and a
prescribed plasma (`wellbalanced.spatial.inductive`), the plasma region
(`wellbalanced.spatial.regions`) and the coupling of the field to a
power-dependent plasma (`wellbalanced.spatial.coupling`), and the driver plasma as a
multi-species fluid heated by that field (`wellbalanced.spatial.fluid`,
`Fluid2D`). The 3D field (`wellbalanced.spatial.inductive3d`, step R4) is on
the edges of a Cartesian `wellbalanced.spatial.edges.EdgeGrid`, or of refined
patches (`wellbalanced.spatial.patch_edges.PatchEdgeGrid`, patchwork's edge space). The neutral gas
(H and H2 as two Navier–Stokes fluids, step R3.5: `wellbalanced.spatial.neutrals`,
`NeutralFluid2D`) is on the plasma's cells and coupled to it by
`couple_neutrals`, and the subpackage is the unit a later
split of wellbalanced moves as a whole (the magnetron's
`wellbalanced.DriftFluid2D` is a candidate to join it then).

Shapes are anything with ``contains_points`` (simgeometry shapes, an optional
dependency: ``pip install "wellbalanced[spatial]"``); simgeometry is imported only for the
cut cells of surface-impedance conductors (`wellbalanced.spatial.cutcell`). patchwork, for `wellbalanced.spatial.patches.PatchGrid`, is in the
same extra and imported when a patch grid is made.
"""

from wellbalanced.spatial.coupling import (
    CoupledState,
    DriverCoupling,
    FluidPlasma,
    PlasmaProfile,
    fluid_plasma,
    global_model_plasma,
)
from wellbalanced.spatial.edges import EdgeGrid
from wellbalanced.spatial.fluid import Fluid2D, Fluid2DState
from wellbalanced.spatial.grid import CellGrid, TransportFaces, stretched_edges
from wellbalanced.spatial.inductive import (
    Conductor,
    InductiveField2D,
    InductiveSolution,
    InductiveState,
    InductiveTimeStepper,
    Winding,
)
from wellbalanced.spatial.inductive3d import (
    Coil3D,
    Conductor3D,
    Feed3D,
    ImpedanceConductor3D,
    InductiveField3D,
    InductiveSolution3D,
)
from wellbalanced.spatial.neutrals import (
    Collision,
    CoupledNeutrals,
    Inlet,
    NeutralFluid2D,
    NeutralSpecies,
    NeutralState,
    Opening,
    couple_neutrals,
    hard_sphere,
)
from wellbalanced.spatial.patch_edges import PatchEdgeGrid
from wellbalanced.spatial.patches import PatchGrid
from wellbalanced.spatial.regions import PlasmaRegion

__all__ = [
    "CellGrid",
    "Coil3D",
    "Collision",
    "Conductor",
    "Conductor3D",
    "CoupledNeutrals",
    "CoupledState",
    "DriverCoupling",
    "EdgeGrid",
    "Feed3D",
    "ImpedanceConductor3D",
    "Fluid2D",
    "Fluid2DState",
    "FluidPlasma",
    "Inlet",
    "InductiveField2D",
    "InductiveField3D",
    "InductiveSolution",
    "InductiveSolution3D",
    "InductiveState",
    "InductiveTimeStepper",
    "NeutralFluid2D",
    "NeutralSpecies",
    "NeutralState",
    "Opening",
    "PatchEdgeGrid",
    "PatchGrid",
    "PlasmaProfile",
    "PlasmaRegion",
    "TransportFaces",
    "Winding",
    "couple_neutrals",
    "fluid_plasma",
    "global_model_plasma",
    "hard_sphere",
    "stretched_edges",
]
