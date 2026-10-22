"""Regions of the plane, marked on the cells of a grid (tensor or patches).

A :class:`PlasmaRegion` is where the plasma is (the driver and the expansion region,
say): the union of its shapes minus the excluded ones. The spatial models set the plasma
to zero outside it; a fluid model (``Fluid2D``) will take its cells as unknowns and its
edges as walls. Shapes are anything with ``contains_points`` (simgeometry shapes), as
everywhere in :mod:`wellbalanced.spatial`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from wellbalanced.spatial.grid import CellGrid
    from wellbalanced.spatial.patches import PatchGrid

__all__ = ["PlasmaRegion"]


@dataclass(frozen=True)
class PlasmaRegion:
    """The cells of a plasma: the union of ``shapes`` minus the union of ``excluded``.

    Attributes:
        shapes: Shapes whose union is the region (e.g. the driver and the expansion
            region as two rectangles, a T in r–z).
        excluded: Shapes cut out of it (e.g. an internal structure).
    """

    shapes: Sequence[Any]
    excluded: Sequence[Any] = ()

    def __post_init__(self) -> None:
        """Check that there is a shape."""
        if not self.shapes:
            raise ValueError("a plasma region needs at least one shape")

    def mask(self, grid: CellGrid | PatchGrid) -> np.ndarray:
        """The cells whose centre is in the region, shape ``grid.shape``."""
        inside = grid.union_mask(self.shapes)
        if self.excluded:
            inside &= ~grid.union_mask(self.excluded)
        return inside

    def wall_faces(self, grid: CellGrid | PatchGrid) -> np.ndarray:
        """The region's cells that have a neighbour outside it (or the domain edge).

        The cells next to a wall, where a fluid model applies its wall conditions; the
        axis of an r–z domain and periodic seams are not walls.
        """
        return grid.wall_cells(self.mask(grid))
