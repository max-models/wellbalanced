"""The two grids of the spatial models, for tests that run on both.

``"tensor"``: the cell edges as a :class:`~wellbalanced.spatial.CellGrid`;
``"patches"``: a :class:`~wellbalanced.spatial.PatchGrid` about as fine
(:meth:`~wellbalanced.spatial.PatchGrid.from_edges`), skipped without patchwork.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from wellbalanced.spatial import InductiveField2D, PatchGrid

GRIDS = ["tensor", "patches"]


def needs_patchwork(kind: str) -> None:
    """Skip a patch-grid case without patchwork."""
    if kind == "patches":
        pytest.importorskip("patchwork")


def patch_grid(
    x_edges: Any,
    y_edges: Any,
    *,
    geometry: str = "cartesian",
    boundaries: tuple[str, str] = ("dirichlet", "dirichlet"),
) -> PatchGrid:
    """The patches of :meth:`PatchGrid.from_edges`."""
    pytest.importorskip("patchwork")
    return PatchGrid.from_edges(
        x_edges,
        y_edges,
        geometry=geometry,  # type: ignore[arg-type]
        boundaries=boundaries,  # type: ignore[arg-type]
    )


def field(kind: str, x_edges: Any, y_edges: Any, **settings: Any) -> InductiveField2D:
    """The field model on the tensor grid of the edges or on patches like it."""
    if kind == "tensor":
        return InductiveField2D(x_edges, y_edges, **settings)
    geometry = settings.pop("geometry", "cartesian")
    boundaries = settings.pop("boundaries", ("dirichlet", "dirichlet"))
    grid = patch_grid(x_edges, y_edges, geometry=geometry, boundaries=boundaries)
    return InductiveField2D(grid=grid, **settings)


def value_at(solution: Any, array: np.ndarray, x: float, y: float):
    """The value of ``array`` in the cell at ``(x, y)``, and that cell's centre."""
    index = solution.grid.locate(x, y)
    return array[index], solution.grid.xx[index], solution.grid.yy[index]
