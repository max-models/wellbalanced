"""Cut cells of a conductor on the edge grid: what of every edge and face is outside it.

The geometry of the surface-impedance conductors of
:class:`~wellbalanced.spatial.inductive3d.InductiveField3D`, in the numbering of an
:class:`~wellbalanced.spatial.edges.EdgeGrid`:

- ``edge_air``: the fraction of every edge's length outside the conductor;
- ``face_air``: the fraction of every face's area outside;
- ``face_width``: the width of the conductor's surface in every face, ``int dl / sin(theta)``
  over the curve where the surface crosses the face, ``theta`` the angle between the
  surface normal and the face's [m]. It is the weight of the surface-impedance term of
  the face (Dey and Mittra's contour through the cut face, with ``E_t = Z_s H x n`` on
  its part along the surface and ``H`` the face's normal component; as ``H`` is tangent
  to a good conductor, ``(H x n) . dl = H_f dl / sin(theta)``).

Two ways to get them:

- :func:`cut_cells`: conformal, from simgeometry's
  :func:`~simgeometry.fractions.yee_fractions` (exact for planes, second order for
  curved surfaces; a slit narrower than a cell keeps its gap). The width is
  ``trace^2 / projected trace`` (``trace / sin(theta)`` for a straight trace), with
  ``sin(theta)`` bounded below by ``min_sine``.
- :func:`staircased_cells`: a cell is metal when its centre is inside; the surface is the
  faces between metal and air cells. An edge or face touching a metal cell is in the
  metal, and an air face's width is the length of its edges on the surface.

simgeometry is imported by :func:`cut_cells` only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np

from wellbalanced.spatial.edges import EdgeGrid

__all__ = ["CutCells", "cut_cells", "staircased_cells"]


@dataclass(frozen=True)
class CutCells:
    """What of the edges and faces is outside a conductor, and its surface's width.

    Attributes:
        edge_air: The fraction of every edge's length outside, ``(n_edges,)``.
        face_air: The fraction of every face's area outside, ``(n_faces,)``.
        face_width: The surface's width in every face [m], ``(n_faces,)``.
    """

    edge_air: np.ndarray
    face_air: np.ndarray
    face_width: np.ndarray


def _simgeometry_grid(grid: EdgeGrid):
    import simgeometry as sg

    return sg.Grid(
        3,
        tuple(grid.lower),
        tuple(grid.upper),
        grid.nodes,
        periodic=tuple(bool(p) for p in grid.periodic),
    )


def cut_cells(
    grid: EdgeGrid,
    shapes: Sequence[Any],
    *,
    min_sine: float = 0.1,
    samples: int = 64,
) -> CutCells:
    """The conformal cut cells of the union of simgeometry ``shapes``.

    Args:
        grid: The edge grid.
        shapes: simgeometry shapes (3D); their union is the conductor.
        min_sine: The lower bound of ``sin(theta)`` in the width ``trace / sin(theta)``:
            a surface almost parallel to a face makes that face's normal field small
            through a large width; the bound keeps the matrix moderate.
        samples: Sampling of shapes without analytic crossings (simgeometry).
    """
    if not shapes:
        raise ValueError("a conductor needs at least one shape")
    union = reduce(lambda a, b: a | b, shapes)
    fractions = union.yee_fractions(_simgeometry_grid(grid), samples=samples)
    for a in range(3):
        if fractions.edges[a].shape != grid.edge_shape(a):
            raise ValueError("simgeometry's edge layout differs from the edge grid's")
    edge_air = 1.0 - np.concatenate([f.ravel() for f in fractions.edges])
    face_air = 1.0 - np.concatenate([f.ravel() for f in fractions.faces])
    trace = np.concatenate([f.ravel() for f in fractions.traces])
    projected = np.concatenate([f.ravel() for f in fractions.projected_traces])
    sine = np.where(trace > 0, projected / np.where(trace > 0, trace, 1.0), 1.0)
    width = trace / np.maximum(sine, min_sine)
    return CutCells(
        np.clip(edge_air, 0.0, 1.0), np.clip(face_air, 0.0, 1.0), np.maximum(width, 0)
    )


def staircased_cells(grid: EdgeGrid, shapes: Sequence[Any]) -> CutCells:
    """The staircased conductor: the cells whose centre one of ``shapes`` contains."""
    centres = np.stack(
        [
            g.ravel()
            for g in np.meshgrid(
                *[grid.cell_coordinates(a) for a in range(3)], indexing="ij"
            )
        ]
    )
    metal = np.zeros(centres.shape[1], dtype=bool)
    for shape in shapes:
        metal |= grid.contains(shape, centres)
    metal = metal.reshape(grid.cells)

    def touching(shape: tuple[int, ...], across: Sequence[int]) -> np.ndarray:
        """Whether an edge or face block (``shape``) touches a metal cell.

        Along the axes in ``across`` an entry sits on a node plane between two cells
        (the cell below and above; none beyond a non-periodic boundary); along the
        others it lies in a cell.
        """
        result = metal.copy()
        for b in across:
            below = np.roll(result, 1, axis=b)
            if grid.periodic[b]:
                result = result | below
            else:
                pad = [(0, 0)] * 3
                pad[b] = (0, 1)
                upper = np.pad(result, pad)
                lower = np.pad(result, [(1, 0) if k == b else (0, 0) for k in range(3)])
                result = upper | lower
        assert result.shape == shape
        return result

    edge_dead, face_dead = [], []
    for a in range(3):
        others = [b for b in range(3) if b != a]
        edge_dead.append(touching(grid.edge_shape(a), others).ravel())
        face_dead.append(touching(grid.face_shape(a), [a]).ravel())
    edge_dead = np.concatenate(edge_dead)
    face_dead = np.concatenate(face_dead)
    # an air face's width: the length of its edges on the surface (in the metal)
    curl = abs(grid.curl())
    width = curl @ (edge_dead * grid.edge_lengths)
    width[face_dead] = 0.0
    return CutCells(
        (~edge_dead).astype(float), (~face_dead).astype(float), np.asarray(width)
    )
