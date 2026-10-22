"""The tensor-product cell grid of the spatial models, in an x–y cut or in r–z.

:class:`CellGrid` is the only place where the spatial models of
:mod:`wellbalanced.spatial` meet the geometry: the cell edges and centres, the
cross-section areas, volumes and face areas of the cells, the metric of the
geometry, the masks of shapes and the finite-volume operators. The solvers use
nothing else, so another grid (a non-uniform cell grid of simgeometry, once it has
one) can replace it in this module alone.

Two geometries:

- ``"cartesian"``: a cut ``(x, y)``, uniform along ``z``; volumes and powers are per
  unit length along ``z``.
- ``"cylindrical"``: an axisymmetric ``(r, z)`` plane, the first axis the radius (its
  first edge is 0 for a domain on the axis); a cell is a ring of volume
  ``2 pi r dr dz``, and volumes and powers are totals.

The operator of both is the curl-curl of a field ``a`` normal to the plane (``a_z`` in
the cut, ``a_phi`` in r–z, which has no divergence): ``curl curl a = -div grad a`` in
the cut and ``-(laplacian - 1 / r^2) a_phi`` in r–z. It is assembled from the energy
``sum_faces c_f (w_+ a_+ - w_- a_-)^2``, the in-plane curl of ``a`` on the faces times
its dual volume, so it is symmetric and positive: ``w = 1`` and
``c = face length / centre distance`` in the cut; across a radial face
``w = r`` (the curl is ``(1 / r) d(r a) / dr``) and ``c = 2 pi dz / (r_f dr)``, across an
axial face ``w = 1`` and ``c = 2 pi r dr / dz``. On the axis ``a_phi = 0``: the half cell
between the axis and the first centre, where ``a_phi`` grows linearly, adds ``4 pi dz a^2``
(without it the ``1 / r`` field, which has no curl, is nearly free near the axis).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

__all__ = ["AxisBoundary", "CellField", "CellGrid", "Geometry", "stretched_edges"]

Geometry = Literal["cartesian", "cylindrical"]
AxisBoundary = Literal["dirichlet", "neumann", "periodic"]
#: A spatial field given as a number, an array on the cells, or a function of the
#: cell-centre coordinate arrays (``(X, Y)`` in the cut, ``(R, Z)`` in r–z).
CellField = float | np.ndarray | Callable[[np.ndarray, np.ndarray], np.ndarray]


def stretched_edges(
    lower: float,
    upper: float,
    fine_lower: float,
    fine_upper: float,
    spacing: float,
    ratio: float = 1.1,
) -> np.ndarray:
    """Cell edges of one axis, fine in the middle and stretched toward the ends.

    Uniform ``spacing`` on ``[fine_lower, fine_upper]``, growing by ``ratio`` per cell
    toward ``lower`` and ``upper``. The outermost cells are shortened to end at
    ``lower`` and ``upper``.
    """
    if not lower <= fine_lower < fine_upper <= upper:
        raise ValueError("need lower <= fine_lower < fine_upper <= upper")
    count = max(1, round((fine_upper - fine_lower) / spacing))
    core = np.linspace(fine_lower, fine_upper, count + 1)

    def outward(start: float, end: float) -> list[float]:
        edges, step, position = [], spacing, start
        direction = 1.0 if end > start else -1.0
        while direction * (end - position) > 1e-12:
            step *= ratio
            position = position + direction * step
            if direction * (end - position) < 0.5 * step:
                position = end
            edges.append(position)
        return edges

    left = outward(fine_lower, lower)[::-1]
    right = outward(fine_upper, upper)
    return np.asarray([*left, *core, *right])


class CellGrid:
    """Cells of a tensor-product grid with any edges, in a cut or in r–z.

    Args:
        x_edges: Cell edges of the first axis (``x``, or ``r`` from ``>= 0``) [m].
        y_edges: Cell edges of the second axis (``y``, or ``z``) [m].
        geometry: ``"cartesian"`` or ``"cylindrical"``.
        boundaries: The boundary of each axis: ``"dirichlet"`` (the field is 0 at
            the edge), ``"neumann"`` (no flux) or ``"periodic"``. In r–z the radial
            axis cannot be periodic, and its lower boundary is the axis when the
            first edge is 0, whatever is given.
    """

    def __init__(
        self,
        x_edges: Sequence[float] | np.ndarray,
        y_edges: Sequence[float] | np.ndarray,
        *,
        geometry: Geometry = "cartesian",
        boundaries: tuple[AxisBoundary, AxisBoundary] = ("dirichlet", "dirichlet"),
    ) -> None:
        """Check the edges and compute the cell geometry."""
        self.x_edges = np.asarray(x_edges, dtype=float)
        self.y_edges = np.asarray(y_edges, dtype=float)
        for edges in (self.x_edges, self.y_edges):
            if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError("cell edges must be increasing, at least two")
        if geometry not in ("cartesian", "cylindrical"):
            raise ValueError(f"unknown geometry {geometry!r}")
        for boundary in boundaries:
            if boundary not in ("dirichlet", "neumann", "periodic"):
                raise ValueError(f"unknown boundary {boundary!r}")
        if geometry == "cylindrical":
            if self.x_edges[0] < 0.0:
                raise ValueError("radial edges must start at r >= 0")
            if boundaries[0] == "periodic":
                raise ValueError("the radius cannot be periodic")
        self.geometry: Geometry = geometry
        self.boundaries = tuple(boundaries)
        self.x = 0.5 * (self.x_edges[1:] + self.x_edges[:-1])
        self.y = 0.5 * (self.y_edges[1:] + self.y_edges[:-1])
        self.shape = (self.x.size, self.y.size)
        self.size = self.shape[0] * self.shape[1]
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")
        #: Cross-section areas of the cells in the plane [m^2].
        self.cell_areas = np.outer(np.diff(self.x_edges), np.diff(self.y_edges))
        #: The length a cell extends out of the plane: 1 m in the cut, the
        #: circumference ``2 pi r`` in r–z.
        self.circumference = (
            np.ones(self.shape)
            if geometry == "cartesian"
            else np.broadcast_to(2.0 * np.pi * self.x[:, None], self.shape).copy()
        )
        #: Cell volumes [m^3] (per metre along ``z`` in the cut).
        self.cell_volumes = self.cell_areas * self.circumference

    @property
    def cylindrical(self) -> bool:
        """Whether the grid is the axisymmetric r–z plane."""
        return self.geometry == "cylindrical"

    @property
    def coordinate_names(self) -> tuple[str, str]:
        """``("x", "y")`` or ``("r", "z")``."""
        return ("r", "z") if self.cylindrical else ("x", "y")

    # --- fields and regions -----------------------------------------------------

    def cell_field(self, value: CellField, dtype: Any = float) -> np.ndarray:
        """A :data:`CellField` evaluated on the cell centres, shape :attr:`shape`."""
        if callable(value):
            result = np.asarray(value(self.xx, self.yy), dtype=dtype)
        else:
            result = np.asarray(value, dtype=dtype)
        return np.broadcast_to(result, self.shape).copy()

    def mask(self, shape: Any) -> np.ndarray:
        """The cells whose centre a shape (anything with ``contains_points``) contains.

        ``contains_points`` takes the ``(2, n)`` centres; a CuPy result is copied to
        the host.
        """
        points = np.stack([self.xx.ravel(), self.yy.ravel()])
        inside = shape.contains_points(points)
        if hasattr(inside, "get"):
            inside = inside.get()
        return np.asarray(inside, dtype=bool).reshape(self.shape)

    def union_mask(self, shapes: Sequence[Any]) -> np.ndarray:
        """The cells any of ``shapes`` contains."""
        result = np.zeros(self.shape, dtype=bool)
        for shape in shapes:
            result |= self.mask(shape)
        return result

    # --- operators ----------------------------------------------------------------

    def _faces(self, axis: int):
        """Interior faces of ``axis``: neighbour index pairs, ``c`` and weights.

        Every entry is an array over the other axis, one per face plane; the periodic
        wrap is the last plane.
        """
        index = np.moveaxis(np.arange(self.size).reshape(self.shape), axis, 0)
        centres = (self.x, self.y)[axis]
        edges = (self.x_edges, self.y_edges)[axis]
        n = centres.size
        planes = []
        for i in range(n - 1):
            distance = centres[i + 1] - centres[i]
            planes.append((index[i], index[i + 1], edges[i + 1], distance))
        if self.boundaries[axis] == "periodic" and n > 1:
            wrap = 0.5 * (edges[1] - edges[0] + edges[-1] - edges[-2])
            planes.append((index[n - 1], index[0], None, wrap))
        return [
            (left, right, *self._metric(axis, face, distance, left, right))
            for left, right, face, distance in planes
        ]

    def _metric(self, axis: int, face: float | None, distance: float, left, right):
        """``c`` and the weights ``(w_-, w_+)`` of the faces between two cell planes."""
        dx, dy = np.diff(self.x_edges), np.diff(self.y_edges)
        ones = np.ones(left.shape)
        if not self.cylindrical:
            length = dy if axis == 0 else dx
            return length / distance, ones, ones
        if axis == 1:  # axial faces at the cells' radius
            return 2.0 * np.pi * self.x * dx / distance, ones, ones
        # radial faces: the curl is (1 / r) d(r a) / dr
        r_left, r_right = self.x[left // self.shape[1]], self.x[right // self.shape[1]]
        assert face is not None  # the radius is never periodic
        return 2.0 * np.pi * dy / (face * distance), r_left, r_right

    @property
    def on_axis(self) -> bool:
        """Whether the r–z domain reaches the axis (its first radial edge is 0)."""
        return self.cylindrical and self.x_edges[0] == 0.0

    def _boundary_faces(self, axis: int):
        """Dirichlet edges (and the axis) of ``axis``: (cells, ``c``, weight) per end.

        The axis: ``a_phi`` grows linearly from 0 to the first centre, so the half
        cell has ``B_z = 2 a / r_0`` and the energy ``4 pi dz a^2`` (``c = 4 pi dz``,
        ``w = 1``); with it a uniform ``B_z`` stays a null field of the operator.
        """
        index = np.moveaxis(np.arange(self.size).reshape(self.shape), axis, 0)
        dx, dy = np.diff(self.x_edges), np.diff(self.y_edges)
        result = []
        if axis == 0 and self.on_axis:
            result.append((index[0], 4.0 * np.pi * dy, np.ones(dy.shape)))
        if self.boundaries[axis] != "dirichlet":
            return result
        centres = (self.x, self.y)[axis]
        edges = (self.x_edges, self.y_edges)[axis]
        for cells, edge, centre in (
            (index[0], edges[0], centres[0]),
            (index[-1], edges[-1], centres[-1]),
        ):
            distance = abs(edge - centre)
            if not self.cylindrical:
                c = (dy if axis == 0 else dx) / distance
                result.append((cells, c, np.ones(cells.shape)))
            elif axis == 1:
                result.append((cells, 2.0 * np.pi * self.x * dx / distance, 1.0))
            elif edge > 0.0:  # a radial wall, not the axis
                c = 2.0 * np.pi * dy / (edge * distance)
                result.append((cells, c, np.full(cells.shape, centre)))
        return result

    def curl_curl(
        self,
        face_coefficient: np.ndarray | None = None,
        active: np.ndarray | None = None,
    ) -> sp.csr_matrix:
        """The curl-curl of the normal field, integrated over the cells (symmetric).

        ``(K a)_i`` is ``V_i (curl curl a)_i`` for the field ``a`` on the cells, zero at
        the Dirichlet edges. ``face_coefficient`` (a cell field) scales each face by the
        mean of its two cells, the coefficient of ``curl (mu curl a)``, e.g. a
        viscosity. With ``active``, only those cells are unknowns: a face to an
        inactive cell becomes a Dirichlet edge (``a = 0``) at the face, and the rows
        and columns of inactive cells are zero.
        """
        mu = (
            np.ones(self.size)
            if face_coefficient is None
            else np.broadcast_to(face_coefficient, self.shape).ravel()
        )
        live = np.ones(self.size, dtype=bool) if active is None else active.ravel()
        rows, cols, values = [], [], []
        diagonal = np.zeros(self.size)
        for axis in (0, 1):
            centres = (self.x, self.y)[axis]
            edges = (self.x_edges, self.y_edges)[axis]
            for left, right, c, w_l, w_r in self._faces(axis):
                left, right = left.ravel(), right.ravel()
                c = np.broadcast_to(c, left.shape)
                w_l, w_r = (
                    np.broadcast_to(w_l, left.shape),
                    np.broadcast_to(w_r, left.shape),
                )
                both = live[left] & live[right]
                c_face = c * 0.5 * (mu[left] + mu[right])
                lower, upper, cf = left[both], right[both], c_face[both]
                coupling = -cf * w_l[both] * w_r[both]
                rows += [lower, upper]
                cols += [upper, lower]
                values += [coupling, coupling]
                np.add.at(diagonal, lower, cf * w_l[both] ** 2)
                np.add.at(diagonal, upper, cf * w_r[both] ** 2)
                if active is None:
                    continue
                # one side active: the field vanishes on the face between them
                for cells, other, weight, side in (
                    (left, right, w_l, 0),
                    (right, left, w_r, 1),
                ):
                    edge = live[cells] & ~live[other]
                    if not edge.any():
                        continue
                    cell = cells[edge]
                    half = self._half_metric(axis, cell, side, centres, edges)
                    np.add.at(diagonal, cell, mu[cell] * half * weight[edge] ** 2)
            for cells, c, weight in self._boundary_faces(axis):
                cells = cells.ravel()
                c = np.broadcast_to(c, cells.shape)
                weight = np.broadcast_to(weight, cells.shape)
                keep = live[cells]
                np.add.at(
                    diagonal,
                    cells[keep],
                    mu[cells[keep]] * c[keep] * weight[keep] ** 2,
                )
        diagonal[~live] = 0.0
        matrix = (
            sp.coo_matrix(
                (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.size, self.size),
            )
            if rows
            else sp.coo_matrix((self.size, self.size))
        )
        return (matrix + sp.diags(diagonal)).tocsr()

    def _half_metric(self, axis: int, cells, side: int, centres, edges) -> np.ndarray:
        """``c`` of the face of ``cells`` toward their neighbour on ``side`` (0: +)."""
        i, j = np.divmod(cells, self.shape[1])
        k = (i, j)[axis]
        face = edges[k + 1] if side == 0 else edges[k]
        distance = np.abs(face - centres[k])
        dx, dy = np.diff(self.x_edges), np.diff(self.y_edges)
        if not self.cylindrical:
            return (dy[j] if axis == 0 else dx[i]) / distance
        if axis == 1:
            return 2.0 * np.pi * self.x[i] * dx[i] / distance
        return 2.0 * np.pi * dy[j] / (face * distance)

    def curl(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The in-plane curl of the normal field ``a`` on the cell centres.

        ``(d_y a, -d_x a)`` in the cut; ``(-d_z a, (1 / r) d(r a) / dr)`` in r–z.
        """
        nx, ny = self.shape
        if not self.cylindrical:
            first = np.gradient(a, self.y, axis=1) if ny > 1 else np.zeros_like(a)
            second = -np.gradient(a, self.x, axis=0) if nx > 1 else np.zeros_like(a)
            return first, second
        b_r = -np.gradient(a, self.y, axis=1) if ny > 1 else np.zeros_like(a)
        # (1/r) d(r a)/dr = a / r + da/dr: exact for a uniform B_z next to the axis too
        da_dr = np.gradient(a, self.x, axis=0) if nx > 1 else np.zeros_like(a)
        return b_r, a / self.x[:, None] + da_dr

    def net_current(self, residual: np.ndarray, region: np.ndarray) -> Any:
        """The current through a region's cross-section from ``V J`` on its cells."""
        return np.sum(residual[region] / self.circumference[region])

    def coordinates(self) -> dict[str, tuple[str, np.ndarray, dict[str, str]]]:
        """The cell-centre coordinates as xarray coordinate entries."""
        first, second = self.coordinate_names
        return {
            first: (first, self.x, {"long_name": first, "units": "m"}),
            second: (second, self.y, {"long_name": second, "units": "m"}),
        }
