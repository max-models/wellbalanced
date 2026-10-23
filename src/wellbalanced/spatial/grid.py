"""The tensor-product cell grid of the spatial models, in an x–y cut or in r–z.

:class:`CellGrid` is the only place where the spatial models of
:mod:`wellbalanced.spatial` meet the geometry: the cell edges and centres, the
cross-section areas, volumes and face areas of the cells, the metric of the
geometry, the masks of shapes and the finite-volume operators. The solvers use
nothing else, so another grid can replace it: :class:`~wellbalanced.spatial.patches.PatchGrid`,
the leaf cells of a patchwork hierarchy, has the same interface (its fields are arrays
over the cells, shape ``(n,)``, where this grid's are ``(nx, ny)``).

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
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

__all__ = [
    "AxisBoundary",
    "CellField",
    "CellGrid",
    "Geometry",
    "TransportFaces",
    "stretched_edges",
]

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


@dataclass(frozen=True)
class TransportFaces:
    """The faces of a grid for finite-volume transport (a divergence, not a curl).

    Interior faces join the cells ``low`` and ``high`` (``high`` is the one further
    along ``axis``; across a periodic seam it is the first cell): ``area`` is the
    face area (the length in a cut, per metre along ``z``; ``2 pi r dz`` for a radial
    and ``pi (r1^2 - r0^2)`` for an axial face in r–z), ``delta_low`` and ``delta_high``
    the distances from the two cell centres to the face along ``axis``. Boundary faces
    of the domain's edges have their ``boundary_cell``, ``boundary_axis``,
    ``boundary_side`` (0 below the cell, 1 above), ``boundary_area`` and
    ``boundary_delta``; ``boundary_symmetry`` marks the edges of ``"neumann"`` axes. The
    axis of an r–z domain (no area) and periodic seams have no boundary face.

    Between leaf cells of different levels (a :class:`~wellbalanced.spatial.PatchGrid`)
    each face is as long as the finer cell's side, so a coarse cell has several faces
    on that side, and its centre is off the line through the finer centre normal to the
    face: ``coarse_side`` (per interior face: 0 if ``low`` is the coarser cell, 1 if
    ``high`` is, -1 between cells of one size) marks these faces, ``coarse_neighbours``
    are the coarser cell's neighbours of its own size along the face (below and above;
    -1 where there is none) and ``coarse_offset`` the distance of the face's middle from
    the coarser centre along the face, in the coarser cell's widths. A finite-volume
    scheme interpolates the coarse value along the face to the face's middle with them
    (a ghost value on the line of the finer centre). ``None`` on a tensor grid.
    """

    low: np.ndarray
    high: np.ndarray
    axis: np.ndarray
    area: np.ndarray
    delta_low: np.ndarray
    delta_high: np.ndarray
    boundary_cell: np.ndarray
    boundary_axis: np.ndarray
    boundary_side: np.ndarray
    boundary_area: np.ndarray
    boundary_delta: np.ndarray
    boundary_symmetry: np.ndarray
    coarse_side: np.ndarray | None = None
    coarse_neighbours: np.ndarray | None = None
    coarse_offset: np.ndarray | None = None

    @property
    def distance(self) -> np.ndarray:
        """The distance between the two cell centres of every interior face."""
        return self.delta_low + self.delta_high

    @classmethod
    def concatenate(
        cls,
        interior: Sequence[tuple[np.ndarray, ...]],
        boundary: Sequence[tuple[np.ndarray, ...]],
    ) -> TransportFaces:
        """Faces from lists of arrays, the faces of no area dropped.

        ``interior``: ``(low, high, axis, area, delta_low, delta_high)``, optionally
        followed by ``(coarse_side, coarse_neighbours, coarse_offset)``; ``boundary``:
        ``(cell, axis, side, area, delta, symmetry)``.
        """

        def stack(parts, k, dtype):
            if not parts:
                return np.zeros(0, dtype=dtype)
            return np.concatenate(
                [np.asarray(p[k], dtype=dtype).ravel() for p in parts]
            )

        inner = [
            stack(interior, k, t) for k, t in enumerate((int, int, int) + (float,) * 3)
        ]
        edge = [
            stack(boundary, k, t)
            for k, t in enumerate((int, int, int, float, float, bool))
        ]
        keep_inner, keep_edge = inner[3] > 0.0, edge[3] > 0.0
        levels: dict[str, np.ndarray] = {}
        if interior and len(interior[0]) > 6:
            levels = {
                "coarse_side": stack(interior, 6, int)[keep_inner],
                "coarse_neighbours": np.concatenate(
                    [np.asarray(p[7], dtype=int).reshape(-1, 2) for p in interior]
                )[keep_inner],
                "coarse_offset": stack(interior, 8, float)[keep_inner],
            }
        return cls(
            *(a[keep_inner] for a in inner), *(a[keep_edge] for a in edge), **levels
        )


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

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions of a field in a dataset: the two axes."""
        return self.coordinate_names

    def locate(self, x: float, y: float) -> tuple[int, int]:
        """The cell whose centre is nearest to ``(x, y)`` along each axis."""
        return int(np.argmin(np.abs(self.x - x))), int(np.argmin(np.abs(self.y - y)))

    def wall_cells(self, inside: np.ndarray) -> np.ndarray:
        """The cells of ``inside`` with a face to a cell outside it or to the domain's edge.

        The axis of an r–z domain and periodic seams are not walls.
        """
        inside = np.asarray(inside, dtype=bool)
        result = np.zeros(self.shape, dtype=bool)
        for axis in (0, 1):
            for step in (1, -1):
                # the neighbour of every cell toward ``step``
                neighbour = np.roll(inside, -step, axis=axis)
                if self.boundaries[axis] != "periodic":
                    end: list[Any] = [slice(None), slice(None)]
                    end[axis] = -1 if step == 1 else 0
                    # beyond the edge: outside, except across the axis
                    neighbour[tuple(end)] = self.on_axis and axis == 0 and step == -1
                result |= inside & ~neighbour
        return result

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

    def transport_faces(self) -> TransportFaces:
        """The faces of the cells for finite-volume transport (see :class:`TransportFaces`)."""
        index = np.arange(self.size).reshape(self.shape)
        widths = (np.diff(self.x_edges), np.diff(self.y_edges))
        interior, boundary = [], []
        for axis in (0, 1):
            cells = np.moveaxis(index, axis, 0)
            centres = (self.x, self.y)[axis]
            edges = (self.x_edges, self.y_edges)[axis]
            other = widths[1 - axis]

            def area(edge: float, axis: int = axis, other: np.ndarray = other):
                if not self.cylindrical:
                    return other
                if axis == 0:  # a radial face: a cylinder of radius ``edge``
                    return 2.0 * np.pi * edge * other
                return 2.0 * np.pi * self.x * other  # an axial face: an annulus

            n = centres.size
            for i in range(n - 1):
                face = edges[i + 1]
                interior.append(
                    (
                        cells[i],
                        cells[i + 1],
                        np.full(cells[i].shape, axis),
                        area(face),
                        np.full(cells[i].shape, face - centres[i]),
                        np.full(cells[i].shape, centres[i + 1] - face),
                    )
                )
            if self.boundaries[axis] == "periodic":
                if n > 1:
                    interior.append(
                        (
                            cells[-1],
                            cells[0],
                            np.full(cells[0].shape, axis),
                            area(edges[-1]),
                            np.full(cells[0].shape, edges[-1] - centres[-1]),
                            np.full(cells[0].shape, centres[0] - edges[0]),
                        )
                    )
                continue
            symmetry = self.boundaries[axis] == "neumann"
            for side, k, edge in ((0, 0, edges[0]), (1, -1, edges[-1])):
                boundary.append(
                    (
                        cells[k],
                        np.full(cells[k].shape, axis),
                        np.full(cells[k].shape, side),
                        np.broadcast_to(area(edge), cells[k].shape),
                        np.full(cells[k].shape, abs(edge - centres[k])),
                        np.full(cells[k].shape, symmetry),
                    )
                )
        return TransportFaces.concatenate(interior, boundary)

    def coordinates(self) -> dict[str, tuple[str, np.ndarray, dict[str, str]]]:
        """The cell-centre coordinates as xarray coordinate entries."""
        first, second = self.coordinate_names
        return {
            first: (first, self.x, {"long_name": first, "units": "m"}),
            second: (second, self.y, {"long_name": second, "units": "m"}),
        }
