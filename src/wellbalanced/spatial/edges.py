"""The uniform Cartesian edge (Yee) grid of the 3D field.

:class:`EdgeGrid` is the staggered grid of the finite integration technique (Weiland
1977; Yee 1966): a vector potential as line integrals ``a_e = int_e A . dl`` on the edges
of the cells, the magnetic flux ``b_f = int_f B . dA = (C a)_f`` through their faces, with
``C`` the face–edge incidence matrix, the discrete curl (entries ``0, +-1``); the
node–edge incidence ``G``, the discrete gradient, has ``C G = 0`` exactly. The metric is
in the diagonal Hodge matrices only: an edge's length and the area of its dual face (the
plane through its midpoint between the centres of the four cells around it), a face's
area and the length of its dual edge (between the centres of the two cells it
separates). Edges and faces on the domain's boundary have half dual faces and half dual
edges.

Boundaries, per side of every axis:

- ``"dirichlet"``: ``A x n = 0`` (a perfectly conducting wall, ``B . n = 0``); the
  edges in the wall are not unknowns.
- ``"natural"``: ``H x n = 0`` (a magnetic wall, a symmetry plane of a field normal to
  it, e.g. the midplane of a solenoid); the edges in the wall are unknowns.
- ``"periodic"`` (both sides of an axis).

Edges and faces are numbered axis by axis: the edges along ``x`` first, each block a
C-ordered array of shape :meth:`edge_shape`. Shapes are anything with
``contains_points`` taking ``(3, n)`` points (simgeometry's 3D shapes and their
Booleans); this module does not import simgeometry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

__all__ = ["EdgeGrid", "SideBoundary"]

SideBoundary = Literal["dirichlet", "natural", "periodic"]
_SIDES = ("dirichlet", "natural", "periodic")


def _sides(boundary: Any) -> tuple[SideBoundary, SideBoundary]:
    """``(lower, upper)`` of one axis from a string or a pair."""
    pair = (boundary, boundary) if isinstance(boundary, str) else tuple(boundary)
    if len(pair) != 2 or any(side not in _SIDES for side in pair):
        raise ValueError(f"unknown boundary {boundary!r}")
    if ("periodic" in pair) and pair != ("periodic", "periodic"):
        raise ValueError("a periodic axis is periodic on both sides")
    return pair  # type: ignore[return-value]


_CYCLE = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


class EdgeGrid:
    """Edges and faces of a uniform Cartesian grid of ``cells[0] x cells[1] x cells[2]``.

    Args:
        lower: The lower corner of the box [m].
        upper: The upper corner of the box [m].
        cells: Cells along each axis.
        boundaries: Per axis, one of ``"dirichlet"``, ``"natural"``, ``"periodic"``, or a
            pair ``(lower side, upper side)``; default Dirichlet everywhere.
    """

    def _edge_count(self, axis: int) -> int:
        return int(np.prod(self.edge_shape(axis)))

    def _face_count(self, axis: int) -> int:
        return int(np.prod(self.face_shape(axis)))

    def _ends(self, shape: tuple[int, ...], axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Masks of the first and last node planes along ``axis`` (none if periodic)."""
        index = np.indices(shape)[axis]
        if self.periodic[axis]:
            none = np.zeros(shape, dtype=bool)
            return none, none
        return index == 0, index == shape[axis] - 1

    def _halved(self, shape, axis: int, area: float) -> np.ndarray:
        """Dual-face areas of the edges along ``axis``: halved per boundary plane."""
        result = np.full(shape, area)
        for b in range(3):
            if b == axis:
                continue
            first, last = self._ends(shape, b)
            result[first | last] *= 0.5
        return result.ravel()

    def _halved_normal(self, axis: int, length: float) -> np.ndarray:
        shape = self.face_shape(axis)
        result = np.full(shape, length)
        first, last = self._ends(shape, axis)
        result[first | last] *= 0.5
        return result.ravel()

    def _free(self, axis: int) -> np.ndarray:
        shape = self.edge_shape(axis)
        free = np.ones(shape, dtype=bool)
        for b in range(3):
            if b == axis:
                continue
            first, last = self._ends(shape, b)
            if self.boundaries[b][0] == "dirichlet":
                free &= ~first
            if self.boundaries[b][1] == "dirichlet":
                free &= ~last
        return free.ravel()

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        cells: Sequence[int],
        boundaries: Sequence[Any] = ("dirichlet", "dirichlet", "dirichlet"),
    ) -> None:
        """Count the nodes, edges and faces and compute the Hodge diagonals."""
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.cells = tuple(int(n) for n in cells)
        if self.lower.shape != (3,) or self.upper.shape != (3,) or len(self.cells) != 3:
            raise ValueError("a 3D grid needs three lower, upper and cell counts")
        if np.any(self.upper <= self.lower) or min(self.cells) < 1:
            raise ValueError("need upper > lower and at least one cell per axis")
        if len(boundaries) != 3:
            raise ValueError("give a boundary per axis")
        self.boundaries = tuple(_sides(b) for b in boundaries)
        self.periodic = tuple(b[0] == "periodic" for b in self.boundaries)
        #: The cell size along each axis [m].
        self.spacing = (self.upper - self.lower) / np.asarray(self.cells)
        #: Nodes along each axis (a periodic axis has no copy of its first node).
        self.nodes = tuple(
            n if p else n + 1 for n, p in zip(self.cells, self.periodic, strict=True)
        )
        self.edge_offsets = np.cumsum(
            [0] + [int(np.prod(self.edge_shape(a))) for a in range(3)]
        )
        self.face_offsets = np.cumsum(
            [0] + [int(np.prod(self.face_shape(a))) for a in range(3)]
        )
        self.n_edges = int(self.edge_offsets[-1])
        self.n_faces = int(self.face_offsets[-1])
        self.n_nodes = int(np.prod(self.nodes))
        h = self.spacing
        #: Length of every edge [m].
        self.edge_lengths = np.concatenate(
            [np.full(self._edge_count(a), h[a]) for a in range(3)]
        )
        #: Area of every edge's dual face [m^2] (halved per boundary it lies in).
        self.dual_areas = np.concatenate(
            [self._halved(self.edge_shape(a), a, h[b] * h[c]) for a, b, c in _CYCLE]
        )
        #: Area of every face [m^2].
        self.face_areas = np.concatenate(
            [np.full(self._face_count(a), h[b] * h[c]) for a, b, c in _CYCLE]
        )
        #: Length of every face's dual edge [m] (halved on a boundary).
        self.dual_lengths = np.concatenate(
            [self._halved_normal(a, h[a]) for a in range(3)]
        )
        #: Which edges are unknowns: all but those in a Dirichlet wall.
        self.free_edges = np.concatenate([self._free(a) for a in range(3)])

    # --- counting ---------------------------------------------------------------

    def edge_shape(self, axis: int) -> tuple[int, int, int]:
        """The array shape of the edges along ``axis``: cells along it, nodes across."""
        return tuple(  # type: ignore[return-value]
            self.cells[b] if b == axis else self.nodes[b] for b in range(3)
        )

    def face_shape(self, axis: int) -> tuple[int, int, int]:
        """The array shape of the faces normal to ``axis``: nodes along it, cells across."""
        return tuple(  # type: ignore[return-value]
            self.nodes[b] if b == axis else self.cells[b] for b in range(3)
        )

    # --- positions ---------------------------------------------------------------

    def node_coordinates(self, axis: int) -> np.ndarray:
        """The node positions along ``axis`` [m]."""
        return self.lower[axis] + self.spacing[axis] * np.arange(self.nodes[axis])

    def cell_coordinates(self, axis: int) -> np.ndarray:
        """The cell-centre positions along ``axis`` [m]."""
        return self.lower[axis] + self.spacing[axis] * (
            np.arange(self.cells[axis]) + 0.5
        )

    def _points(self, staggered: Sequence[bool]) -> np.ndarray:
        """``(3, n)`` points of a C-ordered block: cell centres where ``staggered``."""
        axes = [
            self.cell_coordinates(b) if staggered[b] else self.node_coordinates(b)
            for b in range(3)
        ]
        grids = np.meshgrid(*axes, indexing="ij")
        return np.stack([g.ravel() for g in grids])

    def edge_midpoints(self) -> np.ndarray:
        """The midpoints of all edges, ``(3, n_edges)`` [m]."""
        return np.concatenate(
            [self._points([b == a for b in range(3)]) for a in range(3)], axis=1
        )

    def face_centres(self) -> np.ndarray:
        """The centres of all faces, ``(3, n_faces)`` [m]."""
        return np.concatenate(
            [self._points([b != a for b in range(3)]) for a in range(3)], axis=1
        )

    def edge_axes(self) -> np.ndarray:
        """The axis (0, 1, 2) of every edge."""
        return np.repeat(np.arange(3), np.diff(self.edge_offsets))

    def face_axes(self) -> np.ndarray:
        """The normal axis (0, 1, 2) of every face."""
        return np.repeat(np.arange(3), np.diff(self.face_offsets))

    # --- masks ------------------------------------------------------------------

    @staticmethod
    def contains(shape: Any, points: np.ndarray) -> np.ndarray:
        """``shape.contains_points`` of ``(3, n)`` points, on the host."""
        inside = shape.contains_points(points)
        if hasattr(inside, "get"):
            inside = inside.get()
        return np.asarray(inside, dtype=bool).reshape(points.shape[1])

    def edge_mask(self, shape: Any) -> np.ndarray:
        """The edges whose midpoint ``shape`` contains (a staircased region)."""
        return self.contains(shape, self.edge_midpoints())

    def face_mask(self, shape: Any) -> np.ndarray:
        """The faces whose centre ``shape`` contains."""
        return self.contains(shape, self.face_centres())

    def edge_average(
        self, value: Callable[[np.ndarray], np.ndarray], samples: int = 1
    ) -> np.ndarray:
        """An edge coefficient from a point function, e.g. a conductivity.

        ``samples = 1``: the value at the midpoint. ``samples = s > 1``: ``s`` points
        along the edge times ``s x s`` across its dual face; the harmonic mean along
        the edge (in series: one insulating point blocks the edge) and the arithmetic
        mean across (in parallel), the conductance of the edge's dual cell. ``value``
        takes ``(3, n)`` points and returns ``n`` values (zero is an insulator).
        """
        midpoints = self.edge_midpoints()
        if samples == 1:
            return np.asarray(value(midpoints))
        if samples < 1:
            raise ValueError("samples must be >= 1")
        offsets = (np.arange(samples) + 0.5) / samples - 0.5
        axes = self.edge_axes()
        h = self.spacing
        result = np.zeros(self.n_edges, dtype=complex)
        for across_1 in offsets:
            for across_2 in offsets:
                inverse = np.zeros(self.n_edges, dtype=complex)
                blocked = np.zeros(self.n_edges, dtype=bool)
                for along in offsets:
                    points = midpoints.copy()
                    for a in range(3):
                        b, c = (a + 1) % 3, (a + 2) % 3
                        edges = axes == a
                        points[a, edges] += along * h[a]
                        points[b, edges] += across_1 * h[b]
                        points[c, edges] += across_2 * h[c]
                    sampled = np.asarray(value(points), dtype=complex)
                    zero = sampled == 0
                    blocked |= zero
                    inverse[~zero] += 1.0 / sampled[~zero]
                series = np.where(
                    blocked, 0.0, samples / np.where(blocked, 1.0, inverse)
                )
                result += series / samples**2
        return result

    def nested_dissection(self, edges: np.ndarray, leaf: int = 64) -> np.ndarray:
        """A fill-reducing order of the ``edges`` (a mask) for a sparse LU.

        Geometric nested dissection: split the box at the middle node plane of its
        longest side; the edges in the plane separate the two halves (an edge across
        it has its midpoint off the plane, and the curl-curl couples edges only
        through a common face), so they are numbered after both halves, recursively.
        Returns the positions in ``np.flatnonzero(edges)``, a permutation.
        """
        points = self.edge_midpoints()[:, edges]
        tolerance = 1e-9 * float(np.min(self.spacing))

        def split(ids: np.ndarray) -> list[np.ndarray]:
            if ids.size <= leaf:
                return [ids]
            p = points[:, ids]
            axis = int(np.argmax(np.ptp(p, axis=1)))
            coordinate = p[axis]
            # the node plane nearest the median (a plane of edge midpoints between
            # two node planes would not separate them)
            h, origin = self.spacing[axis], self.lower[axis]
            middle = origin + h * np.round((np.median(coordinate) - origin) / h)
            plane = np.abs(coordinate - middle) <= tolerance
            lower, upper = (
                coordinate < middle - tolerance,
                coordinate > middle + tolerance,
            )
            if not lower.any() or not upper.any():
                return [ids]
            return [*split(ids[lower]), *split(ids[upper]), ids[plane]]

        return np.concatenate(split(np.arange(points.shape[1])))

    # --- operators -----------------------------------------------------------------

    def _edge_index(self, axis: int, index: list[np.ndarray]) -> np.ndarray:
        """Global edge numbers of multi-indices (wrapped on periodic axes)."""
        shape = self.edge_shape(axis)
        wrapped = [i % n for i, n in zip(index, shape, strict=True)]
        return self.edge_offsets[axis] + np.ravel_multi_index(wrapped, shape)

    def curl(self) -> sp.csr_matrix:
        """The face–edge incidence ``C`` (``b = C a``), ``n_faces x n_edges``."""
        rows, cols, values = [], [], []
        for a, b, c in _CYCLE:
            shape = self.face_shape(a)
            index = [i.ravel() for i in np.indices(shape)]
            faces = self.face_offsets[a] + np.arange(index[0].size)

            def shifted(axis: int, index=index) -> list[np.ndarray]:
                result = list(index)
                result[axis] = result[axis] + 1
                return result

            for edge_axis, multi, sign in (
                (b, index, 1.0),
                (c, shifted(b), 1.0),
                (b, shifted(c), -1.0),
                (c, index, -1.0),
            ):
                rows.append(faces)
                cols.append(self._edge_index(edge_axis, multi))
                values.append(np.full(faces.size, sign))
        return sp.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_faces, self.n_edges),
        )

    def gradient(self) -> sp.csr_matrix:
        """The edge–node incidence ``G`` (``C G = 0``), ``n_edges x n_nodes``."""
        rows, cols, values = [], [], []
        for a in range(3):
            shape = self.edge_shape(a)
            index = [i.ravel() for i in np.indices(shape)]
            edges = self.edge_offsets[a] + np.arange(index[0].size)
            end = list(index)
            end[a] = (end[a] + 1) % self.nodes[a]
            for multi, sign in ((index, -1.0), (end, 1.0)):
                rows.append(edges)
                cols.append(np.ravel_multi_index(multi, self.nodes))
                values.append(np.full(edges.size, sign))
        return sp.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_edges, self.n_nodes),
        )

    def face_values_at_cells(self, values: np.ndarray) -> np.ndarray:
        """A face field (e.g. ``B . n``) averaged to the cell centres, ``(3, *cells)``."""
        result = []
        for a in range(3):
            block = values[self.face_offsets[a] : self.face_offsets[a + 1]].reshape(
                self.face_shape(a)
            )
            upper = np.roll(block, -1, axis=a)
            mean = 0.5 * (block + upper)
            result.append(np.take(mean, np.arange(self.cells[a]), axis=a))
        return np.stack(result)

    def edge_block(self, values: np.ndarray, axis: int) -> np.ndarray:
        """The edges along ``axis`` of an edge field, as an array of :meth:`edge_shape`."""
        return values[self.edge_offsets[axis] : self.edge_offsets[axis + 1]].reshape(
            self.edge_shape(axis)
        )
