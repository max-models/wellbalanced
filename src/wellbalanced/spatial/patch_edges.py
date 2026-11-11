"""The edges and faces of refined patches: the 3D field on patchwork's edge space.

`PatchEdgeGrid` puts a 3D `patchwork.AMRHierarchy` behind the interface of
`wellbalanced.spatial.edges.EdgeGrid`, so that
`wellbalanced.spatial.inductive3d.InductiveField3D` (step R4b of the RF-driver
plan) runs on refined patches: the lowest-order Whitney (Yee) edges and faces of the
leaf cells (`patchwork.CompositeEdges`, `patchwork.CompositeFaces`), whose lumped
masses are the diagonal Hodge matrices of the finite integration technique. Inside a
patch the two are the same scheme; at a coarse–fine boundary the fine edges and faces
on a coarse cell's sides hang, their values constrained by the coarse ones
(``T1``, ``T2``, with ``C T1 = T2 C T1``), and the field is solved for the others by
`patchwork.CurlCurl`.

Geometry, in the numbering of the composite spaces:

- the Hodge diagonals from the lumped masses: an edge's dual area
  ``A*_e = (M1)_ee L_e`` and a face's dual length ``L*_f = (M2)_ff A_f`` (halved on
  the box's boundary and on the coarse side of a coarse–fine boundary, where the fine
  side has its own hanging faces);
- the conformal cut cells of the surface-impedance conductors per patch, from
  simgeometry's ``yee_fractions`` on the patch's grid (`yee_fractions`): a fine
  level resolves the conductors' surfaces, slits and corners;
- `around`: a hierarchy refined where the conductors' surfaces cut the cells.

Boundaries as in `EdgeGrid` (``"dirichlet"``, ``"natural"`` or ``"periodic"`` per
side; periodic axes are the hierarchy's). patchwork is imported when a grid is made.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
from methodcard import method

from wellbalanced.spatial.edges import EdgeGrid, SideBoundary, _sides, edge_average

__all__ = ["PatchEdgeGrid", "surface_tags"]

_FACES = ("x", "y", "z")


def surface_tags(hierarchy: Any, level: int, shapes: Sequence[Any]) -> np.ndarray:
    """The cells of ``level`` (in its patches) that the shapes' surfaces cut.

    A cell is cut when of the 27 points of its corners, edge midpoints, face centres
    and centre some are inside one of ``shapes`` and some are not (a surface between
    them; a feature thinner than half a cell can be missed). Returns booleans of the
    level's shape (`patchwork.AMRHierarchy.level_shape`).
    """
    tags = np.zeros(hierarchy.level_shape(level), dtype=bool)
    spacing = np.asarray(hierarchy.spacing(level))
    origin = np.asarray(hierarchy.lower_bound)
    offsets = np.stack(
        np.meshgrid(*([np.array([0.0, 0.5, 1.0])] * 3), indexing="ij")
    ).reshape(3, -1)
    for patch in hierarchy.patches_at_level(level):
        lo = np.array([b[0] for b in patch.box])
        shape = tuple(int(b[1] - b[0]) for b in patch.box)
        index = np.indices(shape).reshape(3, -1) + lo[:, None]
        inside = np.zeros((offsets.shape[1], index.shape[1]), dtype=bool)
        for k in range(offsets.shape[1]):
            points = (
                origin[:, None] + (index + offsets[:, k : k + 1]) * spacing[:, None]
            )
            for shape_ in shapes:
                inside[k] |= EdgeGrid.contains(shape_, points)
        cut = (inside.any(axis=0) & ~inside.all(axis=0)).reshape(shape)
        box = tuple(slice(int(b[0]), int(b[1])) for b in patch.box)
        tags[box] |= cut
    return tags


@method(
    equations={
        "hodge": r"A^*_e = (M_1)_{ee}\, L_e, \qquad L^*_f = (M_2)_{ff}\, A_f \quad (\text{lumped masses of the leaf cells})",
        "constraints": r"a = T_1 a_\mathrm{free}, \qquad C\,T_1 = T_2\,C\,T_1, \qquad K = T_1^\top\big(C^\top M_\nu C + M_e\big)T_1",
    },
    assumptions=(
        r"lowest-order Whitney edges and faces of the leaf cells (patchwork's composite spaces); inside a patch the finite integration scheme of `EdgeGrid`",
        r"cut cells per patch from simgeometry's `yee_fractions` on the patch's own grid",
    ),
    references=(
        "Weiland, AEÜ 31, 116 (1977) (finite integration)",
        "[Bossavit, IEE Proc. A 135, 493 (1988)](https://doi.org/10.1049/ip-a-1.1988.0077)",
    ),
)
class PatchEdgeGrid:
    """The edges and faces of the leaf cells of a 3D patchwork hierarchy.

    The same attributes and methods as `wellbalanced.spatial.edges.EdgeGrid` that
    `wellbalanced.spatial.inductive3d.InductiveField3D` uses, in the numbering
    of patchwork's composite edges and faces (hanging ones included), plus the
    hanging edges, the composite spaces and the faces' conditions for
    `patchwork.CurlCurl`.

    Args:
        hierarchy: A 3D `patchwork.AMRHierarchy` (its periodic axes are periodic).
        boundaries: Per axis, ``"dirichlet"``, ``"natural"`` or ``"periodic"``, or a
            pair ``(lower side, upper side)``; default Dirichlet on the non-periodic
            axes.

    Attributes:
        hierarchy: The hierarchy.
        node_space: Its `patchwork.CompositeNodes`.
        edges: Its `patchwork.CompositeEdges`.
        faces: Its `patchwork.CompositeFaces`.
        conditions: The faces of the box for `patchwork.CurlCurl` (``"electric"``
            for Dirichlet, ``"magnetic"`` for natural).
    """

    # --- masks -----------------------------------------------------------------------

    contains = staticmethod(EdgeGrid.contains)

    def _wall_edges(self) -> np.ndarray:
        """Whether every edge lies in a Dirichlet wall (tangential to it)."""
        edges = self.edges
        wall = np.zeros(self.n_edges, dtype=bool)
        cells = self.node_space._cells
        for axis in range(3):
            if self.periodic[axis]:
                continue
            for side, position in ((0, 0), (1, cells[axis])):
                if self.boundaries[axis][side] == "dirichlet":
                    wall |= (edges.direction != axis) & (
                        edges.lattice[:, axis] == position
                    )
        return wall

    def __init__(
        self,
        hierarchy: Any,
        boundaries: Sequence[Any] | None = None,
    ) -> None:
        """Build the composite spaces and the Hodge diagonals."""
        from patchwork import CompositeEdges, CompositeFaces, CompositeNodes

        if hierarchy.ndim != 3:
            raise ValueError("a patch edge grid needs a 3D hierarchy")
        self.hierarchy = hierarchy
        self.periodic = tuple(bool(p) for p in hierarchy.periodic)
        if boundaries is None:
            boundaries = ["periodic" if p else "dirichlet" for p in self.periodic]
        if len(boundaries) != 3:
            raise ValueError("give a boundary per axis")
        self.boundaries: tuple[tuple[SideBoundary, SideBoundary], ...] = tuple(
            _sides(b) for b in boundaries
        )
        for axis, (pair, periodic) in enumerate(
            zip(self.boundaries, self.periodic, strict=True)
        ):
            if (pair[0] == "periodic") != periodic:
                raise ValueError(
                    f"axis {axis}: periodic boundaries must match the hierarchy's"
                )
        self.conditions = {
            f"{_FACES[axis]}_{side}": (
                "electric" if self.boundaries[axis][k] == "dirichlet" else "magnetic"
            )
            for axis in range(3)
            if not self.periodic[axis]
            for k, side in enumerate(("lower", "upper"))
        }
        self.node_space = CompositeNodes(hierarchy)
        self.edges = CompositeEdges(self.node_space)
        self.faces = CompositeFaces(self.edges)
        nodes = self.node_space
        self.lower = np.asarray(hierarchy.lower_bound, dtype=float)
        self.upper = np.asarray(hierarchy.upper_bound, dtype=float)
        #: The finest cell size along each axis [m].
        self.spacing = np.asarray(nodes._spacing, dtype=float)
        self.n_edges = int(self.edges.num_edges)
        self.n_faces = int(self.faces.num_faces)
        self.n_nodes = int(nodes.num_nodes)
        #: Length of every edge [m].
        self.edge_lengths = np.asarray(self.edges.length, dtype=float)
        #: Area of every edge's dual face [m^2].
        self.dual_areas = self.edges.mass(True).diagonal() * self.edge_lengths
        #: Area of every face [m^2].
        self.face_areas = np.asarray(self.faces.area, dtype=float)
        #: Length of every face's dual edge [m].
        self.dual_lengths = self.faces.mass(True).diagonal() * self.face_areas
        #: Which edges hang (their values follow from the others').
        self.hanging_edges = np.asarray(self.edges.hanging, dtype=bool)
        #: Which edges are unknowns: neither hanging nor in a Dirichlet wall.
        self.free_edges = ~self.hanging_edges & ~self._wall_edges()

    @classmethod
    def around(
        cls,
        lower: Sequence[float],
        upper: Sequence[float],
        cells: Sequence[int],
        shapes: Sequence[Any],
        *,
        levels: int = 1,
        boundaries: Sequence[Any] = ("dirichlet", "dirichlet", "dirichlet"),
        buffer: int = 1,
        **regrid: Any,
    ) -> PatchEdgeGrid:
        """A grid refined ``levels`` times where the surfaces of ``shapes`` cut it.

        The base level has ``cells`` over the box; each level tags the cells the
        shapes' surfaces cut (`surface_tags`) and patchwork's ``regrid`` covers them
        with patches of the next level (``buffer`` cells around them; ``regrid``
        takes its other options, e.g. ``efficiency``, ``max_patch_size``).
        """
        from patchwork import AMRHierarchy

        pairs = [_sides(b) for b in boundaries]
        hierarchy = AMRHierarchy(
            base_shape=tuple(int(n) for n in cells),
            bounds=tuple(
                (float(a), float(b)) for a, b in zip(lower, upper, strict=True)
            ),
            periodic=tuple(p[0] == "periodic" for p in pairs),
        )
        for level in range(int(levels)):
            tags = {k: surface_tags(hierarchy, k, shapes) for k in range(level + 1)}
            if not tags[level].any():
                break
            hierarchy = hierarchy.regrid(tags, buffer=buffer, **regrid)
        return cls(hierarchy, boundaries)

    # --- positions -----------------------------------------------------------------

    def _wrap(self, points: np.ndarray) -> np.ndarray:
        """Points moved into the box along the periodic axes."""
        for axis in range(3):
            if self.periodic[axis]:
                size = self.upper[axis] - self.lower[axis]
                points[axis] = self.lower[axis] + np.mod(
                    points[axis] - self.lower[axis], size
                )
        return points

    def edge_midpoints(self) -> np.ndarray:
        """The midpoints of all edges, ``(3, n_edges)`` [m]."""
        edges = self.edges
        along = np.arange(3)[None, :] == edges.direction[:, None]
        lattice = edges.lattice + 0.5 * edges.span[:, None] * along
        return (self.lower[None, :] + lattice * self.spacing[None, :]).T

    def face_centres(self) -> np.ndarray:
        """The centres of all faces, ``(3, n_faces)`` [m]."""
        faces = self.faces
        across = np.arange(3)[None, :] != faces.normal[:, None]
        lattice = faces.lattice + 0.5 * faces.span[:, None] * across
        return (self.lower[None, :] + lattice * self.spacing[None, :]).T

    def edge_axes(self) -> np.ndarray:
        """The axis (0, 1, 2) of every edge."""
        return np.asarray(self.edges.direction)

    def face_axes(self) -> np.ndarray:
        """The normal axis (0, 1, 2) of every face."""
        return np.asarray(self.faces.normal)

    def cell_centres(self) -> np.ndarray:
        """The centres of the leaf cells, ``(3, m)`` [m]."""
        nodes = self.node_space
        lattice = nodes.cell_lower + 0.5 * nodes.cell_span[:, None]
        return (self.lower[None, :] + lattice * self.spacing[None, :]).T

    def face_dual_points(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """The midpoints of the two halves of every face's dual edge.

        Returns ``((below, valid), (above, valid))``: ``(3, n_faces)`` points a quarter
        of the neighbouring leaf cell below and above every face along its normal, and
        whether that leaf cell has the face (not beyond the box, nor a coarse face's
        fine side, which has its own hanging faces).
        """
        nodes, faces = self.node_space, self.faces
        centres = self.face_centres()
        result = []
        for side, sign in ((1, -1.0), (0, 1.0)):  # a cell's upper face is below it
            points = centres.copy()
            valid = np.zeros(self.n_faces, dtype=bool)
            for axis in range(3):
                face = faces.cell_faces[:, 2 * axis + side]
                quarter = 0.25 * nodes.cell_span * self.spacing[axis]
                points[axis, face] += sign * quarter
                valid[face] = True
            result.append((self._wrap(points), valid))
        return tuple(result)

    def edge_mask(self, shape: Any) -> np.ndarray:
        """The edges whose midpoint ``shape`` contains (a staircased region)."""
        return self.contains(shape, self.edge_midpoints())

    def face_mask(self, shape: Any) -> np.ndarray:
        """The faces whose centre ``shape`` contains."""
        return self.contains(shape, self.face_centres())

    def edge_average(
        self, value: Callable[[np.ndarray], np.ndarray], samples: int = 1
    ) -> np.ndarray:
        """An edge coefficient from a point function, as `EdgeGrid.edge_average`.

        The dual cell of an edge spans the edge's length along every axis (its own
        level's cell size).
        """
        sizes = (self.edges.span[:, None] * self.spacing[None, :]).T
        return edge_average(
            self.edge_midpoints(), self.edge_axes(), sizes, value, samples
        )

    def yee_fractions(
        self, geometry: Any, samples: int = 64
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """What of every edge and face a simgeometry ``geometry`` covers, per patch.

        simgeometry's ``yee_fractions`` on every patch's own grid, put into the
        composite numbering: the fraction of every edge's length and every face's area
        inside, the trace of the surface in every face and its projection. An edge or
        face on the side of two patches of a level takes either's (the same) value.
        """
        import simgeometry as sg

        hierarchy = self.hierarchy
        ratio = int(hierarchy.refinement_ratio)
        finest = max(int(p.level) for p in hierarchy.patches)
        edge_fraction = np.full(self.n_edges, np.nan)
        face_fraction = np.full(self.n_faces, np.nan)
        trace = np.full(self.n_faces, np.nan)
        projected = np.full(self.n_faces, np.nan)
        for patch in hierarchy.patches:
            span = ratio ** (finest - int(patch.level))
            lo = np.array([int(b[0]) for b in patch.box])
            cells = tuple(int(b[1] - b[0]) for b in patch.box)
            bounds = patch.physical_bounds(hierarchy)
            grid = sg.Grid(
                3,
                tuple(float(b[0]) for b in bounds),
                tuple(float(b[1]) for b in bounds),
                tuple(n + 1 for n in cells),
            )
            fractions = geometry.yee_fractions(grid, samples=samples)
            for axis in range(3):
                block = fractions.edges[axis]
                lattice = (np.indices(block.shape).reshape(3, -1) + lo[:, None]) * span
                found = self.edges._find(
                    np.full(block.size, axis), lattice.T, np.full(block.size, span)
                )
                keep = found >= 0
                edge_fraction[found[keep]] = block.reshape(-1)[keep]
                block = fractions.faces[axis]
                lattice = (np.indices(block.shape).reshape(3, -1) + lo[:, None]) * span
                found = self.faces._find(
                    np.full(block.size, axis), lattice.T, np.full(block.size, span)
                )
                keep = found >= 0
                for target, source in (
                    (face_fraction, fractions.faces[axis]),
                    (trace, fractions.traces[axis]),
                    (projected, fractions.projected_traces[axis]),
                ):
                    target[found[keep]] = source.reshape(-1)[keep]
        if np.isnan(edge_fraction).any() or np.isnan(face_fraction).any():
            raise RuntimeError("some edges or faces lie in no patch")
        return edge_fraction, face_fraction, trace, projected

    # --- operators -------------------------------------------------------------------

    def curl(self) -> sp.csr_matrix:
        """The face–edge incidence ``C`` (``b = C a``), ``n_faces x n_edges``."""
        return sp.csr_matrix(self.faces.curl)

    def gradient(self) -> sp.csr_matrix:
        """The edge–node incidence ``G`` (``C G = 0``), ``n_edges x n_nodes``."""
        return sp.csr_matrix(self.edges.gradient)

    def face_values_at_cells(self, values: np.ndarray) -> np.ndarray:
        """A face field (e.g. ``B . n``) averaged to the leaf cells' centres, ``(3, m)``."""
        cell_faces = self.faces.cell_faces
        return np.stack(
            [
                0.5 * (values[cell_faces[:, 2 * a]] + values[cell_faces[:, 2 * a + 1]])
                for a in range(3)
            ]
        )

    def feed_path(
        self, start: Sequence[float], end: Sequence[float], order: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """The edges from the node nearest ``start`` to the one nearest ``end``.

        Axis by axis in ``order``, along the finest edges there are. Returns the edges,
        their signs (+1 along the edge's direction) and the numbers of the two end
        nodes.
        """
        nodes = self.node_space
        coordinates = np.asarray(nodes.coordinates, dtype=float)
        ends = []
        for point in (start, end):
            point = np.asarray(point, dtype=float)
            if np.any(point < self.lower - 0.5 * self.spacing) or np.any(
                point > self.upper + 0.5 * self.spacing
            ):
                raise ValueError("a terminal is outside the grid")
            ends.append(int(np.argmin(np.sum((coordinates - point) ** 2, axis=1))))
        lattice = np.rint((coordinates[ends] - self.lower) / self.spacing).astype(int)
        current, target = lattice[0].copy(), lattice[1]
        spans = np.unique(self.edges.span)
        edges, signs = [], []
        for axis in order:
            while current[axis] != target[axis]:
                step = 1 if target[axis] > current[axis] else -1
                for span in spans:
                    if span > abs(int(target[axis] - current[axis])):
                        continue
                    low = current.copy()
                    if step < 0:
                        low[axis] -= span
                    found = int(
                        self.edges._find(
                            np.array([axis]), low[None, :], np.array([span])
                        )[0]
                    )
                    if found >= 0:
                        break
                else:
                    raise ValueError("no path of edges between the terminals")
                edges.append(found)
                signs.append(float(step))
                current[axis] += step * int(span)
        return np.asarray(edges, dtype=int), np.asarray(signs), ends
