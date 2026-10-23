"""The leaf cells of a patchwork hierarchy as the grid of the spatial models.

:class:`PatchGrid` has the interface of :class:`~wellbalanced.spatial.grid.CellGrid`, so
every spatial model runs on either: fields are arrays over the leaf cells, shape
``(n,)``, instead of ``(nx, ny)``. The geometry and the operator are patchwork's
(:class:`patchwork.CellOperator`): the curl-curl of the field normal to the plane, a
scalar in a cut and the azimuthal component in r–z, assembled from the same energy as
:class:`CellGrid`'s, its coarse–fine faces the symmetric variant (the power balance holds
to round-off), and the curl of a field from the operator's own faces
(:meth:`patchwork.CellOperator.curl`). A coarse base level reaches the far boundary and
finer patches cover the source, so where the stretched tensor grid refines whole rows
and columns, the patches refine only the places that need it.

Layouts: :meth:`PatchGrid.refined` from boxes per level (or a function of the cell
centres), :meth:`PatchGrid.from_edges` as fine as a tensor grid everywhere (for
comparisons), :meth:`PatchGrid.refined_where` one level finer where a field is large
(the skin layer of a solution, say). patchwork is an optional dependency
(``pip install "wellbalanced[spatial]"``), imported when a grid is made.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from wellbalanced.spatial.grid import AxisBoundary, CellField, Geometry, TransportFaces

if TYPE_CHECKING:
    from patchwork import AMRHierarchy, CellOperator

__all__ = ["PatchGrid"]

#: Physical boxes ``((x0, x1), (y0, y1))``.
Box = tuple[tuple[float, float], tuple[float, float]]
_FACE_NAMES = (("x_lower", "x_upper"), ("y_lower", "y_upper"))


class PatchGrid:
    """The leaf cells of a patch hierarchy, in a cut or in r–z.

    Args:
        hierarchy: A :class:`patchwork.AMRHierarchy` (its ``geometry``, ``"cartesian"`` or
            ``"cylindrical"``, is the grid's).
        boundaries: The boundary of each axis, as for :class:`CellGrid`: ``"dirichlet"``,
            ``"neumann"`` or ``"periodic"`` (which the hierarchy's ``periodic`` must
            match); in r–z the axis, when the radius starts at 0.
    """

    def __init__(
        self,
        hierarchy: AMRHierarchy,
        *,
        boundaries: tuple[AxisBoundary, AxisBoundary] = ("dirichlet", "dirichlet"),
    ) -> None:
        """Number the leaf cells and assemble the operator."""
        from patchwork import CellOperator, Dirichlet, Neumann

        for axis, boundary in enumerate(boundaries):
            if boundary not in ("dirichlet", "neumann", "periodic"):
                raise ValueError(f"unknown boundary {boundary!r}")
            if (boundary == "periodic") != hierarchy.periodic[axis]:
                raise ValueError(
                    f"axis {axis}: the boundary {boundary!r} and the hierarchy's "
                    f"periodic={hierarchy.periodic[axis]} disagree"
                )
        self.hierarchy = hierarchy
        self.geometry: Geometry = hierarchy.geometry  # type: ignore[assignment]
        self.boundaries = tuple(boundaries)
        faces: dict[str, Any] = {}
        for axis, boundary in enumerate(boundaries):
            for side, name in enumerate(_FACE_NAMES[axis]):
                if boundary == "periodic" or (axis == 0 and side == 0 and self.on_axis):
                    continue
                faces[name] = (
                    Dirichlet(0.0) if boundary == "dirichlet" else Neumann(0.0)
                )
        self._faces = faces
        self._component = "azimuthal" if self.cylindrical else "scalar"
        self._operator: CellOperator = CellOperator(
            hierarchy,
            component=self._component,
            faces=faces,
            mean="arithmetic",
            factorize=False,
        )
        cells = self._operator.cells
        self.cells = cells
        self.size = cells.num_cells
        self.shape = (self.size,)
        self.x = cells.centers[:, 0].copy()
        self.y = cells.centers[:, 1].copy()
        self.xx, self.yy = self.x, self.y
        #: Cross-section areas of the cells in the plane [m^2].
        self.cell_areas = np.prod(cells.sizes, axis=1)
        #: The length a cell extends out of the plane: 1 m in the cut, ``2 pi r`` in r–z.
        self.circumference = (
            2.0 * np.pi * self.x if self.cylindrical else np.ones(self.size)
        )
        #: Cell volumes [m^3] (per metre along ``z`` in the cut).
        self.cell_volumes = cells.volumes.copy()

    # --- layouts -----------------------------------------------------------------

    @classmethod
    def refined(
        cls,
        bounds: Box,
        base_shape: tuple[int, int],
        refine: Sequence[Sequence[Box]] | Callable[[np.ndarray, np.ndarray, int], Any],
        *,
        levels: int | None = None,
        geometry: Geometry = "cartesian",
        boundaries: tuple[AxisBoundary, AxisBoundary] = ("dirichlet", "dirichlet"),
        buffer: int = 1,
        efficiency: float = 0.8,
    ) -> PatchGrid:
        """A coarse base of ``base_shape`` cells over ``bounds``, finer where asked.

        ``refine[k]`` are the boxes level ``k + 1`` (each level twice as fine) must
        cover; or a function ``refine(x, y, level)`` of the cell centres of a level that
        is true where the next level is needed, for ``levels`` refined levels.
        patchwork's regrid covers them with properly nested patches, ``buffer`` cells
        wider.
        """
        from patchwork import AMRHierarchy

        periodic = (boundaries[0] == "periodic", boundaries[1] == "periodic")
        hierarchy = AMRHierarchy(
            base_shape, bounds, periodic=periodic, geometry=geometry
        )
        function: Callable[[np.ndarray, np.ndarray, int], Any]
        if callable(refine):
            if levels is None:
                raise ValueError("a refinement function needs the number of levels")
            function = refine
        else:
            boxes = [list(level) for level in refine]
            levels = len(boxes)

            def in_boxes(x: np.ndarray, y: np.ndarray, level: int) -> np.ndarray:
                inside = np.zeros(np.shape(x), dtype=bool)
                for (x0, x1), (y0, y1) in boxes[level]:
                    inside |= (x > x0) & (x < x1) & (y > y0) & (y < y1)
                return inside

            function = in_boxes

        if levels:
            tags = {
                level: _level_tags(hierarchy, level, function)
                for level in range(levels)
            }
            hierarchy = hierarchy.regrid(tags, buffer=buffer, efficiency=efficiency)
        return cls(hierarchy, boundaries=boundaries)

    @classmethod
    def from_edges(
        cls,
        x_edges: Sequence[float] | np.ndarray,
        y_edges: Sequence[float] | np.ndarray,
        *,
        geometry: Geometry = "cartesian",
        boundaries: tuple[AxisBoundary, AxisBoundary] = ("dirichlet", "dirichlet"),
        max_levels: int = 5,
        buffer: int = 1,
        efficiency: float = 0.8,
    ) -> PatchGrid:
        """Patches about as fine as the tensor grid of these edges.

        The finest level has the tensor grid's smallest spacing (or finer, along an axis
        with fewer cells than the levels need), the base up to ``max_levels`` levels
        coarser. A cell is refined while it is coarser than the tensor cells it overlaps
        along every stretched axis: where the tensor grid is fine along one axis only
        (the strips a stretched tensor grid refines across the whole domain) the patches
        follow its coarser spacing. A uniform axis sets the level everywhere instead (at
        least as fine as its spacing). The finest lattice is aligned with the edges'
        ends, so a tensor grid that is uniform where it is fine shares its cell faces
        there. For comparing the two grids on one problem.
        """
        edges = [np.asarray(x_edges, dtype=float), np.asarray(y_edges, dtype=float)]
        for axis_edges in edges:
            if axis_edges.ndim != 1 or axis_edges.size < 2:
                raise ValueError("cell edges must be 1D, at least two")
            if np.any(np.diff(axis_edges) <= 0.0):
                raise ValueError("cell edges must be increasing")
        spacing = [np.diff(e) for e in edges]
        h_min = [float(s.min()) for s in spacing]
        lengths = [float(e[-1] - e[0]) for e in edges]
        finest_cells = [length / h for length, h in zip(lengths, h_min, strict=True)]
        spread = max(float(s.max()) / h for s, h in zip(spacing, h_min, strict=True))
        levels = int(min(max_levels, np.floor(np.log2(spread) + 1e-9)))
        shape: list[int] = []
        while True:
            shape = []
            for count in finest_cells:
                if count < 2**levels - 1e-6:
                    shape.append(1)
                    continue
                base = count / 2**levels
                if abs(base - round(base)) > 1e-6:
                    break
                shape.append(round(base))
            if len(shape) == 2 or levels == 0:
                break
            levels -= 1
        if len(shape) != 2:
            raise ValueError(
                "the edges' length is not a whole number of their smallest spacing"
            )

        from patchwork import AMRHierarchy

        periodic = (boundaries[0] == "periodic", boundaries[1] == "periodic")
        bounds = ((edges[0][0], edges[0][-1]), (edges[1][0], edges[1][-1]))
        hierarchy = AMRHierarchy(
            (shape[0], shape[1]), bounds, periodic=periodic, geometry=geometry
        )
        uniform = [float(s.max()) <= float(s.min()) * (1.0 + 1e-6) for s in spacing]
        # a uniform axis: every cell at least as fine as its spacing
        min_level = max(
            (
                int(np.ceil(np.log2(hierarchy.spacing(0)[axis] / h_min[axis]) - 1e-9))
                for axis in (0, 1)
                if uniform[axis]
            ),
            default=0,
        )
        if levels:
            tags = {}
            for level in range(levels):
                shape_l = hierarchy.level_shape(level)
                need = np.full(shape_l, level < min_level)
                stretched = [axis for axis in (0, 1) if not uniform[axis]]
                if stretched:
                    fine = np.ones(shape_l, dtype=bool)
                    for axis in stretched:
                        h = hierarchy.spacing(level)[axis]
                        lower = edges[axis][0] + h * np.arange(shape_l[axis])
                        finest = _min_overlapping(edges[axis], lower, lower + h)
                        row = h > finest * (1.0 + 1e-9)
                        fine &= row[:, None] if axis == 0 else row[None, :]
                    need |= fine
                tags[level] = need
            hierarchy = hierarchy.regrid(tags, buffer=buffer, efficiency=efficiency)
        return cls(hierarchy, boundaries=boundaries)

    def refined_where(
        self, where: np.ndarray, *, buffer: int = 1, efficiency: float = 0.8
    ) -> PatchGrid:
        """The same grid with the leaf cells ``where`` selects one level finer.

        The rest keeps its refinement. E.g. the skin layer of a solution:
        ``grid.refined_where(solution.power_density > 0.05 * solution.power_density.max())``.
        """
        hot = np.asarray(where, dtype=bool)
        if hot.shape != self.shape:
            raise ValueError(f"expected a mask over the {self.size} leaf cells")
        hierarchy = self.hierarchy
        ratio = hierarchy.refinement_ratio
        level_of = np.array([hierarchy.patches[p].level for p in self.cells.patch])
        levels = int(level_of.max()) + (1 if hot.any() else 0)
        tags = {
            level: np.zeros(hierarchy.level_shape(level), dtype=bool)
            for level in range(levels)
        }
        index = self.cells.index
        for level in range(levels):
            # the ancestors of finer leaf cells keep their refinement; hot cells ask
            # for one more level
            keep = (level_of > level) | (hot & (level_of == level))
            scale = ratio ** (level_of[keep] - level)
            tags[level][index[keep, 0] // scale, index[keep, 1] // scale] = True
        if not levels:
            return self
        refined = hierarchy.regrid(tags, buffer=buffer, efficiency=efficiency)
        return PatchGrid(refined, boundaries=self.boundaries)  # type: ignore[arg-type]

    # --- the CellGrid interface ----------------------------------------------------

    @property
    def cylindrical(self) -> bool:
        """Whether the grid is the axisymmetric r–z plane."""
        return self.geometry == "cylindrical"

    @property
    def on_axis(self) -> bool:
        """Whether the r–z domain reaches the axis."""
        return bool(self.hierarchy.on_axis)

    @property
    def coordinate_names(self) -> tuple[str, str]:
        """``("x", "y")`` or ``("r", "z")``."""
        return ("r", "z") if self.cylindrical else ("x", "y")

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions of a field in a dataset: one, the leaf cells."""
        return ("cell",)

    @property
    def num_levels(self) -> int:
        """The number of refinement levels (1: the base alone)."""
        return int(self.hierarchy.max_level) + 1

    def cell_field(self, value: CellField, dtype: Any = float) -> np.ndarray:
        """A :data:`CellField` evaluated on the cell centres, shape ``(n,)``."""
        if callable(value):
            result = np.asarray(value(self.xx, self.yy), dtype=dtype)
        else:
            result = np.asarray(value, dtype=dtype)
        return np.broadcast_to(result, self.shape).copy()

    def mask(self, shape: Any) -> np.ndarray:
        """The cells whose centre a shape (anything with ``contains_points``) contains."""
        inside = shape.contains_points(np.stack([self.xx, self.yy]))
        if hasattr(inside, "get"):
            inside = inside.get()
        return np.asarray(inside, dtype=bool).reshape(self.shape)

    def union_mask(self, shapes: Sequence[Any]) -> np.ndarray:
        """The cells any of ``shapes`` contains."""
        result = np.zeros(self.shape, dtype=bool)
        for shape in shapes:
            result |= self.mask(shape)
        return result

    def curl_curl(
        self,
        face_coefficient: np.ndarray | None = None,
        active: np.ndarray | None = None,
    ) -> sp.csr_matrix:
        """The curl-curl of the normal field, integrated over the cells (symmetric).

        As :meth:`CellGrid.curl_curl`: ``face_coefficient`` scales each face by the mean
        of its two cells; with ``active`` the other cells are a body held at 0 with its
        surface on their faces, and their rows and columns are zero.
        """
        if face_coefficient is None and active is None:
            return self._operator.stiffness.copy()
        from patchwork import CellOperator, cell_body

        live = (
            np.ones(self.size, dtype=bool)
            if active is None
            else np.asarray(active, dtype=bool).reshape(self.shape)
        )
        mu = (
            np.ones(self.size)
            if face_coefficient is None
            else np.broadcast_to(face_coefficient, self.shape).astype(float)
        )
        bodies = [] if live.all() else [cell_body(self.cells, ~live, 0.0)]
        operator = CellOperator(
            self.hierarchy,
            diffusion=np.where(live, mu, 1.0),
            component=self._component,
            faces=self._faces,
            embedded=bodies,
            mean="arithmetic",
            cells=self.cells,
            factorize=False,
        )
        return operator.stiffness

    def curl(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The in-plane curl of the normal field ``a`` on the cell centres.

        ``(d_y a, -d_x a)`` in the cut; ``(-d_z a, (1 / r) d(r a) / dr)`` in r–z.
        """
        result = self._operator.curl(np.asarray(a).reshape(self.shape))
        return result[:, 0], result[:, 1]

    def net_current(self, residual: np.ndarray, region: np.ndarray) -> Any:
        """The current through a region's cross-section from ``V J`` on its cells."""
        return np.sum(residual[region] / self.circumference[region])

    def wall_cells(self, inside: np.ndarray) -> np.ndarray:
        """The cells of ``inside`` with a face to a cell outside it or to the domain's edge.

        The axis of an r–z domain and periodic seams are not walls.
        """
        inside = np.asarray(inside, dtype=bool)
        result = np.zeros(self.size, dtype=bool)
        for axis in (0, 1):
            faces = self.cells.faces(axis)
            low, high = faces["low"], faces["high"]
            crossing = inside[low] != inside[high]
            result[low[crossing & inside[low]]] = True
            result[high[crossing & inside[high]]] = True
            boundary = self.cells.boundary(axis)
            edge = boundary["cell"]
            if axis == 0 and self.on_axis:
                edge = edge[~(boundary["rectangle"] & (boundary["side"] == 0))]
            result[edge[inside[edge]]] = True
        return result & inside

    def transport_faces(self) -> TransportFaces:
        """The faces of the leaf cells for finite-volume transport, across levels too.

        Between two leaf cells of different levels each face is as long as the finer
        cell's side (patchwork's :meth:`~patchwork.CompositeCells.faces`): one flux
        per face, which the coarse cell sums over its faces on that side, is what
        patchwork's refluxing (:class:`~patchwork.FluxRegister`) restores after an
        explicit step; a scheme that takes these faces has it from the start. Their
        ``coarse_side``, ``coarse_neighbours`` and ``coarse_offset`` (see
        :class:`TransportFaces`) locate the coarse value along the face.
        """
        cells = self.cells
        centres = np.stack([self.x, self.y], axis=1)
        (x0, x1), (y0, y1) = self.hierarchy.bounds
        length = (x1 - x0, y1 - y0)
        interior, boundary = [], []
        for axis in (0, 1):
            faces = cells.faces(axis)
            low, high = faces["low"], faces["high"]
            area, position, _ = cells.face_metric(
                axis, faces["plane"], faces["start"], faces["count"]
            )
            delta_low = position - centres[low, axis]
            delta_high = centres[high, axis] - position
            # across a periodic seam the face is at the upper edge
            delta_high = np.where(
                delta_high < 0.0, delta_high + length[axis], delta_high
            )
            interior.append(
                (
                    low,
                    high,
                    np.full(low.shape, axis),
                    area,
                    delta_low,
                    delta_high,
                    *self._coarse_fine(axis, low, high, faces["start"], faces["count"]),
                )
            )
            if self.boundaries[axis] == "periodic":
                continue
            edge = cells.boundary(axis)
            area, position, _ = cells.face_metric(
                axis, edge["plane"], edge["start"], edge["count"]
            )
            cell = edge["cell"]
            boundary.append(
                (
                    cell,
                    np.full(cell.shape, axis),
                    edge["side"],
                    area,
                    np.abs(position - centres[cell, axis]),
                    np.full(cell.shape, self.boundaries[axis] == "neumann"),
                )
            )
        return TransportFaces.concatenate(interior, boundary)

    def _coarse_fine(
        self,
        axis: int,
        low: np.ndarray,
        high: np.ndarray,
        start: np.ndarray,
        count: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The coarser cell of faces between levels, its neighbours and the offset.

        As :class:`TransportFaces` has them: ``coarse_side``, ``coarse_neighbours``
        (the leaf cells of the coarser one's size next to it along the face, on the
        finest lattice; periodic axes wrap) and ``coarse_offset``.
        """
        cells = self.cells
        span_low, span_high = cells.span[low], cells.span[high]
        side = np.where(
            span_low > span_high, 0, np.where(span_high > span_low, 1, -1)
        ).astype(int)
        neighbours = np.full((low.size, 2), -1, dtype=int)
        offset = np.zeros(low.size)
        mixed = np.flatnonzero(side >= 0)
        if not mixed.size:
            return side, neighbours, offset
        coarse = np.where(side[mixed] == 0, low[mixed], high[mixed])
        other = 1 - axis
        span = cells.span[coarse]
        lattice = cells.lattice[coarse]
        middle = start[mixed] + 0.5 * count[mixed]
        offset[mixed] = (middle - (lattice[:, other] + 0.5 * span)) / span
        size = cells.owner.shape[other]
        for column, sign in enumerate((-1, 1)):
            spot = lattice.copy()
            spot[:, other] += sign * span
            if self.hierarchy.periodic[other]:
                spot[:, other] %= size
            inside = (spot[:, other] >= 0) & (spot[:, other] < size)
            found = np.full(coarse.size, -1, dtype=int)
            found[inside] = cells.owner[tuple(spot[inside].T)]
            valid = found >= 0
            valid[valid] &= (cells.span[found[valid]] == span[valid]) & np.all(
                cells.lattice[found[valid]] == spot[valid], axis=1
            )
            neighbours[mixed, column] = np.where(valid, found, -1)
        return side, neighbours, offset

    def locate(self, x: float, y: float) -> int:
        """The leaf cell containing the point ``(x, y)`` (-1 outside the domain)."""
        return int(self.cells.locate(np.array([[x], [y]], dtype=float))[0])

    def coordinates(self) -> dict[str, tuple[str, np.ndarray, dict[str, str]]]:
        """The cell-centre coordinates as xarray coordinate entries along ``cell``."""
        first, second = self.coordinate_names
        return {
            first: ("cell", self.x, {"long_name": first, "units": "m"}),
            second: ("cell", self.y, {"long_name": second, "units": "m"}),
        }


def _level_tags(
    hierarchy: AMRHierarchy,
    level: int,
    function: Callable[[np.ndarray, np.ndarray, int], Any],
) -> np.ndarray:
    """``function`` on the cell centres of a whole level, as tags of that level."""
    dx, dy = hierarchy.spacing(level)
    (x0, _), (y0, _) = hierarchy.bounds
    nx, ny = hierarchy.level_shape(level)
    xx, yy = np.meshgrid(
        x0 + (np.arange(nx) + 0.5) * dx, y0 + (np.arange(ny) + 0.5) * dy, indexing="ij"
    )
    return np.broadcast_to(np.asarray(function(xx, yy, level), dtype=bool), (nx, ny))


def _min_overlapping(
    edges: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """The smallest spacing of the cells of ``edges`` each interval overlaps."""
    spacing = np.diff(edges)
    first = np.clip(
        np.searchsorted(edges, lower, side="right") - 1, 0, spacing.size - 1
    )
    last = np.clip(np.searchsorted(edges, upper, side="left"), first + 1, spacing.size)
    # the minimum over spacing[first:last] for every interval (a sparse table)
    table = [spacing]
    while 2 ** len(table) <= spacing.size:
        previous, step = table[-1], 2 ** (len(table) - 1)
        table.append(np.minimum(previous[:-step], previous[step:]))
    width = last - first
    power = np.floor(np.log2(width)).astype(int)
    result = np.empty(lower.shape)
    for k in np.unique(power):
        mine = power == k
        start, stop = first[mine], last[mine] - 2**k
        result[mine] = np.minimum(table[k][start], table[k][stop])
    return result
