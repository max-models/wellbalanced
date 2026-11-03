"""The time-harmonic magnetoquasistatic field in 3D, on the edges of a Cartesian grid.

The 3D form of `wellbalanced.spatial.inductive.InductiveField2D` (step R4 of
the RF-driver plan): the vector potential on the edges of a uniform
`wellbalanced.spatial.edges.EdgeGrid` (finite integration, Yee's staggering),
in the temporal gauge (``E = -i omega A``):

    curl (nu curl A) + i omega sigma A = J_coil,   nu = 1 / mu0,

discretised as ``(C^T M_nu C + i omega M_sigma) a = j``, ``a`` the line integrals of
``A`` on the edges, ``C`` the curl (face–edge incidence), ``M_nu`` the diagonal
``nu L*_f / A_f`` of the faces, ``M_sigma`` the diagonal ``sigma_e A*_e / L_e`` of the
edges (``L*`` and ``A*`` the dual lengths and areas) and ``j`` the coil current through
the dual faces. The displacement current is dropped, as in 2D.

Coils: `Coil3D`, the current on the surface of a region, ``J = curl T`` with
``T = n I e`` inside it (``n`` turns per metre around the direction ``e``). Discretely
``j = C^T (T L*)``, so ``G^T j = 0`` exactly: the coil current is divergence-free on the
grid whatever the shape. A solenoid is a cylinder with ``T`` along its axis; a
rectangular loop is a box with ``T`` normal to the loop; the sheet sits on the region's
staircased surface. A coil of real wire (copper tube, with its own losses and proximity
effect) is an `ImpedanceConductor3D` driven by a `Feed3D`: the current along grid
edges across a gap between the coil's terminals (a delta-gap source), which returns
through the metal; the terminals must lie in one connected piece of metal on the grid
(about two cells across the wire).

Conductors: `Conductor3D`, a conductivity on the edges inside shapes (the
midpoint rule, staircased; or a sub-cell average, ``samples``), and a background
``conductivity`` (a function of the points, complex for a cold plasma). Good
conductors whose skin depth is below the grid (copper at MHz, a Faraday shield) are
`ImpedanceConductor3D`: their inside drops out and Leontovich's ``E_t = Z_s H x n``
holds on their surface, on cut cells (Dey and Mittra's conformal faces with a surface
impedance; the geometry from simgeometry's ``yee_fractions``,
`wellbalanced.spatial.cutcell`): a cut face's reluctance is
``L*_f / (mu0 A_f^air - i Z_s w_f / omega)``, ``A_f^air`` its area outside and ``w_f``
the surface's width in it. Second order for curved surfaces, exact for planes, and a slit
narrower than a cell keeps its gap; staircased (``cut_cells=False``) the surface is too
long (by ``4 / pi`` for a circle) and the loss does not converge.

Gauge: in the insulator ``C^T M_nu C`` is singular (every gradient ``G phi`` is in its
null space). A small real regularisation ``eps (nu / l^2) M_1 a`` (``M_1`` the diagonal
``A*_e / L_e``, ``l`` the largest side of the box) makes the matrix regular; projected
on the gradients it is the Coulomb gauge ``div A = 0`` in the insulator, weakly, and as
the coil current is discretely divergence-free it changes the curl of ``a`` only by a
relative ``O(eps)``: being real it adds no loss, and it shifts the inductance by
``O(eps)`` (checked in ``tests/unit/test_inductive3d.py``). A tree–cotree gauge is exact
but needs a spanning tree of the insulator, conforming to the conductors and the
boundaries, and makes the matrix harder for the iterative solvers a larger grid will
need; the regularisation keeps every edge an unknown, the same unknowns as picnic's
fields.

Solve: one sparse direct LU (SuperLU, in the grid's nested-dissection order) per
frequency, for small grids: about a minute for 65 000 unknowns (24 x 24 x 40 cells), two
for 130 000, and the cost grows as the square of the unknowns. Larger grids:
``solve(solver="iterative")``, COCG with Hiptmair and Xu's auxiliary-space
preconditioner and a geometric multigrid (`wellbalanced.spatial.iterative`),
whose iterations hardly grow with the grid.

Diagnostics: the complex power the coils deliver, ``S = 1/2 i omega a . conj(j)``, the
impedance ``Z = 2 S / |I|^2 = R + i omega L``, the losses ``1/2 omega^2 a^H M_sigma a`` per
conductor (``Re S`` to round-off), and ``B`` on the faces.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from methodcard import Rule, method
from scipy.constants import mu_0 as MU_0

from wellbalanced.spatial.cutcell import cut_cells as cut_cells_of
from wellbalanced.spatial.cutcell import staircased_cells
from wellbalanced.spatial.edges import EdgeGrid
from wellbalanced.spatial.iterative import AuxiliarySpacePreconditioner, cocg

__all__ = [
    "Coil3D",
    "Conductor3D",
    "Feed3D",
    "ImpedanceConductor3D",
    "InductiveField3D",
    "InductiveSolution3D",
]

#: A conductivity [S/m] as a function of ``(3, n)`` points, returning ``n`` values.
PointField = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Coil3D:
    """A coil as the current sheet on the surface of a region: ``J = curl T``.

    ``T = turns_per_length * I * direction`` inside ``shape`` and zero outside, so the
    sheet carries ``turns_per_length * I`` amperes per metre (along ``direction``),
    circulating around ``direction``. A solenoid of ``N`` turns over a length ``l`` on a
    `simgeometry.Cylinder` along ``z``: ``Coil3D(cylinder, (0, 0, 1), N / l)``; a
    rectangular loop of one turn, ``w`` wide along ``y``, in the x–z plane:
    ``Coil3D(box, (0, 1, 0), 1 / w)``.

    Attributes:
        shape: The region (anything with ``contains_points`` of ``(3, n)`` points).
        direction: The direction of ``T``, the axis the current circulates around.
        turns_per_length: Turns per metre along ``direction`` [1/m].
        name: A label.
    """

    shape: Any
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    turns_per_length: float = 1.0
    name: str = ""


@dataclass(frozen=True)
class Feed3D:
    """A current source along grid edges: the terminals of a coil of real wire.

    The coil current flows along the edges of the grid from the node nearest ``start``
    to the node nearest ``end``, axis by axis in ``order``. Both terminals must lie
    in the metal of the same `ImpedanceConductor3D` (or of the same Dirichlet
    wall): the current enters that conductor at one end of the path, flows on its
    surface, and leaves it at the other, so the coil's own Joule loss and its proximity
    effect are in the solution. A helical coil with its two leads and a gap between
    their ends: the feed bridges the gap, from inside one lead to inside the other
    (a delta-gap source; ``V = -i omega a`` summed along the path is the terminal
    voltage). The edges of the path inside the metal are not unknowns and drop out.

    Attributes:
        start: The first terminal (a point inside the metal) [m].
        end: The second terminal [m].
        order: The axes the path follows, in turn.
        name: A label.
    """

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    order: tuple[int, int, int] = (0, 1, 2)
    name: str = ""


@dataclass(frozen=True)
class Conductor3D:
    """A conductor of uniform conductivity: the union of shapes.

    Attributes:
        shapes: The shapes it is made of (Booleans for slits, e.g. ``tube - box``).
        conductivity: ``sigma`` [S/m], real for a metal.
        name: A label for `InductiveSolution3D.conductor_losses`.
    """

    shapes: Sequence[Any]
    conductivity: float
    name: str = ""


def _impedance_rules() -> dict:
    """The SymPy rule of the surface resistance of `ImpedanceConductor3D` (for the tests)."""
    import sympy

    f, sigma = sympy.symbols("frequency conductivity", positive=True)
    rule = sympy.sqrt(2 * sympy.pi * f * sympy.Float(MU_0) / (2 * sigma))
    return {
        "resistance": Rule(
            rule,
            implementation=lambda frequency, conductivity: (
                ImpedanceConductor3D((), conductivity).surface_impedance(frequency).real
            ),
            ranges={"frequency": (0.5e6, 4.0e6), "conductivity": (1.0e6, 6.0e7)},
            rtol=1e-12,
        )
    }


@method(
    equations={
        "leontovich": r"\mathbf E_t = Z_s\,\mathbf H\times\mathbf n, \qquad Z_s = (1 + i)\sqrt{\frac{\omega\mu_0}{2\sigma}} = \frac{1 + i}{\sigma\delta}",
        "resistance": r"\mathrm{Re}\,Z_s = \sqrt{\frac{\omega\mu_0}{2\sigma}}",
        "loss": r"P = \tfrac12\mathrm{Re}(Z_s)\,|H_t|^2 \text{ per area}",
    },
    assumptions=(
        r"a good conductor whose skin depth is small against its thickness and radii of curvature; $\mathbf n$ into the metal",
    ),
    symbolic=_impedance_rules,
)
@dataclass(frozen=True)
class ImpedanceConductor3D:
    """A good conductor as a surface-impedance (Leontovich) boundary on cut cells.

    Its inside is not part of the field: the edges entirely inside drop out, and on its
    surface ``E_t = Z_s H x n`` (``n`` into the metal) with
    ``Z_s = (1 + i) / (sigma delta)``, ``delta = sqrt(2 / (omega mu0 sigma))`` the skin
    depth. Valid where the skin depth is small against the conductor's thickness and its
    radii of curvature (e.g. copper at MHz). The losses are ``1/2 Re(Z_s) |H_t|^2`` on the
    surface.

    Attributes:
        shapes: The shapes it is made of (their union; Booleans for slits).
        conductivity: ``sigma`` [S/m].
        name: A label for `InductiveSolution3D.conductor_losses`.
    """

    shapes: Sequence[Any]
    conductivity: float
    name: str = ""

    def surface_impedance(self, frequency: float) -> complex:
        """``Z_s = (1 + i) sqrt(omega mu0 / (2 sigma))`` [Ohm]."""
        omega = 2.0 * np.pi * frequency
        return (1.0 + 1.0j) * np.sqrt(omega * MU_0 / (2.0 * self.conductivity))


def _curl(grid: EdgeGrid) -> sp.csr_matrix:
    if not hasattr(grid, "_curl_matrix"):
        grid._curl_matrix = grid.curl()  # type: ignore[attr-defined]
    return grid._curl_matrix  # type: ignore[attr-defined]


@method(
    equations={
        "power": r"S = \tfrac12 i\omega\, a\cdot\bar j, \qquad Z = \frac{2S}{|I|^2} = R + i\omega L",
        "losses": r"P_\sigma = \tfrac12\omega^2 a^H M_\sigma a, \qquad P_Z = \tfrac12\sum_f\mathrm{Re}(Z_s)|H_f|^2 w_f L^*_f",
        "field": r"B_f = \mu_0 H_f \quad \left(\frac{b_f}{A_f} \text{ without impedance conductors}\right)",
    },
    assumptions=(r"the losses sum to $\mathrm{Re}\,S$ to round-off",),
)
@dataclass(frozen=True)
class InductiveSolution3D:
    """The time-harmonic field of `InductiveField3D.solve`, as complex amplitudes.

    Attributes:
        grid: The edge grid.
        frequency: The frequency [Hz].
        coil_current: The coil current amplitude ``I`` [A].
        potential: ``a``, the line integrals of ``A`` on the edges [Wb].
        source: ``j``, the coil current through the dual faces of the edges [A].
        edge_conductance: ``sigma_e A*_e / L_e`` per edge [S m].
        conductor_edges: The edges of each conductor, by name.
        face_areas: The area of every face outside the impedance conductors [m^2]
            (the grid's face areas without them).
        face_field_factor: ``H_f = factor * b_f`` on every face [A/(m Wb)]: ``1 /
            (mu0 A_f)``, or with impedance conductors ``1 / D_f`` (see
            `InductiveField3D`).
        surface_losses: The power dissipated in each impedance conductor [W].
    """

    grid: EdgeGrid
    frequency: float
    coil_current: complex
    potential: np.ndarray
    source: np.ndarray
    edge_conductance: np.ndarray
    conductor_edges: dict[str, np.ndarray] = field(default_factory=dict)
    face_areas: np.ndarray | None = None
    face_field_factor: np.ndarray | None = None
    surface_losses: dict[str, float] = field(default_factory=dict)

    @property
    def omega(self) -> float:
        """The angular frequency [1/s]."""
        return 2.0 * np.pi * self.frequency

    @property
    def electric_field(self) -> np.ndarray:
        """``E . t = -i omega a / L`` on the edges [V/m]."""
        return -1j * self.omega * self.potential / self.grid.edge_lengths

    @property
    def magnetic_flux(self) -> np.ndarray:
        """``b = C a``, the flux through every face [Wb]."""
        return _curl(self.grid) @ self.potential

    @property
    def magnetic_field(self) -> np.ndarray:
        """``B . n`` on the faces [T] (``mu0 H . n``; zero inside impedance conductors)."""
        if self.face_field_factor is None:
            return self.magnetic_flux / self.grid.face_areas
        return MU_0 * self.face_field_factor * self.magnetic_flux

    @property
    def magnetic_field_at_cells(self) -> np.ndarray:
        """``B`` at the cell centres, ``(3, nx, ny, nz)`` [T]."""
        return self.grid.face_values_at_cells(self.magnetic_field)

    @property
    def current(self) -> np.ndarray:
        """The induced current through the dual face of every edge [A]."""
        return self.edge_conductance * (-1j * self.omega * self.potential)

    def _loss(self, edges: np.ndarray | None = None) -> float:
        density = 0.5 * self.omega**2 * np.real(self.edge_conductance)
        values = density * np.abs(self.potential) ** 2
        return float(np.sum(values if edges is None else values[edges]))

    @property
    def loss(self) -> float:
        """The power dissipated in all conductors [W]."""
        return self._loss() + sum(self.surface_losses.values())

    @property
    def conductor_losses(self) -> dict[str, float]:
        """The power dissipated in each conductor [W]."""
        losses = {
            name: self._loss(edges) for name, edges in self.conductor_edges.items()
        }
        return {**losses, **self.surface_losses}

    @property
    def source_power(self) -> complex:
        """The complex power the coils deliver, ``1/2 i omega a . conj(j)`` [W]."""
        return complex(
            0.5j * self.omega * np.sum(self.potential * np.conj(self.source))
        )

    @property
    def impedance(self) -> complex:
        """The coil's impedance ``2 S / |I|^2 = R + i omega L`` [Ohm]."""
        return 2.0 * self.source_power / abs(self.coil_current) ** 2

    @property
    def resistance(self) -> float:
        """The coil's resistance from the induced losses [Ohm]."""
        return float(np.real(self.impedance))

    @property
    def inductance(self) -> float:
        """The coil's inductance [H]."""
        return float(np.imag(self.impedance)) / self.omega


@method(
    equations={
        "field": r"\nabla\times(\nu\nabla\times\mathbf A) + i\omega\sigma\mathbf A = \mathbf J_\mathrm{coil}, \qquad \nu = \frac{1}{\mu_0}, \qquad \mathbf E = -i\omega\mathbf A",
        "system": r"\left(C^T M_\nu C + i\omega M_\sigma + M_g\right) a = j, \qquad a_e = \int_e\mathbf A\cdot d\mathbf l",
        "hodge": r"M_\nu = \mathrm{diag}\frac{L^*_f}{D_f}, \qquad M_\sigma = \mathrm{diag}\frac{\sigma_e A^*_e}{L_e}, \qquad M_g = \frac{\epsilon}{\mu_0 l^2}\mathrm{diag}\frac{A^*_e}{L_e}",
        "coil": r"\mathbf J = \nabla\times\mathbf T, \quad \mathbf T = n I\hat{\mathbf e} \text{ inside the coil}, \qquad j = C^T(T L^*) \ \Rightarrow\ G^T j = 0",
        "cut_face": r"D_f = \mu_0 A^\mathrm{air}_f - \frac{i Z_s w_f}{\omega}, \qquad H_f = \frac{b_f}{D_f}",
    },
    assumptions=(
        r"finite integration on a uniform Yee grid, temporal gauge, no displacement current",
        r"$\epsilon$ = `regularization` ($10^{-6}$): a weak Coulomb gauge in the insulator, real (no loss), shifting the inductance by $O(\epsilon)$; $l$ the box's largest side",
        r"$A^\mathrm{air}_f$ floored at `min_air_fraction` $\cdot A_f$ in the reluctance only; faces with $D_f = 0$ get zero",
        r"volume conductors on the edges by the midpoint or `samples`$^3$ sub-points (harmonic along, arithmetic across the edge)",
        r"direct: SuperLU in a nested-dissection order without pivoting; iterative: COCG with the auxiliary-space preconditioner",
    ),
    references=(
        "[Yee, IEEE Trans. Antennas Propag. 14, 302 (1966)](https://doi.org/10.1109/TAP.1966.1138693)",
        "Weiland, AEÜ 31, 116 (1977) (finite integration)",
        "Dey and Mittra (conformal cut faces)",
    ),
    components=(ImpedanceConductor3D, InductiveSolution3D),
)
class InductiveField3D:
    """The 3D time-harmonic magnetoquasistatic field of coils and conductors.

    Args:
        grid: The edge grid (its boundaries are the field's).
        frequency: The frequency [Hz].
        coils: The coils, all carrying the coil current.
        feeds: Current sources along grid edges (`Feed3D`), also carrying the
            coil current: the terminals of coils of real wire, modelled as
            `ImpedanceConductor3D`.
        conductors: Conductors of uniform conductivity (must not overlap).
        conductivity: A background conductivity [S/m] as a function of ``(3, n)``
            points, e.g. a cold plasma's complex ``sigma``; added outside the
            conductors.
        samples: ``1``: a conductor's edges are those whose midpoint it contains (a
            staircased region). ``s > 1``: the sub-cell conductance of every edge's dual
            cell from ``s^3`` points (`EdgeGrid.edge_average`).
        regularization: ``eps`` of the gauge term (relative to ``nu / l^2``).
        impedance_conductors: Good conductors as surface-impedance boundaries
            (`ImpedanceConductor3D`), on cut cells.
        cut_cells: ``True``: the impedance conductors' cut cells are conformal
            (`wellbalanced.spatial.cutcell.cut_cells`, simgeometry shapes);
            ``False``: staircased (`wellbalanced.spatial.cutcell.staircased_cells`).
        min_air_fraction: The smallest air fraction of a cut face's area in its
            reluctance (the "small cell" bound; the flux and the losses keep the face's
            true area otherwise).
        min_sine: The bound of ``sin(theta)`` in the cut faces' surface widths
            (`wellbalanced.spatial.cutcell.cut_cells`).
        cut_cell_samples: The sampling of the conductors' crossings along each edge
            and line of the cut cells (simgeometry's ``yee_fractions``), for shapes
            without analytic crossings (Booleans, helices); fewer is faster, and
            features thinner than a ``samples``-th of a cell can be missed.
    """

    # --- sources and materials -----------------------------------------------------

    def _coil_source(self) -> np.ndarray:
        """``j = C^T (T L*)`` for 1 A [A], with ``T`` on the faces inside each coil.

        A face's ``T`` is the coil's indicator averaged over its dual edge (its two
        halves' midpoints, the one inside the box on a boundary face), so that a face
        on the coil's surface, e.g. in the plane of a solenoid's end, counts half: with
        the indicator at the face's centre such a coil was half a cell too long.
        """
        grid = self.grid
        centres = grid.face_centres()
        normals = grid.face_axes()
        lower, upper = (
            np.zeros(grid.n_faces, dtype=bool),
            np.zeros(grid.n_faces, dtype=bool),
        )
        for a in range(3):
            first, last = grid._ends(grid.face_shape(a), a)
            block = slice(grid.face_offsets[a], grid.face_offsets[a + 1])
            lower[block], upper[block] = first.ravel(), last.ravel()
        quarter = 0.25 * grid.spacing[normals]
        samples = []
        for sign, beyond in ((-1.0, lower), (1.0, upper)):
            points = centres.copy()
            points[normals, np.arange(grid.n_faces)] += sign * quarter
            samples.append((points, ~beyond))
        t = np.zeros(grid.n_faces)
        for coil in self.coils:
            direction = np.asarray(coil.direction, dtype=float)
            if direction.shape != (3,) or not np.any(direction):
                raise ValueError("a coil needs a nonzero 3D direction")
            direction = direction / np.linalg.norm(direction)
            weight = np.zeros(grid.n_faces)
            for points, valid in samples:
                weight += valid * grid.contains(coil.shape, points)
            weight /= samples[0][1].astype(float) + samples[1][1]
            if not weight.any():
                raise ValueError(f"coil {coil.name!r} contains no face")
            t += weight * coil.turns_per_length * direction[normals]
        return np.asarray(_curl(grid).T @ (t * grid.dual_lengths))

    def _feed_edges(self, feed: Feed3D) -> tuple[np.ndarray, np.ndarray, list]:
        """The edges of a feed's path, their signs and the two terminal nodes."""
        grid = self.grid
        nodes = []
        for point in (feed.start, feed.end):
            index = np.rint(
                (np.asarray(point, dtype=float) - grid.lower) / grid.spacing
            )
            index = index.astype(int)
            if np.any(index < 0) or np.any(index >= np.asarray(grid.nodes)):
                raise ValueError(f"feed {feed.name!r}: a terminal is outside the grid")
            nodes.append(index)
        if sorted(feed.order) != [0, 1, 2]:
            raise ValueError("a feed's order is a permutation of the three axes")
        edges, signs = [], []
        current = nodes[0].copy()
        for axis in feed.order:
            step = 1 if nodes[1][axis] >= current[axis] else -1
            while current[axis] != nodes[1][axis]:
                low = current.copy()
                if step < 0:
                    low[axis] -= 1
                edges.append(
                    int(grid._edge_index(axis, [np.array([i]) for i in low])[0])
                )
                signs.append(float(step))
                current[axis] += step
        if not edges:
            raise ValueError(f"feed {feed.name!r}: its terminals are the same node")
        flat = [int(np.ravel_multi_index(tuple(n), grid.nodes)) for n in nodes]
        return np.asarray(edges), np.asarray(signs), flat

    def _feed_source(self) -> np.ndarray:
        """``j`` of the feeds for 1 A [A]: the current along each path's edges.

        Raises if a feed's terminals are not in the same connected piece of metal (the
        nodes joined by edges that are not unknowns): then its current would have no
        return path on the grid.
        """
        grid = self.grid
        source = np.zeros(grid.n_edges)
        if not self.feeds:
            return source
        import scipy.sparse.csgraph as csgraph

        gradient = abs(self._gradient_matrix())
        metal = sp.diags((~self.free_edges).astype(float))
        graph = (gradient.T @ metal @ gradient).tocsr()
        _, piece = csgraph.connected_components(graph, directed=False)
        isolated = np.asarray(graph.sum(axis=1)).ravel() == 0
        for feed in self.feeds:
            edges, signs, (first, last) = self._feed_edges(feed)
            if isolated[first] or isolated[last] or piece[first] != piece[last]:
                raise ValueError(
                    f"feed {feed.name!r}: its terminals must be inside the metal of"
                    " one conductor (or one Dirichlet wall)"
                )
            np.add.at(source, edges, signs)
        return source

    def _gradient_matrix(self) -> sp.csr_matrix:
        grid = self.grid
        if not hasattr(grid, "_gradient_matrix"):
            grid._gradient_matrix = grid.gradient()  # type: ignore[attr-defined]
        return grid._gradient_matrix  # type: ignore[attr-defined]

    def _edge_conductivity(
        self, background: PointField | None
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        grid = self.grid
        midpoints = grid.edge_midpoints()
        sigma = np.zeros(grid.n_edges, dtype=complex)
        covered = np.zeros(grid.n_edges, dtype=bool)
        edges: dict[str, np.ndarray] = {}
        for k, conductor in enumerate(self.conductors):

            def inside(points, shapes=conductor.shapes):
                result = np.zeros(points.shape[1], dtype=bool)
                for shape in shapes:
                    result |= grid.contains(shape, points)
                return result

            if self.samples == 1:
                mask = inside(midpoints)
                value = np.where(mask, conductor.conductivity, 0.0)
            else:
                value = grid.edge_average(
                    lambda p, inside=inside, s=conductor.conductivity: np.where(
                        inside(p), s, 0.0
                    ),
                    self.samples,
                )
                mask = value != 0
            if np.any(covered & mask):
                raise ValueError("conductors must not overlap")
            covered |= mask
            sigma += value
            edges[conductor.name or f"conductor {k}"] = mask
        if background is not None:
            values = grid.edge_average(
                lambda p: np.asarray(background(p), dtype=complex), self.samples
            )
            sigma[~covered] += values[~covered]
            plasma = (values != 0) & ~covered
            if plasma.any():
                edges["background"] = plasma
        return sigma, edges

    def __init__(
        self,
        grid: EdgeGrid,
        *,
        frequency: float,
        coils: Sequence[Coil3D] = (),
        feeds: Sequence[Feed3D] = (),
        conductors: Sequence[Conductor3D] = (),
        conductivity: PointField | None = None,
        samples: int = 1,
        regularization: float = 1e-6,
        impedance_conductors: Sequence[ImpedanceConductor3D] = (),
        cut_cells: bool = True,
        min_air_fraction: float = 0.01,
        min_sine: float = 0.1,
        cut_cell_samples: int = 64,
    ) -> None:
        """Sample the coils and conductors on the grid and assemble the operator."""
        if frequency <= 0.0:
            raise ValueError("frequency must be positive")
        if regularization <= 0.0:
            raise ValueError("the gauge needs a positive regularization")
        self.grid = grid
        self.frequency = float(frequency)
        self.coils = tuple(coils)
        self.conductors = tuple(conductors)
        self.samples = int(samples)
        self.regularization = float(regularization)
        self._unit_source = self._coil_source()
        self.conductivity, self.conductor_edges = self._edge_conductivity(conductivity)
        self.edge_conductance = self.conductivity * grid.dual_areas / grid.edge_lengths
        self.impedance_conductors = tuple(impedance_conductors)
        self.cut_cells = bool(cut_cells)
        if not 0.0 < min_air_fraction <= 1.0:
            raise ValueError("min_air_fraction must be in (0, 1]")
        self.min_air_fraction = float(min_air_fraction)
        self._surfaces = [
            cut_cells_of(grid, c.shapes, min_sine=min_sine, samples=cut_cell_samples)
            if self.cut_cells
            else staircased_cells(grid, c.shapes)
            for c in self.impedance_conductors
        ]
        metal_edges = sum((1.0 - c.edge_air for c in self._surfaces), np.zeros(1))
        metal_faces = sum((1.0 - c.face_air for c in self._surfaces), np.zeros(1))
        #: The fraction of every edge's length outside the impedance conductors.
        self.edge_air = np.clip(1.0 - metal_edges, 0.0, 1.0) * np.ones(grid.n_edges)
        #: The fraction of every face's area outside the impedance conductors.
        self.face_air = np.clip(1.0 - metal_faces, 0.0, 1.0) * np.ones(grid.n_faces)
        #: The unknowns: the free edges not entirely inside an impedance conductor.
        self.free_edges = grid.free_edges & (self.edge_air > 0.0)
        volume = np.zeros(grid.n_edges, dtype=bool)
        for name, edges in self.conductor_edges.items():
            if name != "background":
                volume |= edges
        if np.any(volume & (self.edge_air < 1.0)):
            raise ValueError("volume conductors must not touch impedance conductors")
        self.feeds = tuple(feeds)
        self._unit_source = self._unit_source + self._feed_source()
        self._factorizations: dict[float, Any] = {}
        self._preconditioners: dict[float, Any] = {}
        self._order: np.ndarray | None = None

    # --- the operator ----------------------------------------------------------------

    @method(
        equations={
            "stokes": r"\mu_0 A^\mathrm{air}_f H_f = b_f + \frac{i Z_s w_f}{\omega} H_f, \qquad \mathbf A_t = \frac{i}{\omega} Z_s\,\mathbf H\times\mathbf n \text{ along the metal}",
            "factor": r"H_f = \frac{b_f}{D_f}, \qquad D_f = \mu_0 A^\mathrm{air}_f - \frac{i Z_s w_f}{\omega} \quad (\mu_0 A_f \text{ without impedance conductors})",
        },
        assumptions=(
            r"$\mathbf H$ tangent to the metal: $(\mathbf H\times\mathbf n)\cdot d\mathbf l = H_f\, dl/\sin\theta$, which integrates to $H_f w_f$",
        ),
        references=("Dey and Mittra (conformal cut faces with a surface impedance)",),
    )
    def face_field_factor(self, frequency: float | None = None) -> np.ndarray:
        """``H_f = factor * b_f`` on every face [A/(m Wb)].

        ``b_f = (C a)_f`` is the flux through the face's part outside the impedance
        conductors, and Stokes' theorem on that part adds the surface's stretch of its
        contour, where ``A_t = (i / omega) Z_s H x n`` (Dey and Mittra's cut faces with
        a surface impedance): ``mu0 A_f^air H_f = b_f + (i Z_s w_f / omega) H_f``, so
        ``factor = 1 / D_f``, ``D_f = mu0 A_f^air - i Z_s w_f / omega`` (``w_f`` the
        surface's width in the face, `wellbalanced.spatial.cutcell`). Without
        impedance conductors ``D_f = mu0 A_f``. Faces entirely inside get zero.
        """
        grid = self.grid
        frequency = self.frequency if frequency is None else float(frequency)
        if not self.impedance_conductors:
            return 1.0 / (MU_0 * grid.face_areas)
        omega = 2.0 * np.pi * frequency
        area = grid.face_areas * np.where(
            self.face_air > 0, np.maximum(self.face_air, self.min_air_fraction), 0.0
        )
        denominator = MU_0 * area.astype(complex)
        for conductor, cells in zip(
            self.impedance_conductors, self._surfaces, strict=True
        ):
            impedance = conductor.surface_impedance(frequency)
            denominator -= 1j * impedance * cells.face_width / omega
        alive = denominator != 0
        return np.where(alive, 1.0 / np.where(alive, denominator, 1.0), 0.0)

    def surface_losses(self, flux: np.ndarray, frequency: float) -> dict[str, float]:
        """The power dissipated in each impedance conductor [W] for face fluxes ``b``.

        ``1/2 Re(Z_s) |H_f|^2 w_f L*_f`` summed over the faces, ``L*`` their dual
        lengths.
        """
        field = self.face_field_factor(frequency) * flux
        losses = {}
        for k, (conductor, cells) in enumerate(
            zip(self.impedance_conductors, self._surfaces, strict=True)
        ):
            resistance = conductor.surface_impedance(frequency).real
            density = resistance * cells.face_width * self.grid.dual_lengths
            losses[conductor.name or f"impedance conductor {k}"] = float(
                0.5 * np.sum(density * np.abs(field) ** 2)
            )
        return losses

    def _gauge(self) -> np.ndarray:
        grid = self.grid
        size = float(np.max(grid.upper - grid.lower))
        return (
            self.regularization / (MU_0 * size**2) * grid.dual_areas / grid.edge_lengths
        )

    def _edge_term(self, omega: float) -> np.ndarray:
        """The diagonal edge term: the conductance's and the gauge's."""
        return 1j * omega * self.edge_conductance + self._gauge()

    def matrix(self, frequency: float | None = None) -> sp.csr_matrix:
        """The full operator on all edges (Dirichlet edges included), symmetric."""
        grid = self.grid
        omega = 2.0 * np.pi * (self.frequency if frequency is None else frequency)
        curl = _curl(grid)
        stiffness = (
            curl.T
            @ sp.diags(grid.dual_lengths * self.face_field_factor(frequency))
            @ curl
        )
        return (stiffness + sp.diags(self._edge_term(omega))).tocsr()

    def _preconditioned(self, frequency: float):
        """The free edges' matrix and its auxiliary-space preconditioner."""
        if frequency not in self._preconditioners:
            free = self.free_edges
            matrix = self.matrix(frequency)[free][:, free]
            omega = 2.0 * np.pi * frequency
            self._preconditioners[frequency] = (
                matrix,
                AuxiliarySpacePreconditioner(
                    self.grid,
                    matrix,
                    free,
                    self.grid.dual_lengths * self.face_field_factor(frequency),
                    self._edge_term(omega),
                    self.grid.edge_lengths * self.edge_air,
                ),
            )
        return self._preconditioners[frequency]

    def _factorized(self, frequency: float):
        """The LU of the free edges' matrix, in nested-dissection order, no pivoting.

        The matrix is complex symmetric with a positive real part on the curl-free
        complement, so SuperLU keeps the diagonal pivots (``diag_pivot_thresh = 0``)
        and the order.
        """
        if frequency not in self._factorizations:
            if self._order is None:
                self._order = self.grid.nested_dissection(self.free_edges)
            free = self.free_edges
            order = self._order
            matrix = self.matrix(frequency)[free][:, free][order][:, order]
            self._factorizations[frequency] = spla.splu(
                sp.csc_matrix(matrix),
                permc_spec="NATURAL",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
        return self._factorizations[frequency]

    def solve(
        self,
        coil_current: complex = 1.0,
        *,
        frequency: float | None = None,
        solver: str = "direct",
        tolerance: float = 1e-8,
        max_iterations: int = 2000,
    ) -> InductiveSolution3D:
        """The time-harmonic field for a coil current amplitude [A].

        Args:
            coil_current: The coil current amplitude [A].
            frequency: The frequency [Hz] (default: the model's).
            solver: ``"direct"`` (a sparse LU, kept per frequency) or ``"iterative"``
                (COCG with the auxiliary-space preconditioner of
                `wellbalanced.spatial.iterative`, kept per frequency), for grids
                beyond about 10^5 unknowns.
            tolerance: The iterative solver's relative residual.
            max_iterations: The iterative solver's limit; it raises if not converged.
        """
        frequency = self.frequency if frequency is None else float(frequency)
        free = self.free_edges
        source = coil_current * self._unit_source
        if solver == "direct":
            lu = self._factorized(frequency)
            assert self._order is not None
            unknowns = np.empty(np.count_nonzero(free), dtype=complex)
            unknowns[self._order] = lu.solve(source[free][self._order].astype(complex))
        elif solver == "iterative":
            matrix, preconditioner = self._preconditioned(frequency)
            result = cocg(
                matrix,
                source[free],
                preconditioner,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            #: The last iterative solve's `IterativeResult`.
            self.last_iterative = result
            if not result.converged:
                raise RuntimeError(
                    f"COCG did not converge: residual {result.residual:.2e} after"
                    f" {result.iterations} iterations"
                )
            unknowns = result.solution
        else:
            raise ValueError(f"unknown solver {solver!r}")
        potential = np.zeros(self.grid.n_edges, dtype=complex)
        potential[free] = unknowns
        flux = _curl(self.grid) @ potential
        return InductiveSolution3D(
            grid=self.grid,
            frequency=frequency,
            coil_current=complex(coil_current),
            potential=potential,
            source=source.astype(complex),
            edge_conductance=self.edge_conductance,
            conductor_edges=self.conductor_edges,
            face_areas=self.grid.face_areas * self.face_air,
            face_field_factor=self.face_field_factor(frequency),
            surface_losses=self.surface_losses(flux, frequency),
        )
