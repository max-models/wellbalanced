"""The time-harmonic magnetoquasistatic field in 3D, on the edges of a Cartesian grid.

The 3D form of :class:`~wellbalanced.spatial.inductive.InductiveField2D` (step R4 of
the RF-driver plan): the vector potential on the edges of a uniform
:class:`~wellbalanced.spatial.edges.EdgeGrid` (finite integration, Yee's staggering),
in the temporal gauge (``E = -i omega A``):

    curl (nu curl A) + i omega sigma A = J_coil,   nu = 1 / mu0,

discretised as ``(C^T M_nu C + i omega M_sigma) a = j``, ``a`` the line integrals of
``A`` on the edges, ``C`` the curl (face–edge incidence), ``M_nu`` the diagonal
``nu L*_f / A_f`` of the faces, ``M_sigma`` the diagonal ``sigma_e A*_e / L_e`` of the
edges (``L*`` and ``A*`` the dual lengths and areas) and ``j`` the coil current through
the dual faces. The displacement current is dropped, as in 2D.

Coils: :class:`Coil3D`, the current on the surface of a region, ``J = curl T`` with
``T = n I e`` inside it (``n`` turns per metre around the direction ``e``). Discretely
``j = C^T (T L*)``, so ``G^T j = 0`` exactly: the coil current is divergence-free on the
grid whatever the shape. A solenoid is a cylinder with ``T`` along its axis; a
rectangular loop is a box with ``T`` normal to the loop; the sheet sits on the region's
staircased surface.

Conductors: :class:`Conductor3D`, a conductivity on the edges inside shapes (the
midpoint rule, staircased; or a sub-cell average, ``samples``), and a background
``conductivity`` (a function of the points, complex for a cold plasma).

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
for 130 000, and the cost grows as the square of the unknowns. Larger grids need an iterative solver
with an auxiliary-space preconditioner (not here).

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
from scipy.constants import mu_0 as MU_0

from wellbalanced.spatial.edges import EdgeGrid

__all__ = ["Coil3D", "Conductor3D", "InductiveField3D", "InductiveSolution3D"]

#: A conductivity [S/m] as a function of ``(3, n)`` points, returning ``n`` values.
PointField = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Coil3D:
    """A coil as the current sheet on the surface of a region: ``J = curl T``.

    ``T = turns_per_length * I * direction`` inside ``shape`` and zero outside, so the
    sheet carries ``turns_per_length * I`` amperes per metre (along ``direction``),
    circulating around ``direction``. A solenoid of ``N`` turns over a length ``l`` on a
    :class:`simgeometry.Cylinder` along ``z``: ``Coil3D(cylinder, (0, 0, 1), N / l)``; a
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
class Conductor3D:
    """A conductor of uniform conductivity: the union of shapes.

    Attributes:
        shapes: The shapes it is made of (Booleans for slits, e.g. ``tube - box``).
        conductivity: ``sigma`` [S/m], real for a metal.
        name: A label for :attr:`InductiveSolution3D.conductor_losses`.
    """

    shapes: Sequence[Any]
    conductivity: float
    name: str = ""


@dataclass(frozen=True)
class InductiveSolution3D:
    """The time-harmonic field of :meth:`InductiveField3D.solve`, as complex amplitudes.

    Attributes:
        grid: The edge grid.
        frequency: The frequency [Hz].
        coil_current: The coil current amplitude ``I`` [A].
        potential: ``a``, the line integrals of ``A`` on the edges [Wb].
        source: ``j``, the coil current through the dual faces of the edges [A].
        edge_conductance: ``sigma_e A*_e / L_e`` per edge [S m].
        conductor_edges: The edges of each conductor, by name.
    """

    grid: EdgeGrid
    frequency: float
    coil_current: complex
    potential: np.ndarray
    source: np.ndarray
    edge_conductance: np.ndarray
    conductor_edges: dict[str, np.ndarray] = field(default_factory=dict)

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
        """``B . n`` on the faces [T]."""
        return self.magnetic_flux / self.grid.face_areas

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
        return self._loss()

    @property
    def conductor_losses(self) -> dict[str, float]:
        """The power dissipated in each conductor [W]."""
        return {name: self._loss(edges) for name, edges in self.conductor_edges.items()}

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


def _curl(grid: EdgeGrid) -> sp.csr_matrix:
    if not hasattr(grid, "_curl_matrix"):
        grid._curl_matrix = grid.curl()  # type: ignore[attr-defined]
    return grid._curl_matrix  # type: ignore[attr-defined]


class InductiveField3D:
    """The 3D time-harmonic magnetoquasistatic field of coils and conductors.

    Args:
        grid: The edge grid (its boundaries are the field's).
        frequency: The frequency [Hz].
        coils: The coils, all carrying the coil current.
        conductors: Conductors of uniform conductivity (must not overlap).
        conductivity: A background conductivity [S/m] as a function of ``(3, n)``
            points, e.g. a cold plasma's complex ``sigma``; added outside the
            conductors.
        samples: ``1``: a conductor's edges are those whose midpoint it contains (a
            staircased region). ``s > 1``: the sub-cell conductance of every edge's dual
            cell from ``s^3`` points (:meth:`EdgeGrid.edge_average`).
        regularization: ``eps`` of the gauge term (relative to ``nu / l^2``).
    """

    def __init__(
        self,
        grid: EdgeGrid,
        *,
        frequency: float,
        coils: Sequence[Coil3D] = (),
        conductors: Sequence[Conductor3D] = (),
        conductivity: PointField | None = None,
        samples: int = 1,
        regularization: float = 1e-6,
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
        self._factorizations: dict[float, Any] = {}
        self._order: np.ndarray | None = None

    # --- sources and materials -----------------------------------------------------

    def _coil_source(self) -> np.ndarray:
        """``j = C^T (T L*)`` for 1 A [A], with ``T`` on the faces inside each coil."""
        grid = self.grid
        centres = grid.face_centres()
        normals = grid.face_axes()
        t = np.zeros(grid.n_faces)
        for coil in self.coils:
            direction = np.asarray(coil.direction, dtype=float)
            if direction.shape != (3,) or not np.any(direction):
                raise ValueError("a coil needs a nonzero 3D direction")
            direction = direction / np.linalg.norm(direction)
            inside = grid.contains(coil.shape, centres)
            if not inside.any():
                raise ValueError(f"coil {coil.name!r} contains no face centre")
            t[inside] += coil.turns_per_length * direction[normals[inside]]
        return np.asarray(_curl(grid).T @ (t * grid.dual_lengths))

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

    # --- the operator ----------------------------------------------------------------

    def matrix(self, frequency: float | None = None) -> sp.csr_matrix:
        """The full operator on all edges (Dirichlet edges included), symmetric."""
        grid = self.grid
        omega = 2.0 * np.pi * (self.frequency if frequency is None else frequency)
        curl = _curl(grid)
        stiffness = (
            curl.T @ sp.diags(grid.dual_lengths / (MU_0 * grid.face_areas)) @ curl
        )
        size = float(np.max(grid.upper - grid.lower))
        gauge = (
            self.regularization / (MU_0 * size**2) * grid.dual_areas / grid.edge_lengths
        )
        return (
            stiffness + sp.diags(1j * omega * self.edge_conductance + gauge)
        ).tocsr()

    def _factorized(self, frequency: float):
        """The LU of the free edges' matrix, in nested-dissection order, no pivoting.

        The matrix is complex symmetric with a positive real part on the curl-free
        complement, so SuperLU keeps the diagonal pivots (``diag_pivot_thresh = 0``)
        and the order.
        """
        if frequency not in self._factorizations:
            if self._order is None:
                self._order = self.grid.nested_dissection(self.grid.free_edges)
            free = self.grid.free_edges
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
        self, coil_current: complex = 1.0, *, frequency: float | None = None
    ) -> InductiveSolution3D:
        """The time-harmonic field for a coil current amplitude [A]."""
        frequency = self.frequency if frequency is None else float(frequency)
        free = self.grid.free_edges
        source = coil_current * self._unit_source
        lu = self._factorized(frequency)
        assert self._order is not None
        unknowns = np.empty(np.count_nonzero(free), dtype=complex)
        unknowns[self._order] = lu.solve(source[free][self._order].astype(complex))
        potential = np.zeros(self.grid.n_edges, dtype=complex)
        potential[free] = unknowns
        return InductiveSolution3D(
            grid=self.grid,
            frequency=frequency,
            coil_current=complex(coil_current),
            potential=potential,
            source=source.astype(complex),
            edge_conductance=self.edge_conductance,
            conductor_edges=self.conductor_edges,
        )
