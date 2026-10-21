"""The inductive RF field of a coil and a plasma in a 2D Cartesian cut.

The magnetoquasistatic field of an RF coil around a driver, as in the 2D models of
Zielke et al. (Plasma Sources Sci. Technol. 31, 035019, 2022, in cylindrical
symmetry; NIBS'22, arXiv:2211.10143, in the Cartesian cut used here): at 1-2 MHz the
vacuum wavelength (150-300 m) is far larger than the source, so the displacement
current is dropped. With all currents along ``z`` (a horizontal cut through one or
several drivers, the coil windings left and right of each) the field is one scalar,
the vector potential ``A_z(x, y)``:

    -div grad A_z = mu0 (J_coil + J_plasma),
    B_x = d_y A_z,   B_y = -d_x A_z,   E_z = -d_t A_z.

The electrostatic part of ``E_z`` (a uniform ``-d_z phi`` along the cut) is zero.

Plasma: cold electrons with friction, ``m_e d_t u = -e E - m_e nu u``, so
``J = -e n_e u`` obeys ``d_t J = eps E - nu J`` with ``eps = n_e e^2 / m_e``. In the
time-harmonic form (amplitudes ``X(r) exp(i omega t)``) this is Ohm's law with the
conductivity ``sigma = eps / (nu + i omega)``; the RF Lorentz force and the electron
viscosity of Zielke's model are not in it (they belong to the self-consistent model,
step R3 of the RF-driver plan). The electron density and the collision frequency are
prescribed.

Coil: every winding is a region (a disk of the tube diameter, say) carrying the coil
current times its sign, spread uniformly over its cells. Outside the tube this is the
field of Zielke's surface current ``I / (pi d_coil)`` on the tube circumference.

Conductors (driver and source backplates, expansion walls, electromagnetic shields):
perfect, ``E_z = 0`` inside. ``grounded`` conductors (the default, Zielke's condition)
have ``A_z = 0``, the value of the far boundary: they and the boundary are joined at
the ends of the cut, so each carries whatever net current the field induces.
``floating`` conductors have one uniform ``A_z``, an unknown, and zero net current:
a closed loop of plates in the cut, such as the two halves of a cylindrical shield
(the zero flux through the loop is the equal ``A_z`` of its plates). The current of a
conductor flows in its surface cells; :attr:`InductiveSolution.conductor_currents`
are their net currents.

Discretisation: finite volumes on the cells of a tensor-product grid (any cell edges,
so the skin layer can be resolved and the cells stretched toward the far boundary).
Each axis is bounded by ``A_z = 0`` (``"dirichlet"``, the default: a distant
conducting enclosure or free space far enough away), a symmetry plane
(``"neumann"``) or periodic. The matrix is complex symmetric; one sparse LU per
frequency (the time-dependent form: per time step size).

Time-dependent form: the trapezoidal rule for ``d_t J = eps E - nu J`` and
``d_t A = -E``, with Ampere's law at the new time. Its periodic steady state under a
coil current ``Re(I exp(i omega t))`` is exactly the time-harmonic solution at the
modified frequency ``omega' = (2 / dt) tan(omega dt / 2)``, which the tests check.

Power: the absorbed power per unit length ``P' = sum 1/2 Re(J E*) dA``, equal, to
round-off, to the active power the windings deliver (``-1/2 Re sum E J_coil* dA``; the
conductors are lossless). A 2D cut gives powers per unit length along ``z``; the
power of a driver is ``depth * P'`` with an effective ``depth``: for a cylindrical
driver of radius ``R`` heated in a skin layer, about the half circumference ``pi R``
(both halves of the cut together make the loop). The plasma resistance seen by the
coil is ``R_p = 2 depth P' / |I|^2`` and the RF power transfer efficiency
``eta = P / (P + R_net |I|^2 / 2) = R_p / (R_p + R_net)``, independent of the power for
a prescribed plasma.

Verification (``tests/unit/test_inductive.py``): a line current and a pair of opposite
currents against ``A_z = -mu0 I / (2 pi) ln r`` and its field; the field decay into a
conducting half-space against ``exp(-sqrt(i omega mu0 sigma) x)``; the power balance
(absorbed equal to delivered) to round-off; grounded and floating conductors (zero
field inside, the net current of a floating group zero); the time-dependent run
against the time-harmonic solution at the modified frequency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import xarray as xr
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy.constants import mu_0 as MU_0
from simgeometry.operations import mask

from wellbalanced.datasets import variable

__all__ = [
    "Conductor",
    "InductiveField2D",
    "InductiveSolution",
    "InductiveState",
    "InductiveTimeStepper",
    "Winding",
    "stretched_edges",
]

AxisBoundary = Literal["dirichlet", "neumann", "periodic"]
#: A spatial field given as a number, an array on the cells, or a function of the
#: cell-centre coordinate arrays ``(X, Y)``.
CellField = float | np.ndarray | Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Winding:
    """One coil winding: a region carrying ``sign`` times the coil current along ``z``.

    Attributes:
        shape: A 2D simgeometry shape (or any object with ``contains_points``), e.g. a
            :class:`simgeometry.Circle` of the tube's radius.
        sign: The direction of the current: ``+1`` or ``-1`` (the two sides of a
            driver carry opposite signs), or a turn count.
        name: A label for the diagnostics.
    """

    shape: Any
    sign: float = 1.0
    name: str = ""


@dataclass(frozen=True)
class Conductor:
    """A perfect conductor (``E_z = 0`` inside): one or several connected shapes.

    Attributes:
        shapes: The shapes it is made of (simgeometry shapes or any objects with
            ``contains_points``).
        floating: ``False`` (grounded, the default): ``A_z = 0`` inside, the value of the
            far boundary, any net current. ``True``: one uniform unknown ``A_z`` and zero
            net current, a closed loop of plates in the cut.
        name: A label for :attr:`InductiveSolution.conductor_currents`.
    """

    shapes: Sequence[Any]
    floating: bool = False
    name: str = ""


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
    toward ``lower`` and ``upper``. The outermost cells are shortened to end at ``lower`` and ``upper``.
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


def _cell_field(value: CellField, xx: np.ndarray, yy: np.ndarray, dtype=float):
    """Evaluate a :data:`CellField` on the cell centres."""
    if callable(value):
        result = np.asarray(value(xx, yy), dtype=dtype)
    else:
        result = np.asarray(value, dtype=dtype)
    return np.broadcast_to(result, xx.shape).copy()


@dataclass(frozen=True)
class InductiveSolution:
    """The time-harmonic field of :meth:`InductiveField2D.solve`, as complex amplitudes.

    Every quantity is per unit length along ``z`` where it is extensive. The instant
    value of an amplitude ``X`` is ``Re(X exp(i omega t))``.
    """

    x: np.ndarray
    y: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    frequency: float
    coil_current: complex
    vector_potential: np.ndarray
    conductivity: np.ndarray
    source_current_density: np.ndarray
    conductor_currents: dict[str, complex] = field(default_factory=dict)

    @property
    def omega(self) -> float:
        """The angular frequency [1/s]."""
        return 2.0 * np.pi * self.frequency

    @property
    def cell_areas(self) -> np.ndarray:
        """The cell areas [m^2], shape ``(nx, ny)``."""
        return np.outer(np.diff(self.x_edges), np.diff(self.y_edges))

    @property
    def electric_field(self) -> np.ndarray:
        """``E_z = -i omega A_z`` [V/m]."""
        return -1j * self.omega * self.vector_potential

    @property
    def magnetic_field(self) -> tuple[np.ndarray, np.ndarray]:
        """``(B_x, B_y) = (d_y A_z, -d_x A_z)`` [T] on the cell centres."""
        a = self.vector_potential
        b_x = np.gradient(a, self.y, axis=1) if a.shape[1] > 1 else np.zeros_like(a)
        b_y = -np.gradient(a, self.x, axis=0) if a.shape[0] > 1 else np.zeros_like(a)
        return b_x, b_y

    @property
    def plasma_current_density(self) -> np.ndarray:
        """``J_z = sigma E_z`` in the plasma [A/m^2]."""
        return self.conductivity * self.electric_field

    @property
    def power_density(self) -> np.ndarray:
        """The time-averaged absorbed power density ``1/2 Re(J E*)`` [W/m^3]."""
        e = self.electric_field
        return 0.5 * np.real(self.conductivity) * np.abs(e) ** 2

    @property
    def absorbed_power(self) -> float:
        """The power absorbed by the plasma per unit length [W/m]."""
        return float(np.sum(self.power_density * self.cell_areas))

    @property
    def source_power(self) -> complex:
        """The complex power the windings deliver per unit length [W/m].

        ``-1/2 sum E J_coil* dA``; its real part equals :attr:`absorbed_power`, its
        imaginary part is the reactive power, positive for the inductive load.
        """
        e = self.electric_field
        return complex(
            -0.5 * np.sum(e * np.conj(self.source_current_density) * self.cell_areas)
        )

    def plasma_resistance(self, depth: float) -> float:
        """The plasma resistance seen by the coil, ``2 depth P' / |I|^2`` [Ohm]."""
        return 2.0 * depth * self.absorbed_power / abs(self.coil_current) ** 2

    def efficiency(self, network_resistance: float, depth: float) -> float:
        """The RF power transfer efficiency ``R_p / (R_p + R_net)``."""
        r_p = self.plasma_resistance(depth)
        return r_p / (r_p + network_resistance)

    def scaled(self, coil_current: complex) -> InductiveSolution:
        """The same solution for another coil current (the problem is linear)."""
        factor = coil_current / self.coil_current
        return InductiveSolution(
            x=self.x,
            y=self.y,
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            frequency=self.frequency,
            coil_current=complex(coil_current),
            vector_potential=self.vector_potential * factor,
            conductivity=self.conductivity,
            source_current_density=self.source_current_density * factor,
            conductor_currents={
                name: value * factor for name, value in self.conductor_currents.items()
            },
        )

    def for_power(self, power: float, depth: float) -> InductiveSolution:
        """The solution whose plasma absorbs ``power`` [W] over ``depth`` [m]."""
        per_length = self.absorbed_power / abs(self.coil_current) ** 2
        if per_length <= 0.0:
            raise ValueError("no power is absorbed: no plasma or no conductivity")
        return self.scaled(np.sqrt(power / (depth * per_length)))

    def to_dataset(self) -> xr.Dataset:
        """The amplitudes and the power density as a labelled dataset."""
        b_x, b_y = self.magnetic_field
        dims = ("x", "y")
        return xr.Dataset(
            {
                "vector_potential": variable(
                    dims, self.vector_potential, "vector potential A_z", "T m"
                ),
                "electric_field": variable(
                    dims, self.electric_field, "electric field E_z", "V/m"
                ),
                "magnetic_field_x": variable(dims, b_x, "magnetic field B_x", "T"),
                "magnetic_field_y": variable(dims, b_y, "magnetic field B_y", "T"),
                "plasma_current_density": variable(
                    dims, self.plasma_current_density, "plasma current density", "A/m^2"
                ),
                "power_density": variable(
                    dims, self.power_density, "absorbed power density", "W/m^3"
                ),
            },
            coords={
                "x": ("x", self.x, {"long_name": "x", "units": "m"}),
                "y": ("y", self.y, {"long_name": "y", "units": "m"}),
            },
            attrs={
                "frequency": self.frequency,
                "coil_current_amplitude": abs(self.coil_current),
                "absorbed_power_per_length": self.absorbed_power,
            },
        )


@dataclass
class InductiveState:
    """The state of the time-dependent form: real instant values on the cells."""

    time: float
    vector_potential: np.ndarray
    plasma_current_density: np.ndarray
    electric_field: np.ndarray
    coil_current: float


class InductiveField2D:
    """The magnetoquasistatic ``A_z`` of coil windings, conductors and a plasma.

    Args:
        x_edges: Cell edges along ``x`` [m], increasing.
        y_edges: Cell edges along ``y`` [m], increasing.
        frequency: The driving frequency [Hz].
        windings: The coil windings; each carries ``sign`` times the coil current.
        conductors: The perfect conductors.
        electron_density: ``n_e`` [m^-3] on the cells (zero outside the plasma; set
            to zero in the windings and conductors).
        collision_frequency: The electron momentum-transfer frequency ``nu`` [1/s].
        boundaries: The boundary of the ``x`` and of the ``y`` axis.
    """

    def __init__(
        self,
        x_edges: Sequence[float] | np.ndarray,
        y_edges: Sequence[float] | np.ndarray,
        *,
        frequency: float,
        windings: Sequence[Winding] = (),
        conductors: Sequence[Conductor] = (),
        electron_density: CellField = 0.0,
        collision_frequency: CellField = 0.0,
        boundaries: tuple[AxisBoundary, AxisBoundary] = ("dirichlet", "dirichlet"),
    ) -> None:
        """Mark the regions on the grid and assemble the field operator."""
        self.x_edges = np.asarray(x_edges, dtype=float)
        self.y_edges = np.asarray(y_edges, dtype=float)
        for edges in (self.x_edges, self.y_edges):
            if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError("cell edges must be increasing, at least two")
        for boundary in boundaries:
            if boundary not in ("dirichlet", "neumann", "periodic"):
                raise ValueError(f"unknown boundary {boundary!r}")
        if frequency <= 0.0:
            raise ValueError("frequency must be positive")
        self.frequency = float(frequency)
        self.boundaries = tuple(boundaries)
        self.x = 0.5 * (self.x_edges[1:] + self.x_edges[:-1])
        self.y = 0.5 * (self.y_edges[1:] + self.y_edges[:-1])
        self.shape = (self.x.size, self.y.size)
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")
        self.cell_areas = np.outer(np.diff(self.x_edges), np.diff(self.y_edges))
        self.windings = tuple(windings)
        self.conductors = tuple(conductors)

        self.electron_density = _cell_field(electron_density, self.xx, self.yy)
        self.collision_frequency = _cell_field(collision_frequency, self.xx, self.yy)
        if np.any(self.electron_density < 0.0) or np.any(
            self.collision_frequency < 0.0
        ):
            raise ValueError("electron density and collision frequency must be >= 0")

        points = np.stack([self.xx.ravel(), self.yy.ravel()])
        self.winding_masks = [self._mask(w.shape, points) for w in self.windings]
        self.conductor_masks = [
            np.logical_or.reduce([self._mask(s, points) for s in c.shapes])
            for c in self.conductors
        ]
        self._check_regions()
        self._source_unit = self._unit_source()
        self._laplacian = self._assemble_laplacian()
        self._projection, self._floating = self._assemble_projection()
        self._factorizations: dict[complex, Any] = {}

    # --- regions -----------------------------------------------------------------

    def _mask(self, shape: Any, points: np.ndarray) -> np.ndarray:
        return mask(shape, points).reshape(self.shape)

    def _check_regions(self) -> None:
        for k, region in enumerate(self.winding_masks):
            if not region.any():
                raise ValueError(
                    f"winding {self.windings[k].name or k} covers no cell centre: "
                    "refine the grid there"
                )
        regions = [*self.winding_masks, *self.conductor_masks]
        covered = np.sum(regions, axis=0) if regions else np.zeros(self.shape)
        if np.any(covered > 1):
            raise ValueError("windings and conductors must not overlap")
        # the plasma is the electron density outside the windings and conductors
        self.electron_density[covered > 0] = 0.0

    def _unit_source(self) -> np.ndarray:
        """``J_coil`` [A/m^2] for a coil current of 1 A."""
        source = np.zeros(self.shape)
        for winding, region in zip(self.windings, self.winding_masks, strict=True):
            source[region] += winding.sign / np.sum(self.cell_areas[region])
        return source

    # --- operators ---------------------------------------------------------------

    def _axis_couplings(self, axis: int):
        """Neighbour pairs and face coefficients ``face length / centre distance``."""
        centres = (self.x, self.y)[axis]
        edges = (self.x_edges, self.y_edges)[axis]
        widths = np.diff((self.y_edges, self.x_edges)[axis])  # face lengths
        n = centres.size
        index = np.arange(self.shape[0] * self.shape[1]).reshape(self.shape)
        index = np.moveaxis(index, axis, 0)
        pairs, coefficients = [], []
        distances = np.diff(centres)
        for i in range(n - 1):
            pairs.append((index[i], index[i + 1]))
            coefficients.append(widths / distances[i])
        boundary = self.boundaries[axis]
        boundary_terms = np.zeros(self.shape)
        if boundary == "periodic" and n > 1:
            wrap = 0.5 * (edges[1] - edges[0] + edges[-1] - edges[-2])
            pairs.append((index[n - 1], index[0]))
            coefficients.append(widths / wrap)
        elif boundary == "dirichlet":
            terms = np.moveaxis(boundary_terms, axis, 0)
            terms[0] += widths / (centres[0] - edges[0])
            terms[-1] += widths / (edges[-1] - centres[-1])
        return pairs, coefficients, boundary_terms

    def _assemble_laplacian(self) -> sp.csr_matrix:
        """The finite-volume ``-div grad`` integrated over the cells (symmetric)."""
        size = self.shape[0] * self.shape[1]
        rows, cols, values = [], [], []
        diagonal = np.zeros(size)
        for axis in (0, 1):
            pairs, coefficients, boundary_terms = self._axis_couplings(axis)
            diagonal += boundary_terms.ravel()
            for (left, right), coefficient in zip(pairs, coefficients, strict=True):
                left, right, c = left.ravel(), right.ravel(), coefficient.ravel()
                rows += [left, right]
                cols += [right, left]
                values += [-c, -c]
                np.add.at(diagonal, left, c)
                np.add.at(diagonal, right, c)
        matrix = (
            sp.coo_matrix(
                (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                shape=(size, size),
            )
            if rows
            else sp.coo_matrix((size, size))
        )
        return (matrix + sp.diags(diagonal)).tocsr()

    def _assemble_projection(self) -> tuple[sp.csr_matrix, list[int]]:
        """Cells -> unknowns: a column per free cell and per floating conductor.

        Grounded conductor cells have no column (``A_z = 0``).
        """
        size = self.shape[0] * self.shape[1]
        column = np.full(size, -1)
        free = np.ones(size, dtype=bool)
        for region in self.conductor_masks:
            free &= ~region.ravel()
        column[free] = np.arange(np.count_nonzero(free))
        count = np.count_nonzero(free)
        floating = []
        for conductor, region in zip(
            self.conductors, self.conductor_masks, strict=True
        ):
            if conductor.floating:
                column[region.ravel()] = count
                floating.append(count)
                count += 1
        cells = np.flatnonzero(column >= 0)
        projection = sp.csr_matrix(
            (np.ones(cells.size), (cells, column[cells])), shape=(size, count)
        )
        return projection, floating

    # --- time-harmonic -----------------------------------------------------------

    def conductivity(self, omega: float) -> np.ndarray:
        """``sigma = n_e e^2 / (m_e (nu + i omega))`` [S/m] on the cells."""
        eps = self.electron_density * ELEMENTARY_CHARGE**2 / ELECTRON_MASS
        return eps / (self.collision_frequency + 1j * omega)

    def _solve_reduced(self, key: Any, matrix_of: Callable[[], Any], rhs):
        if key not in self._factorizations:
            self._factorizations[key] = spla.splu(sp.csc_matrix(matrix_of()))
        return self._factorizations[key].solve(rhs)

    def solve(
        self, coil_current: complex = 1.0, *, frequency: float | None = None
    ) -> InductiveSolution:
        """The time-harmonic field for a coil current amplitude [A].

        ``frequency`` overrides the model's (the time-dependent form's modified
        frequency is the use of it).
        """
        frequency = self.frequency if frequency is None else float(frequency)
        omega = 2.0 * np.pi * frequency
        sigma = self.conductivity(omega)
        diagonal = 1j * omega * MU_0 * (sigma * self.cell_areas).ravel()
        source = coil_current * self._source_unit
        rhs = MU_0 * (source * self.cell_areas).ravel()
        projection = self._projection

        def matrix():
            full = self._laplacian + sp.diags(diagonal)
            return projection.T @ full @ projection

        reduced = self._solve_reduced(
            ("harmonic", frequency), matrix, projection.T @ rhs.astype(complex)
        )
        potential = (projection @ reduced).reshape(self.shape)
        return InductiveSolution(
            x=self.x,
            y=self.y,
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            frequency=frequency,
            coil_current=complex(coil_current),
            vector_potential=potential,
            conductivity=sigma,
            source_current_density=source.astype(complex),
            conductor_currents=self._conductor_currents(potential),
        )

    def _conductor_currents(self, potential: np.ndarray) -> dict[str, Any]:
        """Net currents of the conductors from Ampere's law, ``sum (L A) / mu0``."""
        residual = (self._laplacian @ potential.ravel()).reshape(self.shape) / MU_0
        return {
            conductor.name or f"conductor {k}": np.sum(residual[region])
            for k, (conductor, region) in enumerate(
                zip(self.conductors, self.conductor_masks, strict=True)
            )
        }

    # --- time-dependent ------------------------------------------------------------

    def time_stepper(self, dt: float) -> InductiveTimeStepper:
        """The time-dependent form with steps of ``dt`` [s]."""
        return InductiveTimeStepper(self, dt)

    @staticmethod
    def modified_frequency(frequency: float, dt: float) -> float:
        """The frequency whose harmonic solution the trapezoidal steady state equals.

        ``omega' = (2 / dt) tan(omega dt / 2)``, as a frequency [Hz].
        """
        omega = 2.0 * np.pi * frequency
        return (2.0 / dt) * np.tan(0.5 * omega * dt) / (2.0 * np.pi)


class InductiveTimeStepper:
    """Trapezoidal steps of the plasma current with Ampere's law at the new time.

    ``J^{n+1} (1 + nu dt/2) = J^n (1 - nu dt/2) - eps (A^{n+1} - A^n)`` and
    ``(L + mu0 dA eps / (1 + nu dt/2)) A^{n+1} = mu0 dA (J_coil^{n+1} + ...)``: one
    real sparse LU for all steps.
    """

    def __init__(self, field_model: InductiveField2D, dt: float) -> None:
        """Factorize the step's operator."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.model = field_model
        self.dt = float(dt)
        model = field_model
        self._eps = (
            model.electron_density * ELEMENTARY_CHARGE**2 / ELECTRON_MASS
        ).ravel()
        nu = model.collision_frequency.ravel()
        self._damp = 1.0 + 0.5 * nu * self.dt
        self._keep = 1.0 - 0.5 * nu * self.dt
        self._areas = model.cell_areas.ravel()
        diagonal = MU_0 * self._areas * self._eps / self._damp
        full = model._laplacian + sp.diags(diagonal)
        projection = model._projection
        self._projection = projection
        self._lu = spla.splu(sp.csc_matrix(projection.T @ full @ projection))

    def initial_state(self) -> InductiveState:
        """Everything at rest at ``t = 0``."""
        size = self.model.shape[0] * self.model.shape[1]
        zero = np.zeros(size)
        return InductiveState(0.0, zero, zero.copy(), zero.copy(), 0.0)

    def step(self, state: InductiveState, coil_current: float) -> InductiveState:
        """Advance by ``dt``; ``coil_current`` [A] is the value at the new time."""
        source = coil_current * self.model._source_unit.ravel()
        memory = (
            state.plasma_current_density * self._keep
            + self._eps * state.vector_potential
        ) / self._damp
        rhs = MU_0 * self._areas * (source + memory)
        potential = self._projection @ self._lu.solve(self._projection.T @ rhs)
        change = potential - state.vector_potential
        current = (
            state.plasma_current_density * self._keep - self._eps * change
        ) / self._damp
        electric = -2.0 * change / self.dt - state.electric_field
        return InductiveState(
            state.time + self.dt, potential, current, electric, float(coil_current)
        )

    def absorbed_power(self, state: InductiveState) -> float:
        """The instant power ``sum J E dA`` per unit length [W/m]."""
        return float(
            np.sum(state.plasma_current_density * state.electric_field * self._areas)
        )
