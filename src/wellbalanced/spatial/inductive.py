"""The inductive RF field of a coil and a plasma, in a 2D cut or in r–z.

The magnetoquasistatic field of an RF coil around a driver, as in the models of
Zielke et al. (Plasma Sources Sci. Technol. 31, 035019, 2022, in r–z; NIBS'22,
arXiv:2211.10143, in a Cartesian cut): at 1-2 MHz the vacuum wavelength (150-300 m) is
far larger than the source, so the displacement current is dropped (as in that paper's
eqs. 3-4). All currents are normal to the plane, so the field is one component of the
vector potential, ``a = A_z(x, y)`` in a cut or ``a = A_phi(r, z)`` in r–z:

    curl curl a = mu0 (J_coil + J_plasma),   E = -d_t a,

with ``B = (d_y A_z, -d_x A_z)`` in the cut and ``B = (-d_z A_phi, (1/r) d_r (r A_phi))``
in r–z. The grid, its metric and the operator are the grid's: a tensor
`wellbalanced.spatial.grid.CellGrid` (from cell edges), or the refined patches of a
`wellbalanced.spatial.patches.PatchGrid` (``grid=``), where fields are arrays over
the leaf cells.

Plasma, in one of two forms:

- cold electrons with friction (``electron_temperature`` and ``electron_viscosity``
  not given): ``m_e d_t u = -e E - m_e nu u``, so in the time-harmonic form (amplitudes
  ``X exp(i omega t)``) Ohm's law with ``sigma = n_e e^2 / (m_e (nu + i omega))``;
- the electron momentum balance of Zielke et al. 2022 (eqs. 8-10) with the local
  collisional viscosity (their eq. 7), ``mu_e = (2/3) n_e e T_e / nu``:

      i omega n_e m_e u + n_e m_e nu u + curl (mu_e curl u) = -n_e e E + n_e F_L,

  for the velocity ``u`` normal to the plane (``u_phi``; ``curl curl u = -(laplacian -
  1/r^2) u_phi`` in r–z, ``-laplacian u_z`` in the cut), ``J = -e n_e u``, solved with the
  field as one sparse system. The viscosity is evaluated per face (the mean of its two
  cells), ``curl (mu curl u)``, which is the paper's ``-mu laplacian u`` for a uniform
  ``mu_e``. The electrons do not slip at the edge of the plasma (``u = 0`` on its faces),
  an assumption until the boundary conditions of the model (Zielke's thesis) are in.
  The RF Lorentz force in this equation, ``F_L = -e u_in-plane x B`` of the in-plane
  electron drift, is zero here (the drift is far slower than ``u``); the in-plane force
  the RF motion feels, ``-(e/2) Re(u x B*)``, is `InductiveSolution.lorentz_force`,
  which `wellbalanced.spatial.fluid.Fluid2D` takes.

The electron density, temperature and collision frequency are prescribed; a
`wellbalanced.spatial.regions.PlasmaRegion` sets the density to zero outside the
plasma.

Coil: every winding is a region (the tube's cross-section, say) carrying the coil
current times its sign through the plane, spread uniformly over its cells. Outside the
tube this is the field of Zielke's surface current ``I / (pi d_coil)`` on the tube.

Conductors (backplates, expansion walls, electromagnetic shields): perfect, ``E = 0``
inside. ``grounded`` conductors (the default, Zielke's condition) have ``a = 0``, the
value of the far boundary, and carry whatever net current the field induces. In the cut,
``floating`` conductors have one uniform ``A_z``, an unknown, and zero net current (a
closed loop of plates; ``E_z`` may differ from ``-i omega A_z`` by a uniform ``-d_z phi``).
In r–z there is no such gauge freedom (``E_phi = -i omega A_phi``, no ``d_phi``), so every
conductor is grounded.

Power: the absorbed power ``P = sum 1/2 Re(J E*) dV``, equal, to round-off, to the
active power the windings deliver, ``-1/2 Re sum E J_coil* dV`` (the conductors are
lossless); with viscous electrons it splits into friction ``1/2 sum n_e m_e nu |u|^2 dV``
and viscous dissipation. In r–z the volumes are rings and the powers totals; in the cut
they are per unit length along ``z``, and the power of a driver is ``depth * P'`` with
an effective ``depth``. The plasma resistance seen by the coil is ``R_p = 2 P / |I|^2``
and the RF power transfer efficiency ``eta = R_p / (R_p + R_net)``, independent of the
power for a prescribed plasma.

Verification (``tests/unit/test_inductive.py``, ``tests/unit/test_inductive_rz.py``):
line currents and a conducting half-space in the cut, the field of a current loop
(complete elliptic integrals), a long solenoid and a conducting column in it (Bessel
functions) in r–z, the viscous skin layer of a slab against its dispersion relation, the
power balance to round-off, conductors, the time-dependent form.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import xarray as xr
from methodcard import ConvergenceProblem, Rule, method
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy.constants import mu_0 as MU_0

from wellbalanced.datasets import variable
from wellbalanced.spatial.grid import (
    AxisBoundary,
    CellField,
    CellGrid,
    Geometry,
    stretched_edges,
)
from wellbalanced.spatial.regions import PlasmaRegion

if TYPE_CHECKING:
    from wellbalanced.spatial.patches import PatchGrid

__all__ = [
    "Conductor",
    "InductiveField2D",
    "InductiveSolution",
    "InductiveState",
    "InductiveTimeStepper",
    "Winding",
    "stretched_edges",
]


@dataclass(frozen=True)
class Winding:
    """One coil winding: a region carrying ``sign`` times the coil current.

    Attributes:
        shape: A 2D shape with ``contains_points`` (e.g. a `simgeometry.Circle`
            of the tube's radius).
        sign: The direction of the current: ``+1`` or ``-1`` (the two sides of a
            driver in a cut carry opposite signs), or a turn count.
        name: A label for the diagnostics.
    """

    shape: Any
    sign: float = 1.0
    name: str = ""


@dataclass(frozen=True)
class Conductor:
    """A perfect conductor (``E = 0`` inside): one or several connected shapes.

    Attributes:
        shapes: The shapes it is made of (anything with ``contains_points``).
        floating: ``False`` (grounded, the default): ``a = 0`` inside, the value of the
            far boundary, any net current. ``True`` (in a cut only): one uniform unknown
            ``A_z`` and zero net current, a closed loop of plates.
        name: A label for `InductiveSolution.conductor_currents`.
    """

    shapes: Sequence[Any]
    floating: bool = False
    name: str = ""


@method(
    equations={
        "fields": r"E = -i\omega a, \qquad J = \sigma E \ (\text{cold}), \quad J = -e n_e u \ (\text{viscous})",
        "power": r"P = \sum_\mathrm{cells} \tfrac12\mathrm{Re}(J E^*)\, dV = \mathrm{Re}\left(-\tfrac12\sum E J_\mathrm{coil}^*\, dV\right)",
        "resistance": r"R_p = \frac{2P}{|I|^2}, \qquad \eta = \frac{R_p}{R_p + R_\mathrm{net}}, \qquad |I| = \sqrt{\frac{P\,|I_1|^2}{P_1}} \ (\texttt{for\_power})",
        "lorentz": r"\mathbf F = -\frac{e}{2}\mathrm{Re}\left(u\,\hat{\mathbf n}\times\mathbf B^*\right) = \begin{cases}\left(\tfrac e2\mathrm{Re}(u B_y^*),\ -\tfrac e2\mathrm{Re}(u B_x^*)\right) & \text{cut} \\ \left(-\tfrac e2\mathrm{Re}(u B_z^*),\ \tfrac e2\mathrm{Re}(u B_r^*)\right) & r\text{–}z\end{cases}",
        "friction": r"P_\nu = \tfrac12\sum n_e m_e \nu |u|^2\, dV, \qquad P_\mu = \tfrac12\mathrm{Re}(u^H K_\mu u)",
    },
    assumptions=(
        r"amplitudes of $e^{i\omega t}$; volumes and powers are totals in r–z and per metre in a cut (a device's power is `depth` times it)",
        r"the conductors are lossless, so the absorbed power equals the active power of the windings",
    ),
)
@dataclass(frozen=True)
class InductiveSolution:
    """The time-harmonic field of `InductiveField2D.solve`, as complex amplitudes.

    The instant value of an amplitude ``X`` is ``Re(X exp(i omega t))``. Extensive
    quantities are totals in r–z and per unit length along ``z`` in a cut.
    """

    grid: CellGrid | PatchGrid
    frequency: float
    coil_current: complex
    vector_potential: np.ndarray
    plasma_current_density: np.ndarray
    source_current_density: np.ndarray
    electron_density: np.ndarray
    collision_frequency: np.ndarray
    electron_velocity: np.ndarray | None = None
    viscous_power: float = 0.0
    conductor_currents: dict[str, complex] = field(default_factory=dict)

    @property
    def omega(self) -> float:
        """The angular frequency [1/s]."""
        return 2.0 * np.pi * self.frequency

    @property
    def x(self) -> np.ndarray:
        """Cell centres of the first axis (``x`` or ``r``) [m]; per cell on patches."""
        return self.grid.x

    @property
    def y(self) -> np.ndarray:
        """Cell centres of the second axis (``y`` or ``z``) [m]; per cell on patches."""
        return self.grid.y

    @property
    def cell_areas(self) -> np.ndarray:
        """The cross-section areas of the cells [m^2]."""
        return self.grid.cell_areas

    @property
    def cell_volumes(self) -> np.ndarray:
        """The cell volumes [m^3] (per metre along ``z`` in a cut)."""
        return self.grid.cell_volumes

    @property
    def electric_field(self) -> np.ndarray:
        """``E = -i omega a`` [V/m], normal to the plane."""
        return -1j * self.omega * self.vector_potential

    @property
    def magnetic_field(self) -> tuple[np.ndarray, np.ndarray]:
        """The in-plane field [T]: ``(B_x, B_y)`` in a cut, ``(B_r, B_z)`` in r–z."""
        return self.grid.curl(self.vector_potential)

    @property
    def electron_velocity_amplitude(self) -> np.ndarray:
        """The electrons' velocity normal to the plane [m/s], zero outside the plasma.

        The viscous solution's ``u``, or ``-J / (e n_e)`` for the cold plasma.
        """
        if self.electron_velocity is not None:
            return self.electron_velocity
        n = self.electron_density
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                n > 0.0, -self.plasma_current_density / (ELEMENTARY_CHARGE * n), 0.0
            )

    @property
    def lorentz_force(self) -> tuple[np.ndarray, np.ndarray]:
        """The RF-averaged Lorentz force on an electron, in the plane [N].

        ``-(e / 2) Re(u x B*)`` of the velocity ``u`` normal to the plane and the
        in-plane field ``B``: ``(-(e/2) Re(u B_z*), (e/2) Re(u B_r*))`` in r–z,
        ``((e/2) Re(u B_y*), -(e/2) Re(u B_x*))`` in a cut. The force density
        ``n_e`` times it is the time-averaged ``J x B`` on the plasma current, which
        flows against the coil's: it pushes the electrons away from the coil.
        """
        u = self.electron_velocity_amplitude
        b_1, b_2 = self.magnetic_field
        half = 0.5 * ELEMENTARY_CHARGE
        if self.grid.cylindrical:
            return -half * np.real(u * np.conj(b_2)), half * np.real(u * np.conj(b_1))
        return half * np.real(u * np.conj(b_2)), -half * np.real(u * np.conj(b_1))

    @property
    def power_density(self) -> np.ndarray:
        """The time-averaged absorbed power density ``1/2 Re(J E*)`` [W/m^3]."""
        return 0.5 * np.real(self.plasma_current_density * np.conj(self.electric_field))

    @property
    def absorbed_power(self) -> float:
        """The power the plasma absorbs [W] (per unit length in a cut [W/m])."""
        return float(np.sum(self.power_density * self.cell_volumes))

    @property
    def friction_power(self) -> float:
        """The part of `absorbed_power` the friction with the gas dissipates."""
        if self.electron_velocity is None:
            return self.absorbed_power
        friction = (
            0.5
            * self.electron_density
            * ELECTRON_MASS
            * self.collision_frequency
            * np.abs(self.electron_velocity) ** 2
        )
        return float(np.sum(friction * self.cell_volumes))

    @property
    def source_power(self) -> complex:
        """The complex power the windings deliver [W] (per unit length in a cut).

        ``-1/2 sum E J_coil* dV``; its real part equals `absorbed_power`, its
        imaginary part is the reactive power, positive for the inductive load.
        """
        e = self.electric_field
        return complex(
            -0.5 * np.sum(e * np.conj(self.source_current_density) * self.cell_volumes)
        )

    def _total(self, depth: float | None) -> float:
        """The absorbed power of the device [W]: ``depth`` times the cut's, in a cut."""
        if self.grid.cylindrical:
            if depth is not None:
                raise ValueError("an r–z solution has total powers: no depth")
            return self.absorbed_power
        if depth is None:
            raise ValueError("a cut needs the effective depth of the device")
        return depth * self.absorbed_power

    def plasma_resistance(self, depth: float | None = None) -> float:
        """The plasma resistance seen by the coil, ``2 P / |I|^2`` [Ohm].

        ``depth`` [m] turns the power per unit length of a cut into the device's; an
        r–z solution takes none.
        """
        return 2.0 * self._total(depth) / abs(self.coil_current) ** 2

    def efficiency(
        self, network_resistance: float, depth: float | None = None
    ) -> float:
        """The RF power transfer efficiency ``R_p / (R_p + R_net)``."""
        r_p = self.plasma_resistance(depth)
        return r_p / (r_p + network_resistance)

    def scaled(self, coil_current: complex) -> InductiveSolution:
        """The same solution for another coil current (the problem is linear)."""
        factor = coil_current / self.coil_current
        velocity = self.electron_velocity
        return InductiveSolution(
            grid=self.grid,
            frequency=self.frequency,
            coil_current=complex(coil_current),
            vector_potential=self.vector_potential * factor,
            plasma_current_density=self.plasma_current_density * factor,
            source_current_density=self.source_current_density * factor,
            electron_density=self.electron_density,
            collision_frequency=self.collision_frequency,
            electron_velocity=None if velocity is None else velocity * factor,
            viscous_power=self.viscous_power * abs(factor) ** 2,
            conductor_currents={
                name: value * factor for name, value in self.conductor_currents.items()
            },
        )

    def for_power(self, power: float, depth: float | None = None) -> InductiveSolution:
        """The solution whose plasma absorbs ``power`` [W] (over ``depth`` in a cut)."""
        per_current = self._total(depth) / abs(self.coil_current) ** 2
        if per_current <= 0.0:
            raise ValueError("no power is absorbed: no plasma or no conductivity")
        return self.scaled(np.sqrt(power / per_current))

    def to_dataset(self) -> xr.Dataset:
        """The amplitudes and the power density as a labelled dataset."""
        b_1, b_2 = self.magnetic_field
        first, second = self.grid.coordinate_names
        normal = "phi" if self.grid.cylindrical else "z"
        dims = self.grid.dims
        data = {
            "vector_potential": variable(
                dims, self.vector_potential, f"vector potential A_{normal}", "T m"
            ),
            "electric_field": variable(
                dims, self.electric_field, f"electric field E_{normal}", "V/m"
            ),
            f"magnetic_field_{first}": variable(
                dims, b_1, f"magnetic field B_{first}", "T"
            ),
            f"magnetic_field_{second}": variable(
                dims, b_2, f"magnetic field B_{second}", "T"
            ),
            "plasma_current_density": variable(
                dims, self.plasma_current_density, "plasma current density", "A/m^2"
            ),
            "power_density": variable(
                dims, self.power_density, "absorbed power density", "W/m^3"
            ),
        }
        if self.electron_velocity is not None:
            data["electron_velocity"] = variable(
                dims, self.electron_velocity, f"electron velocity u_{normal}", "m/s"
            )
        return xr.Dataset(
            data,
            coords=self.grid.coordinates(),
            attrs={
                "geometry": self.grid.geometry,
                "frequency": self.frequency,
                "coil_current_amplitude": abs(self.coil_current),
                "absorbed_power": self.absorbed_power,
                "viscous_power": self.viscous_power,
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


@method(
    equations={
        "field": r"\nabla\times\nabla\times a = \mu_0 (J_\mathrm{coil} + J_\mathrm{plasma}), \qquad E = -i\omega a, \qquad a = A_z(x, y) \text{ or } A_\phi(r, z)",
        "cold": r"J = \sigma E, \qquad \sigma = \frac{n_e e^2}{m_e(\nu + i\omega)}",
        "viscous": r"i\omega n_e m_e u + n_e m_e \nu u + \nabla\times(\mu_e\nabla\times u) = -n_e e E, \qquad \mu_e = \tfrac23\frac{n_e e T_e}{\nu}, \qquad J = -e n_e u",
        "cold_system": r"P^T\left(K + i\omega\mu_0\,\mathrm{diag}(\sigma V)\right) P\, a = P^T \mu_0 V J_\mathrm{coil}",
        "viscous_system": r"\begin{pmatrix} K & \mu_0 V e n_e \\ -i\omega V n_e e & V n_e m_e(i\omega + \nu) + K_\mu \end{pmatrix}\begin{pmatrix} a \\ u \end{pmatrix} = \begin{pmatrix} \mu_0 V J_\mathrm{coil} \\ 0 \end{pmatrix}",
        "coil": r"J_\mathrm{coil} = \mathrm{sign}\cdot\frac{I}{A_\mathrm{winding}} \text{ on each winding's cells}",
    },
    assumptions=(
        r"magnetoquasistatic: no displacement current (the vacuum wavelength at 1–2 MHz is far larger than the source)",
        r"perfect conductors: grounded ($a = 0$) or, in a cut, floating (one shared unknown, zero net current); $P$ the projection to the unknowns",
        r"viscous electrons do not slip at the plasma's edge ($u = 0$); the RF Lorentz force of the in-plane drift is not in the momentum balance",
        r"$n_e$, $\nu$ and $T_e$ prescribed; zero viscosity reproduces the cold plasma; one complex LU per frequency",
    ),
    references=(
        "[Zielke et al., Plasma Sources Sci. Technol. 31, 035019 (2022)](https://doi.org/10.1088/1361-6595/ac5845), eqs. 3-10",
        "[Zielke et al., NIBS'22, arXiv:2211.10143](https://arxiv.org/abs/2211.10143)",
    ),
    components=(CellGrid.curl_curl, CellGrid.curl, InductiveSolution),
)
class InductiveField2D:
    """The magnetoquasistatic field of coil windings, conductors and a plasma.

    Args:
        x_edges: Cell edges of the first axis (``x``, or ``r`` from ``>= 0``) [m]: a
            tensor grid; or give ``grid``.
        y_edges: Cell edges of the second axis (``y``, or ``z``) [m].
        frequency: The driving frequency [Hz].
        windings: The coil windings; each carries ``sign`` times the coil current.
        conductors: The perfect conductors.
        electron_density: ``n_e`` [m^-3] on the cells (set to zero in the windings, the
            conductors and outside ``plasma_region``).
        collision_frequency: The electron momentum-transfer frequency ``nu`` [1/s].
        electron_temperature: ``T_e`` [eV]: with it, the electrons carry the collisional
            viscosity ``(2/3) n_e e T_e / nu`` (``nu > 0`` in the plasma).
        electron_viscosity: ``mu_e`` [Pa s] directly, instead of from ``T_e``; zero gives
            the momentum balance without viscosity (the cold plasma, solved with ``u``).
        plasma_region: Where the plasma is; everywhere if ``None``.
        boundaries: The boundary of each axis (see `CellGrid`); default
            ``("dirichlet", "dirichlet")``, or the grid's.
        geometry: ``"cartesian"`` (a cut, ``A_z``) or ``"cylindrical"`` (r–z,
            ``A_phi``); default ``"cartesian"``, or the grid's.
        electron_drift: The in-plane electron velocity for the RF Lorentz force in the
            momentum balance; not implemented (the drift of
            `wellbalanced.spatial.fluid.Fluid2D` is far slower than ``u``).
        grid: A grid instead of the edges, e.g. a
            `wellbalanced.spatial.patches.PatchGrid` (its geometry and
            boundaries are the model's).
    """

    # --- regions -----------------------------------------------------------------

    def _check_regions(self) -> None:
        for k, region in enumerate(self.winding_masks):
            if not region.any():
                raise ValueError(
                    f"winding {self.windings[k].name or k} covers no cell centre: "
                    "refine the grid there"
                )
        regions = [*self.winding_masks, *self.conductor_masks]
        covered = np.sum(regions, axis=0) if regions else np.zeros(self.grid.shape)
        if np.any(covered > 1):
            raise ValueError("windings and conductors must not overlap")
        # the plasma is the electron density outside the windings and conductors
        self.electron_density[covered > 0] = 0.0

    def _viscosity(
        self, temperature: CellField | None, viscosity: CellField | None
    ) -> np.ndarray | None:
        """``mu_e`` on the cells, or ``None`` for the cold plasma (Ohm's law)."""
        if viscosity is not None:
            mu = self.grid.cell_field(viscosity)
        elif temperature is not None:
            t_e = self.grid.cell_field(temperature)
            nu = self.collision_frequency
            if np.any(nu[self.plasma] <= 0.0):
                raise ValueError("a viscous plasma needs nu > 0 where n_e > 0")
            with np.errstate(divide="ignore", invalid="ignore"):
                mu = np.where(
                    self.plasma,
                    2.0 / 3.0 * self.electron_density * ELEMENTARY_CHARGE * t_e / nu,
                    0.0,
                )
        else:
            return None
        if np.any(mu < 0.0):
            raise ValueError("the electron viscosity must be >= 0")
        return np.where(self.plasma, mu, 0.0)

    def _unit_source(self) -> np.ndarray:
        """``J_coil`` [A/m^2] for a coil current of 1 A through every winding."""
        source = np.zeros(self.grid.shape)
        for winding, region in zip(self.windings, self.winding_masks, strict=True):
            source[region] += winding.sign / np.sum(self.grid.cell_areas[region])
        return source

    def _assemble_projection(self) -> tuple[sp.csr_matrix, list[int]]:
        """Cells -> unknowns: a column per free cell and per floating conductor.

        Grounded conductor cells have no column (``a = 0``).
        """
        size = self.grid.size
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

    def __init__(
        self,
        x_edges: Sequence[float] | np.ndarray | None = None,
        y_edges: Sequence[float] | np.ndarray | None = None,
        *,
        frequency: float,
        windings: Sequence[Winding] = (),
        conductors: Sequence[Conductor] = (),
        electron_density: CellField = 0.0,
        collision_frequency: CellField = 0.0,
        electron_temperature: CellField | None = None,
        electron_viscosity: CellField | None = None,
        plasma_region: PlasmaRegion | None = None,
        boundaries: tuple[AxisBoundary, AxisBoundary] | None = None,
        geometry: Geometry | None = None,
        electron_drift: Any = None,
        grid: CellGrid | PatchGrid | None = None,
    ) -> None:
        """Mark the regions on the grid and assemble the field operator."""
        if frequency <= 0.0:
            raise ValueError("frequency must be positive")
        if electron_drift is not None:
            raise NotImplementedError(
                "the Lorentz force of the in-plane drift in the momentum balance"
            )
        if grid is None:
            if x_edges is None or y_edges is None:
                raise ValueError("give the cell edges or a grid")
            grid = CellGrid(
                x_edges,
                y_edges,
                geometry=geometry or "cartesian",
                boundaries=boundaries or ("dirichlet", "dirichlet"),
            )
        else:
            if x_edges is not None or y_edges is not None:
                raise ValueError("give either the cell edges or a grid, not both")
            if geometry is not None and geometry != grid.geometry:
                raise ValueError(f"the grid's geometry is {grid.geometry!r}")
            if boundaries is not None and tuple(boundaries) != grid.boundaries:
                raise ValueError(f"the grid's boundaries are {grid.boundaries!r}")
        self.grid: CellGrid | PatchGrid = grid
        self.frequency = float(frequency)
        self.windings = tuple(windings)
        self.conductors = tuple(conductors)
        if grid.cylindrical and any(c.floating for c in self.conductors):
            raise ValueError("in r–z every conductor is grounded (E_phi = -i omega A)")

        self.electron_density = grid.cell_field(electron_density)
        self.collision_frequency = grid.cell_field(collision_frequency)
        if np.any(self.electron_density < 0.0) or np.any(
            self.collision_frequency < 0.0
        ):
            raise ValueError("electron density and collision frequency must be >= 0")
        if plasma_region is not None:
            self.electron_density[~plasma_region.mask(grid)] = 0.0

        self.winding_masks = [grid.mask(w.shape) for w in self.windings]
        self.conductor_masks = [grid.union_mask(c.shapes) for c in self.conductors]
        self._check_regions()
        self.viscosity = self._viscosity(electron_temperature, electron_viscosity)
        self._source_unit = self._unit_source()
        self._curl_curl = grid.curl_curl()
        self._projection, self._floating = self._assemble_projection()
        self._factorizations: dict[Any, Any] = {}

    # --- compatibility with the cell geometry ------------------------------------------

    @property
    def x(self) -> np.ndarray:
        """Cell centres of the first axis [m]."""
        return self.grid.x

    @property
    def y(self) -> np.ndarray:
        """Cell centres of the second axis [m]."""
        return self.grid.y

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of a field: cells along each axis, or ``(n,)`` leaf cells."""
        return self.grid.shape

    @property
    def plasma(self) -> np.ndarray:
        """The plasma cells: a positive electron density."""
        return self.electron_density > 0.0

    # --- time-harmonic -----------------------------------------------------------

    def conductivity(self, omega: float) -> np.ndarray:
        """``sigma = n_e e^2 / (m_e (nu + i omega))`` [S/m] on the cells (cold plasma)."""
        eps = self.electron_density * ELEMENTARY_CHARGE**2 / ELECTRON_MASS
        return eps / (self.collision_frequency + 1j * omega)

    def _factorized(self, key: Any, matrix_of: Callable[[], Any]):
        if key not in self._factorizations:
            self._factorizations[key] = spla.splu(sp.csc_matrix(matrix_of()))
        return self._factorizations[key]

    def _conductor_currents(self, potential: np.ndarray) -> dict[str, Any]:
        """Net currents of the conductors from Ampere's law, ``(K a) / mu0 = V J``."""
        residual = (self._curl_curl @ potential.ravel()).reshape(self.grid.shape) / MU_0
        return {
            conductor.name or f"conductor {k}": self.grid.net_current(residual, region)
            for k, (conductor, region) in enumerate(
                zip(self.conductors, self.conductor_masks, strict=True)
            )
        }

    def _solution(
        self,
        frequency: float,
        coil_current: complex,
        potential: np.ndarray,
        plasma_current: np.ndarray,
        source: np.ndarray,
        *,
        velocity: np.ndarray | None = None,
        viscous_power: float = 0.0,
    ) -> InductiveSolution:
        return InductiveSolution(
            grid=self.grid,
            frequency=frequency,
            coil_current=complex(coil_current),
            vector_potential=potential,
            plasma_current_density=plasma_current,
            source_current_density=source.astype(complex),
            electron_density=self.electron_density,
            collision_frequency=self.collision_frequency,
            electron_velocity=velocity,
            viscous_power=viscous_power,
            conductor_currents=self._conductor_currents(potential),
        )

    def _solve_cold(self, coil_current: complex, frequency: float) -> InductiveSolution:
        omega = 2.0 * np.pi * frequency
        volumes = self.grid.cell_volumes
        sigma = self.conductivity(omega)
        diagonal = 1j * omega * MU_0 * (sigma * volumes).ravel()
        source = coil_current * self._source_unit
        rhs = MU_0 * (source * volumes).ravel()
        projection = self._projection

        def matrix():
            full = self._curl_curl + sp.diags(diagonal)
            return projection.T @ full @ projection

        lu = self._factorized(("cold", frequency), matrix)
        potential = (projection @ lu.solve(projection.T @ rhs.astype(complex))).reshape(
            self.grid.shape
        )
        return self._solution(
            frequency,
            coil_current,
            potential,
            sigma * (-1j * omega * potential),
            source,
        )

    def _solve_viscous(
        self, coil_current: complex, frequency: float
    ) -> InductiveSolution:
        """Field and electron velocity together: ``[[K, B], [C, D]] [a; u] = [s; 0]``.

        Field rows ``K a + mu0 V e n_e u = mu0 V J_coil``; velocity rows (plasma cells)
        ``-i omega V n_e e a + (V n_e m_e (i omega + nu) + K_mu) u = 0``.
        """
        omega = 2.0 * np.pi * frequency
        grid = self.grid
        volumes = grid.cell_volumes.ravel()
        n = self.electron_density.ravel()
        nu = self.collision_frequency.ravel()
        plasma = self.plasma.ravel()
        cells = np.flatnonzero(plasma)
        select = sp.csr_matrix(
            (np.ones(cells.size), (cells, np.arange(cells.size))),
            shape=(grid.size, cells.size),
        )
        projection = self._projection
        source = coil_current * self._source_unit
        e, m = ELEMENTARY_CHARGE, ELECTRON_MASS
        assert self.viscosity is not None
        k_mu = select.T @ grid.curl_curl(self.viscosity, active=self.plasma) @ select

        def matrix():
            k = projection.T @ self._curl_curl @ projection
            b = projection.T @ sp.diags(MU_0 * volumes * e * n) @ select
            c = select.T @ sp.diags(-1j * omega * volumes * n * e) @ projection
            d = select.T @ sp.diags(volumes * n * m * (1j * omega + nu)) @ select + k_mu
            return sp.bmat([[k, b], [c, d]])

        rhs = np.concatenate(
            [
                projection.T @ (MU_0 * (source.ravel() * volumes)).astype(complex),
                np.zeros(cells.size, dtype=complex),
            ]
        )
        unknowns = np.asarray(
            self._factorized(("viscous", frequency), matrix).solve(rhs)
        )
        count = np.asarray(projection.sum(axis=0)).size  # the field's unknowns
        potential = (projection @ unknowns[:count]).reshape(grid.shape)
        u_plasma = unknowns[count:]
        velocity = (select @ u_plasma).reshape(grid.shape)
        viscous = 0.5 * float(np.real(np.conj(u_plasma) @ (k_mu @ u_plasma)))
        return self._solution(
            frequency,
            coil_current,
            potential,
            -e * self.electron_density * velocity,
            source,
            velocity=velocity,
            viscous_power=viscous,
        )

    def solve(
        self, coil_current: complex = 1.0, *, frequency: float | None = None
    ) -> InductiveSolution:
        """The time-harmonic field for a coil current amplitude [A].

        ``frequency`` overrides the model's (the time-dependent form's modified
        frequency is the use of it).
        """
        frequency = self.frequency if frequency is None else float(frequency)
        if self.viscosity is None:
            return self._solve_cold(coil_current, frequency)
        return self._solve_viscous(coil_current, frequency)

    # --- time-dependent ------------------------------------------------------------

    def time_stepper(self, dt: float) -> InductiveTimeStepper:
        """The time-dependent form with steps of ``dt`` [s] (cold plasma only)."""
        if self.viscosity is not None:
            raise NotImplementedError(
                "the time-dependent form has no viscous electrons"
            )
        return InductiveTimeStepper(self, dt)

    @staticmethod
    def modified_frequency(frequency: float, dt: float) -> float:
        """The frequency whose harmonic solution the trapezoidal steady state equals.

        ``omega' = (2 / dt) tan(omega dt / 2)``, as a frequency [Hz].
        """
        omega = 2.0 * np.pi * frequency
        return (2.0 / dt) * np.tan(0.5 * omega * dt) / (2.0 * np.pi)


def _stepper_rules() -> dict:
    """The SymPy rule of the modified frequency of `InductiveTimeStepper` (for the tests)."""
    import sympy

    f, dt = sympy.symbols("frequency dt", positive=True)
    rule = (2 / dt) * sympy.tan(sympy.pi * f * dt) / (2 * sympy.pi)
    return {
        "modified_frequency": Rule(
            rule,
            implementation=InductiveField2D.modified_frequency,
            ranges={"frequency": (0.5e6, 2.0e6), "dt": (1.0e-9, 5.0e-8)},
        )
    }


def _stepper_problem() -> ConvergenceProblem:
    """A plasma slab between two windings, driven by ``sin^2(pi f t)`` for one period.

    The field and the plasma current at the end against a run with 32 times smaller
    steps; the drive starts smoothly from rest.
    """
    from simgeometry import Rectangle

    frequency = 1.0e6
    edges = np.linspace(-0.03, 0.03, 13)
    model = InductiveField2D(
        edges,
        edges,
        frequency=frequency,
        windings=[
            Winding(Rectangle((-0.02, 0.0), (0.006, 0.006)), 1.0),
            Winding(Rectangle((0.02, 0.0), (0.006, 0.006)), -1.0),
        ],
        electron_density=lambda x, y: np.where(np.abs(x) < 0.012, 1.0e16, 0.0),
        collision_frequency=1.0e7,
    )

    def run(steps: int) -> np.ndarray:
        dt = 1.0 / (frequency * steps)
        stepper = model.time_stepper(dt)
        state = stepper.initial_state()
        for n in range(1, steps + 1):
            state = stepper.step(state, np.sin(np.pi * frequency * n * dt) ** 2)
        a, j = state.vector_potential, state.plasma_current_density
        return np.concatenate([a / np.abs(a).max(), j / np.abs(j).max()])

    reference = run(2048)

    def error(h: float) -> float:
        return float(np.max(np.abs(run(round(1.0 / h)) - reference)))

    return ConvergenceProblem(error, steps=[1 / 16, 1 / 32, 1 / 64])


@method(
    equations={
        "current": r"J^{n+1}\left(1 + \frac{\nu\Delta t}{2}\right) = J^n\left(1 - \frac{\nu\Delta t}{2}\right) - \epsilon\,(a^{n+1} - a^n), \qquad \epsilon = \frac{n_e e^2}{m_e}",
        "field": r"\left(K + \frac{\mu_0 V\epsilon}{1 + \nu\Delta t/2}\right) a^{n+1} = \mu_0 V\left(J_\mathrm{coil}^{n+1} + \frac{J^n(1 - \nu\Delta t/2) + \epsilon a^n}{1 + \nu\Delta t/2}\right)",
        "electric": r"E^{n+1} = -\frac{2(a^{n+1} - a^n)}{\Delta t} - E^n",
        "modified_frequency": r"\omega' = \frac{2}{\Delta t}\tan\frac{\omega\Delta t}{2}",
    },
    order=2,
    assumptions=(
        r"the trapezoidal rule for $\partial_t J = \epsilon E - \nu J$, $E = -\partial_t a$, with Ampère's law at the new time; one real LU for all steps",
        r"its periodic steady state is the harmonic solution at $\omega'$; cold plasma only",
        r"the order is that of the field and the current after one period of a smooth drive",
    ),
    symbolic=_stepper_rules,
    problem=_stepper_problem,
)
class InductiveTimeStepper:
    """Trapezoidal steps of the plasma current with Ampere's law at the new time.

    ``J^{n+1} (1 + nu dt/2) = J^n (1 - nu dt/2) - eps (a^{n+1} - a^n)`` and
    ``(K + mu0 dV eps / (1 + nu dt/2)) a^{n+1} = mu0 dV (J_coil^{n+1} + ...)``: one
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
        self._volumes = model.grid.cell_volumes.ravel()
        diagonal = MU_0 * self._volumes * self._eps / self._damp
        full = model._curl_curl + sp.diags(diagonal)
        projection = model._projection
        self._projection = projection
        self._lu = spla.splu(sp.csc_matrix(projection.T @ full @ projection))

    def initial_state(self) -> InductiveState:
        """Everything at rest at ``t = 0``."""
        zero = np.zeros(self.model.grid.size)
        return InductiveState(0.0, zero, zero.copy(), zero.copy(), 0.0)

    def step(self, state: InductiveState, coil_current: float) -> InductiveState:
        """Advance by ``dt``; ``coil_current`` [A] is the value at the new time."""
        source = coil_current * self.model._source_unit.ravel()
        memory = (
            state.plasma_current_density * self._keep
            + self._eps * state.vector_potential
        ) / self._damp
        rhs = MU_0 * self._volumes * (source + memory)
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
        """The instant power ``sum J E dV`` [W] (per unit length in a cut)."""
        return float(
            np.sum(state.plasma_current_density * state.electric_field * self._volumes)
        )
