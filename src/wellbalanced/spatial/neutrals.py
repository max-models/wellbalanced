"""The neutral gas of the RF driver: H and H2 as two fluids, coupled to the plasma.

Step R3.5 of the RF-driver plan: the neutral model of Zielke, Briefi, Lishev and Fantz
(Plasma Sources Sci. Technol. 31, 035019, 2022; section 2.3), so that the gas depletes
and heats where the plasma is dense, which changes the coupling at high power. Every
neutral species ``n`` (the atoms and the molecules) has its own density, velocity and
temperature, the Navier–Stokes equations of their eqs. (18)–(24):

    dn/dt + div(n u) = R_n,
    m n (du/dt + (u . grad) u) = -grad(n k T) + mu (lap u + 1/3 grad div u)
                                 - sum_j mu_nj n nu_nj (u - u_j),
    d/dt((3 + z) / 2 n k T) + div Q = P,    Q = (5 + z) / 2 k T n u - kappa grad k T,

with ``mu_nj = m m_j / (m + m_j)``, ``nu_nj = n_j K_nj`` the momentum-transfer frequency
on the partner ``j`` (the other neutral, ions, a fixed gas) and ``z`` the internal
degrees of freedom (0 for H, 3 for H2 in the paper: two rotational and one
vibrational). The viscosity and the heat conductivity are local, from the same
collisions (their eqs. 21 and 24; ``j`` includes ``n`` itself),

    mu = 2/3 m n k T / sum_j mu_nj nu_nj,   kappa = (5 + z) / 2 n k T / sum_j mu_nj nu_nj.

The power ``P`` holds the energy exchange of the collisions (friction and temperature,
the form of Schunk, Rev. Geophys. 15, 429, 1977, with the paper's reduced-mass
frequency: ``mu_nj n nu_nj / (m + m_j) (3 k (T_j - T) + m_j |u - u_j|^2)``; the first
part is the "directed energy transfer" that heats the gas where the ions drift), the
heat of the reactions (:attr:`~wellbalanced.reactions.Reaction.gas_heating`, e.g. the
Franck–Condon energy of dissociation, and the electrons' elastic loss), and the thermal
energy of the particles the reactions make and destroy: products are born with the
velocity and temperature of their heavy reactants (their centre of mass), so a
reaction that only moves particles between species conserves momentum and heat.

**The plasma** is an input (:meth:`NeutralFluid2D.set_plasma`, or
:meth:`NeutralFluid2D.plasma_from` a :class:`~wellbalanced.spatial.Fluid2D` state): its
densities and ``T_e`` for the reactions, the ions' velocities and temperature for the
collisions, and the ion flux to each wall face, which comes back as neutrals
(``ion_wall_products``, e.g. ``H3+ -> H2 + H``). The reactions are the
:class:`~wellbalanced.reactions.Reaction` objects of the plasma model (rates from
plasmacoll), the ones that make or destroy a neutral; their other species are plasma
species or ``fixed``. :func:`couple_neutrals` iterates the two models to a common
steady state: the neutrals in the plasma's sources and wall fluxes, the plasma in the
neutrals' densities and mean temperature (the plasma model has one gas temperature).

**Walls**, kinetically as in the paper (eqs. 25–31): each species leaves a wall cell
with the forward flux of a drifting half Maxwellian, ``n u_eff`` (eq. 26), and its
energy flux (eq. 29); it comes back reflected (all but the wall-loss probability),
recombined (:class:`~wellbalanced.global_model.WallReaction`: ``WallReaction("H",
gamma, {"H2": 0.5})`` for atoms that recombine with the probability ``gamma``) and from
the ions; what comes back carries ``(1 - alpha)`` of the incident energy and ``alpha``
of the wall's, ``(2 + z / 2) k T_wall`` per particle (the accommodation ``alpha``; the
ions come back accommodated). An :class:`Opening` (the grid) lets the fraction
``transparency`` of what hits it out: everything that comes back there is scaled by
``1 - transparency``. An :class:`Inlet` injects a species (particles per second over
its faces, ``(2 + z / 2) k T`` per particle). In the momentum equation the walls are
no-slip; the species' fluxes through them are the kinetic ones above.

**Discretisation**: the cells and faces of :class:`~wellbalanced.spatial.Fluid2D`
(a :class:`~wellbalanced.spatial.CellGrid` or a single-level
:class:`~wellbalanced.spatial.PatchGrid`, r–z or a cut; a face to a cell outside the
region is a wall, a ``"neumann"`` edge a symmetry plane). Finite volumes, every species'
particles and the total energy conserved to round-off. The velocities are at the cell
centres; the face mass flux is the mean of the two cells' velocities with a Rhie–Chow
correction (the difference of the face's pressure gradient and the mean of the two
cells', divided by the momentum equation's diagonal), which holds the pressure and
the velocity together without a staggered grid and vanishes for a smooth pressure
(second order). The viscous stress is the vector Laplacian (in r–z with its
``-u_r / r^2``) plus a third of the gradient of the divergence; ``grad mu`` is left out
(as in the paper, eq. 20). Backward Euler, all equations implicit, Newton's method in
the logarithms of the densities and temperatures with a finite-difference Jacobian on a
colouring of the cells (the stencil reaches two cells: the Rhie–Chow term and the
divergence's gradient); :meth:`NeutralFluid2D.steady_state` steps in pseudo-time with a
growing step and then solves the steady equations.

**Validity.** This is the continuum limit. At 0.3 Pa the mean free path of H2 is a few
centimetres against a driver of 24 cm (Knudsen numbers about 0.1–0.3); the paper
names the non-Maxwellian neutrals as the model's main limitation (its atoms have a
hot Franck–Condon population) and a kinetic check is due. The kinetic wall conditions
are what keeps the continuum model sensible at the walls.

Verification (``tests/unit/test_neutrals.py``): the kinetic wall fluxes against a
quadrature of the drifting half Maxwellian; the nuclei (``n_H + 2 n_H2``) conserved to
round-off over time steps with reflecting walls (and changing by exactly the ionized
nuclei with ionization on); the 0D limit (a closed box with a uniform plasma) against a
direct ODE solve, with first-order convergence in the time step; and the BUG driver
coupled to :class:`~wellbalanced.spatial.Fluid2D` (depletion, heating, the nuclei
balance of inlet, grid and ions).

Data: the collision rate coefficients, the wall probabilities, the accommodation and
the Franck–Condon energies are inputs. The paper takes them from Zielke's thesis
(Augsburg, 2021) and its ref. [31], not yet obtained (the plan's R0): the values in the
examples and tests are placeholders of the right size, said where they are used.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import xarray as xr
from plasmacoll.constants import BOLTZMANN, ELEMENTARY_CHARGE
from scipy.special import erf
from scope_profiler import profile

from wellbalanced._profiling import _PROFILE_TAGS
from wellbalanced.datasets import variable
from wellbalanced.global_model import WallReaction
from wellbalanced.reactions import Reaction
from wellbalanced.spatial.grid import CellField, CellGrid
from wellbalanced.spatial.regions import PlasmaRegion
from wellbalanced.species import Species

if TYPE_CHECKING:
    from wellbalanced.spatial.fluid import Fluid2D, Fluid2DState
    from wellbalanced.spatial.patches import PatchGrid

__all__ = [
    "ROOM_TEMPERATURE",
    "Collision",
    "CoupledNeutrals",
    "Inlet",
    "NeutralFluid2D",
    "NeutralSpecies",
    "NeutralState",
    "Opening",
    "couple_neutrals",
    "hard_sphere",
    "half_maxwellian_fluxes",
]

E = ELEMENTARY_CHARGE
#: 300 K in eV.
ROOM_TEMPERATURE = 300.0 * BOLTZMANN / ELEMENTARY_CHARGE
#: Floors of a density (m^-3) and a temperature (eV).
DENSITY_FLOOR = 1.0
TEMPERATURE_FLOOR = 1.0e-4
#: Newton: the largest change of a logarithm per iteration, of a velocity (in units of
#: the largest thermal speed), and the tolerance.
MAX_LOG_STEP = 1.0
MAX_VELOCITY_STEP = 0.2
NEWTON_TOLERANCE = 1.0e-10
SQRT_PI = float(np.sqrt(np.pi))


@dataclass(frozen=True)
class NeutralSpecies:
    """A neutral fluid: its species and internal degrees of freedom.

    Attributes:
        species: Name and mass (a neutral :class:`~wellbalanced.species.Species`).
        internal: The internal degrees of freedom ``z`` that store energy (0 for an
            atom; Zielke et al. take 3 for H2: two rotational, one vibrational).
    """

    species: Species
    internal: float = 0.0

    def __post_init__(self) -> None:
        """Validate the species."""
        if self.species.charge != 0:
            raise ValueError(f"{self.species.name}: a neutral species has no charge")
        if self.internal < 0.0:
            raise ValueError("internal degrees of freedom must be >= 0")

    @property
    def name(self) -> str:
        """The species' name."""
        return self.species.name

    @property
    def mass(self) -> float:
        """The mass [kg]."""
        return self.species.mass

    @property
    def heat_capacity(self) -> float:
        """``(3 + z) / 2``: the thermal energy per particle in units of ``k T``."""
        return 0.5 * (3.0 + self.internal)

    @property
    def enthalpy(self) -> float:
        """``(5 + z) / 2``: the enthalpy per particle in units of ``k T``."""
        return 0.5 * (5.0 + self.internal)

    @property
    def effusive_energy(self) -> float:
        """``2 + z / 2``: the energy per particle of an effusive flux, in ``k T``."""
        return 2.0 + 0.5 * self.internal


RateCoefficient = float | Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Collision:
    """Momentum-transfer collisions of a neutral species with a partner.

    The partner is another neutral of the model (the friction acts on both), the
    species itself (the collision then only enters the viscosity and the heat
    conductivity), a plasma species or a fixed gas. ``rate`` is the momentum-transfer
    rate coefficient ``K`` [m^3/s] (``nu = n_partner K``), a number or a function of
    the pair's temperature ``(m_j T + m T_j) / (m + m_j)`` [eV]; for resonant charge
    exchange (``H+ + H``), which swaps the velocities, ``K`` is twice its rate
    coefficient.
    """

    species: str
    partner: str
    rate: RateCoefficient
    name: str = ""

    def __post_init__(self) -> None:
        """Validate the collision."""
        if isinstance(self.rate, (int, float)) and self.rate < 0.0:
            raise ValueError("a rate coefficient must be >= 0")
        if not self.name:
            object.__setattr__(self, "name", f"{self.species}-{self.partner}")

    def coefficient(self, temperature: np.ndarray) -> np.ndarray:
        """``K`` [m^3/s] at the pair's temperature [eV]."""
        if isinstance(self.rate, (int, float)):
            return np.full(np.shape(temperature), float(self.rate))
        return np.asarray(self.rate(temperature), dtype=float)


def hard_sphere(
    cross_section: float, mass_a: float, mass_b: float
) -> Callable[[np.ndarray], np.ndarray]:
    """``K(T) = sigma sqrt(8 k T / (pi mu))``: hard spheres of ``sigma`` [m^2].

    The mean relative speed of a Maxwellian pair of reduced mass ``mu``; ``T`` in eV.
    """
    if cross_section < 0.0 or min(mass_a, mass_b) <= 0.0:
        raise ValueError("a hard sphere needs a cross section >= 0 and masses > 0")
    reduced = mass_a * mass_b / (mass_a + mass_b)

    def rate(temperature: np.ndarray) -> np.ndarray:
        t = np.maximum(np.asarray(temperature, dtype=float), TEMPERATURE_FLOOR)
        return cross_section * np.sqrt(8.0 * E * t / (np.pi * reduced))

    return rate


@dataclass(frozen=True)
class Opening:
    """Wall faces through which the neutrals leave (the extraction grid).

    ``where`` selects the wall faces by their centres: a shape with
    ``contains_points`` (``(2, n)`` points) or a function of the two coordinates
    returning a mask. The fraction ``transparency`` of what hits them leaves.
    """

    where: Any
    transparency: float
    name: str = "opening"

    def __post_init__(self) -> None:
        """Validate the opening."""
        if not 0.0 <= self.transparency <= 1.0:
            raise ValueError("a transparency must be in [0, 1]")


@dataclass(frozen=True)
class Inlet:
    """A gas inlet: ``rate`` particles of ``species`` per second over its wall faces.

    ``where`` selects the faces as for :class:`Opening`; the gas comes in effusively
    at ``temperature`` [eV] (the paper's eq. 31), spread over the faces by area.
    """

    where: Any
    species: str
    rate: float
    temperature: float = ROOM_TEMPERATURE
    name: str = "inlet"

    def __post_init__(self) -> None:
        """Validate the inlet."""
        if self.rate < 0.0 or self.temperature <= 0.0:
            raise ValueError("an inlet needs a rate >= 0 and a temperature > 0")


def half_maxwellian_fluxes(
    density: np.ndarray,
    temperature: np.ndarray,
    velocity: np.ndarray,
    mass: float,
    internal: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The particle and energy flux of a drifting Maxwellian through a plane.

    Zielke et al.'s eqs. (26) and (29): the particles that cross the plane in the
    direction of the normal of a gas of ``density`` [m^-3], ``temperature`` [eV] and
    velocity ``velocity`` [m/s] along that normal, ``n u_eff`` [m^-2 s^-1], and the
    energy they carry, ``n u_eff U`` [eV m^-2 s^-1] (translation and ``z / 2 k T`` of
    internal energy each). Without drift these are ``n v / 4`` and ``(2 + z / 2) T``
    times it.
    """
    vp = np.sqrt(2.0 * E * np.maximum(temperature, TEMPERATURE_FLOOR) / mass)
    s = velocity / vp
    gauss = np.exp(-(s**2))
    tail = SQRT_PI * s * (1.0 + erf(s))
    particles = density * vp / (2.0 * SQRT_PI) * (gauss + tail)
    normal = (
        density
        * mass
        * vp**3
        / (4.0 * SQRT_PI * E)
        * ((1.0 + s**2) * gauss + tail * (1.5 + s**2))
    )
    energy = normal + particles * (1.0 + 0.5 * internal) * temperature
    return particles, energy


def _select(where: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The points a shape (``contains_points``) or a mask function selects."""
    if hasattr(where, "contains_points"):
        inside = where.contains_points(np.stack([x, y]))
        if hasattr(inside, "get"):
            inside = inside.get()
    else:
        inside = where(x, y)
    return np.asarray(inside, dtype=bool).reshape(x.shape)


@dataclass
class NeutralState:
    """The neutral gas on the grid: fields over the grid's cells, zero outside.

    Attributes:
        grid: The grid.
        region: The gas cells (a mask of the grid's shape).
        unknowns: The model's vector of unknowns, to continue from.
        densities: Density of every neutral species [m^-3].
        velocities: In-plane velocity ``(u_1, u_2)`` [m/s] of every species.
        temperatures: Temperature [eV] of every species.
        reaction_rates: Volume rate of every reaction [m^-3 s^-1].
        time: Time [s]; ``inf`` for a steady state.
        converged: Whether the last solve converged.
    """

    grid: CellGrid | PatchGrid
    region: np.ndarray
    unknowns: np.ndarray
    densities: dict[str, np.ndarray]
    velocities: dict[str, tuple[np.ndarray, np.ndarray]]
    temperatures: dict[str, np.ndarray]
    reaction_rates: dict[str, np.ndarray]
    time: float = 0.0
    converged: bool = True

    def total(self, values: np.ndarray) -> float:
        """The integral of a field over the gas (per metre along ``z`` in a cut)."""
        return float(np.sum(values[self.region] * self.grid.cell_volumes[self.region]))

    @property
    def volume(self) -> float:
        """The gas volume [m^3] (per metre along ``z`` in a cut)."""
        return float(np.sum(self.grid.cell_volumes[self.region]))

    def mean(self, values: np.ndarray) -> float:
        """The volume average of a field over the gas."""
        return self.total(values) / self.volume

    @property
    def pressure(self) -> np.ndarray:
        """The total pressure ``sum n k T`` [Pa]."""
        return sum(  # type: ignore[return-value]
            E * self.densities[n] * self.temperatures[n] for n in self.densities
        )

    @property
    def gas_temperature(self) -> np.ndarray:
        """The density-weighted mean temperature of the species [eV]."""
        density = sum(self.densities.values())
        weighted = sum(self.densities[n] * self.temperatures[n] for n in self.densities)
        return np.where(density > 0, weighted / np.where(density > 0, density, 1), 0.0)

    def to_dataset(self) -> xr.Dataset:
        """The state as a labelled dataset, fields over the grid's cells."""
        dims = self.grid.dims
        species = list(self.densities)
        reactions = list(self.reaction_rates)
        first, second = self.grid.coordinate_names
        data = {
            "density": variable(
                ("species", *dims), [self.densities[s] for s in species], "density",
                "m^-3",
            ),
            "temperature": variable(
                ("species", *dims),
                [self.temperatures[s] for s in species],
                "temperature",
                "eV",
            ),
            "pressure": variable(dims, self.pressure, "pressure", "Pa"),
            "region": variable(dims, self.region, "gas cells"),
            "converged": variable((), self.converged, "converged"),
        }  # fmt: skip
        for k, axis in enumerate((first, second)):
            data[f"velocity_{axis}"] = variable(
                ("species", *dims),
                [self.velocities[s][k] for s in species],
                f"velocity u_{axis}",
                "m/s",
            )
        coords: dict[str, Any] = {
            **self.grid.coordinates(),
            "species": ("species", species, {"long_name": "species"}),
        }
        if reactions:
            data["reaction_rate"] = variable(
                ("reaction", *dims),
                [self.reaction_rates[r] for r in reactions],
                "reaction rate",
                "m^-3 s^-1",
            )
            coords["reaction"] = ("reaction", reactions, {"long_name": "reaction"})
        return xr.Dataset(
            data,
            coords=coords,
            attrs={"time": self.time, "geometry": self.grid.geometry},
        )


def _factorized(matrix: sp.csc_matrix):
    """The sparse LU of a Jacobian, or ``None`` if it is singular or not finite."""
    matrix = sp.csc_matrix(matrix)
    if not np.all(np.isfinite(matrix.data)):
        return None
    try:
        return spla.splu(matrix)
    except RuntimeError:
        return None


class NeutralFluid2D:
    """The neutral gas on the cells of a grid; see the module docstring.

    Args:
        grid: A :class:`CellGrid` or a single-level :class:`PatchGrid`.
        species: The neutral fluids.
        reactions: Volume reactions; those that make or destroy a neutral are used,
            their other species must be plasma species (``plasma_species``) or
            ``fixed``.
        region: Where the gas is: a :class:`PlasmaRegion`, a mask, or ``None`` (every
            cell).
        plasma_species: The plasma's species (names and masses), the electrons
            (``"e"``) included, and the masses of ``fixed`` gases (a fixed gas not
            listed is taken as heavy as the species it collides with).
        collisions: The momentum-transfer collisions of the neutrals.
        wall_reactions: Neutrals lost at the walls with a probability per impact and
            what they come back as (atoms recombining to molecules).
        ion_wall_products: What an ion that reaches a wall comes back as,
            ``{ion: {neutral: number}}``.
        wall_temperature: ``T_wall`` [eV].
        accommodation: The thermal accommodation ``alpha`` at the walls, one or per
            species.
        openings: Wall faces the gas leaves through (the grid).
        inlets: Gas inlets.
        fixed: Densities of species not carried [m^-3] (cell fields), e.g. a
            background gas that only collides.
    """

    #: The largest batch of states the Jacobian evaluates at once (in numbers).
    BATCH_SIZE = 4_000_000

    # --- geometry --------------------------------------------------------------------

    def _faces(self, grid: CellGrid | PatchGrid) -> None:
        """The interior faces and the walls, in region-cell indices (as Fluid2D's).

        The divergence and the gradient are sparse matrices, so that the residual
        takes a batch of states at once (the columns of the Jacobian).
        """
        faces = grid.transport_faces()
        index = np.full(grid.size, -1)
        index[self.cells] = np.arange(self.size)
        low, high = index[faces.low], index[faces.high]
        inner = (low >= 0) & (high >= 0)
        self._low, self._high = low[inner], high[inner]
        self._axis = faces.axis[inner]
        self._area = faces.area[inner]
        self._distance = faces.delta_low[inner] + faces.delta_high[inner]
        to_high = (low >= 0) & (high < 0)
        to_low = (low < 0) & (high >= 0)
        edge = index[faces.boundary_cell]
        edge_wall = (edge >= 0) & ~faces.boundary_symmetry
        self._wall_cell = np.concatenate([low[to_high], high[to_low], edge[edge_wall]])
        self._wall_axis = np.concatenate(
            [faces.axis[to_high], faces.axis[to_low], faces.boundary_axis[edge_wall]]
        )
        self._wall_sign = np.concatenate(
            [
                np.ones(np.count_nonzero(to_high)),
                -np.ones(np.count_nonzero(to_low)),
                np.where(faces.boundary_side[edge_wall] == 1, 1.0, -1.0),
            ]
        )
        self._wall_area = np.concatenate(
            [faces.area[to_high], faces.area[to_low], faces.boundary_area[edge_wall]]
        )
        self._wall_delta = np.concatenate(
            [
                faces.delta_low[to_high],
                faces.delta_high[to_low],
                faces.boundary_delta[edge_wall],
            ]
        )
        n, nf, nw = self.size, self._low.size, self._wall_cell.size
        faces_index = np.arange(nf)
        volume = self.volumes
        self._div_interior = sp.csr_matrix(
            (
                np.concatenate(
                    [self._area / volume[self._low], -self._area / volume[self._high]]
                ),
                (
                    np.concatenate([self._low, self._high]),
                    np.concatenate([faces_index, faces_index]),
                ),
            ),
            shape=(n, nf),
        )
        self._div_wall = sp.csr_matrix(
            (
                self._wall_area / volume[self._wall_cell],
                (self._wall_cell, np.arange(nw)),
            ),
            shape=(n, nw),
        )
        # the gradient: the area-weighted mean of the differences on a cell's
        # interior faces of each axis
        self._grad = []
        for axis in (0, 1):
            along = np.where(self._axis == axis, self._area, 0.0)
            total = np.bincount(self._low, along, n) + np.bincount(self._high, along, n)
            weight = np.where(total > 0.0, 1.0 / np.where(total > 0, total, 1.0), 0.0)
            values = np.concatenate(
                [along * weight[self._low], along * weight[self._high]]
            )
            matrix = sp.csr_matrix(
                (values, (np.concatenate([self._low, self._high]),
                          np.concatenate([faces_index, faces_index]))),
                shape=(n, nf),
            )  # fmt: skip
            self._grad.append(matrix)
        # sum of A / d over a cell's faces per volume: the viscous diagonal
        self._stiffness = (
            (
                np.bincount(self._low, self._area / self._distance, n)
                + np.bincount(self._high, self._area / self._distance, n)
                + np.bincount(self._wall_cell, self._wall_area / self._wall_delta, n)
            )
            / volume
        )[:, None]
        self._first = (self._axis == 0)[:, None]
        self._wall_first = (self._wall_axis == 0)[:, None]
        self._d = self._distance[:, None]

    def _boundaries(self) -> None:
        """The transparency and the injected fluxes of every wall face."""
        cell = self.cells[self._wall_cell]
        x = np.asarray(self.grid.xx).ravel()[cell]
        y = np.asarray(self.grid.yy).ravel()[cell]
        shift = self._wall_sign * self._wall_delta
        x = x + np.where(self._wall_axis == 0, shift, 0.0)
        y = y + np.where(self._wall_axis == 1, shift, 0.0)
        #: The centres of the wall faces, ``(x, y)`` or ``(r, z)``.
        self.wall_centres = (x, y)
        nw = self._wall_cell.size
        self._transparency = np.zeros(nw)
        self._opening = np.full(nw, -1)
        for k, opening in enumerate(self.openings):
            chosen = _select(opening.where, x, y)
            if not chosen.any():
                raise ValueError(f"{opening.name}: no wall face selected")
            self._transparency[chosen] = opening.transparency
            self._opening[chosen] = k
        self._injected = {s.name: np.zeros(nw) for s in self.species}
        self._injected_energy = {s.name: np.zeros(nw) for s in self.species}
        self._inlet = np.full(nw, -1)
        for k, inlet in enumerate(self.inlets):
            chosen = _select(inlet.where, x, y)
            if not chosen.any():
                raise ValueError(f"{inlet.name}: no wall face selected")
            area = float(np.sum(self._wall_area[chosen]))
            flux = np.where(chosen, inlet.rate / area, 0.0)
            neutral = self._by_name[inlet.species]
            self._injected[inlet.species] += flux
            self._injected_energy[inlet.species] += (
                flux * neutral.effusive_energy * inlet.temperature
            )
            self._inlet[chosen] = k

    def _on_cells(self, value: CellField) -> np.ndarray:
        """A cell field (or an array over the region's cells) on the region's cells."""
        if not callable(value):
            array = np.asarray(value)
            if array.shape == (self.size,) and (
                self.size != self.grid.size or array.shape != self.grid.shape
            ):
                return array.astype(float)
        return self.grid.cell_field(value).ravel()[self.cells]

    def _colouring(self) -> np.ndarray:
        """A colouring of the cells for the Jacobian; the residual reaches two cells.

        Two cells share a colour when no residual depends on both, so their columns
        of the Jacobian come out of one evaluation.
        """
        n = self.size
        ones = np.ones(self._low.size)
        adjacency = sp.coo_matrix((ones, (self._low, self._high)), shape=(n, n))
        adjacency = (adjacency + adjacency.T + sp.identity(n)).tocsr()
        adjacency.data[:] = 1.0
        reach = (adjacency @ adjacency).tocsr()
        reach.data[:] = 1.0
        conflict = (reach.T @ reach).tocsr()
        colours = np.full(n, -1)
        for cell in range(n):
            start, stop = conflict.indptr[cell], conflict.indptr[cell + 1]
            used = set(colours[conflict.indices[start:stop]].tolist())
            colour = 0
            while colour in used:
                colour += 1
            colours[cell] = colour
        self._reach = reach.tocoo()
        return colours

    # --- the collisions and reactions ------------------------------------------------

    def _collision_pairs(self) -> list[dict[str, Any]]:
        """The collisions with their masses and the kind of partner."""
        pairs = []
        for c in self.collisions:
            m = self._by_name[c.species].mass
            if c.partner in self._by_name:
                kind, mj = "neutral", self._by_name[c.partner].mass
            elif c.partner in self._fixed:  # at rest; its mass if listed, else m
                listed = self._plasma_species.get(c.partner)
                kind, mj = "fixed", m if listed is None else listed.mass
            else:
                kind, mj = "plasma", self._plasma_species[c.partner].mass
            pairs.append({"collision": c, "kind": kind, "mass": m, "partner_mass": mj})
        return pairs

    def __init__(
        self,
        grid: CellGrid | PatchGrid,
        species: Sequence[NeutralSpecies],
        reactions: Sequence[Reaction] = (),
        *,
        region: PlasmaRegion | np.ndarray | None = None,
        plasma_species: Sequence[Species] = (),
        collisions: Sequence[Collision] = (),
        wall_reactions: Sequence[WallReaction] = (),
        ion_wall_products: Mapping[str, Mapping[str, float]] | None = None,
        wall_temperature: float = ROOM_TEMPERATURE,
        accommodation: float | Mapping[str, float] = 1.0,
        openings: Sequence[Opening] = (),
        inlets: Sequence[Inlet] = (),
        fixed: Mapping[str, CellField] | None = None,
    ) -> None:
        """Index the cells, faces, walls and unknowns; check the names."""
        self.grid = grid
        self.species = tuple(species)
        names = [s.name for s in self.species]
        if not names:
            raise ValueError("the model needs a neutral species")
        if len(set(names)) != len(names):
            raise ValueError("species names must be unique")
        if wall_temperature <= 0.0:
            raise ValueError("the wall temperature must be > 0")
        self.wall_temperature = float(wall_temperature)
        self._by_name = {s.name: s for s in self.species}
        self._plasma_species = {s.name: s for s in plasma_species}
        overlap = set(names) & set(self._plasma_species)
        if overlap:
            raise ValueError(f"species both neutral and plasma: {sorted(overlap)}")
        fixed = dict(fixed or {})
        known = set(names) | set(self._plasma_species) | set(fixed)
        self.reactions = tuple(
            r for r in reactions if set(names) & (set(r.reactants) | set(r.products))
        )
        for reaction in self.reactions:
            unknown = (set(reaction.reactants) | set(reaction.products)) - known
            if unknown:
                raise ValueError(f"{reaction.name}: unknown species {sorted(unknown)}")
        self.collisions = tuple(collisions)
        for collision in self.collisions:
            if collision.species not in self._by_name:
                raise ValueError(f"{collision.name}: {collision.species} is no neutral")
            if collision.partner not in known:
                raise ValueError(
                    f"{collision.name}: unknown partner {collision.partner}"
                )
        self.wall_reactions = tuple(wall_reactions)
        for wall in self.wall_reactions:
            if wall.species not in self._by_name:
                raise ValueError(f"wall reaction: {wall.species} is no neutral")
            if set(wall.products) - set(names):
                raise ValueError("wall reactions must return neutrals of the model")
        lost = dict.fromkeys(names, 0.0)
        for wall in self.wall_reactions:
            lost[wall.species] += wall.probability
        if any(p > 1.0 for p in lost.values()):
            raise ValueError("the wall probabilities of a species add up to > 1")
        self._reflected = {name: 1.0 - p for name, p in lost.items()}
        self.ion_wall_products = {
            ion: dict(products) for ion, products in (ion_wall_products or {}).items()
        }
        for ion, products in self.ion_wall_products.items():
            if ion not in self._plasma_species:
                raise ValueError(f"ion wall products: {ion} is no plasma species")
            if set(products) - set(names):
                raise ValueError("ions must come back as neutrals of the model")
        if isinstance(accommodation, Mapping):
            self.accommodation = {n: float(accommodation.get(n, 1.0)) for n in names}
        else:
            self.accommodation = dict.fromkeys(names, float(accommodation))
        if any(not 0.0 <= a <= 1.0 for a in self.accommodation.values()):
            raise ValueError("an accommodation coefficient must be in [0, 1]")
        for inlet in inlets:
            if inlet.species not in self._by_name:
                raise ValueError(f"{inlet.name}: {inlet.species} is no neutral")
        self.openings = tuple(openings)
        self.inlets = tuple(inlets)

        # the cells and faces
        if region is None:
            mask = np.ones(grid.shape, dtype=bool)
        elif isinstance(region, PlasmaRegion):
            mask = region.mask(grid)
        else:
            mask = np.asarray(region, dtype=bool).reshape(grid.shape)
        if not mask.any():
            raise ValueError("the region covers no cell")
        self.region = mask
        self.cells = np.flatnonzero(mask.ravel())
        self.size = self.cells.size
        self.volumes = grid.cell_volumes.ravel()[self.cells]
        self._faces(grid)
        self._boundaries()
        self._radius = (
            np.asarray(grid.xx).ravel()[self.cells][:, None]
            if grid.cylindrical
            else None
        )
        self._length = float(np.sqrt(np.sum(grid.cell_areas.ravel()[self.cells])))

        # the unknowns per cell and species: log n, u_1, u_2, log T
        self._nvar = 4 * len(self.species)
        self._is_log = np.zeros(self._nvar, dtype=bool)
        self._is_log[0::4] = True
        self._is_log[3::4] = True
        self._fixed = {n: self._on_cells(v)[:, None] for n, v in fixed.items()}
        self._pairs = self._collision_pairs()
        self.set_plasma({}, 0.0)
        self._colours = self._colouring()
        self._scales: dict[str, Any] | None = None
        #: Newton's tolerance of a velocity is relative to this speed [m/s]: the
        #: thermal speed of the lightest species at the wall temperature.
        self._velocity_scale = min(
            float(np.sqrt(2.0 * E * self.wall_temperature / s.mass))
            for s in self.species
        )
        self._iterations = 0

    def on_grid(self, values: np.ndarray, outside: float = 0.0) -> np.ndarray:
        """Values on the region's cells as a field over the grid, ``outside`` elsewhere."""
        values = np.asarray(values).reshape(self.size)
        result = np.full(self.grid.size, outside, dtype=values.dtype)
        result[self.cells] = values
        return result.reshape(self.grid.shape)

    def _gradient(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The gradient on the cells (a batch ``(cells, B)``) from the interior faces."""
        face = (values[self._high] - values[self._low]) / self._d
        return self._grad[0] @ face, self._grad[1] @ face

    def _divergence(self, interior: np.ndarray, walls: np.ndarray | None) -> np.ndarray:
        """``div F`` on the cells from the face fluxes (outward at the walls)."""
        out = self._div_interior @ interior
        if walls is not None:
            out = out + self._div_wall @ walls
        return out

    # --- the plasma ------------------------------------------------------------------

    def set_plasma(
        self,
        densities: Mapping[str, CellField],
        electron_temperature: CellField,
        *,
        ion_velocities: Mapping[str, tuple[CellField, CellField]] | None = None,
        ion_temperature: CellField | None = None,
        ion_wall_flux: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        """Set the plasma the gas sees.

        Args:
            densities: Densities of the plasma species [m^-3], the electrons included.
            electron_temperature: ``T_e`` [eV].
            ion_velocities: In-plane velocities of ions [m/s] (zero if not given).
            ion_temperature: The ions' temperature [eV] (the wall's if not given).
            ion_wall_flux: The flux of every ion into each wall face [m^-2 s^-1],
                over :attr:`wall_faces`; it comes back as ``ion_wall_products``.
        """
        unknown = set(densities) - (set(self._plasma_species) - set(self._fixed))
        if unknown:
            raise ValueError(f"unknown plasma species {sorted(unknown)}")
        self._plasma = {n: self._on_cells(v)[:, None] for n, v in densities.items()}
        te = self._on_cells(electron_temperature)[:, None]
        self._electron_temperature = te
        self._ion_temperature = (
            np.full((self.size, 1), self.wall_temperature)
            if ion_temperature is None
            else self._on_cells(ion_temperature)[:, None]
        )
        self._ion_velocities = {
            n: (self._on_cells(v[0])[:, None], self._on_cells(v[1])[:, None])
            for n, v in (ion_velocities or {}).items()
        }
        flux = {}
        for name, values in (ion_wall_flux or {}).items():
            array = np.asarray(values, dtype=float)
            if array.shape != self._wall_cell.shape:
                raise ValueError(f"{name}: one ion wall flux per wall face")
            flux[name] = array[:, None]
        self._ion_wall_flux = flux
        # the electron-impact coefficients depend on T_e only: once per plasma
        self._electron_coefficients = {
            k: r.coefficients(np.maximum(te, TEMPERATURE_FLOOR), 0.0)
            for k, r in enumerate(self.reactions)
            if r.temperature == "electron"
        }

    def plasma_from(self, fluid: Fluid2D, state: Fluid2DState) -> None:
        """Set the plasma of a :class:`~wellbalanced.spatial.Fluid2D` state.

        Its densities, ``T_e``, ion velocities and gas temperature (the ions'), and its
        ion fluxes to the walls, matched to this model's wall faces (the two must be on
        the same grid; a wall face of this model the plasma does not reach gets none).
        """
        if fluid.grid.size != self.grid.size:
            raise ValueError("the plasma and the gas must be on the same grid")
        cells = self.cells
        densities = {n: v.ravel()[cells] for n, v in state.densities.items()}
        velocities = {
            n: (v[0].ravel()[cells], v[1].ravel()[cells])
            for n, v in state.velocities.items()
        }
        faces = fluid.wall_faces
        fluxes = fluid.wall_fluxes(state)
        key = {
            (int(c), int(a), float(s)): k
            for k, (c, a, s) in enumerate(
                zip(faces["cell"], faces["axis"], faces["sign"], strict=True)
            )
        }
        match = np.array(
            [
                key.get((int(c), int(a), float(s)), -1)
                for c, a, s in zip(
                    self.cells[self._wall_cell], self._wall_axis, self._wall_sign,
                    strict=True,
                )
            ],
            dtype=int,
        )  # fmt: skip
        ion_flux = {
            ion: np.where(
                match >= 0, np.asarray(fluxes[ion])[np.maximum(match, 0)], 0.0
            )
            for ion in self.ion_wall_products
            if ion in fluxes
        }
        self.set_plasma(
            densities,
            state.electron_temperature.ravel()[cells],
            ion_velocities=velocities,
            ion_temperature=np.full(self.size, fluid.gas_temperature),
            ion_wall_flux=ion_flux,
        )

    # --- the unknowns ----------------------------------------------------------------

    def pack(
        self,
        densities: Mapping[str, CellField],
        temperatures: CellField | Mapping[str, CellField] = ROOM_TEMPERATURE,
        velocities: Mapping[str, tuple[CellField, CellField]] | None = None,
    ) -> np.ndarray:
        """The unknowns of fields: densities and temperatures by name, velocities.

        Fields are cell fields of the grid or arrays over the region's cells; the gas
        is at rest unless ``velocities`` are given.
        """
        u = np.zeros((self.size, self._nvar))
        for k, s in enumerate(self.species):
            u[:, 4 * k] = np.log(
                np.maximum(self._on_cells(densities[s.name]), DENSITY_FLOOR)
            )
            if isinstance(temperatures, Mapping):
                t = self._on_cells(temperatures[s.name])
            else:
                t = self._on_cells(temperatures)
            u[:, 4 * k + 3] = np.log(np.maximum(t, TEMPERATURE_FLOOR))
            if velocities is not None and s.name in velocities:
                u[:, 4 * k + 1] = self._on_cells(velocities[s.name][0])
                u[:, 4 * k + 2] = self._on_cells(velocities[s.name][1])
        return u.ravel()

    def _unpack(self, unknowns: np.ndarray):
        """Densities, velocities and temperatures, each ``(cells, B)``."""
        u = unknowns.reshape(self.size, self._nvar, -1)
        densities, velocities, temperatures = {}, {}, {}
        for k, s in enumerate(self.species):
            densities[s.name] = np.exp(np.clip(u[:, 4 * k], -700.0, 700.0))
            velocities[s.name] = (u[:, 4 * k + 1], u[:, 4 * k + 2])
            temperatures[s.name] = np.exp(np.clip(u[:, 4 * k + 3], -50.0, 50.0))
        return densities, velocities, temperatures

    def _partner(self, name, densities, velocities, temperatures):
        """Density, velocity and temperature of any species on the cells."""
        if name in densities:
            return densities[name], velocities[name], temperatures[name]
        zero = np.zeros((self.size, 1))
        if name in self._fixed:
            wall = np.full((self.size, 1), self.wall_temperature)
            return self._fixed[name], (zero, zero), wall
        if name in self._plasma_species:
            if name not in self._plasma:
                raise ValueError(f"no density of the plasma species {name!r} set")
            velocity = self._ion_velocities.get(name, (zero, zero))
            temperature = (
                self._electron_temperature if name == "e" else self._ion_temperature
            )
            return self._plasma[name], velocity, temperature
        raise ValueError(f"unknown species {name!r}")

    def _collisions(self, densities, velocities, temperatures, batch: int):
        """Friction forces, exchanged powers and the viscous denominators.

        Returns ``(force, power, denominator, friction)`` by species: the force
        [N/m^3] per component, the power gained [eV m^-3 s^-1], ``sum_j mu_nj n_j K``
        [kg/s] over every partner and the same sum without the species itself.
        """
        shape = (self.size, batch)
        force = {s.name: [np.zeros(shape), np.zeros(shape)] for s in self.species}
        power = {s.name: np.zeros(shape) for s in self.species}
        denominator = {s.name: np.zeros(shape) for s in self.species}
        friction = {s.name: np.zeros(shape) for s in self.species}
        for pair in self._pairs:
            c: Collision = pair["collision"]
            name, other = c.species, c.partner
            m, mj = pair["mass"], pair["partner_mass"]
            nj, uj, tj = self._partner(other, densities, velocities, temperatures)
            n, u, t = densities[name], velocities[name], temperatures[name]
            reduced = m * mj / (m + mj)
            k = c.coefficient((mj * t + m * tj) / (m + mj))
            if other == name:
                denominator[name] += reduced * n * k
                continue
            denominator[name] += reduced * nj * k
            friction[name] += reduced * nj * k
            rate = reduced * n * nj * k  # kg m^-3 s^-1
            du = (u[0] - uj[0], u[1] - uj[1])
            du2 = du[0] ** 2 + du[1] ** 2
            force[name][0] -= rate * du[0]
            force[name][1] -= rate * du[1]
            power[name] += rate / (m + mj) * (3.0 * (tj - t) + mj * du2 / E)
            if pair["kind"] == "neutral":
                denominator[other] += reduced * n * k
                friction[other] += reduced * n * k
                force[other][0] += rate * du[0]
                force[other][1] += rate * du[1]
                power[other] += rate / (m + mj) * (3.0 * (t - tj) + m * du2 / E)
        return force, power, denominator, friction

    def _mass(self, name: str) -> float:
        if name in self._by_name:
            return self._by_name[name].mass
        if name in self._plasma_species:
            return self._plasma_species[name].mass
        return 0.0  # a fixed gas: no velocity, no weight in the centre of mass

    def _reactions(self, densities, velocities, temperatures, batch: int):
        """The reactions' rates and their sources of particles, momentum and energy.

        Returns ``(rates, particles, momentum, energy)``: rates by reaction; by species
        the net particle source [m^-3 s^-1], the momentum source [N/m^3] per component
        and the energy source [eV m^-3 s^-1] (heat and the thermal energy of the
        particles made and destroyed).
        """
        shape = (self.size, batch)
        particles = {s.name: np.zeros(shape) for s in self.species}
        momentum = {s.name: [np.zeros(shape), np.zeros(shape)] for s in self.species}
        energy = {s.name: np.zeros(shape) for s in self.species}
        rates: dict[str, np.ndarray] = {}
        te = self._electron_temperature
        for index, reaction in enumerate(self.reactions):
            heavy = [r for r in reaction.reactants if r != "e"]
            states = [
                self._partner(r, densities, velocities, temperatures) for r in heavy
            ]
            masses = [self._mass(r) for r in heavy]
            if reaction.temperature == "electron":
                rate = self._electron_coefficients[index]
            else:
                rate = reaction.coefficients(0.0, states[0][2] if states else te)
            for name in reaction.reactants:
                rate = (
                    rate * self._partner(name, densities, velocities, temperatures)[0]
                )
            rate = np.broadcast_to(rate, shape)
            rates[reaction.name] = rates.get(reaction.name, 0.0) + rate
            # the products' velocity and temperature: the heavy reactants' centre of mass
            total = sum(masses)
            if total > 0.0:
                born_u = [
                    sum(mm * st[1][a] for mm, st in zip(masses, states, strict=True))
                    / total
                    for a in (0, 1)
                ]
                born_t = (
                    sum(mm * st[2] for mm, st in zip(masses, states, strict=True))
                    / total
                )
            else:
                born_u, born_t = [0.0, 0.0], self.wall_temperature
            for name, count in reaction.stoichiometry.items():
                if name not in self._by_name:
                    continue
                neutral = self._by_name[name]
                particles[name] += count * rate
                if count > 0:
                    u, t = born_u, born_t
                else:
                    u, t = velocities[name], temperatures[name]
                for a in (0, 1):
                    momentum[name][a] += count * rate * neutral.mass * u[a]
                energy[name] += count * rate * neutral.heat_capacity * t
            # the heat goes to the neutral products (or the neutral reactants)
            receivers = [p for p in reaction.products if p in self._by_name] or [
                r for r in reaction.reactants if r in self._by_name
            ]
            if receivers and (reaction.gas_heating or reaction.elastic_mass):
                target = temperatures[receivers[0]]
                heat = reaction.gas_gain(te, target) * rate  # type: ignore[arg-type]
                for name in receivers:
                    energy[name] += heat / len(receivers)
        return rates, particles, momentum, energy

    # --- the walls -------------------------------------------------------------------

    def _walls(self, densities, velocities, temperatures, batch: int):
        """The net particle and energy flux of every species out through each wall face.

        Returns ``(particles, energy, back)``: by species, the net fluxes and the
        flux that comes back (before the openings' transparency), ``(walls, B)``.
        """
        wc, tw = self._wall_cell, self.wall_temperature
        sign = self._wall_sign[:, None]
        forward, forward_energy = {}, {}
        for s in self.species:
            u = velocities[s.name]
            normal = sign * np.where(self._wall_first, u[0][wc], u[1][wc])
            forward[s.name], forward_energy[s.name] = half_maxwellian_fluxes(
                densities[s.name][wc], temperatures[s.name][wc], normal, s.mass,
                s.internal,
            )  # fmt: skip
        shape = (wc.size, batch)
        back = {s.name: np.zeros(shape) for s in self.species}
        back_energy = {s.name: np.zeros(shape) for s in self.species}

        def comes_back(incident: str, product: str, number: float) -> None:
            alpha = self.accommodation[incident]
            flux = number * forward[incident]
            back[product] += flux
            back_energy[product] += (1.0 - alpha) * number * forward_energy[
                incident
            ] + alpha * flux * self._by_name[product].effusive_energy * tw

        for s in self.species:
            if self._reflected[s.name] > 0.0:
                comes_back(s.name, s.name, self._reflected[s.name])
        for wall in self.wall_reactions:
            for product, count in wall.products.items():
                comes_back(wall.species, product, wall.probability * count)
        for ion, products in self.ion_wall_products.items():
            flux = self._ion_wall_flux.get(ion)
            if flux is None:
                continue
            for product, count in products.items():
                energy = self._by_name[product].effusive_energy * tw
                back[product] += count * flux
                back_energy[product] += count * flux * energy
        keep = (1.0 - self._transparency)[:, None]
        particles, energy = {}, {}
        for s in self.species:
            name = s.name
            particles[name] = (
                forward[name] - keep * back[name] - self._injected[name][:, None]
            )
            energy[name] = (
                forward_energy[name]
                - keep * back_energy[name]
                - self._injected_energy[name][:, None]
            )
        return particles, energy, back

    # --- the equations ---------------------------------------------------------------

    def _terms(self, densities, velocities, temperatures, dt: float):
        """Every flux and source of the equations at a (batch of) state(s)."""
        batch = next(iter(densities.values())).shape[1]
        force, exchange, denominator, friction = self._collisions(
            densities, velocities, temperatures, batch
        )
        rates, particles, momentum, heat = self._reactions(
            densities, velocities, temperatures, batch
        )
        wall_particles, wall_energy, back = self._walls(
            densities, velocities, temperatures, batch
        )
        low, high, d, first = self._low, self._high, self._d, self._first
        wc = self._wall_cell
        fluxes = {}
        for s in self.species:
            name, m = s.name, s.mass
            n, u, t = densities[name], velocities[name], temperatures[name]
            p = E * n * t
            denom = np.maximum(denominator[name], 1e-300)
            viscosity = 2.0 / 3.0 * m * n * E * t / denom
            conductivity = s.enthalpy * n * E * t / denom  # 1/(m s), for T in eV
            diagonal = friction[name] * n + viscosity * self._stiffness
            if np.isfinite(dt):
                diagonal = diagonal + m * n / dt
            gradient = self._gradient(p)
            u_face = 0.5 * np.where(
                first, u[0][low] + u[0][high], u[1][low] + u[1][high]
            )
            g_face = 0.5 * np.where(
                first,
                gradient[0][low] + gradient[0][high],
                gradient[1][low] + gradient[1][high],
            )
            weight = n / np.maximum(diagonal, 1e-300)
            correction = (
                0.5 * (weight[low] + weight[high]) * ((p[high] - p[low]) / d - g_face)
            )
            flux = 0.5 * (n[low] + n[high]) * u_face - correction
            energy = (
                s.enthalpy * 0.5 * (t[low] + t[high]) * flux
                - 0.5
                * (conductivity[low] + conductivity[high])
                * (t[high] - t[low])
                / d
            )
            mu_face = 0.5 * (viscosity[low] + viscosity[high])
            stress = [
                m * flux * 0.5 * (c[low] + c[high]) - mu_face * (c[high] - c[low]) / d
                for c in u
            ]
            # no slip: the wall's viscous stress on the cell's velocity
            wall_mu = viscosity[wc] / self._wall_delta[:, None]
            fluxes[name] = {
                "particles": flux,
                "energy": energy,
                "stress": stress,
                "wall_stress": [wall_mu * c[wc] for c in u],
                "pressure_gradient": gradient,
                "dilatation": self._gradient(self._divergence(u_face, None)),
                "viscosity": viscosity,
            }
        return {
            "fluxes": fluxes,
            "force": force,
            "exchange": exchange,
            "rates": rates,
            "particles": particles,
            "momentum": momentum,
            "heat": heat,
            "wall_particles": wall_particles,
            "wall_energy": wall_energy,
            "wall_back": back,
        }

    def _scale(self, unknowns: np.ndarray) -> dict[str, Any]:
        """The scales of the residuals: the largest density, ``n T`` and pressure.

        Held fixed through a Newton solve (:meth:`_newton`): scales that moved with the
        unknowns would enter the finite-difference Jacobian.
        """
        densities, _, temperatures = self._unpack(np.asarray(unknowns).reshape(-1))
        nt = {n: float(np.max(densities[n] * temperatures[n])) for n in densities}
        return {
            "density": {n: max(float(np.max(v)), 1.0) for n, v in densities.items()},
            "energy": {n: max(v, 1e-300) for n, v in nt.items()},
            "pressure": max(max(nt.values()) * E, 1e-300),
        }

    def _residual(
        self, unknowns: np.ndarray, previous: np.ndarray | None, dt: float
    ) -> np.ndarray:
        """Backward Euler residual (``previous is None``: the steady equations).

        ``unknowns`` is one state or a batch ``(size, B)``; so is the result.
        """
        densities, velocities, temperatures = self._unpack(unknowns)
        batch = next(iter(densities.values())).shape[1]
        terms = self._terms(densities, velocities, temperatures, dt)
        old = None if previous is None else self._unpack(previous)
        res = np.empty((self.size, self._nvar, batch))
        scales = self._scales if self._scales is not None else self._scale(unknowns)
        pressure_scale = scales["pressure"] / self._length
        for k, s in enumerate(self.species):
            name, m = s.name, s.mass
            n, u, t = densities[name], velocities[name], temperatures[name]
            f = terms["fluxes"][name]
            balance = (
                self._divergence(f["particles"], terms["wall_particles"][name])
                - terms["particles"][name]
            )
            if old is not None:
                balance = balance + (n - old[0][name]) / dt
            res[:, 4 * k] = balance / scales["density"][name]
            mu = f["viscosity"]
            for a in (0, 1):
                term = (
                    self._divergence(f["stress"][a], f["wall_stress"][a])
                    + f["pressure_gradient"][a]
                    - mu / 3.0 * f["dilatation"][a]
                    - terms["force"][name][a]
                    - terms["momentum"][name][a]
                )
                if a == 0 and self._radius is not None:
                    term = term + mu * u[0] / self._radius**2
                if old is not None:
                    term = term + m * (n * u[a] - old[0][name] * old[1][name][a]) / dt
                res[:, 4 * k + 1 + a] = term / pressure_scale
            energy = (
                self._divergence(f["energy"], terms["wall_energy"][name])
                - terms["exchange"][name]
                - terms["heat"][name]
            )
            if old is not None:
                energy = (
                    energy
                    + s.heat_capacity * (n * t - old[0][name] * old[2][name]) / dt
                )
            res[:, 4 * k + 3] = energy / scales["energy"][name]
        if np.ndim(unknowns) == 1:
            return res.ravel()
        return res.reshape(self.size * self._nvar, batch)

    def _jacobian(self, unknowns: np.ndarray, previous, dt) -> sp.csc_matrix:
        """The Jacobian by finite differences: one column per colour and unknown.

        The perturbed states go through the residual as one batch (in chunks of at
        most :attr:`BATCH_SIZE` numbers).
        """
        nvar, n = self._nvar, self.size
        base = self._residual(unknowns, previous, dt).reshape(n, nvar)
        rows_reach, cols_reach = self._reach.row, self._reach.col
        columns = [
            (colour, k)
            for colour in range(int(self._colours.max()) + 1)
            for k in range(nvar)
        ]
        chunk = max(1, self.BATCH_SIZE // unknowns.size)
        old = None if previous is None else previous[:, None]
        rows, cols, vals = [], [], []
        for start in range(0, len(columns), chunk):
            group = columns[start : start + chunk]
            batch = np.repeat(unknowns[:, None], len(group), axis=1)
            steps = []
            for j, (colour, k) in enumerate(group):
                index = np.flatnonzero(self._colours == colour) * nvar + k
                step = 1.0e-7 * np.maximum(1.0, np.abs(unknowns[index]))
                batch[index, j] += step
                steps.append(step)
            change = self._residual(batch, old, dt).reshape(n, nvar, len(group))
            change = change - base[:, :, None]
            for j, (colour, k) in enumerate(group):
                chosen = self._colours == colour
                hit = chosen[cols_reach]
                target, source = rows_reach[hit], cols_reach[hit]
                step = np.zeros(n)
                step[chosen] = steps[j]
                block = change[target, :, j] / step[source][:, None]
                for i in range(nvar):
                    rows.append(target * nvar + i)
                    cols.append(source * nvar + k)
                    vals.append(block[:, i])
        size = n * nvar
        return sp.csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(size, size),
        )

    def _damping(self, u: np.ndarray, delta: np.ndarray, is_log: np.ndarray) -> float:
        """The fraction of a Newton step: logarithms and velocities change by a little."""
        fraction = 1.0
        largest = float(np.max(np.abs(delta[is_log])))
        if largest > MAX_LOG_STEP:
            fraction = MAX_LOG_STEP / largest
        _, _, temperatures = self._unpack(u)
        speed = max(
            float(np.sqrt(2.0 * E * np.max(temperatures[s.name]) / s.mass))
            for s in self.species
        )
        velocity = float(np.max(np.abs(delta[~is_log])))
        if velocity > MAX_VELOCITY_STEP * speed:
            fraction = min(fraction, MAX_VELOCITY_STEP * speed / velocity)
        return fraction

    @profile("wellbalanced.NeutralFluid2D.newton", tags=_PROFILE_TAGS)
    def _newton(self, unknowns, previous, dt, max_iterations: int | None = None):
        """Solve the (backward Euler or steady) equations; return the solution and success.

        Newton's method with the finite-difference Jacobian, kept while the steps
        shrink fast (as :class:`~wellbalanced.spatial.Fluid2D`'s), the residuals'
        scales held at those of the start.
        """
        if max_iterations is None:
            max_iterations = 40 if previous is None else 15
        u = unknowns.copy()
        is_log = np.tile(self._is_log, self.size)
        factor = None
        last = np.inf
        self._scales = self._scale(u)
        try:
            for iteration in range(1, max_iterations + 1):
                self._iterations = iteration
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    residual = self._residual(u, previous, dt)
                    if not np.all(np.isfinite(residual)):
                        return u, False
                    if factor is None:
                        factor = _factorized(self._jacobian(u, previous, dt))
                        if factor is None:
                            return u, False
                        last = np.inf
                    delta = factor.solve(-residual)
                if not np.all(np.isfinite(delta)):
                    return u, False
                size = float(
                    np.max(np.abs(delta) / np.where(is_log, 1.0, self._velocity_scale))
                )
                fraction = self._damping(u, delta, is_log)
                if fraction < 1.0 or size > 0.25 * last:
                    factor = None
                last = size
                u = u + fraction * delta
                if size < NEWTON_TOLERANCE:
                    return u, True
            return u, False
        finally:
            self._scales = None

    # --- running ---------------------------------------------------------------------

    def step(self, unknowns: np.ndarray, dt: float) -> tuple[np.ndarray, bool]:
        """One backward-Euler step of ``dt`` seconds; return the new unknowns and success."""
        return self._newton(unknowns, unknowns, dt)

    def advance(self, unknowns: np.ndarray, duration: float, dt: float) -> np.ndarray:
        """Step ``duration`` seconds with steps of ``dt``, halving a failed one."""
        time = 0.0
        u = unknowns
        while time < duration * (1.0 - 1e-12):
            current = min(dt, duration - time)
            new, ok = self.step(u, current)
            if ok:
                u, time = new, time + current
            else:
                dt = 0.5 * current
                if dt < 1e-6 * duration:
                    raise RuntimeError("the time step fell below 1e-6 of the duration")
        return u

    @profile("wellbalanced.NeutralFluid2D.steady_state", tags=_PROFILE_TAGS)
    def steady_state(
        self,
        unknowns: np.ndarray,
        *,
        dt: float = 1.0e-6,
        growth: float = 2.0,
        max_dt: float = 10.0,
        max_steps: int = 400,
    ) -> NeutralState:
        """Return the steady state.

        Steps in pseudo-time from ``unknowns`` with a step growing from ``dt`` to
        ``max_dt`` (by ``growth``, its square after an easy step), then solves the
        steady equations from there. A failed step is retried with a quarter of it;
        below a millionth of the first step the state is returned with
        ``converged=False``.
        """
        u = unknowns
        smallest = 1.0e-6 * dt
        for _ in range(max_steps):
            new, ok = self.step(u, dt)
            if not ok:
                dt *= 0.25
                if dt < smallest:
                    return self.state(u, time=np.inf, converged=False)
                continue
            u = new
            if dt >= max_dt:
                break
            dt *= growth**2 if self._iterations <= 4 else growth
        steady, ok = self._newton(u, None, np.inf)
        if ok:
            u = steady
        return self.state(u, time=np.inf, converged=ok)

    def state(
        self, unknowns: np.ndarray, time: float = 0.0, converged: bool = True
    ) -> NeutralState:
        """The :class:`NeutralState` of a vector of unknowns."""
        densities, velocities, temperatures = self._unpack(unknowns)
        rates = self._reactions(densities, velocities, temperatures, 1)[0]
        return NeutralState(
            grid=self.grid,
            region=self.region,
            unknowns=np.array(unknowns, copy=True),
            densities={n: self.on_grid(v) for n, v in densities.items()},
            velocities={
                n: (self.on_grid(v[0]), self.on_grid(v[1]))
                for n, v in velocities.items()
            },
            temperatures={n: self.on_grid(v) for n, v in temperatures.items()},
            reaction_rates={n: self.on_grid(r) for n, r in rates.items()},
            time=time,
            converged=converged,
        )

    # --- balances --------------------------------------------------------------------

    def _state_terms(self, state: NeutralState) -> dict[str, Any]:
        """The terms of a state, every array flattened to one state."""
        densities, velocities, temperatures = self._unpack(state.unknowns)
        terms = self._terms(densities, velocities, temperatures, np.inf)

        def flat(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: flat(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [flat(v) for v in value]
            return np.asarray(value)[:, 0]

        return {k: flat(v) for k, v in terms.items()}

    def balance(self, state: NeutralState) -> dict[str, float]:
        """The volume source minus the net outflow through the walls of every species.

        In particles per second (per metre along ``z`` in a cut): zero at a steady
        state, the rate of change of the species' total in time.
        """
        terms = self._state_terms(state)
        return {
            s.name: float(
                np.sum(terms["particles"][s.name] * self.volumes)
                - np.sum(terms["wall_particles"][s.name] * self._wall_area)
            )
            for s in self.species
        }

    def wall_budget(self, state: NeutralState) -> dict[str, dict[str, float]]:
        """Where every species crosses the walls [1/s], by kind of face.

        ``"walls"``: the net outflow through the walls that are no opening (the
        inlets' faces included, their injection not), ``"openings"``: the net outflow
        through the openings, ``"inlets"``: injected (positive), ``"ions"``: made from
        the ions at the walls (positive; what is made at an opening is scaled by
        ``1 - transparency``). ``walls + openings - inlets`` is the net outflow.
        """
        terms = self._state_terms(state)
        area = self._wall_area
        opening = self._opening >= 0
        plain = ~opening
        keep = 1.0 - self._transparency
        result = {}
        for s in self.species:
            name = s.name
            net = terms["wall_particles"][name] + self._injected[name]
            ions = np.zeros(area.size)
            for ion, products in self.ion_wall_products.items():
                if ion in self._ion_wall_flux and name in products:
                    ions += products[name] * self._ion_wall_flux[ion][:, 0]
            result[name] = {
                "walls": float(np.sum((net * area)[plain])),
                "openings": float(np.sum((net * area)[opening])),
                "inlets": float(np.sum(self._injected[name] * area)),
                "ions": float(np.sum(keep * ions * area)),
            }
        return result

    def energy_balance(self, state: NeutralState) -> dict[str, float]:
        """The terms of the gas's energy balance, integrated over the region [W].

        ``reactions`` (the heat of the reactions and the thermal energy of the
        particles they make and destroy), ``collisions`` (exchanged with the plasma
        and fixed species; between the neutrals it cancels), ``walls`` (the net energy
        flux in, openings and inlets included). Per metre along ``z`` in a cut; their
        sum is the rate of change of the gas's thermal energy.
        """
        terms = self._state_terms(state)
        v = self.volumes
        heat = sum(float(np.sum(terms["heat"][n] * v)) for n in self._by_name)
        exchange = sum(float(np.sum(terms["exchange"][n] * v)) for n in self._by_name)
        walls = sum(
            float(np.sum(terms["wall_energy"][n] * self._wall_area))
            for n in self._by_name
        )
        return {"reactions": E * heat, "collisions": E * exchange, "walls": -E * walls}

    @property
    def wall_faces(self) -> dict[str, np.ndarray]:
        """The walls' faces: their cell (a grid index), axis, outward sign and area."""
        return {
            "cell": self.cells[self._wall_cell],
            "axis": self._wall_axis,
            "sign": self._wall_sign,
            "area": self._wall_area,
        }


# --- coupling to the plasma ------------------------------------------------------------


@dataclass
class CoupledNeutrals:
    """The plasma and the gas at their common steady state.

    Attributes:
        plasma: The plasma's steady state.
        neutrals: The gas's steady state.
        passes: The plasma–gas passes made.
        converged: Whether the passes reached the tolerance.
        history: The largest relative change of a neutral density after each pass.
    """

    plasma: Fluid2DState
    neutrals: NeutralState
    passes: int
    converged: bool
    history: list[float] = field(default_factory=list)


def couple_neutrals(
    plasma: Fluid2D,
    neutrals: NeutralFluid2D,
    plasma_guess: Fluid2DState | np.ndarray,
    neutral_guess: NeutralState | np.ndarray,
    *,
    tolerance: float = 1.0e-3,
    max_passes: int = 30,
    relaxation: float = 1.0,
    mean_free_path: bool = True,
) -> CoupledNeutrals:
    """The plasma and the neutral gas iterated to a common steady state.

    Each pass: the gas's steady state in the plasma's reactions, collisions and ion
    wall fluxes (:meth:`NeutralFluid2D.plasma_from`), then the plasma's steady state in
    that gas (its densities as the plasma's fixed species, its density-weighted mean
    temperature as the plasma's gas temperature; the densities mixed with the last by
    ``relaxation``). ``mean_free_path`` scales the plasma's ion mean free paths with
    the inverse of the mean gas density (depleted gas, freer ions). Stops when the
    largest change of a neutral density over its peak is below ``tolerance``.
    """
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must be in (0, 1]")
    p_state = (
        plasma_guess
        if not isinstance(plasma_guess, np.ndarray)
        else plasma.state(plasma_guess)
    )
    n_unknowns = (
        neutral_guess.unknowns
        if isinstance(neutral_guess, NeutralState)
        else neutral_guess
    )
    names = [s.name for s in neutrals.species]
    reference_paths = dict(plasma.ion_mean_free_path)
    # the gas the plasma's mean free paths are for: its fixed densities at the start
    start_gas = [plasma._fixed[n] for n in names if n in plasma._fixed]
    volumes = plasma.volumes
    reference_gas = sum(float(np.sum(g * volumes) / np.sum(volumes)) for g in start_gas)
    gas: dict[str, np.ndarray] | None = None
    history: list[float] = []
    converged = False
    neutrals.plasma_from(plasma, p_state)
    n_state = neutrals.state(n_unknowns)
    passes = 0
    for passes in range(1, max_passes + 1):  # noqa: B007
        neutrals.plasma_from(plasma, p_state)
        n_state = neutrals.steady_state(n_unknowns)
        if not n_state.converged:
            break
        n_unknowns = n_state.unknowns
        new = {name: n_state.densities[name] for name in names}
        if gas is None:
            change = 1.0
            gas = new
        else:
            change = max(
                float(np.max(np.abs(new[n] - gas[n])) / np.max(new[n])) for n in names
            )
            gas = {n: relaxation * new[n] + (1.0 - relaxation) * gas[n] for n in names}
        history.append(change)
        mean_gas = sum(n_state.mean(gas[n]) for n in names)
        density = np.sum([gas[n] for n in names], axis=0)
        temperature = float(
            n_state.total(n_state.gas_temperature * density) / n_state.total(density)
        )
        plasma.set_gas(gas, temperature)
        if mean_free_path and reference_paths and reference_gas:
            plasma.ion_mean_free_path = {
                ion: path * reference_gas / mean_gas
                for ion, path in reference_paths.items()
            }
        new_plasma = plasma.steady_state(p_state.unknowns, dt=1.0e-7)
        if not new_plasma.converged:
            new_plasma = plasma.steady_state(p_state.unknowns)
        if not new_plasma.converged:
            break
        p_state = new_plasma
        if change < tolerance:
            converged = True
            break
    return CoupledNeutrals(p_state, n_state, passes, converged, history)
