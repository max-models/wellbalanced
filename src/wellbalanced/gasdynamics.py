"""A compressible neutral-gas model for the gas in front of a sputtering target.

The gas of a magnetron is not at rest: the sputtered atoms leave the target with
electronvolts of energy (the *sputter wind*), the ions that hit the target come back as
neutrals, and the plasma ionizes the gas and drags it along. Together these rarefy the
argon in front of the racetrack, which decides the spoke mode number, the merging of
spokes and the homogeneous regime (Hecimovic and von Keudell, *J. Phys. D* 51, 453001,
2018). :class:`GasDynamics` is the neutral-gas side of that: a two-species gas (argon
and the sputtered metal) with one velocity and one temperature, in one dimension
(axial, ``x`` from the target) or two (``y`` azimuthal, periodic).

**Equations.** The conserved variables are the densities of the gas ``rho_g`` and of the
metal ``rho_m``, the momentum and the total energy ``E``; the equation of state is an ideal
gas of the number density ``n = rho_g / m_g + rho_m / m_m``, ``p = n k T``,
``E = p / (gamma - 1) + rho |u|^2 / 2``, with ``gamma = 5/3`` (monatomic):

    d rho_g/dt + div (rho_g u) = - div J_g + S_g,        J_g = - J_m,
    d rho_m/dt + div (rho_m u) = - div J_m + S_m,        J_m = - rho D grad Y,
    d (rho u)/dt + div (rho u u + p) = S_u,
    dE/dt + div ((E + p) u) = - div (J_m (h_m - h_g)) + div (kappa grad T) + S_E,

with ``Y = rho_m / rho`` the metal mass fraction, ``h_s = gamma k T / ((gamma - 1) m_s)``
the enthalpy per mass of a species (the diffusion carries it) and ``D`` the binary diffusion
coefficient. There is no viscosity and, unless ``thermal_conductivity`` is given, no heat
conduction: the transport that remains is the diffusion of the species. Without
``thermal_conductivity`` the heat a wall exchanges stays in the cell next to it (it reaches
the bulk only by compression), so set it (argon: of the order of 0.02 W/(m K), a
PLACEHOLDER to check) when a wall temperature matters. **This is the
continuum limit.** At 1 Pa the mean free path of argon is a few millimetres, against
centimetres for the ionization region, so the Knudsen number is of order 0.1 to 1 and a
kinetic (DSMC) check of any result is due.

**Sources, all plain inputs** (the model knows nothing about the plasma):

- *The sputter wind*, on a wall boundary: ``metal_flux`` atoms per m^2 and s leave the
  wall into the gas with the mean energy ``metal_energy`` (an effusive flux of the
  temperature ``E / 2k``: a flux-weighted half Maxwellian, with ``2kT`` per atom and the
  normal momentum ``m sqrt(pi k T / 2m)`` per atom).
- *The return of the ions*: ``gas_return_flux`` argon atoms per m^2 and s leave the wall
  at ``gas_return_temperature`` (the target temperature by default).
- *Sticking* of the metal at the wall (the free-molecular flux ``n v / 4`` times
  ``metal_sticking``) and the *heat exchange* with a wall of the temperature
  ``temperature``: the free-molecular energy flux ``alpha (n v / 4) 2k (T - T_w)`` with
  the accommodation coefficient ``alpha``.
- *Volume sources* (:class:`VolumeSource`: atoms created at a temperature and a velocity),
  *sinks* (``model.sinks[species]``, atoms removed per m^3 and s, or
  ``model.sink_frequencies[species]``, a loss frequency in 1/s, e.g. the ionization of
  the gas ``n_e k_iz``: the atoms leave with their own velocity and thermal energy) and *exchange with the plasma ions* (:class:`IonExchange`: momentum
  ``rho nu (u_i - u)`` and the frictional and thermal heating). Arrays of the grid's
  shape, or numbers, and they may be changed between steps: the interface to a plasma
  model that evaluates them from its own state.
- *Cooling by unresolved walls* (:class:`ThermalRelaxation`): a loss frequency of the heat.
  The energy of the sputtered atoms (several eV per atom, against 0.04 eV for the gas) is
  the largest heat source of the gas of a magnetron, and without a way to leave it the
  gas heats until it blows out of the domain.

**Boundaries**, per end (``"left"``, ``"right"``): :meth:`Boundary.wall` (reflecting, with
the sources above), :meth:`Boundary.inflow` (a reservoir of the fill gas at a pressure and
temperature, free of metal: the refill), :meth:`Boundary.outflow` (zero gradient) and
:meth:`Boundary.periodic`. The azimuthal direction is periodic.

**Numerics.** Finite volumes, MUSCL reconstruction of the primitive variables with a
slope limiter, the HLLC flux (the metal and the gas are carried as passive fractions of
it), SSP-RK3. The diffusion and the heat conduction are applied after each step by
backward Euler (tridiagonal, one dimension after the other), conservative, so the step is
limited by the acoustic CFL only. The sum of every conserved quantity changes only by what
the boundaries and the sources bring in, to rounding: :attr:`GasDynamics.budget` records
those, which the tests use.

**Data.** Every number that describes a material (the mean energy of the sputtered
atoms, the diffusion coefficient, the accommodation, the sticking coefficient) is an input,
and the defaults below are **PLACEHOLDERS** of the right order, to be replaced by data:
the energy distribution of the sputtered atoms from SRIM or the Thompson distribution with
the binding energy of the target (the mean is several eV, with a tail to the incident
energy), the diffusion coefficient from :func:`binary_diffusion` with measured collision
diameters or a collision integral, the accommodation of argon on a metal surface from
molecular-beam data.

Units: SI throughout (kg, m, s, Pa, K); the mean energy of the sputtered atoms in eV.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr
from plasmacoll.constants import ATOMIC_MASS, BOLTZMANN, ELEMENTARY_CHARGE

from wellbalanced.datasets import variable

__all__ = [
    "ARGON_MASS",
    "Boundary",
    "GasDynamics",
    "GasState",
    "IonExchange",
    "ThermalRelaxation",
    "VolumeSource",
    "binary_diffusion",
]

#: The mass of an argon atom (kg).
ARGON_MASS = 39.948 * ATOMIC_MASS
#: Floor of a density (kg/m^3), so that an empty metal species stays finite.
DENSITY_FLOOR = 1.0e-300
#: The names of the conserved quantities of :attr:`GasDynamics.budget`.
QUANTITIES = ("gas", "metal", "momentum_x", "momentum_y", "energy")

_SIDES = ("left", "right")
_KINDS = ("wall", "inflow", "outflow", "periodic")


def binary_diffusion(
    mass_a: float,
    mass_b: float,
    diameter: float,
    temperature: float | np.ndarray,
    pressure: float | np.ndarray,
) -> float | np.ndarray:
    """The binary diffusion coefficient of hard spheres (Chapman-Cowling, first approximation).

    ``D = 3 / (16 n pi d^2) sqrt(2 pi k T / mu)`` with the reduced mass ``mu`` and the
    collision diameter ``d`` (the mean of the two atoms' diameters); it scales as
    ``T^(3/2) / p``. For argon on argon it is the hard-sphere self-diffusion
    coefficient ``3 / (8 n d^2) sqrt(k T / pi m)``. A hard-sphere estimate: the
    diameters of real atoms are temperature dependent, so this is good to tens of percent.

    Args:
        mass_a: Mass of the first atom (kg).
        mass_b: Mass of the second atom (kg).
        diameter: The collision diameter in m.
        temperature: The gas temperature in K.
        pressure: The total pressure in Pa.
    """
    reduced = mass_a * mass_b / (mass_a + mass_b)
    density = pressure / (BOLTZMANN * temperature)
    return (
        3.0
        / (16.0 * density * np.pi * diameter**2)
        * np.sqrt(2.0 * np.pi * BOLTZMANN * temperature / reduced)
    )


@dataclass
class Boundary:
    """The condition at one end of the axial direction.

    Use the constructors: :meth:`wall`, :meth:`inflow`, :meth:`outflow`, :meth:`periodic`.
    The fields may be changed between steps (the flux of a wall follows the discharge
    current of the plasma model); fluxes are numbers or arrays along ``y``.

    Attributes:
        kind: ``"wall"``, ``"inflow"``, ``"outflow"`` or ``"periodic"``.
        temperature: Wall temperature in K (``None``: no heat exchange); for an inflow
            the temperature of the fill gas.
        accommodation: The accommodation coefficient ``alpha`` of the wall heat exchange.
        gas_return_flux: Argon atoms per m^2 and s the wall emits (the neutralized ions).
        gas_return_temperature: Their temperature in K (``None``: the wall temperature).
        metal_flux: Metal atoms per m^2 and s the wall emits (the sputtered atoms).
        metal_energy: Their mean energy in eV (an effusive flux of the temperature
            ``E / 2k``).
        metal_sticking: The probability that a metal atom that reaches the wall sticks.
        pressure: The pressure of the fill gas of an inflow in Pa.
    """

    kind: str
    temperature: float | None = None
    accommodation: float = 1.0
    gas_return_flux: float | np.ndarray = 0.0
    gas_return_temperature: float | None = None
    metal_flux: float | np.ndarray = 0.0
    metal_energy: float = 5.0
    metal_sticking: float = 0.0
    pressure: float | None = None

    def __post_init__(self) -> None:
        """Validate the kind."""
        if self.kind not in _KINDS:
            raise ValueError(f"the kind of a boundary is one of {_KINDS}")

    @classmethod
    def wall(
        cls,
        temperature: float | None = None,
        *,
        accommodation: float = 1.0,
        gas_return_flux: float | np.ndarray = 0.0,
        gas_return_temperature: float | None = None,
        metal_flux: float | np.ndarray = 0.0,
        metal_energy: float = 5.0,
        metal_sticking: float = 0.0,
    ) -> Boundary:
        """A reflecting wall, optionally at a temperature, emitting and absorbing atoms.

        The default ``metal_energy`` (5 eV) and ``accommodation`` (1) are PLACEHOLDERS.
        """
        return cls(
            "wall",
            temperature,
            accommodation,
            gas_return_flux,
            gas_return_temperature,
            metal_flux,
            metal_energy,
            metal_sticking,
        )

    @classmethod
    def inflow(cls, pressure: float, temperature: float) -> Boundary:
        """A reservoir of the fill gas (no metal) at a pressure and a temperature."""
        return cls("inflow", temperature, pressure=pressure)

    @classmethod
    def outflow(cls) -> Boundary:
        """Zero gradient: waves leave, nothing comes in."""
        return cls("outflow")

    @classmethod
    def periodic(cls) -> Boundary:
        """Periodic: the other end is the same boundary."""
        return cls("periodic")


@dataclass
class VolumeSource:
    """Atoms created in the volume, at a temperature and a velocity.

    Attributes:
        species: ``"gas"`` or ``"metal"``.
        rate: Atoms per m^3 and s: a number or an array of the grid's shape.
        temperature: The temperature in K they are created at.
        velocity: Their velocity ``(u, v)`` in m/s.
    """

    species: str
    rate: float | np.ndarray
    temperature: float = 300.0
    velocity: tuple[float, float] = (0.0, 0.0)


@dataclass
class IonExchange:
    """Elastic and charge-exchange collisions of a neutral species with the plasma ions.

    Per volume the neutrals gain the momentum ``rho_s nu (u_i - u)`` and the energy
    ``u . S + rho_s nu |u_i - u|^2 / 2 + (3/2) n_s k nu (T_i - T)``: the work of the
    friction, half of its dissipation and the exchange of heat at the same rate (the
    equal-mass estimate; a PLACEHOLDER closure, to be replaced by the energy exchange
    rate of the actual collision).

    Attributes:
        species: ``"gas"`` or ``"metal"``.
        frequency: The collision frequency ``nu`` of a neutral with the ions, 1/s
            (``n_i sigma v``); a number or an array of the grid's shape.
        ion_velocity: The ions' mean velocity ``(u_i, v_i)``, m/s (numbers or arrays).
        ion_temperature: The ion temperature in K.
    """

    species: str
    frequency: float | np.ndarray
    ion_velocity: tuple[float | np.ndarray, float | np.ndarray] = (0.0, 0.0)
    ion_temperature: float | np.ndarray = 300.0


@dataclass
class ThermalRelaxation:
    """Cooling of the gas by walls the model does not resolve.

    A 1D model has no side walls, yet at 1 Pa the atoms reach them in microseconds and
    leave their heat there: free-molecular exchange removes heat far faster than the
    continuum conduction across the region. The gas loses the energy per volume
    ``(3/2) n k nu (T - T_w)``. For a channel of half-width ``R`` and accommodation
    ``alpha`` the free-molecular wall exchange gives ``nu = 2 alpha v / (3 R)`` with the mean
    speed ``v`` of the atoms: of the order of 10^4 1/s for argon at 1 cm. A PLACEHOLDER
    closure, to be replaced by the geometry of the device.

    Attributes:
        frequency: The relaxation frequency ``nu`` in 1/s: a number or an array.
        temperature: The temperature ``T_w`` of the walls in K.
    """

    frequency: float | np.ndarray
    temperature: float = 300.0


@dataclass(frozen=True)
class GasState:
    """The gas on the grid: the primitive variables at one time.

    Attributes:
        time: The time in s.
        x: Cell centres along the axis, m.
        y: Cell centres across, m.
        gas_density: Argon mass density, kg/m^3, shape ``(nx, ny)``.
        metal_density: Metal mass density, kg/m^3.
        u: Axial velocity, m/s.
        v: Azimuthal velocity, m/s.
        pressure: Pressure, Pa.
        temperature: Temperature, K.
        gas_number: Argon number density, m^-3.
        metal_number: Metal number density, m^-3.
    """

    time: float
    x: np.ndarray
    y: np.ndarray
    gas_density: np.ndarray
    metal_density: np.ndarray
    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    gas_number: np.ndarray
    metal_number: np.ndarray

    def to_dataset(self) -> xr.Dataset:
        """The state as a labelled dataset: ``x`` (and ``y`` when the grid has more than one cell)."""
        two_d = self.y.size > 1
        dims = ("x", "y") if two_d else ("x",)

        def values(array: np.ndarray) -> np.ndarray:
            return array if two_d else array[:, 0]

        coords: dict[str, xr.Variable] = {
            "x": variable("x", self.x, "axial position", "m")
        }
        if two_d:
            coords["y"] = variable("y", self.y, "azimuthal position", "m")
        data = {
            "gas_number": variable(
                dims, values(self.gas_number), "argon number density", "m^-3"
            ),
            "metal_number": variable(
                dims, values(self.metal_number), "metal number density", "m^-3"
            ),
            "u": variable(dims, values(self.u), "axial velocity", "m/s"),
            "v": variable(dims, values(self.v), "azimuthal velocity", "m/s"),
            "pressure": variable(dims, values(self.pressure), "pressure", "Pa"),
            "temperature": variable(dims, values(self.temperature), "temperature", "K"),
        }
        return xr.Dataset(data, coords, attrs={"time": self.time})


# -- numerics ---------------------------------------------------------------------------


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b > 0.0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def _vanleer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    product = a * b
    return np.where(
        product > 0.0, 2.0 * product / np.where(product > 0, a + b, 1.0), 0.0
    )


def _mc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    central = 0.5 * (a + b)
    limit = 2.0 * np.where(np.abs(a) < np.abs(b), a, b)
    return np.where(
        a * b > 0.0, np.sign(central) * np.minimum(np.abs(central), np.abs(limit)), 0.0
    )


def _central(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (a + b)


LIMITERS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "minmod": _minmod,
    "vanleer": _vanleer,
    "mc": _mc,
    "none": _central,
}


def _hllc(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    """The HLLC flux in the first direction between two primitive states.

    The states and the flux are ordered ``(gas, metal, u, v, p)``: the flux is that of the
    mass of the gas, the mass of the metal, the momentum along the face's normal, the
    tangential momentum and the energy.
    """
    rg_l, rm_l, u_l, v_l, p_l = left
    rg_r, rm_r, u_r, v_r, p_r = right
    rho_l, rho_r = rg_l + rm_l, rg_r + rm_r
    e_l = p_l / (gamma - 1.0) + 0.5 * rho_l * (u_l**2 + v_l**2)
    e_r = p_r / (gamma - 1.0) + 0.5 * rho_r * (u_r**2 + v_r**2)
    c_l, c_r = np.sqrt(gamma * p_l / rho_l), np.sqrt(gamma * p_r / rho_r)
    s_l = np.minimum(u_l - c_l, u_r - c_r)
    s_r = np.maximum(u_l + c_l, u_r + c_r)
    flux_l = np.array(
        [
            rg_l * u_l,
            rm_l * u_l,
            rho_l * u_l**2 + p_l,
            rho_l * u_l * v_l,
            (e_l + p_l) * u_l,
        ]
    )
    flux_r = np.array(
        [
            rg_r * u_r,
            rm_r * u_r,
            rho_r * u_r**2 + p_r,
            rho_r * u_r * v_r,
            (e_r + p_r) * u_r,
        ]
    )
    a_l, a_r = rho_l * (s_l - u_l), rho_r * (s_r - u_r)
    s_star = (p_r - p_l + a_l * u_l - a_r * u_r) / (a_l - a_r)

    def star(rg, rm, rho, u, v, p, e, s):
        factor = (s - u) / (s - s_star)
        energy = (
            rho * factor * (e / rho + (s_star - u) * (s_star + p / (rho * (s - u))))
        )
        return np.array(
            [rg * factor, rm * factor, rho * factor * s_star, rho * factor * v, energy]
        )

    cons_l = np.array([rg_l, rm_l, rho_l * u_l, rho_l * v_l, e_l])
    cons_r = np.array([rg_r, rm_r, rho_r * u_r, rho_r * v_r, e_r])
    star_l = flux_l + s_l * (star(rg_l, rm_l, rho_l, u_l, v_l, p_l, e_l, s_l) - cons_l)
    star_r = flux_r + s_r * (star(rg_r, rm_r, rho_r, u_r, v_r, p_r, e_r, s_r) - cons_r)
    return np.where(
        s_l >= 0.0,
        flux_l,
        np.where(s_star >= 0.0, star_l, np.where(s_r > 0.0, star_r, flux_r)),
    )


def _thomas(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve tridiagonal systems along axis 0, all columns at once.

    Row ``i`` is ``lower[i] x[i-1] + diag[i] x[i] + upper[i] x[i+1] = rhs[i]``
    (``lower[0]`` and ``upper[-1]`` are not used).
    """
    n = diag.shape[0]
    c = np.empty_like(diag)
    d = np.empty_like(diag)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denominator = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denominator
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denominator
    x = np.empty_like(diag)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _cyclic_thomas(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve cyclic tridiagonal systems along axis 0 (Sherman-Morrison).

    As :func:`_thomas`, with the corners ``lower[0]`` coupling row 0 to ``x[-1]`` and
    ``upper[-1]`` coupling row ``n - 1`` to ``x[0]``.
    """
    n = diag.shape[0]
    if n == 1:
        return rhs / (diag + lower + upper)
    beta, alpha = lower[0], upper[-1]
    gamma = -diag[0]
    modified = diag.copy()
    modified[0] = diag[0] - gamma
    modified[-1] = diag[-1] - alpha * beta / gamma
    y = _thomas(lower, modified, upper, rhs)
    u = np.zeros_like(rhs)
    u[0] = gamma
    u[-1] = alpha
    z = _thomas(lower, modified, upper, u)
    factor = (y[0] + beta * y[-1] / gamma) / (1.0 + z[0] + beta * z[-1] / gamma)
    return y - factor * z


def _diffuse_axis(
    q: np.ndarray,
    w: np.ndarray,
    a: np.ndarray,
    dt: float,
    h: float,
    ends: tuple[str, str],
    values: tuple[np.ndarray | float, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray]:
    """Implicit conservative diffusion ``dq/dt = div(a grad(q / w))`` along axis 0.

    ``ends`` are ``"flux"`` (no flux), ``"dirichlet"`` (the value of ``q / w`` given at the
    face, half a cell away) or ``"periodic"`` for both. Returns the new ``q`` and the flux
    at the ``n + 1`` faces (positive along the axis).
    """
    n = q.shape[0]
    periodic = ends[0] == "periodic"
    coupling = np.zeros((n + 1,) + q.shape[1:])
    coupling[1:n] = 0.5 * (a[:-1] + a[1:])
    if periodic:
        coupling[0] = coupling[n] = 0.5 * (a[-1] + a[0])
    scale = dt / (h * h)
    rhs = q.copy()
    boundary = np.zeros_like(q)
    for index, end, value in ((0, ends[0], values[0]), (n - 1, ends[1], values[1])):
        if end == "dirichlet":  # a face half a cell away: twice the coefficient
            boundary[index] = 2.0 * scale * a[index] / w[index]
            rhs[index] = rhs[index] + 2.0 * scale * a[index] * value
    diag = 1.0 + scale * (coupling[1:] + coupling[:-1]) / w + boundary
    lower = np.zeros_like(q)
    upper = np.zeros_like(q)
    lower[1:] = -scale * coupling[1:n] / w[:-1]
    upper[:-1] = -scale * coupling[1:n] / w[1:]
    if periodic:
        lower[0] = -scale * coupling[0] / w[-1]
        upper[-1] = -scale * coupling[n] / w[0]
        new = _cyclic_thomas(lower, diag, upper, rhs)
    else:
        new = _thomas(lower, diag, upper, rhs)
    fraction = new / w
    flux = np.zeros_like(coupling)
    flux[1:n] = -coupling[1:n] * (fraction[1:] - fraction[:-1]) / h
    if periodic:
        flux[0] = flux[n] = -coupling[0] * (fraction[0] - fraction[-1]) / h
    if ends[0] == "dirichlet":
        flux[0] = -a[0] * (fraction[0] - values[0]) / (0.5 * h)
    if ends[1] == "dirichlet":
        flux[n] = -a[-1] * (values[1] - fraction[-1]) / (0.5 * h)
    return new, flux


class GasDynamics:
    """Two-species compressible gas in 1D or 2D; see the module docstring.

    Args:
        length: Length of the domain along ``x`` in m (the target at ``x = 0``).
        cells: Number of cells along ``x``.
        width: Width in ``y`` in m (the depth of a 1D model: totals are per m^2 for 1).
        cells_y: Number of cells along ``y`` (1: one-dimensional).
        gas_mass: The mass of an atom of the gas in kg (argon).
        metal_mass: The mass of a metal atom in kg (titanium by default).
        gamma: The adiabatic index (5/3: monatomic).
        diffusion: The binary diffusion coefficient ``D`` in m^2/s: a number or a function
            of the temperature (K) and the pressure (Pa) arrays (see
            :func:`binary_diffusion`). ``None``: no diffusion.
        thermal_conductivity: ``kappa`` in W/(m K) (number or function of ``T``); 0: none.
        boundaries: The conditions at ``"left"`` and ``"right"`` (default: a wall at the
            left, an outflow at the right).
        limiter: ``"minmod"``, ``"vanleer"`` (default), ``"mc"`` or ``"none"``
            (central slopes: second order on smooth flow, not monotone).
        cfl: The Courant number of the hyperbolic step.
        source_cfl: The largest share of the content of a cell that a wall emission, a
            source or a sink may add or remove in one step.
    """

    def __init__(
        self,
        length: float,
        cells: int,
        *,
        width: float = 1.0,
        cells_y: int = 1,
        gas_mass: float = ARGON_MASS,
        metal_mass: float = 47.867 * ATOMIC_MASS,
        gamma: float = 5.0 / 3.0,
        diffusion: float | Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        thermal_conductivity: float | Callable[[np.ndarray], np.ndarray] = 0.0,
        boundaries: Mapping[str, Boundary] | None = None,
        limiter: str = "vanleer",
        cfl: float = 0.4,
        source_cfl: float = 0.2,
    ) -> None:
        """Set up the grid and the closures; the gas is empty until :meth:`set_state`."""
        if length <= 0.0 or width <= 0.0 or cells < 2 or cells_y < 1:
            raise ValueError("a positive domain, at least 2 cells along x")
        if limiter not in LIMITERS:
            raise ValueError(f"limiter is one of {sorted(LIMITERS)}")
        if not 0.0 < cfl < 1.0 or gamma <= 1.0:
            raise ValueError("0 < cfl < 1 and gamma > 1")
        self.length, self.width = length, width
        self.nx, self.ny = cells, cells_y
        self.dx, self.dy = length / cells, width / cells_y
        self.masses = {"gas": gas_mass, "metal": metal_mass}
        self.gamma = gamma
        self.diffusion = diffusion
        self.thermal_conductivity = thermal_conductivity
        self.limiter = limiter
        self.cfl = cfl
        self.source_cfl = source_cfl
        self.boundaries: dict[str, Boundary] = {
            "left": Boundary.wall(),
            "right": Boundary.outflow(),
        }
        self.boundaries.update(boundaries or {})
        if set(self.boundaries) != set(_SIDES):
            raise ValueError("boundaries are 'left' and 'right'")
        if (self.boundaries["left"].kind == "periodic") != (
            self.boundaries["right"].kind == "periodic"
        ):
            raise ValueError(
                "a periodic boundary needs a periodic boundary at the other end"
            )
        self.sources: list[VolumeSource] = []
        self.sinks: dict[str, float | np.ndarray] = {}
        self.sink_frequencies: dict[str, float | np.ndarray] = {}
        self.exchange: list[IonExchange] = []
        self.relaxation: list[ThermalRelaxation] = []
        self.x = (np.arange(cells) + 0.5) * self.dx
        self.y = (np.arange(cells_y) + 0.5) * self.dy
        self.time = 0.0
        self.steps = 0
        #: The conserved quantities that boundaries and sources have brought in (the
        #: same units as :meth:`totals`).
        self.budget = dict.fromkeys(QUANTITIES, 0.0)
        self._u = np.zeros((5, cells, cells_y))
        self._ready = False

    # -- state ------------------------------------------------------------------------

    def set_state(
        self,
        gas_density: float | np.ndarray,
        metal_density: float | np.ndarray = 0.0,
        u: float | np.ndarray = 0.0,
        v: float | np.ndarray = 0.0,
        pressure: float | np.ndarray = 1.0,
        *,
        time: float = 0.0,
    ) -> None:
        """Set the gas: mass densities (kg/m^3), velocities (m/s) and the pressure (Pa).

        Each is a number, an array of ``(nx,)`` along ``x`` (``(nx, 1)`` is also
        accepted) or of the grid's shape.
        """
        shape = (self.nx, self.ny)

        def full(value: float | np.ndarray) -> np.ndarray:
            array = np.asarray(value, dtype=float)
            if array.ndim == 1:
                array = array[:, None]
            return np.broadcast_to(array, shape).copy()

        rg, rm = full(gas_density), full(metal_density)
        uu, vv, pp = full(u), full(v), full(pressure)
        if np.any(rg < 0.0) or np.any(rm < 0.0) or np.any(pp <= 0.0):
            raise ValueError("densities >= 0 and the pressure > 0")
        rho = rg + rm
        self._u = np.array(
            [
                rg,
                rm,
                rho * uu,
                rho * vv,
                pp / (self.gamma - 1.0) + 0.5 * rho * (uu**2 + vv**2),
            ]
        )
        self.time = time
        self._ready = True

    def set_uniform(
        self, pressure: float, temperature: float, metal_fraction: float = 0.0
    ) -> None:
        """A gas at rest of one pressure, temperature and metal mass fraction."""
        m_g, m_m = self.masses["gas"], self.masses["metal"]
        mean_inverse_mass = (1.0 - metal_fraction) / m_g + metal_fraction / m_m
        rho = pressure / (BOLTZMANN * temperature * mean_inverse_mass)
        self.set_state(
            (1.0 - metal_fraction) * rho, metal_fraction * rho, pressure=pressure
        )

    def _primitive(self, cons: np.ndarray) -> np.ndarray:
        rho = cons[0] + cons[1]
        u, v = cons[2] / rho, cons[3] / rho
        p = (self.gamma - 1.0) * (cons[4] - 0.5 * rho * (u**2 + v**2))
        return np.array([cons[0], cons[1], u, v, p])

    def _number(self, rg: np.ndarray, rm: np.ndarray) -> np.ndarray:
        return rg / self.masses["gas"] + rm / self.masses["metal"]

    def temperature(self, prim: np.ndarray | None = None) -> np.ndarray:
        """The temperature field in K."""
        prim = self._primitive(self._u) if prim is None else prim
        return prim[4] / (BOLTZMANN * self._number(prim[0], prim[1]))

    def state(self) -> GasState:
        """The current gas as a :class:`GasState`."""
        self._check_ready()
        prim = self._primitive(self._u)
        return GasState(
            time=self.time,
            x=self.x,
            y=self.y,
            gas_density=prim[0].copy(),
            metal_density=prim[1].copy(),
            u=prim[2].copy(),
            v=prim[3].copy(),
            pressure=prim[4].copy(),
            temperature=self.temperature(prim),
            gas_number=prim[0] / self.masses["gas"],
            metal_number=prim[1] / self.masses["metal"],
        )

    def totals(self) -> dict[str, float]:
        """The integrals of the conserved quantities over the domain (per unit depth)."""
        self._check_ready()
        volume = self.dx * self.dy
        return {
            name: float(np.sum(self._u[k]) * volume)
            for k, name in enumerate(QUANTITIES)
        }

    def _check_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("set the state first (set_state or set_uniform)")

    # -- the right-hand side ------------------------------------------------------------

    def _ghosts(self, prim: np.ndarray) -> np.ndarray:
        """The primitive variables with two ghost cells on every side."""
        nx, ny = self.nx, self.ny
        ghosts = np.empty((5, nx + 4, ny + 4))
        columns = np.arange(-2, ny + 2) % ny
        ghosts[:, 2:-2, :] = prim[:, :, columns]
        left, right = self.boundaries["left"], self.boundaries["right"]
        if left.kind == "periodic":
            ghosts[:, :2, :] = ghosts[:, nx : nx + 2, :]
            ghosts[:, -2:, :] = ghosts[:, 2:4, :]
            return ghosts
        for boundary, near, far, first, second in (
            (left, 2, 3, 1, 0),
            (right, -3, -4, -2, -1),
        ):
            if boundary.kind == "wall":
                ghosts[:, first] = ghosts[:, near]
                ghosts[:, second] = ghosts[:, far]
                ghosts[2, first] *= -1.0
                ghosts[2, second] *= -1.0
            elif boundary.kind == "outflow":
                ghosts[:, first] = ghosts[:, near]
                ghosts[:, second] = ghosts[:, near]
            else:  # inflow: the reservoir's state, with the velocity of the gas next to it
                assert (
                    boundary.temperature is not None and boundary.pressure is not None
                )
                density = (
                    boundary.pressure
                    * self.masses["gas"]
                    / (BOLTZMANN * boundary.temperature)
                )
                for index in (first, second):
                    ghosts[0, index] = density
                    ghosts[1, index] = 0.0
                    ghosts[2, index] = ghosts[2, near]
                    ghosts[3, index] = 0.0
                    ghosts[4, index] = boundary.pressure
        return ghosts

    def _directional_flux(self, ghosts: np.ndarray, axis: int) -> np.ndarray:
        """The numerical flux through the faces normal to ``axis`` (0: x, 1: y).

        Shape ``(5, nx + 1, ny)`` for x and ``(5, nx, ny + 1)`` for y, ordered
        ``(gas, metal, p_x, p_y, energy)``.
        """
        if axis == 1:
            ghosts = ghosts.swapaxes(1, 2)[[0, 1, 3, 2, 4]]
        limit = LIMITERS[self.limiter]
        slope = limit(
            ghosts[:, 1:-1, 2:-2] - ghosts[:, :-2, 2:-2],
            ghosts[:, 2:, 2:-2] - ghosts[:, 1:-1, 2:-2],
        )
        cells = ghosts[:, 1:-1, 2:-2]
        left = cells[:, :-1] + 0.5 * slope[:, :-1]
        right = cells[:, 1:] - 0.5 * slope[:, 1:]
        flux = _hllc(left, right, self.gamma)
        if axis == 1:
            flux = flux[[0, 1, 3, 2, 4]].swapaxes(1, 2)
        return flux

    def _emissions(self, boundary: Boundary) -> list[tuple[str, np.ndarray, float]]:
        """The species a wall emits: ``(species, flux along y, temperature in K)``."""
        out = []
        k = BOLTZMANN
        for species, flux, temperature in (
            (
                "gas",
                boundary.gas_return_flux,
                boundary.gas_return_temperature
                if boundary.gas_return_temperature is not None
                else boundary.temperature,
            ),
            (
                "metal",
                boundary.metal_flux,
                boundary.metal_energy * ELEMENTARY_CHARGE / (2.0 * k),
            ),
        ):
            rate = np.broadcast_to(np.asarray(flux, dtype=float), (self.ny,))
            if not np.any(rate):
                continue
            if temperature is None:
                raise ValueError("a wall that returns gas needs a temperature")
            out.append((species, rate, float(temperature)))
        return out

    def _wall_flux(self, side: str, boundary: Boundary, prim: np.ndarray) -> np.ndarray:
        """What the sources, the sticking and the heat exchange of a wall add to its face flux.

        In the convention of a flux along ``+x``, shape ``(5, ny)``.
        """
        inward = 1.0 if side == "left" else -1.0
        cell = prim[:, 0, :] if side == "left" else prim[:, -1, :]
        added = np.zeros((5, self.ny))
        k = BOLTZMANN
        wall_t = boundary.temperature
        for species, rate, temperature in self._emissions(boundary):
            mass = self.masses[species]
            sigma = np.sqrt(k * temperature / mass)
            added[0 if species == "gas" else 1] += inward * rate * mass
            added[2] += rate * mass * sigma * np.sqrt(0.5 * np.pi)
            added[4] += inward * rate * 2.0 * k * temperature
        temperature_cell = cell[4] / (k * self._number(cell[0], cell[1]))
        if boundary.metal_sticking:
            mass = self.masses["metal"]
            thermal = np.sqrt(8.0 * k * temperature_cell / (np.pi * mass))
            lost = boundary.metal_sticking * cell[1] / mass * thermal / 4.0
            added[1] -= inward * lost * mass
            added[4] -= inward * lost * 2.0 * k * temperature_cell
        if wall_t is not None and boundary.accommodation:
            heat = np.zeros(self.ny)
            for species, row in (("gas", 0), ("metal", 1)):
                mass = self.masses[species]
                thermal = np.sqrt(8.0 * k * temperature_cell / (np.pi * mass))
                heat += cell[row] / mass * thermal / 4.0
            added[4] -= (
                inward
                * boundary.accommodation
                * heat
                * 2.0
                * k
                * (temperature_cell - wall_t)
            )
        return added

    def _volume_terms(self, cons: np.ndarray, prim: np.ndarray) -> np.ndarray:
        """The sources, the sinks and the exchange with the ions, per m^3 and s."""
        rate = np.zeros_like(cons)
        k = BOLTZMANN
        temperature = prim[4] / (k * self._number(prim[0], prim[1]))
        speed2 = prim[2] ** 2 + prim[3] ** 2
        for source in self.sources:
            mass = self.masses[source.species]
            created = np.broadcast_to(
                np.asarray(source.rate, dtype=float), cons.shape[1:]
            )
            row = 0 if source.species == "gas" else 1
            rate[row] += created * mass
            rate[2] += created * mass * source.velocity[0]
            rate[3] += created * mass * source.velocity[1]
            rate[4] += (
                created
                * mass
                * (
                    0.5 * (source.velocity[0] ** 2 + source.velocity[1] ** 2)
                    + 1.5 * k * source.temperature / mass
                )
            )
        for species in ("gas", "metal"):
            mass = self.masses[species]
            row = 0 if species == "gas" else 1
            lost = np.zeros(cons.shape[1:])
            if species in self.sinks:
                lost = lost + np.asarray(self.sinks[species], dtype=float)
            if species in self.sink_frequencies:
                lost = (
                    lost + np.asarray(self.sink_frequencies[species]) * prim[row] / mass
                )
            if not np.any(lost):
                continue
            rate[row] -= lost * mass
            rate[2] -= lost * mass * prim[2]
            rate[3] -= lost * mass * prim[3]
            rate[4] -= lost * (0.5 * mass * speed2 + 1.5 * k * temperature)
        for exchange in self.exchange:
            row = 0 if exchange.species == "gas" else 1
            mass = self.masses[exchange.species]
            nu = np.broadcast_to(
                np.asarray(exchange.frequency, dtype=float), cons.shape[1:]
            )
            du = np.asarray(exchange.ion_velocity[0], dtype=float) - prim[2]
            dv = np.asarray(exchange.ion_velocity[1], dtype=float) - prim[3]
            density = prim[row]
            momentum_x, momentum_y = density * nu * du, density * nu * dv
            rate[2] += momentum_x
            rate[3] += momentum_y
            rate[4] += (
                prim[2] * momentum_x
                + prim[3] * momentum_y
                + 0.5 * density * nu * (du**2 + dv**2)
                + 1.5
                * density
                / mass
                * k
                * nu
                * (np.asarray(exchange.ion_temperature) - temperature)
            )
        for cooling in self.relaxation:
            nu = np.asarray(cooling.frequency, dtype=float)
            number = self._number(prim[0], prim[1])
            rate[4] -= 1.5 * number * k * nu * (temperature - cooling.temperature)
        return rate

    def _rhs(self, cons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The time derivative of the conserved variables, and the boundary and source input.

        The second result is the rate (per unit depth, summed over the domain) at which
        the boundaries and the sources bring each conserved quantity in.
        """
        prim = self._primitive(cons)
        if np.any(prim[4] <= 0.0) or np.any(cons[:2] < -1.0e-12 * cons[:2].max()):
            raise FloatingPointError("negative pressure or density: reduce the step")
        ghosts = self._ghosts(prim)
        flux_x = self._directional_flux(ghosts, 0)
        left, right = self.boundaries["left"], self.boundaries["right"]
        if left.kind == "wall":
            flux_x[:, 0, :] += self._wall_flux("left", left, prim)
        if right.kind == "wall":
            flux_x[:, -1, :] += self._wall_flux("right", right, prim)
        tendency = -(flux_x[:, 1:, :] - flux_x[:, :-1, :]) / self.dx
        boundary_input = (flux_x[:, 0, :] - flux_x[:, -1, :]).sum(axis=1) * self.dy
        if self.ny > 1:
            flux_y = self._directional_flux(ghosts, 1)
            tendency -= (flux_y[:, :, 1:] - flux_y[:, :, :-1]) / self.dy
        volume = self._volume_terms(cons, prim)
        tendency += volume
        return tendency, boundary_input + volume.sum(axis=(1, 2)) * self.dx * self.dy

    def stable_step(self) -> float:
        """The acoustic CFL time step of the current state."""
        self._check_ready()
        prim = self._primitive(self._u)
        c = np.sqrt(self.gamma * prim[4] / (prim[0] + prim[1]))
        rate = (np.abs(prim[2]) + c) / self.dx
        if self.ny > 1:
            rate = rate + (np.abs(prim[3]) + c) / self.dy
        step = float(self.cfl / rate.max())
        return min(step, self._source_step(prim))

    def _source_step(self, prim: np.ndarray) -> float:
        """The step in which no source adds or removes more than ``source_cfl`` of a cell.

        Strong sources (the sputter wind is of the order of a hundred times the thermal
        flux of the gas) otherwise fill an empty cell in one step.
        """
        cons = self._u
        density = cons[0] + cons[1]
        limit = np.inf
        fraction = self.source_cfl
        for boundary, row in (
            (self.boundaries["left"], 0),
            (self.boundaries["right"], -1),
        ):
            if boundary.kind != "wall":
                continue
            mass_in = np.zeros(self.ny)
            energy_in = np.zeros(self.ny)
            for species, rate, temperature in self._emissions(boundary):
                mass_in += rate * self.masses[species]
                energy_in += rate * 2.0 * BOLTZMANN * temperature
            active = mass_in > 0.0
            if active.any():
                limit = min(
                    limit,
                    fraction
                    * float(np.min(density[row][active] * self.dx / mass_in[active])),
                    fraction
                    * float(np.min(cons[4, row][active] * self.dx / energy_in[active])),
                )
        for source in self.sources:
            mass = self.masses[source.species]
            created = (
                np.broadcast_to(np.asarray(source.rate, dtype=float), density.shape)
                * mass
            )
            if np.any(created > 0.0):
                active = created > 0.0
                limit = min(
                    limit, fraction * float(np.min(density[active] / created[active]))
                )
        for species, removed in self.sinks.items():
            lost = np.broadcast_to(np.asarray(removed, dtype=float), density.shape)
            lost = lost * self.masses[species]
            row = 0 if species == "gas" else 1
            if np.any(lost > 0.0):
                active = lost > 0.0
                limit = min(
                    limit, fraction * float(np.min(cons[row][active] / lost[active]))
                )
        for frequency in (
            *self.sink_frequencies.values(),
            *(e.frequency for e in self.exchange),
            *(c.frequency for c in self.relaxation),
        ):
            top = float(np.max(np.asarray(frequency, dtype=float)))
            if top > 0.0:
                limit = min(limit, fraction / top)
        return limit

    # -- diffusion ----------------------------------------------------------------------

    def _diffuse(self, dt: float) -> np.ndarray:
        """Diffusion of the metal and heat conduction over ``dt``; returns the input rates.

        The rates are those at which the boundaries bring the five conserved quantities
        in (``QUANTITIES``), per unit depth.
        """
        cons = self._u
        rate = np.zeros(5)
        kappa = self.thermal_conductivity
        has_heat = callable(kappa) or kappa != 0.0
        if self.diffusion is None and not has_heat:
            return rate
        left, right = self.boundaries["left"], self.boundaries["right"]
        if left.kind == "periodic":
            x_ends = ("periodic", "periodic")
        else:
            x_ends = tuple(  # type: ignore[assignment]
                "dirichlet" if b.kind == "inflow" else "flux" for b in (left, right)
            )
        k, gamma = BOLTZMANN, self.gamma
        axes = range(2 if self.ny > 1 else 1)

        def view(axis: int) -> Callable[[np.ndarray], np.ndarray]:
            return (lambda f: f.swapaxes(0, 1)) if axis else (lambda f: f)

        if self.diffusion is not None:
            prim = self._primitive(cons)
            temperature = self.temperature(prim)
            d = (
                self.diffusion(temperature, prim[4])
                if callable(self.diffusion)
                else self.diffusion
            )
            a = (prim[0] + prim[1]) * d
            enthalpy = (
                gamma
                / (gamma - 1.0)
                * k
                * temperature
                * (1.0 / self.masses["metal"] - 1.0 / self.masses["gas"])
            )
            for axis in axes:
                to = view(axis)
                step, other = (self.dx, self.dy) if axis == 0 else (self.dy, self.dx)
                ends = x_ends if axis == 0 else ("periodic", "periodic")
                new, flux = _diffuse_axis(
                    to(cons[1]),
                    to(cons[0] + cons[1]),
                    to(a),
                    dt,
                    step,
                    ends,
                    (0.0, 0.0),
                )
                hd = to(enthalpy)
                face = np.empty_like(flux)
                face[1:-1] = 0.5 * (hd[:-1] + hd[1:])
                face[0], face[-1] = hd[0], hd[-1]
                if ends[0] == "periodic":
                    face[0] = face[-1] = 0.5 * (hd[0] + hd[-1])
                energy_flux = flux * face
                delta = to(new) - cons[1]
                cons[1] += delta
                cons[0] -= delta
                cons[4] += dt * to(-(energy_flux[1:] - energy_flux[:-1]) / step)
                inflow = (flux[0] - flux[-1]).sum() * other
                rate[1] += inflow
                rate[0] -= inflow
                rate[4] += (energy_flux[0] - energy_flux[-1]).sum() * other
        if has_heat:
            prim = self._primitive(cons)
            temperature = self.temperature(prim)
            conductivity = kappa(temperature) if callable(kappa) else kappa
            a = np.broadcast_to(
                np.asarray(conductivity, dtype=float), temperature.shape
            )
            w = self._number(prim[0], prim[1]) * k / (gamma - 1.0)
            internal = prim[4] / (gamma - 1.0)
            reservoir = tuple(
                b.temperature if b.kind == "inflow" else 0.0 for b in (left, right)
            )
            for axis in axes:
                to = view(axis)
                step, other = (self.dx, self.dy) if axis == 0 else (self.dy, self.dx)
                ends = x_ends if axis == 0 else ("periodic", "periodic")
                new, flux = _diffuse_axis(
                    to(internal),
                    to(w),
                    to(a),
                    dt,
                    step,
                    ends,
                    reservoir,  # type: ignore[arg-type]
                )
                new = to(new)
                cons[4] += new - internal
                rate[4] += (flux[0] - flux[-1]).sum() * other
                internal = new
        return rate

    # -- stepping -----------------------------------------------------------------------

    def step(self, dt: float | None = None) -> float:
        """Advance one step (the stable step by default); returns the step taken."""
        self._check_ready()
        dt = self.stable_step() if dt is None else dt
        if dt <= 0.0:
            raise ValueError("dt > 0")
        u0 = self._u
        k1, b1 = self._rhs(u0)
        u1 = u0 + dt * k1
        k2, b2 = self._rhs(u1)
        u2 = 0.75 * u0 + 0.25 * (u1 + dt * k2)
        k3, b3 = self._rhs(u2)
        self._u = u0 / 3.0 + 2.0 / 3.0 * (u2 + dt * k3)
        self._accumulate(dt * (b1 + b2 + 4.0 * b3) / 6.0)
        self._accumulate(dt * self._diffuse(dt))
        self.time += dt
        self.steps += 1
        return dt

    def _accumulate(self, amount: np.ndarray) -> None:
        for name, value in zip(QUANTITIES, amount, strict=True):
            self.budget[name] += float(value)

    def run(
        self,
        t_end: float,
        *,
        callback: Callable[[GasDynamics], None] | None = None,
        every: int = 1,
    ) -> GasState:
        """Step until ``time == t_end`` (the last step is shortened); returns the state.

        Args:
            t_end: The end time in s.
            callback: Called with the model after every ``every`` steps.
            every: The interval of the callback in steps.
        """
        self._check_ready()
        while self.time < t_end * (1.0 - 1.0e-12):
            dt = min(self.stable_step(), t_end - self.time)
            self.step(dt)
            if callback is not None and self.steps % every == 0:
                callback(self)
        return self.state()

    def steady_state(
        self,
        *,
        tolerance: float = 1.0e-8,
        max_time: float = 1.0,
        check_every: int = 200,
    ) -> GasState:
        """Step until the density of the gas and of the metal stop changing.

        Stops when the largest relative change of the densities over ``check_every``
        steps, per unit time of ``length / c``, is below ``tolerance``.

        Raises:
            RuntimeError: If ``max_time`` is reached first.
        """
        self._check_ready()
        reference = self._u[:2].copy()
        t_reference = self.time
        while self.time < max_time:
            for _ in range(check_every):
                self.step()
            scale = np.maximum(np.abs(self._u[:2]), 1.0e-6 * np.abs(self._u[:2]).max())
            change = np.max(np.abs(self._u[:2] - reference) / scale)
            sound = float(
                np.sqrt(
                    self.gamma
                    * self._primitive(self._u)[4].max()
                    / (self._u[0] + self._u[1]).min()
                )
            )
            if (
                change / max(self.time - t_reference, 1e-300) * self.length / sound
                < tolerance
            ):
                return self.state()
            reference, t_reference = self._u[:2].copy(), self.time
        warnings.warn("steady_state reached max_time", RuntimeWarning, stacklevel=2)
        return self.state()
