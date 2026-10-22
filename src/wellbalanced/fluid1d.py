"""A 1D fluid model of a low-temperature plasma: drift-diffusion, electron energy, Poisson.

Every fluid species ``s`` of charge ``z_s`` obeys the continuity equation with
the drift-diffusion flux, and the electrons their energy equation, coupled to
Poisson's equation for the potential ``phi``:

    dn_s/dt + d Gamma_s/dx = S_s,           Gamma_s = z_s mu_s n_s E - D_s dn_s/dx,
    dw/dt + d Gamma_w/dx = -Gamma_e E - P_loss + P_ext,       w = 3/2 n_e T_e,
    d^2 phi/dx^2 = -e/eps0 sum_s z_s n_s,                      E = -d phi/dx,

with the energy flux of Hagelaar and Pitchford (2005): the electron flux with
``5/3`` times the mobility and diffusion, applied to ``w``. The sources come
from the same :class:`~wellbalanced.reactions.Reaction` objects as the global
model, with rates at the local electron temperature.

An ion species with ``momentum=True`` obeys its momentum equation instead of
drift-diffusion, so it can accelerate into a sheath (the Bohm criterion):

    d Gamma/dt + d(Gamma^2 / n)/dx = (z e / M) n E - (e T_g / M) dn/dx - (nu + L / n) Gamma,

isothermal at the gas temperature, with the friction ``nu = e / (M mu)`` of
its mobility on the gas, and the loss rate ``L`` (m^-3 s^-1) of the reactions
taking their momentum with them (ions are born at rest). Without inertia it is
drift-diffusion with ``D = mu T_g``; with a large mobility it is the
collisionless fluid of a presheath.

Discretisation: finite volumes on uniform cells; the fluxes between cells are
Scharfetter-Gummel fluxes, exact for a constant field between two cell
centres, so the discrete steady states are those of the equations without an
artificial diffusion. The velocity of a species with momentum is an unknown on
the faces between cells (a staggered grid; the flux is it times the mean
density of the two cells), whose electric and pressure force is
the Scharfetter-Gummel flux times ``e / (M mu)``: without inertia the scheme
is the drift-diffusion scheme exactly; ``Gamma^2 / n`` is taken on the cells. Every equation is implicit (backward Euler, Poisson
included, so the step is not limited by the dielectric relaxation time),
solved by Newton's method in the logarithms of the densities and of ``w``.
:meth:`Fluid1D.steady_state` steps in pseudo-time with a growing step, then
solves the steady equations themselves.

Boundaries, per species and end (``"left"``, ``"right"``), may be changed between
steps (the fluid side of a fluid-PIC coupling): an absorbing wall (the drift
towards it plus half the thermal flux, after Hagelaar et al. 2000, with a
reflection coefficient), a prescribed density, a prescribed flux, or symmetry. The potential at each end is a prescribed value or a prescribed field.
For a species with momentum a wall takes the outflow of its momentum balance
over the half cell next to it (supersonic, as into a sheath), and a
prescribed density flows in or out at the velocity of the last face between
cells.

Not yet: variable transport coefficients, a gas temperature equation, an ion
energy equation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import xarray as xr
from plasmacoll.constants import ELEMENTARY_CHARGE
from scope_profiler import profile

from wellbalanced._profiling import _PROFILE_TAGS
from wellbalanced.datasets import variable
from wellbalanced.global_model import VolumeSource
from wellbalanced.reactions import Reaction
from wellbalanced.species import Species

__all__ = [
    "Boundary",
    "Fluid1D",
    "FluidSpecies",
    "FluidState",
    "PotentialBoundary",
    "scharfetter_gummel",
]

E = ELEMENTARY_CHARGE
EPSILON_0 = 8.8541878128e-12
#: Floor of the electron temperature (eV) where the electron density vanishes.
TE_FLOOR = 1.0e-3
#: Newton: the largest change of a logarithm per iteration, and the tolerance.
MAX_LOG_STEP = 2.0
NEWTON_TOLERANCE = 1.0e-10


def _bernoulli(x: np.ndarray) -> np.ndarray:
    """``x / (exp(x) - 1)``, with its limit 1 at 0."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1.0e-6
    safe = np.where(small, 1.0, x)
    with np.errstate(over="ignore"):
        value = safe / np.expm1(safe)
    return np.where(small, 1.0 - 0.5 * x, value)


def scharfetter_gummel(
    z: int,
    mu: np.ndarray,
    d: np.ndarray,
    efield: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    distance: float | np.ndarray,
) -> np.ndarray:
    """The Scharfetter-Gummel flux from ``left`` to ``right``, a ``distance`` apart.

    ``z`` is the charge number, ``mu`` and ``d`` the mobility and diffusion and
    ``efield`` the field along the line from ``left`` to ``right``; exact for a constant
    field between the two points. Without diffusion the flux is the upwind drift.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        peclet = np.where(
            d > 0.0, z * mu * efield * distance / np.where(d > 0, d, 1.0), 0.0
        )
    drift_only = d <= 0.0
    flux = (d / distance) * (_bernoulli(-peclet) * left - _bernoulli(peclet) * right)
    if np.any(drift_only):
        v = z * mu * efield
        upwind = np.where(v > 0.0, v * left, v * right)
        flux = np.where(drift_only, upwind, flux)
    return flux


@dataclass(frozen=True)
class FluidSpecies:
    """A species carried by the fluid model.

    Attributes:
        species: Name, mass and charge.
        mobility: ``mu`` in m^2/(V s) (charged species).
        diffusion: ``D`` in m^2/s; for a charged species ``None`` takes the
            Einstein relation ``D = mu T`` (``T_e`` for electrons, ``T_g`` for ions).
        momentum: Solve the momentum equation of this ion species (with the
            friction ``e / (M mu)`` of its mobility) instead of drift-diffusion.
    """

    species: Species
    mobility: float = 0.0
    diffusion: float | None = None
    momentum: bool = False

    def __post_init__(self) -> None:
        """Validate the transport coefficients."""
        if self.mobility < 0.0 or (self.diffusion is not None and self.diffusion < 0):
            raise ValueError(
                f"{self.species.name}: mobility and diffusion must be >= 0"
            )
        if self.species.charge == 0 and self.diffusion is None:
            raise ValueError(f"{self.species.name}: a neutral species needs diffusion")
        if self.species.charge != 0 and self.mobility == 0.0:
            raise ValueError(f"{self.species.name}: a charged species needs mobility")
        if self.momentum and (self.species.charge == 0 or self.species.is_electron):
            raise ValueError(f"{self.species.name}: momentum is for ion species")
        if self.momentum and self.diffusion is not None:
            raise ValueError(
                f"{self.species.name}: with momentum the pressure follows from the "
                "gas temperature, not a diffusion coefficient"
            )

    @property
    def name(self) -> str:
        """The species name."""
        return self.species.name


@dataclass
class Boundary:
    """The condition of one species at one end.

    Use the constructors: :meth:`wall`, :meth:`density`, :meth:`flux`,
    :meth:`symmetry`. ``value`` is the density (m^-3) or the flux into the
    domain (m^-2 s^-1); ``temperature`` (eV) the electron temperature of a
    prescribed electron density; ``reflection`` the wall's reflection coefficient.
    """

    kind: str
    value: float = 0.0
    temperature: float | None = None
    reflection: float = 0.0

    @classmethod
    def wall(cls, reflection: float = 0.0) -> Boundary:
        """An absorbing wall: outflow ``(1 - r)/(1 + r) (max(v_drift, 0) + v_th / 2) n``."""
        return cls("wall", reflection=reflection)

    @classmethod
    def density(cls, value: float, temperature: float | None = None) -> Boundary:
        """A prescribed density at the boundary (and, for electrons, temperature)."""
        return cls("density", value, temperature)

    @classmethod
    def flux(cls, value: float, temperature: float | None = None) -> Boundary:
        """A prescribed flux into the domain (and, for electrons, their temperature)."""
        return cls("flux", value, temperature)

    @classmethod
    def symmetry(cls) -> Boundary:
        """No flux."""
        return cls("symmetry")


@dataclass
class PotentialBoundary:
    """The potential at one end: a value (V) or the field ``E`` (V/m) there."""

    kind: str
    value: float = 0.0

    @classmethod
    def potential(cls, value: float) -> PotentialBoundary:
        """A prescribed potential."""
        return cls("potential", value)

    @classmethod
    def field(cls, value: float = 0.0) -> PotentialBoundary:
        """A prescribed field, zero for a symmetry plane."""
        return cls("field", value)


@dataclass
class FluidState:
    """The fields on the cells and the fluxes on the faces.

    Attributes:
        x: Cell centres (m).
        faces: Face positions (m).
        densities: Density of every fluid species on the cells (m^-3).
        electron_temperature: ``T_e`` on the cells (eV).
        potential: ``phi`` on the cells (V).
        fluxes: Flux of every fluid species on the faces (m^-2 s^-1), positive
            along ``x``; ``"energy"`` is the electron energy flux (eV m^-2 s^-1).
        reaction_rates: Volume rate of every reaction on the cells (m^-3 s^-1).
        time: Time (s) of the state; ``inf`` for a steady state.
        converged: Whether the last solve converged.
    """

    x: np.ndarray
    faces: np.ndarray
    densities: dict[str, np.ndarray]
    electron_temperature: np.ndarray
    potential: np.ndarray
    fluxes: dict[str, np.ndarray]
    reaction_rates: dict[str, np.ndarray]
    time: float = 0.0
    converged: bool = True

    def integral(self, values: np.ndarray) -> float:
        """The integral of a cell field over the domain."""
        return float(np.sum(values) * (self.faces[1] - self.faces[0]))

    def velocity(self, name: str) -> np.ndarray:
        """The mean velocity of a species on the faces (m/s): its flux over the density there.

        The density on a face between cells is the mean of the two; on an end
        face, that of the cell next to it.
        """
        n = self.densities[name]
        face = np.concatenate([n[:1], 0.5 * (n[:-1] + n[1:]), n[-1:]])
        return self.fluxes[name] / face

    def to_dataset(self) -> xr.Dataset:
        """The state as a labelled dataset.

        Fields on the cells ``x``, fluxes on the faces ``x_face``, per ``species`` and
        ``reaction``.
        """
        species = list(self.densities)
        reactions = list(self.reaction_rates)
        data = {
            "density": variable(
                ("species", "x"),
                [self.densities[s] for s in species],
                "density",
                "m^-3",
            ),
            "electron_temperature": variable(
                "x", self.electron_temperature, "electron temperature", "eV"
            ),
            "potential": variable("x", self.potential, "potential", "V"),
            "flux": variable(
                ("species", "x_face"),
                [self.fluxes[s] for s in species],
                "flux along x",
                "m^-2 s^-1",
            ),
            "converged": variable((), self.converged, "converged"),
        }
        if "energy" in self.fluxes:
            data["energy_flux"] = variable(
                "x_face", self.fluxes["energy"], "electron energy flux", "eV m^-2 s^-1"
            )
        if reactions:
            data["reaction_rate"] = variable(
                ("reaction", "x"),
                [
                    np.broadcast_to(self.reaction_rates[r], self.x.shape)
                    for r in reactions
                ],
                "reaction rate",
                "m^-3 s^-1",
            )
        coords = {
            "x": ("x", self.x, {"long_name": "x", "units": "m"}),
            "x_face": ("x_face", self.faces, {"long_name": "x (faces)", "units": "m"}),
            "species": ("species", species, {"long_name": "species"}),
        }
        if reactions:
            coords["reaction"] = ("reaction", reactions, {"long_name": "reaction"})
        return xr.Dataset(data, coords=coords, attrs={"time": self.time})


@dataclass
class Fluid1D:
    """A plasma between ``x = 0`` and ``length`` on ``cells`` cells; see the module docstring.

    Attributes:
        species: The fluid species, the electrons (``"e"``) included.
        reactions: The volume reactions; reactants that are not fluid species
            must be in ``fixed``.
        length: The domain length (m).
        cells: The number of cells.
        fixed: Densities of species not carried by the model, e.g. the gas
            (m^-3, a number or an array on the cells).
        gas_temperature: ``T_g`` in eV.
        electron_temperature: ``None`` solves the electron energy equation;
            a number holds ``T_e`` fixed (isothermal electrons).
        boundaries: ``{"left": {species: Boundary}, "right": {...}}``; a
            species missing at an end has a wall there.
        potential: ``{"left": PotentialBoundary, "right": ...}``; grounded if missing.
        heating: Power to the electrons (W/m^3), a number or an array on the cells.
        sources: External volume sources ``{species: rate}`` (m^-3 s^-1), numbers
            or arrays on the cells.
        volume_sources: External sources as for the global model, e.g. a beam's
            (:meth:`wellbalanced.neutralizer.Neutralizer.beam_sources`): rates per
            unit density of fluid or fixed species, numbers or arrays on the
            cells, with the energy they give the electrons.
    """

    species: Sequence[FluidSpecies]
    reactions: Sequence[Reaction]
    length: float
    cells: int
    fixed: Mapping[str, float | np.ndarray] = field(default_factory=dict)
    gas_temperature: float = 0.025
    electron_temperature: float | None = None
    boundaries: dict[str, dict[str, Boundary]] = field(default_factory=dict)
    potential: dict[str, PotentialBoundary] = field(default_factory=dict)
    heating: float | np.ndarray = 0.0
    sources: Mapping[str, float | np.ndarray] = field(default_factory=dict)
    volume_sources: Sequence[VolumeSource] = ()

    def __post_init__(self) -> None:
        """Check the species and index the unknowns."""
        if self.length <= 0.0 or self.cells < 3:
            raise ValueError("need length > 0 and at least 3 cells")
        names = [s.name for s in self.species]
        if len(set(names)) != len(names):
            raise ValueError("species names must be unique")
        if "e" not in names:
            raise ValueError('the model needs the electrons, a species named "e"')
        known = set(names) | set(self.fixed)
        for reaction in self.reactions:
            unknown = set(reaction.reactants) | set(reaction.products)
            unknown -= known
            if unknown:
                raise ValueError(f"{reaction.name}: unknown species {sorted(unknown)}")
        for source in self.volume_sources:
            unknown = (set(source.changes) | set(source.per)) - known
            if unknown:
                raise ValueError(f"{source.name}: unknown species {sorted(unknown)}")
            if not callable(source.rate) and np.size(source.rate) not in (
                1,
                self.cells,
            ):
                raise ValueError(f"{source.name}: a rate is a number or one per cell")
        for end in self.boundaries:
            if end not in ("left", "right"):
                raise ValueError(f"boundary ends are 'left' and 'right', not {end!r}")
        self._names = names
        self._energy = self.electron_temperature is None
        #: the species with momentum; their face velocities come after the densities
        #: (and the electron energy), before the potential
        self._momentum = [f.name for f in self.species if f.momentum]
        self._nvar = len(names) + int(self._energy) + len(self._momentum) + 1
        self._velocity_column = {
            name: len(names) + int(self._energy) + k
            for k, name in enumerate(self._momentum)
        }
        self.h = self.length / self.cells
        self.x = (np.arange(self.cells) + 0.5) * self.h
        self.faces = np.arange(self.cells + 1) * self.h
        self._fixed = {
            name: np.broadcast_to(np.asarray(v, dtype=float), (self.cells,))
            for name, v in self.fixed.items()
        }
        self._electron = names.index("e")

    # -- the boundaries -------------------------------------------------------------

    def boundary(self, end: str, name: str) -> Boundary:
        """The condition of a species at an end (a wall unless set)."""
        return self.boundaries.get(end, {}).get(name, Boundary.wall())

    def set_boundary(self, end: str, name: str, boundary: Boundary) -> None:
        """Change the condition of a species at an end, e.g. between steps."""
        if end not in ("left", "right") or name not in self._names:
            raise ValueError(f"no species {name!r} or end {end!r}")
        self.boundaries.setdefault(end, {})[name] = boundary

    def _potential_boundary(self, end: str) -> PotentialBoundary:
        return self.potential.get(end, PotentialBoundary.potential(0.0))

    # -- the unknowns ---------------------------------------------------------------

    def _unpack(
        self, unknowns: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Densities, ``T_e`` and ``phi`` on the cells, and the fluxes with momentum.

        The unknown of a species with momentum is its velocity on the face to the
        right of each cell (the last cell's, an end face, is not used and is
        zero), so it is of the same size whatever the density; the flux is the
        velocity times the mean density of the two cells.
        """
        u = unknowns.reshape(self.cells, self._nvar)
        densities = {name: np.exp(u[:, k]) for k, name in enumerate(self._names)}
        if self._energy:
            w = np.exp(u[:, len(self._names)])
            te = np.maximum(w / (1.5 * densities["e"]), TE_FLOOR)
        else:
            te = np.full(self.cells, float(self.electron_temperature))  # type: ignore[arg-type]
        inner = {
            name: u[:-1, k] * 0.5 * (densities[name][:-1] + densities[name][1:])
            for name, k in self._velocity_column.items()
        }
        return densities, te, u[:, -1], inner

    def pack(
        self,
        densities: Mapping[str, np.ndarray | float],
        electron_temperature: np.ndarray | float = 2.0,
        potential: np.ndarray | float = 0.0,
        fluxes: Mapping[str, np.ndarray | float] | None = None,
    ) -> np.ndarray:
        """The unknowns of fields on the cells (densities by name, ``T_e``, ``phi``).

        ``fluxes`` start the species with momentum: on the ``cells + 1`` faces
        (as :attr:`FluidState.fluxes`), on the ``cells - 1`` faces between
        cells, or a number; zero if missing.
        """
        u = np.zeros((self.cells, self._nvar))
        for k, name in enumerate(self._names):
            u[:, k] = np.log(
                np.maximum(np.broadcast_to(densities[name], (self.cells,)), 1.0)
            )
        if self._energy:
            ne = np.exp(u[:, self._electron])
            u[:, len(self._names)] = np.log(1.5 * ne * np.asarray(electron_temperature))
        for name, k in self._velocity_column.items():
            value = np.asarray((fluxes or {}).get(name, 0.0), dtype=float)
            if value.size == self.cells + 1:
                value = value[1:-1]
            n = np.exp(u[:, self._names.index(name)])
            u[:-1, k] = np.broadcast_to(value, (self.cells - 1,)) / (
                0.5 * (n[:-1] + n[1:])
            )
        u[:, -1] = potential
        return u.ravel()

    # -- fluxes ---------------------------------------------------------------------

    def _coefficients(
        self, fluid: FluidSpecies, te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mobility and diffusion on the faces (Einstein relation if no diffusion)."""
        faces = self.cells + 1
        mu = np.full(faces, fluid.mobility)
        if fluid.diffusion is not None:
            return mu, np.full(faces, fluid.diffusion)
        if fluid.name == "e":
            t_face = np.empty(faces)
            t_face[1:-1] = 0.5 * (te[:-1] + te[1:])
            t_face[0], t_face[-1] = te[0], te[-1]
        else:
            t_face = np.full(faces, self.gas_temperature)
        return mu, mu * t_face

    def _field(self, phi: np.ndarray) -> np.ndarray:
        """``E`` on the faces (V/m), from the cells and the potential boundaries."""
        h = self.h
        efield = np.empty(self.cells + 1)
        efield[1:-1] = -(phi[1:] - phi[:-1]) / h
        for end, face, cell, sign in (("left", 0, 0, 1.0), ("right", -1, -1, -1.0)):
            condition = self._potential_boundary(end)
            if condition.kind == "potential":
                # the boundary face is half a cell from the cell centre
                efield[face] = sign * -(phi[cell] - condition.value) / (0.5 * h)
            else:
                efield[face] = condition.value
        return efield

    def _sg(
        self,
        z: int,
        mu: np.ndarray,
        d: np.ndarray,
        efield: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        distance: float,
    ) -> np.ndarray:
        """Scharfetter-Gummel flux between densities ``left`` and ``right`` a ``distance`` apart."""
        return scharfetter_gummel(z, mu, d, efield, left, right, distance)

    def _boundary_flux(
        self,
        fluid: FluidSpecies,
        end: str,
        n_cell: float,
        te_cell: float,
        mu: float,
        d: float,
        efield: float,
        energy: bool = False,
    ) -> float:
        """Flux along ``x`` at an end face, of the density (or electron energy) ``n_cell``."""
        condition = self.boundary(end, fluid.name)
        outward = 1.0 if end == "right" else -1.0
        z = fluid.species.charge
        scale = 5.0 / 3.0 if energy else 1.0
        if condition.kind == "symmetry":
            return 0.0
        if condition.kind == "flux":
            # value is into the domain
            flux = -outward * condition.value
            if energy:
                t = (
                    condition.temperature
                    if condition.temperature is not None
                    else te_cell
                )
                flux *= 2.0 * t  # mean energy of a flux of Maxwellian particles
            return flux
        if condition.kind == "density":
            value = condition.value
            if energy:
                t = (
                    condition.temperature
                    if condition.temperature is not None
                    else te_cell
                )
                value = 1.5 * condition.value * t
            mu_e, d_e = scale * mu, scale * d
            if end == "left":
                return float(
                    self._sg(
                        z,
                        np.array([mu_e]),
                        np.array([d_e]),
                        np.array([efield]),
                        np.array([value]),
                        np.array([n_cell]),
                        0.5 * self.h,
                    )[0]
                )
            return float(
                self._sg(
                    z,
                    np.array([mu_e]),
                    np.array([d_e]),
                    np.array([efield]),
                    np.array([n_cell]),
                    np.array([value]),
                    0.5 * self.h,
                )[0]
            )
        # an absorbing wall: the drift towards it, if any, plus half the thermal
        # flux (the density next to an absorbing wall is that of the outgoing half)
        r = condition.reflection
        temperature = te_cell if fluid.name == "e" else self.gas_temperature
        thermal = np.sqrt(8.0 * E * temperature / (np.pi * fluid.species.mass))
        drift = max(z * mu * efield * outward, 0.0)  # outward drift speed
        if energy:
            outflow = scale * drift * n_cell + (2.0 / 3.0) * thermal * n_cell
        else:
            outflow = drift * n_cell + 0.5 * thermal * n_cell
        return float(outward * (1.0 - r) / (1.0 + r) * outflow)

    def _momentum_boundary(
        self,
        fluid: FluidSpecies,
        end: str,
        n_cell: float,
        inner: float,
        n_inner: float,
        efield: float,
    ) -> float:
        """Flux at an end face of a species with momentum.

        ``inner`` is the flux on the face between cells next to the end, where
        the density is ``n_inner``, and ``efield`` the field at the end. A wall
        takes the outflow of the momentum balance over the half cell next to it,
        steady and with nothing coming back from the wall (supersonic, as into a
        sheath): ``(2 / h) (Gamma^2 / n - Pi) = (z e / M) n E - nu Gamma``, with
        ``Pi`` the momentum flux arriving from the inner face; without inertia
        that is the drift ``mu z E n``. Half the thermal flux is added, as for
        drift-diffusion.
        """
        condition = self.boundary(end, fluid.name)
        outward = 1.0 if end == "right" else -1.0
        if condition.kind == "symmetry":
            return 0.0
        if condition.kind == "flux":
            return -outward * condition.value
        if condition.kind == "density":
            # upwind, at the inner face's velocity: the held density flows in,
            # the cell's flows out
            velocity = inner / n_inner
            leaving = velocity * outward > 0.0
            return velocity * (n_cell if leaving else condition.value)
        mass = fluid.species.mass
        arriving = max(inner * outward, 0.0) ** 2 / n_inner
        a = 2.0 / (self.h * n_cell)
        b = E / (mass * fluid.mobility)
        c = (
            2.0 * arriving / self.h
            + fluid.species.charge * E * efield * outward * n_cell / mass
        )
        # the positive root of a G^2 + b G - c = 0, none if nothing pushes outwards
        drift = 2.0 * c / (b + np.sqrt(b * b + 4.0 * a * c)) if c > 0.0 else 0.0
        r = condition.reflection
        thermal = np.sqrt(8.0 * E * self.gas_temperature / (np.pi * mass))
        outflow = drift + 0.5 * thermal * n_cell
        return float(outward * (1.0 - r) / (1.0 + r) * outflow)

    def _fluxes(self, densities, te, phi, inner=None):
        """Fluxes of every species and of the electron energy on the faces, and ``E``.

        ``inner`` holds the fluxes of the species with momentum between cells.
        """
        h = self.h
        efield = self._field(phi)
        fluxes = {}
        for fluid in self.species:
            n = densities[fluid.name]
            if fluid.momentum:
                assert inner is not None
                g = inner[fluid.name]
                flux = np.empty(self.cells + 1)
                flux[1:-1] = g
                flux[0] = self._momentum_boundary(
                    fluid, "left", n[0], g[0], 0.5 * (n[0] + n[1]), efield[0]
                )
                flux[-1] = self._momentum_boundary(
                    fluid, "right", n[-1], g[-1], 0.5 * (n[-2] + n[-1]), efield[-1]
                )
                fluxes[fluid.name] = flux
                continue
            mu, d = self._coefficients(fluid, te)
            z = fluid.species.charge
            flux = np.empty(self.cells + 1)
            flux[1:-1] = self._sg(z, mu[1:-1], d[1:-1], efield[1:-1], n[:-1], n[1:], h)
            flux[0] = self._boundary_flux(
                fluid, "left", n[0], te[0], mu[0], d[0], efield[0]
            )
            flux[-1] = self._boundary_flux(
                fluid, "right", n[-1], te[-1], mu[-1], d[-1], efield[-1]
            )
            fluxes[fluid.name] = flux
            if self._energy and fluid.name == "e":
                w = 1.5 * n * te
                energy = np.empty(self.cells + 1)
                energy[1:-1] = self._sg(
                    z, 5 / 3 * mu[1:-1], 5 / 3 * d[1:-1], efield[1:-1], w[:-1], w[1:], h
                )
                energy[0] = self._boundary_flux(
                    fluid, "left", w[0], te[0], mu[0], d[0], efield[0], energy=True
                )
                energy[-1] = self._boundary_flux(
                    fluid,
                    "right",
                    w[-1],
                    te[-1],
                    mu[-1],
                    d[-1],
                    efield[-1],
                    energy=True,
                )
                fluxes["energy"] = energy
        return fluxes, efield

    # -- the equations --------------------------------------------------------------

    def _reactions(self, densities, te):
        """Volume rates of the reactions and sources, the source of every species, the energy loss.

        The energy loss is that of the reactions minus what the volume sources
        give; the sinks are the loss rates of the species with momentum.
        """
        n_all = {**self._fixed, **densities}
        sources = {name: np.zeros(self.cells) for name in self._names}
        for name, value in self.sources.items():
            if name in sources:
                sources[name] = sources[name] + np.broadcast_to(value, (self.cells,))
        loss = np.zeros(self.cells)
        sinks = {name: np.zeros(self.cells) for name in self._momentum}
        rates = {}
        tg = self.gas_temperature
        for reaction in self.reactions:
            rate = reaction.coefficients(te, tg) * np.ones(self.cells)
            for name in reaction.reactants:
                rate = rate * n_all[name]
            rates[reaction.name] = rates.get(reaction.name, 0.0) + rate
            for name, count in reaction.stoichiometry.items():
                if name in sources:
                    sources[name] = sources[name] + count * rate
                if name in sinks and count < 0:
                    sinks[name] = sinks[name] - count * rate
            loss = loss + reaction.electron_loss(te, tg) * rate
        for source, rate in zip(
            self.volume_sources, self._source_rates(n_all, te), strict=True
        ):
            rates[source.name] = rates.get(source.name, 0.0) + rate
            for name, count in source.changes.items():
                if name in sources:
                    sources[name] = sources[name] + count * rate
                if name in sinks and count < 0:
                    sinks[name] = sinks[name] - count * rate
            loss = loss - source.electron_energy * rate
        return rates, sources, loss, sinks

    def _source_rates(self, n_all, te) -> list[np.ndarray]:
        """The volume rate of every volume source on the cells (m^-3 s^-1)."""
        out = []
        for source in self.volume_sources:
            if callable(source.rate):
                coefficient = source.rate(te, self.gas_temperature)  # type: ignore[arg-type]
            else:
                coefficient = source.rate
            rate = np.broadcast_to(np.asarray(coefficient, dtype=float), (self.cells,))
            for name in source.per:
                rate = rate * n_all[name]
            out.append(np.asarray(rate, dtype=float))
        return out

    def _residual(
        self, unknowns: np.ndarray, previous: np.ndarray | None, dt: float
    ) -> np.ndarray:
        """Backward Euler residual (``previous is None``: the steady equations)."""
        densities, te, phi, inner = self._unpack(unknowns)
        fluxes, efield = self._fluxes(densities, te, phi, inner)
        _, sources, loss, sinks = self._reactions(densities, te)
        h = self.h
        res = np.empty((self.cells, self._nvar))
        old = None if previous is None else self._unpack(previous)
        for k, name in enumerate(self._names):
            n = densities[name]
            divergence = (fluxes[name][1:] - fluxes[name][:-1]) / h
            balance = divergence - sources[name]
            if old is not None:
                balance = balance + (n - old[0][name]) / dt
            scale = max(float(np.max(n)), 1.0)
            res[:, k] = balance / scale
        if self._energy:
            n = densities["e"]
            w = 1.5 * n * te
            gamma = fluxes["e"]
            joule = -0.5 * (gamma[1:] * efield[1:] + gamma[:-1] * efield[:-1])
            heating = np.broadcast_to(self.heating, (self.cells,)) / E
            divergence = (fluxes["energy"][1:] - fluxes["energy"][:-1]) / h
            balance = divergence - joule + loss - heating
            if old is not None:
                w_old = 1.5 * old[0]["e"] * old[1]
                balance = balance + (w - w_old) / dt
            res[:, len(self._names)] = balance / max(float(np.max(w)), 1.0)
        for fluid in self.species:
            if fluid.momentum:
                column = self._velocity_column[fluid.name]
                res[:-1, column] = self._momentum_residual(
                    fluid,
                    densities[fluid.name],
                    fluxes[fluid.name],
                    efield,
                    sinks[fluid.name],
                    te,
                    None if old is None else old[3][fluid.name],
                    dt,
                )
                res[-1, column] = unknowns.reshape(self.cells, self._nvar)[-1, column]
        charge = sum(f.species.charge * densities[f.name] for f in self.species)
        laplacian = np.empty(self.cells)
        laplacian[:] = (efield[:-1] - efield[1:]) / h  # d^2 phi/dx^2 = -dE/dx
        res[:, -1] = (laplacian + E * charge / EPSILON_0) * h * h
        return res.ravel()

    def _momentum_residual(self, fluid, n, flux, efield, sink, te, old, dt):
        """The momentum balance on the faces between cells, scaled."""
        h = self.h
        mass = fluid.species.mass
        tg = self.gas_temperature
        inner = flux[1:-1]
        # the electric and pressure force, (e / mu) times the drift-diffusion flux
        sg = self._sg(
            fluid.species.charge,
            np.full(self.cells - 1, fluid.mobility),
            np.full(self.cells - 1, fluid.mobility * tg),
            efield[1:-1],
            n[:-1],
            n[1:],
            h,
        )
        friction = E / (mass * fluid.mobility)
        # Gamma^2 / n on the cells, upwind: from the face the flow comes through
        face = np.concatenate([n[:1], 0.5 * (n[:-1] + n[1:]), n[-1:]])
        carried = flux**2 / face
        rightward = flux[:-1] + flux[1:] > 0.0
        convection = np.where(rightward, carried[:-1], carried[1:])
        rate = sink / n
        loss = 0.5 * (rate[:-1] + rate[1:])
        balance = (
            (convection[1:] - convection[:-1]) / h
            - friction * sg
            + (friction + loss) * inner
        )
        if old is not None:
            balance = balance + (inner - old) / dt
        sound = E * (float(np.mean(te)) + tg) / mass
        scale = max(float(np.max(n)), 1.0) * sound / self.length
        return balance / scale

    def _jacobian(self, unknowns: np.ndarray, previous, dt) -> sp.csc_matrix:
        """The Jacobian by finite differences, three cells apart per evaluation."""
        nvar, cells = self._nvar, self.cells
        base = self._residual(unknowns, previous, dt)
        rows, cols, vals = [], [], []
        for k in range(nvar):
            for colour in range(3):
                perturbed = unknowns.copy()
                index = np.arange(colour, cells, 3) * nvar + k
                step = 1.0e-7 * np.maximum(1.0, np.abs(unknowns[index]))
                perturbed[index] += step
                change = (self._residual(perturbed, previous, dt) - base).reshape(
                    cells, nvar
                )
                for offset in (-1, 0, 1):
                    target = np.arange(colour, cells, 3) + offset
                    ok = (target >= 0) & (target < cells)
                    source = np.arange(colour, cells, 3)[ok]
                    target = target[ok]
                    block = change[target] / step[ok][:, None]
                    for j in range(nvar):
                        rows.append(target * nvar + j)
                        cols.append(source * nvar + k)
                        vals.append(block[:, j])
        size = cells * nvar
        return sp.csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(size, size),
        )

    @profile("wellbalanced.Fluid1D.newton", tags=_PROFILE_TAGS)
    def _newton(self, unknowns, previous, dt, max_iterations: int = 30):
        """Solve the (backward Euler or steady) equations; return the solution and success."""
        u = unknowns.copy()
        log_columns = np.ones(self._nvar, dtype=bool)
        log_columns[-1] = False  # the potential
        log_columns[list(self._velocity_column.values())] = False  # velocities
        is_log = np.tile(log_columns, self.cells)
        linear = np.flatnonzero(~log_columns)
        for _ in range(max_iterations):
            residual = self._residual(u, previous, dt)
            jacobian = self._jacobian(u, previous, dt)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", spla.MatrixRankWarning)
                    delta = np.asarray(spla.spsolve(jacobian, -residual))
            except (RuntimeError, spla.MatrixRankWarning):
                return u, False  # a singular Jacobian: the step fails
            if not np.all(np.isfinite(delta)):
                return u, False
            largest = float(np.max(np.abs(delta[is_log]))) if is_log.any() else 0.0
            if largest > MAX_LOG_STEP:
                delta *= MAX_LOG_STEP / largest
            u = u + delta
            # the potential and the velocities each converge relative to their own size
            columns = u.reshape(self.cells, self._nvar)[:, linear]
            changes = delta.reshape(self.cells, self._nvar)[:, linear]
            scales = 1.0 + np.max(np.abs(columns), axis=0)
            if np.max(np.abs(delta[is_log])) < NEWTON_TOLERANCE and np.all(
                np.max(np.abs(changes), axis=0) < NEWTON_TOLERANCE * scales
            ):
                return u, True
        return u, False

    # -- running --------------------------------------------------------------------

    def step(self, unknowns: np.ndarray, dt: float) -> tuple[np.ndarray, bool]:
        """One backward-Euler step of ``dt`` seconds; return the new unknowns and success."""
        return self._newton(unknowns, unknowns, dt)

    def advance(self, unknowns: np.ndarray, duration: float, dt: float) -> np.ndarray:
        """Step ``duration`` seconds with steps of at most ``dt``, halving failed ones."""
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

    @profile("wellbalanced.Fluid1D.steady_state", tags=_PROFILE_TAGS)
    def steady_state(
        self,
        unknowns: np.ndarray,
        *,
        dt: float = 1.0e-9,
        growth: float = 2.0,
        max_dt: float = 1.0,
        max_steps: int = 400,
    ) -> FluidState:
        """Return the steady state.

        Steps in pseudo-time from ``unknowns`` with a step growing from ``dt`` up
        to ``max_dt``, then solves the steady equations from there. A failed step
        is retried with a quarter of it; below a millionth of the first step the
        search stops and the state is returned with ``converged=False``.
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
            dt *= growth
        steady, ok = self._newton(u, None, np.inf)
        if ok:
            u = steady
        return self.state(u, time=np.inf, converged=ok)

    def state(
        self, unknowns: np.ndarray, time: float = 0.0, converged: bool = True
    ) -> FluidState:
        """The :class:`FluidState` of a vector of unknowns."""
        densities, te, phi, inner = self._unpack(unknowns)
        fluxes, _ = self._fluxes(densities, te, phi, inner)
        rates, _, _, _ = self._reactions(densities, te)
        return FluidState(
            x=self.x.copy(),
            faces=self.faces.copy(),
            densities=densities,
            electron_temperature=te,
            potential=phi.copy(),
            fluxes=fluxes,
            reaction_rates=rates,
            time=time,
            converged=converged,
        )

    def energy_balance(self, state: FluidState) -> dict[str, float]:
        """Return the electron energy terms, integrated over the slab (eV m^-2 s^-1).

        ``heating`` (external), ``field`` (work of the field, ``-Gamma_e E``),
        ``reactions`` (lost in collisions, negative), ``sources`` (given by the
        volume sources) and ``walls`` (the energy flux out through both ends,
        negative); at a steady state they sum to zero.
        """
        if not self._energy:
            raise ValueError("isothermal electrons have no energy balance")
        unknowns = self.pack(
            state.densities, state.electron_temperature, state.potential, state.fluxes
        )
        densities, te, phi, inner = self._unpack(unknowns)
        fluxes, efield = self._fluxes(densities, te, phi, inner)
        _, _, loss, _ = self._reactions(densities, te)
        rates = self._source_rates({**self._fixed, **densities}, te)
        gained = sum(
            (
                s.electron_energy * r
                for s, r in zip(self.volume_sources, rates, strict=True)
            ),
            np.zeros(self.cells),
        )
        gamma = fluxes["e"]
        joule = -0.5 * (gamma[1:] * efield[1:] + gamma[:-1] * efield[:-1])
        heating = np.broadcast_to(self.heating, (self.cells,)) / E
        return {
            "heating": state.integral(heating),
            "field": state.integral(joule),
            "reactions": -state.integral(loss + gained),
            "sources": state.integral(gained),
            "walls": -(fluxes["energy"][-1] - fluxes["energy"][0]),
        }

    def balance(self, state: FluidState) -> dict[str, float]:
        """Return the volume source minus the net outflow of every species.

        In m^-2 s^-1 (per unit area of the slab); zero at a steady state.
        """
        unknowns = self.pack(
            state.densities, state.electron_temperature, state.potential, state.fluxes
        )
        densities, te, _, _ = self._unpack(unknowns)
        _, sources, _, _ = self._reactions(densities, te)
        out = {}
        for name in self._names:
            outflow = state.fluxes[name][-1] - state.fluxes[name][0]
            out[name] = state.integral(sources[name]) - outflow
        return out
