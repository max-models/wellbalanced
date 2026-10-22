"""A global (volume-averaged, 0D) model of a low-temperature plasma.

The model balances, averaged over the volume ``V`` of a chamber:

- the density of every species, from the reactions, the losses to the walls,
  the feed and pumping of gases and external volume sources; the electron
  density follows from quasi-neutrality, ``n_e = sum_s Z_s n_s``;
- the electron energy, ``3/2 n_e T_e``: absorbed power and the energy of
  external electrons in, the energy lost in reactions, in elastic collisions and
  at the walls out;
- optionally the gas temperature, heated by the reactions and cooled by
  conduction to the walls.

Positive ions leave through the walls at the Bohm speed ``u_B = sqrt(e T_e / M)``
(Lieberman and Lichtenberg, *Principles of Plasma Discharges*, ch. 10): through
the unconfined area times an edge-to-centre ratio ``h``, and through the leak
area of magnetic cusps, ``L w`` with the hybrid leak width
``w = 4 sqrt(rho_e rho_i)`` (Bosch and Merlino 1986). Negative ions are held by
the sheath and do not reach the walls. Floating walls draw as many electrons
as ions (plus any net negative charge the sources add); every electron lost
takes ``2 T_e`` and every ion ``T_e / 2 + V_f`` with the floating potential
``V_f = T_e ln sqrt(M / (2 pi m_e))``. Neutral species are lost at the walls
with a probability (recombination, quenching) at a quarter of their mean speed
and returned as products.

All temperatures are in eV, densities in m^-3, rates in m^-3 s^-1 and powers in
W. The steady state is found by integrating the balances in the logarithms of
the densities and temperatures (they stay positive) with an implicit method,
then refining it with a root finder: :meth:`GlobalModel.steady_state`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr
from plasmacoll.constants import BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scope_profiler import profile

from wellbalanced._profiling import _PROFILE_TAGS
from wellbalanced.datasets import variable
from wellbalanced.reactions import Reaction
from wellbalanced.species import Species

__all__ = [
    "Chamber",
    "Cusp",
    "GasTemperature",
    "GlobalModel",
    "GlobalState",
    "VolumeSource",
    "WallReaction",
]

E = ELEMENTARY_CHARGE


@dataclass(frozen=True)
class Cusp:
    """Multicusp magnetic confinement: cusp lines with a field at the wall.

    ``line_length`` (m) is the total length of the cusp lines, ``field`` (T) the
    field at the wall. The ions leave through a strip of the hybrid leak width
    ``w = 4 sqrt(rho_e rho_i)`` along the lines, with ``rho = sqrt(m T) / (e B)``
    the gyroradius at the thermal speed ``sqrt(T / m)``.
    """

    field: float
    line_length: float

    def __post_init__(self) -> None:
        """Validate the cusp."""
        if self.field <= 0.0 or self.line_length < 0.0:
            raise ValueError("a cusp needs field > 0 and line_length >= 0")

    def leak_width(
        self, electron_temperature: float, ion_temperature: float, ion_mass: float
    ) -> float:
        """Return the leak width in m."""
        rho_e = np.sqrt(ELECTRON_MASS * electron_temperature * E) / (E * self.field)
        rho_i = np.sqrt(ion_mass * ion_temperature * E) / (E * self.field)
        return 4.0 * float(np.sqrt(rho_e * rho_i))

    def area(
        self, electron_temperature: float, ion_temperature: float, ion_mass: float
    ) -> float:
        """Return the leak area in m^2."""
        return self.line_length * self.leak_width(
            electron_temperature, ion_temperature, ion_mass
        )


@dataclass(frozen=True)
class Chamber:
    """The volume and walls of the plasma.

    Attributes:
        volume: The volume in m^3.
        wall_area: The total wall area in m^2, seen by the neutrals.
        ion_loss_area: The area in m^2 through which ions leave unconfined;
            the whole wall if ``None``.
        h_factor: The edge-to-centre density ratio at that area.
        cusp: Magnetic cusps through which ions leave as well.
    """

    volume: float
    wall_area: float
    ion_loss_area: float | None = None
    h_factor: float = 1.0
    cusp: Cusp | None = None

    def __post_init__(self) -> None:
        """Validate the chamber."""
        if self.volume <= 0.0 or self.wall_area <= 0.0:
            raise ValueError("a chamber needs volume > 0 and wall_area > 0")
        if not 0.0 < self.h_factor <= 1.0:
            raise ValueError("h_factor must be in (0, 1]")

    def ion_area(
        self, electron_temperature: float, ion_temperature: float, ion_mass: float
    ) -> float:
        """Return the effective area in m^2 through which an ion leaves at ``u_B``."""
        area = self.h_factor * (
            self.wall_area if self.ion_loss_area is None else self.ion_loss_area
        )
        if self.cusp is not None:
            area += self.cusp.area(electron_temperature, ion_temperature, ion_mass)
        return area


@dataclass(frozen=True)
class GasTemperature:
    """The gas temperature as a balance: heating by the reactions against conduction.

    The conduction loss is ``kappa (T_g - T_w) / Lambda^2`` (W/m^3), with the
    thermal conductivity ``kappa`` (W/m/K) and the diffusion length ``Lambda``
    (m) of the chamber. ``heat_capacity`` is the heat capacity per particle in
    units of ``k`` (3/2 for atoms, 5/2 for diatomic molecules), which only sets
    how fast the temperature moves towards its steady state.
    """

    wall_temperature: float
    conductivity: float
    diffusion_length: float
    heat_capacity: float = 2.5

    def __post_init__(self) -> None:
        """Validate the gas model."""
        if min(self.wall_temperature, self.conductivity, self.diffusion_length) <= 0:
            raise ValueError("wall_temperature, conductivity and diffusion_length > 0")


@dataclass(frozen=True)
class VolumeSource:
    """An external volume source, e.g. ionization or stripping by a fast beam.

    Its rate is ``rate * prod n_per`` (m^-3 s^-1): ``rate`` alone, or per unit
    density of the species ``per`` (for a beam of density ``n_b`` and speed
    ``v_b`` ionizing a gas with the cross section ``sigma``,
    ``rate = n_b sigma v_b`` and ``per = ("gas",)``). Each event changes the
    densities by ``changes`` and gives the electrons ``electron_energy`` (eV).
    ``rate`` may be a function of the electron and gas temperatures in eV,
    ``rate(T_e, T_g)``, for a source that follows the plasma, e.g. sputtering
    by the ion flux to a target (``per = ("Ar+",)``, ``rate`` the Bohm flux
    times the yield per volume).
    """

    changes: Mapping[str, int]
    rate: float | Callable[[float, float], float]
    per: Sequence[str] = ()
    electron_energy: float = 0.0
    name: str = "source"

    def __post_init__(self) -> None:
        """Validate the source."""
        if not callable(self.rate) and self.rate < 0.0:
            raise ValueError("a source rate must be >= 0")
        object.__setattr__(self, "changes", dict(self.changes))
        object.__setattr__(self, "per", tuple(self.per))

    def coefficient(self, electron_temperature: float, gas_temperature: float) -> float:
        """Return ``rate`` at the temperatures (eV)."""
        if callable(self.rate):
            return float(self.rate(electron_temperature, gas_temperature))
        return float(self.rate)


@dataclass(frozen=True)
class WallReaction:
    """A neutral species lost at the walls, with a probability per impact.

    It returns ``products`` (``{name: number}``), e.g.
    ``WallReaction("H", 0.1, {"H2": 0.5})`` for atoms recombining to molecules.
    """

    species: str
    probability: float
    products: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the wall reaction."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("a wall probability must be in [0, 1]")
        object.__setattr__(self, "products", dict(self.products))


@dataclass
class GlobalState:
    """A solution of the global model, with every term of its balances.

    Attributes:
        densities: Every species' density in m^-3, the electrons included.
        electron_temperature: ``T_e`` in eV.
        gas_temperature: ``T_g`` in eV.
        reaction_rates: Volume rate of every reaction (m^-3 s^-1), by name.
        wall_losses: Volume rate at which every species reaches the walls
            (m^-3 s^-1), the electrons included.
        power: Electron power terms in W (positive: gained), by channel; at
            steady state they sum to zero.
        particle_balance: ``dn/dt`` of every species (m^-3 s^-1); zero at steady
            state, except for a species held fixed, where it is the rate the
            reactions consume it at (and its feed would have to make up).
        converged: Whether the steady state was reached.
    """

    densities: dict[str, float]
    electron_temperature: float
    gas_temperature: float
    reaction_rates: dict[str, float]
    wall_losses: dict[str, float]
    power: dict[str, float]
    particle_balance: dict[str, float]
    converged: bool = True

    @property
    def electron_density(self) -> float:
        """``n_e`` in m^-3."""
        return self.densities["e"]

    def ionization_degree(self, gases: Iterable[str]) -> float:
        """``n_e / (n_e + sum n_gas)`` for the named gases."""
        gas = sum(self.densities[g] for g in gases)
        return self.electron_density / (self.electron_density + gas)

    def electronegativity(self, negative_ions: Iterable[str]) -> float:
        """``sum n_- / n_e`` for the named negative ions."""
        return sum(self.densities[s] for s in negative_ions) / self.electron_density

    def to_dataset(self) -> xr.Dataset:
        """The state as a labelled dataset (dimensions ``species``, ``reaction``, ``channel``)."""
        species = list(self.densities)
        reactions = list(self.reaction_rates)
        channels = list(self.power)
        return xr.Dataset(
            {
                "density": variable(
                    "species", [self.densities[s] for s in species], "density", "m^-3"
                ),
                "wall_loss": variable(
                    "species",
                    [self.wall_losses.get(s, 0.0) for s in species],
                    "loss to the walls",
                    "m^-3 s^-1",
                ),
                "particle_balance": variable(
                    "species",
                    [self.particle_balance[s] for s in species],
                    "rate of change of the density",
                    "m^-3 s^-1",
                ),
                "reaction_rate": variable(
                    "reaction",
                    [self.reaction_rates[r] for r in reactions],
                    "reaction rate",
                    "m^-3 s^-1",
                ),
                "electron_power": variable(
                    "channel",
                    [self.power[c] for c in channels],
                    "electron power gained",
                    "W",
                ),
                "electron_temperature": variable(
                    (), self.electron_temperature, "electron temperature", "eV"
                ),
                "gas_temperature": variable(
                    (), self.gas_temperature, "gas temperature", "eV"
                ),
                "converged": variable((), self.converged, "converged"),
            },
            coords={
                "species": ("species", species, {"long_name": "species"}),
                "reaction": ("reaction", reactions, {"long_name": "reaction"}),
                "channel": ("channel", channels, {"long_name": "power channel"}),
            },
        )


@dataclass
class GlobalModel:
    """The balances of a plasma in a chamber; see the module docstring.

    Attributes:
        species: Every species, the electrons (named ``"e"``) included.
        reactions: The volume reactions.
        chamber: Volume, walls and confinement.
        power: Power absorbed by the electrons in W, or a function of the time in
            s, ``power(t)``, for :meth:`GlobalModel.evolve` (a steady state is
            then taken at ``t = 0``).
        fixed: Densities held fixed (m^-3), e.g. a gas at a given pressure.
        feed: Particles per second fed into the chamber, by species.
        pumping_speed: Volume per second (m^3/s) pumped out, of every neutral
            species that is not held fixed.
        sources: External volume sources.
        wall_reactions: Losses of neutral species at the walls.
        ion_wall_products: What an ion returns to the volume after reaching
            the wall, ``{ion: {product: number}}``, e.g. ``{"H2+": {"H2": 1}}``.
        gas_temperature: A fixed gas temperature in eV, or its balance.
        thermalize_electrons: Keep ``T_e >= T_g``. Without sources of energy the
            wall losses cool the electrons of a decaying plasma below the gas,
            which no elastic collision allows; set it for afterglows.
    """

    species: Sequence[Species]
    reactions: Sequence[Reaction]
    chamber: Chamber
    power: float | Callable[[float], float] = 0.0
    fixed: Mapping[str, float] = field(default_factory=dict)
    feed: Mapping[str, float] = field(default_factory=dict)
    pumping_speed: float = 0.0
    sources: Sequence[VolumeSource] = ()
    wall_reactions: Sequence[WallReaction] = ()
    ion_wall_products: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    gas_temperature: float | GasTemperature = 0.025
    thermalize_electrons: bool = False

    def __post_init__(self) -> None:
        """Check that every name refers to a species and index the unknowns."""
        names = [s.name for s in self.species]
        if len(set(names)) != len(names):
            raise ValueError("species names must be unique")
        if "e" not in names:
            raise ValueError('the model needs the electrons, a species named "e"')
        self._by_name = {s.name: s for s in self.species}
        known = set(names)

        def check(where: str, used: Iterable[str]) -> None:
            unknown = set(used) - known
            if unknown:
                raise ValueError(f"{where}: unknown species {sorted(unknown)}")

        for reaction in self.reactions:
            check(reaction.name, [*reaction.reactants, *reaction.products])
        for source in self.sources:
            check(source.name, [*source.changes, *source.per])
        for wall in self.wall_reactions:
            check("wall reaction", [wall.species, *wall.products])
            if not self._by_name[wall.species].is_neutral:
                raise ValueError(f"{wall.species}: wall reactions are for neutrals")
        for ion, products in self.ion_wall_products.items():
            check("ion wall products", [ion, *products])
        check("fixed", self.fixed)
        check("feed", self.feed)
        if "e" in self.fixed:
            raise ValueError("the electron density follows from quasi-neutrality")
        self._charged = [s for s in self.species if s.charge != 0 and s.name != "e"]
        if not any(s.charge > 0 for s in self._charged):
            raise ValueError("the model needs a positive ion")
        #: The species whose densities are unknowns, in the order of the state.
        self._unknown = [
            s.name for s in self.species if s.name != "e" and s.name not in self.fixed
        ]
        self._gas_balance = isinstance(self.gas_temperature, GasTemperature)

    # -- the balances -----------------------------------------------------------------

    def _densities(self, state: np.ndarray) -> dict[str, float]:
        densities = dict(self.fixed)
        for k, name in enumerate(self._unknown):
            densities[name] = float(np.exp(np.clip(state[k], -600.0, 600.0)))
        densities["e"] = sum(s.charge * densities[s.name] for s in self._charged)
        return densities

    def _temperatures(self, state: np.ndarray) -> tuple[float, float]:
        n = len(self._unknown)
        te = float(np.exp(np.clip(state[n], -100.0, 100.0)))
        if self._gas_balance:
            tg = float(np.exp(np.clip(state[n + 1], -100.0, 100.0)))
        else:
            tg = float(self.gas_temperature)  # type: ignore[arg-type]
        if self.thermalize_electrons:
            te = max(te, tg)
        return te, tg

    def _absorbed(self, t: float) -> float:
        """The absorbed power in W at the time ``t``."""
        return float(self.power(t)) if callable(self.power) else float(self.power)

    def _terms(self, state: np.ndarray, t: float = 0.0) -> dict[str, Any]:
        """Every term of the balances at ``state`` and the time ``t``."""
        chamber = self.chamber
        volume = chamber.volume
        n = self._densities(state)
        te, tg = self._temperatures(state)
        ne = max(n["e"], 1.0)  # quasi-neutral; guarded against a negative excess
        change = dict.fromkeys(n, 0.0)
        gross = dict.fromkeys(n, 0.0)  # every gain and loss, to judge a residual
        power: dict[str, float] = {"absorbed": self._absorbed(t)}
        gas_heating = 0.0

        rates = {}
        for reaction in self.reactions:
            rate = reaction.coefficient(te, tg)
            for name in reaction.reactants:
                rate *= n[name]
            rates[reaction.name] = rates.get(reaction.name, 0.0) + rate
            for name, count in reaction.stoichiometry.items():
                change[name] += count * rate
                gross[name] += abs(count) * rate
            loss = reaction.electron_loss(te, tg) * rate * E * volume
            if loss:
                power[reaction.name] = power.get(reaction.name, 0.0) - loss
            gas_heating += reaction.gas_gain(te, tg) * rate * E

        source_rates = []
        for source in self.sources:
            rate = source.coefficient(te, tg)
            for name in source.per:
                rate *= n[name]
            source_rates.append(rate)
            for name, count in source.changes.items():
                change[name] += count * rate
                gross[name] += abs(count) * rate
            if source.electron_energy:
                power[source.name] = source.electron_energy * rate * E * volume

        walls = dict.fromkeys(n, 0.0)
        ion_energy = 0.0
        for ion in self._charged:
            if ion.charge < 0:
                continue  # held by the sheath
            area = chamber.ion_area(te, tg, ion.mass)
            loss = n[ion.name] * np.sqrt(E * te / ion.mass) * area / volume
            walls[ion.name] = loss
            change[ion.name] -= loss
            gross[ion.name] += loss
            floating = te * np.log(np.sqrt(ion.mass / (2.0 * np.pi * ELECTRON_MASS)))
            ion_energy += loss * (0.5 * te + ion.charge * floating)
            for product, count in self.ion_wall_products.get(ion.name, {}).items():
                change[product] += count * loss
        for wall in self.wall_reactions:
            mass = self._by_name[wall.species].mass
            mean_speed = np.sqrt(8.0 * E * tg / (np.pi * mass))
            loss = (
                wall.probability
                * 0.25
                * mean_speed
                * n[wall.species]
                * chamber.wall_area
                / volume
            )
            walls[wall.species] += loss
            change[wall.species] -= loss
            gross[wall.species] += loss
            for product, count in wall.products.items():
                change[product] += count * loss
                gross[product] += count * loss
        for name, rate in self.feed.items():
            change[name] += rate / volume
            gross[name] += rate / volume
        if self.pumping_speed:
            for name in self._unknown:
                if self._by_name[name].is_neutral:
                    change[name] -= n[name] * self.pumping_speed / volume
                    gross[name] += n[name] * self.pumping_speed / volume
        # floating walls: the electrons leave as fast as the positive charge does,
        # and as fast as the sources add net negative charge
        net_charge_source = sum(
            -self._by_name[name].charge * count * rate
            for source, rate in zip(self.sources, source_rates, strict=True)
            for name, count in source.changes.items()
        )
        electron_wall = (
            sum(ion.charge * walls[ion.name] for ion in self._charged if ion.charge > 0)
            + net_charge_source
        )
        walls["e"] = electron_wall
        power["walls"] = -(electron_wall * 2.0 * te + ion_energy) * E * volume

        # the electron density follows the charged species
        change["e"] = sum(s.charge * change[s.name] for s in self._charged)
        net_power = sum(power.values())  # W
        # d(3/2 n_e T_e)/dt = net power / V, in eV m^-3 s^-1
        dte = (net_power / (E * volume) - 1.5 * te * change["e"]) / (1.5 * ne)

        terms = {
            "densities": n,
            "te": te,
            "tg": tg,
            "change": change,
            "rates": rates,
            "walls": walls,
            "power": power,
            "dte": dte,
            "gross": gross,
        }
        if self._gas_balance:
            gas = self.gas_temperature
            assert isinstance(gas, GasTemperature)
            neutrals = sum(n[s.name] for s in self.species if s.is_neutral)
            conduction = (
                gas.conductivity
                * (tg - gas.wall_temperature)
                * E
                / BOLTZMANN
                / gas.diffusion_length**2
            )  # W/m^3
            terms["gas_power"] = {"heating": gas_heating, "conduction": -conduction}
            terms["dtg"] = (gas_heating - conduction) / (
                gas.heat_capacity * neutrals * E
            )
        return terms

    def _derivative(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        """``d state / dt``: the balances in the logarithms of the unknowns."""
        terms = self._terms(state, t)
        n, change = terms["densities"], terms["change"]
        rates = [change[name] / n[name] for name in self._unknown]
        dlog_te = terms["dte"] / terms["te"]
        if self.thermalize_electrons:
            # at the floor T_e cannot fall; below it (the state is a log) it is pulled up
            floor = np.log(terms["tg"])
            n = len(self._unknown)
            if state[n] <= floor:
                dlog_te = max(dlog_te, 0.0) + (floor - state[n]) / 1.0e-8
        rates.append(dlog_te)
        if self._gas_balance:
            rates.append(terms["dtg"] / terms["tg"])
        return np.array(rates)

    def _state(self, guess: Mapping[str, float] | GlobalState | None) -> np.ndarray:
        """The state vector of a guess (densities by name and temperatures)."""
        if isinstance(guess, GlobalState):
            values = {
                **guess.densities,
                "Te": guess.electron_temperature,
                "Tg": guess.gas_temperature,
            }
        else:
            values = dict(guess or {})
        gas = sum(self.fixed.values()) or 1.0e19
        positives = [s for s in self._charged if s.charge > 0]
        state = []
        for name in self._unknown:
            species = self._by_name[name]
            if name in values:
                value = values[name]
            elif species.charge > 0:
                value = 1.0e-4 * gas / len(positives)
            elif species.charge < 0:
                value = 1.0e-6 * gas
            else:
                value = 1.0e-2 * gas
            state.append(np.log(max(value, 1.0)))
        state.append(np.log(values.get("Te", 3.0)))
        if self._gas_balance:
            gas_model = self.gas_temperature
            assert isinstance(gas_model, GasTemperature)
            state.append(np.log(values.get("Tg", gas_model.wall_temperature)))
        return np.array(state)

    def evaluate(
        self, state: np.ndarray, converged: bool = True, t: float = 0.0
    ) -> GlobalState:
        """Return the :class:`GlobalState` of a state vector at ``t``, with every term."""
        terms = self._terms(state, t)
        power = {k: float(v) for k, v in terms["power"].items()}
        balance = {name: float(terms["change"][name]) for name in terms["densities"]}
        result = GlobalState(
            densities=terms["densities"],
            electron_temperature=terms["te"],
            gas_temperature=terms["tg"],
            reaction_rates={k: float(v) for k, v in terms["rates"].items()},
            wall_losses={k: float(v) for k, v in terms["walls"].items()},
            power=power,
            particle_balance=balance,
            converged=converged,
        )
        if self._gas_balance:
            result.power.update(
                {
                    f"gas {k}": v * self.chamber.volume
                    for k, v in terms["gas_power"].items()
                }
            )
        return result

    # -- solving ----------------------------------------------------------------------

    @profile("wellbalanced.GlobalModel.steady_state", tags=_PROFILE_TAGS)
    def steady_state(
        self,
        initial: Mapping[str, float] | GlobalState | None = None,
        *,
        t_max: float = 10.0,
        tolerance: float = 1.0e-8,
    ) -> GlobalState:
        """Return the steady state.

        The balances are integrated in pseudo-time from ``initial`` (densities
        by name, ``"Te"`` and ``"Tg"`` in eV; a previous :class:`GlobalState`
        for a continuation) with an implicit method for up to ``t_max``
        seconds, and the end point refined with a root finder. ``converged`` is
        set when every relative rate of change is below ``tolerance`` times the
        fastest rate of the model's processes.
        """
        start = self._state(initial)

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            return self._derivative(y)

        def settled(_t: float, y: np.ndarray) -> float:
            rates = np.abs(self._derivative(y))
            return float(np.max(rates) - 1.0e-6 * self._scale(y))

        settled.terminal = True  # type: ignore[attr-defined]
        run = solve_ivp(
            rhs,
            (0.0, t_max),
            start,
            method="BDF",
            rtol=1.0e-6,
            atol=1.0e-9,
            events=settled,
        )
        polished = root(self._derivative, run.y[:, -1], method="hybr", tol=1e-13)
        state = polished.x if polished.success else run.y[:, -1]
        residual = float(np.max(np.abs(self._derivative(state))))
        converged = bool(residual < tolerance * self._scale(state))
        return self.evaluate(state, converged=converged)

    @profile("wellbalanced.GlobalModel.evolve", tags=_PROFILE_TAGS)
    def evolve(
        self,
        times: Sequence[float] | np.ndarray,
        initial: Mapping[str, float] | GlobalState | None = None,
        *,
        breakpoints: Sequence[float] = (),
    ) -> list[GlobalState]:
        """The states at ``times`` (s) of the balances integrated in time.

        The integration starts at ``times[0]`` from ``initial`` (see
        :meth:`steady_state`) with an implicit method, driven by ``power(t)`` if
        that is a function. Stop at every ``breakpoints`` time inside the
        interval, where the drive jumps (the edges of a pulse), so the step
        size does not have to find them.
        """
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be a strictly increasing 1D sequence")
        edges = sorted({float(b) for b in breakpoints if times[0] < b < times[-1]})
        edges = [float(times[0]), *edges, float(times[-1])]
        y = self._state(initial)
        states = [self.evaluate(y, t=float(times[0]))]
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            inside = times[(times > lo) & (times <= hi)]
            if len(inside) == 0 or inside[-1] < hi:
                inside = np.append(inside, hi)
            run = solve_ivp(
                lambda t, y: self._derivative(y, t),
                (lo, hi),
                y,
                method="BDF",
                t_eval=inside,
                rtol=1.0e-6,
                atol=1.0e-9,
            )
            if not run.success:
                raise RuntimeError(f"the time integration failed: {run.message}")
            for t, column in zip(run.t, run.y.T, strict=True):
                if t in times:
                    states.append(self.evaluate(column, t=float(t)))
            y = run.y[:, -1]
        return states

    def _scale(self, state: np.ndarray) -> float:
        """The fastest relative gain or loss (1/s) of any unknown, to judge a residual."""
        terms = self._terms(state)
        n, gross = terms["densities"], terms["gross"]
        return max([gross[name] / n[name] for name in self._unknown] + [1.0])

    def scan(
        self, parameter: str, values: Sequence[Any], **guess: float
    ) -> list[GlobalState]:
        """Steady states for every value of a field of the model.

        Each starts from the last (a continuation); ``guess`` starts the first.
        """
        states: list[GlobalState] = []
        previous: GlobalState | Mapping[str, float] | None = guess or None
        for value in values:
            model = dataclasses.replace(self, **{parameter: value})
            state = model.steady_state(previous)
            states.append(state)
            previous = state
        return states
