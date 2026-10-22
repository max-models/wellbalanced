"""A beam-driven plasma neutralizer: the beam's charge states and the plasma it makes.

A fast beam of negative ions crosses a channel of length ``L`` and cross section
``w h`` filled with gas. On the gas, and on the plasma the beam itself makes,
the beam changes charge state (``-``, ``0``, ``+``): the fractions along the
channel follow from :func:`plasmacoll.charge_state_fractions` with the line
densities ``n_gas z`` and ``n_e z``, where the gas counts the molecules and,
weighted (``atoms``: ``{name: (weight, ion)}``, e.g. half a molecule each), the
atoms the plasma dissociates them into. In turn every charge change and every
ionization of the gas by a fast particle feeds the plasma, which a
:class:`~wellbalanced.GlobalModel` of the channel balances:

- a transition that frees beam electrons (``- -> 0``, ``- -> +``, ``0 -> +``)
  adds that many fast electrons at the beam velocity, of energy
  ``(m_e / M) E_beam`` each (272 eV for 1 MeV D-), a share
  ``fast_electron_deposition`` of which the plasma keeps;
- a capture from the gas (``+ -> 0``, ``0 -> -``) ionizes a gas molecule
  (the electron goes to the beam);
- the ionization of the gas by a fast particle of state ``s`` (cross section
  ``gas_ionization[s]``) makes an ion and a secondary electron of mean
  energy ``secondary_energy[s]``: the mean of its energy distribution, e.g.
  Rudd's (:class:`plasmacoll.RuddIonization`), at the beam's energy.

The sources are functions of the local beam (:meth:`Neutralizer.beam_sources`):
with a particle flux ``Gamma`` (beam particles per m^2 and s), the fractions
``F_s`` of its states and its energy ``E``, a process of cross section
``sigma(E)`` on a target of density ``n`` happens at the volume rate
``Gamma F_s sigma(E) n``: a :class:`~wellbalanced.VolumeSource` per unit target
density, so a model treats it self-consistently with its own gas and electron
densities. The global model takes them with the beam's mean flux
``N / (w h)`` (``N`` particles per second) and the fractions averaged along the
channel; a :class:`~wellbalanced.Fluid1D` takes them with profiles across the
channel. The cross sections are numbers at the beam's energy
(:class:`BeamData`) or functions of it (:class:`BeamCrossSections`). The gas
is fed at ``gas_inflow`` and leaves through the two open ends of the channel at
the free-molecular rate ``2 T (v_mean / 4) w h`` (transmission ``T``), with the
mean speed at the current gas temperature: gas heating lowers the density.

The beam fractions depend on the densities through the line densities, so the
two are iterated to a fixed point (:meth:`Neutralizer.solve`). The plasma
target's density is the electron density (equal to the ion density in a
quasi-neutral plasma); captures on the plasma are not counted as sources.
Assumptions to check against kinetic runs (``neutralizer_plan.md``, step 4):
Maxwellian electrons, the cusp leak width, uniform plasma and gas.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import xarray as xr
from plasmacoll import ChargeStateTransitions, charge_state_fractions
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE
from scope_profiler import profile

from wellbalanced._profiling import _PROFILE_TAGS
from wellbalanced.datasets import variable
from wellbalanced.global_model import (
    Chamber,
    Cusp,
    GasTemperature,
    GlobalModel,
    GlobalState,
    VolumeSource,
    WallReaction,
)
from wellbalanced.reactions import Reaction
from wellbalanced.species import ELECTRON, Species

__all__ = [
    "BeamCrossSections",
    "BeamData",
    "FastBeam",
    "Neutralizer",
    "NeutralizerResult",
    "SecondaryElectrons",
    "illustrative_deuterium_neutralizer",
]

E = ELEMENTARY_CHARGE
#: The charge (in e) of every beam state.
STATE_CHARGES = {"-": -1, "0": 0, "+": 1}


@dataclass(frozen=True)
class FastBeam:
    """The incoming beam.

    Attributes:
        energy: Energy per particle in eV.
        current: Electric current of the incoming beam in A.
        mass: Mass of a beam particle in kg.
        initial: Charge-state fractions at the entrance.
    """

    energy: float
    current: float
    mass: float
    initial: Mapping[str, float] = field(default_factory=lambda: {"-": 1.0})

    def __post_init__(self) -> None:
        """Validate the beam."""
        if self.energy <= 0.0 or self.current < 0.0 or self.mass <= 0.0:
            raise ValueError("a beam needs energy > 0, current >= 0 and mass > 0")
        object.__setattr__(self, "initial", dict(self.initial))

    @property
    def particle_rate(self) -> float:
        """Beam particles per second: the current over the charge of the entering beam."""
        charge = sum(abs(STATE_CHARGES[s]) * f for s, f in self.initial.items())
        return self.current / (E * charge) if charge else self.current / E

    @property
    def stripped_electron_energy(self) -> float:
        """Energy in eV of an electron moving with the beam, ``(m_e / M) E``."""
        return ELECTRON_MASS / self.mass * self.energy


@runtime_checkable
class SecondaryElectrons(Protocol):
    """Ionization of the gas by a fast particle, with its secondary electrons' energies.

    :class:`plasmacoll.RuddIonization` is one.
    """

    def total(self, projectile_energy: float) -> float:
        """The ionization cross section (m^2) at the projectile's energy (eV)."""
        ...

    def mean_energy(self, projectile_energy: float) -> float:
        """The mean energy (eV) of the secondary electrons."""
        ...


#: A cross section (m^2) as a function of the beam energy (eV), e.g. a
#: :class:`plasmacoll.CrossSection` in the lab frame.
CrossSectionFunction = Callable[[np.ndarray], Any]


@dataclass(frozen=True)
class BeamData:
    """The beam's cross sections at its energy (m^2).

    Attributes:
        transitions: Charge changes on the targets ``"gas"`` and ``"plasma"``.
        gas_ionization: Cross section for ionizing the gas, per beam state.
        secondary_energy: Mean energy (eV) of the secondary electrons of gas
            ionization, per beam state; the neutralizer's ``secondary_energy``
            for a state missing here.
    """

    transitions: ChargeStateTransitions
    gas_ionization: Mapping[str, float] = field(default_factory=dict)
    secondary_energy: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the targets and states."""
        unknown = set(self.transitions.targets) - {"gas", "plasma"}
        if unknown:
            raise ValueError(
                f"beam targets are 'gas' and 'plasma', not {sorted(unknown)}"
            )
        if set(self.transitions.states) - set(STATE_CHARGES):
            raise ValueError(f"beam states must be among {tuple(STATE_CHARGES)}")
        object.__setattr__(self, "gas_ionization", dict(self.gas_ionization))
        object.__setattr__(self, "secondary_energy", dict(self.secondary_energy))
        if any(e < 0.0 for e in self.secondary_energy.values()):
            raise ValueError("secondary energies must be >= 0")


@dataclass(frozen=True)
class BeamCrossSections:
    """The beam's cross sections as functions of its energy; :meth:`at` gives :class:`BeamData`.

    Every cross section is a number (m^2) or a function of the beam energy
    (eV), e.g. a :class:`plasmacoll.CrossSection` tabulated in the lab frame.
    The gas ionization may be a :class:`SecondaryElectrons` instead, such as
    :class:`plasmacoll.RuddIonization` with the beam particle's mass: then the
    secondary electrons' mean energy follows the beam energy too.

    Attributes:
        transitions: ``{target: {(from_state, to_state): sigma}}``, targets
            ``"gas"`` and ``"plasma"``.
        gas_ionization: Ionization of the gas, per beam state.
        states: The charge states, in the order of the fractions.
    """

    transitions: Mapping[str, Mapping[tuple[str, str], float | CrossSectionFunction]]
    gas_ionization: Mapping[str, float | CrossSectionFunction | SecondaryElectrons] = (
        field(default_factory=dict)
    )
    states: Sequence[str] = ("-", "0", "+")

    def at(self, energy: float) -> BeamData:
        """The cross sections at the beam energy ``energy`` (eV)."""
        if energy <= 0.0:
            raise ValueError("the beam energy must be > 0")
        transitions = {
            target: {pair: _evaluate(sigma, energy) for pair, sigma in table.items()}
            for target, table in self.transitions.items()
        }
        ionization, secondaries = {}, {}
        for state, value in self.gas_ionization.items():
            if isinstance(value, SecondaryElectrons):
                ionization[state] = float(value.total(energy))
                secondaries[state] = float(value.mean_energy(energy))
            else:
                ionization[state] = _evaluate(value, energy)
        return BeamData(
            ChargeStateTransitions(transitions, tuple(self.states)),
            ionization,
            secondaries,
        )


def _evaluate(sigma: float | CrossSectionFunction, energy: float) -> float:
    """A cross section (m^2) at the beam energy (eV): a number, or a function of it."""
    if callable(sigma):
        return float(np.asarray(sigma(np.array([energy]))).reshape(-1)[0])
    return float(sigma)


@dataclass
class NeutralizerResult:
    """A converged neutralizer.

    Attributes:
        position: Points along the channel (m).
        fractions: Charge-state fractions at ``position``, ``(points, states)``.
        states: The state names, in the order of ``fractions``.
        plasma: The steady state of the channel's plasma.
        gas_density: Density of the gas target, molecules and weighted atoms (m^-3).
        beam_power: Power the beam gives the plasma's electrons (W).
        iterations: Fixed-point iterations.
        converged: Whether the fixed point and the plasma converged.
    """

    position: np.ndarray
    fractions: np.ndarray
    states: tuple[str, ...]
    plasma: GlobalState
    gas_density: float
    beam_power: float
    iterations: int
    converged: bool

    @property
    def neutral_fraction(self) -> float:
        """The neutral fraction at the exit."""
        return float(self.fractions[-1, self.states.index("0")])

    @property
    def exit_fractions(self) -> dict[str, float]:
        """Every state's fraction at the exit."""
        return {
            s: float(f) for s, f in zip(self.states, self.fractions[-1], strict=True)
        }

    def to_dataset(self) -> xr.Dataset:
        """The beam along the channel (``state``, ``position``) and the plasma's state."""
        plasma = self.plasma.to_dataset().rename({"converged": "plasma_converged"})
        beam = xr.Dataset(
            {
                "fraction": variable(
                    ("state", "position"), self.fractions.T, "fraction of the beam", "1"
                ),
                "neutral_fraction": variable(
                    (), self.neutral_fraction, "neutral fraction at the exit", "1"
                ),
                "gas_density": variable(
                    (), self.gas_density, "gas target density", "m^-3"
                ),
                "beam_power": variable((), self.beam_power, "power from the beam", "W"),
                "converged": variable((), self.converged, "converged"),
            },
            coords={
                "state": ("state", list(self.states), {"long_name": "charge state"}),
                "position": (
                    "position",
                    self.position,
                    {"long_name": "position along the channel", "units": "m"},
                ),
            },
        )
        # keep every variable's long_name and units
        merged = xr.merge([beam, plasma], combine_attrs="override")
        merged.attrs = {}
        return merged


@dataclass
class Neutralizer:
    """One channel of a beam-driven plasma neutralizer; see the module docstring.

    Attributes:
        length: Channel length along the beam (m).
        width: Channel width (m).
        height: Channel height (m).
        beam: The incoming beam.
        data: The beam's cross sections, at its energy or as functions of it.
        species: The plasma and gas species, the electrons included.
        reactions: The plasma's volume reactions.
        gas: The name of the fed gas.
        gas_ion: The ion the beam makes from the gas.
        gas_inflow: Gas molecules fed per second.
        atoms: Other neutrals that act as gas targets, ``{name: (weight, ion)}``:
            their cross sections are the gas's times ``weight``, and ``ion`` is
            the ion the beam makes from them.
        transmission: Share of the molecules reaching an end that leave.
        cusp: Magnetic cusps on the side walls, if any; then the ions leave
            unconfined only through the two open ends.
        h_factor: Edge-to-centre ratio where ions leave unconfined.
        gas_temperature: A fixed gas temperature (eV) or its balance.
        wall_reactions: Neutral losses at the walls.
        ion_wall_products: What an ion returns from the wall.
        fast_electron_deposition: Share of the fast electrons' energy the plasma keeps.
        secondary_energy: Mean energy (eV) of a secondary electron of gas
            ionization, for the beam states whose data give none.
        points: Points along the channel for the beam fractions.
        relaxation: Under-relaxation of the beam fractions between iterations.
        tolerance: Convergence of the fixed point (largest change of a fraction).
        max_iterations: Fixed-point iterations at most.
    """

    length: float
    width: float
    height: float
    beam: FastBeam
    data: BeamData | BeamCrossSections
    species: Sequence[Species]
    reactions: Sequence[Reaction]
    gas: str
    gas_ion: str
    gas_inflow: float
    atoms: Mapping[str, tuple[float, str]] = field(default_factory=dict)
    transmission: float = 1.0
    cusp: Cusp | None = None
    h_factor: float = 0.5
    gas_temperature: float | GasTemperature = 0.025
    wall_reactions: Sequence[WallReaction] = ()
    ion_wall_products: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    fast_electron_deposition: float = 1.0
    secondary_energy: float = 10.0
    points: int = 201
    relaxation: float = 0.5
    tolerance: float = 1.0e-6
    max_iterations: int = 200

    def __post_init__(self) -> None:
        """Validate the geometry and the names."""
        if min(self.length, self.width, self.height) <= 0.0:
            raise ValueError("the channel needs a positive length, width and height")
        names = {s.name for s in self.species}
        for name in (
            self.gas,
            self.gas_ion,
            *self.atoms,
            *(i for _, i in self.atoms.values()),
        ):
            if name not in names:
                raise ValueError(f"no species {name!r}")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation must be in (0, 1]")

    # -- geometry and gas -----------------------------------------------------------

    @property
    def volume(self) -> float:
        """The channel volume (m^3)."""
        return self.length * self.width * self.height

    @property
    def cross_section(self) -> float:
        """The channel cross section ``w h`` (m^2)."""
        return self.width * self.height

    def pumping_speed(self, gas_temperature: float) -> float:
        """Volume per second (m^3/s) leaving through both ends at the gas temperature (eV)."""
        mass = next(s.mass for s in self.species if s.name == self.gas)
        mean_speed = np.sqrt(8.0 * E * gas_temperature / (np.pi * mass))
        return 2.0 * self.transmission * 0.25 * mean_speed * self.cross_section

    def gas_target_density(self, densities: Mapping[str, float]) -> float:
        """The density of the gas target: the molecules plus the weighted atoms (m^-3)."""
        return densities[self.gas] + sum(
            weight * densities[name] for name, (weight, _) in self.atoms.items()
        )

    # -- the beam -------------------------------------------------------------------

    def beam_data(self, energy: float | None = None) -> BeamData:
        """The beam's cross sections at ``energy`` (eV; the beam's by default)."""
        if isinstance(self.data, BeamData):
            if energy is not None and energy != self.beam.energy:
                raise ValueError(
                    "BeamData holds the cross sections at the beam's energy; "
                    "use BeamCrossSections for others"
                )
            return self.data
        return self.data.at(self.beam.energy if energy is None else energy)

    @property
    def mean_flux(self) -> float:
        """The beam's particle flux (m^-2 s^-1) averaged over the cross section."""
        return self.beam.particle_rate / self.cross_section

    def fractions(
        self, gas_density: float, plasma_density: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """The positions along the channel and the beam's fractions there."""
        transitions = self.beam_data().transitions
        z = np.linspace(0.0, self.length, self.points)
        line = {"gas": gas_density * z}
        if "plasma" in transitions.targets:
            line["plasma"] = plasma_density * z
        line = {t: v for t, v in line.items() if t in transitions.targets}
        fractions = charge_state_fractions(transitions, line, initial=self.beam.initial)
        return z, fractions

    def _sources(self, mean: np.ndarray) -> list[VolumeSource]:
        """The beam's sources for the plasma, with the fractions averaged along the channel."""
        return self.beam_sources(self.mean_flux, mean)

    def beam_sources(
        self,
        flux: float | np.ndarray,
        fractions: Sequence[float] | np.ndarray,
        energy: float | None = None,
    ) -> list[VolumeSource]:
        """The beam's sources for the plasma where the beam is ``flux``, ``fractions``, ``energy``.

        Args:
            flux: Beam particles per m^2 and s, all states together: a number,
                or an array along a coordinate (the cells of a
                :class:`~wellbalanced.Fluid1D`).
            fractions: The fractions of the states (in the order of the data's
                states), ``(states,)`` or ``(points, states)`` with ``points``
                matching ``flux``.
            energy: The beam energy in eV, the incoming beam's by default.

        Returns:
            One :class:`~wellbalanced.VolumeSource` per process, its rate per
            unit density of its target (a number, or an array like ``flux``).
        """
        energy = self.beam.energy if energy is None else float(energy)
        data = self.beam_data(energy)
        states = data.transitions.states
        shares = np.asarray(fractions, dtype=float)
        if shares.shape[-1] != len(states):
            raise ValueError(f"fractions need one value per state {tuple(states)}")
        rate = np.asarray(flux, dtype=float)
        if np.any(rate < 0.0) or np.any(shares < 0.0):
            raise ValueError("flux and fractions must be >= 0")
        # the energy of an electron moving with the beam, (m_e / M) E
        stripped = ELECTRON_MASS / self.beam.mass * energy
        fast = stripped * self.fast_electron_deposition
        # every neutral that is a gas target: (name, weight, the ion the beam makes)
        gases = [(self.gas, 1.0, self.gas_ion)] + [
            (name, weight, ion) for name, (weight, ion) in self.atoms.items()
        ]
        sources = []

        def coefficient(state: str, sigma: float) -> float | np.ndarray:
            value = rate * shares[..., states.index(state)] * sigma
            return float(value) if value.ndim == 0 else value

        for target, table in data.transitions.cross_sections.items():
            for (start, end), sigma in table.items():
                if sigma <= 0.0 or not np.any(shares[..., states.index(start)] > 0.0):
                    continue
                freed = STATE_CHARGES[end] - STATE_CHARGES[start]
                name = f"beam {start}->{end} on {target}"
                if target == "plasma":
                    if freed > 0:  # captures from the plasma are not counted
                        sources.append(
                            VolumeSource(
                                {"e": freed},
                                coefficient(start, sigma),
                                per=("e",),
                                electron_energy=freed * fast,
                                name=name,
                            )
                        )
                    continue
                for gas, weight, ion in gases:
                    label = name if gas == self.gas else f"{name} ({gas})"
                    if freed > 0:
                        changes = {"e": freed}
                        gained = freed * fast
                    else:  # a capture ionizes the target
                        changes = {gas: freed, ion: -freed}
                        gained = 0.0
                    sources.append(
                        VolumeSource(
                            changes,
                            coefficient(start, sigma * weight),
                            per=(gas,),
                            electron_energy=gained,
                            name=label,
                        )
                    )
        for state, sigma in data.gas_ionization.items():
            if sigma <= 0.0 or not np.any(shares[..., states.index(state)] > 0.0):
                continue
            secondary = data.secondary_energy.get(state, self.secondary_energy)
            for gas, weight, ion in gases:
                label = f"gas ionization by {state}" + (
                    "" if gas == self.gas else f" ({gas})"
                )
                sources.append(
                    VolumeSource(
                        {gas: -1, ion: 1, "e": 1},
                        coefficient(state, sigma * weight),
                        per=(gas,),
                        electron_energy=secondary,
                        name=label,
                    )
                )
        return sources

    def plasma_model(self, mean: np.ndarray, gas_temperature: float) -> GlobalModel:
        """The global model of the channel with the beam sources of the mean fractions."""
        # the side walls hold the gas; ions leave through the open ends, and through
        # the side walls too unless cusps confine them there
        wall = 2.0 * self.length * (self.width + self.height)
        ends = 2.0 * self.cross_section
        chamber = Chamber(
            self.volume,
            wall,
            ion_loss_area=ends if self.cusp is not None else wall + ends,
            h_factor=self.h_factor,
            cusp=self.cusp,
        )
        return GlobalModel(
            species=self.species,
            reactions=self.reactions,
            chamber=chamber,
            feed={self.gas: self.gas_inflow},
            pumping_speed=self.pumping_speed(gas_temperature),
            sources=self._sources(mean),
            wall_reactions=self.wall_reactions,
            ion_wall_products=self.ion_wall_products,
            gas_temperature=self.gas_temperature,
        )

    # -- solving --------------------------------------------------------------------

    @profile("wellbalanced.Neutralizer.solve", tags=_PROFILE_TAGS)
    def solve(self, guess: GlobalState | None = None) -> NeutralizerResult:
        """Iterate the beam fractions and the plasma to a fixed point."""
        tg = (
            self.gas_temperature.wall_temperature
            if isinstance(self.gas_temperature, GasTemperature)
            else float(self.gas_temperature)
        )
        gas_density = self.gas_inflow / self.pumping_speed(tg)
        z, fractions = self.fractions(gas_density, 0.0)
        mean = np.trapezoid(fractions, z, axis=0) / self.length
        state = guess
        converged = False
        iterations = 0
        for _ in range(self.max_iterations):
            iterations += 1
            model = self.plasma_model(mean, tg)
            start = state or {self.gas: gas_density, "Te": 3.0}
            state = model.steady_state(start)
            tg = state.gas_temperature
            gas_density = self.gas_target_density(state.densities)
            z, fractions = self.fractions(gas_density, state.electron_density)
            new_mean = np.trapezoid(fractions, z, axis=0) / self.length
            change = float(np.max(np.abs(new_mean - mean)))
            mean = (1.0 - self.relaxation) * mean + self.relaxation * new_mean
            if change < self.tolerance and state.converged:
                converged = True
                break
        assert state is not None
        beam_power = sum(
            v
            for k, v in state.power.items()
            if k.startswith(("beam", "gas ionization"))
        )
        return NeutralizerResult(
            position=z,
            fractions=fractions,
            states=tuple(self.beam_data().transitions.states),
            plasma=state,
            gas_density=gas_density,
            beam_power=float(beam_power),
            iterations=iterations,
            converged=converged,
        )

    def scan(self, parameter: str, values: Sequence[Any]) -> list[NeutralizerResult]:
        """Results for every value of a field, each starting from the last plasma."""
        results: list[NeutralizerResult] = []
        guess: GlobalState | None = None
        for value in values:
            result = dataclasses.replace(self, **{parameter: value}).solve(guess)
            results.append(result)
            guess = result.plasma
        return results


def illustrative_deuterium_neutralizer(**overrides: Any) -> Neutralizer:
    """A deuterium neutralizer channel with illustrative numbers, not data.

    The cross sections and rates are round numbers of plausible size, chosen so
    that a gas target alone gives at most about 0.6 neutrals near
    1.4e20 m^-2 and a plasma target more: placeholders that let the model run
    and the tests check it, until real data are supplied (beam stripping:
    e.g. the ORNL Redbook, IAEA ALADDIN; plasma reactions: Janev et al.). It
    warns when called. ``overrides`` replace fields of the :class:`Neutralizer`.
    """
    warnings.warn(
        "illustrative_deuterium_neutralizer uses placeholder cross sections and "
        "rates, not data",
        UserWarning,
        stacklevel=2,
    )
    d2 = Species("D2", 4.028 * ATOMIC_MASS)
    d = Species("D", 2.014 * ATOMIC_MASS)
    d2_ion = Species("D2+", 4.028 * ATOMIC_MASS - ELECTRON_MASS, 1)
    d_ion = Species("D+", 2.014 * ATOMIC_MASS - ELECTRON_MASS, 1)
    reactions = [
        Reaction(("e", "D2"), ("e", "e", "D2+"), _arrhenius(2.0e-14, 15.4),
                 electron_energy_loss=15.4, name="ionization of D2"),
        Reaction(("e", "D2"), ("e", "D", "D"), _arrhenius(1.0e-14, 8.8),
                 electron_energy_loss=8.8, gas_heating=2.0, name="dissociation of D2"),
        Reaction(("e", "D"), ("e", "e", "D+"), _arrhenius(1.5e-14, 13.6),
                 electron_energy_loss=13.6, name="ionization of D"),
        Reaction(("e", "D2+"), ("D", "D"), lambda t: 1.0e-14 / np.sqrt(t),
                 gas_heating=5.0, name="dissociative recombination"),
        Reaction(("e", "D2"), ("e", "D2"), 3.0e-14, elastic_mass=d2.mass,
                 name="elastic e-D2"),
    ]  # fmt: skip
    data = BeamData(
        ChargeStateTransitions(
            {
                "gas": {("-", "0"): 1.2e-20, ("-", "+"): 5.0e-22, ("0", "+"): 3.7e-21},
                "plasma": {
                    ("-", "0"): 2.0e-20,
                    ("-", "+"): 1.0e-21,
                    ("0", "+"): 2.5e-21,
                },
            }
        ),
        gas_ionization={"-": 1.0e-20, "0": 8.0e-21, "+": 6.0e-21},
    )
    defaults: dict[str, Any] = {
        "length": 3.0,
        "width": 0.1,
        "height": 1.7,
        "beam": FastBeam(energy=1.0e6, current=10.0, mass=2.014 * ATOMIC_MASS),
        "data": data,
        "species": [ELECTRON, d2, d, d2_ion, d_ion],
        "reactions": reactions,
        "gas": "D2",
        "gas_ion": "D2+",
        "gas_inflow": 2.0e21,
        "atoms": {"D": (0.5, "D+")},
        "cusp": Cusp(field=0.1, line_length=2 * 3.0 * 20),
        "h_factor": 0.3,
        "gas_temperature": 0.03,
        "wall_reactions": [WallReaction("D", 0.1, {"D2": 0.5})],
        "ion_wall_products": {"D2+": {"D2": 1}, "D+": {"D": 1}},
    }
    return Neutralizer(**{**defaults, **overrides})


def _arrhenius(a: float, c: float):
    """``K = a exp(-c / T)`` (m^3/s, T in eV), a placeholder rate."""
    return lambda t: a * np.exp(-c / np.asarray(t, dtype=float))
