"""A 0D model of an inductive RF driver: a global model and a transformer model.

The driver is a cylinder of radius ``R``, its plasma heated by an RF coil of ``N``
turns, radius ``b`` and length ``l`` around it. :class:`RFDriver` couples two models:

- a :class:`~wellbalanced.GlobalModel` of the driver plasma, which fixes the electron
  density and temperature for the power the plasma absorbs;
- a transformer model of the coupling (Piejak, Godyak and Alexandrovich, Plasma
  Sources Sci. Technol. 1, 179, 1992; Lieberman and Lichtenberg, *Principles of
  Plasma Discharges*, sec. 12.2), which turns that plasma into the resistance the
  coil sees, ``R_s``, and so into the coil current ``I = sqrt(2 P / R_s)`` and the RF
  power transfer efficiency ``eta = R_s / (R_s + R_net)``: the network (matching,
  coil, Faraday shield) dissipates ``R_net I^2 / 2`` besides.

Transformer model: the coil is the primary, the plasma under it a one-turn secondary
in a skin layer at its surface. With the long-solenoid inductances

    L_11 = mu0 pi b^2 N^2 / l,   M = mu0 pi R^2 N / l,   L_2 = mu0 pi R^2 / l

and the impedance of the skin layer around the plasma,

    Z_p = (2 pi R / l) (k / sigma),   k = sqrt(i omega mu0 sigma),

the coil sees ``Z_s = i omega L_11 + omega^2 M^2 / (Z_p + i omega L_2)``. The plasma is
the cold, collisional one of :mod:`wellbalanced.inductive`,
``sigma = n_e e^2 / (m_e (nu_m + i omega))``: ``k / sigma`` is the surface impedance of
the skin layer, ``(1 + i) / (sigma delta)`` in the collisional limit (``nu_m >> omega``,
``delta = sqrt(2 / (omega mu0 sigma))``) and ``i omega mu0 c / omega_pe`` with the
electron-inertia skin depth ``c / omega_pe`` when ``nu_m << omega``, so both
limits, and the resistance ``R_p = Re Z_p`` and electron-inertia inductance
``Im Z_p / omega`` of the textbook model, come out of one expression. It is the
thin-skin limit of the uniform plasma column in a long solenoid, whose exact
impedance :func:`solenoid_impedance` gives; :func:`transformer_impedance` agrees
with it to order ``delta / R``.

A real driver coil is short (six turns over 17 cm around a 24 cm plasma): its
inductance is Nagaoka's coefficient ``K`` (:func:`nagaoka_coefficient`) times the
long-solenoid one. It is modelled as the long solenoid of ``N sqrt(K)`` turns, which
has that inductance: ``L_11`` is multiplied by ``K`` and ``M`` by ``sqrt(K)``, so the
plasma's share of the impedance scales with ``K`` as well, as the exact solution's
(:func:`solenoid_impedance`) does; ``field_factor`` overrides ``K``.

The plasma coupled to the coil is the length of the coil; the global model takes
the whole driver, which may be longer (a longer driver at the same coil spreads the
power over more plasma).

What a 0D model cannot give: the distance between the coil and an electromagnetic
shield and the mutual coupling of several drivers change the field around the
coil, not the plasma; they are in the 2D field of :mod:`wellbalanced.inductive`.
Neither is the non-local (stochastic) heating of low pressures, which the cold
plasma's collision frequency leaves out.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import xarray as xr
from plasmacoll import CrossSection
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy.constants import mu_0 as MU_0
from scipy.special import ellipe, ellipk, jve

from wellbalanced.datasets import variable
from wellbalanced.global_model import Chamber, GlobalModel, GlobalState
from wellbalanced.reactions import Reaction
from wellbalanced.species import ELECTRON, Species

__all__ = [
    "Coil",
    "DriverState",
    "RFDriver",
    "illustrative_hydrogen_driver",
    "nagaoka_coefficient",
    "plasma_conductivity",
    "solenoid_impedance",
    "transformer_impedance",
]

E = ELEMENTARY_CHARGE


def plasma_conductivity(
    electron_density: float, collision_frequency: float, frequency: float
) -> complex:
    """``sigma = n_e e^2 / (m_e (nu_m + i omega))`` in S/m, for ``exp(i omega t)``."""
    omega = 2.0 * np.pi * frequency
    return (
        electron_density * E**2 / (ELECTRON_MASS * (collision_frequency + 1j * omega))
    )


def _wavenumber(sigma: complex, frequency: float) -> complex:
    """``k = sqrt(i omega mu0 sigma)``, the root with ``Re k > 0`` (decay into the plasma)."""
    k = np.sqrt(1j * 2.0 * np.pi * frequency * MU_0 * sigma)
    return complex(k if k.real >= 0.0 else -k)


def nagaoka_coefficient(diameter: float, length: float) -> float:
    """Nagaoka's coefficient of a solenoid: its inductance over the long-solenoid one.

    ``K = 4 / (3 pi k') ((k'^2 / k^2) (K(k) - E(k)) + E(k) - k)`` with
    ``k^2 = d^2 / (d^2 + l^2)`` and ``k'^2 = 1 - k^2``; 1 for a long coil, 0.6884 for
    ``d = l``.
    """
    if diameter <= 0.0 or length <= 0.0:
        raise ValueError("a coil needs diameter > 0 and length > 0")
    m = diameter**2 / (diameter**2 + length**2)
    k, kp = math.sqrt(m), math.sqrt(1.0 - m)
    big_k, big_e = float(ellipk(m)), float(ellipe(m))
    return 4.0 / (3.0 * math.pi * kp) * ((kp**2 / k**2) * (big_k - big_e) + big_e - k)


@dataclass(frozen=True)
class Coil:
    """The RF coil: ``turns`` windings of ``radius`` (m) over ``length`` (m).

    ``field_factor`` is the ratio of its inductance to the long-solenoid one;
    Nagaoka's coefficient of the coil if ``None``.
    """

    turns: float
    radius: float
    length: float
    field_factor: float | None = None

    def __post_init__(self) -> None:
        """Validate the coil."""
        if min(self.turns, self.radius, self.length) <= 0.0:
            raise ValueError("a coil needs turns, radius and length > 0")
        if self.field_factor is not None and not 0.0 < self.field_factor <= 1.0:
            raise ValueError("field_factor must be in (0, 1]")

    @property
    def factor(self) -> float:
        """The inductance ratio used: ``field_factor`` or Nagaoka's coefficient."""
        if self.field_factor is not None:
            return self.field_factor
        return nagaoka_coefficient(2.0 * self.radius, self.length)

    @property
    def inductance(self) -> float:
        """The coil's inductance without plasma, ``K mu0 pi b^2 N^2 / l`` in H."""
        return self.factor * MU_0 * np.pi * self.radius**2 * self.turns**2 / self.length


def transformer_impedance(
    coil: Coil,
    plasma_radius: float,
    electron_density: float,
    collision_frequency: float,
    frequency: float,
) -> complex:
    """The impedance in Ohm the coil sees with a uniform plasma column inside.

    The transformer model of the module docstring; ``Re`` is the resistance ``R_s``
    through which the plasma absorbs ``R_s |I|^2 / 2``. A thin-skin model, for skin
    depths up to a fraction of ``R``: for a faint plasma (``delta >> R``) its ``R_s``
    falls like ``sqrt(sigma)`` instead of ``sigma``; :func:`solenoid_impedance` holds
    there.
    """
    if not 0.0 < plasma_radius < coil.radius:
        raise ValueError("the plasma must be inside the coil")
    omega = 2.0 * np.pi * frequency
    factor = coil.factor
    l_11 = factor * MU_0 * np.pi * coil.radius**2 * coil.turns**2 / coil.length
    mutual = (
        math.sqrt(factor) * MU_0 * np.pi * plasma_radius**2 * coil.turns / coil.length
    )
    l_2 = MU_0 * np.pi * plasma_radius**2 / coil.length
    if electron_density <= 0.0:
        return complex(1j * omega * l_11)
    sigma = plasma_conductivity(electron_density, collision_frequency, frequency)
    loop = 2.0 * np.pi * plasma_radius / coil.length  # the skin layer's circuit
    z_p = loop * _wavenumber(sigma, frequency) / sigma
    return complex(1j * omega * l_11 + omega**2 * mutual**2 / (z_p + 1j * omega * l_2))


def solenoid_impedance(
    coil: Coil,
    plasma_radius: float,
    electron_density: float,
    collision_frequency: float,
    frequency: float,
) -> complex:
    """The exact impedance of a uniform plasma column in a long solenoid, in Ohm.

    Inside the column ``B_z = B_0 J_0(q r) / J_0(q R)`` with ``q^2 = -i omega mu0 sigma``,
    so ``Z = i omega mu0 N^2 / l (pi (b^2 - R^2) + 2 pi R J_1(q R) / (q J_0(q R)))``,
    times the coil's inductance ratio ``K``; any skin depth, the reference of
    :func:`transformer_impedance`.
    """
    if not 0.0 < plasma_radius < coil.radius:
        raise ValueError("the plasma must be inside the coil")
    omega = 2.0 * np.pi * frequency
    scale = coil.factor * 1j * omega * MU_0 * coil.turns**2 / coil.length
    gap = np.pi * (coil.radius**2 - plasma_radius**2)
    if electron_density <= 0.0:
        return complex(scale * (gap + np.pi * plasma_radius**2))
    sigma = plasma_conductivity(electron_density, collision_frequency, frequency)
    q = np.sqrt(-1j * omega * MU_0 * sigma)
    x = q * plasma_radius
    # jve: J scaled by exp(-|Im x|), the same in both, so the ratio does not overflow
    ratio = jve(1, x) / jve(0, x)
    return complex(scale * (gap + 2.0 * np.pi * plasma_radius * ratio / q))


@dataclass
class DriverState:
    """A driver at one absorbed power: its plasma and the coil's view of it.

    Attributes:
        plasma: The steady state of the driver's global model.
        power: The power the plasma absorbs in W.
        frequency: The RF frequency in Hz.
        collision_frequency: The electrons' momentum-transfer frequency ``nu_m`` (1/s).
        coil_impedance: ``Z_s`` in Ohm; its real part is ``R_s``.
        network_resistance: ``R_net`` in Ohm.
        skin_depth: ``1 / Re k`` in m.
    """

    plasma: GlobalState
    power: float
    frequency: float
    collision_frequency: float
    coil_impedance: complex
    network_resistance: float
    skin_depth: float

    @property
    def plasma_resistance(self) -> float:
        """``R_s = Re Z_s`` in Ohm."""
        return self.coil_impedance.real

    @property
    def coil_current(self) -> float:
        """The coil current amplitude ``sqrt(2 P / R_s)`` in A."""
        return math.sqrt(2.0 * self.power / self.plasma_resistance)

    @property
    def coil_voltage(self) -> float:
        """The voltage amplitude across the coil, ``|Z_s| I``, in V."""
        return abs(self.coil_impedance) * self.coil_current

    @property
    def efficiency(self) -> float:
        """The RF power transfer efficiency ``R_s / (R_s + R_net)``."""
        r_s = self.plasma_resistance
        return r_s / (r_s + self.network_resistance)

    @property
    def rf_power(self) -> float:
        """The power the generator delivers, ``P / eta``, in W."""
        return self.power / self.efficiency

    def to_dataset(self) -> xr.Dataset:
        """The plasma's dataset with the coupling's quantities."""
        scalars = {
            "absorbed_power": (self.power, "power absorbed by the plasma", "W"),
            "rf_power": (self.rf_power, "RF power", "W"),
            "coil_current": (self.coil_current, "coil current amplitude", "A"),
            "coil_voltage": (self.coil_voltage, "coil voltage amplitude", "V"),
            "plasma_resistance": (self.plasma_resistance, "plasma resistance", "Ohm"),
            "coil_reactance": (self.coil_impedance.imag, "coil reactance", "Ohm"),
            "efficiency": (self.efficiency, "RF power transfer efficiency", ""),
            "skin_depth": (self.skin_depth, "skin depth", "m"),
            "collision_frequency": (
                self.collision_frequency,
                "electron momentum-transfer frequency",
                "1/s",
            ),
        }
        dataset = self.plasma.to_dataset()
        for name, (value, long_name, units) in scalars.items():
            dataset[name] = variable((), value, long_name, units)
        return dataset


@dataclass
class RFDriver:
    """A cylindrical RF driver: its plasma's global model and its coil.

    The chamber is the driver's cylinder; the ions leave through its whole wall,
    the end open to the expansion region included, with the low-pressure
    edge-to-centre ratios ``h_L = 0.86 / sqrt(3 + l / (2 lambda_i))`` (ends) and
    ``h_R = 0.8 / sqrt(4 + R / lambda_i)`` (side) of Lieberman and Lichtenberg
    (sec. 10.2) for the ion mean free path ``lambda_i``.

    Attributes:
        species: Every species, the electrons (``"e"``) included.
        reactions: The volume reactions.
        fixed: Densities held fixed (m^-3), the gases.
        radius: The plasma radius ``R`` in m.
        length: The driver's length in m.
        coil_turns: The coil's turns ``N``.
        coil_radius: The coil's radius ``b`` in m.
        coil_length: The coil's length ``l`` in m; the driver's if ``None``.
        frequency: The RF frequency in Hz.
        network_resistance: ``R_net`` in Ohm.
        momentum_transfer: Names of the reactions whose rates per electron make up
            ``nu_m`` (the elastic electron collisions).
        ion_mean_free_path: ``lambda_i`` in m.
        skin_density_factor: The electron density of the skin layer over the
            volume average.
        field_factor: The coil's inductance over the long-solenoid one;
            Nagaoka's coefficient if ``None``.
        coupling: ``"transformer"`` (:func:`transformer_impedance`) or
            ``"solenoid"``, the exact uniform column (:func:`solenoid_impedance`).
        model_options: Further fields of the :class:`GlobalModel` (gas
            temperature, wall reactions, ion wall products, ...).
    """

    species: Sequence[Species]
    reactions: Sequence[Reaction]
    fixed: Mapping[str, float]
    radius: float
    length: float
    coil_turns: float
    coil_radius: float
    frequency: float
    network_resistance: float
    momentum_transfer: Sequence[str]
    ion_mean_free_path: float
    skin_density_factor: float = 1.0
    coil_length: float | None = None
    field_factor: float | None = None
    coupling: Literal["transformer", "solenoid"] = "transformer"
    model_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the driver."""
        if min(self.radius, self.length, self.frequency, self.ion_mean_free_path) <= 0:
            raise ValueError(
                "radius, length, frequency, ion_mean_free_path must be > 0"
            )
        if self.coil_radius <= self.radius:
            raise ValueError("the coil must be outside the plasma")
        if self.coil_length is not None and not 0.0 < self.coil_length <= self.length:
            raise ValueError("coil_length must be in (0, length]")
        if self.network_resistance < 0.0 or self.skin_density_factor <= 0.0:
            raise ValueError("network_resistance >= 0 and skin_density_factor > 0")
        if self.coupling not in ("transformer", "solenoid"):
            raise ValueError('coupling must be "transformer" or "solenoid"')
        names = {reaction.name for reaction in self.reactions}
        missing = set(self.momentum_transfer) - names
        if missing:
            raise ValueError(f"momentum_transfer: unknown reactions {sorted(missing)}")

    @property
    def coil(self) -> Coil:
        """The coil."""
        return Coil(
            self.coil_turns,
            self.coil_radius,
            self.coil_length or self.length,
            self.field_factor,
        )

    @property
    def chamber(self) -> Chamber:
        """The driver's cylinder with its effective edge-to-centre ratio."""
        r, length, mfp = self.radius, self.length, self.ion_mean_free_path
        ends, side = 2.0 * np.pi * r**2, 2.0 * np.pi * r * length
        h_l = 0.86 / math.sqrt(3.0 + length / (2.0 * mfp))
        h_r = 0.8 / math.sqrt(4.0 + r / mfp)
        return Chamber(
            volume=np.pi * r**2 * length,
            wall_area=ends + side,
            h_factor=(h_l * ends + h_r * side) / (ends + side),
        )

    def model(self, power: float) -> GlobalModel:
        """The global model of the driver plasma absorbing ``power`` W."""
        return GlobalModel(
            species=self.species,
            reactions=self.reactions,
            chamber=self.chamber,
            power=power,
            fixed=self.fixed,
            **self.model_options,
        )

    def couple(self, plasma: GlobalState, power: float) -> DriverState:
        """The coil's view of a plasma absorbing ``power`` W."""
        ne = plasma.electron_density
        nu = sum(plasma.reaction_rates[name] for name in self.momentum_transfer) / ne
        n_skin = self.skin_density_factor * ne
        impedance = (
            transformer_impedance
            if self.coupling == "transformer"
            else solenoid_impedance
        )
        z_s = impedance(self.coil, self.radius, n_skin, nu, self.frequency)
        sigma = plasma_conductivity(n_skin, nu, self.frequency)
        return DriverState(
            plasma=plasma,
            power=power,
            frequency=self.frequency,
            collision_frequency=nu,
            coil_impedance=z_s,
            network_resistance=self.network_resistance,
            skin_depth=1.0 / _wavenumber(sigma, self.frequency).real,
        )

    def solve(
        self, power: float, guess: GlobalState | Mapping[str, float] | None = None
    ) -> DriverState:
        """The driver absorbing ``power`` W; ``guess`` starts the global model."""
        plasma = self.model(power).steady_state(guess)
        if not plasma.converged:
            warnings.warn(
                f"the driver's global model did not converge at {power:g} W",
                RuntimeWarning,
                stacklevel=2,
            )
        return self.couple(plasma, power)

    def scan(
        self,
        parameter: str,
        values: Sequence[Any],
        *,
        power: float | None = None,
        guess: GlobalState | Mapping[str, float] | None = None,
    ) -> list[DriverState]:
        """Driver states for every value of ``"power"`` or a field of the driver.

        Each global model starts from the last state (a continuation); a scan of a
        field needs the absorbed ``power``.
        """
        states: list[DriverState] = []
        previous = guess
        for value in values:
            if parameter == "power":
                driver, absorbed = self, float(value)
            else:
                if power is None:
                    raise ValueError("a scan of a driver field needs the power")
                driver = dataclasses.replace(self, **{parameter: value})
                absorbed = power
            state = driver.solve(absorbed, previous)
            states.append(state)
            previous = state.plasma
        return states


def illustrative_hydrogen_driver(**overrides: Any) -> RFDriver:
    """The BUG driver in hydrogen at 0.3 Pa, with representative rates, not data.

    Geometry of Zielke et al. (NIBS'22, arXiv:2211.10143): a plasma of 23.7 cm
    diameter and 17 cm length, six turns at 1 MHz, ``R_net = 0.6`` Ohm; the gas
    densities of their model, ``n_H2 = 3e19`` and ``n_H = 1.5e19`` m^-3. The coil
    radius (2 cm outside the plasma, for the wall and the Faraday shield), the gas
    temperature and the ion mean free path are assumptions.

    The cross sections are constants of plausible size above their thresholds
    (``CrossSection.constant``), round numbers, not hydrogen data: they let the
    model run until checked e-H2 and e-H data are supplied (LXCat, Janev et al.).
    It warns when called. ``overrides`` replace fields of the :class:`RFDriver`.
    """
    warnings.warn(
        "illustrative_hydrogen_driver uses placeholder cross sections, not data",
        UserWarning,
        stacklevel=2,
    )
    h2 = Species("H2", 2.016 * ATOMIC_MASS)
    h = Species("H", 1.008 * ATOMIC_MASS)
    h_ion = Species.atomic("H+", 1.008, 1)
    h2_ion = Species.atomic("H2+", 2.016, 1)
    h3_ion = Species.atomic("H3+", 3.024, 1)

    def electron_impact(reactants, products, sigma, threshold, name, **energy):
        return Reaction.from_cross_section(
            reactants,
            products,
            CrossSection.constant(sigma, threshold=threshold),
            ELECTRON_MASS,
            name=name,
            **energy,
        )

    reactions = [
        electron_impact(("e", "H2"), ("e", "e", "H2+"), 1.0e-20, 15.43,
                        "ionization of H2", electron_energy_loss=15.43),
        electron_impact(("e", "H"), ("e", "e", "H+"), 0.7e-20, 13.6,
                        "ionization of H", electron_energy_loss=13.6),
        electron_impact(("e", "H2"), ("e", "H", "H"), 0.6e-20, 10.0,
                        "dissociation of H2", electron_energy_loss=10.0,
                        gas_heating=2.0),
        electron_impact(("e", "H2"), ("e", "H2"), 0.8e-20, 12.0,
                        "electronic excitation of H2", electron_energy_loss=12.0),
        electron_impact(("e", "H2"), ("e", "H2"), 1.5e-20, 0.52,
                        "vibrational excitation of H2", electron_energy_loss=0.52),
        electron_impact(("e", "H"), ("e", "H"), 0.5e-20, 10.2,
                        "excitation of H", electron_energy_loss=10.2),
        Reaction.from_cross_section(("e", "H2"), ("e", "H2"),
                                    CrossSection.constant(1.0e-19), ELECTRON_MASS,
                                    elastic_mass=h2.mass, name="elastic e-H2"),
        Reaction.from_cross_section(("e", "H"), ("e", "H"),
                                    CrossSection.constant(1.0e-19), ELECTRON_MASS,
                                    elastic_mass=h.mass, name="elastic e-H"),
        Reaction(("H2+", "H2"), ("H3+", "H"), 2.0e-15, temperature="gas",
                 name="H3+ formation"),
        Reaction(("e", "H2+"), ("H", "H"), lambda t: 1.0e-14 / np.sqrt(t),
                 gas_heating=5.0, name="dissociative recombination of H2+"),
        Reaction(("e", "H3+"), ("H2", "H"), lambda t: 1.0e-14 / np.sqrt(t),
                 gas_heating=4.0, name="dissociative recombination of H3+"),
    ]  # fmt: skip
    defaults: dict[str, Any] = {
        "species": [ELECTRON, h2, h, h_ion, h2_ion, h3_ion],
        "reactions": reactions,
        "fixed": {"H2": 3.0e19, "H": 1.5e19},
        "radius": 0.1185,
        "length": 0.17,
        "coil_turns": 6,
        "coil_radius": 0.1385,
        "frequency": 1.0e6,
        "network_resistance": 0.6,
        "momentum_transfer": ("elastic e-H2", "elastic e-H"),
        "ion_mean_free_path": 0.03,
        "model_options": {
            "gas_temperature": 0.08,
            "ion_wall_products": {
                "H+": {"H": 1},
                "H2+": {"H2": 1},
                "H3+": {"H2": 1, "H": 1},
            },
        },
    }
    return RFDriver(**{**defaults, **overrides})
