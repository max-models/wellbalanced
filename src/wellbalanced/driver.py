"""A 0D model of an inductive RF driver: a global model and a transformer model.

The driver is a cylinder of radius ``R``, its plasma heated by an RF coil of ``N``
turns, radius ``b`` and length ``l`` around it. `RFDriver` couples two models:

- a `wellbalanced.GlobalModel` of the driver plasma, which fixes the electron
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
the cold, collisional one of `wellbalanced.spatial.inductive`,
``sigma = n_e e^2 / (m_e (nu_m + i omega))``: ``k / sigma`` is the surface impedance of
the skin layer, ``(1 + i) / (sigma delta)`` in the collisional limit (``nu_m >> omega``,
``delta = sqrt(2 / (omega mu0 sigma))``) and ``i omega mu0 c / omega_pe`` with the
electron-inertia skin depth ``c / omega_pe`` when ``nu_m << omega``, so both
limits, and the resistance ``R_p = Re Z_p`` and electron-inertia inductance
``Im Z_p / omega`` of the textbook model, come out of one expression. It is the
thin-skin limit of the uniform plasma column in a long solenoid, whose exact
impedance `solenoid_impedance` gives; `transformer_impedance` agrees
with it to order ``delta / R``.

A real driver coil is short (six turns over 17 cm around a 24 cm plasma): its
inductance is Nagaoka's coefficient ``K`` (`nagaoka_coefficient`) times the
long-solenoid one. It is modelled as the long solenoid of ``N sqrt(K)`` turns, which
has that inductance: ``L_11`` is multiplied by ``K`` and ``M`` by ``sqrt(K)``, so the
plasma's share of the impedance scales with ``K`` as well, as the exact solution's
(`solenoid_impedance`) does; ``field_factor`` overrides ``K``.

The plasma coupled to the coil is the length of the coil; the global model takes
the whole driver, which may be longer (a longer driver at the same coil spreads the
power over more plasma).

What a 0D model cannot give: the distance between the coil and an electromagnetic
shield and the mutual coupling of several drivers change the field around the
coil, not the plasma; they are in the 2D field of `wellbalanced.spatial.inductive`.
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
from methodcard import Rule, method
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

_PLASMA_RANGES = {
    "electron_density": (1.0e16, 1.0e18),
    "collision_frequency": (1.0e6, 1.0e8),
    "frequency": (0.5e6, 4.0e6),
}


def _sympy_conductivity():
    """``sigma`` in SymPy, with its symbols (for the rules)."""
    import sympy

    ne, nu, f = sympy.symbols(
        "electron_density collision_frequency frequency", positive=True
    )
    omega = 2 * sympy.pi * f
    sigma = (
        ne * sympy.Float(E) ** 2 / (sympy.Float(ELECTRON_MASS) * (nu + sympy.I * omega))
    )
    return sigma, omega


def _conductivity_rules() -> dict:
    """The SymPy rule of `plasma_conductivity` (for the tests)."""
    import sympy

    sigma, _ = _sympy_conductivity()
    return {
        "resistive": Rule(
            sympy.re(sigma),
            implementation=lambda **kwargs: plasma_conductivity(**kwargs).real,
            ranges=_PLASMA_RANGES,
            rtol=1e-12,
        )
    }


@method(
    equations={
        "conductivity": r"\sigma = \frac{n_e e^2}{m_e (\nu_m + i\omega)}",
        "resistive": r"\mathrm{Re}\,\sigma = \frac{n_e e^2 \nu_m}{m_e (\nu_m^2 + \omega^2)}",
    },
    assumptions=(
        r"cold electrons with friction, $m_e \dot u = -eE - m_e \nu_m u$; time dependence $e^{i\omega t}$",
    ),
    references=(
        "Lieberman and Lichtenberg, Principles of Plasma Discharges and Materials Processing, 2nd ed. (2005), sec. 12.2",
    ),
    symbolic=_conductivity_rules,
)
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


def _nagaoka_rules() -> dict:
    """The SymPy rule of `nagaoka_coefficient` (for the tests)."""
    import sympy

    d, length = sympy.symbols("diameter length", positive=True)
    m = d**2 / (d**2 + length**2)
    k, kp = sympy.sqrt(m), sympy.sqrt(1 - m)
    big_k, big_e = sympy.elliptic_k(m), sympy.elliptic_e(m)
    rule = 4 / (3 * sympy.pi * kp) * ((kp**2 / k**2) * (big_k - big_e) + big_e - k)
    ranges = {"diameter": (0.05, 0.5), "length": (0.05, 0.5)}
    return {"coefficient": Rule(rule, ranges=ranges, rtol=1e-10)}


@method(
    equations={
        "coefficient": r"K = \frac{4}{3\pi k'}\left(\frac{k'^2}{k^2}\big(\mathrm{K}(k) - \mathrm{E}(k)\big) + \mathrm{E}(k) - k\right), \qquad k^2 = \frac{d^2}{d^2 + l^2}, \quad k'^2 = 1 - k^2",
    },
    assumptions=(
        r"a single-layer solenoid of diameter $d$ and length $l$; $K = 1$ for a long coil, 0.6884 for $d = l$",
        r"$\mathrm{K}$, $\mathrm{E}$ the complete elliptic integrals (SciPy's `ellipk`, `ellipe` of the parameter $k^2$)",
    ),
    references=("Nagaoka's coefficient (no bibliographic details in the code)",),
    symbolic=_nagaoka_rules,
)
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


_COIL_RANGES = {
    "turns": (3.0, 10.0),
    "coil_radius": (0.12, 0.2),
    "coil_length": (0.1, 0.3),
    "plasma_radius": (0.05, 0.1),
}


def _transformer_rules() -> dict:
    """The SymPy rule of the resistance of `transformer_impedance` (for the tests)."""
    import sympy

    sigma, omega = _sympy_conductivity()
    n, b, length, r = sympy.symbols(
        "turns coil_radius coil_length plasma_radius", positive=True
    )
    mu0 = sympy.Float(MU_0)
    d = 2 * b
    m = d**2 / (d**2 + length**2)
    k, kp = sympy.sqrt(m), sympy.sqrt(1 - m)
    factor = (
        4
        / (3 * sympy.pi * kp)
        * (
            (kp**2 / k**2) * (sympy.elliptic_k(m) - sympy.elliptic_e(m))
            + sympy.elliptic_e(m)
            - k
        )
    )
    mutual = sympy.sqrt(factor) * mu0 * sympy.pi * r**2 * n / length
    l_2 = mu0 * sympy.pi * r**2 / length
    wavenumber = sympy.sqrt(sympy.I * omega * mu0 * sigma)
    z_p = 2 * sympy.pi * r / length * wavenumber / sigma
    rule = sympy.re(omega**2 * mutual**2 / (z_p + sympy.I * omega * l_2))

    def resistance(turns, coil_radius, coil_length, plasma_radius, **plasma):
        coil = Coil(turns, coil_radius, coil_length)
        return transformer_impedance(coil, plasma_radius, **plasma).real

    return {
        "resistance": Rule(
            rule,
            implementation=resistance,
            ranges={**_COIL_RANGES, **_PLASMA_RANGES},
            rtol=1e-9,
        )
    }


@method(
    equations={
        "impedance": r"Z_s = i\omega L_{11} + \frac{\omega^2 M^2}{Z_p + i\omega L_2}",
        "inductances": r"L_{11} = K \frac{\mu_0 \pi b^2 N^2}{l}, \qquad M = \sqrt{K}\, \frac{\mu_0 \pi R^2 N}{l}, \qquad L_2 = \frac{\mu_0 \pi R^2}{l}",
        "skin_layer": r"Z_p = \frac{2\pi R}{l}\, \frac{k}{\sigma}, \qquad k = \sqrt{i\omega\mu_0\sigma}\ (\mathrm{Re}\,k > 0)",
        "resistance": r"R_s = \mathrm{Re}\, Z_s",
    },
    assumptions=(
        r"a thin skin layer: the thin-skin limit of `solenoid_impedance`, error $O(\delta/R)$; for $\delta \gg R$ its $R_s$ falls like $\sqrt\sigma$ instead of $\sigma$",
        r"the short coil as the long solenoid of $N\sqrt{K}$ turns ($K$ Nagaoka's coefficient or `field_factor`)",
        r"$\nu_m \gg \omega$: $k/\sigma = (1 + i)/(\sigma\delta)$; $\nu_m \ll \omega$: $k/\sigma = i\omega\mu_0 c/\omega_{pe}$; with $n_e \le 0$ the empty coil $i\omega L_{11}$",
    ),
    references=(
        "[Piejak, Godyak and Alexandrovich, Plasma Sources Sci. Technol. 1, 179 (1992)](https://doi.org/10.1088/0963-0252/1/3/006)",
        "Lieberman and Lichtenberg, Principles of Plasma Discharges and Materials Processing, 2nd ed. (2005), sec. 12.2",
    ),
    components=(plasma_conductivity, nagaoka_coefficient),
    symbolic=_transformer_rules,
)
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
    falls like ``sqrt(sigma)`` instead of ``sigma``; `solenoid_impedance` holds
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


def _solenoid_rules() -> dict:
    """The SymPy rule of the resistance of `solenoid_impedance` (for the tests)."""
    import sympy

    sigma, omega = _sympy_conductivity()
    n, b, length, r = sympy.symbols(
        "turns coil_radius coil_length plasma_radius", positive=True
    )
    mu0 = sympy.Float(MU_0)
    q = sympy.sqrt(-sympy.I * omega * mu0 * sigma)
    ratio = sympy.besselj(1, q * r) / sympy.besselj(0, q * r)
    gap = sympy.pi * (b**2 - r**2)
    rule = sympy.re(
        sympy.I * omega * mu0 * n**2 / length * (gap + 2 * sympy.pi * r * ratio / q)
    )

    # the gap outside the plasma is reactive only: the resistance has no coil radius
    def resistance(turns, coil_length, plasma_radius, coil_radius=0.2, **plasma):
        coil = Coil(turns, coil_radius, coil_length, field_factor=1.0)
        return solenoid_impedance(coil, plasma_radius, **plasma).real

    return {
        "resistance": Rule(
            rule,
            implementation=resistance,
            ranges={**_COIL_RANGES, **_PLASMA_RANGES},
            rtol=1e-9,
        )
    }


@method(
    equations={
        "field": r"B_z = B_0 \frac{J_0(q r)}{J_0(q R)}, \qquad q^2 = -i\omega\mu_0\sigma",
        "impedance": r"Z = K\, \frac{i\omega\mu_0 N^2}{l}\left(\pi (b^2 - R^2) + \frac{2\pi R\, J_1(qR)}{q\, J_0(qR)}\right)",
        "resistance": r"R_s = \mathrm{Re}\, Z",
    },
    assumptions=(
        r"a uniform plasma column of radius $R$ in a long solenoid of radius $b > R$: exact for any skin depth",
        r"$J_1/J_0$ with SciPy's `jve` (both scaled by $e^{-|\mathrm{Im}\,x|}$), so the ratio does not overflow",
    ),
    references=(
        "Lieberman and Lichtenberg, Principles of Plasma Discharges and Materials Processing, 2nd ed. (2005), sec. 12.2",
    ),
    components=(plasma_conductivity,),
    symbolic=_solenoid_rules,
)
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
    `transformer_impedance`.
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


def _driver_rules() -> dict:
    """The SymPy rule of the edge-to-centre ratio of `RFDriver.chamber` (for the tests)."""
    import sympy

    r, length, mfp = sympy.symbols("radius length ion_mean_free_path", positive=True)
    ends, side = 2 * sympy.pi * r**2, 2 * sympy.pi * r * length
    h_l = sympy.Float("0.86") / sympy.sqrt(3 + length / (2 * mfp))
    h_r = sympy.Float("0.8") / sympy.sqrt(4 + r / mfp)
    rule = (h_l * ends + h_r * side) / (ends + side)

    def h_factor(radius, length, ion_mean_free_path):
        driver = RFDriver(
            species=(),
            reactions=(),
            fixed={},
            radius=radius,
            length=length,
            coil_turns=6,
            coil_radius=2.0 * radius,
            frequency=1.0e6,
            network_resistance=0.0,
            momentum_transfer=(),
            ion_mean_free_path=ion_mean_free_path,
        )
        return driver.chamber.h_factor

    ranges = {
        "radius": (0.05, 0.2),
        "length": (0.1, 0.4),
        "ion_mean_free_path": (0.005, 0.1),
    }
    return {"h_factor": Rule(rule, implementation=h_factor, ranges=ranges, rtol=1e-12)}


@method(
    equations={
        "h_factor": r"h = \frac{h_L A_\mathrm{ends} + h_R A_\mathrm{side}}{A_\mathrm{ends} + A_\mathrm{side}}, \qquad h_L = \frac{0.86}{\sqrt{3 + l_d/(2\lambda_i)}}, \qquad h_R = \frac{0.8}{\sqrt{4 + R/\lambda_i}}",
        "collision_frequency": r"\nu_m = \frac{\sum_\mathrm{momentum\ transfer} R}{n_e}, \qquad n_\mathrm{skin} = f_\mathrm{skin}\, n_e",
        "current": r"I = \sqrt{\frac{2P}{R_s}}, \qquad \eta = \frac{R_s}{R_s + R_\mathrm{net}}, \qquad P_\mathrm{RF} = \frac{P}{\eta}, \qquad V_\mathrm{coil} = |Z_s|\, I",
        "skin_depth": r"\delta = \frac{1}{\mathrm{Re}\, k}",
    },
    assumptions=(
        r"the global model takes the whole driver; the coupled plasma is the coil's length",
        r"no stochastic (non-local) heating, no shield distance or mutual coupling of drivers: those are in the 2D field",
        r"the low-pressure edge-to-centre ratios for the ion mean free path $\lambda_i$",
    ),
    references=(
        "Lieberman and Lichtenberg, Principles of Plasma Discharges and Materials Processing, 2nd ed. (2005), sec. 10.2 and 12.2",
    ),
    components=(GlobalModel, transformer_impedance, solenoid_impedance),
    symbolic=_driver_rules,
)
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
        coupling: ``"transformer"`` (`transformer_impedance`) or
            ``"solenoid"``, the exact uniform column (`solenoid_impedance`).
        model_options: Further fields of the `GlobalModel` (gas
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
    It warns when called. ``overrides`` replace fields of the `RFDriver`.
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
