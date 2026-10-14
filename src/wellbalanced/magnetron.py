"""A global model of the ionization region of a (HiPIMS) magnetron discharge.

The model balances, averaged over the volume ``V`` of the ionization region in
front of the racetrack of a sputtering magnetron, five species: electrons,
argon ``Ar`` and ``Ar+``, and the sputtered target metal ``M`` and ``M+``. It is
a :class:`~wellbalanced.GlobalModel` with the closures of the pulsed-magnetron
global models (Raadu et al., *Plasma Sources Sci. Technol.* 20, 065007, 2011;
Huo et al., *J. Phys. D* 46, 045204, 2013: the ionization region model):

- **Ionization** of Ar and M by electron impact, charge exchange ``Ar+ + M``,
  elastic and excitation losses on Ar.
- **Ion losses** at the Bohm speed through the target (area ``A_t``) and the
  other boundaries of the region, times an edge-to-centre ratio ``h``. The
  ions that reach the target sputter it, ``Y_Ar`` atoms per ``Ar+`` and
  ``Y_M`` per ``M+`` (self-sputtering): a :class:`~wellbalanced.VolumeSource`
  per unit ion density, ``Y h u_B(T_e) A_t / V``. An ``Ar+`` that reaches a wall
  returns as ``Ar``; an ``M+`` that does is lost (deposited).
- **Return of the metal**: the sputtered atoms leave the region at a quarter of
  their mean speed through the whole boundary (they stick, probability 1), unless
  they are ionized first. The ionized fraction that returns to the target, and
  so the self-sputtering, is the share of the ion loss area that is the target;
  there is no separate return probability.
- **Gas**: Ar is fed and flows out on the time ``refill_time``, so it relaxes
  to its filling density. With ``wind_efficiency`` ``eta``, every sputtered atom
  also pushes ``eta`` Ar atoms out of the region (the sputter wind), the
  rarefaction that limits the discharge current of a HiPIMS pulse.
- **Heating**: a share ``heating_fraction`` of the energy ``e V_D`` that the
  secondary electrons (``gamma``) gain in the cathode fall goes into the plasma
  electrons, ``P = f gamma / (1 + gamma) I_D V_D``: the prescribed discharge
  current ``I_D(t)`` and voltage ``V_D(t)`` of an experiment (a pulse, or the dc
  limit) give the absorbed power, and the model returns the densities, the
  electron temperature, the ion current to the target it implies and the metal
  fraction of the ions. A drive by the voltage alone has no steady state: the
  absorbed power and the losses both scale with the density.

Everything not listed here is missing: Ar metastables and Penning ionization,
multiply charged metal ions, excitation and elastic scattering on the metal,
gas heating and the energy of the sputtered atoms (they leave at the gas
temperature), the radial structure of the plasma and the ionization zones
(spokes) that make the discharge non-uniform.

**Data.** plasmacoll ships no argon or metal cross sections. The argon rates are
the fits of Lieberman and Lichtenberg, *Principles of Plasma Discharges and
Materials Processing*, 2nd ed., ch. 3; the metal ionization rate is the Lotz
approximation (Lotz, *Z. Phys.* 206, 205, 1967) averaged over a Maxwellian in
closed form. Every other number (sputter and secondary electron yields, the
charge-exchange rate, the geometry, the heating fraction, the wind efficiency,
the refill time, the seed source) is a **placeholder** of the right order of
magnitude, to be replaced by data for the target and the device: yields from
TRIM/SRIM or measured tables at the ion energy of the cathode fall, secondary
electron yields from the literature on ion-induced emission for the target
material, geometry from the magnetron. A :class:`Magnetron` with a placeholder
material warns.

All temperatures are in eV, densities in m^-3, areas in m^2, currents in A and
voltages in V.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from plasmacoll.constants import BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy.optimize import brentq
from scipy.special import exp1

from wellbalanced.datasets import stack, variable
from wellbalanced.global_model import (
    Chamber,
    GlobalModel,
    GlobalState,
    VolumeSource,
    WallReaction,
)
from wellbalanced.reactions import Reaction
from wellbalanced.species import ELECTRON, Species

__all__ = [
    "COPPER",
    "MATERIALS",
    "TITANIUM",
    "Magnetron",
    "TargetMaterial",
    "argon_elastic",
    "argon_excitation",
    "argon_ionization",
    "lotz_ionization",
    "pulse",
]

E = ELEMENTARY_CHARGE

#: Argon ionization and excitation energies in eV.
ARGON_IONIZATION_ENERGY = 15.76
ARGON_EXCITATION_ENERGY = 11.56

#: The constant of the Lotz formula, ``a = 4.5e-14 cm^2 eV^2``, in m^2 eV^2.
LOTZ_CONSTANT = 4.5e-18


def _fit_range(te: np.ndarray | float) -> np.ndarray | float:
    """``T_e`` limited to 0.01-1000 eV, where the fits are evaluated (they overflow beyond)."""
    return np.clip(te, 1.0e-2, 1.0e3)


def argon_ionization(te: np.ndarray | float) -> np.ndarray | float:
    """``K_iz = 2.34e-14 T_e^0.59 exp(-17.44 / T_e)`` m^3/s (Lieberman and Lichtenberg)."""
    te = _fit_range(te)
    return 2.34e-14 * te**0.59 * np.exp(-17.44 / te)


def argon_excitation(te: np.ndarray | float) -> np.ndarray | float:
    """``K_ex = 2.48e-14 T_e^0.33 exp(-12.78 / T_e)`` m^3/s (Lieberman and Lichtenberg)."""
    te = _fit_range(te)
    return 2.48e-14 * te**0.33 * np.exp(-12.78 / te)


def argon_elastic(te: np.ndarray | float) -> np.ndarray | float:
    """The elastic rate of electrons on argon in m^3/s (Lieberman and Lichtenberg's fit).

    ``K_el = 2.336e-14 T_e^1.609 exp(0.0618 ln(T_e)^2 - 0.1171 ln(T_e)^3)``.
    """
    te = _fit_range(te)
    ln = np.log(te)
    return 2.336e-14 * te**1.609 * np.exp(0.0618 * ln**2 - 0.1171 * ln**3)


def lotz_ionization(
    ionization_energy: float, electrons: int
) -> Callable[[np.ndarray | float], np.ndarray | float]:
    """The Maxwellian rate of the Lotz ionization cross section, in m^3/s.

    The cross section ``sigma = xi a ln(E / I) / (E I)`` above the threshold
    ``I`` (eV; ``a = 4.5e-14 cm^2 eV^2``, ``xi`` the electrons of the outer
    shell) averaged over a Maxwellian of temperature ``T`` gives, in closed form,

        K = sqrt(8 e / (pi m_e)) xi a E_1(I / T) / (I sqrt(T)),

    with ``E_1`` the exponential integral. An approximation good to a factor of
    about two, to replace by measured or recommended data for a given metal.
    """
    prefactor = (
        np.sqrt(8.0 * E / (np.pi * ELECTRON_MASS))
        * electrons
        * LOTZ_CONSTANT
        / ionization_energy
    )

    def rate(te: np.ndarray | float) -> np.ndarray | float:
        te = _fit_range(te)
        return prefactor * exp1(ionization_energy / te) / np.sqrt(te)

    return rate


@dataclass(frozen=True)
class TargetMaterial:
    """The sputtered metal and the response of the target to the ions.

    Attributes:
        name: The element, e.g. ``"Ti"``; the species are ``name`` and ``name+``.
        mass_u: The atomic mass in u.
        ionization_energy: The first ionization energy in eV.
        outer_electrons: The electrons of the outer shell in the Lotz formula.
        argon_yield: Atoms sputtered per ``Ar+`` at the cathode-fall energy.
        self_yield: Atoms sputtered per metal ion (self-sputtering).
        argon_secondary: Secondary electrons emitted per ``Ar+``.
        self_secondary: Secondary electrons emitted per metal ion.
        charge_exchange: Rate of ``Ar+ + M -> Ar + M+`` in m^3/s.
        verified: ``False`` while the yields and rates are placeholders.
    """

    name: str
    mass_u: float
    ionization_energy: float
    outer_electrons: int
    argon_yield: float
    self_yield: float
    argon_secondary: float
    self_secondary: float
    charge_exchange: float
    verified: bool = False

    def __post_init__(self) -> None:
        """Validate the material."""
        if self.mass_u <= 0.0 or self.ionization_energy <= 0.0:
            raise ValueError("mass_u and ionization_energy must be > 0")
        if min(self.argon_yield, self.self_yield) < 0.0:
            raise ValueError("a sputter yield must be >= 0")
        if min(self.argon_secondary, self.self_secondary) < 0.0:
            raise ValueError("a secondary electron yield must be >= 0")


def _material(
    name: str,
    mass_u: float,
    ionization_energy: float,
    outer_electrons: int,
    argon_yield: float,
    self_yield: float,
    argon_secondary: float = 0.1,
    self_secondary: float = 0.05,
) -> TargetMaterial:
    """A target material of the table below."""
    return TargetMaterial(
        name=name,
        mass_u=mass_u,
        ionization_energy=ionization_energy,
        outer_electrons=outer_electrons,
        argon_yield=argon_yield,
        self_yield=self_yield,
        argon_secondary=argon_secondary,
        self_secondary=self_secondary,
        charge_exchange=1.0e-15,
    )


#: The targets of the spoke literature. The atomic masses and first ionization
#: energies (eV) are data. Everything else is a PLACEHOLDER of the right order and
#: not checked against a source: the outer-shell electron count of the Lotz
#: formula; the Ar and self-sputter yields at a few hundred eV (to take from
#: Ziegler's SRIM or Yamamura-Tawara tables at the ion energy of the cathode
#: fall: the review of Hecimovic and von Keudell, *J. Phys. D* 51, 453001, 2018,
#: tabulates them for these metals, but this table was not copied from it);
#: the ion-induced secondary electron yields (Depla et al.'s measurements for
#: metals and oxides are the source to check); and the charge-exchange rate.
MATERIALS: dict[str, TargetMaterial] = {
    m.name: m
    for m in (
        _material("Al", 26.982, 5.986, 3, 0.65, 0.55),
        _material("Cr", 51.996, 6.767, 1, 0.9, 0.8),
        _material("Cu", 63.546, 7.726, 1, 1.0, 1.0),
        _material("Ti", 47.867, 6.828, 2, 0.4, 0.3),
        _material("Au", 196.967, 9.226, 1, 2.4, 2.5),
        _material("Mo", 95.95, 7.092, 1, 0.7, 0.8),
        _material("Nb", 92.906, 6.759, 1, 0.5, 0.6),
    )
}

#: Titanium and copper, the defaults of the examples (placeholder yields, see :data:`MATERIALS`).
TITANIUM = MATERIALS["Ti"]
COPPER = MATERIALS["Cu"]


@dataclass(frozen=True)
class Magnetron:
    """The ionization region of a magnetron: geometry, gas and target; see the module docstring.

    The defaults are placeholders of the order of a 3 inch magnetron, not a
    particular device.

    Attributes:
        material: The target material.
        target_area: The area of the racetrack the ions hit, in m^2.
        region_height: The height of the ionization region above it in m; the
            volume is ``target_area * region_height``.
        side_area: The other boundaries ions leave the region through, in m^2.
        h_factor: The edge-to-centre density ratio at the boundaries.
        pressure: The argon filling pressure in Pa, at ``gas_temperature``.
        gas_temperature: The gas temperature in K.
        refill_time: The time in s in which the argon of the region is replaced.
        wind_efficiency: Ar atoms pushed out per sputtered atom (0: no sputter wind).
        heating_fraction: The share of the secondary electron energy the plasma
            electrons keep: at most 1 if the secondary electron yield is right;
            a larger value, as :meth:`calibrate_heating` may find, means the yield
            of the placeholder material is too low (only the product matters).
        seed: A pair source in m^-3 s^-1 that stands for the residual plasma
            and preionization at the start of a pulse; it keeps the density from
            decaying to zero in the afterglow.
    """

    material: TargetMaterial = TITANIUM
    target_area: float = 1.0e-3
    region_height: float = 0.02
    side_area: float = 1.0e-4
    h_factor: float = 0.5
    pressure: float = 1.0
    gas_temperature: float = 300.0
    refill_time: float = 1.0e-4
    wind_efficiency: float = 0.0
    heating_fraction: float = 1.0
    seed: float = 1.0e19

    def __post_init__(self) -> None:
        """Validate the magnetron, and warn about placeholder data."""
        positive = (
            self.target_area,
            self.region_height,
            self.pressure,
            self.gas_temperature,
            self.refill_time,
        )
        if min(positive) <= 0.0 or self.side_area < 0.0:
            raise ValueError("areas, height, pressure, temperature and times > 0")
        if not 0.0 < self.h_factor <= 1.0:
            raise ValueError("h_factor must be in (0, 1]")
        if self.heating_fraction < 0.0:
            raise ValueError("heating_fraction must be >= 0")
        if self.wind_efficiency < 0.0 or self.seed < 0.0:
            raise ValueError("wind_efficiency and seed must be >= 0")
        if not self.material.verified:
            warnings.warn(
                f"{self.material.name}: the sputter and secondary electron yields and "
                "the charge-exchange rate are placeholders, not data",
                UserWarning,
                stacklevel=2,
            )

    # -- the plasma -------------------------------------------------------------------

    @property
    def volume(self) -> float:
        """The volume of the ionization region in m^3."""
        return self.target_area * self.region_height

    @property
    def gas_density(self) -> float:
        """The argon filling density in m^-3."""
        return self.pressure / (BOLTZMANN * self.gas_temperature)

    @property
    def species(self) -> dict[str, Species]:
        """The species by role: ``e``, ``gas``, ``gas_ion``, ``metal``, ``metal_ion``."""
        m = self.material
        return {
            "e": ELECTRON,
            "gas": Species.atomic("Ar", 39.948),
            "gas_ion": Species.atomic("Ar+", 39.948, 1),
            "metal": Species.atomic(m.name, m.mass_u),
            "metal_ion": Species.atomic(f"{m.name}+", m.mass_u, 1),
        }

    @property
    def chamber(self) -> Chamber:
        """The region as a chamber: ions leave through the target and the sides."""
        wall = 2.0 * self.target_area + self.side_area
        return Chamber(
            volume=self.volume,
            wall_area=wall,
            ion_loss_area=self.target_area + self.side_area,
            h_factor=self.h_factor,
        )

    def bohm_flux_coefficient(self, te: float, ion: str) -> float:
        """The loss rate of the ion named ``ion`` to the target per unit density, in 1/s.

        ``h u_B(T_e) A_t / V``.
        """
        mass = next(s.mass for s in self.species.values() if s.name == ion)
        return self.h_factor * np.sqrt(E * te / mass) * self.target_area / self.volume

    def reactions(self) -> list[Reaction]:
        """The ionization, excitation, elastic and charge-exchange reactions."""
        s = self.species
        m = self.material
        gas, metal = s["gas"].name, s["metal"].name
        metal_ion = s["metal_ion"].name
        return [
            Reaction(
                ("e", gas),
                ("e", "e", s["gas_ion"].name),
                argon_ionization,
                electron_energy_loss=ARGON_IONIZATION_ENERGY,
                name="Ar ionization",
            ),
            Reaction(
                ("e", gas),
                ("e", gas),
                argon_excitation,
                electron_energy_loss=ARGON_EXCITATION_ENERGY,
                name="Ar excitation",
            ),
            Reaction(
                ("e", gas),
                ("e", gas),
                argon_elastic,
                elastic_mass=s["gas"].mass,
                name="Ar elastic",
            ),
            Reaction(
                ("e", metal),
                ("e", "e", metal_ion),
                lotz_ionization(m.ionization_energy, m.outer_electrons),
                electron_energy_loss=m.ionization_energy,
                name=f"{m.name} ionization",
            ),
            Reaction(
                (s["gas_ion"].name, metal),
                (gas, metal_ion),
                m.charge_exchange,
                temperature="gas",
                name="charge exchange",
            ),
            Reaction(
                (gas,),
                (),
                1.0 / self.refill_time,
                temperature="gas",
                name="Ar flow out",
            ),
        ]

    def sources(self) -> list[VolumeSource]:
        """Sputtering and the sputter wind by the ion flux to the target, and the seed."""
        s = self.species
        m = self.material
        sources = []
        for ion, sputter in (("gas_ion", m.argon_yield), ("metal_ion", m.self_yield)):
            name = s[ion].name

            def flux(te: float, _tg: float, name: str = name) -> float:
                return self.bohm_flux_coefficient(te, name)

            def yielded(
                te: float, tg: float, flux=flux, sputter: float = sputter
            ) -> float:
                return sputter * flux(te, tg)

            def wind(te: float, tg: float, yielded=yielded) -> float:
                return self.wind_efficiency * yielded(te, tg)

            sources.append(
                VolumeSource(
                    {s["metal"].name: 1},
                    yielded,
                    per=(name,),
                    name=f"sputtering by {name}",
                )
            )
            if self.wind_efficiency:
                sources.append(
                    VolumeSource(
                        {s["gas"].name: -1},
                        wind,
                        per=(name,),
                        name=f"sputter wind of {name}",
                    )
                )
        if self.seed:
            sources.append(
                VolumeSource(
                    {s["gas_ion"].name: 1, "e": 1}, self.seed, name="preionization"
                )
            )
        return sources

    def heating_power(self, current: float, voltage: float) -> float:
        """The power the plasma electrons absorb at a discharge current and voltage, in W.

        ``P = f gamma / (1 + gamma) I_D V_D``, with ``gamma`` the secondary
        electron yield of the argon ions.
        """
        gamma = self.material.argon_secondary
        return self.heating_fraction * gamma / (1.0 + gamma) * current * voltage

    def model(
        self,
        current: float | Callable[[float], float],
        voltage: float | Callable[[float], float],
    ) -> GlobalModel:
        """The global model driven by the discharge current ``I_D(t)`` and voltage ``V_D(t)``.

        Each is a number (dc) or a function of the time in s, e.g. from
        :func:`pulse`.
        """

        def absorbed(t: float) -> float:
            i = current(t) if callable(current) else current
            v = voltage(t) if callable(voltage) else voltage
            return self.heating_power(i, v)

        s = self.species
        walls = [WallReaction(s["metal"].name, 1.0)]
        return GlobalModel(
            species=list(s.values()),
            reactions=self.reactions(),
            chamber=self.chamber,
            power=absorbed,
            feed={s["gas"].name: self.gas_density * self.volume / self.refill_time},
            sources=self.sources(),
            wall_reactions=walls,
            ion_wall_products={s["gas_ion"].name: {s["gas"].name: 1}},
            gas_temperature=BOLTZMANN * self.gas_temperature / E,
            thermalize_electrons=True,
        )

    def guess(self, ion_density: float = 1.0e17) -> dict[str, float]:
        """A starting state: the filling gas and a plasma of ``ion_density``."""
        s = self.species
        return {
            s["gas"].name: self.gas_density,
            s["gas_ion"].name: ion_density,
            s["metal"].name: 1.0e-3 * self.gas_density,
            s["metal_ion"].name: 0.1 * ion_density,
            "Te": 3.0,
        }

    def steady_state(
        self,
        current: float,
        voltage: float,
        initial: Mapping[str, float] | GlobalState | None = None,
    ) -> GlobalState:
        """The dc state at a discharge current and voltage."""
        model = self.model(current, voltage)
        with np.errstate(over="ignore", invalid="ignore"):  # trial states of the solver
            return model.steady_state(self.guess() if initial is None else initial)

    def calibrate_heating(self, current: float, voltage: float) -> Magnetron:
        """A copy whose heating fraction makes the dc discharge current match ``current``.

        The share of the secondary electron energy that heats the plasma is the
        least known closure. The model's own ion current to the target,
        ``I_D`` of :meth:`target_currents`, rises with it; this fits it to one
        measured point so that the others are predictions. A value above 1
        means the secondary electron yield is too low; it raises a
        ``ValueError`` if no value up to 10 does, which says that the sputter
        yields or the geometry are off.
        """

        def mismatch(fraction: float) -> float:
            magnetron = dataclasses.replace(self, heating_fraction=fraction)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                state = magnetron.steady_state(current, voltage)
            return magnetron.target_currents(state)["I_D"] / current - 1.0

        lo, hi = 1.0e-3, 10.0
        if mismatch(lo) * mismatch(hi) > 0.0:
            raise ValueError(
                "no heating fraction up to 10 reproduces the current: "
                "check the yields and the geometry"
            )
        fraction = brentq(mismatch, lo, hi, xtol=1.0e-6)
        return dataclasses.replace(self, heating_fraction=float(fraction))

    # -- the target -------------------------------------------------------------------

    def target_currents(self, state: GlobalState) -> dict[str, float]:
        """The currents and fluxes at the target for a state.

        Returns the ion currents ``I_<ion>`` to the target in A, the discharge
        current ``I_D`` including the secondary electrons, the sputtered atom
        flux ``sputtered`` in 1/s, the ``metal_ion_fraction`` of the ion flux to
        the target and the ``metal_ion_density_fraction`` of the ion density.
        """
        s = self.species
        m = self.material
        te = state.electron_temperature
        currents = {}
        for role, secondary in (
            ("gas_ion", m.argon_secondary),
            ("metal_ion", m.self_secondary),
        ):
            name = s[role].name
            rate = state.densities[name] * self.bohm_flux_coefficient(te, name)
            currents[f"I_{name}"] = E * rate * self.volume
            currents[f"I_D_{name}"] = (1.0 + secondary) * currents[f"I_{name}"]
        gas_ion, metal_ion = s["gas_ion"].name, s["metal_ion"].name
        sputtered = (
            m.argon_yield * currents[f"I_{gas_ion}"]
            + m.self_yield * currents[f"I_{metal_ion}"]
        ) / E
        ions = state.densities[gas_ion] + state.densities[metal_ion]
        ion_current = currents[f"I_{gas_ion}"] + currents[f"I_{metal_ion}"]
        return {
            **currents,
            "I_D": currents[f"I_D_{gas_ion}"] + currents[f"I_D_{metal_ion}"],
            "sputtered": sputtered,
            "metal_ion_fraction": currents[f"I_{metal_ion}"] / ion_current,
            "metal_ion_density_fraction": state.densities[metal_ion] / ions,
        }

    def diagnostics(self, state: GlobalState) -> dict[str, float]:
        """The quantities that decide the spoke regime, for a state.

        - ``rarefaction``: ``n_Ar`` in the region over the filling density
          (1: none; the sputter wind and the ionization of the gas lower it);
        - ``metal_to_argon``: the neutral metal over the neutral argon density;
        - ``metal_ion_fraction``: the metal share of the ion flux to the target;
        - ``self_sputtering_share``: the share of the sputtered flux that the
          metal ions make (1: the discharge runs on the metal alone);
        - ``electron_density`` in m^-3, and the electron temperature in eV;
        - ``discharge_current``: the model's ``I_D`` in A, to compare with the
          prescribed one.
        """
        s = self.species
        m = self.material
        currents = self.target_currents(state)
        gas_ion, metal_ion = s["gas_ion"].name, s["metal_ion"].name
        from_argon = m.argon_yield * currents[f"I_{gas_ion}"]
        from_metal = m.self_yield * currents[f"I_{metal_ion}"]
        density = state.densities
        return {
            "rarefaction": density[s["gas"].name] / self.gas_density,
            "metal_to_argon": density[s["metal"].name] / density[s["gas"].name],
            "metal_ion_fraction": currents["metal_ion_fraction"],
            "self_sputtering_share": from_metal / (from_argon + from_metal),
            "electron_density": density["e"],
            "electron_temperature": state.electron_temperature,
            "discharge_current": currents["I_D"],
        }

    def metal_ionization(self, state: GlobalState) -> float:
        """The ionized fraction of the metal in the region, ``n_M+ / (n_M + n_M+)``."""
        s = self.species
        metal, metal_ion = s["metal"].name, s["metal_ion"].name
        total = state.densities[metal] + state.densities[metal_ion]
        return state.densities[metal_ion] / total

    # -- pulses -----------------------------------------------------------------------

    def pulse(
        self,
        current: float,
        voltage: float,
        on_time: float,
        period: float,
        *,
        periods: int = 20,
        points: int = 200,
        tolerance: float = 1.0e-3,
        initial: Mapping[str, float] | GlobalState | None = None,
    ) -> xr.Dataset:
        """The periodic state of a rectangular pulse, as a dataset over the time ``t`` in a period.

        The discharge current and voltage are ``current`` and ``voltage`` for
        ``on_time`` seconds of every ``period``, and zero in between. Periods
        are run until the densities at the start of a period change by less than
        ``tolerance`` (relative, in the log) from one period to the next, or
        ``periods`` are done; ``attrs["periodicity_error"]`` is the last change
        and ``attrs["periods"]`` how many were run. The variables are the
        densities, the electron temperature, the model's currents at the target
        (``I_D``, ``I_<ion>``), the sputtered flux and the metal ion fraction.
        """
        if not 0.0 < on_time < period:
            raise ValueError("on_time must be in (0, period)")
        drive_current = pulse(current, on_time, period)
        drive_voltage = pulse(voltage, on_time, period)
        model = self.model(drive_current, drive_voltage)
        times = np.linspace(0.0, period, points)
        state = self.guess(1.0e14) if initial is None else initial
        error = np.inf
        run = 0
        states: list[GlobalState] = []
        for run in range(1, periods + 1):
            # the state at the start of the period is the state at its end
            shifted = (run - 1) * period + times
            with np.errstate(over="ignore", invalid="ignore"):
                states = model.evolve(
                    shifted, state, breakpoints=[shifted[0] + on_time]
                )
            previous = state if isinstance(state, GlobalState) else None
            state = states[-1]
            if previous is not None:
                error = max(
                    abs(np.log(state.densities[k] / previous.densities[k]))
                    for k in state.densities
                )
                if error < tolerance:
                    break
        data = self._series(states, times)
        data.attrs.update(
            periodicity_error=float(error), periods=run, on_time=on_time, period=period
        )
        return data

    def _series(self, states: list[GlobalState], times: np.ndarray) -> xr.Dataset:
        """The time series of a list of states as a dataset."""
        data = stack(states, "t", times, long_name="time in the period", units="s")
        columns = [self.target_currents(state) for state in states]
        for key, long_name, units in (
            ("I_D", "discharge current", "A"),
            ("sputtered", "sputtered atoms", "s^-1"),
            ("metal_ion_fraction", "metal fraction of the ion flux", ""),
        ):
            data[key] = variable("t", [c[key] for c in columns], long_name, units)
        diagnostics = [self.diagnostics(state) for state in states]
        for key, long_name in (
            ("rarefaction", "argon density over the filling density"),
            ("self_sputtering_share", "self-sputtering share of the sputtered flux"),
            ("metal_to_argon", "neutral metal over argon density"),
        ):
            data[key] = variable("t", [d[key] for d in diagnostics], long_name)
        data["metal_ionization"] = variable(
            "t",
            [self.metal_ionization(state) for state in states],
            "ionized fraction of the metal",
        )
        return data

    def current_scan(
        self,
        currents: Sequence[float],
        voltage: float | Callable[[float], float],
    ) -> xr.Dataset:
        """The dc states against the discharge current, with the spoke diagnostics.

        ``voltage`` is a number or a function of the current ``V_D(I_D)``, e.g.
        the measured characteristic. Every state starts from the last (a
        continuation); the dataset has the densities (``species``), the
        :meth:`diagnostics` and the model's ``I_D`` over ``current``.
        """
        states: list[GlobalState] = []
        previous: GlobalState | None = None
        for current in currents:
            v = voltage(current) if callable(voltage) else voltage
            state = self.steady_state(current, v, previous)
            states.append(state)
            previous = state
        data = stack(states, "current", currents, long_name="discharge current", units="A")
        columns = [self.diagnostics(state) for state in states]
        for key, long_name, units in (
            ("rarefaction", "argon density over the filling density", ""),
            ("metal_to_argon", "neutral metal over argon density", ""),
            ("metal_ion_fraction", "metal fraction of the ion flux", ""),
            ("self_sputtering_share", "self-sputtering share of the sputtered flux", ""),
            ("electron_density", "electron density", "m^-3"),
            ("discharge_current", "discharge current of the model", "A"),
        ):
            data[key] = variable("current", [c[key] for c in columns], long_name, units)
        data["converged"] = variable("current", [s.converged for s in states], "converged")
        return data

    def scan(
        self, parameter: str, values: list, current: float, voltage: float
    ) -> list[GlobalState]:
        """Dc states for values of a field of the magnetron, each started from the last."""
        states: list[GlobalState] = []
        previous: GlobalState | None = None
        for value in values:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                magnetron = dataclasses.replace(self, **{parameter: value})
            state = magnetron.steady_state(current, voltage, previous)
            states.append(state)
            previous = state
        return states


def pulse(
    on: float, on_time: float, period: float, off: float = 0.0
) -> Callable[[float], float]:
    """A rectangular waveform: ``on`` for ``on_time`` of every ``period``, else ``off``."""
    if not 0.0 < on_time < period:
        raise ValueError("on_time must be in (0, period)")

    def waveform(t: float) -> float:
        return on if (t % period) < on_time else off

    return waveform

