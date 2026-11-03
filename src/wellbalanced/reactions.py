"""Reactions: reactants, products, a rate coefficient and the energy they exchange.

A `Reaction` turns its reactants into its products at the volume rate
``R = K(T) prod n_reactant`` (m^-3 s^-1), with ``K`` in m^3/s for two
reactants. ``K`` depends on the electron temperature for electron-impact
reactions and on the gas temperature for reactions between heavy species. The
electrons lose ``electron_energy_loss`` (eV) per reaction and the gas gains
``gas_heating`` (eV). An elastic reaction changes no density; the electrons
lose ``3 (m_e / M) (T_e - T_g)`` per collision to the target of mass ``M``,
which the gas gains.

Rates come from cross sections through plasmacoll (`Reaction.from_process`,
`Reaction.from_cross_section`), or are given directly: a constant, a
`plasmacoll.RateTable` or any function of the temperature in eV.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from methodcard import Rule, method
from plasmacoll import CollisionProcess, CrossSection, RateTable
from plasmacoll.constants import ELECTRON_MASS
from plasmacoll.units import convert_units, to_unit, unit_field

__all__ = ["TEMPERATURES", "Reaction"]

#: The temperature a rate coefficient depends on.
TEMPERATURES = ("electron", "gas")

#: A rate coefficient as a function of temperature, on numbers or arrays.
RateFunction = Callable[[np.ndarray | float], np.ndarray | float]


def _elastic_rules() -> dict:
    """The SymPy rule of the elastic loss of `Reaction` (for the tests)."""
    import sympy

    te, tg, mass = sympy.symbols("te tg mass", positive=True)
    rule = 3 * sympy.Float(ELECTRON_MASS) / mass * (te - tg)

    def loss(te: float, tg: float, mass: float) -> float:
        reaction = Reaction(("e", "X"), ("e", "X"), 1.0, elastic_mass=mass)
        return reaction.electron_loss(te, tg)

    ranges = {"te": (0.5, 20.0), "tg": (0.01, 0.1), "mass": (1.6e-27, 7.0e-26)}
    return {"elastic": Rule(rule, implementation=loss, ranges=ranges, rtol=1e-12)}


@method(
    equations={
        "rate": r"R = K(T) \prod_{r} n_r \quad [\mathrm{m^{-3}\,s^{-1}}], \qquad T = T_e \text{ (electron impact) or } T_g \text{ (heavy particles)}",
        "stoichiometry": r"\nu_s = (\text{products} - \text{reactants})_s, \qquad \frac{dn_s}{dt} \mathrel{+}= \nu_s R",
        "energy": r"\frac{dw}{dt} \mathrel{-}= R\,\varepsilon_e, \qquad \text{the gas gains } R\,\varepsilon_g",
        "elastic": r"\varepsilon_e = 3\,\frac{m_e}{M}\,(T_e - T_g) = \varepsilon_g",
    },
    assumptions=(
        r"$K$ a constant, a plasmacoll `RateTable` or a function of the temperature in eV",
        r"an elastic reaction keeps its reactants and moves $3 (m_e/M)(T_e - T_g)$ per collision from the electrons to the gas",
        r"a reactant that is also a product (the electron of `e + H2 -> e + H + H`) nets out of $\nu_s$",
    ),
    references=(
        "Lieberman and Lichtenberg, Principles of Plasma Discharges and Materials Processing, 2nd ed. (2005), ch. 3",
    ),
    symbolic=_elastic_rules,
)
@dataclass(frozen=True)
class Reaction:
    """One reaction of the model.

    Attributes:
        reactants: Species names, repeated for a species that appears twice.
        products: Species names, repeated likewise.
        rate: ``K`` in m^3/s (for two reactants): a number, a
            `plasmacoll.RateTable`, or a function of the temperature in eV.
        temperature: ``"electron"`` or ``"gas"``, the temperature ``K`` depends on.
        electron_energy_loss: Energy in eV the electrons lose per reaction
            (negative for a gain, e.g. superelastic or detachment energy).
        gas_heating: Energy in eV the gas gains per reaction (Franck-Condon
            fragments, charge exchange).
        elastic_mass: For an elastic electron collision, the target mass in kg:
            the electrons then lose ``3 (m_e / M) (T_e - T_g)`` per collision.
        name: A name for diagnostics; built from the equation if empty.

    The energies, the mass and a constant rate take astropy Quantities,
    strings such as ``"15.76 eV"`` or numbers in the units above (a constant
    rate in m^(3(k-1))/s for ``k`` reactants).
    """

    reactants: Sequence[str]
    products: Sequence[str]
    rate: float | RateTable | RateFunction
    temperature: str = "electron"
    electron_energy_loss: float = unit_field("eV", 0.0)
    gas_heating: float = unit_field("eV", 0.0)
    elastic_mass: float | None = unit_field("kg", None)
    name: str = ""
    _counts: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the reaction and fill in its name."""
        convert_units(self)
        object.__setattr__(self, "reactants", tuple(self.reactants))
        if not callable(self.rate):
            unit = f"m^{3 * (len(self.reactants) - 1)} / s"
            object.__setattr__(self, "rate", to_unit(self.rate, unit, "rate"))
        object.__setattr__(self, "products", tuple(self.products))
        if not self.reactants:
            raise ValueError("a reaction needs at least one reactant")
        if self.temperature not in TEMPERATURES:
            raise ValueError(f"temperature must be one of {TEMPERATURES}")
        if self.elastic_mass is not None:
            if self.elastic_mass <= 0.0:
                raise ValueError("elastic_mass must be > 0")
            if Counter(self.reactants) != Counter(self.products):
                raise ValueError("an elastic reaction must keep its reactants")
        if isinstance(self.rate, (int, float)) and self.rate < 0.0:
            raise ValueError("a rate coefficient must be >= 0")
        counts = Counter(self.products)
        counts.subtract(Counter(self.reactants))
        object.__setattr__(
            self, "_counts", {name: n for name, n in counts.items() if n != 0}
        )
        if not self.name:
            object.__setattr__(
                self,
                "name",
                " + ".join(self.reactants) + " -> " + " + ".join(self.products),
            )

    @property
    def stoichiometry(self) -> dict[str, int]:
        """Net change of every species per reaction (products minus reactants)."""
        return dict(self._counts)

    def coefficient(self, electron_temperature: float, gas_temperature: float) -> float:
        """Return ``K`` at the temperatures (eV)."""
        return float(self.coefficients(electron_temperature, gas_temperature))

    def coefficients(
        self,
        electron_temperature: np.ndarray | float,
        gas_temperature: np.ndarray | float,
    ) -> np.ndarray:
        """Return ``K`` at arrays of temperatures (eV), e.g. on the cells of a mesh."""
        t = electron_temperature if self.temperature == "electron" else gas_temperature
        rate = self.rate
        if isinstance(rate, (int, float)):
            return np.full(np.shape(t), float(rate))
        return np.asarray(rate(t), dtype=float)

    def electron_loss(
        self, electron_temperature: float, gas_temperature: float
    ) -> float:
        """Energy in eV the electrons lose per reaction."""
        if self.elastic_mass is not None:
            ratio = 3.0 * ELECTRON_MASS / self.elastic_mass
            return ratio * (electron_temperature - gas_temperature)
        return self.electron_energy_loss

    def gas_gain(self, electron_temperature: float, gas_temperature: float) -> float:
        """Energy in eV the gas gains per reaction."""
        if self.elastic_mass is not None:
            return self.electron_loss(electron_temperature, gas_temperature)
        return self.gas_heating

    @classmethod
    @method(
        equations={
            "maxwellian": r"K(T) = \langle\sigma v\rangle = \sqrt{\frac{8}{\pi \mu}}\,(eT)^{-3/2} \int_0^\infty \sigma(\varepsilon)\, \varepsilon\, e^{-\varepsilon / eT}\, d\varepsilon",
        },
        assumptions=(
            r"a Maxwellian at the reduced mass $\mu$ (the electron mass for electron impact); plasmacoll's `RateTable.from_cross_section`",
            r"`incident_mass` only for a cross section tabulated in the lab frame",
        ),
    )
    def from_cross_section(
        cls,
        reactants: Sequence[str],
        products: Sequence[str],
        cross_section: CrossSection,
        reduced_mass: float,
        *,
        incident_mass: float | None = None,
        temperature: str = "electron",
        electron_energy_loss: float = 0.0,
        gas_heating: float = 0.0,
        elastic_mass: float | None = None,
        name: str = "",
    ) -> Reaction:
        """A reaction whose rate is ``cross_section`` averaged over a Maxwellian.

        ``reduced_mass`` (kg) is the electron mass for electron impact;
        ``incident_mass`` (kg) is needed for a table in the lab frame (see
        `plasmacoll.rate_coefficient`). The masses and energies also take
        Quantities or strings such as ``"1 u"``.
        """
        table = RateTable.from_cross_section(
            cross_section,
            to_unit(reduced_mass, "kg", "reduced_mass"),
            incident_mass=to_unit(incident_mass, "kg", "incident_mass"),
        )
        return cls(
            reactants,
            products,
            table,
            temperature=temperature,
            electron_energy_loss=electron_energy_loss,
            gas_heating=gas_heating,
            elastic_mass=elastic_mass,
            name=name or cross_section.label,
        )

    @classmethod
    @method(
        equations={
            "reduced_mass": r"\mu = \frac{m_i m_t}{m_i + m_t} \quad (\mu = m_i \text{ without a target mass})",
            "loss": r"\varepsilon_e = \begin{cases} 0 \ (\text{the } 3 m_e/M \text{ transfer}) & \text{elastic, } m_i < 10\,m_e \\ \text{the process's loss} & \text{otherwise} \end{cases}",
        },
        assumptions=(r"an elastic electron process needs `target_mass`",),
        components=(from_cross_section,),
    )
    def from_process(
        cls,
        process: CollisionProcess,
        reactants: Sequence[str],
        products: Sequence[str],
        *,
        incident_mass: float = ELECTRON_MASS,
        target_mass: float | None = None,
        temperature: str = "electron",
        gas_heating: float = 0.0,
    ) -> Reaction:
        """A reaction from a plasmacoll process: its cross section, frame and energy loss.

        The reactants and products are given here, since a process only names
        the species it tracks. ``target_mass`` (kg) gives the reduced mass and,
        for an elastic process, the energy transfer; without it the target is
        taken as infinitely heavy. The masses also take Quantities or strings
        such as ``"2 u"``.
        """
        incident_mass = to_unit(incident_mass, "kg", "incident_mass")
        target_mass = to_unit(target_mass, "kg", "target_mass")
        mu = (
            incident_mass
            if target_mass is None
            else incident_mass * target_mass / (incident_mass + target_mass)
        )
        elastic = process.kind == "elastic" and incident_mass < 10.0 * ELECTRON_MASS
        if elastic and target_mass is None:
            raise ValueError("an elastic electron process needs target_mass")
        return cls.from_cross_section(
            reactants,
            products,
            process.cross_section,
            mu,
            incident_mass=incident_mass if process.energy_frame == "lab" else None,
            temperature=temperature,
            electron_energy_loss=0.0 if elastic else process.loss,
            gas_heating=gas_heating,
            elastic_mass=target_mass if elastic else None,
            name=process.name,
        )
