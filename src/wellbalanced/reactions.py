"""Reactions: reactants, products, a rate coefficient and the energy they exchange.

A :class:`Reaction` turns its reactants into its products at the volume rate
``R = K(T) prod n_reactant`` (m^-3 s^-1), with ``K`` in m^3/s for two
reactants. ``K`` depends on the electron temperature for electron-impact
reactions and on the gas temperature for reactions between heavy species. The
electrons lose ``electron_energy_loss`` (eV) per reaction and the gas gains
``gas_heating`` (eV). An elastic reaction changes no density; the electrons
lose ``3 (m_e / M) (T_e - T_g)`` per collision to the target of mass ``M``,
which the gas gains.

Rates come from cross sections through plasmacoll (:meth:`Reaction.from_process`,
:meth:`Reaction.from_cross_section`), or are given directly: a constant, a
:class:`plasmacoll.RateTable` or any function of the temperature in eV.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from plasmacoll import CollisionProcess, CrossSection, RateTable
from plasmacoll.constants import ELECTRON_MASS

__all__ = ["TEMPERATURES", "Reaction"]

#: The temperature a rate coefficient depends on.
TEMPERATURES = ("electron", "gas")

#: A rate coefficient as a function of temperature, on numbers or arrays.
RateFunction = Callable[[np.ndarray | float], np.ndarray | float]


@dataclass(frozen=True)
class Reaction:
    """One reaction of the model.

    Attributes:
        reactants: Species names, repeated for a species that appears twice.
        products: Species names, repeated likewise.
        rate: ``K`` in m^3/s (for two reactants): a number, a
            :class:`plasmacoll.RateTable`, or a function of the temperature in eV.
        temperature: ``"electron"`` or ``"gas"``, the temperature ``K`` depends on.
        electron_energy_loss: Energy in eV the electrons lose per reaction
            (negative for a gain, e.g. superelastic or detachment energy).
        gas_heating: Energy in eV the gas gains per reaction (Franck-Condon
            fragments, charge exchange).
        elastic_mass: For an elastic electron collision, the target mass in kg:
            the electrons then lose ``3 (m_e / M) (T_e - T_g)`` per collision.
        name: A name for diagnostics; built from the equation if empty.
    """

    reactants: Sequence[str]
    products: Sequence[str]
    rate: float | RateTable | RateFunction
    temperature: str = "electron"
    electron_energy_loss: float = 0.0
    gas_heating: float = 0.0
    elastic_mass: float | None = None
    name: str = ""
    _counts: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the reaction and fill in its name."""
        object.__setattr__(self, "reactants", tuple(self.reactants))
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
        ``incident_mass`` is needed for a table in the lab frame (see
        :func:`plasmacoll.rate_coefficient`).
        """
        table = RateTable.from_cross_section(
            cross_section, reduced_mass, incident_mass=incident_mass
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
        taken as infinitely heavy.
        """
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
