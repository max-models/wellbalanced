"""Species of a plasma: name, mass and charge."""

from __future__ import annotations

from dataclasses import dataclass

from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS

__all__ = ["ELECTRON", "Species"]


@dataclass(frozen=True)
class Species:
    """A species of the plasma or the gas.

    Attributes:
        name: A unique name, used by the reactions (``"e"`` for electrons).
        mass: The mass in kg.
        charge: The charge in elementary charges.
    """

    name: str
    mass: float
    charge: int = 0

    def __post_init__(self) -> None:
        """Validate the species."""
        if not self.name:
            raise ValueError("a species needs a name")
        if self.mass <= 0.0:
            raise ValueError(f"{self.name}: mass must be > 0")

    @classmethod
    def atomic(cls, name: str, mass_u: float, charge: int = 0) -> Species:
        """A species of mass ``mass_u`` in atomic mass units, minus the electrons of its charge."""
        return cls(name, mass_u * ATOMIC_MASS - charge * ELECTRON_MASS, charge)

    @property
    def is_electron(self) -> bool:
        """Whether this is the electron species."""
        return self.charge == -1 and self.mass < 10.0 * ELECTRON_MASS

    @property
    def is_neutral(self) -> bool:
        """Whether the species has no charge."""
        return self.charge == 0


#: The electrons.
ELECTRON = Species("e", ELECTRON_MASS, -1)
