"""The coil current and the plasma together: a power-dependent plasma in the RF field.

For a prescribed plasma the field is linear in the coil current, so the absorbed power
fixes the current (:meth:`~wellbalanced.spatial.inductive.InductiveSolution.for_power`).
A real plasma depends on the power it absorbs; :class:`DriverCoupling` closes the loop
between a field model and a plasma model, each a callable:

- ``plasma(P)``: the plasma absorbing ``P`` W, as a :class:`PlasmaProfile` (and anything
  else to keep, e.g. a global model's state);
- ``field(profile)``: the :class:`~wellbalanced.spatial.inductive.InductiveField2D` of
  that plasma.

:meth:`DriverCoupling.at_absorbed_power` sets the power the plasma absorbs (one pass, the
coil current follows). :meth:`DriverCoupling.at_rf_power` sets the generator power, as
Zielke et al. (PSST 31, 035019, 2022) do with their integral controller: the plasma
absorbs ``P = eta P_RF`` with ``eta = R_p / (R_p + R_net)``, and ``R_p`` depends on the
plasma, so ``P`` is the fixed point of ``P -> eta(P) P_RF``, iterated (with relaxation)
to a relative tolerance.

:func:`global_model_plasma` makes the plasma model of an
:class:`~wellbalanced.driver.RFDriver`'s global model: its electron density, temperature
and momentum-transfer frequency on a fixed spatial shape. With it the 0D plasma sits in
the real 2D (or r–z) field instead of the transformer model, which the returned
:class:`~wellbalanced.driver.DriverState` still gives for comparison. A fluid model
(``Fluid2D``) will replace the fixed shape.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wellbalanced.driver import DriverState, RFDriver
from wellbalanced.global_model import GlobalState
from wellbalanced.spatial.grid import CellField
from wellbalanced.spatial.inductive import InductiveField2D, InductiveSolution

__all__ = ["CoupledState", "DriverCoupling", "PlasmaProfile", "global_model_plasma"]


@dataclass(frozen=True)
class PlasmaProfile:
    """A plasma for the field model: electron density, collision frequency, temperature.

    Attributes:
        electron_density: ``n_e`` [m^-3].
        collision_frequency: ``nu`` [1/s].
        electron_temperature: ``T_e`` [eV], for viscous electrons; ``None``: cold.
    """

    electron_density: CellField
    collision_frequency: CellField
    electron_temperature: CellField | None = None

    def settings(self) -> dict[str, Any]:
        """The keyword arguments of :class:`InductiveField2D` this plasma sets."""
        settings: dict[str, Any] = {
            "electron_density": self.electron_density,
            "collision_frequency": self.collision_frequency,
        }
        if self.electron_temperature is not None:
            settings["electron_temperature"] = self.electron_temperature
        return settings


@dataclass
class CoupledState:
    """A plasma and its field at one operating point.

    Attributes:
        solution: The field, scaled to the coil current of the operating point.
        plasma: What the plasma model returned besides the profile (a
            :class:`~wellbalanced.driver.DriverState` for :func:`global_model_plasma`).
        rf_power: The generator power ``P + R_net I^2 / 2`` [W].
        absorbed_power: The power the plasma absorbs [W].
        network_resistance: ``R_net`` [Ohm].
        depth: The effective depth of a cut [m]; ``None`` in r–z.
        iterations: The passes of the fixed point (1 at a set absorbed power).
        converged: Whether the fixed point reached the tolerance.
        history: The absorbed power after each pass [W].
    """

    solution: InductiveSolution
    plasma: Any
    rf_power: float
    absorbed_power: float
    network_resistance: float
    depth: float | None
    iterations: int = 1
    converged: bool = True
    history: list[float] = field(default_factory=list)

    @property
    def coil_current(self) -> float:
        """The coil current amplitude [A]."""
        return abs(self.solution.coil_current)

    @property
    def plasma_resistance(self) -> float:
        """``R_p`` [Ohm]."""
        return self.solution.plasma_resistance(self.depth)

    @property
    def efficiency(self) -> float:
        """The RF power transfer efficiency ``P / P_RF``."""
        return self.absorbed_power / self.rf_power


@dataclass
class DriverCoupling:
    """A field model and a power-dependent plasma model, solved together.

    Attributes:
        field: ``profile -> InductiveField2D``.
        plasma: ``absorbed power [W] -> (PlasmaProfile, anything)``.
        network_resistance: ``R_net`` [Ohm].
        depth: The effective depth of a cut [m]; ``None`` in r–z.
    """

    field: Callable[[PlasmaProfile], InductiveField2D]
    plasma: Callable[[float], tuple[PlasmaProfile, Any]]
    network_resistance: float
    depth: float | None = None

    def _pass(self, power: float) -> tuple[InductiveSolution, Any]:
        profile, extra = self.plasma(power)
        return self.field(profile).solve(1.0), extra

    def at_absorbed_power(self, power: float) -> CoupledState:
        """The operating point where the plasma absorbs ``power`` W."""
        if power <= 0.0:
            raise ValueError("power must be positive")
        unit, extra = self._pass(power)
        solution = unit.for_power(power, self.depth)
        current = abs(solution.coil_current)
        return CoupledState(
            solution=solution,
            plasma=extra,
            rf_power=power + 0.5 * self.network_resistance * current**2,
            absorbed_power=power,
            network_resistance=self.network_resistance,
            depth=self.depth,
            history=[power],
        )

    def at_rf_power(
        self,
        rf_power: float,
        *,
        guess: float | None = None,
        tolerance: float = 1.0e-6,
        max_iterations: int = 50,
        relaxation: float = 1.0,
    ) -> CoupledState:
        """The operating point at the generator power ``rf_power`` W.

        Iterates ``P <- P + relaxation (eta(P) P_RF - P)`` from ``guess`` (half the
        generator power by default) until ``|eta P_RF - P| < tolerance P_RF``.
        """
        if rf_power <= 0.0:
            raise ValueError("rf_power must be positive")
        if not 0.0 < relaxation <= 1.0:
            raise ValueError("relaxation must be in (0, 1]")
        power = 0.5 * rf_power if guess is None else float(guess)
        history: list[float] = []
        converged = False
        unit, extra = self._pass(power)
        for _ in range(max_iterations):
            r_p = unit.plasma_resistance(self.depth)
            target = rf_power * r_p / (r_p + self.network_resistance)
            history.append(target)
            if abs(target - power) < tolerance * rf_power:
                power, converged = target, True
                unit, extra = self._pass(power)
                break
            power += relaxation * (target - power)
            unit, extra = self._pass(power)
        solution = unit.for_power(power, self.depth)
        return CoupledState(
            solution=solution,
            plasma=extra,
            rf_power=power
            + 0.5 * self.network_resistance * abs(solution.coil_current) ** 2,
            absorbed_power=power,
            network_resistance=self.network_resistance,
            depth=self.depth,
            iterations=len(history),
            converged=converged,
            history=history,
        )


def global_model_plasma(
    driver: RFDriver,
    shape: CellField = 1.0,
    *,
    viscous: bool = False,
    guess: GlobalState | Mapping[str, float] | None = None,
) -> Callable[[float], tuple[PlasmaProfile, DriverState]]:
    """The plasma model of ``driver``'s global model, on a fixed spatial shape.

    ``n_e = shape * <n_e>`` with the global model's ``<n_e>`` (scale ``shape`` to a
    volume average of 1 over the plasma), and its ``T_e`` (if ``viscous``) and
    momentum-transfer frequency uniform; ``shape`` is zero outside the plasma or the
    field model's plasma region does it. Each call continues from the last state.
    Returns ``(profile, DriverState)``: the driver state is the transformer model's
    view of the same plasma.
    """
    last: list[GlobalState | Mapping[str, float] | None] = [guess]

    def plasma(power: float) -> tuple[PlasmaProfile, DriverState]:
        state = driver.solve(power, last[0])
        last[0] = state.plasma
        density = state.plasma.electron_density
        if callable(shape):
            function = shape

            def profile_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
                return density * np.asarray(function(x, y))

            electron_density: CellField = profile_density
        else:
            electron_density = density * np.asarray(shape)
        profile = PlasmaProfile(
            electron_density=electron_density,
            collision_frequency=state.collision_frequency,
            electron_temperature=(
                state.plasma.electron_temperature if viscous else None
            ),
        )
        return profile, state

    return plasma
