"""The time-dependent drive, temperature-dependent sources and the afterglow floor."""

from __future__ import annotations

import math

import numpy as np
import pytest
from plasmacoll import CrossSection
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

from wellbalanced import (
    ELECTRON,
    Chamber,
    GlobalModel,
    Reaction,
    Species,
    VolumeSource,
)

E = ELEMENTARY_CHARGE
AR = Species.atomic("Ar", 39.948)
AR_ION = Species.atomic("Ar+", 39.948, 1)
CHAMBER = Chamber(volume=1.0e-3, wall_area=0.06, h_factor=0.5)
GAS = 1.0e20


def argon(**settings) -> GlobalModel:
    ionization = Reaction.from_cross_section(
        ("e", "Ar"),
        ("e", "e", "Ar+"),
        CrossSection.constant(2.0e-20, threshold=15.76),
        ELECTRON_MASS,
        electron_energy_loss=15.76,
    )
    defaults = {
        "species": [ELECTRON, AR, AR_ION],
        "reactions": [ionization],
        "chamber": CHAMBER,
        "fixed": {"Ar": GAS},
    }
    return GlobalModel(**{**defaults, **settings})


def test_a_source_rate_may_depend_on_the_temperature() -> None:
    """A temperature-dependent source with a constant value gives the constant source."""
    source = {"changes": {"Ar+": 1, "e": 1}, "per": ("Ar",)}
    fixed = argon(power=50.0, sources=[VolumeSource(rate=1.0e-12, **source)])
    follows = argon(
        power=50.0, sources=[VolumeSource(rate=lambda te, tg: 1.0e-12, **source)]
    )
    a, b = fixed.steady_state(), follows.steady_state()
    assert b.electron_density == pytest.approx(a.electron_density, rel=1e-9)
    assert b.electron_temperature == pytest.approx(a.electron_temperature, rel=1e-9)
    assert VolumeSource({"Ar+": 1}, lambda te, tg: te * tg).coefficient(
        2.0, 3.0
    ) == pytest.approx(6.0)


def test_a_source_that_follows_the_electron_temperature_moves_the_balance() -> None:
    """Ions that make ions in proportion to ``T_e`` raise the density at a given power."""
    plain = argon(power=50.0).steady_state()
    boosted = argon(
        power=50.0,
        sources=[
            VolumeSource(
                {"Ar+": 1, "e": 1},
                lambda te, tg: 1.0e4 * te,
                per=("Ar+",),
                name="boost",
            )
        ],
    )
    state = boosted.steady_state()
    assert state.converged
    assert state.electron_temperature < plain.electron_temperature


def test_the_power_may_be_a_function_of_the_time() -> None:
    """A constant function of ``t`` gives the same steady state as the number."""
    constant = argon(power=80.0).steady_state()
    function = argon(power=lambda t: 80.0).steady_state()
    assert function.electron_density == pytest.approx(constant.electron_density)


def test_evolve_reaches_the_steady_state_and_follows_the_drive() -> None:
    steady = argon(power=100.0).steady_state()
    times = np.linspace(0.0, 2.0e-2, 5)
    states = argon(power=100.0).evolve(times, {"Te": 3.0, "Ar+": 1.0e16, "Ar": GAS})
    assert len(states) == len(times)
    assert states[-1].electron_density == pytest.approx(
        steady.electron_density, rel=1e-3
    )
    # a pulse: the density rises with the power on and decays with it off
    pulse = lambda t: 100.0 if t < 5.0e-3 else 0.0  # noqa: E731
    ts = np.linspace(0.0, 1.0e-2, 11)
    on_off = argon(power=pulse).evolve(
        ts, {"Te": 3.0, "Ar+": 1.0e16, "Ar": GAS}, breakpoints=[5.0e-3]
    )
    density = [s.electron_density for s in on_off]
    assert density[5] > density[0]
    assert density[-1] < density[5]


def test_evolve_rejects_bad_times() -> None:
    with pytest.raises(ValueError, match="increasing"):
        argon(power=1.0).evolve([0.0, 0.0, 1.0])


def test_thermalized_electrons_do_not_cool_below_the_gas() -> None:
    """The wall losses of a decaying plasma would cool it below the gas; the floor holds."""
    model = argon(power=0.0, thermalize_electrons=True, gas_temperature=0.05)
    states = model.evolve(
        np.linspace(0.0, 2.0e-4, 6), {"Te": 3.0, "Ar+": 1.0e17, "Ar": GAS}
    )
    te = [s.electron_temperature for s in states]
    assert min(te) >= 0.05 * (1.0 - 1.0e-9)
    assert te[-1] == pytest.approx(0.05, rel=1e-2)
    assert math.isfinite(states[-1].electron_density)
