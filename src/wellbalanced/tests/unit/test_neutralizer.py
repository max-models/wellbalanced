"""The beam-driven plasma neutralizer; the numbers are the illustrative placeholders."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from plasmacoll import ChargeStateTransitions, charge_state_fractions
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE

from wellbalanced.neutralizer import (
    BeamData,
    FastBeam,
    illustrative_deuterium_neutralizer,
)


def neutralizer(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return illustrative_deuterium_neutralizer(**overrides)


@pytest.fixture(scope="module")
def solved():
    model = neutralizer()
    return model, model.solve()


def test_the_illustrative_data_warn_that_they_are_not_data() -> None:
    with pytest.warns(UserWarning, match="placeholder"):
        illustrative_deuterium_neutralizer()


def test_a_stripped_electron_moves_with_the_beam() -> None:
    beam = FastBeam(energy=1.0e6, current=40.0, mass=2.014 * ATOMIC_MASS)
    assert beam.stripped_electron_energy == pytest.approx(272.4, abs=0.2)
    assert beam.particle_rate == pytest.approx(40.0 / ELEMENTARY_CHARGE)


def test_the_fixed_point_converges_and_every_balance_closes(solved) -> None:
    model, result = solved
    assert result.converged
    plasma = result.plasma
    assert plasma.converged
    assert sum(plasma.power.values()) == pytest.approx(
        0.0, abs=1e-6 * result.beam_power
    )
    for name in ("e", "D2", "D", "D2+", "D+"):
        gross = max(abs(v) for v in plasma.reaction_rates.values())
        assert abs(plasma.particle_balance[name]) < 1e-6 * gross
    assert sum(result.exit_fractions.values()) == pytest.approx(1.0)
    assert 0.0 < plasma.ionization_degree(["D2", "D"]) < 0.05


def test_the_beam_fractions_are_plasmacolls_on_the_line_densities(solved) -> None:
    model, result = solved
    z = result.position
    expected = charge_state_fractions(
        model.data.transitions,
        {
            "gas": model.gas_target_density(result.plasma.densities) * z,
            "plasma": result.plasma.electron_density * z,
        },
    )
    assert result.fractions == pytest.approx(expected, abs=2e-6)


def test_the_stripped_electrons_bring_the_beam_power(solved) -> None:
    model, result = solved
    plasma = result.plasma
    mean = np.trapezoid(result.fractions, result.position, axis=0) / model.length
    sigma = model.data.transitions.cross_sections["gas"][("-", "0")]
    rate = (
        model.beam.particle_rate
        / model.cross_section
        * mean[0]
        * sigma
        * plasma.densities["D2"]
    )
    expected = rate * model.volume * model.beam.stripped_electron_energy
    # the fractions moved by less than the tolerance in the last iteration
    assert plasma.power["beam -->0 on gas"] == pytest.approx(
        expected * ELEMENTARY_CHARGE, rel=1e-4
    )


def test_a_weak_beam_leaves_the_gas_at_inflow_over_pumping() -> None:
    model = neutralizer(beam=FastBeam(1.0e6, 0.05, 2.014 * ATOMIC_MASS))
    result = model.solve()
    plasma = result.plasma
    speed = model.pumping_speed(plasma.gas_temperature)
    # deuterium nuclei: fed as molecules, pumped as molecules and atoms
    nuclei = 2 * plasma.densities["D2"] + plasma.densities["D"]
    assert nuclei * speed == pytest.approx(2 * model.gas_inflow, rel=0.02)


def test_the_plasma_adds_neutrals_through_stripping(solved) -> None:
    _, result = solved
    gas_only = BeamData(
        ChargeStateTransitions(
            {"gas": neutralizer().data.transitions.cross_sections["gas"]}
        ),
        neutralizer().data.gas_ionization,
    )
    without = neutralizer(data=gas_only).solve()
    assert result.neutral_fraction > without.neutral_fraction


def test_bad_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="targets"):
        BeamData(ChargeStateTransitions({"wall": {("-", "0"): 1e-20}}))
    with pytest.raises(ValueError, match="no species"):
        neutralizer(gas="H2")
    with pytest.raises(ValueError, match="positive"):
        neutralizer(length=0.0)
