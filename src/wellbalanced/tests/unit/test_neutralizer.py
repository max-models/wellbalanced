"""The beam-driven plasma neutralizer; the numbers are the illustrative placeholders."""

from __future__ import annotations

import warnings

import numpy as np
import plasmacoll
import pytest
from plasmacoll import ChargeStateTransitions, charge_state_fractions
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE

from wellbalanced.neutralizer import (
    BeamCrossSections,
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


class PowerLawSecondaries:
    """Synthetic gas ionization: ``sigma`` and the mean secondary energy as powers of E."""

    def total(self, projectile_energy: float) -> float:
        return 1.0e-20 * (projectile_energy / 1.0e6) ** -0.8

    def mean_energy(self, projectile_energy: float) -> float:
        return 30.0 * (projectile_energy / 1.0e6) ** 0.3


def energy_dependent(data: BeamData, power: float = 0.0) -> BeamCrossSections:
    """The cross sections of ``data`` at 1 MeV, scaled as ``(E / 1 MeV)^power``."""
    return BeamCrossSections(
        {
            target: {
                pair: (lambda e, s=sigma: s * (np.asarray(e) / 1.0e6) ** power)
                for pair, sigma in table.items()
            }
            for target, table in data.transitions.cross_sections.items()
        },
        dict(data.gas_ionization),
    )


def test_cross_sections_of_the_energy_give_the_numbers_at_the_beam_energy() -> None:
    model = neutralizer()
    assert isinstance(model.data, BeamData)
    functions = energy_dependent(model.data)
    assert functions.at(model.beam.energy) == model.data
    as_functions = neutralizer(data=functions)
    mean = np.array([0.4, 0.5, 0.1])
    assert as_functions.beam_sources(model.mean_flux, mean) == model.beam_sources(
        model.mean_flux, mean
    )
    with pytest.raises(ValueError, match="BeamCrossSections"):
        model.beam_sources(1.0e19, mean, energy=5.0e5)


def test_the_sources_follow_the_local_flux_energy_and_charge_state() -> None:
    data = energy_dependent(neutralizer().data, power=-1.0)
    secondaries = PowerLawSecondaries()
    data = BeamCrossSections(data.transitions, {"-": secondaries, "0": 8.0e-21})
    model = neutralizer(data=data)
    flux = np.array([0.0, 1.0e19, 3.0e19])
    fractions = np.array([[1.0, 0.0, 0.0], [0.5, 0.4, 0.1], [0.2, 0.6, 0.2]])
    energy = 5.0e5
    sources = {s.name: s for s in model.beam_sources(flux, fractions, energy)}
    stripping = sources["beam -->0 on gas"]
    sigma = 1.2e-20 * 2.0  # (E / 1 MeV)^-1
    assert stripping.rate == pytest.approx(flux * fractions[:, 0] * sigma)
    assert stripping.per == ("D2",)
    assert stripping.electron_energy == pytest.approx(
        ELECTRON_MASS / model.beam.mass * energy
    )
    atoms = sources["beam -->0 on gas (D)"]
    assert atoms.rate == pytest.approx(0.5 * stripping.rate)  # weighted atoms
    on_plasma = sources["beam 0->+ on plasma"]
    assert on_plasma.per == ("e",) and on_plasma.changes == {"e": 1}
    assert on_plasma.rate == pytest.approx(flux * fractions[:, 1] * 2.5e-21 * 2.0)
    ionization = sources["gas ionization by -"]
    assert ionization.rate == pytest.approx(
        flux * fractions[:, 0] * secondaries.total(energy)
    )
    assert ionization.electron_energy == pytest.approx(secondaries.mean_energy(energy))
    assert sources["gas ionization by 0"].electron_energy == model.secondary_energy
    assert "gas ionization by +" not in sources
    with pytest.raises(ValueError, match="one value per state"):
        model.beam_sources(1.0e19, [1.0, 0.0])


def test_the_secondary_electrons_bring_their_mean_energy() -> None:
    secondaries = PowerLawSecondaries()
    base = neutralizer().data
    data = BeamCrossSections(
        energy_dependent(base).transitions, {"-": secondaries, "0": secondaries}
    )
    model = neutralizer(data=data)
    result = model.solve()
    assert result.converged
    plasma = result.plasma
    mean = np.trapezoid(result.fractions, result.position, axis=0) / model.length
    energy = model.beam.energy
    for k, state in enumerate(("-", "0")):
        rate = (
            model.mean_flux
            * mean[k]
            * secondaries.total(energy)
            * plasma.densities["D2"]
        )
        # the fractions moved by less than the tolerance in the last iteration
        assert plasma.power[f"gas ionization by {state}"] == pytest.approx(
            rate * model.volume * secondaries.mean_energy(energy) * ELEMENTARY_CHARGE,
            rel=1e-4,
        )


@pytest.mark.skipif(
    not hasattr(plasmacoll, "RuddIonization"), reason="plasmacoll without Rudd's model"
)
def test_rudd_secondaries_of_deuterons_on_d2() -> None:
    mass = 2.014 * ATOMIC_MASS
    rudd = plasmacoll.RUDD_H2.with_projectile(mass)  # type: ignore[attr-defined]
    data = BeamCrossSections({"gas": {("-", "0"): 1.2e-20}}, {"+": rudd})
    at = data.at(1.0e6)
    assert at.gas_ionization["+"] == pytest.approx(rudd.total(1.0e6))
    # Rudd's H2 parameters at 1 MeV / 2 (the same speed as a proton at 0.5 MeV)
    assert at.secondary_energy["+"] == pytest.approx(36.8, abs=0.1)
