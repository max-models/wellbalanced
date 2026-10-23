"""The neutralizer with the gas along the channel; the numbers are the placeholders."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
from plasmacoll import charge_state_fractions
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE

from wellbalanced.neutralizer import GasProfile, illustrative_deuterium_neutralizer

D2 = 4.028 * ATOMIC_MASS


def neutralizer(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return illustrative_deuterium_neutralizer(**overrides)


def mean_speed(temperature: float) -> float:
    return float(np.sqrt(8.0 * ELEMENTARY_CHARGE * temperature / (np.pi * D2)))


@pytest.mark.parametrize("feed", [1.5, 0.6])
def test_knudsen_diffusion_from_the_feed_leaves_through_both_ends(feed) -> None:
    length, width, height, inflow, tg = 3.0, 0.1, 1.7, 2.0e21, 0.03
    profile = GasProfile.knudsen(length, width, height, D2, inflow, tg, feed=feed)
    z, n = profile.position, profile.density
    area = width * height
    # Marshak's exit: Gamma = n v_mean / 2; the two ends take all of the inflow
    out = 0.5 * mean_speed(tg) * area * (n[0] + n[-1])
    assert out == pytest.approx(inflow, rel=1e-12)
    # Fick's law on each side with Knudsen's diffusivity (4/3) v_mean A / P
    diffusivity = 4.0 / 3.0 * mean_speed(tg) * area / (2.0 * (width + height))
    left = 0.5 * mean_speed(tg) * area * n[0]
    slope = np.gradient(n, z)
    assert slope[z < feed - 0.1] == pytest.approx(left / (diffusivity * area))
    assert np.argmax(n) == np.flatnonzero(z == feed)[0]
    # free-molecular: the density scales with 1 / v_mean, so the pumping speed with it
    hot = GasProfile.knudsen(length, width, height, D2, inflow, 4 * tg, feed=feed)
    assert hot.mean_density == pytest.approx(profile.mean_density / 2.0)
    assert profile.pumping_speed(4 * tg) == pytest.approx(
        inflow / hot.mean_density, rel=1e-12
    )
    assert profile.shape(z).mean() == pytest.approx(1.0, rel=0.01)


def test_a_flat_profile_with_the_end_pumping_is_the_uniform_gas() -> None:
    model = neutralizer(points=101)
    uniform = model.solve()
    density = model.gas_inflow / model.pumping_speed(0.03)
    z = np.linspace(0.0, model.length, 11)
    flat = GasProfile(z, np.full(z.shape, density), model.gas_inflow, 0.03)
    result = dataclasses.replace(model, gas_profile=flat).solve()
    assert result.neutral_fraction == pytest.approx(uniform.neutral_fraction, rel=1e-6)
    assert result.plasma.electron_density == pytest.approx(
        uniform.plasma.electron_density, rel=1e-5
    )
    assert result.line_density == pytest.approx(uniform.line_density, rel=1e-6)


@pytest.fixture(scope="module")
def peaked():
    model = neutralizer(points=101)
    profile = GasProfile.knudsen(
        model.length, model.width, model.height, D2, model.gas_inflow, 0.03
    )
    model = dataclasses.replace(model, gas_profile=profile)
    return model, model.solve()


def test_the_beam_crosses_the_line_density_of_the_profile(peaked) -> None:
    model, result = peaked
    assert result.converged
    plasma = result.plasma
    # the global model balances the profile's mean: inflow over its pumping speed
    nuclei = 2 * plasma.densities["D2"] + plasma.densities["D"]
    speed = model.pumping_speed(plasma.gas_temperature)
    assert nuclei * speed == pytest.approx(2 * model.gas_inflow, rel=0.05)
    z = result.position
    shape = model.gas_profile.shape(z)
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (shape[1:] + shape[:-1]))))
    cumulative *= z[1] - z[0]
    expected = charge_state_fractions(
        model.data.transitions,
        {
            "gas": plasma.densities["D2"] * cumulative
            + 0.5 * plasma.densities["D"] * z,
            "plasma": plasma.electron_density * z,
        },
    )
    assert result.fractions == pytest.approx(expected, abs=2e-6)
    assert result.gas_profile == pytest.approx(
        plasma.densities["D2"] * shape + 0.5 * plasma.densities["D"]
    )
    # the uniform estimate (end conductances only) leaves 3.5 times less gas
    uniform = model.gas_inflow / neutralizer().pumping_speed(0.03)
    assert plasma.densities["D2"] / uniform > 3.0


def test_the_sources_on_the_gas_take_the_fractions_weighted_with_it(peaked) -> None:
    model, result = peaked
    plasma = result.plasma
    z = result.position
    shape = model.gas_profile.shape(z)
    weighted = np.trapezoid(result.fractions[:, 0] * shape, z) / model.length
    plain = np.trapezoid(result.fractions[:, 0], z) / model.length
    # the D- fraction falls convexly along the channel: the dense middle has fewer
    # of them than the plain average (0.39 against 0.43), so they strip less
    assert weighted < 0.95 * plain
    sigma = model.data.transitions.cross_sections["gas"][("-", "0")]
    rate = model.mean_flux * weighted * sigma * plasma.densities["D2"]
    expected = rate * model.volume * model.beam.stripped_electron_energy
    assert plasma.power["beam -->0 on gas"] == pytest.approx(
        expected * ELEMENTARY_CHARGE, rel=1e-4
    )


def test_the_neutral_fraction_follows_the_line_density_not_the_shape(peaked) -> None:
    """Stripping on the gas depends on the line density alone; the plasma adds little."""
    model, result = peaked
    profile = model.gas_profile
    flat = GasProfile(
        profile.position,
        np.full(profile.position.shape, profile.mean_density),
        profile.inflow,
        profile.temperature,
    )
    same = dataclasses.replace(model, gas_profile=flat).solve()
    assert same.line_density == pytest.approx(result.line_density, rel=1e-3)
    assert same.neutral_fraction == pytest.approx(result.neutral_fraction, rel=1e-3)


def test_every_slice_balances_its_own_gas_and_beam(peaked) -> None:
    model, result = peaked
    local = model.local_plasma(result, points=5)
    assert bool(local.converged.all())
    electrons = local.density.sel(species="e").values
    # the plasma follows the gas and the beam's D-, which peak in the middle
    assert electrons[2] > electrons[0] and electrons[2] > electrons[4]
    # beam sources make electrons faster than the ions leave, lowering the sheath
    potential = local.plasma_potential.values
    assert np.all(potential > 0.0)
    assert np.isfinite(result.plasma_potential) and result.plasma_potential > 0.0
    assert local.position.values == pytest.approx(np.linspace(0.0, model.length, 5))


def test_a_profile_must_span_the_channel() -> None:
    profile = GasProfile(np.array([0.0, 2.0]), np.array([1.0, 1.0]), 1.0, 0.03)
    with pytest.raises(ValueError, match="span the channel"):
        neutralizer(gas_profile=profile)
    with pytest.raises(ValueError, match="increase"):
        GasProfile(np.array([1.0, 0.0]), np.array([1.0, 1.0]), 1.0, 0.03)
    with pytest.raises(ValueError, match="feed"):
        GasProfile.knudsen(3.0, 0.1, 1.7, D2, 1e21, 0.03, feed=4.0)
