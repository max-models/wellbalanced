"""The 0D RF driver: the coupling models against closed forms, the driver's bookkeeping."""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import pytest
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy.constants import epsilon_0 as EPS_0
from scipy.constants import mu_0 as MU_0

from wellbalanced import Coil, RFDriver
from wellbalanced.driver import (
    illustrative_hydrogen_driver,
    nagaoka_coefficient,
    plasma_conductivity,
    solenoid_impedance,
    transformer_impedance,
)

E = ELEMENTARY_CHARGE
FREQUENCY = 1.0e6
OMEGA = 2.0 * np.pi * FREQUENCY
RADIUS = 0.12
COIL = Coil(turns=6, radius=0.14, length=0.17)
K = COIL.factor


@pytest.fixture
def driver() -> RFDriver:
    with pytest.warns(UserWarning, match="placeholder"):
        return illustrative_hydrogen_driver()


def test_nagaoka_coefficient() -> None:
    assert nagaoka_coefficient(1.0, 1.0) == pytest.approx(0.6884, abs=1e-4)
    assert nagaoka_coefficient(0.001, 1.0) == pytest.approx(1.0, abs=1e-3)
    assert nagaoka_coefficient(2.0, 1.0) < nagaoka_coefficient(1.0, 1.0)
    assert Coil(6, 0.14, 0.17, field_factor=0.5).factor == 0.5
    assert COIL.inductance == pytest.approx(K * MU_0 * np.pi * 0.14**2 * 36 / 0.17)
    with pytest.raises(ValueError):
        nagaoka_coefficient(0.0, 1.0)


def _transformer(z_p: complex) -> complex:
    """The textbook transformer with a given skin-layer impedance (sqrt(K) N turns)."""
    l_11 = K * MU_0 * np.pi * COIL.radius**2 * COIL.turns**2 / COIL.length
    mutual = math.sqrt(K) * MU_0 * np.pi * RADIUS**2 * COIL.turns / COIL.length
    l_2 = MU_0 * np.pi * RADIUS**2 / COIL.length
    return 1j * OMEGA * l_11 + OMEGA**2 * mutual**2 / (z_p + 1j * OMEGA * l_2)


def test_collisional_limit_is_the_textbook_skin_resistance() -> None:
    # nu >> omega: sigma real, Z_p = (1 + i) R_p, R_p = 2 pi R / (sigma l delta)
    ne, nu = 1.0e18, 1.0e11
    sigma = ne * E**2 / (ELECTRON_MASS * nu)
    delta = math.sqrt(2.0 / (OMEGA * MU_0 * sigma))
    r_p = 2.0 * np.pi * RADIUS / (sigma * COIL.length * delta)
    expected = _transformer((1.0 + 1.0j) * r_p)
    z_s = transformer_impedance(COIL, RADIUS, ne, nu, FREQUENCY)
    assert z_s.real == pytest.approx(expected.real, rel=1e-4)
    assert z_s.imag == pytest.approx(expected.imag, rel=1e-4)


def test_inertial_limit_is_purely_reactive() -> None:
    # nu = 0: the skin depth is c / omega_pe and the layer an inductance
    ne = 1.0e18
    omega_pe = math.sqrt(ne * E**2 / (EPS_0 * ELECTRON_MASS))
    z_p = 1j * OMEGA * MU_0 * (2.0 * np.pi * RADIUS / COIL.length) / omega_pe * 2.998e8
    z_s = transformer_impedance(COIL, RADIUS, ne, 0.0, FREQUENCY)
    assert abs(z_s.real) < 1e-12 * abs(z_s)
    assert z_s.imag == pytest.approx(_transformer(z_p).imag, rel=1e-3)


@pytest.mark.parametrize("impedance", [transformer_impedance, solenoid_impedance])
def test_no_plasma_leaves_the_coil_alone(impedance) -> None:
    empty = impedance(COIL, RADIUS, 0.0, 1.0e7, FREQUENCY)
    assert empty == pytest.approx(1j * OMEGA * COIL.inductance)
    # fainter and more collisional plasmas absorb less, towards nothing
    densities = [impedance(COIL, RADIUS, n, 1.0e7, FREQUENCY).real for n in (1e8, 1e14)]
    frequencies = [
        impedance(COIL, RADIUS, 1e18, nu, FREQUENCY).real for nu in (1e14, 1e18)
    ]
    assert 0.0 < densities[0] < 1e-2 * densities[1]
    assert 0.0 < frequencies[1] < 1e-1 * frequencies[0]


def test_a_weak_conductor_absorbs_the_eddy_current_power() -> None:
    # |q R| << 1: P = sigma omega^2 B^2 pi R^4 l / 16, so R_s = K pi sigma (omega mu0 N)^2
    # R^4 / (8 l)
    ne, nu = 1.0e12, 1.0e9
    sigma = plasma_conductivity(ne, nu, FREQUENCY)
    expected = (
        K * np.pi * sigma.real * (OMEGA * MU_0 * COIL.turns) ** 2 * RADIUS**4
    ) / (8.0 * COIL.length)
    z_s = solenoid_impedance(COIL, RADIUS, ne, nu, FREQUENCY)
    assert z_s.real == pytest.approx(expected, rel=1e-6)


def test_the_transformer_is_the_thin_skin_limit_of_the_column() -> None:
    errors = []
    for ne in (1.0e17, 1.0e18, 1.0e19, 1.0e20):
        exact = solenoid_impedance(COIL, RADIUS, ne, 1.0e7, FREQUENCY)
        model = transformer_impedance(COIL, RADIUS, ne, 1.0e7, FREQUENCY)
        errors.append(abs(model.real / exact.real - 1.0))
        assert model.imag == pytest.approx(exact.imag, rel=0.1)
    assert errors == sorted(errors, reverse=True)  # better as the skin thins
    assert errors[-1] < 0.02


def test_a_driver_balances_its_power(driver) -> None:
    state = driver.solve(25.0e3)
    assert state.plasma.converged
    assert 0.5 * state.plasma_resistance * state.coil_current**2 == pytest.approx(
        25.0e3
    )
    loss = 0.5 * driver.network_resistance * state.coil_current**2
    assert state.rf_power == pytest.approx(25.0e3 + loss)
    assert state.efficiency == pytest.approx(25.0e3 / state.rf_power)
    assert state.coil_voltage == pytest.approx(
        abs(state.coil_impedance) * state.coil_current
    )
    nu = sum(state.plasma.reaction_rates[r] for r in driver.momentum_transfer)
    assert state.collision_frequency == pytest.approx(
        nu / state.plasma.electron_density
    )
    dataset = state.to_dataset()
    assert float(dataset.efficiency) == pytest.approx(state.efficiency)
    assert dataset.coil_current.attrs["units"] == "A"


def test_scans_take_seconds_and_continue(driver) -> None:
    start = time.perf_counter()
    powers = driver.scan("power", [5.0e3, 10.0e3, 20.0e3, 40.0e3])
    frequencies = driver.scan("frequency", [1.0e6, 2.0e6], power=25.0e3)
    assert time.perf_counter() - start < 10.0
    densities = [s.plasma.electron_density for s in powers]
    assert densities == sorted(densities)
    # the plasma does not depend on the frequency in 0D, the coupling does
    first, second = (s.plasma.electron_density for s in frequencies)
    assert first == pytest.approx(second, rel=1e-6)
    assert frequencies[1].skin_depth < frequencies[0].skin_depth
    with pytest.raises(ValueError, match="power"):
        driver.scan("frequency", [1.0e6])


def test_the_exact_column_can_couple_the_driver(driver) -> None:
    transformer = driver.solve(25.0e3)
    exact = dataclasses.replace(driver, coupling="solenoid").couple(
        transformer.plasma, 25.0e3
    )
    assert exact.plasma_resistance == pytest.approx(
        transformer.plasma_resistance, rel=0.2
    )


def test_a_longer_driver_keeps_the_coil(driver) -> None:
    longer = dataclasses.replace(driver, length=0.30, coil_length=0.17)
    assert longer.coil.length == 0.17
    assert longer.chamber.volume > driver.chamber.volume


def test_driver_checks_its_input(driver) -> None:
    with pytest.raises(ValueError, match="outside"):
        dataclasses.replace(driver, coil_radius=driver.radius)
    with pytest.raises(ValueError, match="coil_length"):
        dataclasses.replace(driver, coil_length=2.0 * driver.length)
    with pytest.raises(ValueError, match="momentum_transfer"):
        dataclasses.replace(driver, momentum_transfer=("no such reaction",))
    with pytest.raises(ValueError, match="coupling"):
        dataclasses.replace(driver, coupling="other")
    with pytest.raises(ValueError, match="inside"):
        transformer_impedance(COIL, 0.2, 1.0e18, 1.0e7, FREQUENCY)
