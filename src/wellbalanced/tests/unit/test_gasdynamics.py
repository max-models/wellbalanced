"""The gas model: shock tube, conservation, uniform states, diffusion, wall closures.

The gas masses are synthetic in the shock tube (unit mass) so that it matches the textbook
case; everywhere else argon and titanium.
"""

from __future__ import annotations

import numpy as np
import pytest
from plasmacoll.constants import BOLTZMANN, ELEMENTARY_CHARGE
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from wellbalanced.gasdynamics import (
    ARGON_MASS,
    Boundary,
    GasDynamics,
    IonExchange,
    VolumeSource,
    _cyclic_thomas,
    _thomas,
    binary_diffusion,
)

M_AR = ARGON_MASS
T0 = 300.0
P0 = 1.0


def exact_riemann(x, t, gamma, left, right, x0):
    """The exact solution of a Riemann problem with a left rarefaction and a right shock."""
    (rl, ul, pl), (rr, ur, pr) = left, right
    cl, cr = np.sqrt(gamma * pl / rl), np.sqrt(gamma * pr / rr)

    def f(p, rk, pk, ck):
        if p > pk:
            a = 2.0 / ((gamma + 1.0) * rk)
            b = (gamma - 1.0) / (gamma + 1.0) * pk
            return (p - pk) * np.sqrt(a / (p + b))
        return (
            2.0
            * ck
            / (gamma - 1.0)
            * ((p / pk) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        )

    ps = brentq(lambda p: f(p, rl, pl, cl) + f(p, rr, pr, cr) + ur - ul, 1e-12, 1e3)
    us = 0.5 * (ul + ur) + 0.5 * (f(ps, rr, pr, cr) - f(ps, rl, pl, cl))
    g = gamma
    rsl = rl * (ps / pl) ** (1.0 / g)
    csl = cl * (ps / pl) ** ((g - 1.0) / (2.0 * g))
    rsr = rr * (
        (ps / pr + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * ps / pr + 1.0)
    )
    shock = ur + cr * np.sqrt((g + 1.0) / (2.0 * g) * ps / pr + (g - 1.0) / (2.0 * g))
    out = np.empty((3, x.size))
    for i, xi in enumerate(x):
        s = (xi - x0) / t
        if s < ul - cl:
            r, u, p = rl, ul, pl
        elif s < us - csl:
            u = 2.0 / (g + 1.0) * (cl + (g - 1.0) / 2.0 * ul + s)
            c = 2.0 / (g + 1.0) * (cl + (g - 1.0) / 2.0 * (ul - s))
            r = rl * (c / cl) ** (2.0 / (g - 1.0))
            p = pl * (c / cl) ** (2.0 * g / (g - 1.0))
        elif s < us:
            r, u, p = rsl, us, ps
        elif s < shock:
            r, u, p = rsr, us, ps
        else:
            r, u, p = rr, ur, pr
        out[:, i] = (r, u, p)
    return out


def box(cells=40, cells_y=1, **settings) -> GasDynamics:
    """A periodic box of argon and titanium."""
    defaults = {
        "length": 0.01,
        "cells": cells,
        "cells_y": cells_y,
        "width": 0.01,
        "boundaries": {"left": Boundary.periodic(), "right": Boundary.periodic()},
    }
    return GasDynamics(**{**defaults, **settings})


# -- the linear solvers -------------------------------------------------------------------


def test_tridiagonal_solvers_match_a_dense_solve() -> None:
    rng = np.random.default_rng(1)
    n, m = 9, 3
    lower, upper = rng.random((n, m)) * -0.3, rng.random((n, m)) * -0.3
    diag = 1.0 + rng.random((n, m))
    rhs = rng.random((n, m))
    for cyclic in (False, True):
        solved = (_cyclic_thomas if cyclic else _thomas)(lower, diag, upper, rhs)
        for column in range(m):
            matrix = np.diag(diag[:, column])
            matrix += np.diag(lower[1:, column], -1) + np.diag(upper[:-1, column], 1)
            if cyclic:
                matrix[0, -1] = lower[0, column]
                matrix[-1, 0] = upper[-1, column]
            expected = np.linalg.solve(matrix, rhs[:, column])
            assert solved[:, column] == pytest.approx(expected, rel=1e-12)


def test_binary_diffusion_is_the_hard_sphere_self_diffusion_for_equal_atoms() -> None:
    d, t, p = 3.6e-10, 300.0, 1.0
    n = p / (BOLTZMANN * t)
    self_diffusion = 3.0 / (8.0 * n * d**2) * np.sqrt(BOLTZMANN * t / (np.pi * M_AR))
    assert binary_diffusion(M_AR, M_AR, d, t, p) == pytest.approx(
        self_diffusion, rel=1e-12
    )
    assert binary_diffusion(M_AR, M_AR, d, 4 * t, p) == pytest.approx(
        8.0 * self_diffusion, rel=1e-12
    )  # T^(3/2)


# -- the shock tube and the order of the scheme -------------------------------------------


def sod(cells: int, limiter: str = "vanleer") -> tuple[GasDynamics, np.ndarray]:
    model = GasDynamics(
        1.0,
        cells,
        gamma=1.4,
        gas_mass=1.0,
        metal_mass=1.0,
        limiter=limiter,
        boundaries={"left": Boundary.outflow(), "right": Boundary.outflow()},
    )
    left = model.x < 0.5
    rho = np.where(left, 1.0, 0.125)
    p = np.where(left, 1.0, 0.1)
    model.set_state(0.9 * rho, 0.1 * rho, pressure=p)
    model.run(0.2)
    exact = exact_riemann(model.x, 0.2, 1.4, (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.5)
    return model, exact


def test_sod_shock_tube_matches_the_exact_riemann_solution() -> None:
    errors = []
    for cells in (100, 200, 400):
        model, exact = sod(cells)
        state = model.state()
        density = state.gas_density[:, 0] + state.metal_density[:, 0]
        errors.append(np.mean(np.abs(density - exact[0])))
        assert np.mean(np.abs(state.u[:, 0] - exact[1])) < 0.05
        assert np.mean(np.abs(state.pressure[:, 0] - exact[2])) < 0.02
    assert errors[-1] < 2.0e-3
    # first order across the discontinuities, as for any shock-capturing scheme
    assert errors[0] / errors[-1] > 2.4
    assert errors[0] > errors[1] > errors[2]


def test_the_species_split_is_passive() -> None:
    """A uniform metal fraction stays uniform through a shock tube, to rounding."""
    model, _ = sod(200)
    state = model.state()
    fraction = state.metal_density / (state.metal_density + state.gas_density)
    assert fraction == pytest.approx(0.1, abs=1e-12)


def test_a_species_contact_moves_with_the_gas_and_leaves_u_and_p_alone() -> None:
    """A jump in the composition at uniform velocity and pressure is a contact: it is
    carried at the gas velocity, and the velocity and the pressure stay uniform."""
    model = box(
        200, length=1.0, width=1.0, gas_mass=1.0, metal_mass=4.0, limiter="minmod"
    )
    inside = (model.x > 0.2) & (model.x < 0.5)
    model.set_state(
        np.where(inside, 0.2, 1.0), np.where(inside, 0.8, 0.0), u=1.0, pressure=1.0
    )
    model.run(0.25)
    state = model.state()
    assert state.u == pytest.approx(1.0, abs=1e-9)
    assert state.pressure == pytest.approx(1.0, abs=1e-9)
    metal = state.metal_density[:, 0]
    assert metal.min() > -1e-9 and metal.max() < 0.8 + 1e-9
    centroid = np.sum(metal * model.x) / np.sum(metal)
    assert centroid == pytest.approx(0.35 + 0.25, abs=2e-3)


@pytest.mark.parametrize(
    "limiter, order", [("none", 1.9), ("vanleer", 1.4), ("mc", 1.5)]
)
def test_smooth_waves_converge_at_second_order(limiter: str, order: float) -> None:
    """An advected density wave, once round a periodic box: the order of the scheme."""
    errors = []
    sizes = (16, 32, 64)
    for cells in sizes:
        model = GasDynamics(
            1.0,
            cells,
            gamma=5.0 / 3.0,
            gas_mass=1.0,
            metal_mass=1.0,
            limiter=limiter,
            cfl=0.2,
            boundaries={"left": Boundary.periodic(), "right": Boundary.periodic()},
        )
        rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * model.x)
        model.set_state(rho, 0.0, u=1.0, pressure=1.0)
        model.run(1.0)
        errors.append(np.mean(np.abs(model.state().gas_density[:, 0] - rho)))
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert rates[-1] > order, (limiter, errors, rates)


# -- uniform states and conservation ------------------------------------------------------


@pytest.mark.parametrize("cells_y", [1, 6])
def test_a_uniform_state_stays_uniform(cells_y: int) -> None:
    """Walls and a reservoir at the state's own temperature and pressure change nothing."""
    model = GasDynamics(
        0.01,
        24,
        cells_y=cells_y,
        width=0.01,
        diffusion=3.0,
        thermal_conductivity=0.1,
        boundaries={
            "left": Boundary.wall(T0, accommodation=1.0),
            "right": Boundary.inflow(P0, T0),
        },
    )
    model.set_uniform(P0, T0)
    before = model._u.copy()
    for _ in range(60):
        model.step()
    assert model._u == pytest.approx(
        before, rel=1e-12, abs=1e-12 * np.abs(before).max()
    )
    state = model.state()
    assert np.max(np.abs(state.pressure - P0)) < 1e-12 * P0
    assert np.max(np.abs(state.temperature - T0)) < 1e-11 * T0
    assert np.max(np.abs(state.u)) + np.max(np.abs(state.v)) < 1e-9


def conserved_change(
    model: GasDynamics, steps: int
) -> tuple[dict[str, float], dict[str, float]]:
    start = model.totals()
    for _ in range(steps):
        model.step()
    end = model.totals()
    return {k: end[k] - start[k] for k in end}, dict(model.budget)


@pytest.mark.parametrize("cells_y", [1, 8])
def test_periodic_boxes_conserve_mass_momentum_and_energy(cells_y: int) -> None:
    model = box(
        32,
        cells_y,
        diffusion=lambda t, p: 5.0 * (t / T0) ** 1.5 / p,
        thermal_conductivity=lambda t: 0.05 * t / T0,
        limiter="mc",
    )
    x, y = np.meshgrid(model.x, model.y, indexing="ij")
    phase = 2 * np.pi * x / model.length
    gas = (
        1e-6
        * (1.0 + 0.3 * np.sin(phase))
        * (1.0 + 0.1 * np.cos(2 * np.pi * y / model.width))
    )
    metal = 3e-7 * (1.0 + 0.5 * np.cos(phase))
    model.set_state(
        gas,
        metal,
        u=30.0 + 20.0 * np.sin(phase),
        v=15.0 * np.cos(phase),
        pressure=P0 * (1 + 0.1 * np.cos(phase)),
    )
    change, budget = conserved_change(model, 80)
    scale = model.totals()
    for name in change:
        assert change[name] == pytest.approx(
            0.0, abs=1e-11 * max(abs(scale[name]), 1e-30)
        ), name
        assert budget[name] == pytest.approx(
            0.0, abs=1e-11 * max(abs(scale[name]), 1e-30)
        ), name


def test_boundaries_and_sources_account_for_every_change() -> None:
    """With emitting, sticking and heat-exchanging walls, an inflow and volume terms, the
    change of the totals is exactly what the budget says the boundaries and sources brought."""
    model = GasDynamics(
        0.01,
        30,
        cells_y=4,
        width=0.004,
        diffusion=2.0,
        thermal_conductivity=0.05,
        boundaries={
            "left": Boundary.wall(
                500.0,
                accommodation=0.7,
                gas_return_flux=np.array([1e21, 2e21, 3e21, 1e21]),
                metal_flux=2e21,
                metal_energy=5.0,
                metal_sticking=0.3,
            ),
            "right": Boundary.inflow(P0, T0),
        },
    )
    model.set_uniform(P0, T0, metal_fraction=0.05)
    model.sources.append(
        VolumeSource("metal", 1e22, temperature=1000.0, velocity=(20.0, 0.0))
    )
    model.sinks["gas"] = 5e21
    model.exchange.append(
        IonExchange("gas", 1e4, ion_velocity=(0.0, 50.0), ion_temperature=900.0)
    )
    change, budget = conserved_change(model, 120)
    totals = model.totals()
    for name in change:
        assert change[name] == pytest.approx(
            budget[name], rel=1e-9, abs=1e-12 * abs(totals[name])
        ), name
    assert budget["metal"] != 0.0 and budget["energy"] != 0.0


# -- the sources of the gas ---------------------------------------------------------------


def test_a_loss_frequency_removes_atoms_exponentially_at_the_local_state() -> None:
    model = box(8)
    model.set_state(1e-6, 0.0, u=40.0, pressure=P0)
    n0 = 1e-6 / M_AR
    nu, t_end = 3.0e3, 1.0e-4
    model.sink_frequencies["gas"] = nu
    model.run(t_end)
    state = model.state()
    assert state.gas_number == pytest.approx(n0 * np.exp(-nu * t_end), rel=1e-3)
    # the atoms leave with their own velocity and temperature: neither changes
    assert state.u == pytest.approx(40.0, rel=1e-9)
    assert state.temperature == pytest.approx(P0 / (n0 * BOLTZMANN), rel=1e-6)


def test_a_constant_sink_removes_atoms_linearly() -> None:
    model = box(8)
    model.set_state(1e-6, 0.0, pressure=P0)
    n0 = 1e-6 / M_AR
    model.sinks["gas"] = 0.3 * n0 / 1e-4
    model.run(1e-4)
    assert model.state().gas_number == pytest.approx(0.7 * n0, rel=1e-9)


def test_friction_with_the_ions_relaxes_the_velocity_exponentially() -> None:
    model = box(8)
    model.set_state(1e-6, 0.0, u=80.0, pressure=P0)
    nu = 2.0e3
    model.exchange.append(
        IonExchange("gas", nu, ion_velocity=(10.0, 0.0), ion_temperature=T0)
    )
    model.run(4.0e-4)
    state = model.state()
    expected = 10.0 + 70.0 * np.exp(-nu * model.time)
    assert state.u == pytest.approx(expected, rel=1e-3)
    assert np.all(state.temperature > T0)  # the friction heats the neutrals


def test_a_volume_source_creates_atoms_with_their_energy() -> None:
    model = box(8)
    model.set_uniform(P0, T0)
    before = model.totals()
    model.sources.append(VolumeSource("metal", 1e23, temperature=2000.0))
    model.run(1e-4)
    after = model.totals()
    volume = model.length * model.width
    created = 1e23 * 1e-4 * volume
    assert after["metal"] - before["metal"] == pytest.approx(
        created * model.masses["metal"], rel=1e-9
    )
    assert after["energy"] - before["energy"] == pytest.approx(
        created * 1.5 * BOLTZMANN * 2000.0, rel=1e-9
    )


# -- the walls ----------------------------------------------------------------------------


def closed_box(**wall) -> GasDynamics:
    model = GasDynamics(
        0.01,
        40,
        boundaries={"left": Boundary.wall(**wall), "right": Boundary.wall(**wall)},
    )
    return model


def test_an_emitting_wall_brings_the_mass_and_energy_of_its_flux() -> None:
    flux, energy_ev = 5e22, 6.0
    model = closed_box(metal_flux=flux, metal_energy=energy_ev)
    model.boundaries["right"] = Boundary.wall()
    model.set_uniform(P0, T0)
    before = model.totals()
    model.run(2e-5)
    after = model.totals()
    assert after["metal"] - before["metal"] == pytest.approx(
        flux * 2e-5 * model.masses["metal"] * model.width, rel=1e-9
    )
    assert after["energy"] - before["energy"] == pytest.approx(
        flux * 2e-5 * energy_ev * ELEMENTARY_CHARGE * model.width, rel=1e-9
    )
    # the wind pushes the gas away from the wall (later the gas strikes the far wall and the
    # net momentum of the closed box reverses, so look at an early time)
    early = closed_box(metal_flux=flux, metal_energy=energy_ev)
    early.boundaries["right"] = Boundary.wall()
    early.set_uniform(P0, T0)
    early.run(1e-6)
    assert early.totals()["momentum_x"] > 0.0


def test_a_hot_wall_heats_the_gas_at_the_free_molecular_rate_and_relaxes_to_its_temperature() -> (
    None
):
    alpha, t_wall = 0.8, 600.0
    model = closed_box(temperature=t_wall, accommodation=alpha)
    model.thermal_conductivity = 0.02  # the wall's heat reaches the bulk by conduction
    model.set_uniform(P0, T0)
    n = P0 / (BOLTZMANN * T0)
    thermal = np.sqrt(8 * BOLTZMANN * T0 / (np.pi * M_AR))
    expected_power = (
        2 * alpha * (n * thermal / 4) * 2 * BOLTZMANN * (t_wall - T0)
    )  # two walls
    start = model.totals()["energy"]
    model.run(2e-7)
    rate = (model.totals()["energy"] - start) / model.time / model.width
    assert rate == pytest.approx(expected_power, rel=0.05)
    model.run(3e-3)
    assert model.state().temperature.mean() == pytest.approx(t_wall, rel=1e-3)


def test_a_sticking_wall_takes_the_metal_at_the_free_molecular_flux() -> None:
    model = GasDynamics(
        0.01,
        40,
        boundaries={
            "left": Boundary.wall(metal_sticking=1.0),
            "right": Boundary.wall(),
        },
    )
    model.set_uniform(P0, T0, metal_fraction=0.2)
    n_metal = model.state().metal_number[0, 0]
    thermal = np.sqrt(8 * BOLTZMANN * T0 / (np.pi * model.masses["metal"]))
    expected = -n_metal * thermal / 4 * model.masses["metal"]  # kg per m^2 and s
    start = model.totals()["metal"]
    model.run(1e-7)
    rate = (model.totals()["metal"] - start) / model.time / model.width
    assert rate == pytest.approx(expected, rel=0.02)


def test_the_return_of_the_ions_replaces_what_the_plasma_ionizes() -> None:
    """An ionization sink that is balanced by the return flux of the neutralized ions keeps
    the gas: the loss per volume times the length equals the flux from the wall."""
    length = 0.01
    sink = 2e21 / length
    model = GasDynamics(
        length,
        40,
        boundaries={
            "left": Boundary.wall(T0, gas_return_flux=2e21),
            "right": Boundary.wall(T0),
        },
    )
    model.set_uniform(P0, T0)
    model.sinks["gas"] = sink
    start = model.totals()["gas"]
    model.run(3e-5)
    assert model.totals()["gas"] == pytest.approx(start, rel=2e-3)


# -- diffusion ----------------------------------------------------------------------------


def test_a_small_metal_perturbation_diffuses_at_the_diffusion_coefficient() -> None:
    d = 0.4
    model = box(64, diffusion=d)
    eps, k = 1e-3, 2 * np.pi / model.length
    fraction = eps * (1.0 + np.sin(k * model.x))
    m_g, m_m = model.masses["gas"], model.masses["metal"]
    # uniform number density: the mass densities follow from the metal fraction
    n = P0 / (BOLTZMANN * T0)
    metal = n * fraction * m_m
    gas = n * (1.0 - fraction) * m_g
    model.set_state(gas, metal, pressure=P0)
    t_end = 0.2 / (d * k**2)
    model.run(t_end)
    state = model.state()
    y = state.metal_density / (state.metal_density + state.gas_density)
    amplitude = np.sum((y[:, 0] - y[:, 0].mean()) * np.sin(k * model.x)) * 2 / model.nx
    initial_y = metal / (metal + gas)
    initial_amplitude = (
        np.sum((initial_y - initial_y.mean()) * np.sin(k * model.x)) * 2 / model.nx
    )
    assert amplitude / initial_amplitude == pytest.approx(
        np.exp(-d * k**2 * t_end), rel=0.01
    )
    assert state.metal_density.sum() == pytest.approx(metal.sum(), rel=1e-12)


def test_the_steady_sputter_wind_into_argon_is_the_stefan_diffusion_profile() -> None:
    d, length = 0.2, 0.01
    n = P0 / (BOLTZMANN * T0)
    flux = (
        0.5 * n * d / length
    )  # Gamma L / (n D) = 0.5: a flux that makes a clear profile
    model = GasDynamics(
        length,
        50,
        diffusion=d,
        boundaries={
            "left": Boundary.wall(
                T0,
                metal_flux=flux,
                metal_energy=2.0 * BOLTZMANN * T0 / ELEMENTARY_CHARGE,
            ),
            "right": Boundary.inflow(P0, T0),
        },
    )
    model.set_uniform(P0, T0)
    model.run(4.0 * length**2 / d)
    state = model.state()
    fraction = state.metal_density[:, 0] / (
        state.metal_density[:, 0] + state.gas_density[:, 0]
    )
    expected = solve_ivp(
        lambda x, y: (
            -flux
            * model.masses["metal"]
            * (1.0 - y)
            * ((1.0 - y) / model.masses["gas"] + y / model.masses["metal"])
            / (n * d)
        ),
        (length, 0.0),
        [0.0],
        dense_output=True,
        rtol=1e-10,
        atol=1e-14,
    ).sol(model.x)[0]
    assert fraction.max() > 0.2  # a real profile, not a trace
    assert np.max(np.abs(fraction - expected)) < 0.03 * fraction.max()
    # the argon does not flow through the wall: it is stationary in number, the gas leaves
    # as the metal arrives, at the Stefan flow speed of the mixture
    assert np.max(np.abs(state.pressure - P0)) < 2e-3 * P0
    # the atoms are emitted with 2 kT and leave with the enthalpy 5/2 kT: the gas is a
    # little cooler (the wall supplies the difference through the heat exchange)
    assert np.max(np.abs(state.temperature - T0)) < 0.03 * T0


# -- two dimensions -----------------------------------------------------------------------


def test_a_two_dimensional_gas_uniform_along_y_is_the_one_dimensional_gas() -> None:
    def wall_case(cells_y: int) -> GasDynamics:
        model = GasDynamics(
            0.01,
            30,
            cells_y=cells_y,
            width=0.004,
            diffusion=1.0,
            boundaries={
                "left": Boundary.wall(T0, metal_flux=3e21, gas_return_flux=1e21),
                "right": Boundary.inflow(P0, T0),
            },
        )
        model.set_uniform(P0, T0)
        model.sinks["gas"] = 4e21
        for _ in range(120):
            model.step(2.0e-7)
        return model

    one, two = wall_case(1), wall_case(8)
    a, b = one.state(), two.state()
    for name in ("gas_density", "metal_density", "u", "pressure"):
        assert getattr(b, name)[:, 3] == pytest.approx(
            getattr(a, name)[:, 0], rel=1e-10, abs=1e-12
        )
    assert np.max(np.abs(b.v)) < 1e-9


def test_a_two_dimensional_wave_is_advected_along_a_diagonal_at_second_order() -> None:
    errors = []
    for cells in (16, 32):
        model = GasDynamics(
            1.0,
            cells,
            width=1.0,
            cells_y=cells,
            gas_mass=1.0,
            metal_mass=1.0,
            cfl=0.3,
            limiter="none",
            boundaries={"left": Boundary.periodic(), "right": Boundary.periodic()},
        )
        x, y = np.meshgrid(model.x, model.y, indexing="ij")
        rho = 1.0 + 0.2 * np.sin(2 * np.pi * (x + y))
        model.set_state(rho, 0.0, u=1.0, v=1.0, pressure=1.0)
        model.run(1.0)
        errors.append(np.mean(np.abs(model.state().gas_density - rho)))
    assert np.log2(errors[0] / errors[1]) > 1.8


def test_a_wind_from_a_patch_of_the_target_drives_a_lateral_flow() -> None:
    """A wall that emits hot atoms over a quarter of the azimuth builds a pressure there that
    pushes the gas sideways at hundreds of m/s, carrying the metal round the ring; the
    solution is symmetric about the middle of the patch and the metal is all accounted for."""
    cells_y = 16
    flux = np.zeros(cells_y)
    flux[:4] = 4e22
    model = GasDynamics(
        0.01,
        20,
        cells_y=cells_y,
        width=0.02,
        diffusion=0.5,
        boundaries={
            "left": Boundary.wall(T0, metal_flux=flux, metal_energy=8.0),
            "right": Boundary.inflow(P0, T0),
        },
    )
    model.set_uniform(P0, T0)
    model.run(5e-6)
    state = model.state()
    assert np.abs(state.v).max() > 300.0
    column = state.metal_number.sum(axis=0)
    assert column[4:6].min() > 0.2 * column[:4].min()  # beyond the patch already
    assert (
        column[8:12].max() < column[:4].min()
    )  # the far side has not yet been reached
    mirror = np.roll(
        column[::-1], 4
    )  # cell j to 3 - j: the mirror about the patch middle
    assert column == pytest.approx(mirror, rel=1e-6)
    model.run(2e-5)
    assert model.state().metal_number.sum(axis=0)[8:12].min() > 0.0
    # every metal atom is the emitted ones, less the ones that left through the reservoir
    emitted = flux.sum() * model.dy * model.time * model.masses["metal"]
    assert model.totals()["metal"] <= emitted * (1.0 + 1e-9)
    assert model.totals()["metal"] - model.budget["metal"] == pytest.approx(
        0.0, abs=1e-9 * emitted
    )


def test_a_small_metal_perturbation_diffuses_along_y_at_the_diffusion_coefficient() -> (
    None
):
    d = 0.4
    model = box(4, cells_y=48, diffusion=d)
    eps, k = 1e-3, 2 * np.pi / model.width
    fraction = eps * (1.0 + np.sin(k * model.y))[None, :] * np.ones((model.nx, 1))
    n = P0 / (BOLTZMANN * T0)
    gas, metal = (
        n * (1.0 - fraction) * model.masses["gas"],
        n * fraction * model.masses["metal"],
    )
    model.set_state(gas, metal, pressure=P0)
    t_end = 0.2 / (d * k**2)
    model.run(t_end)
    state = model.state()
    y = state.metal_density / (state.metal_density + state.gas_density)
    y0 = metal / (metal + gas)  # the mass fraction, which is what diffuses
    amplitude = np.sum((y[0] - y[0].mean()) * np.sin(k * model.y)) * 2 / model.ny
    initial = np.sum((y0[0] - y0[0].mean()) * np.sin(k * model.y)) * 2 / model.ny
    assert amplitude / initial == pytest.approx(np.exp(-d * k**2 * t_end), rel=0.01)


def test_invalid_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        GasDynamics(1.0, 1)
    with pytest.raises(ValueError):
        GasDynamics(1.0, 10, limiter="superbee")
    with pytest.raises(ValueError):
        GasDynamics(1.0, 10, boundaries={"left": Boundary.periodic()})
    with pytest.raises(RuntimeError):
        GasDynamics(1.0, 10).step()
    with pytest.raises(ValueError):
        Boundary("magic")
