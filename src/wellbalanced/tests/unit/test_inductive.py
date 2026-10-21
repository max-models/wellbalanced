"""The inductive field: line currents, the skin effect, power balance, conductors, time.

Closed forms: a line current ``A_z = -mu0 I / (2 pi) ln r``, its field
``mu0 I / (2 pi r)``; a conducting half-space ``A_z ~ exp(-sqrt(i omega mu0 sigma) x)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy.constants import mu_0 as MU_0
from simgeometry import Circle, Rectangle

from wellbalanced.inductive import (
    Conductor,
    InductiveField2D,
    Winding,
    stretched_edges,
)

FREQUENCY = 1.0e6


def square(centre, side):
    return Rectangle(centre, (side, side))


def line_current_model(spacing=1e-3, half=0.4, windings=None, **settings):
    edges = stretched_edges(-half, half, -0.03, 0.03, spacing, ratio=1.08)
    windings = windings or [Winding(square((0.0, 0.0), 2.0 * spacing))]
    return InductiveField2D(
        edges, edges, frequency=FREQUENCY, windings=windings, **settings
    )


def value_at(solution, array, x, y):
    i = np.argmin(np.abs(solution.x - x))
    j = np.argmin(np.abs(solution.y - y))
    return array[i, j], solution.x[i], solution.y[j]


def test_stretched_edges_cover_the_interval():
    edges = stretched_edges(-1.0, 2.0, 0.0, 0.5, 0.01, ratio=1.2)
    assert edges[0] == -1.0 and edges[-1] == 2.0
    assert np.all(np.diff(edges) > 0)
    core = np.diff(edges)[(edges[:-1] >= 0.0) & (edges[1:] <= 0.5)]
    np.testing.assert_allclose(core, 0.01)


def test_line_current_potential_and_field():
    model = line_current_model()
    solution = model.solve(100.0)
    a = solution.vector_potential
    # A(r1) - A(r2) = -mu0 I / (2 pi) ln(r1 / r2) for r small against the box
    a1, x1, _ = value_at(solution, a, 0.01, 0.0)
    a2, x2, _ = value_at(solution, a, 0.025, 0.0)
    expected = -MU_0 * 100.0 / (2 * np.pi) * np.log(abs(x1) / abs(x2))
    assert abs((a1 - a2) - expected) < 2e-3 * abs(expected)
    _, b_y = solution.magnetic_field
    for radius in (0.01, 0.015, 0.025):
        b, x, _ = value_at(solution, b_y, radius, 0.0)
        assert abs(abs(b) - MU_0 * 100.0 / (2 * np.pi * x)) < 5e-3 * abs(b)


def test_pair_of_opposite_currents():
    spacing, d = 1e-3, 0.02
    windings = [
        Winding(square((-d / 2, 0.0), 2 * spacing), +1.0),
        Winding(square((d / 2, 0.0), 2 * spacing), -1.0),
    ]
    model = line_current_model(windings=windings)
    solution = model.solve(50.0)
    a = solution.vector_potential
    for x, y in ((0.0, 0.01), (0.02, 0.005), (-0.025, -0.01)):
        value, xc, yc = value_at(solution, a, x, y)
        r1 = np.hypot(xc + d / 2, yc)
        r2 = np.hypot(xc - d / 2, yc)
        expected = -MU_0 * 50.0 / (2 * np.pi) * np.log(r1 / r2)
        assert abs(value - expected) < 5e-3 * MU_0 * 50.0 / (2 * np.pi)


def test_skin_depth_of_a_conducting_half_space():
    # 1D: periodic along y with one cell, a current sheet at x ~ 0, a plasma beyond
    density, nu = 1.0e18, 1.0e7
    edges_x = np.linspace(0.0, 0.2, 2001)
    plasma_start = 0.05
    density_of = lambda x, y: np.where(x > plasma_start, density, 0.0)  # noqa: E731
    model = InductiveField2D(
        edges_x,
        [0.0, 1.0],
        frequency=FREQUENCY,
        windings=[Winding(Rectangle((0.0105, 0.5), (0.001, 1.0)))],
        electron_density=density_of,
        collision_frequency=nu,
        boundaries=("dirichlet", "periodic"),
    )
    solution = model.solve(1.0)
    omega = 2 * np.pi * FREQUENCY
    sigma = density * ELEMENTARY_CHARGE**2 / (ELECTRON_MASS * (nu + 1j * omega))
    k = np.sqrt(1j * omega * MU_0 * sigma)
    a = solution.vector_potential[:, 0]
    x = solution.x
    i1 = np.argmin(np.abs(x - (plasma_start + 0.005)))
    i2 = np.argmin(np.abs(x - (plasma_start + 0.015)))
    k_numerical = -np.log(a[i2] / a[i1]) / (x[i2] - x[i1])
    assert abs(k_numerical - k) < 1e-3 * abs(k)


def plasma_model(**settings):
    edges = np.linspace(-0.1, 0.1, 81)
    windings = [
        Winding(Circle((-0.06, 0.0), 0.004), +1.0, "left"),
        Winding(Circle((0.06, 0.0), 0.004), -1.0, "right"),
    ]
    density = lambda x, y: 8e17 * np.exp(-(x**2 + y**2) / 0.03**2)  # noqa: E731
    return InductiveField2D(
        edges,
        edges,
        frequency=FREQUENCY,
        windings=windings,
        electron_density=density,
        collision_frequency=2e7,
        **settings,
    )


def test_power_balance_to_round_off():
    solution = plasma_model().solve(30.0 + 10.0j)
    assert solution.absorbed_power > 0.0
    assert abs(solution.source_power.real - solution.absorbed_power) < 1e-11 * abs(
        solution.source_power
    )


def test_efficiency_and_power_scaling():
    solution = plasma_model().solve(1.0)
    depth = 0.3
    target = solution.for_power(25e3, depth)
    assert abs(depth * target.absorbed_power - 25e3) < 1e-9 * 25e3
    r_p = target.plasma_resistance(depth)
    eta = target.efficiency(0.6, depth)
    current = abs(target.coil_current)
    assert abs(eta - 25e3 / (25e3 + 0.5 * 0.6 * current**2)) < 1e-12
    assert abs(r_p - solution.plasma_resistance(depth)) < 1e-12 * r_p


def test_grounded_and_floating_conductors():
    plates = [
        Rectangle((-0.085, 0.0), (0.006, 0.12)),
        Rectangle((0.085, 0.0), (0.006, 0.12)),
    ]
    grounded = plasma_model(conductors=[Conductor(plates, name="shield")])
    floating = plasma_model(
        conductors=[Conductor(plates, floating=True, name="shield")]
    )
    free = plasma_model().solve(1.0)
    for model in (grounded, floating):
        solution = model.solve(1.0)
        inside = model.conductor_masks[0]
        a = solution.vector_potential[inside]
        np.testing.assert_allclose(
            a, a[0], rtol=0, atol=1e-14 * np.abs(a).max() + 1e-30
        )
        assert abs(solution.source_power.real - solution.absorbed_power) < 1e-10 * abs(
            solution.source_power
        )
    floating_solution = floating.solve(1.0)
    assert abs(floating_solution.conductor_currents["shield"]) < 1e-9
    grounded_solution = grounded.solve(1.0)
    assert np.all(
        grounded_solution.vector_potential[grounded.conductor_masks[0]] == 0.0
    )
    # shields near the coil lower the plasma resistance at a fixed coil current
    assert grounded_solution.absorbed_power < free.absorbed_power


def test_regions_are_checked():
    edges = np.linspace(-0.1, 0.1, 21)
    with pytest.raises(ValueError, match="covers no cell"):
        InductiveField2D(
            edges,
            edges,
            frequency=FREQUENCY,
            windings=[Winding(Circle((0.001, 0.001), 1e-4))],
        )
    with pytest.raises(ValueError, match="overlap"):
        InductiveField2D(
            edges,
            edges,
            frequency=FREQUENCY,
            windings=[Winding(Circle((0.0, 0.0), 0.02))],
            conductors=[Conductor([Circle((0.0, 0.0), 0.03)])],
        )


def test_time_dependent_reaches_the_harmonic_solution():
    # a dense, collisional plasma: the transients decay within a period
    edges = np.linspace(-0.03, 0.03, 31)
    windings = [
        Winding(Rectangle((-0.02, 0.0), (0.004, 0.004)), +1.0),
        Winding(Rectangle((0.02, 0.0), (0.004, 0.004)), -1.0),
    ]
    model = InductiveField2D(
        edges,
        edges,
        frequency=FREQUENCY,
        windings=windings,
        electron_density=lambda x, y: np.where(np.abs(x) < 0.012, 1e18, 0.0),
        collision_frequency=1e9,
    )
    steps_per_period = 100
    dt = 1.0 / (FREQUENCY * steps_per_period)
    stepper = model.time_stepper(dt)
    state = stepper.initial_state()
    current = 10.0
    for n in range(1, 12 * steps_per_period + 1):
        state = stepper.step(state, current * np.cos(2 * np.pi * FREQUENCY * n * dt))
    exact = model.solve(current, frequency=model.modified_frequency(FREQUENCY, dt))
    phase = np.exp(1j * 2 * np.pi * FREQUENCY * state.time)
    expected = np.real(exact.vector_potential * phase).ravel()
    scale = np.abs(exact.vector_potential).max()
    assert np.abs(state.vector_potential - expected).max() < 1e-8 * scale
    expected_j = np.real(exact.plasma_current_density * phase).ravel()
    scale_j = np.abs(exact.plasma_current_density).max()
    assert np.abs(state.plasma_current_density - expected_j).max() < 1e-8 * scale_j
    # and the modified frequency is close to the true one at 100 steps per period
    true = model.solve(current)
    relative = np.abs(true.vector_potential - exact.vector_potential).max() / scale
    assert relative < 1e-2
