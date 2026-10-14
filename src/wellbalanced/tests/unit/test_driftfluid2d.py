"""The 2D drift fluid: symmetry, the E x B drift, the dispersion relation, the diagnostics.

The rates and transport coefficients are synthetic; the tests check the numerics.
"""

from __future__ import annotations

import numpy as np
import pytest

from wellbalanced.driftfluid2d import (
    DriftFluid2D,
    Run,
    exponential_field,
    fit_mode,
    gradient_drift_roots,
    mode_amplitudes,
    spoke_report,
)

E = 1.602176634e-19


def test_a_state_uniform_in_y_stays_uniform_to_round_off() -> None:
    model = DriftFluid2D(
        0.02,
        0.016,
        16,
        8,
        voltage=260.0,
        magnetic_field=exponential_field(0.12, 0.006),
        electron_temperature=None,
    )
    profile = 1e15 * np.exp(-(((model.x - 0.012) / 0.004) ** 2)) + 1e13
    state = model.initial_state(profile[:, None] * np.ones((1, 8)))
    dt = model.suggest_timestep(state)
    for _ in range(15):
        state = model.step(state, dt)
    fields = (
        state.electron_density,
        state.ion_density,
        state.potential,
        state.ion_velocity[0],
        state.electron_temperature,
    )
    for field in fields:
        assert np.abs(field - field.mean(axis=1, keepdims=True)).max() <= 1e-12 * (
            np.abs(field).max() + 1.0
        )


def test_a_passive_blob_moves_at_the_exb_speed() -> None:
    model = DriftFluid2D(
        0.02,
        0.04,
        20,
        64,
        voltage=100.0,
        magnetic_field=0.05,
        momentum_rate=1e-20,  # almost collisionless: h >> 1
        electron_temperature=1e-3,
    )
    x, y = np.meshgrid(model.x, model.y, indexing="ij")
    floor = 1e10
    density = (
        1e15 * np.exp(-((x - 0.01) ** 2 + (y - 0.02) ** 2) / 0.004**2) + floor
    )
    state = model.initial_state(density)
    fields = model._fields(state.potential)
    v_exb = -fields[4].mean() / 0.05  # -E_x / B
    assert v_exb == pytest.approx(1e5)

    def rhs(n: np.ndarray) -> np.ndarray:
        gx, gy, _ = model._electron_like_flux(
            n, state.electron_temperature, fields, state.gas_density
        )
        gx[0] = gx[-1] = 0.0
        flux = model._divx @ gx.ravel() + model._divy @ gy.ravel()
        return -flux.reshape(n.shape)

    duration = 4e-8
    steps = int(duration / (0.3 * model.dy / v_exb)) + 1
    dt = duration / steps
    n = density
    for _ in range(steps):
        k1 = n + dt * rhs(n)
        k2 = 0.75 * n + 0.25 * (k1 + dt * rhs(k1))
        n = n / 3.0 + 2.0 / 3.0 * (k2 + dt * rhs(k2))
    weight = (n - floor).sum(axis=0)
    phase = np.angle((weight * np.exp(2j * np.pi * model.y / model.length_y)).sum())
    displacement = phase / (2 * np.pi) * model.length_y - 0.02
    displacement = (displacement + model.length_y / 2) % model.length_y - model.length_y / 2
    assert displacement / duration == pytest.approx(v_exb, rel=1e-3)
    # the drift is azimuthal: no motion along x
    cx = (n - floor).sum(axis=1) @ model.x / (n - floor).sum()
    assert cx == pytest.approx(0.01, abs=1e-5)


def test_the_poisson_equation_holds_for_the_new_densities() -> None:
    model = DriftFluid2D(
        0.02,
        0.016,
        16,
        8,
        voltage=260.0,
        magnetic_field=exponential_field(0.12, 0.006),
        electron_temperature=5.0,
    )
    state = model.initial_state(1e15)
    dt = model.suggest_timestep(state)
    # one Euler stage is exact; the Runge-Kutta combination is consistent to O(dt)
    u0 = model._pack(state)
    u1, phi = model._euler(u0, state.potential, dt)
    stage = model._unpack(u1, phi, dt)
    assert model.charge_residual(stage) < 1e-6


def test_stationary_sources_hold_an_equilibrium() -> None:
    nx, ny = 12, 8
    model = DriftFluid2D(
        0.03,
        0.016,
        nx,
        ny,
        voltage=66.0,
        magnetic_field=0.006,
        electron_temperature=5.0,
        ionization_rate=lambda te: 0.0 * te,
        refill_rate=0.0,
        secondary_emission=0.0,
    )
    density = 1e16 * np.exp(-model.x / 0.01)[:, None] * np.ones((1, ny))
    state = model.step(model.initial_state(density), 1e-13)
    model.stationary_sources(state)
    after = state
    for _ in range(10):
        after = model.step(after, 2e-10)
    assert after.electron_density == pytest.approx(state.electron_density, rel=1e-6)
    assert after.ion_density == pytest.approx(state.ion_density, rel=1e-6)


def test_fit_mode_recovers_growth_and_phase_speed() -> None:
    length, m, ny = 0.016, 3, 32
    y = (np.arange(ny) + 0.5) * length / ny
    k = 2 * np.pi * m / length
    times = np.linspace(0.0, 2e-6, 60)
    growth, speed = 2e6, -350.0
    frames = np.array(
        [
            1.0 + 1e-3 * np.exp(growth * t) * np.cos(k * (y - speed * t))
            for t in times
        ]
    )
    amplitudes = mode_amplitudes(frames)
    fit = fit_mode(times, amplitudes, m, length)
    assert fit.growth_rate == pytest.approx(growth, rel=1e-6)
    assert fit.phase_speed == pytest.approx(speed, rel=1e-6)
    with pytest.raises(ValueError):
        fit_mode(times, amplitudes, m, length, window=(0.0, 1e-9))


def test_spoke_report_reads_a_synthetic_spoke() -> None:
    length, m, ny = 0.08, 5, 64
    y = (np.arange(ny) + 0.5) * length / ny
    k = 2 * np.pi * m / length
    times = np.linspace(0.0, 1e-4, 80)
    speed = 1500.0
    cathode = np.array([1e16 * (1 + 0.25 * np.cos(k * (y - speed * t))) for t in times])
    potential = np.array([2.0 + 9.0 * np.cos(k * (y - speed * t)) for t in times])
    e_y = np.array([-1e3 * np.sin(k * (y - speed * t)) for t in times])
    v_exb = np.full((len(times), ny), 2.0e5)
    run = Run(
        times,
        cathode,
        0.01,
        mode_amplitudes(cathode),
        None,  # type: ignore[arg-type]
        0,
        length,
        {
            "density_cathode": cathode,
            "potential": potential,
            "e_y": e_y,
            "v_exb_y": v_exb,
        },
    )
    report = spoke_report(run)
    assert report.mode == m
    assert report.phase_velocity == pytest.approx(speed, rel=1e-6)
    assert report.relative_direction == 1
    assert report.speed_ratio == pytest.approx(speed / 2e5, rel=1e-6)
    assert report.cathode_modulation == pytest.approx(0.25, abs=0.01)
    assert report.potential_hump == pytest.approx(9.0, abs=0.3)
    assert report.azimuthal_field == pytest.approx(1e3, rel=0.02)
    run.series["v_exb_y"] = -v_exb
    assert spoke_report(run).relative_direction == -1


def test_the_dispersion_relation_is_damped_without_a_drive() -> None:
    ion_mass = 39.948 * 1.66054e-27
    # no field and no gradient: ion-acoustic and diffusive roots, all damped
    roots = gradient_drift_roots(
        2 * np.pi / 0.008, 1e9, 0.0, 0.006, 5.0, ion_mass, 1e6, 2e7
    )
    assert roots.imag.max() < 0.0
    # the drift of the electrons (a field, no gradient) and the gradient drive it
    driven = gradient_drift_roots(
        2 * np.pi / 0.008, 0.01, -2200.0, 0.006, 5.0, ion_mass, 1e6, 2e7
    )
    assert driven.imag.max() > 1e6
    assert roots.size == 2
