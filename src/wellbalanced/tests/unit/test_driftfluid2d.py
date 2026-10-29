"""The 2D drift fluid: symmetry, the E x B drift, the dispersion relation, the diagnostics.

The rates and transport coefficients are synthetic; the tests check the numerics.
"""

from __future__ import annotations

import numpy as np
import pytest

from wellbalanced.driftfluid2d import (
    DriftFluid2D,
    Run,
    argon_excitation_rate,
    argon_ionization_rate,
    argon_momentum_rate,
    boeuf_dispersion_roots,
    exponential_field,
    fit_mode,
    gradient_drift_roots,
    mode_amplitudes,
    spoke_report,
    uniform_plasma_roots,
)

E = 1.602176634e-19
ION_MASS = 39.948 * 1.66054e-27


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
    density = 1e15 * np.exp(-((x - 0.01) ** 2 + (y - 0.02) ** 2) / 0.004**2) + floor
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
    displacement = (
        displacement + model.length_y / 2
    ) % model.length_y - model.length_y / 2
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
        [1.0 + 1e-3 * np.exp(growth * t) * np.cos(k * (y - speed * t)) for t in times]
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


def uniform_plasma(ny: int = 32, nx: int = 4, **settings) -> DriftFluid2D:
    """A uniform plasma, periodic in x and y, in a uniform axial field of -2200 V/m."""
    defaults = {
        "voltage": 0.0,
        "magnetic_field": 0.02,
        "gas_density": 1.43e20,
        "electron_temperature": 5.0,
        "momentum_rate": 2e7 / 1.43e20,
        "ion_friction_rate": 1e6 / 1.43e20,
        "ionization_rate": lambda te: 0.0 * te,
        "refill_rate": 0.0,
        "secondary_emission": 0.0,
        "ion_temperature": 1e-4,
        "field_bc": -2200.0,
        "periodic_x": True,
    }
    return DriftFluid2D(0.0012 * nx, 0.016, nx, ny, **{**defaults, **settings})


@pytest.mark.parametrize("mode", [1, 2])
def test_the_uniform_plasma_grows_as_the_dispersion_relation_says(mode: int) -> None:
    # An x-uniform perturbation of a uniform plasma drifting in crossed fields: the
    # growth rate and the phase speed are those of the full-Poisson dispersion relation
    # (an E x B-driven ion-acoustic-like branch with omega ~ gamma), to ~1 %
    # at 32 cells per wavelength.
    density = 1e16
    model = uniform_plasma()
    state = model.initial_state(density, electron_temperature=5.0)
    state.ion_velocity[0] = E * -2200.0 / (model.ion_mass * 1e6)
    state = model.step(state, 1e-13)
    model.stationary_sources(state)
    k = 2 * np.pi * mode / model.length_y
    wave = 1.0 + 1e-4 * np.cos(k * model.y)[None, :]
    state.electron_density = state.electron_density * wave
    state.ion_density = state.ion_density * wave
    roots = uniform_plasma_roots(
        0.0, k, density, -2200.0, 0.02, 5.0, model.ion_mass, 1e6, 2e7
    )
    growing = roots[np.argmax(roots.real)]
    duration = 3.0 / growing.real
    run = model.run(state, duration, record_every=duration / 120, cfl=0.4)
    window = (0.45 * duration, duration)
    fit = fit_mode(run.times, run.amplitudes, mode, model.length_y, window)
    assert fit.growth_rate == pytest.approx(growing.real, rel=0.04)
    assert fit.frequency == pytest.approx(-growing.imag, rel=0.04)
    assert fit.phase_speed > 0.0  # along +y, the E x B direction


def test_a_density_gradient_is_carried_by_the_azimuthal_field() -> None:
    # the Hall tensor with a density gradient: a potential perturbation moves the
    # density along x, d(delta n)/dt = n_0 phi_1 [k^2 mu_perp cos(ky) + (k mu_perp h / L) sin(ky)]
    gradient_length = 0.02
    model = DriftFluid2D(
        0.04,
        0.016,
        24,
        48,
        voltage=0.0,
        magnetic_field=0.02,
        gas_density=1.43e20,
        electron_temperature=5.0,
        momentum_rate=2e7 / 1.43e20,
        field_bc=0.0,
    )
    n0 = 1e16 * np.exp(-model.x / gradient_length)[:, None] * np.ones((1, model.ny))
    phi = np.zeros_like(n0)
    k = 2 * np.pi * 2 / model.length_y
    phi_1 = 0.5
    perturbed = phi + phi_1 * np.cos(k * model.y)[None, :]
    temperature = np.full_like(n0, 5.0)
    gas = np.full_like(n0, 1.43e20)

    def tendency(potential: np.ndarray) -> np.ndarray:
        fields = model._fields(potential)
        gx, gy, _ = model._electron_like_flux(n0, temperature, fields, gas)
        gx[0] = gx[-1] = 0.0
        return -(model._divx @ gx.ravel() + model._divy @ gy.ravel()).reshape(n0.shape)

    response = (tendency(perturbed) - tendency(phi))[8:16]
    omega = E * 0.02 / 9.1093837e-31
    nu = 2e7
    mu_perp = E * nu / (9.1093837e-31 * (nu**2 + omega**2))
    h = omega / nu
    y = model.y[None, :]
    expected = (
        n0[8:16]
        * phi_1
        * (
            k**2 * mu_perp * np.cos(k * y)
            + k * mu_perp * h / gradient_length * np.sin(k * y)
        )
    )
    assert np.abs(response - expected).max() < 0.03 * np.abs(expected).max()


def test_the_dispersion_relation_is_damped_without_a_drive() -> None:
    # no field and no gradient: ion-acoustic and diffusive roots, all damped
    roots = gradient_drift_roots(
        2 * np.pi / 0.008, 1e9, 0.0, 0.006, 5.0, ION_MASS, 1e6, 2e7
    )
    assert roots.imag.max() < 0.0
    # with the density the Poisson equation is part of the relation, and for a
    # large density it approaches the quasi-neutral roots in the gradient-drift case
    quasi = gradient_drift_roots(
        2 * np.pi / 0.008, 0.01, -2200.0, 0.006, 5.0, ION_MASS, 1e6, 2e7
    )
    full = gradient_drift_roots(
        2 * np.pi / 0.008, 0.01, -2200.0, 0.006, 5.0, ION_MASS, 1e6, 2e7, density=1e22
    )
    assert quasi.imag.max() > 1e6
    assert full.imag.max() == pytest.approx(quasi.imag.max(), rel=0.02)


BOEUF_18MM = {
    "density": 0.8e15,
    "e_field": -2200.0,
    "b_field": 6e-3,
    "electron_temperature": 11.4,
    "density_length": -2.2e-3,
    "field_length": -6e-3,
    "electron_collision_frequency": 2e7,
}


def _fastest(parameters: dict) -> tuple[float, float, float]:
    """Largest growth rate of Boeuf's relation over 0.3-8 cm, its wavelength and speed."""
    best = (-np.inf, 0.0, 0.0)
    for wavelength in np.linspace(0.003, 0.08, 300):
        k = 2 * np.pi / wavelength
        roots = boeuf_dispersion_roots(k, **parameters)
        root = roots[np.argmax(roots.imag)]
        if root.imag > best[0]:
            best = (float(root.imag), float(wavelength), float(root.real / k))
    return best


def test_boeuf_relation_gives_his_linear_phase_and_hinges_on_the_gradient() -> None:
    # his numbers at x = 1.8 cm: 1.1e7 /s at k = 400 /m (1.6 cm), 17 km/s along +E x B
    gamma, wavelength, speed = _fastest(BOEUF_18MM)
    assert gamma == pytest.approx(1.1e7, rel=0.1)
    assert wavelength == pytest.approx(0.016, rel=0.1)
    assert speed == pytest.approx(17e3, rel=0.1)
    # the drive is the density gradient: at L_n = -5 mm (the drift fluid's 1D state) the
    # growth rate falls by two orders of magnitude
    gentle = _fastest({**BOEUF_18MM, "density_length": -5e-3})
    assert gentle[0] < 0.02 * gamma


def test_the_fitted_rates_are_bounded_and_give_the_quoted_collision_frequency() -> None:
    te = np.logspace(-2, 4, 60)
    for rate in (argon_ionization_rate, argon_excitation_rate, argon_momentum_rate):
        values = rate(te)
        assert np.all(np.isfinite(values)) and np.all(values > 0.0)
        assert values.max() < 1e-12  # no rate grows without bound with a runaway T_e
    # nu_en = k N is close to the 2e7 /s Boeuf quotes at 0.6 Pa (N = 1.45e20)
    assert argon_momentum_rate(3.0) * 1.45e20 == pytest.approx(2e7, rel=0.2)


def test_the_cathode_emits_the_prescribed_current() -> None:
    model = DriftFluid2D(
        0.02,
        0.016,
        16,
        8,
        voltage=260.0,
        magnetic_field=exponential_field(0.12, 0.006),
        cathode_current_density=0.03,
    )
    state = model.initial_state(1e15)
    ion_flux = np.full(model.ny, -1e18)
    assert model._cathode_flux(ion_flux) == pytest.approx(0.03 / E)
    model.cathode_current_density = None
    assert model._cathode_flux(ion_flux) == pytest.approx(0.06 * 1e18)
    model.cathode_current_density = 0.03
    u1, phi = model._euler(model._pack(state), state.potential, 1e-10)
    assert model.charge_residual(model._unpack(u1, phi, 1e-10)) < 1e-6


def _electrode_model(**settings) -> DriftFluid2D:
    return DriftFluid2D(
        0.02,
        0.016,
        16,
        8,
        voltage=260.0,
        magnetic_field=exponential_field(0.12, 0.006),
        cathode_current_density=0.03,
        **settings,
    )


def test_the_anode_sheath_holds_back_the_wall_flux_by_its_boltzmann_factor() -> None:
    """Above the plasma the anode takes ``n v_bar / 4``; below it, that times exp(-dphi/T)."""
    model = _electrode_model(anode_sheath=True)
    state = model.initial_state(1e15)
    ne = state.electron_density
    te = np.full_like(ne, 4.0)
    gas = state.gas_density
    gas_c = model._collision_gas(gas, te)
    saturation = 0.25 * 1e15 * np.sqrt(8.0 * E * 4.0 / (np.pi * 9.1093837015e-31))
    zero = np.zeros(model.ny)
    # E_x < 0: the anode above the plasma, electrons drift to it
    toward = np.full(model.ny, -100.0)
    attracting = model._anode_flux(ne, te, gas_c, toward, zero)
    model.anode_sheath = False
    drift = model._anode_flux(ne, te, gas_c, toward, zero)
    model.anode_sheath = True
    assert np.all(drift > 0.0)
    assert attracting == pytest.approx(saturation + drift)
    # E_x > 0: the plasma 3 T_e above the anode over the half cell, no drift towards it
    field = 3.0 * 4.0 / (0.5 * model.dx)
    repelling = model._anode_flux(ne, te, gas_c, np.full(model.ny, field), zero)
    assert repelling == pytest.approx(saturation * np.exp(-3.0))
    model.anode_sheath = False
    assert np.all(model._anode_flux(ne, te, gas_c, np.full(model.ny, field), zero) == 0)


def test_an_absorbing_cathode_takes_the_electrons_driven_back() -> None:
    """The emitted flux, less the drift into the cathode where the field drives it there."""
    model = _electrode_model(cathode_absorbs=True)
    state = model.initial_state(1e15)
    ne = state.electron_density
    te = np.full_like(ne, 4.0)
    gas_c = model._collision_gas(state.gas_density, te)
    ion_flux = np.full(model.ny, -1e18)
    zero = np.zeros(model.ny)
    emitted = 0.03 / E
    # E_x < 0 at the cathode: electrons pushed away from it, only the emission
    away = model._cathode_flux(ion_flux, ne, te, gas_c, np.full(model.ny, -1e4), zero)
    assert away == pytest.approx(emitted)
    # E_x > 0: the field drives electrons into the cathode, which takes them
    back = model._cathode_flux(ion_flux, ne, te, gas_c, np.full(model.ny, 1e5), zero)
    assert np.all(back < emitted)
    model.cathode_absorbs = False
    assert model._cathode_flux(
        ion_flux, ne, te, gas_c, np.full(model.ny, 1e5), zero
    ) == pytest.approx(emitted)
    # with both options a stage keeps the discrete charge balance
    model = _electrode_model(cathode_absorbs=True, anode_sheath=True)
    u1, phi = model._euler(model._pack(state), state.potential, 1e-10)
    assert model.charge_residual(model._unpack(u1, phi, 1e-10)) < 1e-6


def test_the_energy_loss_is_implicit_so_a_huge_step_stays_bounded() -> None:
    model = DriftFluid2D(
        0.02,
        0.016,
        16,
        8,
        voltage=260.0,
        magnetic_field=exponential_field(0.12, 0.006),
        electron_temperature=None,
    )
    state = model.initial_state(1e15, electron_temperature=5.0)
    u = model._pack(state)
    for _ in range(3):  # far above the explicit limit of the energy equation
        u, phi = model._euler(u, state.potential, 1e-8)
    te = u["w"] / (1.5 * u["ne"])
    assert np.all(np.isfinite(te))
    assert te.max() < 200.0


def test_a_short_1d_run_with_the_energy_equation_does_not_run_away() -> None:
    gas = 0.6 / (1.380649e-23 * 300.0)
    model = DriftFluid2D(
        0.02,
        0.008,
        16,
        4,
        voltage=260.0,
        magnetic_field=exponential_field(0.12, 0.006),
        gas_density=gas,
        electron_temperature=None,
        cathode_current_density=0.03,
    )
    state = model.initial_state(1e15, electron_temperature=3.0)
    run = model.run(state, 2e-7)
    final = run.state
    assert final.electron_density.max() < 5e15
    assert final.electron_temperature.max() < 40.0
    assert np.all(np.isfinite(final.potential))


@pytest.mark.parametrize("mode", [1, 2])
def test_the_jacobian_and_the_time_domain_agree_with_the_dispersion_relation(
    mode: int,
) -> None:
    # three independent routes to the growth rate and the phase speed of a mode of a uniform
    # plasma: the full-Poisson dispersion relation, the eigenvalues of the discretised operator
    # (linear_modes) and a time-domain run fitted in its exponential-growth window
    density = 1e16
    model = uniform_plasma(ny=32)
    state = model.initial_state(density, electron_temperature=5.0)
    state.ion_velocity[0] = E * -2200.0 / (model.ion_mass * 1e6)
    state = model.step(state, 1e-13)
    k = 2 * np.pi * mode / model.length_y
    roots = uniform_plasma_roots(
        0.0, k, density, -2200.0, 0.02, 5.0, model.ion_mass, 1e6, 2e7
    )
    growing = roots[np.argmax(roots.real)]
    sources = model.external_sources
    modes = model.linear_modes(state, mode)
    assert model.external_sources is sources  # restored
    gamma, omega, speed = modes.leading()
    assert gamma == pytest.approx(growing.real, rel=0.03)
    assert omega == pytest.approx(-growing.imag, rel=0.03)
    assert speed == pytest.approx(-growing.imag / k, rel=0.03)
    assert speed > 0.0
    # the time domain, started from the equilibrium and a small perturbation
    model.stationary_sources(state)
    wave = 1.0 + 1e-4 * np.cos(k * model.y)[None, :]
    state.electron_density = state.electron_density * wave
    state.ion_density = state.ion_density * wave
    duration = 3.0 / gamma
    run = model.run(state, duration, record_every=duration / 120)
    fit = fit_mode(
        run.times, run.amplitudes, mode, model.length_y, (0.45 * duration, duration)
    )
    assert fit.growth_rate == pytest.approx(gamma, rel=0.04)
    assert fit.frequency == pytest.approx(omega, rel=0.04)


def test_frozen_responses_change_the_eigenvalues_and_leave_the_model_alone() -> None:
    gas = 0.6 / (1.380649e-23 * 300.0)
    model = DriftFluid2D(
        0.02,
        0.08,
        12,
        16,
        voltage=260.0,
        magnetic_field=exponential_field(0.12, 0.006),
        gas_density=gas,
        electron_temperature=None,
        cathode_current_density=0.03,
    )
    base = model.initial_state(1e15, electron_temperature=5.0)
    for _ in range(40):
        base = model.step(base, model.suggest_timestep(base))
    full = model.linear_modes(base, 2)
    frozen = model.linear_modes(base, 2, frozen=("temperature", "ionization", "gas"))
    assert model.electron_temperature is None  # restored
    assert full.matrix.shape[0] > frozen.matrix.shape[0]  # no energy, no gas unknowns
    assert not np.allclose(full.growth_rates[0], frozen.growth_rates[0])
    with pytest.raises(ValueError):
        tilted = base.copy()
        tilted.electron_density = tilted.electron_density * (
            1.0 + 0.1 * np.arange(16)[None, :] / 16
        )
        model.linear_modes(tilted, 2)
