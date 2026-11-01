"""The local kinetic dispersion relation: its limits, the compressibility, the scans."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from wellbalanced.dispersion import (
    ARGON_MASS,
    ColdIons,
    E,
    KineticIons,
    LocalDispersion,
    LocalRoots,
    Plasma,
    plasma_dispersion_function,
    scan_profile,
)
from wellbalanced.driftfluid2d import boeuf_dispersion_roots

K = 2 * np.pi / 0.016  # Boeuf's 1.6 cm


def _clean(**overrides) -> Plasma:
    """A case where the corrections are below 1 %: ``k rho_e``, ``k^2 L_n v_E / omega_ce``,
    ``v_B / v_E = T_e / (E L_B)`` all small, ``S = e B^2 L_n / (m_i E)`` far below 4."""
    base = dict(
        density=1e16,
        e_field=-1e4,
        b_field=0.1,
        electron_temperature=3.0,
        density_length=-3e-3,
    )
    return Plasma(**{**base, **overrides})


def _growth(plasma: Plasma, ions=None, quasineutral=True) -> float:
    ions = ColdIons() if ions is None else ions
    return LocalDispersion(plasma, ions, quasineutral).roots(0.0, K).growth_rate


def test_plasma_dispersion_function_values_and_asymptotics() -> None:
    assert plasma_dispersion_function(0.0) == pytest.approx(1j * np.sqrt(np.pi))
    assert plasma_dispersion_function(25j) == pytest.approx(-1 / 25j, rel=1e-3)
    # the switch to the series at |zeta| = 30 is continuous
    inside = plasma_dispersion_function(30 - 1e-9)
    outside = plasma_dispersion_function(30 + 1e-9)
    assert abs(inside - outside) < 1e-6 * abs(inside)
    # vectorised and finite deep in the lower half-plane, where wofz overflows
    values = plasma_dispersion_function(np.array([1.0 + 1j, -2e3j, 50.0 - 50.0j]))
    assert np.all(np.isfinite(values))


def test_simon_hoh_limit_is_rayleigh_taylor_with_the_field_as_gravity() -> None:
    # cold ions, uniform B, quasineutral: Omega^2 - S Omega + S = 0, so for S << 1
    # gamma = k sqrt(e E L_n / m_i) and omega_r / k = c_s^2 / (2 v_*)
    plasma = _clean()
    roots = LocalDispersion(plasma, ColdIons(), quasineutral=True).roots(0.0, K)
    gamma = K * np.sqrt(E * 1e4 * 3e-3 / ARGON_MASS)
    assert roots.growth_rate == pytest.approx(gamma, rel=0.01)
    v_star = 3.0 / 0.1 / 3e-3
    assert roots.phase_speed == pytest.approx(
        E * 3.0 / ARGON_MASS / (2 * v_star), rel=0.05
    )
    s = (E * 3.0 / ARGON_MASS) / (v_star * plasma.exb_velocity)
    omega = roots.omega[0] / (K * plasma.exb_velocity)
    assert omega**2 - s * omega + s == pytest.approx(0.0, abs=1e-4)


def test_compressibility_of_the_model_field_and_of_a_vacuum_field() -> None:
    # 1 / L_eff = 1 / L_n - 1 / L_B (model field), - 2 / L_B (curl-free); gamma ~ sqrt(L_eff)
    uniform = _growth(_clean())
    model = _growth(_clean(field_length=-1.2e-2))
    vacuum = _growth(_clean(field_length=-1.2e-2, curl_free_field=True))
    assert model / uniform == pytest.approx(
        np.sqrt((1 / 3e-3) / (1 / 3e-3 - 1 / 1.2e-2)), rel=0.03
    )
    assert vacuum / model == pytest.approx(
        np.sqrt((1 / 3e-3 - 1 / 1.2e-2) / (1 / 3e-3 - 2 / 1.2e-2)), rel=0.02
    )
    # at L_n = L_B / 2 the vacuum field has no drive left: no growing root
    marginal = LocalDispersion(
        _clean(density_length=-6e-3, field_length=-1.2e-2, curl_free_field=True),
        ColdIons(),
        quasineutral=True,
    ).roots(0.0, K)
    assert marginal.growth_rate < 1e-3 * uniform


def test_poisson_closure_matches_boeuf_relation_without_grad_b() -> None:
    # large B (k rho_e -> 0), uniform B, no collisions: the two relations are the same cubic
    plasma = Plasma(0.8e15, -2200.0, 0.2, 11.4, -2.2e-3)
    mine = LocalDispersion(plasma, ColdIons(), quasineutral=False).roots(0.0, K).omega
    his = boeuf_dispersion_roots(
        K, 0.8e15, -2200.0, 0.2, 11.4, -2.2e-3, -6e-3, 0.0, grad_b=False
    )
    for root in his:
        assert np.min(np.abs(mine - root)) < 1e-3 * abs(root)


def test_grad_b_treatment_differs_from_boeuf_relation() -> None:
    # at his point the grad-B terms are as large as E x B; his factor 2 (growth 1e7 /s)
    # against the model-field compressibility here (near marginal): the PIC decides
    plasma = Plasma(0.8e15, -2200.0, 6e-3, 11.4, -2.2e-3, -6e-3, np.inf, 2e7)
    mine = LocalDispersion(plasma, ColdIons()).roots(0.0, K)
    his = boeuf_dispersion_roots(K, 0.8e15, -2200.0, 6e-3, 11.4, -2.2e-3, -6e-3, 2e7)
    assert his.imag.max() > 5e6
    assert 0.0 < mine.growth_rate < 0.1 * his.imag.max()


def test_kinetic_ions_reduce_to_cold_and_shift_with_the_flow() -> None:
    plasma = _clean()
    cold = LocalDispersion(plasma, ColdIons(), quasineutral=True).roots(0.0, K)
    very_cold = LocalDispersion(plasma, KineticIons(temperature=1e-4), True).roots(
        0.0, K
    )
    assert very_cold.omega[0] == pytest.approx(cold.omega[0], rel=1e-4)
    # at 300 K the fast-growing mode has zeta ~ 25 i: still the cold limit within 1 %
    warm = LocalDispersion(plasma, KineticIons(temperature=0.026), True).roots(0.0, K)
    assert warm.growth_rate == pytest.approx(cold.growth_rate, rel=0.01)
    assert warm.growth_rate < cold.growth_rate
    # an ion flow toward the cathode Doppler-shifts a tilted mode by k_x v_0x
    kx = 1 / 3e-3
    still = LocalDispersion(plasma, KineticIons(temperature=0.026), True).roots(kx, K)
    flow = KineticIons(temperature=0.026, flow=(-5e3, 0.0))
    moving = LocalDispersion(plasma, flow, True).roots(kx, K)
    shift = moving.omega[0].real - still.omega[0].real
    assert shift == pytest.approx(kx * -5e3, rel=0.05)
    # the electron-cyclotron root of the momentum balance is outside the drift ordering
    assert np.all(np.abs(still.omega) < 0.7 * plasma.omega_ce)


def test_cold_ion_flow_collisions_ionization_and_field_shear_are_smooth() -> None:
    plasma = _clean(
        field_length=-1.2e-2,
        e_field_length=-2e-2,
        electron_collision_frequency=1e7,
        ionization_frequency=1e5,
    )
    ions = ColdIons(flow=(-3e3, 1.5e3), collision_frequency=2e5)
    roots = LocalDispersion(plasma, ions).roots(0.0, K)
    reference = _growth(_clean(field_length=-1.2e-2))
    assert roots.omega.size >= 1
    assert 0.5 * reference < roots.growth_rate < 2.0 * reference


def test_scan_follows_the_branch_with_growth_linear_in_k() -> None:
    model = LocalDispersion(_clean(), ColdIons(), quasineutral=True)
    ky = np.array([200.0, 400.0, 800.0])
    scan = model.scan(ky)
    assert set(scan) == {
        "growth_rate",
        "frequency",
        "phase_speed",
        "omega_re",
        "omega_im",
    }
    assert scan["omega_re"].dims == ("ky", "root") and scan.attrs["branch"] == "drift"
    # the followed root is among the roots returned
    assert np.isclose(
        scan["omega_im"].max("root").values, scan["growth_rate"].values
    ).all()
    growth = scan["growth_rate"].values
    assert growth[1] / growth[0] == pytest.approx(2.0, rel=0.05)
    assert growth[2] / growth[1] == pytest.approx(2.0, rel=0.05)
    assert np.allclose(scan["phase_speed"].values, scan["frequency"].values / ky)


def test_scan_profile_grows_fastest_where_the_gradient_is_gentlest() -> None:
    # gamma = k sqrt(e E L_eff / m_i): the mode grows faster the longer L_eff, as long as the
    # gradient has the unstable sign; with no gradient there is no drive (uniform B: no shear)
    x = np.linspace(2e-3, 10e-3, 9)
    density = 1e16 * np.exp(-(((x - 2e-3) / 3e-3) ** 2))  # |L_n| falls with x
    profile = xr.Dataset(
        {
            "density": ("x", density),
            "e_field": ("x", np.full_like(x, -1e4)),
            "b_field": ("x", np.full_like(x, 0.1)),
            "electron_temperature": ("x", np.full_like(x, 3.0)),
        },
        coords={"x": x},
    )
    result = scan_profile(profile, K, quasineutral=True)
    assert set(result) >= {
        "growth_rate",
        "phase_speed",
        "density_length",
        "field_length",
    }
    growth = result["growth_rate"].values
    assert np.isfinite(growth).all()
    # np.gradient is one-sided at the edge, so x[0] has a finite, very gentle L_n: fastest
    assert growth[0] > growth[1] > growth[4] > growth[8] > 0.0
    assert np.all(np.diff(growth) < 0.0)


def test_the_drift_ordering_drops_the_electron_scale_roots() -> None:
    # Boeuf's profile at 14 mm (B 11.6 mT, 4.5e7 /s collisions, |L_n| > a): no gradient-drift
    # mode, a slow lower-hybrid root, and electron-scale roots at Re omega ~ 0.3 omega_ce
    plasma = Plasma(
        density=2.079e15,
        e_field=-6326.0,
        b_field=0.01164,
        electron_temperature=11.76,
        density_length=-8.431e-3,
        field_length=-5.993e-3,
        e_field_length=-6.29e-3,
        electron_collision_frequency=4.501e7,
    )
    ky = 2 * np.pi * 3 / 0.08
    roots = LocalDispersion(plasma, ColdIons()).roots(0.0, ky)
    assert roots.growth_rate > 1e7  # the electron-scale root is the fastest of all
    assert abs(roots.phase_speed) > 3 * plasma.exb_velocity
    drift = roots.drift(plasma)
    assert 0 < drift.omega.size < roots.omega.size
    assert drift.growth_rate < 1e6
    assert abs(drift.phase_speed) < 0.3 * plasma.exb_velocity
    assert (np.abs(drift.omega.imag) < 0.02 * plasma.omega_ce).all()
    omega_lh = np.sqrt(plasma.omega_ce * E * plasma.b_field / ARGON_MASS)
    assert abs(abs(drift.omega[0].real) - omega_lh) < 0.6 * omega_lh
    # the scans follow the drift branch by default and still return every root
    fastest = LocalDispersion(plasma, ColdIons()).scan(np.array([ky]), branch="fastest")
    following = LocalDispersion(plasma, ColdIons()).scan(np.array([ky]))
    assert float(fastest["growth_rate"][0]) > 1e7 > float(following["growth_rate"][0])
    assert np.nanmax(following["omega_im"].values) > 1e7
    with pytest.raises(ValueError, match="branch"):
        LocalDispersion(plasma, ColdIons()).scan(np.array([ky]), branch="ions")
    empty = LocalRoots(0.0, ky, np.array([], dtype=complex)).drift(plasma)
    assert empty.omega.size == 0 and np.isnan(empty.growth_rate)
