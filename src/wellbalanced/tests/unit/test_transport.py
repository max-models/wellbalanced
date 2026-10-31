"""The momentum budget, the loop identity and the mode fluxes on constructed moments."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

from wellbalanced.transport import loop_identity, mode_fluxes, momentum_budget

E = ELEMENTARY_CHARGE
B = 0.02
NU = 2e7
LY = 0.08


def _moments(
    *, wave: float = 0.3, phase: float = 0.6, collisions: bool = True, nt: int = 3
) -> xr.Dataset:
    """Electron moments built so that the momentum identity holds exactly.

    ``n = n_0(x) (1 + wave cos(k y))``, ``phi = phi_1 cos(k y + phase)``, a Pedersen
    drift from ``nu`` and ``E_x``, the ``E x B`` drift ``E_y / B`` along ``x`` and a
    stress ``n <v_x v_y>`` whose divergence is prescribed; ``u_x`` is then the sum the
    identity says, so that the budget must close to rounding.
    """
    x = np.linspace(0.0, 0.02, 41)
    y = np.arange(64) * LY / 64
    t = np.arange(nt) * 1e-8
    k = 2 * np.pi * 2 / LY
    xx, yy = np.meshgrid(x, y, indexing="ij")
    n0 = 2e15 * np.exp(-(((xx - 0.012) / 0.006) ** 2))
    phi1 = 5.0
    frames = []
    for time in t:
        n = n0 * (1 + wave * np.cos(k * yy))
        phi = phi1 * np.cos(k * yy + phase) * (1 + 0.1 * time / 1e-8)
        ey = phi1 * k * np.sin(k * yy + phase) * (1 + 0.1 * time / 1e-8)
        ex = -5e3 * np.ones_like(xx)
        uy = -ex / B * np.ones_like(xx)  # E x B along y, uniform in time
        nu = NU if collisions else 0.0
        mxy = 2e26 * np.sin(np.pi * xx / 0.02) * np.ones_like(yy)  # n <v_x v_y>
        dmxy = 2e26 * np.pi / 0.02 * np.cos(np.pi * xx / 0.02)
        factor = -ELECTRON_MASS / (-E * B)
        # the pointwise identity: n u_x = n E_y / B + factor (nu n u_y + d_x M_xy)
        nux = n * ey / B + factor * (nu * n * uy + dmxy)
        frames.append(
            xr.Dataset(
                {
                    "density": (("x", "y"), n),
                    "flux_x": (("x", "y"), nux),
                    "flux_y": (("x", "y"), n * uy),
                    "moment_xy": (("x", "y"), mxy),
                    "e_field_y": (("x", "y"), ey),
                    "potential": (("x", "y"), phi),
                }
            )
        )
    ds = xr.concat(frames, dim="t").assign_coords(t=t, x=x, y=y)
    ds["b_field_z"] = ("x", np.full(x.size, B))
    ds.attrs = {"mass": ELECTRON_MASS, "charge": -E}
    return ds


def test_the_budget_closes_on_consistent_moments() -> None:
    ds = _moments()
    budget = momentum_budget(ds, collision_frequency=NU, spoke_modes=3)
    scale = float(np.abs(budget["flux"]).max())
    # the stress divergence is differenced on x, so its error is the only residual
    assert float(np.abs(budget["stress"]).max()) > 0.1 * scale
    assert float(np.abs(budget["residual"]).max()) < 3e-3 * scale
    assert (
        float(
            np.abs(budget["stress"] - budget["pressure"] - budget["convective"]).max()
        )
        < 1e-9 * scale
    )
    assert (
        float(np.abs(budget["exb_spoke"] + budget["exb_rest"] - budget["exb"]).max())
        < 1e-9 * scale
    )
    # the wave is mode 2: all of the E x B flux is in the spoke band
    assert budget["exb_rest"].values == pytest.approx(0.0, abs=1e-9 * scale)
    # <n E_y> / B = n_0 wave phi_1 k sin(phase) / (2 B)
    k = 2 * np.pi * 2 / LY
    expected = ds["density"].mean("y") * 0.3 * 5.0 * k * np.sin(0.6) / (2 * B)
    assert budget["exb"].isel(t=0).values == pytest.approx(
        expected.isel(t=0).values, rel=1e-6
    )
    # the Pedersen flux: m nu n u_y / (e B), positive toward the anode for E_x < 0
    pedersen = ELECTRON_MASS * NU * ds["flux_y"].mean("y") / (E * B)
    assert budget["collisional"].values == pytest.approx(pedersen.values, rel=1e-9)
    assert (budget["collisional"] > 0).all()
    assert budget["hall_parameter"].dims == ("t", "x")
    assert budget["exb"].attrs["units"] == "1/(m^2 s)"


def test_collisionless_and_steady_terms_vanish() -> None:
    ds = _moments(collisions=False, nt=1)
    budget = momentum_budget(ds)
    assert budget["collisional"].values == pytest.approx(0.0)
    assert budget["inertial"].values == pytest.approx(0.0)
    assert budget["flux"].dims == ("t", "x")


def test_the_species_dimension_is_selected() -> None:
    ds = _moments(nt=1)
    both = xr.concat([ds, ds * 2], dim="species").assign_coords(
        species=["electrons", "ions"]
    )
    both["b_field_z"] = ds["b_field_z"]
    both.attrs = {"mass": [ELECTRON_MASS, 40 * ELECTRON_MASS], "charge": [-E, E]}
    with pytest.raises(ValueError, match="several species"):
        momentum_budget(both)
    electrons = momentum_budget(
        both, species="electrons", mass=ELECTRON_MASS, charge=-E
    )
    assert electrons["flux"].values == pytest.approx(momentum_budget(ds)["flux"].values)


def test_the_loop_identity_holds_with_the_matching_derivative() -> None:
    ds = _moments(wave=0.5, phase=1.1)
    # the analytic E_y is the spectral derivative of phi
    identity = loop_identity(ds, derivative="spectral")
    scale = float(np.abs(identity["exb_from_field"]).max())
    assert float(np.abs(identity["difference"]).max()) < 1e-9 * scale
    assert (identity["exb_from_field"].isel(t=0) > 0).all()
    # a field differenced on the grid: the centred form is exact, the spectral one off
    # by 1 - sin(k dy) / (k dy) of the mode (k dy = 2 pi 2 / 64)
    dy = float(ds["y"][1] - ds["y"][0])
    phi = ds["potential"]
    differenced = ds.assign(e_field_y=-(phi.roll(y=-1) - phi.roll(y=1)) / (2 * dy))
    centred = loop_identity(differenced)
    assert float(np.abs(centred["difference"]).max()) < 1e-9 * scale
    spectral = loop_identity(differenced, derivative="spectral")
    k_dy = 2 * np.pi * 2 / 64
    ratio = spectral["exb_from_field"] / spectral["exb_from_potential"]
    assert float(ratio.isel(t=0, x=20)) == pytest.approx(np.sin(k_dy) / k_dy, rel=1e-6)
    with pytest.raises(ValueError, match="centred"):
        loop_identity(ds, derivative="fourier")


def test_mode_fluxes_resolve_the_single_wave() -> None:
    ds = _moments(wave=0.3, phase=0.6)
    modes = mode_fluxes(ds, modes=4)
    assert modes["exb_mode"].dims == ("t", "x", "m")
    flux = modes["exb_mode"].isel(t=0)
    total = momentum_budget(ds)["exb"].isel(t=0)
    # mode 2 carries all of it; the amplitudes are those of the construction
    assert flux.sel(m=2).values == pytest.approx(total.values, rel=1e-6)
    assert flux.sel(m=[1, 3, 4]).values == pytest.approx(
        0.0, abs=1e-9 * float(np.abs(total).max())
    )
    assert modes["potential_amplitude"].sel(m=2).isel(t=0).values == pytest.approx(
        5.0, rel=1e-9
    )
    assert modes["density_amplitude"].sel(m=2).isel(t=0, x=20).values == pytest.approx(
        0.3 * float(ds["density"].isel(t=0, x=20).mean("y")), rel=1e-9
    )
    # n ~ cos(k y), phi ~ cos(k y + phase): the density lags the potential by phase
    assert modes["phase"].sel(m=2).isel(t=0).values == pytest.approx(0.6, rel=1e-6)
