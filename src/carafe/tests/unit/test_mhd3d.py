"""MHD in 3D on a patchwork hierarchy: convergence, the 2D limit, div B, refinement."""

import numpy as np
import pytest
from patchwork import AMRHierarchy

from carafe.mhd import MHD2D
from carafe.mhd3d import CONSERVED_3D, MHD3D

BOX = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))


def test_alfven_wave_converges_at_second_order():
    errors = []
    for cells in (8, 16):
        model, exact = MHD3D.alfven_wave(cells, limiter="none")
        model.run(model.period)
        by = model.dataset()["magnetic_field_y"].values
        errors.append(np.mean(np.abs(by - exact["magnetic_field_y"].values)))
        assert model.divergence() < 1e-14
    assert np.log2(errors[0] / errors[1]) > 1.8


def test_a_state_uniform_along_z_is_the_2d_solution():
    flat = MHD3D.orszag_tang(16, layers=2)
    plane = MHD2D.orszag_tang(16)
    for _ in range(10):
        dt = plane.stable_step()
        plane.step(dt)
        flat.step(dt)
    solid, sheet = flat.dataset(), plane.dataset()
    for name in ("density", "pressure", "velocity_x", "magnetic_field_y"):
        values = solid[name].values
        np.testing.assert_allclose(
            values, np.repeat(sheet[name].values[:, :, None], 2, axis=2), atol=1e-13
        )
    assert np.abs(solid["velocity_z"].values).max() < 1e-15
    assert np.abs(solid["magnetic_field_z"].values).max() < 1e-15


def _refined_alfven() -> MHD3D:
    """The Alfvén wave of `MHD3D.alfven_wave` at 8 cells, with a refined patch."""
    template, _ = MHD3D.alfven_wave(8)
    mesh = AMRHierarchy(
        (16, 8, 8), ((0.0, 2.0), (0.0, 1.0), (0.0, 1.0)), periodic=(True,) * 3
    )
    mesh.add_patch(1, ((8, 20), (4, 12), (2, 10)))
    model = MHD3D(mesh, mean_field=template.mean_field)
    wavevector = 2 * np.pi * np.array([0.5, 1.0, 1.0])
    k = np.linalg.norm(wavevector)
    n = wavevector / k
    e1 = np.cross(n, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    def perpendicular(x, y, z):
        phase = wavevector[0] * x + wavevector[1] * y + wavevector[2] * z
        return [0.1 * (np.sin(phase) * e1[i] + np.cos(phase) * e2[i]) for i in range(3)]

    def potential(i):
        return lambda x, y, z: perpendicular(x, y, z)[i] / k

    model.set_state(
        1.0,
        lambda x, y, z: [-b for b in perpendicular(x, y, z)],
        0.1,
        vector_potential=(potential(0), potential(1), potential(2)),
    )
    return model


def test_a_refined_patch_conserves_and_keeps_div_b_zero():
    model = _refined_alfven()
    start = model.totals()
    model.run(0.1)
    end = model.totals()
    for name in CONSERVED_3D:
        assert abs(end[name] - start[name]) < 1e-14
    assert model.divergence() < 1e-14
    assert model.composite_divergence() < 1e-14


def test_regridding_on_the_current_keeps_div_b_zero():
    model = _refined_alfven()
    # |J| of the circularly polarized wave is uniform: the whole base level is tagged
    model.regrid(model.current_tags(0.1, levels=1), max_patch_size=8)
    assert model.hierarchy.max_level == 1
    assert len(model.hierarchy.patches_at_level(1)) > 1
    start = model.totals()
    model.run(0.05)
    assert model.composite_divergence() < 1e-14
    assert abs(model.totals()["energy"] - start["energy"]) < 1e-13


def test_resistivity_damps_a_force_free_field_at_the_diffusion_rate():
    """``B = curl A``, ``A = (sin kx z_hat-ish)``: a Beltrami field decays as ``exp(-eta k^2 t)``."""
    mesh = AMRHierarchy((4, 4, 16), BOX, periodic=(True,) * 3)
    eta, k = 0.01, 2.0 * np.pi
    model = MHD3D(mesh, resistivity=eta)
    # A = (cos kz, sin kz, 0) / k: B = (cos kz, sin kz, 0), J = k B, no force
    model.set_state(
        1.0,
        (0.0, 0.0, 0.0),
        1.0,
        vector_potential=(
            lambda x, y, z: np.cos(k * z) / k,
            lambda x, y, z: np.sin(k * z) / k,
            0.0,
        ),
    )
    before = model.dataset()["magnetic_pressure"].values.mean()
    duration = 0.5
    model.run(duration)
    after = model.dataset()
    ratio = after["magnetic_pressure"].values.mean() / before
    # the discrete wavenumber of the second difference on 16 cells
    h = 1.0 / 16
    k_discrete = 2.0 * np.sin(k * h / 2) / h
    assert ratio == pytest.approx(np.exp(-2 * eta * k_discrete**2 * duration), rel=2e-3)
    assert np.abs(after["velocity_x"].values).max() < 1e-12


def test_a_uniform_state_stays_uniform_with_outflow_faces():
    mesh = AMRHierarchy((6, 4, 5), BOX, periodic=(False, True, False))
    model = MHD3D(mesh, mean_field=(0.3, -0.2, 0.5))
    model.set_state(1.2, (0.1, -0.2, 0.3), 0.7)
    for _ in range(4):
        model.step()
    state = model.dataset()
    for name, value in (("density", 1.2), ("velocity_z", 0.3), ("pressure", 0.7)):
        np.testing.assert_allclose(state[name].values, value, atol=1e-13)
    np.testing.assert_allclose(state["magnetic_field_z"].values, 0.5, atol=1e-13)


def test_free_fall_gains_the_momentum_of_gravity():
    mesh = AMRHierarchy((4, 4, 4), BOX, periodic=(True,) * 3)
    model = MHD3D(mesh, gravity=(0.0, 0.0, -2.0))
    model.set_state(1.5, (0.0, 0.0, 0.0), 1.0)
    model.run(0.1)
    np.testing.assert_allclose(model.dataset()["velocity_z"].values, -0.2, atol=1e-13)


def test_blast_wave_conserves_and_runs_along_the_field():
    model = MHD3D.blast_wave(12)
    start = model.totals()
    model.run(0.02)
    end = model.totals()
    for name in CONSERVED_3D:
        assert abs(end[name] - start[name]) < 1e-12 * max(1.0, abs(start[name]))
    assert model.divergence() < 1e-14
    pressure = model.dataset()["pressure"].values
    assert pressure.max() < 10.0
    assert np.all(pressure > 0.0)


def test_rejects_unsupported_hierarchies_and_options():
    with pytest.raises(ValueError, match="3D hierarchy"):
        MHD3D(AMRHierarchy((4, 4)))
    mesh = AMRHierarchy((4, 4, 4))
    with pytest.raises(ValueError, match="limiter"):
        MHD3D(mesh, limiter="superbee")
    with pytest.raises(ValueError, match="cfl"):
        MHD3D(mesh, cfl=0.0)
    with pytest.raises(ValueError, match="gamma"):
        MHD3D(mesh, gamma=1.0)
    with pytest.raises(ValueError, match="resistivity"):
        MHD3D(mesh, resistivity=-1.0)
    with pytest.raises(ValueError, match="three components"):
        MHD3D(mesh, mean_field=(1.0, 0.0))


def test_needs_a_positive_state_first():
    model = MHD3D(AMRHierarchy((4, 4, 4), BOX, periodic=(True,) * 3))
    with pytest.raises(RuntimeError, match="set the state"):
        model.step()
    with pytest.raises(ValueError, match="positive"):
        model.set_state(1.0, (0.0, 0.0, 0.0), -1.0)
    with pytest.raises(ValueError, match="three values"):
        model.set_state(1.0, 0.0, 1.0)


def test_reports_a_negative_pressure():
    model = MHD3D(AMRHierarchy((4, 4, 4), BOX, periodic=(True,) * 3))
    model.set_state(1.0, (0.0, 0.0, 0.0), 1.0)
    patch = model.hierarchy.patches[0]
    patch.field(model.names["energy"]).local[1, 1, 1] = -1.0
    with pytest.raises(FloatingPointError, match="non-positive"):
        model.step(1e-3)


def _sheared(axis: int, eta: float) -> MHD3D:
    """``B_axis = sin(2 pi s)``, ``s`` the next axis, in pressure balance; 32 cells along ``s``."""
    shape = [4, 4, 4]
    across = (axis + 1) % 3
    shape[across] = 32
    mesh = AMRHierarchy(tuple(shape), BOX, periodic=(True,) * 3)
    k = 2.0 * np.pi
    model = MHD3D(mesh, resistivity=eta)
    # B_axis = d_s of A along the third axis
    potential = [0.0, 0.0, 0.0]
    potential[(axis + 2) % 3] = lambda x, y, z: -np.cos(k * (x, y, z)[across]) / k
    model.set_state(
        1.0,
        (0.0, 0.0, 0.0),
        lambda x, y, z: 10.0 - 0.5 * np.sin(k * (x, y, z)[across]) ** 2,
        vector_potential=tuple(potential),
    )
    return model


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_resistive_energy_flux_is_eta_j_cross_b(axis):
    """With ``B_a = sin(k s)`` the energy changes by ``-div(eta J x B) = eta k^2 cos(2 k s)``.

    At rest, so over a short time the ideal fluxes add only what the run without
    resistivity adds, and the difference is the resistive energy flux's divergence.
    """
    eta, duration = 1e-3, 0.002
    across = (axis + 1) % 3
    energies = []
    for resistivity in (eta, 0.0):
        model = _sheared(axis, resistivity)
        (patch,) = model.hierarchy.patches
        start = model.hierarchy.gather(patch, model.names["energy"])
        model.run(duration, max_steps=100)
        energies.append(model.hierarchy.gather(patch, model.names["energy"]) - start)
    k = 2.0 * np.pi
    s = (np.arange(32) + 0.5) / 32
    expected = eta * k**2 * np.cos(2.0 * k * s)
    rate = (energies[0] - energies[1]) / duration
    rate = rate.mean(axis=tuple(a for a in range(3) if a != across))
    np.testing.assert_allclose(rate, expected, rtol=0.0, atol=0.03 * eta * k**2)
