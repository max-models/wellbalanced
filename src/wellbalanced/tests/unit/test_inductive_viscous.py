"""Viscous electrons: the slab's skin layer, the cold limit, the power split.

The closed form: in a uniform plasma ``x > 0`` the field and the electron momentum
balance with viscosity ``mu`` (Zielke et al., PSST 31, 035019, 2022, eqs. 7-10) give,
for ``A, u ~ exp(-q x)``,

    A'' = mu0 e n u,   n m (i omega + nu) u - mu u'' = i omega n e A
    =>  mu q^4 - n m (i omega + nu) q^2 + i omega mu0 e^2 n^2 = 0,

two decaying modes ``q_1, q_2``. With the electrons at rest at the surface (no slip,
``u_1 + u_2 = 0``, ``u_j = q_j^2 A_j / (mu0 e n)``) the surface admittance is
``-A'(0) / A(0) = q_1 q_2 / (q_1 + q_2)``, which for ``mu -> 0`` is the cold
``sqrt(i omega mu0 sigma)``. In the vacuum in front, driven by a current layer, ``A`` is
piecewise linear and quadratic, so the field there follows in closed form. Every check
runs on the tensor grid and on refined patches (``kind``).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import e as E
from scipy.constants import m_e as M_E
from scipy.constants import mu_0 as MU_0

from wellbalanced.spatial import InductiveField2D, Winding, stretched_edges
from wellbalanced.tests.unit._grids import GRIDS, field, value_at

simgeometry = pytest.importorskip("simgeometry")
Rectangle = simgeometry.Rectangle

FREQUENCY, DENSITY, NU, TE = 1.0e6, 1.0e18, 1.0e7, 10.0
LEFT, LAYER, RIGHT = -0.05, (-0.04, -0.035), 0.6


def slab(kind="tensor", spacing=5e-4, **settings):
    """Vacuum on ``[-0.05, 0]`` with a current layer, plasma on ``[0, 0.6]``."""
    edges = np.unique(
        np.concatenate(
            [
                np.arange(LEFT, 0.0, spacing),
                np.arange(0.0, 0.05, spacing / 4),
                stretched_edges(0.05, RIGHT, 0.05, 0.051, spacing / 4, ratio=1.05),
                [0.0, RIGHT],
            ]
        )
    )
    s1, s2 = LAYER
    return field(
        kind,
        edges,
        [0.0, 1.0],
        frequency=FREQUENCY,
        boundaries=("dirichlet", "neumann"),
        windings=[Winding(Rectangle(((s1 + s2) / 2, 0.5), (s2 - s1, 2.0)))],
        electron_density=lambda x, y: np.where(x > 0.0, DENSITY, 0.0),
        collision_frequency=NU,
        **settings,
    )


def surface_admittance(viscosity):
    omega = 2.0 * np.pi * FREQUENCY
    if viscosity == 0.0:
        sigma = DENSITY * E**2 / (M_E * (NU + 1j * omega))
        return np.sqrt(1j * omega * MU_0 * sigma)
    d0 = DENSITY * M_E * (1j * omega + NU)
    root = np.sqrt(d0**2 - 4.0 * viscosity * 1j * omega * MU_0 * E**2 * DENSITY**2)
    q = np.sqrt(np.array([d0 + root, d0 - root]) / (2.0 * viscosity))
    q = np.where(q.real < 0.0, -q, q)
    return q[0] * q[1] / (q[0] + q[1])


def vacuum_potential(x, admittance, current_density):
    """``A`` in the vacuum between the layer and the plasma, ``A(LEFT) = 0``."""
    s1, s2 = LAYER
    w = s2 - s1
    j = MU_0 * current_density
    alpha = j * (w + admittance * (w**2 / 2 - w * s2)) / (1.0 - admittance * LEFT)
    at_layer = alpha * (s2 - LEFT) - j * w**2 / 2
    return at_layer + (alpha - j * w) * (x - s2)


@pytest.mark.parametrize("kind", GRIDS)
@pytest.mark.parametrize("viscous", [True, False])
def test_the_skin_layer_of_a_slab(viscous, kind):
    settings = {"electron_temperature": TE} if viscous else {}
    solution = slab(kind, **settings).solve(1.0)
    viscosity = 2.0 / 3.0 * DENSITY * E * TE / NU if viscous else 0.0
    value, x, _ = value_at(solution, solution.vector_potential, -0.02, 0.5)
    layer_area = LAYER[1] - LAYER[0]  # times the cell's 1 m along y
    expected = vacuum_potential(x, surface_admittance(viscosity), 1.0 / layer_area)
    assert value == pytest.approx(expected, rel=1e-4)


def depth(solution):
    """The power-weighted mean depth of the absorption."""
    power = solution.power_density * solution.cell_volumes
    x = solution.grid.xx
    return np.sum(power * np.clip(x, 0, None)) / np.sum(power)


@pytest.mark.parametrize("kind", GRIDS)
def test_viscosity_widens_the_skin_layer(kind):
    cold = slab(kind).solve(1.0)
    viscous = slab(kind, electron_temperature=TE).solve(1.0)
    assert depth(viscous) > 2.0 * depth(cold)


@pytest.mark.parametrize("kind", GRIDS)
def test_no_viscosity_is_the_cold_plasma_to_round_off(kind):
    cold = slab(kind).solve(1.0)
    zero = slab(kind, electron_viscosity=0.0).solve(1.0)
    np.testing.assert_allclose(
        zero.vector_potential, cold.vector_potential, rtol=1e-10, atol=1e-18
    )
    assert zero.absorbed_power == pytest.approx(cold.absorbed_power, rel=1e-10)
    assert zero.viscous_power == pytest.approx(0.0, abs=1e-12 * cold.absorbed_power)


@pytest.mark.parametrize("kind", GRIDS)
def test_power_splits_into_friction_and_viscosity(kind):
    solution = slab(kind, electron_temperature=TE).solve(3.0)
    assert solution.viscous_power > 0.0 and solution.friction_power > 0.0
    total = solution.friction_power + solution.viscous_power
    assert total == pytest.approx(solution.absorbed_power, rel=1e-9)
    assert solution.source_power.real == pytest.approx(
        solution.absorbed_power, rel=1e-9
    )
    scaled = solution.scaled(6.0)
    assert scaled.viscous_power == pytest.approx(4.0 * solution.viscous_power)


@pytest.mark.parametrize("kind", GRIDS)
def test_viscous_electrons_in_r_z_recover_the_cold_column(kind):
    def column(**settings):
        return field(
            kind,
            stretched_edges(0.0, 0.2, 0.0, 0.14, 1e-3, ratio=1.1),
            np.linspace(0.0, 0.01, 3),
            frequency=FREQUENCY,
            geometry="cylindrical",
            boundaries=("neumann", "periodic"),
            windings=[Winding(Rectangle((0.132, 0.005), (0.004, 0.02)), sign=6)],
            electron_density=lambda r, z: np.where(r < 0.1, DENSITY, 0.0),
            collision_frequency=NU,
            **settings,
        ).solve(1.0)

    cold, zero, viscous = (
        column(),
        column(electron_viscosity=0.0),
        column(electron_temperature=TE),
    )
    assert zero.plasma_resistance() == pytest.approx(cold.plasma_resistance(), 1e-10)
    # the velocity vanishes on the axis and at the edge of the plasma
    assert viscous.electron_velocity is not None
    speed = np.abs(viscous.electron_velocity)
    on_axis = viscous.grid.xx == viscous.grid.xx.min()
    assert speed[on_axis].max() < 0.1 * speed.max()
    assert viscous.friction_power + viscous.viscous_power == pytest.approx(
        viscous.absorbed_power, rel=1e-9
    )


def test_a_viscous_plasma_needs_a_collision_frequency():
    with pytest.raises(ValueError, match="nu > 0"):
        InductiveField2D(
            [0.0, 1.0, 2.0],
            [0.0, 1.0],
            frequency=FREQUENCY,
            electron_density=1e18,
            collision_frequency=0.0,
            electron_temperature=5.0,
        )
    with pytest.raises(NotImplementedError, match="Fluid2D"):
        InductiveField2D(
            [0.0, 1.0], [0.0, 1.0], frequency=FREQUENCY, electron_drift=(0.0, 0.0)
        )
    model = slab(electron_temperature=TE)
    with pytest.raises(NotImplementedError, match="viscous"):
        model.time_stepper(1e-9)
