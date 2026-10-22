"""The inductive field in r–z: a current loop, a long solenoid, a plasma column in it.

Closed forms: the vector potential of a loop of radius ``a``,
``A_phi = mu0 I / (pi k) sqrt(a / r) ((1 - k^2/2) K(k) - E(k))`` with
``k^2 = 4 a r / ((a + r)^2 + z^2)`` (Jackson, *Classical Electrodynamics*, eq. 5.37),
and ``B_z = mu0 I / (2 a)`` at its centre; the uniform ``B_z = mu0 N I / l`` of a long
solenoid; the exact impedance of a uniform conducting column in it, the Bessel solution
of :func:`wellbalanced.driver.solenoid_impedance`.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0 as MU_0
from scipy.special import ellipe, ellipk

from wellbalanced.driver import Coil, solenoid_impedance
from wellbalanced.spatial import Conductor, InductiveField2D, Winding, stretched_edges

simgeometry = pytest.importorskip("simgeometry")
Circle, Rectangle = simgeometry.Circle, simgeometry.Rectangle

FREQUENCY = 1.0e6


def loop_potential(r, z, radius, current=1.0):
    m = 4.0 * radius * r / ((radius + r) ** 2 + z**2)
    k = np.sqrt(m)
    shape = (1.0 - m / 2.0) * ellipk(m) - ellipe(m)
    return MU_0 * current / (np.pi * k) * np.sqrt(radius / r) * shape


def nearest(model, r, z):
    return int(np.argmin(np.abs(model.x - r))), int(np.argmin(np.abs(model.y - z)))


def test_the_field_of_a_current_loop():
    radius, spacing = 0.1, 1.0e-3
    model = InductiveField2D(
        stretched_edges(0.0, 2.0, 0.0, 0.2, spacing, ratio=1.06),
        stretched_edges(-2.0, 2.0, -0.1, 0.1, spacing, ratio=1.06),
        frequency=FREQUENCY,
        geometry="cylindrical",
        windings=[Winding(Circle((radius, 0.0), 0.003))],
    )
    solution = model.solve(1.0)
    for r, z in [(0.05, 0.0), (0.1, 0.02), (0.15, 0.0), (0.1, -0.05), (0.02, 0.03)]:
        i, j = nearest(model, r, z)
        expected = loop_potential(model.x[i], model.y[j], radius)
        assert solution.vector_potential[i, j].real == pytest.approx(expected, rel=1e-3)
    _, b_z = solution.magnetic_field
    i, j = nearest(model, 0.0, 0.0)
    assert b_z[i, j].real == pytest.approx(MU_0 / (2.0 * radius), rel=1e-3)
    # no plasma: no absorbed power, a purely reactive load
    assert solution.absorbed_power == 0.0 and solution.source_power.imag > 0.0


def solenoid(electron_density=0.0, collision_frequency=1.0e7, spacing=5e-4, **settings):
    """An infinitely long solenoid of 6 turns per centimetre: periodic in ``z``.

    Outside an infinite solenoid ``B_z = 0``: the natural ("neumann") edge.
    """
    length = 0.01
    return InductiveField2D(
        stretched_edges(0.0, 0.2, 0.0, 0.14, spacing, ratio=1.1),
        np.linspace(0.0, length, 3),
        frequency=FREQUENCY,
        geometry="cylindrical",
        boundaries=("neumann", "periodic"),
        windings=[Winding(Rectangle((0.132, length / 2), (0.004, 2 * length)), sign=6)],
        electron_density=electron_density,
        collision_frequency=collision_frequency,
        **settings,
    ), length


def test_a_long_solenoid_has_a_uniform_field():
    model, length = solenoid()
    _, b_z = model.solve(2.0).magnetic_field
    inside = model.x < 0.12
    np.testing.assert_allclose(b_z[inside].real, MU_0 * 6 * 2.0 / length, rtol=1e-9)


@pytest.mark.parametrize("density", [1.0e17, 1.0e18])
def test_a_plasma_column_in_a_solenoid_against_the_bessel_solution(density):
    radius = 0.1
    model, length = solenoid(lambda r, z: np.where(r < radius, density, 0.0), 1.0e7)
    solution = model.solve(1.0)
    exact = solenoid_impedance(
        Coil(6, 0.13, length, field_factor=1.0), radius, density, 1.0e7, FREQUENCY
    )
    assert solution.plasma_resistance() == pytest.approx(exact.real, rel=1e-3)
    # absorbed equals delivered to round-off
    assert solution.source_power.real == pytest.approx(
        solution.absorbed_power, rel=1e-12
    )


def test_power_scaling_and_efficiency_in_r_z():
    model, _ = solenoid(lambda r, z: np.where(r < 0.1, 1.0e18, 0.0))
    solution = model.solve(1.0).for_power(25e3)
    assert solution.absorbed_power == pytest.approx(25e3)
    r_p = solution.plasma_resistance()
    assert solution.efficiency(0.6) == pytest.approx(r_p / (r_p + 0.6))
    with pytest.raises(ValueError, match="no depth"):
        solution.plasma_resistance(0.3)


def test_conductors_are_grounded_in_r_z():
    edges = stretched_edges(0.0, 1.0, 0.0, 0.3, 2e-3, ratio=1.1)
    plate = Rectangle((0.1, -0.05), (0.2, 0.01))
    model = InductiveField2D(
        edges,
        stretched_edges(-1.0, 1.0, -0.1, 0.1, 2e-3, ratio=1.1),
        frequency=FREQUENCY,
        geometry="cylindrical",
        windings=[Winding(Circle((0.12, 0.0), 0.003))],
        conductors=[Conductor([plate], name="backplate")],
    )
    solution = model.solve(1.0)
    inside = model.grid.mask(plate)
    assert np.all(solution.vector_potential[inside] == 0.0)
    # the image current in the plate opposes the coil's
    assert solution.conductor_currents["backplate"].real < 0.0
    with pytest.raises(ValueError, match="grounded"):
        InductiveField2D(
            edges,
            [-0.1, 0.1],
            frequency=FREQUENCY,
            geometry="cylindrical",
            conductors=[Conductor([plate], floating=True)],
        )
