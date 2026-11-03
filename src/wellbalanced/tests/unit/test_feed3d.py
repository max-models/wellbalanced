"""Coils of real wire: a feed (delta-gap source) driving surface-impedance conductors.

Closed form: a coaxial line of surface-impedance conductors, inner radius ``a`` and outer
``b``, has per metre ``L' = mu0 / (2 pi) ln(b / a)`` outside the metal and the surface
impedance's ``Z_s / (2 pi) (1 / a + 1 / b)`` (``R' = Re``, and as much again in
``omega L'``: the skin's internal inductance). The line is shorted at one end by a
perfectly conducting wall and fed across a gap at the other; the difference of two
lengths removes the ends.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0 as MU_0

from wellbalanced.spatial import (
    EdgeGrid,
    Feed3D,
    ImpedanceConductor3D,
    InductiveField3D,
)

simgeometry = pytest.importorskip("simgeometry")
Cuboid, Cylinder, Torus, Tube = (
    simgeometry.Cuboid,
    simgeometry.Cylinder,
    simgeometry.Torus,
    simgeometry.Tube,
)

COPPER = 5.8e7
INNER, OUTER, OUTER_WALL, HALF, GAP = 0.004, 0.012, 0.014, 0.015, 0.002


def coax(length, h, frequency=1e6, regularization=1e-6, reverse=False):
    """A coax along z from the fed end (z = 0) to the shorted end (z = length)."""
    cells = round(2 * HALF / h)
    grid = EdgeGrid(
        (-HALF, -HALF, 0.0), (HALF, HALF, length), (cells, cells, round(length / h))
    )
    inner = Cylinder((0, 0, GAP), (0, 0, length + 1.0), INNER)
    outer = Tube((0, 0, -1.0), (0, 0, length + 1.0), OUTER, OUTER_WALL)
    # from the wall at z = 0 (joined to the outer conductor) into the inner one
    ends = ((0, 0, 0), (0, 0, GAP + 0.003))
    feed = Feed3D(*(ends[::-1] if reverse else ends), name="feed")
    return InductiveField3D(
        grid,
        frequency=frequency,
        feeds=[feed],
        impedance_conductors=[
            ImpedanceConductor3D([inner], COPPER, "inner"),
            ImpedanceConductor3D([outer], COPPER, "outer"),
        ],
        regularization=regularization,
    )


def test_coax_impedance_per_length_matches_the_closed_form():
    h, frequency = 0.001, 1e6
    short, long = (
        coax(length, h, frequency).solve(1.0, solver="iterative")
        for length in (0.008, 0.016)
    )
    per_length = (long.impedance - short.impedance) / 0.008
    surface = ImpedanceConductor3D((), COPPER).surface_impedance(frequency)
    expected = 2j * np.pi * frequency * MU_0 / (2 * np.pi) * np.log(
        OUTER / INNER
    ) + surface / (2 * np.pi) * (1 / INNER + 1 / OUTER)
    # the coil's power is the conductors' losses, to round-off
    assert long.loss == pytest.approx(long.source_power.real, rel=1e-8)
    assert set(long.conductor_losses) == {"inner", "outer"}
    # inductance to 0.3 %, resistance to 5 % (first order: H is a face average
    # where it varies as 1 / r, at a / h = 4)
    assert per_length.imag == pytest.approx(expected.imag, rel=3e-3)
    assert per_length.real == pytest.approx(expected.real, rel=0.05)


def test_feed_is_compatible_and_the_gauge_does_not_matter():
    """The feed's current returns through the metal: no gradient of the source."""
    results = [
        coax(0.008, 0.002, regularization=eps).solve(1.0).impedance
        for eps in (1e-4, 1e-6)
    ]
    assert results[0] == pytest.approx(results[1], rel=1e-4)
    # the current the other way round: the same impedance
    reverse = coax(0.008, 0.002, reverse=True).solve(1.0).impedance
    assert reverse == pytest.approx(results[1], rel=1e-10)


def test_ring_of_wire():
    """A ring of copper wire with a feed across a gap: L and R of the right size."""
    radius, wire, half, gap, h = 0.02, 0.004, 0.05, 0.002, 0.004
    cells = round(2 * half / h)
    grid = EdgeGrid((-half,) * 3, (half,) * 3, (cells,) * 3)
    ring = Torus((0, 0, 0), radius, wire) - Cuboid((radius, 0, 0), (0.02, gap, 0.02))
    feed = Feed3D(
        (radius, -gap / 2 - 2.5e-3, 0), (radius, gap / 2 + 2.5e-3, 0), order=(1, 0, 2)
    )
    model = InductiveField3D(
        grid,
        frequency=1e6,
        feeds=[feed],
        impedance_conductors=[ImpedanceConductor3D([ring], COPPER, "ring")],
    )
    solution = model.solve(1.0, solver="iterative")
    # thin-ring closed forms (a / R = 0.2): the curvature (+18 % in R) and the box
    # (-1 % in L) aside
    inductance = MU_0 * radius * (np.log(8 * radius / wire) - 2)
    skin = ImpedanceConductor3D((), COPPER).surface_impedance(1e6).real
    assert solution.inductance == pytest.approx(inductance, rel=0.03)
    assert solution.resistance == pytest.approx(skin * radius / wire, rel=0.25)
    assert solution.conductor_losses["ring"] == pytest.approx(solution.loss)


def test_feed_errors():
    grid = EdgeGrid((0, 0, 0), (1, 1, 1), (4, 4, 4))
    block = ImpedanceConductor3D([Cuboid((0.5, 0.5, 0.5), (0.6, 0.6, 0.6))], COPPER)
    with pytest.raises(ValueError, match="outside the grid"):
        InductiveField3D(grid, frequency=1.0, feeds=[Feed3D((0, 0, 0), (2, 0, 0))])
    with pytest.raises(ValueError, match="same node"):
        InductiveField3D(
            grid, frequency=1.0, feeds=[Feed3D((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    with pytest.raises(ValueError, match="permutation"):
        InductiveField3D(
            grid, frequency=1.0, feeds=[Feed3D((0, 0, 0), (1, 1, 1), order=(0, 0, 1))]
        )
    # from the metal block into the air: no return path
    fine = EdgeGrid((0, 0, 0), (1, 1, 1), (8, 8, 8))
    with pytest.raises(ValueError, match="inside the metal"):
        InductiveField3D(
            fine,
            frequency=1.0,
            feeds=[Feed3D((0.5, 0.5, 0.5), (0.5, 0.5, 0.875))],
            impedance_conductors=[block],
        )
    # from the block to the Dirichlet wall: two pieces of metal
    with pytest.raises(ValueError, match="inside the metal"):
        InductiveField3D(
            fine,
            frequency=1.0,
            feeds=[Feed3D((0.5, 0.5, 0.5), (0.5, 0.5, 1.0))],
            impedance_conductors=[block],
        )
