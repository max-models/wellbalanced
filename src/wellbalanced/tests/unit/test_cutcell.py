"""Surface-impedance conductors on cut cells, and the iterative solver.

Closed form: in a z-invariant field (a long solenoid of ``n`` turns per metre, a
perfectly conducting box, z-periodic) ``H_z`` is uniform in each connected region of
air, so a conductor of perimeter ``P`` and cross-section ``A_m`` inside the coil, with
Leontovich's ``E_t = Z_s H x n`` on its surface, has

    P Z_s H + i omega mu0 [H (A_c - A_m) + (H - n I) (A_box - A_c)] = 0,

the loss ``1/2 Re(Z_s) |H|^2 P`` per metre and the impedance ``n (i omega mu0 H
(A_c - A_m) + P Z_s H) / I`` per metre (Faraday's law around the box's wall and the
conductor). The discrete field is uniform on faces joined by unknown edges too, so for a
polygonal conductor, whose cut cells are exact, the cut-cell solution is exact up to the
gauge's ``eps``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0 as MU_0

from wellbalanced.spatial import (
    Coil3D,
    Conductor3D,
    EdgeGrid,
    ImpedanceConductor3D,
    InductiveField3D,
)
from wellbalanced.spatial.cutcell import cut_cells, staircased_cells

simgeometry = pytest.importorskip("simgeometry")
if not hasattr(simgeometry.Circle((0, 0), 1), "yee_fractions"):
    pytest.skip("needs simgeometry with yee_fractions", allow_module_level=True)
Cuboid, Tube = simgeometry.Cuboid, simgeometry.Tube

SIGMA, TURNS, BOX, COIL = 1e6, 100.0, 0.06, 0.045


def impedance_closed_form(frequency, perimeter, metal_area):
    """Loss [W/m] and impedance [Ohm/m] of the whole cross-section at 1 A."""
    omega = 2 * np.pi * frequency
    z_s = (1 + 1j) * np.sqrt(omega * MU_0 / (2 * SIGMA))
    coil_area, box_area = (2 * COIL) ** 2, (2 * BOX) ** 2
    field = (
        1j
        * omega
        * MU_0
        * TURNS
        * (box_area - coil_area)
        / (perimeter * z_s + 1j * omega * MU_0 * (box_area - metal_area))
    )
    loss = 0.5 * z_s.real * abs(field) ** 2 * perimeter
    impedance = TURNS * (
        1j * omega * MU_0 * field * (coil_area - metal_area) + perimeter * z_s * field
    )
    return loss, impedance


def quarter(cells, shapes, frequency, **kwargs):
    """A quarter of the cross-section, z-periodic, two cells long, at 1 A."""
    h = BOX / cells
    grid = EdgeGrid(
        (0, 0, 0),
        (BOX, BOX, 2 * h),
        (cells, cells, 2),
        ("dirichlet",) * 2 + ("periodic",),
    )
    coil = Coil3D(Cuboid((0, 0, 0), (2 * COIL, 2 * COIL, 4.0)), (0, 0, 1), TURNS)
    model = InductiveField3D(
        grid,
        frequency=frequency,
        coils=[coil],
        impedance_conductors=[ImpedanceConductor3D(shapes, SIGMA, "metal")],
        **kwargs,
    )
    return model, model.solve(1.0), 4 / (2 * h)


def test_cut_and_staircased_cells_agree_on_walls_on_grid_lines():
    grid = EdgeGrid(
        (0, 0, 0), (1, 1, 1), (8, 8, 8), ("dirichlet", "natural", "periodic")
    )
    box = Cuboid((0.5, 0.4375, 0.5), (0.5, 0.375, 0.25))
    slit = Cuboid((0.5, 0.5, 0.5), (2, 2, 2)) - Cuboid((0.5, 0.5, 0.5), (0.25, 4, 4))
    for shape in (box, slit):
        cut, stairs = cut_cells(grid, [shape]), staircased_cells(grid, [shape])
        np.testing.assert_array_equal(cut.edge_air, stairs.edge_air)
        np.testing.assert_array_equal(cut.face_air, stairs.face_air)
        np.testing.assert_allclose(cut.face_width, stairs.face_width, atol=1e-15)


def test_plane_conductor_is_exact_on_cut_cells():
    """A tilted plane wall: the cut cells give the closed form to the gauge's eps.

    The conductor is the half-space ``n . x < d`` (a big rotated box): in the quarter a
    triangle with its corners on the symmetry planes, in the whole cross-section a
    rhombus of four times its area and hypotenuse.
    """
    angle, offset, big = 0.4, 0.012, 1.0
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    rotation = np.array(
        [[normal[0], -normal[1], 0], [normal[1], normal[0], 0], [0, 0, 1]]
    )
    wall = Cuboid((-big / 2, 0, 0), (big, big, 4.0)).transformed(
        rotation, offset * normal
    )
    a, b = offset / normal[1], offset / normal[0]  # the triangle's legs on y and x
    frequency = 5e5
    loss, impedance = impedance_closed_form(frequency, 4 * np.hypot(a, b), 2 * a * b)
    model, solution, scale = quarter(24, [wall], frequency, min_air_fraction=1e-12)
    assert scale * solution.loss == pytest.approx(loss, rel=1e-5)
    assert scale * solution.impedance.imag == pytest.approx(impedance.imag, rel=1e-5)
    # the coil's power is the surface's loss
    assert solution.source_power.real == pytest.approx(solution.loss, rel=1e-10)
    _, stairs, _ = quarter(24, [wall], frequency, cut_cells=False)
    assert abs(scale * stairs.loss / loss - 1) > 0.05


def test_impedance_tube_converges_at_second_order_and_staircase_does_not():
    """A tube in the solenoid: 2 pi r_out, its outer surface; the bore sees no field."""
    inner, outer, frequency = 0.030, 0.036, 5e5
    loss, impedance = impedance_closed_form(
        frequency, 2 * np.pi * outer, np.pi * outer**2
    )
    tube = Tube((0, 0, -2.0), (0, 0, 2.0), inner, outer)
    errors = []
    for cells in (24, 48):
        _, solution, scale = quarter(cells, [tube], frequency)
        errors.append(abs(scale * solution.loss / loss - 1))
        assert scale * solution.impedance.imag == pytest.approx(
            impedance.imag, rel=2e-4
        )
    assert errors[0] < 2e-4 and errors[1] < 5e-5
    assert errors[0] / errors[1] > 3
    # the staircase's surface is 4/pi too long: no convergence
    _, stairs, scale = quarter(48, [tube], frequency, cut_cells=False)
    assert scale * stairs.loss / loss - 1 > 0.2


def test_iterative_solver_matches_the_direct_one():
    h = 0.005
    grid = EdgeGrid(
        (0, 0, 0),
        (0.06, 0.06, 0.1),
        (12, 12, 20),
        ("dirichlet", "dirichlet", ("natural", "dirichlet")),
    )
    tube = Tube((0, 0, -0.05), (0, 0, 0.05), 0.03, 0.036) - Cuboid(
        (0, 0, 0), (0.24, 0.004, 0.08)
    )
    model = InductiveField3D(
        grid,
        frequency=5e5,
        coils=[Coil3D(Cuboid((0, 0, 0), (0.09, 0.09, 0.16)), (0, 0, 1), TURNS)],
        impedance_conductors=[ImpedanceConductor3D([tube], SIGMA)],
    )
    direct = model.solve(1.0)
    iterative = model.solve(1.0, solver="iterative", tolerance=1e-10)
    assert model.last_iterative.converged
    assert model.last_iterative.iterations < 200
    assert iterative.loss == pytest.approx(direct.loss, rel=1e-7)
    assert iterative.inductance == pytest.approx(direct.inductance, rel=1e-8)
    assert h == grid.spacing[0]
    with pytest.raises(ValueError, match="solver"):
        model.solve(1.0, solver="magic")


def test_volume_conductors_must_not_touch_impedance_conductors():
    grid = EdgeGrid((0, 0, 0), (1, 1, 1), (4, 4, 4))
    block = Cuboid((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="must not touch"):
        InductiveField3D(
            grid,
            frequency=1e3,
            conductors=[Conductor3D([block], 1e6)],
            impedance_conductors=[ImpedanceConductor3D([block], 1e6)],
        )
    with pytest.raises(ValueError, match="min_air_fraction"):
        InductiveField3D(grid, frequency=1e3, min_air_fraction=0.0)
