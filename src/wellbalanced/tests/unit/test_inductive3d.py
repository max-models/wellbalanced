"""The 3D field on the edges: a long solenoid, the 2D result, symmetry planes, the gauge.

Closed forms: a long solenoid of ``n`` turns per metre and cross-section ``A_s`` in a
perfectly conducting box of cross-section ``A_box`` (periodic along the axis) has
``L' = mu0 n^2 A_s (1 - A_s / A_box)`` (Ampere's law, and zero net flux through the box,
whose walls keep ``A x n = 0``); the z-invariant field of a driver in a cut is
:class:`~wellbalanced.spatial.InductiveField2D`'s; a long conducting tube in that
solenoid has the Bessel solution of :func:`tube_in_box`.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0 as MU_0
from scipy.integrate import quad
from scipy.special import iv, kv

from wellbalanced.spatial import (
    Coil3D,
    Conductor3D,
    EdgeGrid,
    InductiveField2D,
    InductiveField3D,
    Winding,
)

simgeometry = pytest.importorskip("simgeometry")
Cuboid, Cylinder, Tube = simgeometry.Cuboid, simgeometry.Cylinder, simgeometry.Tube
Rectangle = simgeometry.Rectangle

PERIODIC_Z = ("dirichlet", "dirichlet", "periodic")


def solenoid_grid(cells, half=0.1, length=0.02):
    return EdgeGrid(
        (-half, -half, 0.0), (half, half, length), (cells, cells, 2), PERIODIC_Z
    )


def discrete_area(grid, shape):
    """The cross-section the staircased coil encloses: its z-faces in one plane."""
    z_faces = (grid.face_axes() == 2) & grid.face_mask(shape)
    return z_faces.sum() / grid.cells[2] * grid.spacing[0] * grid.spacing[1]


def box_solenoid_inductance(n, area, box_area):
    return MU_0 * n**2 * area * (1.0 - area / box_area)


def tube_in_box(frequency, sigma, inner, outer, turns, coil_area, box_area):
    """Loss [W/m] and impedance [Ohm/m] of an infinite tube in a solenoid in a box, at 1 A.

    ``H_z`` is uniform in every insulating region: the bore ``H_b``, the gap to the
    coil ``H_g`` and outside the coil ``H_o = H_g - n I``; in the wall
    ``H = c1 I0(k r) + c2 K0(k r)``, ``k^2 = i omega mu0 sigma``; Faraday's law on the
    bore gives ``H'(inner) = k^2 inner H_b / 2``, and the box's walls zero net flux.
    """
    omega = 2 * np.pi * frequency
    k = np.sqrt(1j * omega * MU_0 * sigma)

    def integral(r):  # of r I0(k r) and r K0(k r)
        return r * iv(1, k * r) / k, -r * kv(1, k * r) / k

    (i_out, k_out), (i_in, k_in) = integral(outer), integral(inner)
    ring = np.pi * outer**2
    system = np.array(
        [
            [iv(0, k * outer), kv(0, k * outer), 0, -1, 0],
            [iv(0, k * inner), kv(0, k * inner), -1, 0, 0],
            [k * iv(1, k * inner), -k * kv(1, k * inner), -(k**2) * inner / 2, 0, 0],
            [0, 0, 0, 1, -1],
            [
                2 * np.pi * (i_out - i_in),
                2 * np.pi * (k_out - k_in),
                np.pi * inner**2,
                coil_area - ring,
                box_area - coil_area,
            ],
        ]
    )
    c1, c2, h_b, h_g, _ = np.linalg.solve(system, [0, 0, 0, turns, 0])

    def gradient(r):
        return k * (c1 * iv(1, k * r) - c2 * kv(1, k * r))

    loss = np.pi / sigma * quad(lambda r: abs(gradient(r)) ** 2 * r, inner, outer)[0]
    flux = MU_0 * (
        h_b * np.pi * inner**2
        + 2 * np.pi * (c1 * (i_out - i_in) + c2 * (k_out - k_in))
        + h_g * (coil_area - ring)
    )
    return loss, 1j * omega * turns * flux


def test_curl_of_gradient_vanishes_and_coil_current_is_divergence_free():
    grid = EdgeGrid(
        (0, 0, 0),
        (1, 2, 1),
        (3, 4, 5),
        ("periodic", ("natural", "dirichlet"), ("dirichlet", "natural")),
    )
    assert abs(grid.curl() @ grid.gradient()).max() == 0.0
    coil = Coil3D(Cylinder((0.5, 1.0, -1.0), (0.5, 1.0, 0.6), 0.4), (1, 1, 1), 7.0)
    model = InductiveField3D(grid, frequency=1.0, coils=[coil])
    assert abs(grid.gradient().T @ model._unit_source).max() < 1e-12


def test_grid_counts_and_boundary_halves():
    grid = EdgeGrid(
        (0, 0, 0), (1, 1, 1), (2, 3, 4), ("dirichlet", "natural", "periodic")
    )
    assert grid.nodes == (3, 4, 4)
    assert grid.n_edges == 2 * 4 * 4 + 3 * 3 * 4 + 3 * 4 * 4
    # the dual faces and edges add up to the box: sum over x-edges of L A* = volume
    x_edges = slice(0, grid.edge_offsets[1])
    volume = np.sum(grid.edge_lengths[x_edges] * grid.dual_areas[x_edges])
    assert volume == pytest.approx(1.0)
    z_faces = slice(grid.face_offsets[2], grid.face_offsets[3])
    assert np.sum(
        grid.face_areas[z_faces] * grid.dual_lengths[z_faces]
    ) == pytest.approx(1.0)
    # Dirichlet walls (x) remove the tangential edges, natural walls (y) keep them
    y_edges = grid.edge_block(grid.free_edges.astype(float), 1)
    assert np.all(y_edges[[0, -1]] == 0.0) and np.all(y_edges[1:-1] == 1.0)


@pytest.mark.parametrize("cells", [8, 20])
def test_square_solenoid_inductance_is_exact(cells):
    half, side, n = 0.1, 0.1, 50.0
    grid = solenoid_grid(cells, half)
    coil = Coil3D(Cuboid((0.0, 0.0, 0.0), (side, side, 1.0)), (0, 0, 1), n)
    model = InductiveField3D(grid, frequency=1e3, coils=[coil])
    per_length = model.solve(1.0).inductance / 0.02
    expected = box_solenoid_inductance(n, side**2, (2 * half) ** 2)
    assert per_length == pytest.approx(expected, rel=1e-6)


def test_long_solenoid_inductance_staircased_circle():
    half, radius, n = 0.1, 0.05, 20.0
    box_area = (2 * half) ** 2
    exact = box_solenoid_inductance(n, np.pi * radius**2, box_area)
    errors = []
    for cells in (10, 20, 40):
        grid = solenoid_grid(cells, half)
        shape = Cylinder((0, 0, -1), (0, 0, 1), radius)
        model = InductiveField3D(
            grid, frequency=1e3, coils=[Coil3D(shape, (0, 0, 1), n)]
        )
        per_length = model.solve(1.0).inductance / 0.02
        # exact for the staircased cross-section the grid sees
        area = discrete_area(grid, shape)
        assert per_length == pytest.approx(
            box_solenoid_inductance(n, area, box_area), rel=1e-6
        )
        errors.append(abs(per_length / exact - 1.0))
    assert errors[-1] < 0.01 and errors[-1] < errors[0]


def test_z_invariant_driver_recovers_the_2d_field():
    """A z-periodic driver: the z-edges are the cells of a cut, to the gauge's eps."""
    half, cells, frequency = 0.1, 30, 1.0e6
    h = 2 * half / cells
    d, width, radius = 8 * h, 9 * h, 0.0413
    density, collisions, current = 1e17, 2e7, 50.0
    grid = EdgeGrid(
        (-half, -half, 0.0), (half, half, 2 * h), (cells, cells, 2), ("periodic",) * 3
    )
    # the cut's cells are centred on the 3D grid's nodes
    edges = -half - h / 2 + h * np.arange(cells + 1)
    cut = InductiveField2D(
        edges,
        edges,
        frequency=frequency,
        boundaries=("periodic", "periodic"),
        windings=[
            Winding(Rectangle((-d, 0.0), (h, width)), 1.0),
            Winding(Rectangle((d, 0.0), (h, width)), -1.0),
        ],
        electron_density=lambda x, y: np.where(x**2 + y**2 < radius**2, density, 0.0),
        collision_frequency=collisions,
    )
    sigma = cut.conductivity(2 * np.pi * frequency)[cells // 2, cells // 2]
    # the loop: a box between the two sides, T along y, one turn over its width
    loop = Coil3D(Cuboid((0.0, 0.0, 0.0), (2 * d, width, 1.0)), (0, 1, 0), 1.0 / width)
    field = InductiveField3D(
        grid,
        frequency=frequency,
        coils=[loop],
        conductivity=lambda p: np.where(p[0] ** 2 + p[1] ** 2 < radius**2, sigma, 0.0),
    )
    three = field.solve(current)
    two = cut.solve(current)
    a_z = grid.edge_block(three.potential, 2)[:, :, 0] / h
    a_ref = two.vector_potential
    assert np.max(np.abs(a_z - a_ref)) < 1e-6 * np.max(np.abs(a_ref))
    assert three.loss / (2 * h) == pytest.approx(two.absorbed_power, rel=1e-6)
    assert three.impedance / (2 * h) == pytest.approx(
        2 * two.source_power / current**2, rel=1e-6
    )


def test_infinite_tube_loss_converges_to_the_bessel_solution():
    """A staircased tube, z-periodic, a quarter of the box: first order in h."""
    box, coil, inner, outer = 0.06, 0.045, 0.030, 0.036
    sigma, turns, frequency = 1e6, 100.0, 5e4  # skin depth 2.25 mm, wall 6 mm
    loss, impedance = tube_in_box(
        frequency, sigma, inner, outer, turns, (2 * coil) ** 2, (2 * box) ** 2
    )
    assert np.real(impedance) == pytest.approx(2 * loss, rel=1e-8)
    errors = []
    for cells in (48, 96):
        h = box / cells
        grid = EdgeGrid((0, 0, 0), (box, box, 2 * h), (cells, cells, 2), PERIODIC_Z)
        model = InductiveField3D(
            grid,
            frequency=frequency,
            coils=[
                Coil3D(Cuboid((0, 0, 0), (2 * coil, 2 * coil, 1.0)), (0, 0, 1), turns)
            ],
            conductors=[
                Conductor3D([Tube((0, 0, -1), (0, 0, 1), inner, outer)], sigma)
            ],
        )
        solution = model.solve(1.0)
        per_length = 4 * solution.loss / (2 * h)
        assert solution.loss == pytest.approx(solution.source_power.real, rel=1e-9)
        errors.append(per_length / loss - 1)
        inductance = 4 * solution.inductance / (2 * h)
        assert inductance == pytest.approx(
            np.imag(impedance) / (2 * np.pi * frequency), rel=0.01
        )
    assert abs(errors[1]) < 0.025 and abs(errors[1]) < 0.5 * abs(errors[0])


def tube_model(lower, upper, cells, boundaries, **settings):
    grid = EdgeGrid(lower, upper, cells, boundaries)
    # no surface through a node plane or a coil sample (a quarter cell off a node
    # plane): the two grids sample the same points
    coil = Coil3D(Cuboid((0, 0, 0), (0.12, 0.12, 0.146)), (0, 0, 1), 100.0)
    tube = Tube((0, 0, -0.043), (0, 0, 0.043), 0.0253, 0.0447)
    slit = Cuboid((0, 0, 0), (0.2, 0.012, 0.065))
    return InductiveField3D(
        grid,
        frequency=2e4,
        coils=[coil],
        conductors=[Conductor3D([tube - slit], 1e6, "tube")],
        **settings,
    )


def test_power_balance_and_symmetry_planes():
    """The loss is Re S; the eighth with its symmetry planes is the whole box."""
    whole = tube_model((-0.08,) * 3, (0.08,) * 3, (16, 16, 16), ("dirichlet",) * 3)
    eighth = tube_model(
        (0, 0, 0),
        (0.08,) * 3,
        (8, 8, 8),
        ("dirichlet", "dirichlet", ("natural", "dirichlet")),
    )
    full, part = whole.solve(2.0), eighth.solve(2.0)
    assert full.loss > 0.0
    assert full.loss == pytest.approx(full.source_power.real, rel=1e-9)
    assert full.conductor_losses["tube"] == pytest.approx(full.loss)
    # the azimuthal current crosses x = 0 and y = 0 at right angles (A x n = 0 there,
    # the slit is symmetric about both); z = 0 is the midplane (H x n = 0)
    assert 8 * part.loss == pytest.approx(full.loss, rel=1e-6)
    assert 8 * part.inductance == pytest.approx(full.inductance, rel=1e-6)


def test_regularization_changes_the_result_by_order_eps():
    results = []
    for eps in (1e-4, 1e-6, 1e-8):
        model = tube_model(
            (0, 0, 0),
            (0.08,) * 3,
            (8, 8, 8),
            ("dirichlet", "dirichlet", ("natural", "dirichlet")),
            regularization=eps,
        )
        solution = model.solve(1.0)
        results.append((solution.loss, solution.inductance))
    loss, inductance = np.array(results).T
    assert abs(loss[1] / loss[2] - 1) < 1e-5
    assert abs(inductance[1] / inductance[2] - 1) < 1e-5
    assert abs(loss[0] / loss[2] - 1) < 1e-3


def test_subcell_conductance_blocks_a_thin_slit():
    """A slit thinner than a cell: missed at the midpoints, blocked by the sub-cells."""
    grid = EdgeGrid((0, 0, 0), (1, 1, 1), (4, 4, 4))
    plate = Cuboid((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)) - Cuboid(
        (0.30, 0.5, 0.5), (0.05, 2.0, 2.0)
    )

    def sigma(points):
        return np.where(grid.contains(plate, points), 1.0, 0.0)

    x_edges = slice(0, grid.edge_offsets[1])
    midpoint = grid.edge_average(sigma)[x_edges].reshape(grid.edge_shape(0))
    averaged = grid.edge_average(sigma, samples=4)[x_edges].reshape(grid.edge_shape(0))
    # the x-edges from 0.25 to 0.5 cross the slit (0.275 to 0.325)
    assert np.all(midpoint[1] == 1.0)
    assert np.all(averaged[1] == 0.0)
    # inside the plate the other edges conduct fully (its outer faces are halved)
    assert np.allclose(averaged[[0, 2, 3]][:, 1:-1, 1:-1], 1.0)


def test_invalid_input():
    grid = EdgeGrid((0, 0, 0), (1, 1, 1), (2, 2, 2))
    with pytest.raises(ValueError, match="frequency"):
        InductiveField3D(grid, frequency=0.0)
    with pytest.raises(ValueError, match="regularization"):
        InductiveField3D(grid, frequency=1.0, regularization=0.0)
    with pytest.raises(ValueError, match="contains no face"):
        InductiveField3D(
            grid, frequency=1.0, coils=[Coil3D(Cuboid((5, 5, 5), (0.1, 0.1, 0.1)))]
        )
    block = Cuboid((0.5, 0.5, 0.5), (0.6, 0.6, 0.6))
    with pytest.raises(ValueError, match="overlap"):
        InductiveField3D(
            grid,
            frequency=1.0,
            conductors=[Conductor3D([block], 1.0), Conductor3D([block], 2.0)],
        )
    with pytest.raises(ValueError, match="periodic"):
        EdgeGrid(
            (0, 0, 0),
            (1, 1, 1),
            (2, 2, 2),
            (("periodic", "dirichlet"), "dirichlet", "dirichlet"),
        )
    with pytest.raises(ValueError, match="unknown boundary"):
        EdgeGrid((0, 0, 0), (1, 1, 1), (2, 2, 2), ("open", "dirichlet", "dirichlet"))
