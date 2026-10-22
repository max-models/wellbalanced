"""The cell grid: geometry, the curl-curl operator against known fields, masks."""

from __future__ import annotations

import numpy as np
import pytest

from wellbalanced.spatial import CellGrid


class Disk:
    """A disk with ``contains_points``, so these tests need no simgeometry."""

    def __init__(self, centre, radius):
        self.centre, self.radius = np.asarray(centre), radius

    def contains_points(self, points):
        return np.hypot(*(points - self.centre[:, None])) < self.radius


def test_volumes_of_a_cut_and_of_rings():
    cut = CellGrid(np.linspace(0, 2, 5), np.linspace(0, 1, 3))
    assert cut.cell_volumes.sum() == pytest.approx(2.0)
    assert cut.coordinate_names == ("x", "y")
    rings = CellGrid(
        np.linspace(0, 0.3, 31), np.linspace(0, 0.5, 6), geometry="cylindrical"
    )
    assert rings.cell_volumes.sum() == pytest.approx(np.pi * 0.3**2 * 0.5)
    assert rings.coordinate_names == ("r", "z") and rings.on_axis


def test_curl_curl_of_known_fields_in_r_z():
    """``a = r`` (uniform B_z) has no curl curl, to the axis; ``a = r^3`` gives ``-8 r``."""
    grid = CellGrid(
        np.linspace(0, 1, 41),
        np.linspace(0, 1, 21),
        geometry="cylindrical",
        boundaries=("dirichlet", "periodic"),
    )
    operator = grid.curl_curl()
    uniform = (operator @ grid.xx.ravel()).reshape(grid.shape) / grid.cell_volumes
    np.testing.assert_allclose(uniform[:-1], 0.0, atol=1e-10)
    cubic = (operator @ (grid.xx**3).ravel()).reshape(grid.shape) / grid.cell_volumes
    np.testing.assert_allclose(cubic[1:-1], -8.0 * grid.xx[1:-1], rtol=1e-10)


def test_curl_curl_in_a_cut_is_minus_the_laplacian():
    grid = CellGrid(np.linspace(0, 1, 21), np.linspace(0, 1, 11))
    quadratic = grid.xx**2 + 3.0 * grid.yy**2
    result = (grid.curl_curl() @ quadratic.ravel()).reshape(grid.shape)
    interior = (result / grid.cell_volumes)[1:-1, 1:-1]
    np.testing.assert_allclose(interior, -8.0, rtol=1e-10)


def test_the_curl_of_a_uniform_field_next_to_the_axis():
    grid = CellGrid(np.linspace(0, 1, 11), np.linspace(0, 1, 5), geometry="cylindrical")
    b_r, b_z = grid.curl(0.5 * 3.0 * grid.xx)  # B_z = 3
    np.testing.assert_allclose(b_z, 3.0)
    np.testing.assert_allclose(b_r, 0.0)


def test_inactive_cells_are_walls_at_their_faces():
    """``-u'' = 1`` on the active cells ``[0, 1]`` of a larger grid, ``u = 0`` at its ends."""
    edges = np.linspace(-0.5, 1.5, 201)
    grid = CellGrid(edges, [0.0, 1.0], boundaries=("dirichlet", "neumann"))
    active = (grid.xx > 0.0) & (grid.xx < 1.0)
    operator = grid.curl_curl(active=active)
    cells = np.flatnonzero(active.ravel())
    sub = operator[cells][:, cells]
    u = np.linalg.solve(sub.toarray(), grid.cell_volumes.ravel()[cells])
    x = grid.xx.ravel()[cells]
    np.testing.assert_allclose(u, 0.5 * x * (1.0 - x), atol=2e-5)
    assert np.all(operator[np.flatnonzero(~active.ravel())].toarray() == 0.0)


def test_masks_take_any_shape_with_contains_points():
    grid = CellGrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
    disk = grid.mask(Disk((0.0, 0.0), 0.5))
    assert disk.sum() == pytest.approx(np.pi * 0.25 / 0.01, rel=0.1)
    both = grid.union_mask([Disk((-0.5, 0.0), 0.2), Disk((0.5, 0.0), 0.2)])
    assert both.sum() == 2 * grid.mask(Disk((0.5, 0.0), 0.2)).sum()


def test_invalid_grids():
    with pytest.raises(ValueError, match="increasing"):
        CellGrid([0.0, 0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="geometry"):
        CellGrid([0.0, 1.0], [0.0, 1.0], geometry="spherical")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="r >= 0"):
        CellGrid([-1.0, 1.0], [0.0, 1.0], geometry="cylindrical")
    with pytest.raises(ValueError, match="periodic"):
        CellGrid(
            [0.0, 1.0],
            [0.0, 1.0],
            geometry="cylindrical",
            boundaries=("periodic", "dirichlet"),
        )
    with pytest.raises(ValueError, match="boundary"):
        CellGrid([0.0, 1.0], [0.0, 1.0], boundaries=("open", "dirichlet"))  # type: ignore[arg-type]
