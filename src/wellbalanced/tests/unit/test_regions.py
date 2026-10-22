"""The plasma region: masks on the grid, wall cells, the field's plasma."""

from __future__ import annotations

import numpy as np
import pytest

from wellbalanced.spatial import CellGrid, InductiveField2D, PlasmaRegion


class Box:
    """An axis-aligned box with ``contains_points``, so these tests need no simgeometry."""

    def __init__(self, x0, x1, y0, y1):
        self.bounds = (x0, x1, y0, y1)

    def contains_points(self, points):
        x0, x1, y0, y1 = self.bounds
        return (points[0] > x0) & (points[0] < x1) & (points[1] > y0) & (points[1] < y1)


# a driver (r < 0.12, 0 < z < 0.17) opening into an expansion (r < 0.19, z < 0.42)
T_SHAPE = PlasmaRegion([Box(0.0, 0.12, 0.0, 0.17), Box(0.0, 0.19, 0.17, 0.42)])


def grid():
    return CellGrid(
        np.linspace(0.0, 0.25, 51),
        np.linspace(-0.05, 0.45, 101),
        geometry="cylindrical",
    )


def test_a_t_shaped_region():
    cells = T_SHAPE.mask(grid())
    g = grid()
    assert cells[g.x < 0.12][:, (g.y > 0.0) & (g.y < 0.42)].all()
    assert not cells[(g.x > 0.12)][:, g.y < 0.17].any()
    assert cells[(g.x > 0.12) & (g.x < 0.19)][:, (g.y > 0.17) & (g.y < 0.42)].all()
    hole = PlasmaRegion(T_SHAPE.shapes, excluded=[Box(0.0, 0.05, 0.3, 0.35)])
    assert hole.mask(g).sum() < cells.sum()
    with pytest.raises(ValueError, match="shape"):
        PlasmaRegion([])


def test_wall_cells_are_at_the_edge_but_not_on_the_axis():
    g = grid()
    walls = T_SHAPE.wall_faces(g)
    cells = T_SHAPE.mask(g)
    assert (walls <= cells).all()
    # the axis is not a wall: the axis cells are walls only at the two ends
    axis_walls = np.flatnonzero(walls[0])
    assert set(g.y[axis_walls].round(4)) <= {0.0025, 0.4175}
    # the radial wall of the driver and the step to the expansion are walls
    i = int(np.argmin(np.abs(g.x - 0.1175)))
    j = int(np.argmin(np.abs(g.y - 0.1)))
    assert walls[i, j]


def test_the_field_has_no_plasma_outside_the_region():
    g = grid()
    model = InductiveField2D(
        g.x_edges,
        g.y_edges,
        frequency=1e6,
        geometry="cylindrical",
        electron_density=1e18,
        collision_frequency=1e7,
        plasma_region=T_SHAPE,
    )
    np.testing.assert_array_equal(model.plasma, T_SHAPE.mask(g))
