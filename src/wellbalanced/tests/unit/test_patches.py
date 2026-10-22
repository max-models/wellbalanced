"""The patch grid: the tensor grid's operator on one level, layouts, regions, datasets.

On a single uniform level the patches are the tensor grid's cells in the same order,
so the operators agree to round-off, except that across a radial wall or surface in r–z
patchwork divides by the mean radius of the two values it couples (which keeps the
uniform axial field a null field) where the tensor grid divides by the wall's: both
second order. Refined, the closed-form checks of the field models
(``test_inductive*.py``) run on both grids.
"""

from __future__ import annotations

import numpy as np
import pytest

from wellbalanced.spatial import CellGrid, InductiveField2D, PatchGrid, PlasmaRegion

patchwork = pytest.importorskip("patchwork")


class Box:
    """An axis-aligned box with ``contains_points``."""

    def __init__(self, x0, x1, y0, y1):
        self.bounds = (x0, x1, y0, y1)

    def contains_points(self, points):
        x0, x1, y0, y1 = self.bounds
        return (points[0] > x0) & (points[0] < x1) & (points[1] > y0) & (points[1] < y1)


T_SHAPE = PlasmaRegion([Box(0.0, 0.12, 0.0, 0.17), Box(0.0, 0.19, 0.17, 0.42)])


@pytest.mark.parametrize(
    ("geometry", "bounds"),
    [
        ("cartesian", ((-0.1, 0.2), (0.0, 0.1))),
        ("cylindrical", ((0.0, 0.3), (-0.1, 0.2))),
    ],
)
@pytest.mark.parametrize(
    "boundaries", [("dirichlet", "dirichlet"), ("neumann", "periodic")]
)
def test_one_uniform_level_is_the_tensor_grid(geometry, bounds, boundaries):
    shape = (12, 9)
    (x0, x1), (y0, y1) = bounds
    tensor = CellGrid(
        np.linspace(x0, x1, shape[0] + 1),
        np.linspace(y0, y1, shape[1] + 1),
        geometry=geometry,
        boundaries=boundaries,
    )
    hierarchy = patchwork.AMRHierarchy(
        shape, bounds, periodic=(False, boundaries[1] == "periodic"), geometry=geometry
    )
    patches = PatchGrid(hierarchy, boundaries=boundaries)
    np.testing.assert_allclose(patches.xx, tensor.xx.ravel())
    np.testing.assert_allclose(patches.cell_volumes, tensor.cell_volumes.ravel())
    mu = tensor.cell_field(lambda x, y: 1.0 + x**2)
    live = tensor.mask(Box(x0 + 0.05, x1 - 0.05, -1.0, 1.0))
    # rows next to a radial wall or surface differ in r–z (see the module docstring)
    radial_wall = boundaries[0] == "dirichlet" and geometry == "cylindrical"
    for args, wall in (((), radial_wall), ((mu, live), geometry == "cylindrical")):
        k_tensor = tensor.curl_curl(*args)
        k_patches = patches.curl_curl(*(a.ravel() for a in args))
        difference = abs(k_patches - k_tensor).tocsr()
        scale = abs(k_tensor).max()
        if not wall:
            assert difference.max() < 1e-12 * scale
            continue
        coo = difference.tocoo()
        i, j = np.divmod(np.unique(coo.row[coo.data > 1e-12 * scale]), shape[1])
        # on the outer radial wall, or with a radial face to an inactive cell
        beside = i == shape[0] - 1
        if args:
            inactive = np.pad(~live, ((1, 1), (0, 0)), constant_values=False)
            beside |= inactive[i, j] | inactive[i + 2, j]
        assert np.all(beside)


def test_the_volumes_of_refined_rings_fill_the_cylinder():
    grid = PatchGrid.refined(
        ((0.0, 0.3), (0.0, 0.5)),
        (6, 10),
        [[((0.0, 0.15), (0.1, 0.3))], [((0.05, 0.1), (0.15, 0.25))]],
        geometry="cylindrical",
    )
    assert grid.num_levels == 3 and grid.on_axis
    assert grid.cell_volumes.sum() == pytest.approx(np.pi * 0.3**2 * 0.5, rel=1e-12)
    assert grid.coordinate_names == ("r", "z") and grid.dims == ("cell",)
    # the finest cells are where the boxes asked for them
    finest = grid.cell_areas == grid.cell_areas.min()
    assert np.all((grid.xx[finest] > 0.03) & (grid.xx[finest] < 0.12))


def test_refining_by_a_function_of_the_centres():
    def near_the_coil(x, y, level):
        return np.hypot(x - 0.13, y - 0.07) < 0.03 * 2.0 ** (-level)

    grid = PatchGrid.refined(
        ((0.0, 0.5), (-0.1, 0.4)),
        (25, 25),
        near_the_coil,
        levels=3,
        geometry="cylindrical",
    )
    assert grid.num_levels == 4
    with pytest.raises(ValueError, match="number of levels"):
        PatchGrid.refined(((0.0, 1.0), (0.0, 1.0)), (4, 4), near_the_coil)
    k = grid.locate(0.13, 0.07)
    assert grid.cell_areas[k] == grid.cell_areas.min()


def test_from_edges_follows_the_tensor_grid_where_it_is_fine():
    from wellbalanced.spatial import stretched_edges

    edges = stretched_edges(-0.4, 0.4, -0.03, 0.03, 1e-3, ratio=1.08)
    grid = PatchGrid.from_edges(edges, edges)
    core = (np.abs(grid.xx) < 0.03) & (np.abs(grid.yy) < 0.03)
    np.testing.assert_allclose(grid.cell_areas[core], 1e-6, rtol=1e-9)
    # about as many cells as the tensor grid, not its fine strips across the domain
    assert grid.size < 1.2 * (edges.size - 1) ** 2


def test_refining_where_a_field_is_large():
    grid = PatchGrid.refined(
        ((0.0, 1.0), (0.0, 1.0)), (8, 8), [[((0.25, 0.75), (0.25, 0.75))]]
    )
    hot = np.hypot(grid.xx - 0.5, grid.yy - 0.5) < 0.1
    finer = grid.refined_where(hot)
    assert finer.num_levels == grid.num_levels + 1
    assert finer.size > grid.size
    k = finer.locate(0.5, 0.5)
    assert finer.cell_areas[k] == pytest.approx(grid.cell_areas.min() / 4)
    # the rest keeps its refinement
    assert finer.cell_areas[finer.locate(0.3, 0.3)] == pytest.approx(
        grid.cell_areas.min()
    )
    with pytest.raises(ValueError, match="leaf cells"):
        grid.refined_where(hot[:-1])


def test_wall_cells_of_a_region_on_patches():
    grid = PatchGrid.refined(
        ((0.0, 0.25), (-0.05, 0.45)),
        (25, 50),
        [[((0.08, 0.15), (0.0, 0.2))]],
        geometry="cylindrical",
    )
    cells = T_SHAPE.mask(grid)
    walls = T_SHAPE.wall_faces(grid)
    assert (walls <= cells).all() and walls.any()
    # the axis is not a wall: the axis cells are walls only at the two ends
    axis = cells & (grid.xx == grid.xx.min())
    assert set(np.round(grid.yy[axis & walls], 4)) <= {0.005, 0.415}
    # the radial wall of the driver (refined there) is a wall
    assert walls[grid.locate(0.119, 0.1)]


def test_a_field_on_patches_and_its_dataset():
    grid = PatchGrid.refined(
        ((-0.1, 0.1), (-0.1, 0.1)), (20, 20), [[((-0.05, 0.05), (-0.05, 0.05))]]
    )
    model = InductiveField2D(
        grid=grid,
        frequency=1e6,
        electron_density=lambda x, y: 1e18 * np.exp(-(x**2 + y**2) / 0.03**2),
        collision_frequency=1e7,
        plasma_region=PlasmaRegion([Box(-0.08, 0.08, -0.08, 0.08)]),
    )
    assert model.shape == (grid.size,)
    np.testing.assert_array_equal(model.plasma, model.electron_density > 0.0)
    dataset = model.solve(1.0).to_dataset()
    assert dataset["vector_potential"].dims == ("cell",)
    assert set(dataset.coords) == {"x", "y"}


def test_grid_and_edges_are_checked():
    hierarchy = patchwork.AMRHierarchy((4, 4))
    with pytest.raises(ValueError, match="disagree"):
        PatchGrid(hierarchy, boundaries=("dirichlet", "periodic"))
    grid = PatchGrid(hierarchy)
    with pytest.raises(ValueError, match="not both"):
        InductiveField2D([0.0, 1.0], [0.0, 1.0], frequency=1e6, grid=grid)
    with pytest.raises(ValueError, match="geometry"):
        InductiveField2D(frequency=1e6, grid=grid, geometry="cylindrical")
    with pytest.raises(ValueError, match="edges or a grid"):
        InductiveField2D(frequency=1e6)
    with pytest.raises(ValueError, match="whole number"):
        PatchGrid.from_edges([0.0, 0.3, 1.0], [0.0, 1.0])
