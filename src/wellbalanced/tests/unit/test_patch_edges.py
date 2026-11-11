"""The 3D field on refined patches: patchwork's edge space behind `InductiveField3D`.

On one level the composite edges and faces are the uniform grid's, so the field is
`EdgeGrid`'s to round-off. With patches over the conductors' surfaces it is the fine
uniform grid's wherever the coarse cells away from them carry the same field.
"""

from __future__ import annotations

import numpy as np
import pytest

from wellbalanced.spatial import (
    Coil3D,
    Conductor3D,
    EdgeGrid,
    Feed3D,
    ImpedanceConductor3D,
    InductiveField3D,
)
from wellbalanced.spatial.cutcell import cut_cells, staircased_cells

simgeometry = pytest.importorskip("simgeometry")
patchwork = pytest.importorskip("patchwork")
from wellbalanced.spatial.patch_edges import PatchEdgeGrid, surface_tags  # noqa: E402

Cuboid, Cylinder, Tube = simgeometry.Cuboid, simgeometry.Cylinder, simgeometry.Tube
AMRHierarchy = patchwork.AMRHierarchy

SIGMA, TURNS, BOX, COIL = 1e6, 100.0, 0.06, 0.045
PERIODIC_Z = ("dirichlet", "dirichlet", "periodic")
COIL_3D = Coil3D(Cuboid((0, 0, 0), (2 * COIL, 2 * COIL, 4.0)), (0, 0, 1), TURNS)
TUBE = Tube((0, 0, -1.0), (0, 0, 1.0), 0.02, 0.026)


def quarter_box(cells: int) -> tuple[tuple, tuple, tuple]:
    """A quarter of a square cross-section, z-periodic, two cells long."""
    h = BOX / cells
    return (0, 0, 0), (BOX, BOX, 2 * h), (cells, cells, 2)


def one_level(lower, upper, cells, boundaries) -> PatchEdgeGrid:
    periodic = tuple(b == "periodic" for b in boundaries)
    hierarchy = AMRHierarchy(
        base_shape=cells,
        bounds=tuple(zip(lower, upper, strict=True)),
        periodic=periodic,
    )
    return PatchEdgeGrid(hierarchy, boundaries)


def tube_field(grid, **kwargs):
    model = InductiveField3D(
        grid,
        frequency=1e6,
        coils=[COIL_3D],
        impedance_conductors=[ImpedanceConductor3D([TUBE], SIGMA, "metal")],
        **kwargs,
    )
    return model, model.solve(1.0)


def test_one_level_has_the_uniform_grids_geometry():
    box = quarter_box(6)
    uniform = EdgeGrid(*box, PERIODIC_Z)
    patches = one_level(*box, PERIODIC_Z)
    assert (patches.n_edges, patches.n_faces) == (uniform.n_edges, uniform.n_faces)
    assert not patches.hanging_edges.any()
    assert np.count_nonzero(patches.free_edges) == np.count_nonzero(uniform.free_edges)
    for name in ("edge_lengths", "dual_areas", "face_areas", "dual_lengths"):
        a, b = np.sort(getattr(patches, name)), np.sort(getattr(uniform, name))
        assert np.allclose(a, b, rtol=1e-12, atol=0), name
    # the same fractions, numbered differently
    for a, b in zip(
        cut_cells(patches, [TUBE]).__dict__.values(),
        cut_cells(uniform, [TUBE]).__dict__.values(),
        strict=True,
    ):
        assert np.allclose(np.sort(a), np.sort(b), atol=1e-13)


@pytest.mark.parametrize(
    "boundaries", [PERIODIC_Z, (("natural", "dirichlet"), "dirichlet", "periodic")]
)
def test_one_level_gives_the_uniform_grids_field(boundaries):
    box = quarter_box(8)
    _, uniform = tube_field(EdgeGrid(*box, boundaries))
    _, patches = tube_field(one_level(*box, boundaries))
    assert abs(patches.impedance - uniform.impedance) < 1e-12 * abs(uniform.impedance)
    assert patches.loss == pytest.approx(uniform.loss, rel=1e-12)
    assert patches.magnetic_field_at_cells.shape == (3, 8 * 8 * 2)


def test_one_level_volume_conductors_with_subcell_samples():
    box = quarter_box(6)
    slab = Cuboid((0.015, 0.015, 0.0), (0.01, 0.02, 4.0))
    results = []
    for grid in (EdgeGrid(*box, PERIODIC_Z), one_level(*box, PERIODIC_Z)):
        model = InductiveField3D(
            grid,
            frequency=1e5,
            coils=[COIL_3D],
            conductors=[Conductor3D([slab], 1e5, "slab")],
            samples=2,
        )
        results.append(model.solve(1.0))
    assert results[1].impedance == pytest.approx(results[0].impedance, rel=1e-12)
    assert results[1].conductor_losses["slab"] == pytest.approx(
        results[0].conductor_losses["slab"], rel=1e-12
    )


def test_refined_patches_give_the_fine_grids_field_on_fewer_edges():
    lower, upper, cells = quarter_box(8)
    fine = EdgeGrid(lower, upper, tuple(2 * n for n in cells), PERIODIC_Z)
    patches = PatchEdgeGrid.around(
        lower, upper, cells, [TUBE], boundaries=PERIODIC_Z, efficiency=0.95
    )
    assert patches.hierarchy.max_level == 1 and patches.hanging_edges.any()
    assert patches.n_edges < fine.n_edges
    _, reference = tube_field(fine)
    model, solution = tube_field(patches)
    assert solution.impedance == pytest.approx(reference.impedance, rel=1e-6)
    assert solution.loss == pytest.approx(reference.loss, rel=1e-6)
    # the losses are what the source delivers, hanging edges included
    assert solution.loss == pytest.approx(solution.source_power.real, rel=1e-9)
    # hanging edges follow from the others
    edges = patches.edges
    assert np.allclose(edges.reduction @ solution.potential, solution.potential)


def test_the_coil_current_is_divergence_free_on_patches():
    patches = PatchEdgeGrid.around(
        *quarter_box(8), [TUBE], boundaries=PERIODIC_Z, efficiency=0.95
    )
    model = InductiveField3D(patches, frequency=1e6, coils=[COIL_3D])
    edges, nodes = patches.edges, patches.node_space
    gradient = edges.gradient @ nodes.reduction
    load = edges.reduction.T @ model._unit_source
    assert abs(gradient.T @ load).max() < 1e-12 * abs(load).max()
    assert abs(patches.curl() @ patches.gradient()).max() == 0


def test_a_feed_on_patches_matches_the_fine_grid():
    """A short coax fed across a gap, with patches over the conductors' surfaces."""
    half, length, gap = 0.015, 0.006, 0.002
    inner = Cylinder((0, 0, gap), (0, 0, 1.0), 0.004)
    outer = Tube((0, 0, -1.0), (0, 0, 1.0), 0.012, 0.014)

    def solve(grid):
        model = InductiveField3D(
            grid,
            frequency=1e6,
            feeds=[Feed3D((0, 0, 0), (0, 0, gap + 0.003), name="feed")],
            impedance_conductors=[
                ImpedanceConductor3D([inner], 5.8e7, "inner"),
                ImpedanceConductor3D([outer], 5.8e7, "outer"),
            ],
        )
        return model.solve(1.0)

    lower, upper = (-half, -half, 0.0), (half, half, length)
    fine = solve(EdgeGrid(lower, upper, (30, 30, 6)))
    patches = PatchEdgeGrid.around(
        lower, upper, (15, 15, 3), [inner, outer], efficiency=0.95
    )
    assert patches.hanging_edges.any()
    solution = solve(patches)
    assert solution.impedance == pytest.approx(fine.impedance, rel=1e-5)
    assert solution.loss == pytest.approx(solution.source_power.real, rel=1e-9)


def test_feed_paths_take_coarse_edges_where_there_are_no_fine_ones():
    hierarchy = AMRHierarchy(base_shape=(4, 4, 4), bounds=((0, 1),) * 3)
    hierarchy.add_patch(level=1, box=((0, 4), (0, 4), (0, 4)))
    grid = PatchEdgeGrid(hierarchy)
    edges, signs, ends = grid.feed_path((0, 0, 0), (1.0, 0.75, 0.0), (0, 1, 2))
    assert np.all(signs == 1)
    assert np.isclose(grid.edge_lengths[edges].sum(), 1.75)
    assert grid.edge_lengths[edges].min() == pytest.approx(0.125)
    assert grid.edge_lengths[edges].max() == pytest.approx(0.25)
    back, signs, _ = grid.feed_path((1.0, 0.75, 0.0), (0, 0, 0), (1, 0, 2))
    assert np.all(signs == -1) and set(back) == set(edges)
    with pytest.raises(ValueError, match="outside"):
        grid.feed_path((2.0, 0, 0), (0, 0, 0), (0, 1, 2))


def test_surface_tags_mark_the_cells_a_surface_cuts():
    hierarchy = AMRHierarchy(base_shape=(8, 4, 4), bounds=((0, 1),) * 3)
    plane = Cuboid((0, 0.5, 0.5), (0.6, 2, 2))  # its face at x = 0.3 cuts cells 2
    tags = surface_tags(hierarchy, 0, [plane])
    assert tags[2].all() and np.count_nonzero(tags) == 16


def test_what_patches_do_not_support():
    box = quarter_box(4)
    grid = one_level(*box, PERIODIC_Z)
    model = InductiveField3D(grid, frequency=1e6, coils=[COIL_3D])
    with pytest.raises(NotImplementedError, match="direct"):
        model.solve(1.0, solver="iterative")
    with pytest.raises(TypeError, match="uniform"):
        staircased_cells(grid, [TUBE])
    with pytest.raises(ValueError, match="periodic"):
        PatchEdgeGrid(grid.hierarchy, ("dirichlet",) * 3)
    with pytest.raises(ValueError, match="per axis"):
        PatchEdgeGrid(grid.hierarchy, ("dirichlet",) * 2)
    with pytest.raises(ValueError, match="3D"):
        PatchEdgeGrid(AMRHierarchy(base_shape=(4, 4)))


def test_positions_and_masks():
    hierarchy = AMRHierarchy(
        base_shape=(4, 4, 4), bounds=((0, 1),) * 3, periodic=(False, False, True)
    )
    hierarchy.add_patch(level=1, box=((0, 4), (0, 4), (0, 8)))
    grid = PatchEdgeGrid(hierarchy)
    centres = grid.cell_centres()
    assert centres.shape == (3, (64 - 16) + 16 * 8)  # 16 base cells refined
    assert np.all((centres > 0) & (centres < 1))
    half = Cuboid((0.25, 0.5, 0.5), (0.5, 2, 2))
    assert np.array_equal(grid.edge_mask(half), grid.edge_midpoints()[0] <= 0.5)
    assert np.array_equal(grid.face_mask(half), grid.face_centres()[0] <= 0.5)
    # the dual halves of the z-faces on the periodic seam wrap into the box
    (below, valid_below), (above, valid_above) = grid.face_dual_points()
    z_faces = grid.face_axes() == 2
    assert valid_below[z_faces].all() and valid_above[z_faces].all()
    assert np.all((below[2] >= 0) & (below[2] <= 1))
    # a coarse face on the patch's side has only its coarse half
    x_faces = grid.face_axes() == 0
    assert not (valid_below[x_faces] & valid_above[x_faces]).all()
