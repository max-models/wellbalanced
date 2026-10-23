"""`Fluid2D` on refined patches: conservation across levels, convergence, sharp features.

One implicit step of all levels together on the leaf cells, one flux per leaf face
(a coarse cell has one face per finer neighbour), the coarse value of a face between
levels interpolated along it. The cross sections, mobilities and the hydrogen
reaction set are synthetic (as in ``test_fluid2d.py``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from plasmacoll.constants import ELEMENTARY_CHARGE
from scipy.special import erf

from wellbalanced import ELECTRON, Reaction, Species
from wellbalanced.driver import illustrative_hydrogen_driver
from wellbalanced.fluid1d import FluidSpecies
from wellbalanced.spatial import Fluid2D, PatchGrid

pytest.importorskip("patchwork")

E = ELEMENTARY_CHARGE
H2_ION = Species.atomic("H2+", 2.016, 1)
ARGON_ION = Species.atomic("Ar+", 39.948, 1)
TE, MU_E, MU_I = 4.0, 2.0e4, 1.0e3
TG = 0.05
D_A = MU_I * MU_E * (TE + TG) / (MU_I + MU_E)  # ambipolar diffusion [m^2/s]


@pytest.fixture(scope="module")
def driver():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return illustrative_hydrogen_driver(coupling="solenoid", field_factor=1.0)


def hydrogen_fluid(driver, grid, **settings) -> Fluid2D:
    tg = driver.model_options["gas_temperature"]
    species = [FluidSpecies(ELECTRON, 1.0)]
    for s in driver.species:
        if s.charge and s.name != "e":
            thermal = np.sqrt(8.0 * E * tg / (np.pi * s.mass))
            mobility = E * driver.ion_mean_free_path / (s.mass * thermal)
            species.append(FluidSpecies(s, mobility))
    return Fluid2D(
        grid,
        species,
        driver.reactions,
        fixed=driver.fixed,
        gas_temperature=tg,
        momentum_transfer=driver.momentum_transfer,
        ion_mean_free_path=driver.ion_mean_free_path,
        **settings,
    )


def driver_patches(driver, base: int = 4) -> PatchGrid:
    """The r–z driver, two levels finer at the side wall and one at the end plates."""
    radius, length = driver.radius, driver.length
    rows = round(base * length / radius)
    return PatchGrid.refined(
        ((0.0, radius), (0.0, length)),
        (base, rows),
        [
            [((0.7 * radius, radius), (0.0, length))],
            [((0.85 * radius, radius), (0.2 * length, 0.8 * length))],
        ],
        geometry="cylindrical",
    )


def levels_of(grid: PatchGrid) -> np.ndarray:
    hierarchy = grid.hierarchy
    return np.array([hierarchy.patches[p].level for p in grid.cells.patch])


# --- conservation ---------------------------------------------------------------------


@pytest.mark.parametrize("closure", ["quasineutral", "poisson"])
def test_a_time_step_on_three_levels_conserves_particles_and_energy(
    driver, closure
) -> None:
    """Over a backward Euler step the totals change by the step times the balances."""
    if closure == "quasineutral":
        grid = driver_patches(driver)
        fluid = hydrogen_fluid(driver, grid, heating=1e6)
        shape = 0.5 + np.cos(0.5 * np.pi * grid.xx / driver.radius)
        ions = {"H+": 0.25, "H2+": 0.5, "H3+": 0.25}
        start = fluid.pack({k: v * 1e18 * shape for k, v in ions.items()}, 5.0, 25.0)
    else:
        grid = PatchGrid.refined(
            ((0.0, 0.01), (0.0, 0.008)),
            (10, 8),
            [
                [((0.004, 0.006), (0.003, 0.005))],
                [((0.0045, 0.0055), (0.0035, 0.0045))],
            ],
        )
        ionization = Reaction(("e", "Ar"), ("e", "e", "Ar+"), 1e-16, name="ionization")
        fluid = Fluid2D(
            grid,
            [FluidSpecies(ELECTRON, 30.0), FluidSpecies(ARGON_ION, 0.1)],
            [ionization],
            fixed={"Ar": 1e21},
            potential="poisson",
            heating=1e4,
        )
        bump = 1e14 * np.exp(-((grid.xx - 0.005) ** 2 + (grid.yy - 0.004) ** 2) / 1e-5)
        start = fluid.pack({"e": bump + 1e12, "Ar+": 1.1 * bump + 1e12}, 3.0, 0.0)
    assert set(levels_of(grid)) == {0, 1, 2}
    dt = 1e-9
    new, ok = fluid.step(start, dt)
    assert ok
    before, after = fluid.state(start), fluid.state(new)
    balance = fluid.balance(after)
    for s in fluid.species:
        change = (
            after.total(after.densities[s.name])
            - before.total(before.densities[s.name])
        ) / dt
        gross = abs(change) + abs(balance[s.name]) + 1.0
        assert abs(change - balance[s.name]) < 1e-8 * gross, s.name
    energy = [
        1.5 * E * st.total(st.electron_density * st.electron_temperature)
        for st in (before, after)
    ]
    terms = fluid.energy_balance(after)
    change = (energy[1] - energy[0]) / dt
    assert abs(change - sum(terms.values())) < 1e-8 * max(
        abs(v) for v in terms.values()
    )


def test_the_faces_between_levels_cancel_to_round_off(driver) -> None:
    """Whatever the state, the interior fluxes of every species and of the energy
    cancel in the sum over the cells: only the walls change a total."""
    grid = driver_patches(driver)
    fluid = hydrogen_fluid(driver, grid, heating=1e6)
    rng = np.random.default_rng(1)
    ions = ("H+", "H2+", "H3+")
    start = fluid.pack(
        {k: 1e18 * rng.uniform(0.2, 1.0, grid.size) for k in ions},
        rng.uniform(2.0, 8.0, grid.size),
        rng.uniform(0.0, 30.0, grid.size),
        {k: tuple(rng.normal(0.0, 3e3, (2, grid.size))) for k in ions},
    )
    densities, velocities, te, phi = fluid._unpack(start)
    interior, walls, _, momentum = fluid._transport(densities, velocities, te, phi)
    assert fluid._coarse_fine is not None and fluid._coarse_fine[0].size > 0
    for name in interior:
        divergence = fluid._divergence(interior[name], walls[name])
        total = float(divergence @ fluid.volumes)
        out = float(walls[name] @ fluid._wall_area)
        gross = float(np.abs(interior[name]) @ fluid._area)
        assert abs(total - out) < 1e-14 * gross, name


def test_the_composite_update_is_patchworks_refluxed_update(driver) -> None:
    """Fluid2D's fluxes in an explicit step of patchwork, refluxed by its FluxRegister.

    Each patch takes the fluxes of its faces that are leaf faces and a wrong flux (zero)
    on the coarse faces under finer patches; after the register's reflux the leaf cells
    have changed as the divergence of Fluid2D's faces says, to round-off.
    """
    patchwork = pytest.importorskip("patchwork")
    if not hasattr(patchwork, "FluxRegister"):
        pytest.skip("needs patchwork's FluxRegister (its refluxing)")
    FluxRegister = patchwork.FluxRegister

    grid = driver_patches(driver)
    fluid = hydrogen_fluid(driver, grid, heating=1e6)
    shape = 0.5 + np.cos(0.5 * np.pi * grid.xx / driver.radius)
    shape = shape * (1.0 + 0.3 * np.sin(7.0 * grid.yy / driver.length))
    start = fluid.pack({"H+": 3e17 * shape, "H2+": 5e17 * shape, "H3+": 2e17}, 5.0)
    densities, velocities, te, phi = fluid._unpack(start)
    interior, walls, _, _ = fluid._transport(densities, velocities, te, phi)

    cells, hierarchy = grid.cells, grid.hierarchy
    finest = hierarchy.max_level
    level = finest - np.log2(cells.span).round().astype(int)

    def key(axis, cell, plane):
        """A face of a leaf cell: its axis, level and indices on that level."""
        span = cells.span[cell]
        index = [plane // span, cells.lattice[cell, 1 - axis] // span][:: 1 - 2 * axis]
        return list(zip(np.full(cell.size, axis), level[cell], *index, strict=True))

    low, high, axis = fluid._low, fluid._high, fluid._axis
    finer = np.where(cells.span[low] <= cells.span[high], low, high)
    plane = cells.lattice[high, axis]
    wall, sign = fluid._wall_cell, fluid._wall_sign
    wall_plane = cells.lattice[wall, fluid._wall_axis] + (sign > 0) * cells.span[wall]
    faces = {}
    for a in (0, 1):
        on = axis == a
        for k, face in zip(
            np.flatnonzero(on), key(a, finer[on], plane[on]), strict=True
        ):
            faces[face] = ("interior", k)
        on = fluid._wall_axis == a
        for k, face in zip(
            np.flatnonzero(on), key(a, wall[on], wall_plane[on]), strict=True
        ):
            faces[face] = ("wall", k)

    hierarchy.allocate_field("u", halo=0, split=None)
    register = FluxRegister(hierarchy, cells)
    dt = 1e-9
    for name in ("H2+", "energy"):
        values = 1e18 * (1.0 + 0.1 * np.sin(np.arange(grid.size)))
        cells.to_patches(values, "u")
        register.reset()

        def fluxes(patch, name=name):
            (i0, i1), (j0, j1) = patch.box
            result = []
            for a in (0, 1):
                shape = (i1 - i0 + (a == 0), j1 - j0 + (a == 1))
                flux = np.zeros(shape)
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        found = faces.get((a, patch.level, i0 + i, j0 + j))
                        if found is None:
                            continue
                        kind, k = found
                        if kind == "interior":
                            flux[i, j] = interior[name][k]
                        else:
                            flux[i, j] = sign[k] * walls[name][k]
                result.append(flux)
            return tuple(result)

        register.update("u", fluxes, dt)
        register.reflux("u")
        expected = values - dt * fluid._divergence(interior[name], walls[name])
        assert cells.from_patches("u") == pytest.approx(expected, rel=1e-12), name


# --- a blob of plasma: convergence and a sharp feature ----------------------------------

SIZE = 0.1


def blob_fluid(
    grid: PatchGrid, width: float, centre: tuple[float, float], length: float
) -> tuple[Fluid2D, np.ndarray]:
    """A Gaussian source of ``width`` at ``centre``, lost in the volume over ``length``.

    Isothermal electrons and drift-diffusion ions, as in the ambipolar tests of
    ``test_fluid2d.py``; both are lost at ``nu = D_a / length^2`` (two first-order
    reactions with a fixed partner), so the plasma stays near the source, plus a
    uniform source of 1 % of the peak that keeps a floor everywhere. The source is the
    exact mean of the Gaussian over every cell, so that every grid makes as many.
    Returns the model and a first guess, the local balance of source and loss.
    """
    centres, sizes = grid.cells.centers, grid.cells.sizes
    mean = np.ones(grid.size)
    for axis in (0, 1):
        scale = np.sqrt(2.0) * width
        lower = (centres[:, axis] - 0.5 * sizes[:, axis] - centre[axis]) / scale
        upper = (centres[:, axis] + 0.5 * sizes[:, axis] - centre[axis]) / scale
        mean *= (
            0.5 * np.sqrt(np.pi) * scale * (erf(upper) - erf(lower)) / sizes[:, axis]
        )
    source = 1e22 * (mean + 1e-2)
    nu, partner = D_A / length**2, 1e20
    losses = [
        Reaction((name, "X"), ("X",), nu / partner, name=f"{name} loss")
        for name in ("e", "H2+")
    ]
    fluid = Fluid2D(
        grid,
        [FluidSpecies(ELECTRON, MU_E), FluidSpecies(H2_ION, MU_I)],
        losses,
        fixed={"X": partner},
        ions="drift-diffusion",
        electron_temperature=TE,
        gas_temperature=TG,
        sources={"e": source, "H2+": source},
    )
    return fluid, source / nu


def blob(grid: PatchGrid, width: float, centre=(0.5 * SIZE, 0.5 * SIZE), length=5e-3):
    """The steady density of :func:`blob_fluid`, and the model's imbalance."""
    fluid, guess = blob_fluid(grid, width, centre, length)
    potential = TE * (5.0 + np.log(guess / guess.max()))  # Boltzmann electrons
    state = fluid.steady_state(
        fluid.pack({"H2+": guess}, TE, potential), dt=3e-11, max_dt=1e-6
    )
    assert state.converged
    balance = fluid.balance(state)["H2+"]
    made = float(fluid._sources["H2+"] @ fluid.volumes)
    assert abs(balance) < 1e-10 * made
    return state.densities["H2+"]


def square(base: int, *boxes: float) -> PatchGrid:
    """The square of side :data:`SIZE`, refined about its middle by half-widths."""
    return PatchGrid.refined(
        ((0.0, SIZE), (0.0, SIZE)),
        (base, base),
        [[((0.5 * SIZE - b, 0.5 * SIZE + b),) * 2] for b in boxes],
    )


def near_the_middle(grid: PatchGrid, half_width: float) -> np.ndarray:
    """The cells within ``half_width`` of the middle of the square along both axes."""
    return (np.abs(grid.x - 0.5 * SIZE) < half_width) & (
        np.abs(grid.y - 0.5 * SIZE) < half_width
    )


def on_cells(grid: PatchGrid, fine: PatchGrid, values: np.ndarray) -> np.ndarray:
    """The volume mean over the cells of ``grid`` of values on a finer ``fine``."""
    leaf = grid.cells.locate(np.stack([fine.x, fine.y]))
    volume = np.bincount(leaf, fine.cell_volumes, grid.size)
    return np.bincount(leaf, values * fine.cell_volumes, grid.size) / volume


def test_refined_patches_converge_with_the_uniform_fine_grid() -> None:
    """A smooth blob (width 1 cm, loss length 5 mm in a 10 cm square).

    The uniform grids converge at about second order (the peak 31 % below that of
    32 x 32 cells on 8 x 8, 8 % on 16 x 16); one level of patches over the blob gives
    the peak of the uniform grid as fine as the patches to 1e-4 and its density about
    the blob to 3e-4 of the peak, with two thirds of its cells or fewer, where the
    uniform grid as coarse as the base is off by far more. Outside the patches the
    coarse cells keep their own error (the wall layer of the floor, 5 mm thick, on
    cells of 12.5 and 6.25 mm).
    """
    peaks, uniform = [], {}
    for base in (8, 16, 32):
        grid = square(base)
        uniform[base] = (grid, blob(grid, 0.01))
        peaks.append(uniform[base][1].max())
    # about second order: the successive changes of the peak shrink by 2.9 (and 3.2
    # to 64 x 64 cells, not run here)
    assert (peaks[1] - peaks[0]) / (peaks[2] - peaks[1]) > 2.5
    for base in (8, 16):
        refined = square(base, 0.025)
        assert set(levels_of(refined)) == {0, 1}
        density = blob(refined, 0.01)
        fine, reference = uniform[2 * base]
        coarse, coarse_density = uniform[base]
        assert refined.size < 0.7 * fine.size
        peak = reference.max()
        assert density.max() == pytest.approx(peak, rel=1e-4)
        difference = np.abs(density - on_cells(refined, fine, reference)) / peak
        blob_cells = near_the_middle(refined, 0.025)
        assert np.all(levels_of(refined)[blob_cells] == 1)
        assert difference[blob_cells].max() < 3e-4
        coarse_difference = np.abs(coarse_density - on_cells(coarse, fine, reference))
        coarse_blob = coarse_difference[near_the_middle(coarse, 0.025)] / peak
        assert coarse_blob.max() > 20.0 * difference[blob_cells].max()


def test_two_levels_resolve_a_sharp_blob_the_base_cannot() -> None:
    """A narrow blob (width 3 mm, off the cell centres) on 8 x 8 cells of 12.5 mm.

    Two levels of patches about it reach the cells of the 32 x 32 grid there: its
    peak to 5e-4 and its density on the finest patch to 3e-3 of the peak, with a
    fifth of its cells, where the base alone has 45 % of the peak. (Without the
    interpolation along the faces between levels the finest patch is off by 3.4e-3.)
    """
    width, length = 3e-3, 3e-3
    centre = (0.52 * SIZE, 0.47 * SIZE)
    fine = square(32)
    reference = blob(fine, width, centre, length)
    peak = reference.max()
    assert blob(square(8), width, centre, length).max() < 0.5 * peak
    refined = square(8, 0.012, 0.006)
    assert set(levels_of(refined)) == {0, 1, 2}
    assert refined.size < 0.25 * fine.size
    density = blob(refined, width, centre, length)
    assert density.max() == pytest.approx(peak, rel=2e-3)
    finest = levels_of(refined) == 2
    difference = np.abs(density - on_cells(refined, fine, reference)) / peak
    assert difference[finest].max() < 5e-3
