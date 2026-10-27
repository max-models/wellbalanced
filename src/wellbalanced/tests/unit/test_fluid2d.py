"""The 2D driver fluid: conservation, Fluid1D, closed forms, patches, the global model.

The cross sections, mobilities and the hydrogen reaction set are synthetic (the
placeholders of :func:`~wellbalanced.driver.illustrative_hydrogen_driver`).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
from plasmacoll import CrossSection
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

from wellbalanced import ELECTRON, Reaction, Species
from wellbalanced.driver import RFDriver, illustrative_hydrogen_driver
from wellbalanced.fluid1d import Fluid1D, FluidSpecies
from wellbalanced.global_model import Chamber
from wellbalanced.spatial import (
    CellGrid,
    DriverCoupling,
    Fluid2D,
    InductiveField2D,
    PatchGrid,
    Winding,
    fluid_plasma,
    global_model_plasma,
    stretched_edges,
)

E = ELEMENTARY_CHARGE
ARGON_ION = Species.atomic("Ar+", 39.948, 1)
H2_ION = Species.atomic("H2+", 2.016, 1)
IONIZATION = Reaction.from_cross_section(
    ("e", "Ar"),
    ("e", "e", "Ar+"),
    CrossSection.constant(2e-20, threshold=15.76),
    ELECTRON_MASS,
    electron_energy_loss=15.76,
)


def slab_grid(length: float, cells: int) -> CellGrid:
    """A slab along ``x``: one cell along a periodic ``y``."""
    return CellGrid(
        np.linspace(0.0, length, cells + 1),
        [0.0, 1.0],
        boundaries=("dirichlet", "periodic"),
    )


@pytest.fixture(scope="module")
def driver():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return illustrative_hydrogen_driver(coupling="solenoid", field_factor=1.0)


def hydrogen_species(driver: RFDriver) -> list[FluidSpecies]:
    """The driver's charged species; the ions' mobilities from its mean free path."""
    tg = driver.model_options["gas_temperature"]
    species = [FluidSpecies(ELECTRON, 1.0)]  # the mobility comes from nu
    for s in driver.species:
        if s.charge and s.name != "e":
            thermal = np.sqrt(8.0 * E * tg / (np.pi * s.mass))
            mobility = E * driver.ion_mean_free_path / (s.mass * thermal)
            species.append(FluidSpecies(s, mobility))
    return species


def hydrogen_fluid(driver: RFDriver, grid, **settings) -> Fluid2D:
    return Fluid2D(
        grid,
        hydrogen_species(driver),
        driver.reactions,
        fixed=driver.fixed,
        gas_temperature=driver.model_options["gas_temperature"],
        momentum_transfer=driver.momentum_transfer,
        ion_mean_free_path=driver.ion_mean_free_path,
        **settings,
    )


def hydrogen_guess(fluid: Fluid2D, density: float, te: float, shape):
    ions = {"H+": 0.25, "H2+": 0.5, "H3+": 0.25}
    return fluid.pack(
        {name: share * density * shape for name, share in ions.items()}, te, 5.0 * te
    )


# --- the 1D limit and conservation ----------------------------------------------------


def test_a_slab_with_poissons_equation_is_fluid1d() -> None:
    """The same discharge as Fluid1D's test, on the faces of a 2D grid: to round-off."""
    species = [FluidSpecies(ELECTRON, 1.0e3), FluidSpecies(ARGON_ION, 10.0)]
    settings = {"fixed": {"Ar": 1.0e21}, "gas_temperature": 0.025, "heating": 2.0e4}
    one = Fluid1D(species, [IONIZATION], 0.05, 60, **settings)
    guess = 1e15 * np.sin(np.pi * one.x / 0.05) + 1e12
    reference = one.steady_state(one.pack({"e": guess, "Ar+": guess}, 3.0, 0.0))
    two = Fluid2D(
        slab_grid(0.05, 60), species, [IONIZATION], potential="poisson", **settings
    )
    column = guess[:, None]
    state = two.steady_state(two.pack({"e": column, "Ar+": column}, 3.0, 0.0))
    assert reference.converged and state.converged
    for name in ("e", "Ar+"):
        assert state.densities[name][:, 0] == pytest.approx(
            reference.densities[name], rel=1e-10
        )
    assert state.electron_temperature[:, 0] == pytest.approx(
        reference.electron_temperature, rel=1e-10
    )
    assert state.potential[:, 0] == pytest.approx(reference.potential, abs=1e-9)
    energy = two.energy_balance(state)
    assert energy["heating"] == pytest.approx(2.0e4 * 0.05)  # W per m^2 of the slab
    assert abs(sum(energy.values())) < 1e-10 * energy["heating"]


def driver_cylinder(driver: RFDriver, cells: int) -> CellGrid:
    radius, length = driver.radius, driver.length
    return CellGrid(
        np.linspace(0.0, radius, cells + 1),
        np.linspace(0.0, length, round(cells * length / radius) + 1),
        geometry="cylindrical",
    )


@pytest.mark.parametrize("closure", ["quasineutral", "poisson"])
def test_a_time_step_conserves_particles_and_energy(driver, closure) -> None:
    """The change over a backward Euler step is the step times the balances."""
    if closure == "quasineutral":
        grid = driver_cylinder(driver, 4)
        fluid = hydrogen_fluid(driver, grid, heating=1e6)
        shape = 0.5 + np.cos(0.5 * np.pi * grid.xx / driver.radius)
        start = hydrogen_guess(fluid, 1e18, 5.0, shape)
    else:
        grid = CellGrid(np.linspace(0, 0.01, 7), np.linspace(0, 0.008, 5))
        species = [FluidSpecies(ELECTRON, 30.0), FluidSpecies(ARGON_ION, 0.1)]
        fluid = Fluid2D(
            grid,
            species,
            [IONIZATION],
            fixed={"Ar": 1e21},
            potential="poisson",
            heating=1e4,
        )
        bump = 1e14 * np.exp(-((grid.xx - 0.005) ** 2 + (grid.yy - 0.004) ** 2) / 1e-5)
        start = fluid.pack({"e": bump + 1e12, "Ar+": 1.1 * bump + 1e12}, 3.0, 0.0)
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
    energy = (
        1.5
        * E
        * np.array(
            [
                st.total(st.electron_density * st.electron_temperature)
                for st in (before, after)
            ]
        )
    )
    terms = fluid.energy_balance(after)
    change = (energy[1] - energy[0]) / dt
    assert abs(change - sum(terms.values())) < 1e-8 * max(
        abs(v) for v in terms.values()
    )


# --- closed forms ---------------------------------------------------------------------

TE, MU_E, MU_I, SOURCE = 4.0, 2.0e4, 1.0e3, 1.0e22


def ambipolar_error(geometry: str, cells: int, grid_kind: str = "tensor") -> float:
    """Ambipolar diffusion from a uniform source, against its closed form.

    Slab: ``n = S L / (2 u_B) + S (L^2 / 4 - x^2) / (2 D_a)``; infinite cylinder:
    ``n = S R / (2 u_B) + S (R^2 - r^2) / (4 D_a)``; the Bohm flux at the walls.
    """
    tg = 0.05
    d_a = MU_I * MU_E * (TE + tg) / (MU_I + MU_E)
    u_b = np.sqrt(E * TE / H2_ION.mass)
    size = 0.1
    if geometry == "slab":
        edges = (np.linspace(0, size, cells + 1), [0.0, 1.0])
        settings = {"boundaries": ("dirichlet", "periodic")}
    else:
        edges = (np.linspace(0, size, cells + 1), [0.0, 0.01, 0.02])
        settings = {"geometry": "cylindrical", "boundaries": ("dirichlet", "periodic")}
    if grid_kind == "tensor":
        grid = CellGrid(*edges, **settings)
    else:
        pytest.importorskip("patchwork")
        grid = PatchGrid.from_edges(*edges, **settings)
    fluid = Fluid2D(
        grid,
        [FluidSpecies(ELECTRON, MU_E), FluidSpecies(H2_ION, MU_I)],
        ions="drift-diffusion",
        electron_temperature=TE,
        gas_temperature=tg,
        sources={"e": SOURCE, "H2+": SOURCE},
    )
    x = grid.xx
    if geometry == "slab":
        exact = SOURCE * size / (2 * u_b) + SOURCE / (2 * d_a) * (
            size**2 / 4 - (x - size / 2) ** 2
        )
    else:
        exact = SOURCE * size / (2 * u_b) + SOURCE / (4 * d_a) * (size**2 - x**2)
    state = fluid.steady_state(fluid.pack({"H2+": exact}, TE, 5 * TE))
    assert state.converged
    made = SOURCE * state.volume
    for imbalance in fluid.balance(state).values():
        assert abs(imbalance) < 1e-12 * made
    return float(np.max(np.abs(state.densities["H2+"] / exact - 1.0)))


@pytest.mark.parametrize("geometry", ["slab", "cylinder"])
def test_ambipolar_diffusion_converges_to_its_closed_form(geometry) -> None:
    """First order: the Bohm flux takes the wall cell's density, ``u_B h / (2 D)`` off."""
    coarse, fine = ambipolar_error(geometry, 25), ambipolar_error(geometry, 50)
    assert fine < 4e-3
    assert coarse / fine == pytest.approx(2.0, rel=0.05)


def test_a_single_level_patch_grid_is_the_tensor_grid() -> None:
    assert ambipolar_error("cylinder", 32, "patches") == pytest.approx(
        ambipolar_error("cylinder", 32), rel=1e-9
    )


def collisionless_slab(cells: int):
    """Uniform ionization, cold ions without collisions, Boltzmann electrons.

    The closed form: ``n (1 + u^2) = n_0`` (momentum) and ``n u u_B = S x`` with
    ``u = v / u_B``, so ``u = (1 - sqrt(1 - y^2)) / y`` (``y = 2 x / L`` from the
    middle), ``n_0 = S L / u_B`` and ``n = n_0 / 2`` at the wall, where ``u = 1``.
    """
    length = 0.1
    u_b = np.sqrt(E * TE / H2_ION.mass)
    grid = slab_grid(length, cells)
    fluid = Fluid2D(
        grid,
        [FluidSpecies(ELECTRON, 1e5), FluidSpecies(H2_ION, 1e9)],
        electron_temperature=TE,
        gas_temperature=1e-4,
        sources={"e": SOURCE, "H2+": SOURCE},
    )
    y = np.abs(2 * grid.xx / length - 1)
    u = np.where(y > 0, (1 - np.sqrt(1 - y**2)) / np.where(y > 0, y, 1), 0.0)
    n_0 = SOURCE * length / u_b
    velocity = np.sign(grid.xx - length / 2) * u * u_b
    state = fluid.steady_state(
        fluid.pack({"H2+": n_0 / (1 + u**2)}, TE, 5 * TE, {"H2+": (velocity, 0.0)})
    )
    assert state.converged
    density = state.densities["H2+"][:, 0]
    return density, n_0 / (1 + u[:, 0] ** 2), n_0, state, u_b


def test_ions_with_inertia_reach_the_collisionless_closed_form() -> None:
    """Slowly (about half an order: the sonic point at the wall is singular)."""
    errors, speeds = [], []
    for cells in (25, 50):
        density, exact, n_0, state, u_b = collisionless_slab(cells)
        errors.append(float(np.sqrt(np.mean((density - exact) ** 2)) / n_0))
        assert density.max() < n_0  # the centre density converges from below
        assert density[-1] == pytest.approx(0.5 * n_0, rel=1e-3)  # n_s = n_0 / 2
        speeds.append(state.velocities["H2+"][0][-1, 0] / u_b)
    assert errors[1] < 0.11 and errors[1] < 0.75 * errors[0]
    assert 0.8 < speeds[1] < 1.0 and speeds[1] > speeds[0]  # towards u_B at the wall


# --- the driver ---------------------------------------------------------------------


def column(driver, spacing=5e-3):
    """An infinitely long driver: the plasma column in its solenoid, periodic in z."""
    radius, length = driver.radius, driver.length
    x_edges = np.concatenate(
        [
            np.linspace(0, radius, round(radius / spacing) + 1),
            stretched_edges(radius, 0.2, radius, radius + 0.01, spacing, 1.2)[1:],
        ]
    )
    grid = CellGrid(
        x_edges,
        np.linspace(0, length, 2),
        geometry="cylindrical",
        boundaries=("neumann", "periodic"),
    )
    return grid, grid.xx < radius


def side_wall_driver(driver: RFDriver, h_factor: float) -> RFDriver:
    """The driver whose ions leave through the side wall only, with ``h_factor``."""
    chamber = Chamber(
        volume=np.pi * driver.radius**2 * driver.length,
        wall_area=2 * np.pi * driver.radius * driver.length,
        h_factor=h_factor,
    )

    class SideWall(RFDriver):
        @property
        def chamber(self) -> Chamber:
            return chamber

    fields = {f.name: getattr(driver, f.name) for f in dataclasses.fields(driver)}
    return SideWall(**fields)


def _coil(driver):
    simgeometry = pytest.importorskip("simgeometry")
    return Winding(
        simgeometry.Rectangle((driver.coil_radius, driver.length / 2), (0.004, 1.0)),
        sign=driver.coil_turns,
    )


def test_the_lorentz_force_pushes_the_electrons_off_the_coil(driver) -> None:
    grid, plasma = column(driver)
    field = InductiveField2D(
        grid=grid,
        frequency=driver.frequency,
        windings=[_coil(driver)],
        electron_density=np.where(plasma, 1e18, 0.0),
        collision_frequency=1e7,
    )
    solution = field.solve(1.0).for_power(25e3)
    force_r, force_z = solution.lorentz_force
    assert np.all(force_r[plasma] <= 0.0) and force_r[plasma].min() < 0.0
    assert np.all(force_r[~plasma] == 0.0)
    assert np.max(np.abs(force_z)) < 1e-6 * np.max(np.abs(force_r))  # z-invariant
    # the force density is the averaged J x B on the plasma current
    b_r, b_z = solution.magnetic_field
    j_x_b = 0.5 * np.real(solution.plasma_current_density * np.conj(b_z))
    assert force_r * solution.electron_density == pytest.approx(
        j_x_b, rel=1e-9, abs=1e-12
    )


def test_the_coupled_column_balances_power_as_the_global_model(driver) -> None:
    """The field and the fluid at 25 kW absorbed, against the global model.

    The global model takes the fluid's edge-to-mean density ratio for its ions' loss
    area and the fluid's shape for the field: then it has the fluid's plasma and the
    coil sees the same resistance. With :class:`RFDriver`'s edge-to-centre ratio of
    Lieberman and Lichtenberg on the mean density the global plasma is denser.
    """
    power = 25e3
    grid, plasma = column(driver)
    volume = np.pi * driver.radius**2 * driver.length
    fluid = hydrogen_fluid(
        driver, grid, region=plasma, heating=np.where(plasma, power / volume, 0.0)
    )
    shape = np.cos(0.4 * np.pi * grid.xx / driver.radius)
    uniform = fluid.steady_state(hydrogen_guess(fluid, 2e18, 5.0, shape))
    assert uniform.converged

    def field(profile):
        settings = profile.settings()
        return InductiveField2D(
            grid=grid, frequency=driver.frequency, windings=[_coil(driver)], **settings
        )

    coupled = DriverCoupling(
        field=field,
        plasma=fluid_plasma(fluid, field, uniform, lorentz=False),
        network_resistance=driver.network_resistance,
    ).at_absorbed_power(power)
    result = coupled.plasma
    state = result.state
    assert result.converged and result.passes <= 4
    assert float(fluid.heating @ fluid.volumes) == pytest.approx(power, rel=1e-10)
    terms = fluid.energy_balance(state)
    assert abs(sum(terms.values())) < 1e-9 * power
    assert coupled.rf_power == pytest.approx(power / coupled.efficiency)

    ne = state.electron_density
    walls = fluid.wall_faces["cell"]
    h_fluid = np.average(
        ne.ravel()[walls], weights=fluid.wall_faces["area"]
    ) / state.mean(ne)
    # the fluid's edge-to-centre ratio of the main ion is Lieberman's h_R
    h2 = state.densities["H2+"]
    h_r = 0.8 / np.sqrt(4 + driver.radius / driver.ion_mean_free_path)
    assert h2.ravel()[walls].mean() / h2[0, 0] == pytest.approx(h_r, rel=0.2)

    same = side_wall_driver(driver, h_fluid)
    reference = DriverCoupling(
        field=field,
        plasma=global_model_plasma(same, np.where(plasma, ne / state.mean(ne), 0.0)),
        network_resistance=driver.network_resistance,
    ).at_absorbed_power(power)
    global_plasma = reference.plasma.plasma
    assert global_plasma.electron_density == pytest.approx(state.mean(ne), rel=0.06)
    assert global_plasma.electron_temperature == pytest.approx(
        state.mean(state.electron_temperature), rel=0.01
    )
    assert reference.coil_current == pytest.approx(coupled.coil_current, rel=0.03)
    assert reference.efficiency == pytest.approx(coupled.efficiency, abs=0.01)

    lieberman = side_wall_driver(driver, h_r).model(power).steady_state()
    assert lieberman.electron_density > 1.3 * state.mean(ne)


def test_a_driver_reaches_a_steady_state(driver) -> None:
    """The r–z driver heated uniformly: ions leave through the side and both ends."""
    grid = driver_cylinder(driver, 5)
    volume = np.pi * driver.radius**2 * driver.length
    fluid = hydrogen_fluid(driver, grid, heating=25e3 / volume)
    shape = np.cos(0.4 * np.pi * grid.xx / driver.radius) * np.sin(
        np.pi * (0.9 * grid.yy / driver.length + 0.05)
    )
    state = fluid.steady_state(hydrogen_guess(fluid, 1e18, 6.0, shape))
    assert state.converged
    terms = fluid.energy_balance(state)
    assert abs(sum(terms.values())) < 1e-9 * 25e3
    assert terms["walls"] < 0 and terms["reactions"] < 0
    made = state.total(state.reaction_rates["ionization of H2"])
    for imbalance in fluid.balance(state).values():
        assert abs(imbalance) < 1e-9 * made
    # it stays where it is when stepped further
    later = fluid.state(fluid.advance(state.unknowns, 1e-4, 1e-4))
    assert later.electron_density == pytest.approx(state.electron_density, rel=1e-6)
    dataset = state.to_dataset()
    assert set(dataset.ion.values) == {"H+", "H2+", "H3+"}
    assert float(dataset.electron_temperature.max()) > 1.0


def test_bad_models_are_refused() -> None:
    grid = slab_grid(0.01, 10)
    with pytest.raises(ValueError, match="electrons"):
        Fluid2D(grid, [FluidSpecies(H2_ION, 1.0)])
    with pytest.raises(ValueError, match="positive ion"):
        Fluid2D(grid, [FluidSpecies(ELECTRON, 1.0)])
    with pytest.raises(ValueError, match="inertia"):
        Fluid2D(
            grid,
            [FluidSpecies(ELECTRON, 1.0), FluidSpecies(H2_ION, 1.0)],
            potential="poisson",
            ions="momentum",
        )
    with pytest.raises(ValueError, match="unknown species"):
        Fluid2D(
            grid, [FluidSpecies(ELECTRON, 1.0), FluidSpecies(H2_ION, 1.0)], [IONIZATION]
        )
    with pytest.raises(ValueError, match="no cell"):
        Fluid2D(
            grid,
            [FluidSpecies(ELECTRON, 1.0), FluidSpecies(H2_ION, 1.0)],
            region=np.zeros(grid.shape, bool),
        )
    with pytest.raises(ValueError, match="mean free path"):
        Fluid2D(
            grid,
            [FluidSpecies(ELECTRON, 1.0), FluidSpecies(H2_ION, 1.0)],
            ion_mean_free_path=0.0,
        )
