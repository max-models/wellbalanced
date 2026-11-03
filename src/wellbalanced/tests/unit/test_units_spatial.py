"""Quantities in the setup of the spatial models: converted once, numbers kept as they are."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from plasmacoll.constants import ATOMIC_MASS, BOLTZMANN, ELEMENTARY_CHARGE

from wellbalanced import ELECTRON, Species
from wellbalanced.fluid1d import FluidSpecies
from wellbalanced.spatial import (
    CellGrid,
    Coil3D,
    Collision,
    Conductor3D,
    DriverCoupling,
    EdgeGrid,
    Fluid2D,
    ImpedanceConductor3D,
    InductiveField2D,
    InductiveField3D,
    Inlet,
    NeutralFluid2D,
    NeutralSpecies,
    PatchGrid,
    PlasmaProfile,
    Winding,
    hard_sphere,
    stretched_edges,
)

KELVIN_IN_EV = BOLTZMANN / ELEMENTARY_CHARGE
H2 = Species("H2", 2.016 * ATOMIC_MASS)
H_ION = Species.atomic("H+", 1.008, 1)


class Box:
    """An axis-aligned rectangle with ``contains_points``."""

    def __init__(self, x0: float, x1: float, y0: float, y1: float) -> None:
        self.bounds = (x0, x1, y0, y1)

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        x0, x1, y0, y1 = self.bounds
        return (points[0] > x0) & (points[0] < x1) & (points[1] > y0) & (points[1] < y1)


def grid() -> CellGrid:
    return CellGrid(np.linspace(0.0, 0.04, 5), np.linspace(0.0, 0.04, 5))


def test_cell_grid_edges():
    edges = np.linspace(0.0, 4.0, 5)
    numbers = CellGrid(edges / 100, edges / 100)
    quantities = CellGrid(edges * u.cm, "[0, 1, 2, 3, 4] cm")
    np.testing.assert_allclose(quantities.x_edges, numbers.x_edges)
    np.testing.assert_allclose(quantities.y_edges, numbers.y_edges)
    with pytest.raises(ValueError, match="x_edges"):
        CellGrid(edges * u.s, edges)


def test_stretched_edges():
    numbers = stretched_edges(0.0, 0.1, 0.04, 0.06, 0.005)
    quantities = stretched_edges(
        0.0 * u.m, "10 cm", 40 * u.mm, "60 mm", 5 * u.mm, ratio=1.1
    )
    np.testing.assert_allclose(quantities, numbers)
    with pytest.raises(ValueError, match="spacing"):
        stretched_edges(0.0, 0.1, 0.04, 0.06, "5 eV")


def test_edge_grid_corners():
    numbers = EdgeGrid((0.0, 0.0, 0.0), (0.1, 0.1, 0.2), (2, 2, 2))
    quantities = EdgeGrid([0, 0, 0] * u.cm, "[10, 10, 20] cm", (2, 2, 2))
    np.testing.assert_allclose(quantities.upper, numbers.upper)
    np.testing.assert_allclose(quantities.lower, numbers.lower)
    with pytest.raises(ValueError, match="upper"):
        EdgeGrid((0.0, 0.0, 0.0), [1, 1, 1] * u.kg, (2, 2, 2))


def test_patch_grid_bounds_and_edges():
    pytest.importorskip("patchwork")
    numbers = PatchGrid.refined(((0.0, 0.01), (0.0, 0.008)), (10, 8), [])
    quantities = PatchGrid.refined("[[0, 10], [0, 8]] mm", (10, 8), [])
    np.testing.assert_allclose(quantities.cell_volumes, numbers.cell_volumes)
    boxes = [[((0.004, 0.006), (0.003, 0.005))]]
    numbers = PatchGrid.refined(((0.0, 0.01), (0.0, 0.008)), (10, 8), boxes)
    quantities = PatchGrid.refined(
        [[0, 1], [0, 0.8]] * u.cm, (10, 8), [[[[4, 6], [3, 5]] * u.mm]]
    )
    np.testing.assert_allclose(quantities.cell_volumes, numbers.cell_volumes)
    edges = np.linspace(0.0, 0.01, 11)
    numbers = PatchGrid.from_edges(edges, edges)
    quantities = PatchGrid.from_edges(edges * 1e3 * u.mm, edges * u.m)
    np.testing.assert_allclose(quantities.cell_volumes, numbers.cell_volumes)
    with pytest.raises(ValueError, match="bounds"):
        PatchGrid.refined(((0.0, 0.01), (0.0, 0.008)) * u.s, (10, 8), [])


def test_inductive_field_2d():
    coil = Winding(Box(0.03, 0.04, 0.0, 0.04))
    numbers = InductiveField2D(
        grid=grid(),
        frequency=1.0e6,
        windings=[coil],
        electron_density=1.0e17,
        collision_frequency=1.0e7,
        electron_temperature=2.0,
    )
    quantities = InductiveField2D(
        np.linspace(0.0, 4.0, 5) * u.cm,
        "[0, 1, 2, 3, 4] cm",
        frequency="1 MHz",
        windings=[coil],
        electron_density=1.0e11 * u.cm**-3,
        collision_frequency="1e7 / s",
        electron_temperature=2.0 * u.eV,
    )
    assert quantities.frequency == pytest.approx(1.0e6)
    np.testing.assert_allclose(quantities.electron_density, numbers.electron_density)
    np.testing.assert_allclose(quantities.viscosity, numbers.viscosity)
    in_kelvin = InductiveField2D(
        grid=grid(),
        frequency=1.0e6,
        windings=[coil],
        electron_density=1.0e17,
        collision_frequency=1.0e7,
        electron_temperature=(2.0 / KELVIN_IN_EV) * u.K,
    )
    np.testing.assert_allclose(in_kelvin.viscosity, numbers.viscosity)
    viscous = InductiveField2D(
        grid=grid(),
        frequency=1.0e6,
        windings=[coil],
        electron_density=1.0e17,
        electron_viscosity="1e-6 Pa s",
    )
    assert viscous.viscosity.max() == pytest.approx(1.0e-6)
    stepper = InductiveField2D(
        grid=grid(), frequency=1e6, windings=[coil]
    ).time_stepper("10 ns")
    assert stepper.dt == pytest.approx(1.0e-8)
    with pytest.raises(ValueError, match="frequency"):
        InductiveField2D(grid=grid(), frequency="1 m", windings=[coil])
    with pytest.raises(ValueError, match="electron_density"):
        InductiveField2D(
            grid=grid(), frequency=1e6, windings=[coil], electron_density="1 eV"
        )


def test_inductive_3d_settings():
    coil = Coil3D(None, turns_per_length="1 / cm")
    assert coil.turns_per_length == pytest.approx(100.0)
    assert Coil3D(None, turns_per_length=100.0).turns_per_length == 100.0
    assert Conductor3D((), "58 MS/m").conductivity == pytest.approx(5.8e7)
    assert ImpedanceConductor3D((), 5.8e7 * u.S / u.m).conductivity == 5.8e7
    with pytest.raises(ValueError, match="conductivity"):
        Conductor3D((), "1 Ohm")
    edge_grid = EdgeGrid((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (2, 2, 2))
    field = InductiveField3D(edge_grid, frequency="2 MHz")
    assert field.frequency == pytest.approx(2.0e6)
    with pytest.raises(ValueError, match="frequency"):
        InductiveField3D(edge_grid, frequency="2 m")


def test_plasma_profile_and_coupling():
    profile = PlasmaProfile("1e17 m^-3", [1e7, 2e7] * u.Hz, "23209 K")
    assert profile.electron_density == pytest.approx(1.0e17)
    np.testing.assert_allclose(profile.collision_frequency, [1e7, 2e7])
    assert profile.electron_temperature == pytest.approx(2.0, rel=1e-4)
    shape = np.ones((2, 2))
    kept = PlasmaProfile(shape, 1e7)
    assert kept.electron_density is shape and kept.electron_temperature is None
    with pytest.raises(ValueError, match="collision_frequency"):
        PlasmaProfile(1e17, "1 m")
    coupling = DriverCoupling(None, None, "600 mOhm", depth=10 * u.cm)
    assert coupling.network_resistance == pytest.approx(0.6)
    assert coupling.depth == pytest.approx(0.1)
    assert DriverCoupling(None, None, 0.6).depth is None
    with pytest.raises(ValueError, match="network_resistance"):
        DriverCoupling(None, None, "0.6 W")
    with pytest.raises(ValueError, match="power"):
        coupling.at_absorbed_power("1 m")
    with pytest.raises(ValueError, match="rf_power"):
        coupling.at_rf_power("1 s")


def test_neutral_settings():
    collision = Collision("H2", "H2", "1e-15 m^3/s")
    assert collision.rate == pytest.approx(1.0e-15)
    assert Collision("H2", "H2", 2e-15).rate == 2e-15
    assert callable(Collision("H2", "H2", lambda t: t).rate)
    with pytest.raises(ValueError, match="rate"):
        Collision("H2", "H2", "1 m^2")
    rate = hard_sphere("1e-19 m^2", H2.mass * u.kg, H2.mass)
    reference = hard_sphere(1e-19, H2.mass, H2.mass)
    assert rate(0.1) == pytest.approx(reference(0.1))
    with pytest.raises(ValueError, match="cross_section"):
        hard_sphere("1 m", H2.mass, H2.mass)
    inlet = Inlet(None, "H2", "1e19 / s", temperature=300 * u.K)
    assert inlet.rate == pytest.approx(1.0e19)
    assert inlet.temperature == pytest.approx(300 * KELVIN_IN_EV)
    assert Inlet(None, "H2", "1e19 / (cm s)").rate == pytest.approx(1.0e21)
    assert Inlet(None, "H2", 5.0, temperature=0.03).temperature == 0.03
    with pytest.raises(ValueError, match="rate"):
        Inlet(None, "H2", "1 m")
    with pytest.raises(ValueError, match="temperature"):
        Inlet(None, "H2", 1.0, temperature="1 m")


def test_neutral_fluid_2d():
    species = [NeutralSpecies(H2, 3.0)]
    numbers = NeutralFluid2D(
        grid(), species, plasma_species=[ELECTRON], wall_temperature=0.03
    )
    quantities = NeutralFluid2D(
        grid(),
        species,
        plasma_species=[ELECTRON, H_ION],
        wall_temperature=(0.03 / KELVIN_IN_EV) * u.K,
        fixed={"Ar": "1e20 m^-3"},
    )
    assert quantities.wall_temperature == pytest.approx(numbers.wall_temperature)
    np.testing.assert_allclose(quantities._fixed["Ar"], 1.0e20)
    quantities.set_plasma(
        {"e": "1e17 m^-3", "H+": 1e11 * u.cm**-3},
        "3 eV",
        ion_velocities={"H+": ("1 km/s", 0.0)},
        ion_temperature="0.1 eV",
        ion_wall_flux={
            "H+": np.ones(quantities.wall_faces["cell"].size) / u.m**2 / u.s
        },
    )
    np.testing.assert_allclose(quantities._plasma["H+"], 1.0e17)
    np.testing.assert_allclose(quantities._electron_temperature, 3.0)
    np.testing.assert_allclose(quantities._ion_velocities["H+"][0], 1.0e3)
    with pytest.raises(ValueError, match="wall_temperature"):
        NeutralFluid2D(grid(), species, wall_temperature="1 m")
    with pytest.raises(ValueError, match="electron_temperature"):
        quantities.set_plasma({"e": 1e17}, "1 m")


def test_fluid_2d():
    species = [FluidSpecies(ELECTRON, 1.0), FluidSpecies(H_ION, 500.0)]
    numbers = Fluid2D(
        grid(),
        species,
        fixed={"H2": 1e20},
        gas_temperature=0.03,
        electron_temperature=2.0,
        heating=1e4,
        electron_force=(1e-20, 0.0),
        sources={"H+": 1e20},
        ion_mean_free_path=0.01,
        wall_potential=5.0,
    )
    quantities = Fluid2D(
        grid(),
        species,
        fixed={"H2": 1e14 * u.cm**-3},
        gas_temperature=(0.03 / KELVIN_IN_EV) * u.K,
        electron_temperature="2 eV",
        heating="10 kW / m^3",
        electron_force=(1e-20 * u.N, "0 N"),
        sources={"H+": "1e20 m^-3 s^-1"},
        ion_mean_free_path={"H+": "1 cm"},
        wall_potential=5000 * u.mV,
    )
    assert quantities.gas_temperature == pytest.approx(numbers.gas_temperature)
    assert quantities.electron_temperature == pytest.approx(2.0)
    assert quantities.wall_potential == pytest.approx(5.0)
    assert quantities.ion_mean_free_path == pytest.approx(numbers.ion_mean_free_path)
    np.testing.assert_allclose(quantities._fixed["H2"], numbers._fixed["H2"])
    np.testing.assert_allclose(quantities._sources["H+"], numbers._sources["H+"])
    np.testing.assert_allclose(quantities.heating, numbers.heating)
    np.testing.assert_allclose(quantities.electron_force[0], numbers.electron_force[0])
    quantities.set_gas({"H2": "2e20 m^-3"}, "300 K")
    np.testing.assert_allclose(quantities._fixed["H2"], 2e20)
    assert quantities.gas_temperature == pytest.approx(300 * KELVIN_IN_EV)
    with pytest.raises(ValueError, match="heating"):
        Fluid2D(grid(), species, fixed={"H2": 1e20}, heating="1 W")
    with pytest.raises(ValueError, match="gas_temperature"):
        Fluid2D(grid(), species, fixed={"H2": 1e20}, gas_temperature="1 m")
