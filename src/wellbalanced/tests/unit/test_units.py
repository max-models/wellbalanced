"""Quantities in the setup of the models: converted once, numbers kept as they are."""

from __future__ import annotations

import warnings

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from plasmacoll import ChargeStateTransitions
from plasmacoll.constants import ATOMIC_MASS, BOLTZMANN, ELEMENTARY_CHARGE

from wellbalanced import (
    ELECTRON,
    Boundary,
    Chamber,
    Coil,
    Cusp,
    DriftFluid2D,
    Fluid1D,
    FluidSpecies,
    GasDynamics,
    GasTemperature,
    GlobalModel,
    Magnetron,
    PotentialBoundary,
    Reaction,
    Species,
    TargetMaterial,
    VolumeSource,
)
from wellbalanced import gasdynamics as gd
from wellbalanced.dispersion import ColdIons, KineticIons, Plasma, scan_profile
from wellbalanced.driftfluid2d import exponential_field
from wellbalanced.driver import illustrative_hydrogen_driver
from wellbalanced.magnetron import TITANIUM, pulse
from wellbalanced.neutralizer import (
    BeamCrossSections,
    BeamData,
    FastBeam,
    GasProfile,
    illustrative_deuterium_neutralizer,
)

KELVIN_IN_EV = BOLTZMANN / ELEMENTARY_CHARGE
ARGON = Species.atomic("Ar", 39.948)
ARGON_ION = Species.atomic("Ar+", 39.948, 1)
IONIZATION = Reaction(("e", "Ar"), ("e", "e", "Ar+"), 1.0e-14)


@pytest.fixture
def quiet():
    """Silence the placeholder warnings of the illustrative models."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


# -- species and reactions -----------------------------------------------------------


def test_species_mass_takes_quantities_strings_and_kg():
    assert Species("Ar", 39.948 * u.u).mass == pytest.approx(39.948 * ATOMIC_MASS)
    assert Species("Ar", "39.948 u").mass == pytest.approx(39.948 * ATOMIC_MASS)
    assert Species("Ar", 6.6e-26).mass == 6.6e-26
    assert Species.atomic("Ar", "39.948 u").mass == pytest.approx(ARGON.mass)
    with pytest.raises(ValueError, match="mass"):
        Species("Ar", "40 m")


def test_reaction_energies_masses_and_constant_rate():
    reaction = Reaction(
        ("e", "Ar"),
        ("e", "e", "Ar+"),
        "1e-8 cm3 / s",
        electron_energy_loss=15.76 * u.eV,
        gas_heating="1 eV",
    )
    assert reaction.rate == pytest.approx(1.0e-14)
    assert reaction.electron_energy_loss == pytest.approx(15.76)
    assert reaction.gas_heating == 1.0
    assert Reaction(("Ar",), (), "2 kHz").rate == pytest.approx(2.0e3)
    assert Reaction(("e", "Ar"), ("e", "Ar"), 1e-13, elastic_mass="40 u").elastic_mass
    assert Reaction(("e", "Ar"), ("e", "Ar"), 1e-14).rate == 1e-14
    with pytest.raises(ValueError, match="rate"):
        Reaction(("e", "Ar"), ("e", "Ar"), "1 1/s")
    with pytest.raises(ValueError, match="electron_energy_loss"):
        Reaction(("e", "Ar"), ("e", "Ar"), 1e-14, electron_energy_loss="1 m")


def test_reaction_from_a_cross_section_takes_masses():
    from plasmacoll import CrossSection

    sigma = CrossSection.constant(1e-20, threshold=15.76)
    a = Reaction.from_cross_section(("e", "Ar"), ("e", "Ar"), sigma, ELECTRON.mass)
    b = Reaction.from_cross_section(
        ("e", "Ar"), ("e", "Ar"), sigma, ELECTRON.mass * u.kg
    )
    assert a.coefficient(3.0, 0.03) == pytest.approx(b.coefficient(3.0, 0.03))
    with pytest.raises(ValueError, match="reduced_mass"):
        Reaction.from_cross_section(("e", "Ar"), ("e", "Ar"), sigma, "1 eV")


# -- the global model ----------------------------------------------------------------


def test_chamber_cusp_and_gas_temperature():
    chamber = Chamber("1 L", 600 * u.cm**2, "0.01 m2", cusp=Cusp("50 mT", "2 m"))
    assert chamber.volume == pytest.approx(1e-3)
    assert chamber.wall_area == pytest.approx(0.06)
    assert chamber.ion_loss_area == pytest.approx(0.01)
    assert chamber.cusp.field == pytest.approx(0.05)
    gas = GasTemperature("300 K", "0.018 W / (m K)", 5 * u.cm)
    assert gas.wall_temperature == pytest.approx(300 * KELVIN_IN_EV)
    assert gas.diffusion_length == pytest.approx(0.05)
    assert GasTemperature(0.03, 0.02, 0.05).wall_temperature == 0.03
    with pytest.raises(ValueError, match="volume"):
        Chamber("1 m2", 0.06)


def test_volume_source_rate_unit_follows_per():
    assert VolumeSource({"Ar+": 1}, "1e14 cm^-3 s^-1").rate == pytest.approx(1e20)
    assert VolumeSource({"Ar+": 1}, "1 kHz", per=("Ar",)).rate == pytest.approx(1e3)
    source = VolumeSource({"Ar+": 1}, 1e-16 * u.m**3 / u.s, per=("Ar", "e"))
    assert source.rate == pytest.approx(1e-16)
    assert VolumeSource({"e": 1}, 1.0, electron_energy="2 eV").electron_energy == 2.0
    with pytest.raises(ValueError, match="rate"):
        VolumeSource({"Ar+": 1}, "1 1/s")


def test_global_model_settings():
    model = GlobalModel(
        species=[ELECTRON, ARGON, ARGON_ION],
        reactions=[IONIZATION],
        chamber=Chamber(1e-3, 0.06),
        power="0.1 kW",
        fixed={"Ar": "1e14 cm^-3"},
        feed={"Ar": 1e18 / u.s},
        pumping_speed="1 L / s",
        gas_temperature="300 K",
    )
    assert model.power == pytest.approx(100.0)
    assert model.fixed["Ar"] == pytest.approx(1e20)
    assert model.feed["Ar"] == pytest.approx(1e18)
    assert model.pumping_speed == pytest.approx(1e-3)
    assert model.gas_temperature == pytest.approx(300 * KELVIN_IN_EV)
    plain = GlobalModel([ELECTRON, ARGON, ARGON_ION], [IONIZATION], Chamber(1, 1))
    assert plain.gas_temperature == 0.025
    with pytest.raises(ValueError, match="power"):
        GlobalModel([ELECTRON, ARGON, ARGON_ION], [IONIZATION], Chamber(1, 1), "1 V")
    with pytest.raises(ValueError, match="times"):
        plain.evolve("[0, 1, 2] m")


# -- the magnetron -------------------------------------------------------------------


def test_magnetron_settings(quiet):
    magnetron = Magnetron(
        pressure="7.5 mTorr",
        gas_temperature=0.025 * u.eV,
        region_height="2 cm",
        target_area=10 * u.cm**2,
        refill_time="100 us",
        seed="1e13 cm^-3 s^-1",
    )
    assert magnetron.pressure == pytest.approx(1.0, rel=1e-3)
    assert magnetron.gas_temperature == pytest.approx(0.025 / KELVIN_IN_EV)
    assert magnetron.region_height == pytest.approx(0.02)
    assert magnetron.target_area == pytest.approx(1e-3)
    assert magnetron.refill_time == pytest.approx(1e-4)
    assert magnetron.seed == pytest.approx(1e19)
    assert Magnetron().gas_temperature == 300.0
    with pytest.raises(ValueError, match="pressure"):
        Magnetron(pressure="1 K")


def test_magnetron_drive(quiet):
    magnetron = Magnetron()
    model = magnetron.model("20 A", 500 * u.V)
    assert model.power(0.0) == pytest.approx(magnetron.heating_power(20.0, 500.0))
    assert magnetron.guess("1e11 cm^-3")["Ar+"] == pytest.approx(1e17)
    with pytest.raises(ValueError, match="current"):
        magnetron.calibrate_heating("1 V", 500.0)
    with pytest.raises(ValueError, match="on_time"):
        magnetron.pulse("20 A", "500 V", "200 us", "100 us")
    with pytest.raises(ValueError, match="voltage"):
        magnetron.current_scan(["5 A", "10 A"], "1 A")
    waveform = pulse(1.0, "50 us", 0.1 * u.ms)
    assert waveform(40e-6) == 1.0
    assert waveform(60e-6) == 0.0


def test_target_material():
    material = TargetMaterial("Ti", "47.867 u", 6.828 * u.eV, 2, 0.4, 0.3, 0.1, 0.05,
                              "1e-9 cm3 / s")  # fmt: skip
    assert material.mass_u == pytest.approx(TITANIUM.mass_u)
    assert material.charge_exchange == pytest.approx(1e-15)
    with pytest.raises(ValueError, match="ionization_energy"):
        TargetMaterial("Ti", 47.867, "1 m", 2, 0.4, 0.3, 0.1, 0.05, 1e-15)


# -- the 1D fluid --------------------------------------------------------------------


def test_fluid_species_and_boundaries():
    fluid = FluidSpecies(ARGON, diffusion="1 cm2 / s")
    assert fluid.diffusion == pytest.approx(1e-4)
    ions = FluidSpecies(ARGON_ION, mobility=100 * u.cm**2 / (u.V * u.s))
    assert ions.mobility == pytest.approx(1e-2)
    assert Boundary.density("1e10 cm^-3", "3 eV").value == pytest.approx(1e16)
    assert Boundary.density(1e16, 300 * u.K).temperature == pytest.approx(
        300 * KELVIN_IN_EV
    )
    assert Boundary.flux("1e12 cm^-2 s^-1").value == pytest.approx(1e16)
    assert Boundary.outflow("1 km / s").value == pytest.approx(1e3)
    assert Boundary.wall().value == 0.0
    assert PotentialBoundary.potential("10 V").value == 10.0
    assert PotentialBoundary.field("1 kV / m").value == pytest.approx(1e3)
    with pytest.raises(ValueError, match="value"):
        Boundary.flux("1 m / s")
    with pytest.raises(ValueError, match="mobility"):
        FluidSpecies(ARGON_ION, mobility="1 m")


def test_fluid1d_settings():
    model = Fluid1D(
        species=[FluidSpecies(ELECTRON, 1e3), FluidSpecies(ARGON_ION, 10.0)],
        reactions=[IONIZATION],
        length="5 cm",
        cells=10,
        fixed={"Ar": np.full(10, 1e21) * u.m**-3},
        gas_temperature="300 K",
        heating="2 W / cm3",
        sources={"Ar+": "1e14 cm^-3 s^-1"},
    )
    assert model.length == pytest.approx(0.05)
    assert model.gas_temperature == pytest.approx(300 * KELVIN_IN_EV)
    assert model.heating == pytest.approx(2e6)
    assert model.sources["Ar+"] == pytest.approx(1e20)
    assert isinstance(model.fixed["Ar"], np.ndarray)
    heating = np.linspace(1.0, 2.0, 10)
    plain = Fluid1D(
        model.species, [IONIZATION], 0.05, 10, {"Ar": 1e21}, heating=heating
    )
    assert plain.heating is heating
    packed = model.pack({"e": "1e9 cm^-3", "Ar+": 1e15}, "3 eV", 0.0 * u.V)
    assert np.allclose(packed, model.pack({"e": 1e15, "Ar+": 1e15}, 3.0, 0.0))
    with pytest.raises(ValueError, match="dt"):
        model.advance(packed, "1 ns", "1 m")
    with pytest.raises(ValueError, match="length"):
        Fluid1D(model.species, [IONIZATION], "5 s", 10, {"Ar": 1e21})


# -- the RF driver -------------------------------------------------------------------


def test_coil_and_driver(quiet):
    coil = Coil(6, "13.85 cm", 0.17 * u.m)
    assert coil.radius == pytest.approx(0.1385)
    driver = illustrative_hydrogen_driver(
        frequency="1 MHz", radius="11.85 cm", fixed={"H2": "3e13 cm^-3", "H": 1.5e19}
    )
    assert driver.frequency == pytest.approx(1e6)
    assert driver.radius == pytest.approx(0.1185)
    assert driver.fixed["H2"] == pytest.approx(3e19)
    assert driver.model("25 kW").power == pytest.approx(25e3)
    with pytest.raises(ValueError, match="power"):
        driver.solve("1 m")
    with pytest.raises(ValueError, match="power"):
        driver.couple(None, "1 m")
    with pytest.raises(ValueError, match="power"):
        driver.scan("power", ["1 m"])
    with pytest.raises(ValueError, match="frequency"):
        illustrative_hydrogen_driver(frequency="1 m")


# -- the neutralizer -----------------------------------------------------------------


def test_beam_and_cross_sections():
    beam = FastBeam("1 MeV", "10 A", "2.014 u")
    assert beam.energy == pytest.approx(1e6)
    assert beam.mass == pytest.approx(2.014 * ATOMIC_MASS)
    transitions = ChargeStateTransitions({"gas": {("-", "0"): 1e-20}})
    data = BeamData(transitions, {"-": "1e-16 cm2"}, {"-": 300 * u.K})
    assert data.gas_ionization["-"] == pytest.approx(1e-20)
    assert data.secondary_energy["-"] == pytest.approx(300 * KELVIN_IN_EV)
    table = BeamCrossSections(
        {"gas": {("-", "0"): "1e-16 cm2", ("0", "+"): lambda e: 1e-21 + 0 * e}},
        {"-": 2e-20 * u.m**2},
    )
    at = table.at("1 MeV")
    assert at.transitions.cross_sections["gas"][("-", "0")] == pytest.approx(1e-20)
    assert at.gas_ionization["-"] == pytest.approx(2e-20)
    with pytest.raises(ValueError, match="energy"):
        FastBeam("1 A", 10.0, 3e-27)
    with pytest.raises(ValueError, match="transitions"):
        BeamCrossSections({"gas": {("-", "0"): "1 m"}})


def test_gas_profile():
    mass = 4.028 * ATOMIC_MASS
    plain = GasProfile.knudsen(3.0, 0.1, 1.7, mass, 2e21, 0.03)
    quantity = GasProfile.knudsen(
        "3 m", "10 cm", 1.7 * u.m, "4.028 u", "2e21 1/s", 0.03 * u.eV, feed="1.5 m"
    )
    assert np.allclose(quantity.density, plain.density)
    profile = GasProfile([0, 1, 2] * u.m, "[1, 2, 1] 1e14 / cm3", 1e20, "300 K")
    assert profile.density[1] == pytest.approx(2e20)
    assert profile.temperature == pytest.approx(300 * KELVIN_IN_EV)
    with pytest.raises(ValueError, match="temperature"):
        GasProfile.knudsen(3.0, 0.1, 1.7, mass, 2e21, "1 m")


def test_neutralizer(quiet):
    model = illustrative_deuterium_neutralizer(
        length="300 cm", gas_inflow="2e21 1/s", gas_temperature="350 K"
    )
    assert model.length == pytest.approx(3.0)
    assert model.gas_temperature == pytest.approx(350 * KELVIN_IN_EV)
    assert model.beam_data("1 MeV") is model.data
    sources = model.beam_sources("1e15 cm^-2 s^-1", [1.0, 0.0, 0.0], "1 MeV")
    reference = model.beam_sources(1e19, [1.0, 0.0, 0.0], 1e6)
    assert [s.rate for s in sources] == pytest.approx([s.rate for s in reference])
    with pytest.raises(ValueError, match="width"):
        illustrative_deuterium_neutralizer(width="1 s")


# -- gas dynamics --------------------------------------------------------------------


def test_gas_dynamics_closures():
    wall = gd.Boundary.wall(
        "300 K",
        gas_return_flux="1e16 cm^-2 s^-1",
        metal_flux=np.ones(3) * 1e20 / (u.m**2 * u.s),
        metal_energy="5 eV",
    )
    assert wall.temperature == 300.0
    assert wall.gas_return_flux == pytest.approx(1e20)
    assert isinstance(wall.metal_flux, np.ndarray)
    inflow = gd.Boundary.inflow("7.5 mTorr", 0.025 * u.eV)
    assert inflow.pressure == pytest.approx(1.0, rel=1e-3)
    assert inflow.temperature == pytest.approx(0.025 / KELVIN_IN_EV)
    source = gd.VolumeSource("metal", "1e14 cm^-3 s^-1", "0.5 eV", ["1 km/s", "0 m/s"])
    assert source.rate == pytest.approx(1e20)
    assert source.velocity == pytest.approx((1e3, 0.0))
    exchange = gd.IonExchange("gas", "1 kHz", ion_velocity=[1e3, 0.0] * u.m / u.s)
    assert exchange.frequency == pytest.approx(1e3)
    assert exchange.ion_velocity == pytest.approx((1e3, 0.0))
    velocity = (np.ones(3), 0.0)
    assert gd.IonExchange("gas", 1.0, velocity).ion_velocity[0] is velocity[0]
    relaxation = gd.ThermalRelaxation("1e4 1/s", "0.03 eV")
    assert relaxation.temperature == pytest.approx(0.03 / KELVIN_IN_EV)
    with pytest.raises(ValueError, match="metal_energy"):
        gd.Boundary.wall(metal_energy="5 m")


def test_binary_diffusion_takes_arrays_and_quantities():
    mass = gd.ARGON_MASS
    plain = gd.binary_diffusion(mass, mass, 3.6e-10, np.array([300.0, 600.0]), 1.0)
    quantity = gd.binary_diffusion(
        "39.948 u", mass * u.kg, "0.36 nm", [300, 600] * u.K, "7.5 mTorr"
    )
    assert quantity == pytest.approx(plain, rel=1e-3)


def test_gas_dynamics_setup():
    model = GasDynamics(
        "5 cm", 8, width=1 * u.cm, metal_mass="47.867 u", diffusion="1 cm2/s"
    )
    assert model.length == pytest.approx(0.05)
    assert model.width == pytest.approx(0.01)
    assert model.diffusion == pytest.approx(1e-4)
    model.set_uniform("1 Pa", "300 K")
    uniform = model.state()
    model.set_state(uniform.gas_density * u.kg / u.m**3, pressure="1 Pa")
    assert np.allclose(model.state().pressure, uniform.pressure)
    state = model.run("1 ns")
    assert state.time == pytest.approx(1e-9)
    with pytest.raises(ValueError, match="thermal_conductivity"):
        GasDynamics(0.05, 8, thermal_conductivity="1 W")


# -- the dispersion relations and the 2D drift fluid ---------------------------------


def test_dispersion_settings():
    plasma = Plasma("1e11 cm^-3", "-2.2 kV/m", "6 mT", "11.4 eV", "-2.2 mm")
    assert plasma.density == pytest.approx(1e17)
    assert plasma.e_field == pytest.approx(-2200.0)
    assert plasma.b_field == pytest.approx(6e-3)
    assert plasma.density_length == pytest.approx(-2.2e-3)
    assert plasma.field_length == np.inf
    assert ColdIons("39.948 u", ["1 km/s", "0 m/s"]).flow == pytest.approx((1e3, 0.0))
    assert KineticIons(temperature="300 K").temperature == pytest.approx(
        300 * KELVIN_IN_EV
    )
    with pytest.raises(ValueError, match="b_field"):
        Plasma(1e17, -2200.0, "6 m", 11.4, -2.2e-3)
    x = np.linspace(0.0, 0.01, 3)
    profile = xr.Dataset(
        {
            "density": ("x", np.full(3, 1e15) * np.exp(-x / 0.003)),
            "e_field": ("x", np.full(3, -2200.0)),
            "b_field": ("x", np.full(3, 6e-3)),
            "electron_temperature": ("x", np.full(3, 11.4)),
        },
        coords={"x": x},
    )
    with pytest.raises(ValueError, match="ky"):
        scan_profile(profile, "1 m")


def test_drift_fluid_settings():
    model = DriftFluid2D(
        "2.5 cm",
        5 * u.cm,
        4,
        4,
        voltage="260 V",
        magnetic_field="20 mT",
        gas_density="1.45e14 cm^-3",
        electron_temperature="5 eV",
        momentum_rate="1e-7 cm3/s",
        ion_mass_u="39.948 u",
        ion_temperature="300 K",
    )
    assert model.length_x == pytest.approx(0.025)
    assert model.magnetic_field == pytest.approx(0.02)
    assert model.gas_density == pytest.approx(1.45e20)
    assert model.momentum_rate == pytest.approx(1e-13)
    assert model.ion_temperature == pytest.approx(300 * KELVIN_IN_EV)
    state = model.initial_state("1e9 cm^-3", "3 eV")
    assert state.electron_density[0, 0] == pytest.approx(1e15)
    assert state.electron_temperature[0, 0] == 3.0
    with pytest.raises(ValueError, match="probe_x"):
        model.run(state, "1 ns", probe_x="1 s")
    field = exponential_field("120 mT", 0.3 * 2.5 * u.cm)
    assert field(0.0) == pytest.approx(0.12)
    with pytest.raises(ValueError, match="voltage"):
        DriftFluid2D(0.025, 0.05, 4, 4, voltage="1 A")
