"""The global model against closed forms; the cross sections and rates are synthetic."""

from __future__ import annotations

import math

import numpy as np
import pytest
from plasmacoll import CrossSection
from plasmacoll.constants import BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy.optimize import brentq

from wellbalanced import (
    ELECTRON,
    Chamber,
    Cusp,
    GasTemperature,
    GlobalModel,
    Reaction,
    Species,
    VolumeSource,
    WallReaction,
)

E = ELEMENTARY_CHARGE
AR = Species.atomic("Ar", 39.948)
AR_ION = Species.atomic("Ar+", 39.948, 1)
GAS = 1.0e20  # m^-3
CHAMBER = Chamber(volume=1.0e-3, wall_area=0.06, h_factor=0.5)
IONIZATION_ENERGY = 15.76


def ionization() -> Reaction:
    """A step cross section: K(T) = sigma <g> (1 + E0 / T) exp(-E0 / T)."""
    step = CrossSection.constant(2.0e-20, threshold=IONIZATION_ENERGY)
    return Reaction.from_cross_section(
        ("e", "Ar"),
        ("e", "e", "Ar+"),
        step,
        ELECTRON_MASS,
        electron_energy_loss=IONIZATION_ENERGY,
    )


def argon(power: float = 100.0, **settings) -> GlobalModel:
    defaults = {
        "species": [ELECTRON, AR, AR_ION],
        "reactions": [ionization()],
        "chamber": CHAMBER,
        "fixed": {"Ar": GAS},
    }
    return GlobalModel(power=power, **{**defaults, **settings})


def bohm(te: float, mass: float) -> float:
    return math.sqrt(E * te / mass)


def test_the_particle_balance_sets_te_and_the_power_balance_sets_ne() -> None:
    state = argon().steady_state()
    assert state.converged
    reaction = ionization()
    area = CHAMBER.h_factor * CHAMBER.wall_area

    def particle_balance(te: float) -> float:
        rate = reaction.coefficient(te, 0.025) * GAS
        return rate - bohm(te, AR_ION.mass) * area / CHAMBER.volume

    te = brentq(particle_balance, 0.5, 50.0, xtol=1e-12)
    assert state.electron_temperature == pytest.approx(te, rel=1e-6)

    floating = te * math.log(math.sqrt(AR_ION.mass / (2 * math.pi * ELECTRON_MASS)))
    energy = IONIZATION_ENERGY + 2.0 * te + 0.5 * te + floating
    ne = 100.0 / (E * bohm(te, AR_ION.mass) * area * energy)
    assert state.electron_density == pytest.approx(ne, rel=1e-6)
    assert state.densities["Ar+"] == pytest.approx(ne, rel=1e-12)


def test_every_balance_closes_at_steady_state() -> None:
    elastic = Reaction(
        ("e", "Ar"), ("e", "Ar"), 1.0e-13, elastic_mass=AR.mass, name="elastic"
    )
    state = argon(reactions=[ionization(), elastic]).steady_state()
    assert sum(state.power.values()) == pytest.approx(0.0, abs=1e-8 * 100.0)
    assert state.power["elastic"] < 0.0
    scale = state.reaction_rates[ionization().name]
    for name in ("e", "Ar+"):
        assert abs(state.particle_balance[name]) < 1e-8 * scale
    # floating walls: as many electrons as ions reach them
    assert state.wall_losses["e"] == pytest.approx(state.wall_losses["Ar+"])


def test_te_does_not_depend_on_the_power_and_ne_is_proportional_to_it() -> None:
    states = argon().scan("power", [50.0, 100.0, 400.0])
    te = [s.electron_temperature for s in states]
    ne = np.array([s.electron_density for s in states])
    assert te == pytest.approx([te[0]] * 3, rel=1e-6)
    assert ne / ne[0] == pytest.approx([1.0, 2.0, 8.0], rel=1e-6)


def test_negative_ions_held_by_the_sheath_reach_attachment_detachment_balance() -> None:
    negative = Species.atomic("Ar-", 39.948, -1)
    k_att, k_det = 2.0e-17, 5.0e-14
    model = argon(
        species=[ELECTRON, AR, AR_ION, negative],
        reactions=[
            ionization(),
            Reaction(("e", "Ar"), ("Ar-",), k_att),
            Reaction(("e", "Ar-"), ("e", "e", "Ar"), k_det, electron_energy_loss=1.0),
        ],
    )
    state = model.steady_state()
    assert state.converged
    # n_e K_att n_g = n_e K_det n_-, whatever n_e is
    assert state.densities["Ar-"] / GAS == pytest.approx(k_att / k_det, rel=1e-6)
    assert state.wall_losses["Ar-"] == 0.0
    assert state.electronegativity(["Ar-"]) > 0.0
    # quasi-neutrality
    assert state.densities["Ar+"] == pytest.approx(
        state.electron_density + state.densities["Ar-"], rel=1e-12
    )


def test_atoms_from_dissociation_recombine_at_the_walls() -> None:
    h2 = Species.atomic("H2", 2.016)
    h = Species.atomic("H", 1.008)
    h2_ion = Species.atomic("H2+", 2.016, 1)
    k_dis, gamma, tg = 1.0e-14, 0.05, 0.05
    model = GlobalModel(
        species=[ELECTRON, h2, h, h2_ion],
        reactions=[
            Reaction.from_cross_section(
                ("e", "H2"),
                ("e", "e", "H2+"),
                CrossSection.constant(1e-20, threshold=15.4),
                ELECTRON_MASS,
                electron_energy_loss=15.4,
            ),
            Reaction(("e", "H2"), ("e", "H", "H"), k_dis, electron_energy_loss=8.8),
        ],
        chamber=CHAMBER,
        power=50.0,
        fixed={"H2": GAS},
        wall_reactions=[WallReaction("H", gamma, {"H2": 0.5})],
        gas_temperature=tg,
    )
    state = model.steady_state()
    assert state.converged
    mean_speed = math.sqrt(8 * E * tg / (math.pi * h.mass))
    wall_rate = gamma * 0.25 * mean_speed * CHAMBER.wall_area / CHAMBER.volume
    expected = 2.0 * k_dis * state.electron_density * GAS / wall_rate
    assert state.densities["H"] == pytest.approx(expected, rel=1e-6)


def test_cusps_confine_ions_through_a_hybrid_leak_width() -> None:
    cusp = Cusp(field=0.1, line_length=2.0)
    te, ti = 4.0, 0.05
    rho_e = math.sqrt(ELECTRON_MASS * te * E) / (E * 0.1)
    rho_i = math.sqrt(AR_ION.mass * ti * E) / (E * 0.1)
    assert cusp.leak_width(te, ti, AR_ION.mass) == pytest.approx(
        4 * math.sqrt(rho_e * rho_i)
    )
    confined = Chamber(1.0e-3, 0.06, ion_loss_area=0.01, h_factor=0.5, cusp=cusp)
    assert confined.ion_area(te, ti, AR_ION.mass) == pytest.approx(
        0.005 + 2.0 * cusp.leak_width(te, ti, AR_ION.mass)
    )
    # fewer losses: a lower Te sustains the plasma, and more density for the power
    open_state = argon().steady_state()
    cusp_state = argon(chamber=confined).steady_state()
    assert cusp_state.electron_temperature < open_state.electron_temperature
    assert cusp_state.electron_density > open_state.electron_density


def test_the_gas_temperature_balances_heating_against_conduction() -> None:
    gas = GasTemperature(
        wall_temperature=0.025, conductivity=0.02, diffusion_length=0.02
    )
    heating = Reaction(
        ("e", "Ar"), ("e", "Ar"), 1e-14, electron_energy_loss=0.5, gas_heating=0.5
    )
    state = argon(reactions=[ionization(), heating], gas_temperature=gas).steady_state()
    assert state.converged
    rate = state.reaction_rates[heating.name]
    rise = rate * 0.5 * E * 0.02**2 / 0.02 * BOLTZMANN / E  # eV
    assert state.gas_temperature - 0.025 == pytest.approx(rise, rel=1e-6)
    assert state.power["gas heating"] == pytest.approx(
        -state.power["gas conduction"], rel=1e-6
    )


def test_a_beam_source_of_electrons_drives_the_plasma() -> None:
    """Stripped electrons: a net negative charge the walls must carry away."""
    rate, energy = 1.0e19, 300.0  # m^-3 s^-1, eV
    stripping = VolumeSource({"e": 1}, rate, electron_energy=energy, name="stripping")
    pairs = VolumeSource(
        {"e": 1, "Ar+": 1}, 1.0e-2, per=("Ar",), name="beam ionization"
    )
    state = argon(power=0.0, sources=[stripping, pairs]).steady_state()
    assert state.converged
    assert state.power["stripping"] == pytest.approx(rate * energy * E * CHAMBER.volume)
    assert state.wall_losses["e"] == pytest.approx(state.wall_losses["Ar+"] + rate)
    assert sum(state.power.values()) == pytest.approx(
        0.0, abs=1e-8 * state.power["stripping"]
    )
    # every electron leaves over the presheath and a sheath lowered by the extra ones:
    # Gamma_e (5/2 T_e + V_f - T_e ln(Gamma_e / Gamma_+)) at the walls
    te = state.electron_temperature
    electrons, ions = state.wall_losses["e"], state.wall_losses["Ar+"]
    floating = te * math.log(math.sqrt(AR_ION.mass / (2.0 * math.pi * ELECTRON_MASS)))
    sheath = floating - te * math.log(electrons / ions)
    assert -state.power["walls"] == pytest.approx(
        electrons * (2.5 * te + sheath) * E * CHAMBER.volume, rel=1e-12
    )


def test_bad_models_are_refused() -> None:
    with pytest.raises(ValueError, match="electrons"):
        GlobalModel([AR, AR_ION], [], CHAMBER)
    with pytest.raises(ValueError, match="positive ion"):
        GlobalModel([ELECTRON, AR], [], CHAMBER)
    with pytest.raises(ValueError, match="unknown species"):
        GlobalModel(
            [ELECTRON, AR, AR_ION], [Reaction(("e", "Xe"), ("e",), 1.0)], CHAMBER
        )
    with pytest.raises(ValueError, match="quasi-neutrality"):
        GlobalModel([ELECTRON, AR, AR_ION], [], CHAMBER, fixed={"e": 1e16})
    with pytest.raises(ValueError, match="elastic"):
        Reaction(("e", "Ar"), ("e",), 1.0, elastic_mass=1e-26)
