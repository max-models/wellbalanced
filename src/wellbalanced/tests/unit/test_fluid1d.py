"""The 1D fluid model: conservation, the ambipolar profile, a discharge, boundaries.

The cross sections and transport coefficients are synthetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from plasmacoll import CrossSection
from plasmacoll.constants import ELECTRON_MASS

from wellbalanced import ELECTRON, Reaction, Species, VolumeSource
from wellbalanced.fluid1d import (
    Boundary,
    Fluid1D,
    FluidSpecies,
    PotentialBoundary,
)

ION = Species.atomic("Ar+", 39.948, 1)
TE, TG = 2.0, 0.025


def plasma(length: float, cells: int, **settings) -> Fluid1D:
    defaults = {
        "species": [FluidSpecies(ELECTRON, 30.0), FluidSpecies(ION, 0.1)],
        "reactions": [],
        "length": length,
        "cells": cells,
        "electron_temperature": TE,
        "gas_temperature": TG,
    }
    return Fluid1D(**{**defaults, **settings})


def test_a_vacuum_gap_has_a_linear_potential() -> None:
    model = plasma(
        0.01,
        20,
        potential={
            "left": PotentialBoundary.potential(0.0),
            "right": PotentialBoundary.potential(100.0),
        },
    )
    unknowns, ok = model.step(model.pack({"e": 1.0, "Ar+": 1.0}, TE, 0.0), 1e-9)
    assert ok
    phi = model.state(unknowns).potential
    assert phi == pytest.approx(100.0 * model.x / 0.01, abs=1e-6)


def test_a_closed_box_conserves_every_species() -> None:
    closed = {name: Boundary.symmetry() for name in ("e", "Ar+")}
    model = plasma(
        0.01,
        60,
        boundaries={"left": dict(closed), "right": dict(closed)},
        potential={"right": PotentialBoundary.field(0.0)},
    )
    x = model.x
    electrons = 1e14 * np.exp(-(((x - 0.004) / 0.001) ** 2)) + 1e12
    ions = 1e14 * np.exp(-(((x - 0.006) / 0.001) ** 2)) + 1e12
    start = model.state(model.pack({"e": electrons, "Ar+": ions}, TE, 0.0))
    end = model.state(
        model.advance(model.pack({"e": electrons, "Ar+": ions}, TE, 0.0), 2e-7, 1e-8)
    )
    for name in ("e", "Ar+"):
        assert end.integral(end.densities[name]) == pytest.approx(
            start.integral(start.densities[name]), rel=1e-10
        )
    # the separated charges have moved towards each other
    assert not np.allclose(end.densities["e"], electrons, rtol=1e-3)


@pytest.mark.parametrize(("length", "tolerance"), [(0.05, 0.15), (0.5, 0.05)])
def test_ambipolar_diffusion_from_a_uniform_source_is_a_parabola(
    length: float, tolerance: float
) -> None:
    """n = S x (L - x) / (2 D_a), up to the sheaths, which shrink with Debye length / L."""
    mu_e, mu_i = 30.0, 0.1
    d_a = mu_i * mu_e / (mu_i + mu_e) * (TE + TG)
    peak = 1.0e16
    source = 8.0 * d_a * peak / length**2
    model = plasma(length, 400, sources={"e": source, "Ar+": source})
    guess = 0.5 * peak * np.sin(np.pi * model.x / length) + 1e10
    state = model.steady_state(model.pack({"e": guess, "Ar+": guess}, TE, 0.0))
    assert state.converged
    density = state.densities["Ar+"]
    assert density.max() == pytest.approx(peak, rel=tolerance)
    assert density.max() < peak  # the sheaths shorten the slab
    # every particle made leaves through the walls
    made = source * length
    for name, imbalance in model.balance(state).items():
        assert abs(imbalance) < 1e-10 * made, name
    # quasi-neutral in the middle
    middle = np.abs(model.x - length / 2) < 0.25 * length
    assert state.densities["e"][middle] == pytest.approx(density[middle], rel=1e-3)


def test_a_discharge_sustains_itself_and_closes_its_balances() -> None:
    ionization = Reaction.from_cross_section(
        ("e", "Ar"),
        ("e", "e", "Ar+"),
        CrossSection.constant(2e-20, threshold=15.76),
        ELECTRON_MASS,
        electron_energy_loss=15.76,
    )
    model = Fluid1D(
        [FluidSpecies(ELECTRON, 1.0e3), FluidSpecies(ION, 10.0)],
        [ionization],
        0.05,
        200,
        fixed={"Ar": 1.0e21},
        gas_temperature=TG,
        heating=2.0e4,
    )
    guess = 1e15 * np.sin(np.pi * model.x / 0.05) + 1e12
    state = model.steady_state(model.pack({"e": guess, "Ar+": guess}, 3.0, 0.0))
    assert state.converged
    made = state.integral(state.reaction_rates[ionization.name])
    for imbalance in model.balance(state).values():
        assert abs(imbalance) < 1e-10 * made
    energy = model.energy_balance(state)
    assert energy["heating"] == pytest.approx(2.0e4 * 0.05 / 1.602176634e-19)
    assert abs(sum(energy.values())) < 1e-8 * energy["heating"]
    assert energy["reactions"] < 0.0 and energy["walls"] < 0.0
    assert 1.0 < state.electron_temperature.mean() < 5.0
    # the plasma sits above the grounded walls
    assert state.potential.max() > 3.0 * state.electron_temperature.mean()


def test_boundaries_can_be_changed_between_steps() -> None:
    """Inject electrons and ions at the left: at steady state the same flux leaves."""
    model = plasma(0.02, 100, potential={"left": PotentialBoundary.field(0.0)})
    injected = 1.0e16  # about 1e15 m^-3 at the left: the Debye length is resolved
    model.set_boundary("left", "e", Boundary.flux(injected))
    model.set_boundary("left", "Ar+", Boundary.flux(injected))
    guess = 1e14 * (1.0 - model.x / 0.02) + 1e10
    state = model.steady_state(model.pack({"e": guess, "Ar+": guess}, TE, 0.0))
    assert state.converged
    for name in ("e", "Ar+"):
        assert state.fluxes[name][0] == pytest.approx(injected)
        assert state.fluxes[name][-1] == pytest.approx(injected, rel=1e-9)
    # then hold half the density the injection gave at the left end instead, with
    # the potential there (values in, as across a coupling interface: a held density
    # with a prescribed zero field would let the electrons diffuse out unrestrained)
    held_density = 0.5 * state.densities["Ar+"][0]
    model.set_boundary("left", "e", Boundary.density(held_density))
    model.set_boundary("left", "Ar+", Boundary.density(held_density))
    model.potential["left"] = PotentialBoundary.potential(float(state.potential[0]))
    unknowns = model.pack(state.densities, TE, state.potential)
    held = model.steady_state(unknowns)
    assert held.converged
    assert held.densities["Ar+"][0] == pytest.approx(held_density, rel=0.05)
    assert held.fluxes["Ar+"][-1] < injected  # less density, less flux
    with pytest.raises(ValueError, match="no species"):
        model.set_boundary("middle", "e", Boundary.wall())


def test_volume_sources_are_sources_per_unit_density_with_their_energy() -> None:
    """A beam-like source on a fixed gas equals the same rates given as sources and heating."""
    cells, gas = 50, 2.0e19
    x = (np.arange(cells) + 0.5) / cells
    rate = 1.0e21 * np.exp(-(((x - 0.5) / 0.2) ** 2))  # m^-3 s^-1
    energy = 30.0  # eV per event
    plain = plasma(
        0.05,
        cells,
        electron_temperature=None,
        fixed={"Ar": gas},
        sources={"e": rate, "Ar+": rate},
        heating=rate * energy * 1.602176634e-19,
    )
    source = VolumeSource(
        {"e": 1, "Ar+": 1, "Ar": -1},
        rate / gas,
        per=("Ar",),
        electron_energy=energy,
        name="beam",
    )
    beam = plasma(
        0.05,
        cells,
        electron_temperature=None,
        fixed={"Ar": gas},
        volume_sources=[source],
    )
    rng = np.random.default_rng(1)
    density = 1e15 * (1.0 + 0.1 * rng.random(cells))
    unknowns = beam.pack(
        {"e": density, "Ar+": density * 1.01},
        2.0 + rng.random(cells),
        rng.random(cells),
    )
    assert beam._residual(unknowns, None, np.inf) == pytest.approx(
        plain._residual(unknowns, None, np.inf), rel=1e-12, abs=1e-300
    )
    state = beam.state(unknowns)
    assert state.reaction_rates["beam"] == pytest.approx(rate)
    terms = beam.energy_balance(state)
    assert terms["sources"] == pytest.approx(state.integral(rate * energy))
    assert terms["reactions"] == 0.0
    with pytest.raises(ValueError, match="one per cell"):
        plasma(
            0.05,
            cells,
            fixed={"Ar": gas},
            volume_sources=[VolumeSource({"e": 1}, np.ones(3), per=("Ar",))],
        )
    with pytest.raises(ValueError, match="1D model"):
        source.coefficient(2.0, 0.025)


def test_bad_models_are_refused() -> None:
    with pytest.raises(ValueError, match="electrons"):
        Fluid1D([FluidSpecies(ION, 0.1)], [], 0.01, 10)
    with pytest.raises(ValueError, match="needs diffusion"):
        FluidSpecies(Species.atomic("Ar", 39.948))
    with pytest.raises(ValueError, match="needs mobility"):
        FluidSpecies(ION)
    with pytest.raises(ValueError, match="unknown species"):
        plasma(0.01, 10, reactions=[Reaction(("e", "Xe"), ("e",), 1.0)])
