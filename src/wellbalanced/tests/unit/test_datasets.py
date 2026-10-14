"""The results as labelled xarray datasets; synthetic rates."""

from __future__ import annotations

import numpy as np
import pytest
from plasmacoll import CrossSection
from plasmacoll.constants import ELECTRON_MASS

from wellbalanced import (
    ELECTRON,
    Chamber,
    Fluid1D,
    FluidSpecies,
    GlobalModel,
    Reaction,
    Species,
    stack,
)

AR = Species.atomic("Ar", 39.948)
AR_ION = Species.atomic("Ar+", 39.948, 1)
IONIZATION = Reaction.from_cross_section(
    ("e", "Ar"),
    ("e", "e", "Ar+"),
    CrossSection.constant(2e-20, threshold=15.76),
    ELECTRON_MASS,
    electron_energy_loss=15.76,
    name="ionization",
)


def argon() -> GlobalModel:
    return GlobalModel(
        [ELECTRON, AR, AR_ION],
        [IONIZATION],
        Chamber(1e-3, 0.06, h_factor=0.5),
        power=100.0,
        fixed={"Ar": 1e20},
    )


def test_a_global_state_is_a_labelled_dataset() -> None:
    state = argon().steady_state()
    ds = state.to_dataset()
    assert set(ds.species.values) == {"e", "Ar", "Ar+"}
    assert float(ds.density.sel(species="e")) == pytest.approx(state.electron_density)
    assert float(ds.electron_temperature) == pytest.approx(state.electron_temperature)
    assert float(ds.electron_power.sum()) == pytest.approx(0.0, abs=1e-8)
    assert ds.density.attrs == {"long_name": "density", "units": "m^-3"}
    assert ds.reaction_rate.sel(reaction="ionization") > 0.0


def test_a_scan_stacks_along_its_parameter() -> None:
    powers = [50.0, 100.0, 200.0]
    ds = stack(argon().scan("power", powers), "power", powers, units="W")
    assert ds.density.dims == ("power", "species")
    assert ds.electron_power.dims == ("power", "channel")  # energy terms per power
    assert ds.coords["power"].attrs == {"long_name": "power", "units": "W"}
    assert ds.density.attrs == {"long_name": "density", "units": "m^-3"}
    ne = ds.density.sel(species="e").values
    assert ne / ne[0] == pytest.approx([1.0, 2.0, 4.0], rel=1e-6)
    assert bool(ds.converged.all())
    with pytest.raises(ValueError, match="one value per result"):
        stack(argon().scan("power", powers), "power", powers[:2])


def test_a_fluid_state_is_a_labelled_dataset() -> None:
    model = Fluid1D(
        [FluidSpecies(ELECTRON, 30.0), FluidSpecies(AR_ION, 0.1)],
        [],
        0.02,
        50,
        electron_temperature=2.0,
        sources={"e": 1e19, "Ar+": 1e19},
    )
    guess = 1e14 * np.sin(np.pi * model.x / 0.02) + 1e10
    state = model.steady_state(model.pack({"e": guess, "Ar+": guess}, 2.0, 0.0))
    ds = state.to_dataset()
    assert ds.density.dims == ("species", "x")
    assert ds.flux.dims == ("species", "x_face")
    assert ds.x.attrs["units"] == "m" and ds.sizes["x_face"] == 51
    assert np.array_equal(ds.potential.values, state.potential)
    assert "energy_flux" not in ds  # isothermal electrons
