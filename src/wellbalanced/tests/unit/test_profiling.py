"""The scope-profiler regions: off by default, recorded when the tag is enabled."""

from __future__ import annotations

import pytest
import scope_profiler as sp
from plasmacoll import CrossSection
from plasmacoll.constants import ELECTRON_MASS

from wellbalanced import ELECTRON, Chamber, GasDynamics, GlobalModel, Reaction, Species
from wellbalanced.gasdynamics import Boundary


@pytest.fixture(autouse=True)
def fresh_profiler(monkeypatch):
    """No profiling state leaks into or out of a test."""
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)
    sp.reset()
    yield
    sp.reset()


def argon() -> GlobalModel:
    ionization = Reaction.from_cross_section(
        ("e", "Ar"),
        ("e", "e", "Ar+"),
        CrossSection.constant(2.0e-20, threshold=15.76),
        ELECTRON_MASS,
        electron_energy_loss=15.76,
    )
    return GlobalModel(
        species=[
            ELECTRON,
            Species.atomic("Ar", 39.948),
            Species.atomic("Ar+", 39.948, 1),
        ],
        reactions=[ionization],
        chamber=Chamber(volume=1.0e-3, wall_area=0.06, h_factor=0.5),
        power=100.0,
        fixed={"Ar": 1.0e20},
    )


def recorded_regions(**setup) -> list[str]:
    sp.setup(output=None, **setup)
    assert argon().steady_state().converged
    results = sp.finalize(verbose=False, return_results=True)
    return [name for name in results.region_names if name.startswith("wellbalanced.")]


def test_regions_are_off_by_default():
    assert recorded_regions() == []


def test_enabling_the_tag_records_the_regions():
    assert recorded_regions(enable_tags=["wellbalanced"]) == [
        "wellbalanced.GlobalModel.steady_state"
    ]


@pytest.mark.parametrize("enable_tags", [[], ["wellbalanced"]])
def test_a_warning_from_a_region_points_at_the_caller(enable_tags):
    sp.setup(output=None, enable_tags=enable_tags)
    periodic = Boundary.periodic()
    model = GasDynamics(0.01, 8, boundaries={"left": periodic, "right": periodic})
    model.set_uniform(1.0, 300.0)
    with pytest.warns(RuntimeWarning, match="max_time") as record:
        model.steady_state(max_time=0.0)
    sp.finalize(verbose=False)
    assert record[0].filename == __file__


def test_the_opt_in_tag_has_an_entry_point():
    """``scope-profiler tags`` finds the tag through the package's entry point."""
    from importlib.metadata import entry_points

    from wellbalanced._profiling import _PROFILE_TAGS

    found = {
        entry_point.name: entry_point.value
        for entry_point in entry_points(group="scope_profiler.opt_in_tags")
    }
    assert found.get("wellbalanced") == "wellbalanced"
    assert "wellbalanced" in _PROFILE_TAGS
