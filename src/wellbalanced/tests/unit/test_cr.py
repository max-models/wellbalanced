"""The hydrogen reactions from OpenCR and the driver that uses them."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("opencr")

from wellbalanced.cr import (  # noqa: E402
    CR_TEMPERATURES,
    hydrogen_driver,
    opencr_hydrogen_reactions,
)
from wellbalanced.driver import illustrative_hydrogen_driver  # noqa: E402


@pytest.fixture(scope="module")
def reactions():
    return {r.name: r for r in opencr_hydrogen_reactions(1e17, n_max=8)}


def test_rates_are_positive_and_follow_the_table(reactions) -> None:
    t = np.array([1.0, 5.0, 30.0])
    for reaction in reactions.values():
        k = reaction.coefficients(t, 0.08)
        assert np.all(np.isfinite(k)) and np.all(k >= 0.0)
    ionization = reactions["CR ionization of H"].coefficients(t, 0.08)
    assert np.all(np.diff(ionization) > 0.0)
    # outside the table the end values hold
    low = reactions["CR ionization of H"].coefficient(0.01, 0.08)
    assert low == pytest.approx(
        reactions["CR ionization of H"].coefficient(CR_TEMPERATURES[0], 0.08)
    )


def test_energy_loss_covers_the_ionization_energy(reactions) -> None:
    for t in (2.0, 10.0, 50.0):
        loss = reactions["CR energy loss on H (loss)"].coefficient(t, 0.08)
        ionization = reactions["CR ionization of H"].coefficient(t, 0.08)
        assert loss > 13.6 * ionization


def test_against_the_placeholders(reactions) -> None:
    # the placeholder cross sections were round numbers; the CR rates are within
    # an order of magnitude of them at 5 eV, the dissociation above them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        old = {r.name: r for r in illustrative_hydrogen_driver().reactions}
    pairs = (
        ("CR ionization of H", "ionization of H"),
        ("ionization of H2", "ionization of H2"),
        ("dissociation of H2", "dissociation of H2"),
    )
    for new, placeholder in pairs:
        ratio = reactions[new].coefficient(5.0, 0.08) / old[placeholder].coefficient(
            5.0, 0.08
        )
        assert 0.1 < ratio < 10.0


def test_driver_solves() -> None:
    driver = hydrogen_driver(1e17)
    state = driver.solve(20e3)
    assert state.plasma.converged
    assert 1.0 < state.plasma.electron_temperature < 20.0
    assert state.plasma.densities["e"] > 1e16


def test_deuterium_neutralizer() -> None:
    from wellbalanced.cr import deuterium_neutralizer

    deuterium = {r.name: r for r in opencr_hydrogen_reactions(isotope="D")}
    assert "CR ionization of D" in deuterium
    assert deuterium["ionization of D2"].reactants == ("e", "D2")
    with pytest.raises(ValueError):
        opencr_hydrogen_reactions(isotope="T")
    with pytest.warns(UserWarning, match="placeholder"):
        channel = deuterium_neutralizer()
    names = {r.name for r in channel.reactions}
    assert "CR ionization of D" in names and "H3+ formation" not in names
    assert not any("D3+" in r.products for r in channel.reactions)
