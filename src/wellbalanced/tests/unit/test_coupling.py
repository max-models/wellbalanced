"""A power-dependent plasma in the field: the fixed point and the 0D driver's plasma.

The reference: in an infinitely long solenoid (periodic ``z``) with a uniform plasma
column, the r–z field gives exactly the coupling of :class:`~wellbalanced.RFDriver`
with ``coupling="solenoid"`` and ``field_factor=1``, so at the same absorbed power the
two plasma resistances agree to the discretisation error. On the tensor grid and on
refined patches (``kind``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from wellbalanced.driver import illustrative_hydrogen_driver
from wellbalanced.spatial import (
    DriverCoupling,
    InductiveField2D,
    PlasmaProfile,
    Winding,
    global_model_plasma,
    stretched_edges,
)
from wellbalanced.tests.unit._grids import GRIDS, patch_grid

simgeometry = pytest.importorskip("simgeometry")
Rectangle = simgeometry.Rectangle


@pytest.fixture(scope="module")
def driver():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return illustrative_hydrogen_driver(coupling="solenoid", field_factor=1.0)


def column_field(driver, kind="tensor"):
    """The driver as an infinite solenoid: its turns per length, its plasma column.

    The grid is made once, for every pass of the fixed point.
    """
    edges = (
        stretched_edges(0.0, 0.2, 0.0, 0.15, 5e-4, ratio=1.1),
        np.linspace(0.0, driver.length, 3),
    )
    geometry, boundaries = "cylindrical", ("neumann", "periodic")
    if kind == "tensor":
        where = {"x_edges": edges[0], "y_edges": edges[1]}
        where.update(geometry=geometry, boundaries=boundaries)
    else:
        grid = patch_grid(*edges, geometry=geometry, boundaries=boundaries)
        where = {"grid": grid}

    def field(profile: PlasmaProfile) -> InductiveField2D:
        settings = profile.settings()
        density = settings.pop("electron_density")
        return InductiveField2D(
            **where,
            frequency=driver.frequency,
            windings=[
                Winding(
                    Rectangle((driver.coil_radius, driver.length / 2), (0.004, 1.0)),
                    sign=driver.coil_turns,
                )
            ],
            electron_density=lambda r, z: np.where(r < driver.radius, density, 0.0),
            **settings,
        )

    return field


def coupling(driver, kind="tensor"):
    return DriverCoupling(
        field=column_field(driver, kind),
        plasma=global_model_plasma(driver),
        network_resistance=driver.network_resistance,
    )


@pytest.mark.parametrize("kind", GRIDS)
def test_the_0d_plasma_in_the_field_matches_the_exact_0d_coupling(driver, kind):
    state = coupling(driver, kind).at_absorbed_power(25e3)
    zero_d = state.plasma  # the DriverState of the same plasma
    assert state.plasma_resistance == pytest.approx(zero_d.plasma_resistance, rel=2e-3)
    assert state.coil_current == pytest.approx(zero_d.coil_current, rel=2e-3)
    assert state.solution.absorbed_power == pytest.approx(25e3)
    assert state.rf_power == pytest.approx(25e3 / state.efficiency)


@pytest.mark.parametrize("kind", GRIDS)
def test_the_generator_power_is_a_converged_fixed_point(driver, kind):
    state = coupling(driver, kind).at_rf_power(40e3, tolerance=1e-8)
    assert state.converged and 1 < state.iterations < 50
    r_p = state.plasma_resistance
    eta = r_p / (r_p + driver.network_resistance)
    assert state.absorbed_power == pytest.approx(eta * 40e3, rel=1e-6)
    assert state.rf_power == pytest.approx(40e3, rel=1e-6)
    assert state.efficiency == pytest.approx(eta, rel=1e-6)


def test_more_power_gives_a_denser_plasma_and_a_lower_efficiency(driver):
    """The 0D driver's trend: a denser plasma has a thinner skin, a lower R_s."""
    model = coupling(driver)
    low, high = model.at_absorbed_power(10e3), model.at_absorbed_power(60e3)
    assert high.plasma.plasma.electron_density > low.plasma.plasma.electron_density
    assert high.efficiency < low.efficiency
    assert (high.efficiency < low.efficiency) == (
        high.plasma.efficiency < low.plasma.efficiency
    )


def test_invalid_coupling_inputs(driver):
    model = coupling(driver)
    with pytest.raises(ValueError, match="power"):
        model.at_absorbed_power(0.0)
    with pytest.raises(ValueError, match="relaxation"):
        model.at_rf_power(1e3, relaxation=1.5)
