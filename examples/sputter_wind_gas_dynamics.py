"""Sputter wind.

The sputter wind: the argon in front of a sputtering target, as a compressible gas.

The target of a magnetron is a wall in front of a gas at 1 Pa that emits sputtered metal atoms
and, at the same time, the ions that hit it return as neutrals. Here only the sputtered
titanium is emitted, at the flux the global model (:class:`wellbalanced.Magnetron`) finds for a
discharge current of 1 to 100 A on a target of 10 cm^2, into 2 cm of argon that is refilled
from a reservoir at the far end (`wellbalanced.gasdynamics`, the argon and the titanium as one
compressible gas with one velocity and temperature and the diffusion of the titanium through
the argon). Two cases of the energy of the sputtered atoms: cold atoms (they leave at the gas
temperature, as the global model assumes: only their number and momentum push the gas) and
5 eV atoms (a mean energy of the order of the one of sputtered metals, a placeholder: their
heat goes into the gas).

The first figure is the argon density profile in front of the target for several currents with
cold atoms; the second its mean value against the global model's rarefaction; the third how
the argon returns once the sputtering stops.

This is a demonstration of the model, not a validation. The model is the continuum limit
(Euler with species diffusion): at 1 Pa the mean free path of argon is a few mm, so the
Knudsen number of the region is of order 0.1 to 1 and a kinetic (DSMC) calculation is due. A
continuum gas thermalizes the 5 eV atoms in the cell next to the target, whereas at this
Knudsen number energetic atoms leave ballistically: with them the model overestimates the
heating of the gas by an order of magnitude, and the 5 eV result at 10 A and above (the gas
heated to thousands of kelvin) is that overestimate, not a prediction. The sputtered flux of
the global model uses its placeholder yields and the diffusion coefficient a placeholder
hard-sphere diameter of titanium.
"""

import warnings

import numpy as np
import plasma_plots
import xarray as xr
from plasmacoll.constants import ATOMIC_MASS, BOLTZMANN

from wellbalanced import MATERIALS, Magnetron
from wellbalanced.gasdynamics import (
    ARGON_MASS,
    Boundary,
    GasDynamics,
    ThermalRelaxation,
    binary_diffusion,
)

warnings.filterwarnings("ignore", message=".*placeholders, not data")

LENGTH = 0.02  # m: the height of the ionization region, the refill reservoir above it
PRESSURE, TEMPERATURE = 1.0, 300.0  # Pa, K
TITANIUM = 47.867 * ATOMIC_MASS
FILL_DENSITY = PRESSURE / (BOLTZMANN * TEMPERATURE)
CURRENTS = (1.0, 3.0, 10.0, 30.0, 100.0)


def voltage(current: float) -> float:
    """A placeholder current-voltage characteristic, 400 V at 0.1 A rising slowly."""
    return 400.0 + 60.0 * np.log10(1.0 + current)


def diffusion(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Ar-Ti diffusion: hard spheres of 3.6 and 3.0 angstrom (PLACEHOLDER diameters)."""
    return binary_diffusion(ARGON_MASS, TITANIUM, 3.3e-10, temperature, pressure)


def gas(flux: float, energy: float, cells: int = 80) -> GasDynamics:
    """The gas in front of a target that emits ``flux`` atoms/m^2/s of ``energy`` eV."""
    mean_speed = np.sqrt(8.0 * BOLTZMANN * TEMPERATURE / (np.pi * ARGON_MASS))
    model = GasDynamics(
        LENGTH,
        cells,
        diffusion=diffusion,
        thermal_conductivity=0.02,  # PLACEHOLDER: argon at 300 K, to check
        boundaries={
            "left": Boundary.wall(TEMPERATURE, metal_flux=flux, metal_energy=energy),
            "right": Boundary.inflow(PRESSURE, TEMPERATURE),
        },
    )
    # the side walls of a 1 cm channel, free-molecular exchange (PLACEHOLDER closure)
    model.relaxation.append(ThermalRelaxation(2.0 * mean_speed / (3.0 * 0.01)))
    model.set_uniform(PRESSURE, TEMPERATURE)
    return model


# the sputtered flux and the rarefaction of the global model at each current
magnetron = Magnetron(material=MATERIALS["Ti"], wind_efficiency=0.3)
magnetron = magnetron.calibrate_heating(2.0, voltage(2.0))
fluxes, global_rarefaction = [], []
for current in CURRENTS:
    state = magnetron.steady_state(current, voltage(current))
    sputtered = magnetron.target_currents(state)["sputtered"] / magnetron.target_area
    fluxes.append(sputtered)
    global_rarefaction.append(magnetron.diagnostics(state)["rarefaction"])

# cold atoms: the argon in front of the target, brought to its steady state
profiles, rows = [], []
for current, flux in zip(CURRENTS, fluxes, strict=True):
    model = gas(flux, energy=0.05)
    model.run(3.0e-4)
    state = model.state()
    argon = state.gas_number[:, 0] / FILL_DENSITY
    profiles.append(argon)
    rows.append(
        (current, flux, argon.mean(), argon[0], state.temperature[0, 0], state.u[-1, 0])
    )
profile = xr.DataArray(
    np.array(profiles),
    dims=("current", "x"),
    coords={"current": [f"{c:g} A" for c in CURRENTS], "x": model.x * 1e3},
    attrs={"long_name": "argon density over the filling density"},
)
profile.x.attrs.update(long_name="distance from the target", units="mm")

# 5 eV atoms at the two lowest currents: the heat of the sputtered atoms
hot = []
for current, flux in list(zip(CURRENTS, fluxes, strict=True))[:3:2]:
    model = gas(flux, energy=5.0, cells=60)
    model.run(3.0e-4)
    state = model.state()
    hot.append(
        (
            current,
            state.gas_number.mean() / FILL_DENSITY,
            state.temperature[0, 0],
            state.u[-1, 0],
        )
    )

print("1D gas model with cold sputtered atoms against the global model's rarefaction")
print("  I [A]  flux [1/m^2/s]   1D mean   1D at target   global model")
for (current, flux, mean, near, _t, _u), rarefaction in zip(
    rows, global_rarefaction, strict=True
):
    print(
        f"  {current:5.0f}  {flux:12.2e}   {mean:7.3f}   {near:11.3f}   {rarefaction:11.3f}"
    )
print("with 5 eV atoms (the heat of the sputtered atoms goes into the gas):")
for current, mean, temperature, velocity in hot:
    print(
        f"  {current:5.0f} A: argon {mean:.3f} of the filling density on average, "
        f"{temperature:.0f} K at the target, outflow {velocity:.0f} m/s"
    )

# refill: stop the sputtering at 30 A (cold atoms) and follow the argon back
model = gas(fluxes[3], energy=0.05)
model.run(3.0e-4)
deficit = 1.0 - model.state().gas_number.mean() / FILL_DENSITY
model.boundaries["left"].metal_flux = 0.0
step, stop = 5.0e-6, 4.0e-4
times, mean_argon = [0.0], [1.0 - deficit]
for k in range(1, int(round(stop / step)) + 1):
    model.run(3.0e-4 + k * step)
    times.append(k * step)
    mean_argon.append(model.state().gas_number.mean() / FILL_DENSITY)
mean_argon = np.array(mean_argon)
refill = xr.DataArray(
    mean_argon,
    dims=("t",),
    coords={"t": np.array(times) * 1e6},
    attrs={"long_name": "mean argon density over the filling density"},
)
refill.t.attrs.update(long_name="time after the sputtering stops", units="us")
recovered = np.nonzero(mean_argon >= 1.0 - 0.5 * deficit)[0]
refill_time = times[recovered[0]] if recovered.size else float("nan")
print(
    f"30 A: half of the mean argon deficit of {deficit:.2f} is back after "
    f"{refill_time * 1e6:.0f} us, with an overshoot from the sound waves (the global "
    f"model's placeholder refill time is {magnetron.refill_time * 1e6:.0f} us)"
)

with plasma_plots.figure(
    1, 3, title="Sputtered titanium into argon at 1 Pa, cold atoms"
) as fig:
    profile.plasma.plot.profiles(x="x", over="current", ax=fig[0])
    fig[0].set_title("argon in front of the target")
    fig[1].plot(CURRENTS, [r[2] for r in rows], marker="o", label="gas model, mean")
    fig[1].plot(
        CURRENTS, [r[3] for r in rows], marker="o", label="gas model, at target"
    )
    fig[1].plot(CURRENTS, global_rarefaction, marker="s", label="global model")
    fig[1].set_xlabel("discharge current [A]")
    fig[1].set_ylabel("argon density over the filling density")
    fig[1].set_title("rarefaction against the discharge current")
    fig[1].legend()
    refill.plasma.plot.timeseries(ax=fig[2])
    fig[2].set_title("refill after the sputtering stops (30 A)")
