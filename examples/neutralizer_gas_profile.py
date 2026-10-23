"""A beam-driven plasma neutralizer with the gas along the channel: scans of flow and current.

Gas fed into the middle of a neutralizer channel flows to its two open ends as a free-molecular
gas: a molecule hits the walls far more often than other molecules and random-walks to an end.
So the density falls linearly from the feed, and the channel holds 3.5 times more gas than the
conductance of its two ends alone suggests. ``GasProfile.knudsen`` is Knudsen diffusion with
free-molecular exits; picnic's ``NeutralParticleTransport`` with diffuse walls (the points, from
its ``neutralizer_gas_profile`` example, kept in ``examples/data``) has the same peak and an
8% lower mean. The neutralizer model keeps a profile's shape and balances its mean.

Against the gas flow, the profiles reach the best neutral fraction (0.57) at a 3.5 times lower
flow than the uniform gas of the end conductances; against the line density all three are one
curve: stripping on the gas depends on the line density alone, and the plasma, ionized below
1%, adds little. The denser gas also makes a less ionized plasma with a lower potential.
Against the beam current the plasma density grows almost in proportion, its potential stays
near 9 V and the neutral fraction moves in the fourth digit. Along the channel, every slice
balancing its own gas and beam (``Neutralizer.local_plasma``: a plasma that does not move along
the channel, with no losses through the ends) peaks upstream of the feed, where the D- that
make the electrons are, and its potential dips where the gas is densest. The cross sections and
rates are illustrative placeholders, not data (see
``wellbalanced.neutralizer.illustrative_deuterium_neutralizer``).
"""

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import plasma_plots
import xarray as xr
from plasmacoll.constants import ATOMIC_MASS

from wellbalanced import stack
from wellbalanced.neutralizer import (
    FastBeam,
    GasProfile,
    illustrative_deuterium_neutralizer,
)

warnings.simplefilter("ignore", UserWarning)  # the placeholder data warn
model = illustrative_deuterium_neutralizer(points=101)
tg = 0.03  # the placeholder's gas temperature, eV
knudsen = GasProfile.knudsen(
    model.length, model.width, model.height, 4.028 * ATOMIC_MASS, 2.0e21, tg
)
# NeutralParticleTransport's profile of the same channel at 2e21 molecules/s
# (picnic, src/picnic/examples/neutralizer_gas_profile.py), if it is at hand
markers = Path(__file__).parent / "data" / "neutralizer_gas_npt.csv"
npt = np.loadtxt(markers, delimiter=",", comments="#") if markers.exists() else None

# -- the gas flow ---------------------------------------------------------------------
flows = np.geomspace(1.0e21, 2.0e22, 6)  # molecules/s
cases = {
    "end conductances": model,
    "Knudsen profile": dataclasses.replace(model, gas_profile=knudsen),
}
if npt is not None:
    markers_profile = GasProfile(npt[:, 0], npt[:, 1], 2.0e21, tg)
    cases["markers' profile"] = dataclasses.replace(model, gas_profile=markers_profile)
scans = []
for case in cases.values():
    results = case.scan("gas_inflow", list(flows))
    scans.append(
        stack(results, "gas_inflow", flows, long_name="gas inflow", units="1/s")
    )
flow = xr.concat(scans, dim="case").assign_coords(case=list(cases))
electrons = flow.density.sel(species="e")
gas = flow.density.sel(species="D2") + flow.density.sel(species="D")
ionization = 100 * electrons / (electrons + gas)
ionization.attrs.update(long_name="ionization degree", units="%")

with plasma_plots.figure(1, 3, figsize=(13, 3.8)) as fig:
    ax = fig[0]
    ax.plot(knudsen.position, knudsen.density, label="Knudsen diffusion")
    if npt is not None:
        ax.plot(npt[:, 0], npt[:, 1], "o", ms=3, label="NeutralParticleTransport")
    uniform = 2.0e21 / model.pumping_speed(tg)
    ax.axhline(uniform, ls="--", color="gray", label="end conductances")
    ax.set_xlabel("position along the channel [m]")
    ax.set_ylabel("D$_2$ density [m$^{-3}$]")
    ax.set_title("the gas at 2e21 molecules/s, no beam")
    ax.legend()
    flow.neutral_fraction.plasma.plot.profiles(
        x="gas_inflow", over="case", logx=True, ax=fig[1]
    )
    ax = fig[2]
    for case in cases:
        scan = flow.sel(case=case)
        ax.plot(scan.line_density, scan.neutral_fraction, "o-", label=case)
    ax.set_xlabel("gas line density [m$^{-2}$]")
    ax.set_ylabel("neutral fraction at the exit")
    ax.set_title("against the line density: one curve")
    ax.legend()

with plasma_plots.figure(1, 2, figsize=(9, 3.6)) as fig:
    ionization.plasma.plot.profiles(x="gas_inflow", over="case", logx=True, ax=fig[0])
    flow.plasma_potential.plasma.plot.profiles(
        x="gas_inflow", over="case", logx=True, ax=fig[1]
    )

# -- the beam current, at the flow of the best neutral fraction with the profile -------
profiled = flow.sel(case="Knudsen profile")
best = float(flows[int(np.argmax(profiled.neutral_fraction.values))])
channel = dataclasses.replace(cases["Knudsen profile"], gas_inflow=best)
currents = np.array([2.5, 5.0, 10.0, 20.0, 40.0])  # A
beams = [FastBeam(model.beam.energy, i, model.beam.mass) for i in currents]
current = stack(
    channel.scan("beam", beams),
    "current",
    currents,
    long_name="beam current",
    units="A",
)
print(f"best flow with the profile {best:.3g} molecules/s")
for i, row in zip(currents, current.neutral_fraction.values, strict=True):
    print(f"  {i:5.1f} A: neutral fraction {row:.4f}")

with plasma_plots.figure(1, 3, figsize=(13, 3.6)) as fig:
    panels = [
        (current.neutral_fraction, "neutral fraction at the exit"),
        (current.density.sel(species="e"), "electron density [m$^{-3}$]"),
        (current.plasma_potential, "plasma potential [V]"),
    ]
    for ax, (values, label) in zip(fig, panels, strict=True):
        ax.semilogx(currents, values, "o-")
        ax.set_xticks(currents, [f"{i:g}" for i in currents])
        ax.minorticks_off()
        ax.set_xlabel("beam current [A]")
        ax.set_ylabel(label)

# -- along the channel: every slice in its own gas and beam ----------------------------
result = channel.solve()
local = channel.local_plasma(result, points=13)
with plasma_plots.figure(1, 3, figsize=(13, 3.6)) as fig:
    result.to_dataset().fraction.plasma.plot.profiles(
        x="position", over="state", title="charge states", ax=fig[0]
    )
    ax = fig[1]
    ax.plot(local.position, local.density.sel(species="e"), "o-", label="local slices")
    ax.axhline(result.plasma.electron_density, ls="--", label="global model")
    ax.set_xlabel("position along the channel [m]")
    ax.set_ylabel("electron density [m$^{-3}$]")
    ax.legend()
    ax = fig[2]
    ax.plot(local.position, local.plasma_potential, "o-", label="local slices")
    ax.axhline(result.plasma_potential, ls="--", label="global model")
    ax.set_xlabel("position along the channel [m]")
    ax.set_ylabel("plasma potential [V]")
    ax.legend()
