"""Neutralizer gas scan.

A beam-driven plasma neutralizer: the neutral fraction against the gas flow.

A 1 MeV, 10 A deuterium negative-ion beam crosses a 3 m channel. On the gas,
and on the plasma it makes itself, it changes charge state; the plasma is a
global model of the channel fed by the beam's stripped and secondary electrons.
Magnetic cusps on the side walls confine the plasma: it is far denser, but its
ionization degree stays below a per cent, too little for stripping on the plasma
to raise the neutral fraction much over that of the gas alone. The cross sections
and rates are illustrative placeholders, not data (see
``wellbalanced.neutralizer.illustrative_deuterium_neutralizer``): the curves show
what the model computes, not what a neutralizer achieves. Every scan is one
xarray dataset, plotted with plasma-plots.
"""

import warnings

import numpy as np
import plasma_plots
import xarray as xr

from wellbalanced import stack
from wellbalanced.neutralizer import illustrative_deuterium_neutralizer

warnings.simplefilter("ignore", UserWarning)  # the placeholder data warn
flows = np.geomspace(1.0e21, 2.0e22, 9)  # molecules/s
cases = {"cusps": {}, "no cusps": {"cusp": None}}
scans = []
for overrides in cases.values():
    model = illustrative_deuterium_neutralizer(**overrides)
    results = model.scan("gas_inflow", list(flows))
    scans.append(
        stack(results, "gas_inflow", flows, long_name="gas inflow", units="1/s")
    )
result = xr.concat(scans, dim="case").assign_coords(case=list(cases))

electrons = result.density.sel(species="e")
gas = result.density.sel(species="D2") + result.density.sel(species="D")
ionization = 100 * electrons / (electrons + gas)
ionization.attrs.update(long_name="ionization degree", units="%")

with plasma_plots.figure(1, 3, figsize=(13, 3.8)) as fig:
    result.neutral_fraction.plasma.plot.profiles(
        x="gas_inflow", over="case", logx=True, ax=fig[0]
    )
    ionization.plasma.plot.profiles(x="gas_inflow", over="case", logx=True, ax=fig[1])
    # the beam along the channel, cusps, at the flow nearest the best neutral fraction
    best = int(np.argmax(result.neutral_fraction.sel(case="cusps").values))
    result.fraction.sel(case="cusps").isel(gas_inflow=best).plasma.plot.profiles(
        x="position",
        over="state",
        title=f"charge states at {flows[best]:.2g} molecules/s",
        ax=fig[2],
    )
