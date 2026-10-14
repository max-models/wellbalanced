"""A beam-driven plasma neutralizer: the neutral fraction against the gas flow.

A 1 MeV, 10 A deuterium negative-ion beam crosses a 3 m channel. On the gas,
and on the plasma it makes itself, it changes charge state; the plasma is a
global model of the channel fed by the beam's stripped and secondary electrons.
Magnetic cusps on the side walls confine the plasma: it is far denser, but its
ionization degree stays below a per cent, too little for stripping on the plasma
to raise the neutral fraction much over that of the gas alone. The cross sections and rates are
illustrative placeholders, not data (see
``wellbalanced.neutralizer.illustrative_deuterium_neutralizer``): the curves show
what the model computes, not what a neutralizer achieves.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from wellbalanced.neutralizer import illustrative_deuterium_neutralizer

warnings.simplefilter("ignore", UserWarning)  # the placeholder data warn
flows = np.geomspace(1.0e21, 2.0e22, 9)  # molecules/s
fig, (ax_f, ax_i) = plt.subplots(1, 2, figsize=(9, 3.6))
for label, cusp in (("cusps", True), ("no cusps", False)):
    model = illustrative_deuterium_neutralizer()
    if not cusp:
        model = illustrative_deuterium_neutralizer(cusp=None)
    results = model.scan("gas_inflow", list(flows))
    line = [r.gas_density * model.length for r in results]
    ax_f.plot(line, [r.neutral_fraction for r in results], "o-", label=label)
    ax_i.plot(
        line,
        [100 * r.plasma.ionization_degree(["D2", "D"]) for r in results],
        "o-",
        label=label,
    )
ax_f.set(xlabel="gas line density (m$^{-2}$)", ylabel="neutral fraction at the exit")
ax_i.set(xlabel="gas line density (m$^{-2}$)", ylabel="ionization degree (%)")
ax_f.legend()
fig.tight_layout()
