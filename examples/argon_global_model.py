"""A global model of an argon discharge: electron temperature and density against pressure.

The textbook global model (Lieberman and Lichtenberg, ch. 10): one ion species,
ionization from a step cross section averaged over a Maxwellian (with plasmacoll),
elastic losses, ions leaving at the Bohm speed. The particle balance alone sets the
electron temperature, which falls with the pressure; the power balance then sets the
density. Magnetic cusps that cover most of the wall lower both the losses and the
electron temperature. The cross sections are illustrative step functions, not
argon data.
"""

import matplotlib.pyplot as plt
import numpy as np
from plasmacoll import CrossSection
from plasmacoll.constants import BOLTZMANN, ELECTRON_MASS

from wellbalanced import ELECTRON, Chamber, Cusp, GlobalModel, Reaction, Species

argon = Species.atomic("Ar", 39.948)
argon_ion = Species.atomic("Ar+", 39.948, 1)
reactions = [
    Reaction.from_cross_section(
        ("e", "Ar"),
        ("e", "e", "Ar+"),
        CrossSection.constant(2.0e-20, threshold=15.76),
        ELECTRON_MASS,
        electron_energy_loss=15.76,
        name="ionization",
    ),
    Reaction.from_cross_section(
        ("e", "Ar"),
        ("e", "Ar"),
        CrossSection.constant(1.0e-20, threshold=11.5),
        ELECTRON_MASS,
        electron_energy_loss=11.5,
        name="excitation",
    ),
    Reaction(
        ("e", "Ar"), ("e", "Ar"), 1.0e-13, elastic_mass=argon.mass, name="elastic"
    ),
]
# a cylinder of 10 cm radius and 20 cm length
radius, length = 0.1, 0.2
volume = np.pi * radius**2 * length
wall = 2 * np.pi * radius * (radius + length)
chambers = {
    "open walls": Chamber(volume, wall, h_factor=0.3),
    "cusps on the side": Chamber(
        volume,
        wall,
        ion_loss_area=2 * np.pi * radius**2,
        h_factor=0.3,
        cusp=Cusp(field=0.1, line_length=16 * length),
    ),
}

pressures = np.geomspace(0.1, 10.0, 15)  # Pa
gas = pressures / (BOLTZMANN * 300.0)  # m^-3
fig, (ax_te, ax_ne) = plt.subplots(1, 2, figsize=(9, 3.6))
for label, chamber in chambers.items():
    model = GlobalModel(
        species=[ELECTRON, argon, argon_ion],
        reactions=reactions,
        chamber=chamber,
        power=500.0,
        fixed={"Ar": float(gas[0])},
    )
    states = model.scan("fixed", [{"Ar": float(n)} for n in gas])
    ax_te.semilogx(
        pressures, [s.electron_temperature for s in states], "o-", label=label
    )
    ax_ne.loglog(pressures, [s.electron_density for s in states], "o-", label=label)
ax_te.set(xlabel="pressure (Pa)", ylabel="electron temperature (eV)")
ax_ne.set(xlabel="pressure (Pa)", ylabel="electron density (m$^{-3}$)")
ax_te.legend()
fig.suptitle("Argon, 500 W absorbed")
fig.tight_layout()
