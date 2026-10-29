"""RF driver neutrals.

The BUG driver's gas: H and H2 as Navier–Stokes fluids, depleted and heated by the plasma.

Step R3.5 of the RF-driver plan: :class:`wellbalanced.spatial.NeutralFluid2D`, the neutral
model of Zielke et al. (PSST 31, 035019, 2022, section 2.3), coupled to the driver plasma
of :class:`wellbalanced.spatial.Fluid2D` by :func:`wellbalanced.spatial.couple_neutrals`.
The driver is the r–z cylinder of :mod:`examples.rf_driver_fluid2d` (radius 12 cm, 17 cm
long), uniformly heated (no field here: the field coupling is
:func:`wellbalanced.spatial.fluid_plasma`); H2 comes in through the backplate near the
axis at the rate that fills the driver to 0.3 Pa at 300 K without plasma, and both
species leave through the open end (the expansion region, here a grid of transparency
0.3); the atoms recombine on the walls with a probability of 0.1, the ions come back as
gas. At each power the plasma and the gas are iterated to a common steady state.

It shows what the fixed background of NIBS'22 leaves out: the gas thins where the plasma
is, the atoms (Franck–Condon fragments, charge exchange with the drifting ions) are
hotter than the molecules, and the plasma's density follows.

Every rate is a placeholder: the plasma's reactions are those of
:func:`wellbalanced.driver.illustrative_hydrogen_driver` (constant cross sections), the
neutral collisions hard spheres (H2–H2 of the size that gives H2's viscosity at 300 K),
the wall probability, transparency and Franck–Condon energies round numbers; the grid
is coarse (2 cm). The numbers show the model working, not the BUG driver.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from plasmacoll.constants import BOLTZMANN, ELEMENTARY_CHARGE

from wellbalanced import ELECTRON, WallReaction
from wellbalanced.driver import illustrative_hydrogen_driver
from wellbalanced.fluid1d import FluidSpecies
from wellbalanced.spatial import (
    CellGrid,
    Collision,
    Fluid2D,
    Inlet,
    NeutralFluid2D,
    NeutralSpecies,
    Opening,
    couple_neutrals,
    hard_sphere,
)
from wellbalanced.spatial.neutrals import ROOM_TEMPERATURE

E = ELEMENTARY_CHARGE
KELVIN = BOLTZMANN / E
POWERS = [10e3, 25e3, 50e3]
RADIUS, LENGTH = 0.12, 0.17
TRANSPARENCY, RECOMBINATION = 0.3, 0.1

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)  # placeholder rates, said above
    driver = illustrative_hydrogen_driver(radius=RADIUS, length=LENGTH)
gas_temperature = driver.model_options["gas_temperature"]
mass = {s.name: s.mass for s in driver.species}
grid = CellGrid(
    np.linspace(0.0, RADIUS, 7), np.linspace(0.0, LENGTH, 10), geometry="cylindrical"
)
volume = np.pi * RADIUS**2 * LENGTH

# the plasma: electrons, H+, H2+ and H3+, ions with a 3 cm mean free path
species = [FluidSpecies(ELECTRON, 1.0)]
for s in driver.species:
    if s.charge and s.name != "e":
        thermal = np.sqrt(8 * E * gas_temperature / (np.pi * s.mass))
        species.append(
            FluidSpecies(s, E * driver.ion_mean_free_path / (s.mass * thermal))
        )
plasma = Fluid2D(
    grid,
    species,
    driver.reactions,
    fixed=driver.fixed,
    gas_temperature=gas_temperature,
    momentum_transfer=driver.momentum_transfer,
    ion_mean_free_path=driver.ion_mean_free_path,
    heating=POWERS[0] / volume,
)

# the gas: H2 (two rotational and one vibrational degree of freedom, as the paper) and H
fill = 0.3 / (E * ROOM_TEMPERATURE)
mean_speed = np.sqrt(8 * E * ROOM_TEMPERATURE / (np.pi * mass["H2"]))
feed = TRANSPARENCY * fill * mean_speed / 4 * np.pi * RADIUS**2
collisions = [
    Collision("H2", "H2", hard_sphere(2.6e-19, mass["H2"], mass["H2"])),
    Collision("H", "H2", hard_sphere(2.0e-19, mass["H"], mass["H2"])),
    Collision("H", "H", hard_sphere(1.5e-19, mass["H"], mass["H"])),
    Collision("H", "H+", 3e-15),  # charge exchange: twice its rate coefficient
] + [
    Collision(n, ion, 2e-15)
    for n in ("H", "H2")
    for ion in ("H+", "H2+", "H3+")
    if (n, ion) != ("H", "H+")
]
gas = NeutralFluid2D(
    grid,
    [
        NeutralSpecies(next(s for s in driver.species if s.name == "H2"), 3.0),
        NeutralSpecies(next(s for s in driver.species if s.name == "H"), 0.0),
    ],
    driver.reactions,
    plasma_species=[s for s in driver.species if s.charge],
    collisions=collisions,
    wall_reactions=[WallReaction("H", RECOMBINATION, {"H2": 0.5})],
    ion_wall_products=driver.model_options["ion_wall_products"],
    openings=[Opening(lambda r, z: z > LENGTH - 1e-9, TRANSPARENCY, "grid")],
    inlets=[Inlet(lambda r, z: (z < 1e-9) & (r < 0.04), "H2", feed)],
)

# a first plasma in the fixed background, then the power scan
shape = np.cos(0.4 * np.pi * grid.xx / RADIUS) * np.sin(
    np.pi * (0.9 * grid.yy / LENGTH + 0.05)
)
ions = {"H+": 0.25, "H2+": 0.5, "H3+": 0.25}
start = time.perf_counter()
fixed_state = plasma.steady_state(
    plasma.pack({n: f * 1e18 * shape for n, f in ions.items()}, 6.0, 30.0)
)
neutral_guess = gas.pack({"H2": fill, "H": 0.1 * fill}, ROOM_TEMPERATURE)
plasma_guess = fixed_state
results = []
print(f"fill: {fill:.3g} m^-3 of H2 at 300 K (0.3 Pa), fed at {feed:.3g} /s")
for power in POWERS:
    plasma.heating = np.full(plasma.size, power / volume)
    reference = plasma.steady_state(plasma_guess.unknowns)  # in the fixed background
    coupled = couple_neutrals(plasma, gas, reference, neutral_guess, tolerance=3e-3)
    state, ions_state = coupled.neutrals, coupled.plasma
    results.append((power, reference, coupled))
    total = state.densities["H"] + state.densities["H2"]
    print(
        f"{power / 1e3:4.0f} kW: {coupled.passes} passes (converged: {coupled.converged}); "
        f"n_H + n_H2 = {state.mean(total):.3g} m^-3 ({state.mean(total) / fill:.2f} of "
        f"the fill), n_H / n_H2 = {state.mean(state.densities['H']) / state.mean(state.densities['H2']):.2f}, "
        f"p = {state.mean(state.pressure):.2f} Pa; "
        f"T_H2 = {state.mean(state.temperatures['H2']) / KELVIN:.0f} K, "
        f"T_H = {state.mean(state.temperatures['H']) / KELVIN:.0f} K; "
        f"n_e = {ions_state.mean(ions_state.electron_density):.3g} m^-3 "
        f"(fixed gas: {reference.mean(reference.electron_density):.3g}), "
        f"T_e = {ions_state.mean(ions_state.electron_temperature):.2f} eV "
        f"(fixed gas: {reference.mean(reference.electron_temperature):.2f})"
    )
    heat = gas.energy_balance(state)
    print(
        f"        gas heating: reactions {heat['reactions']:.0f} W, collisions with "
        f"the ions {heat['collisions']:.0f} W, to the walls {-heat['walls']:.0f} W"
    )
    neutral_guess, plasma_guess = state.unknowns, ions_state
print(f"{time.perf_counter() - start:.0f} s")

# --- figures ------------------------------------------------------------------------------
power, reference, coupled = results[-1]
state = coupled.neutrals
r, z = grid.x * 100, grid.y * 100
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
panels = [
    (state.densities["H2"], "n_H2 (m⁻³)"),
    (state.densities["H"], "n_H (m⁻³)"),
    (state.temperatures["H"] / KELVIN, "T_H (K)"),
    (coupled.plasma.electron_density, "n_e (m⁻³)"),
]
for ax, (values, title) in zip(axes, panels, strict=True):
    mesh = ax.pcolormesh(z, r, values, shading="auto")
    ax.set_xlabel("z (cm)")
    ax.set_title(f"{title}, {power / 1e3:.0f} kW", loc="left")
    fig.colorbar(mesh, ax=ax)
axes[0].set_ylabel("r (cm)")
fig.tight_layout()

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
kw = [p / 1e3 for p, _, _ in results]
for name, style in (("H2", "-o"), ("H", "-s")):
    left.plot(
        kw,
        [c.neutrals.mean(c.neutrals.densities[name]) / fill for _, _, c in results],
        style,
        label=name,
    )
    right.plot(
        kw,
        [
            c.neutrals.mean(c.neutrals.temperatures[name]) / KELVIN
            for _, _, c in results
        ],
        style,
        label=name,
    )
left.plot(
    kw,
    [
        c.neutrals.mean(c.neutrals.densities["H"] + c.neutrals.densities["H2"]) / fill
        for _, _, c in results
    ],
    "-^",
    label="H + H2",
)
left.set_xlabel("absorbed power (kW)")
left.set_ylabel("mean density / fill")
left.set_title("the gas thins as the power rises", loc="left")
left.legend(frameon=False)
right.set_xlabel("absorbed power (kW)")
right.set_ylabel("mean temperature (K)")
right.set_title("and heats, the atoms most", loc="left")
right.legend(frameon=False)
fig.tight_layout()
