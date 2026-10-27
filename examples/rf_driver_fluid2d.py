"""The BUG driver in r–z as a 2D fluid heated by its inductive field.

Step R3.3 of the RF-driver plan: :class:`wellbalanced.spatial.Fluid2D`, the driver plasma as
a multi-species fluid (electrons, H+, H2+ and H3+ in a fixed H2 and H background at
0.3 Pa), on the cells of the r–z field of :mod:`examples.rf_driver_rz` (six coil turns at
1 MHz, the backplates as conductors). The ions have inertia and a constant mean free path,
the electrons drift and diffuse and carry their energy, the potential is quasi-neutral with
sheath walls; the plasma is the driver cylinder (radius 12 cm, 17 cm long), its open end a
wall, as the chamber of :class:`wellbalanced.RFDriver`'s global model. It computes:

1. the fluid with a uniform 25 kW against the global model: the fluid loses its ions
   faster, because the global model applies Lieberman and Lichtenberg's edge-to-centre
   ratio to the mean density; with the fluid's own edge-to-mean ratio the global model
   gives the fluid's plasma;
2. the fluid in its field at 25 kW absorbed (:func:`wellbalanced.spatial.fluid_plasma`
   with :class:`wellbalanced.spatial.DriverCoupling`), against the global model's plasma
   in the same field and the 0D transformer model;
3. the RF Lorentz force of that field, ``-(e/2) Re(u x B*)`` per electron, which pushes
   the electrons away from the coil: as an equivalent field it is several times the
   ambipolar one, ``T_e`` over the skin depth (the time-averaged RF magnetic pressure
   exceeds the plasma pressure at the edge). With the force at full strength the
   field–fluid fixed point does not converge yet (``fluid_plasma(lorentz=True)``, with
   ``ramp`` and ``relaxation``, loses the fluid's steady state), so it is shown, not
   applied, here.

The rates are placeholders (constant cross sections), the ion mobilities follow from the
assumed 3 cm mean free path, the wall conditions are textbook ones (Zielke's thesis has
the model's), the grid is coarse (6 mm) and the transport first order: the numbers show
the model working, not the BUG driver.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from plasmacoll.constants import ELEMENTARY_CHARGE
from simgeometry import Circle, Rectangle

from wellbalanced.driver import illustrative_hydrogen_driver
from wellbalanced.fluid1d import FluidSpecies
from wellbalanced.global_model import Chamber, GlobalModel
from wellbalanced.spatial import (
    CellGrid,
    Conductor,
    DriverCoupling,
    Fluid2D,
    InductiveField2D,
    PlasmaProfile,
    PlasmaRegion,
    Winding,
    fluid_plasma,
    global_model_plasma,
    stretched_edges,
)

POWER = 25e3
DRIVER_RADIUS, DRIVER_END = 0.12, 0.17
COIL_RADIUS, TUBE_RADIUS = 0.128, 0.003
COIL_Z = np.linspace(0.035, 0.10, 6)
SPACING = 6e-3


def box(r0, r1, z0, z1):
    """A rectangle from its bounds."""
    return Rectangle(((r0 + r1) / 2, (z0 + z1) / 2), (r1 - r0, z1 - z0))


with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)  # placeholder rates, said above
    driver = illustrative_hydrogen_driver(radius=DRIVER_RADIUS, length=DRIVER_END)
gas_temperature = driver.model_options["gas_temperature"]

# The grid: 6 mm over the driver (its cells are the fluid's), 3 mm over the coil,
# stretched out to the far boundary for the field.
r_edges = np.concatenate(
    [
        np.linspace(0.0, DRIVER_RADIUS, round(DRIVER_RADIUS / SPACING) + 1),
        stretched_edges(DRIVER_RADIUS, 0.5, DRIVER_RADIUS, 0.135, 3e-3, 1.15)[1:],
    ]
)
z_edges = np.concatenate(
    [
        stretched_edges(-0.3, 0.0, -0.01, 0.0, 5e-3, 1.15)[:-1],
        np.linspace(0.0, DRIVER_END, round(DRIVER_END / SPACING) + 1),
        stretched_edges(DRIVER_END, 0.6, DRIVER_END, DRIVER_END + 0.01, 5e-3, 1.15)[1:],
    ]
)
REGION = PlasmaRegion([box(0.0, DRIVER_RADIUS, 0.0, DRIVER_END)])
WINDINGS = [
    Winding(Circle((COIL_RADIUS, z), TUBE_RADIUS), name=f"turn {k}")
    for k, z in enumerate(COIL_Z)
]
CONDUCTORS = [Conductor([box(0.0, 0.125, -0.04, 0.0)], name="driver backplate")]


grid = CellGrid(r_edges, z_edges, geometry="cylindrical")


def field(profile: PlasmaProfile) -> InductiveField2D:
    """The r–z field of the coil for a plasma, on the fluid's grid."""
    return InductiveField2D(
        grid=grid,
        frequency=driver.frequency,
        windings=WINDINGS,
        conductors=CONDUCTORS,
        plasma_region=REGION,
        **profile.settings(),
    )


species = [FluidSpecies(s, 1.0) for s in driver.species if s.name == "e"] + [
    FluidSpecies(
        s,
        ELEMENTARY_CHARGE
        * driver.ion_mean_free_path
        / (
            s.mass * np.sqrt(8 * ELEMENTARY_CHARGE * gas_temperature / (np.pi * s.mass))
        ),
    )
    for s in driver.species
    if s.charge > 0
]
inside = REGION.mask(grid)
volume = np.pi * DRIVER_RADIUS**2 * DRIVER_END
fluid = Fluid2D(
    grid,
    species,
    driver.reactions,
    region=REGION,
    fixed=driver.fixed,
    gas_temperature=gas_temperature,
    momentum_transfer=driver.momentum_transfer,
    ion_mean_free_path=driver.ion_mean_free_path,
    heating=np.where(inside, POWER / volume, 0.0),
)

# --- 1. a uniform 25 kW against the global model ----------------------------------------
start = time.perf_counter()
zero_d = driver.model(POWER).steady_state()
shape = np.cos(0.4 * np.pi * grid.xx / DRIVER_RADIUS) * np.sin(
    np.pi * (0.9 * np.clip(grid.yy / DRIVER_END, 0, 1) + 0.05)
)
guess = fluid.pack(
    {
        name: zero_d.densities[name] * shape / np.mean(shape[inside])
        for name in ("H+", "H2+", "H3+")
    },
    zero_d.electron_temperature,
    5 * zero_d.electron_temperature,
)
uniform = fluid.steady_state(guess)
ne = uniform.electron_density
walls = fluid.wall_faces  # the edge density, weighted by the walls' areas
edge = np.average(ne.ravel()[walls["cell"]], weights=walls["area"]) / uniform.mean(ne)
own = GlobalModel(
    species=driver.species,
    reactions=driver.reactions,
    chamber=Chamber(driver.chamber.volume, driver.chamber.wall_area, h_factor=edge),
    power=POWER,
    fixed=driver.fixed,
    **driver.model_options,
).steady_state()
print(
    f"fluid: {fluid.size} cells, steady {uniform.converged}, "
    f"{time.perf_counter() - start:.0f} s"
)
print("uniform 25 kW                <n_e> (m^-3)   T_e (eV)   loss ratio")
print(
    f"fluid                         {uniform.mean(ne):9.3e}   {uniform.mean(uniform.electron_temperature):6.2f}"
    f"     {edge:.3f} (edge / mean)"
)
print(
    f"global model, RFDriver        {zero_d.electron_density:9.3e}   {zero_d.electron_temperature:6.2f}"
    f"     {driver.chamber.h_factor:.3f} (Lieberman, edge / centre)"
)
print(
    f"global model, fluid's ratio   {own.electron_density:9.3e}   {own.electron_temperature:6.2f}"
)
energy = fluid.energy_balance(uniform)
print(
    "fluid energy balance (W):",
    {k: round(v) for k, v in energy.items()},
    f"sum {sum(energy.values()):.1e}",
)

# --- 2. the fluid in its field, and 3. the Lorentz force ------------------------------
start = time.perf_counter()
coupled = DriverCoupling(
    field=field,
    plasma=fluid_plasma(fluid, field, uniform, lorentz=False),
    network_resistance=driver.network_resistance,
).at_absorbed_power(POWER)
result = coupled.plasma
plasma = result.state
print(
    f"\nfield and fluid: {result.passes} passes, converged {result.converged}, "
    f"{time.perf_counter() - start:.0f} s"
)
reference = DriverCoupling(
    field=field,
    plasma=global_model_plasma(
        driver, np.where(inside, shape / np.mean(shape[inside]), 0)
    ),
    network_resistance=driver.network_resistance,
).at_absorbed_power(POWER)

print("25 kW absorbed                coil current   eta     <n_e> (m^-3)  T_e (eV)")
print(
    f"fluid in its field               {coupled.coil_current:6.0f} A    "
    f"{coupled.efficiency:5.1%}   {plasma.mean(plasma.electron_density):9.3e}   "
    f"{plasma.mean(plasma.electron_temperature):5.2f}"
)
print(
    f"global model in the field        {reference.coil_current:6.0f} A    "
    f"{reference.efficiency:5.1%}   {reference.plasma.plasma.electron_density:9.3e}   "
    f"{reference.plasma.plasma.electron_temperature:5.2f}"
)
print(
    f"global model, transformer        {reference.plasma.coil_current:6.0f} A    "
    f"{reference.plasma.efficiency:5.1%}"
)
energy = fluid.energy_balance(plasma)
print("fluid energy balance (W):", {k: round(v) for k, v in energy.items()})

force_r, force_z = result.solution.lorentz_force
ambipolar = plasma.mean(plasma.electron_temperature) / 0.01  # T_e over a 1 cm skin
print(
    f"Lorentz force as a field: up to {np.max(np.abs(force_r)) / ELEMENTARY_CHARGE:.0f} V/m "
    f"inward, against T_e / delta = {ambipolar:.0f} V/m (delta = 1 cm)"
)

# --- figures ------------------------------------------------------------------------------
r, z = grid.x, grid.y
keep_r, keep_z = r < DRIVER_RADIUS, (z > 0) & (z < DRIVER_END)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
panels = [
    (plasma.electron_density, "n_e in its field (m⁻³)"),
    (plasma.electron_temperature, "T_e (eV)"),
    (force_r / ELEMENTARY_CHARGE / 1e3, "Lorentz force F_r / e (kV/m)"),
]
for ax, (values, title) in zip(axes, panels, strict=True):
    mesh = ax.pcolormesh(
        z[keep_z] * 100, r[keep_r] * 100, values[np.ix_(keep_r, keep_z)], shading="auto"
    )
    ax.set_xlabel("z (cm)")
    ax.set_title(title, loc="left")
    fig.colorbar(mesh, ax=ax)
axes[0].set_ylabel("r (cm)")
fig.tight_layout()

fig, ax = plt.subplots(figsize=(6, 3.5))
j = int(np.argmin(np.abs(z - 0.07)))
ax.plot(
    r[keep_r] * 100, plasma.electron_density[keep_r, j], label="heated by its field"
)
ax.plot(
    r[keep_r] * 100, uniform.electron_density[keep_r, j], "--", label="uniform heating"
)
ax.set_xlabel("r (cm), at the coil's centre z = 7 cm")
ax.set_ylabel("n_e (m⁻³)")
ax.set_title("the skin layer heats the edge", loc="left")
ax.legend(frameon=False)
fig.tight_layout()
