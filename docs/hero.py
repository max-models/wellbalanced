"""Draw the animated figure of the docs home page: a 1D argon discharge settling down.

The 1D fluid model (``Fluid1D``) of the argon discharge of ``examples/argon_discharge_1d.py``
(5 cm between two grounded walls, uniform heating), started from a small, lopsided seed and
stepped in time to its steady state. The electron density (top) and the potential (bottom) at
successive times are drawn as two families of lines that light up in turn, the steady state in
the accent colour, with ions running from the middle of the plasma to the walls. Rerun with
``make docs-hero`` (needs ``plasma_plots.hero``); it writes ``docs/src/assets/hero.svg``.
"""

from pathlib import Path

import numpy as np
from plasma_plots.hero import HeroFigure
from plasmacoll import CrossSection
from plasmacoll.constants import ELECTRON_MASS

from wellbalanced import ELECTRON, Fluid1D, FluidSpecies, Reaction, Species

OUTPUT = Path(__file__).parent / "src" / "assets" / "hero.svg"
PERIOD = 12.0  # seconds for the evolution to play once
LENGTH = 0.05

ionization = Reaction.from_cross_section(
    ("e", "Ar"),
    ("e", "e", "Ar+"),
    CrossSection.constant(2.0e-20, threshold=15.76),
    ELECTRON_MASS,
    electron_energy_loss=15.76,
    name="ionization",
)
model = Fluid1D(
    species=[
        FluidSpecies(ELECTRON, mobility=1.0e3),
        FluidSpecies(Species.atomic("Ar+", 39.948, 1), mobility=10.0),
    ],
    reactions=[ionization],
    length=LENGTH,
    cells=120,
    fixed={"Ar": 1.0e21},
    heating=2.0e4,  # W/m^3
)
seed = 1.0e13 * np.exp(-(((model.x - 0.3 * LENGTH) / (0.08 * LENGTH)) ** 2)) + 1.0e11
u = model.pack({"e": seed, "Ar+": seed}, 3.0, 0.0)
states, time, dt = [], 0.0, 1.0e-8
while dt < 1.0e-2:
    u, ok = model.step(u, dt)
    if not ok:
        raise RuntimeError("a time step failed")
    time += dt
    states.append(model.state(u, time=time))
    dt *= 1.6
states = states[:: max(1, len(states) // 14)]
states.append(model.steady_state(u))

x = model.x / LENGTH * 6.0 + 3.6  # the discharge spans x = 3.6 ... 9.6 of the picture
density = np.array([s.densities["e"] for s in states])
potential = np.array([s.potential for s in states])
density = density / density.max()
potential = potential / potential.max()

fig = HeroFigure(xlim=(0.0, 10.0), ylim=(-1.15, 1.15))
count = len(states)
for i in range(count):
    final = i == count - 1
    glow = np.full(count + 1, 0.12)
    glow[i] = 0.9
    glow[-1] = glow[0]
    for values, top in ((density, 0.08), (potential, -1.08)):
        line = fig.line(
            x,
            top + values[i] * 0.98,
            tone="accent" if final else "muted",
            width=2.4 if final else 1.2,
            opacity=0.9 if final else 0.35,
            title="steady state" if final else f"t = {states[i].time * 1e6:.2g} µs",
        )
        if not final:
            line.animate("opacity", glow, PERIOD)
peak = int(np.argmax(density[-1]))
rng = np.random.default_rng(2)
for side in (slice(peak, None), slice(peak, None, -1)):
    for _ in range(4):
        duration = rng.uniform(3.0, 6.0)
        fig.tracer(
            x[side],
            duration=duration,
            offset=rng.uniform(0.0, duration),
            r=3.0,
            y=0.08 + density[-1][side] * 0.98,
        )
fig.save(OUTPUT)
