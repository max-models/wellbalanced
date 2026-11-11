"""Settled 1D state and its linear spectrum.

Relaxes the y-uniform 1D state of Boeuf's DC magnetron for a long time (the gas time is 50 us,
so about 100 us) with a chosen anomalous frequency and anode closure, saves it, and prints
the Jacobian eigenvalues per azimuthal mode with the responses frozen one at a time
(none, ``temperature``, ``ionization``, ``ionization_frequency``, the consistent form).
It runs about 15 minutes at ``--nx 24``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from wellbalanced import DriftFluid2D
from wellbalanced.driftfluid2d import exponential_field

BOLTZMANN = 1.380649e-23
GAP, PERIOD, PRESSURE, B_MAX = 0.02, 0.08, 0.6, 0.12
FROZEN = {
    "full": (),
    "T_e frozen": ("temperature",),
    "ionization frozen": ("ionization",),
    "consistent": ("ionization_frequency",),
}


def build(nx: int, ny: int, **options) -> DriftFluid2D:
    """The magnetron plane of the gallery's study 07."""
    return DriftFluid2D(
        length_x=GAP,
        length_y=PERIOD,
        nx=nx,
        ny=ny,
        voltage=260.0,
        magnetic_field=exponential_field(B_MAX, 0.3 * GAP),
        gas_density=PRESSURE / (BOLTZMANN * 300.0),
        electron_temperature=None,
        cathode_current_density=0.03,
        refill_rate=2.0e4,
        **options,
    )


def relax(model: DriftFluid2D, duration: float):
    """Step the 1D state to ``duration`` seconds, printing the anode current."""
    state = model.initial_state(1.0e15, electron_temperature=3.0)
    steps, step, start = 0, 0.0, time.time()
    while state.time < duration:
        if steps % 20 == 0:
            step = model.suggest_timestep(state)
            if steps % 2000 == 0:
                gx, _ = model.electron_flux(state)
                print(
                    f"t = {state.time * 1e6:6.1f} us  j_anode = "
                    f"{1.602176634e-19 * gx[-1, 0]:.2f} A/m2  ({time.time() - start:.0f} s)",
                    flush=True,
                )
        state = model.step(state, min(step, duration - state.time + 1e-15))
        steps += 1
    return state


def main() -> None:
    """Relax, save and print the spectra."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--relax", type=float, default=100e-6, help="seconds")
    parser.add_argument("--bohm", type=float, default=0.3)
    parser.add_argument(
        "--anode", choices=["none", "thermal", "sheath"], default="none"
    )
    parser.add_argument("--modes", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--save", default="settled_state.npz")
    args = parser.parse_args()
    options = {"bohm": args.bohm}
    if args.anode != "none":
        options[
            "anode_" + ("thermal_flux" if args.anode == "thermal" else "sheath")
        ] = True
    model = build(args.nx, 4, **options)
    state = relax(model, args.relax)
    ne = state.electron_density[:, 0]
    print(
        f"L_n at 1.8 cm: {1e3 / np.gradient(np.log(ne), model.x)[np.argmin(abs(model.x - 0.018))]:.2f} mm"
    )
    np.savez(
        args.save,
        x=model.x,
        ne=ne,
        te=state.electron_temperature[:, 0],
        field=model.electric_field(state)[0][:, 0],
    )
    wide = build(args.nx, 16, **options)
    uniform = state.copy()
    uniform.time = 0.0
    for name in (
        "electron_density",
        "ion_density",
        "electron_temperature",
        "gas_density",
        "potential",
    ):
        setattr(uniform, name, np.repeat(getattr(uniform, name)[:, :1], 16, axis=1))
    uniform.ion_velocity = np.repeat(uniform.ion_velocity[:, :, :1], 16, axis=2)
    print("mode  " + "  ".join(f"{n:>22}" for n in FROZEN))
    for mode in args.modes:
        row = []
        for frozen in FROZEN.values():
            gamma, _, speed = wide.linear_modes(uniform, mode, frozen=frozen).leading()
            row.append(f"{gamma:9.2e} {speed / 1e3:+7.1f} km/s")
        print(f"{mode:>4}  " + "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
