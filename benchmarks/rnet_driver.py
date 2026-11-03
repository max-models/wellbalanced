"""R_net of an RF driver: the losses of its coil and Faraday shield, computed in 3D.

Step R4's output. The network resistance ``R_net`` in ``eta = P_plasma / (P_plasma +
1/2 R_net I^2)`` is measured without plasma (0.6 Ohm at BUG; 1.28 Ohm for ELISE's two
upper drivers in series, Zielke et al., NIBS'22, arXiv:2211.10143); at BUG 25 % of it
is the coil's Joule heating and 75 % the Faraday shield's eddy currents (Zielke et al.,
J. Phys. D 54, 155202, 2021). Here it is the resistance the coil's terminals see without
plasma, ``Re Z = 2 P / |I|^2`` from `InductiveField3D` with:

- the coil: a copper `simgeometry.Helix` with two radial leads and an axial return bar,
  all `ImpedanceConductor3D`, driven by a `Feed3D` across a gap in the bar (its own
  Joule loss and proximity effect included);
- the Faraday shield: a copper tube with axial slits closed at both ends, on the back
  plate;
- the housing: a perfectly conducting box (the back plate at z = 0, side walls, a wall
  beyond the driver's open end standing in for the expansion chamber), lossless;
- optionally an electromagnetic shield (EMS, ELISE): a copper tube around the coil
  with four axial support bars to the back plate.

Geometry (BUG; ``*`` marks an assumption, to be replaced by IPP's drawings): plasma
radius 0.1185 m, driver length 0.17 m (Zielke et al.); the shield's inner radius at the
plasma, its wall 8 mm* thick (real shields are thinner: a wall below two cells leaks
the field, see below), 24* slits 2 mm* wide ending 10 mm* before its ends; six turns of
radius 0.1385 m (2 cm outside the plasma*, as ``illustrative_hydrogen_driver``), pitch
2 cm*, an 8 mm* copper tube; copper at 5.8e7 S/m, 1 MHz.

Resolution: the solver needs about two cells across every metal part (an edge longer
than the wall reaches through it, and a feed needs its terminals joined through the
metal), and the losses converge at first order in h at the 3D corners (the slits' ends;
``docs/.../field-3d.md``). The grids here (5, 4 and 3.33 mm) are the reduced ones; the
converged R_net, with the real shield's thickness (about 2 mm) and h about 1 mm (some
10^8 edges), needs a production run on mpiarray and cunumpy.

Run ``python benchmarks/rnet_driver.py VARIANT H_MM[,H_MM...]`` with ``VARIANT`` one
of ``coil`` (the coil alone), ``bug`` (coil and shield), ``elise`` (with the EMS at
4 cm); rows are added to ``benchmarks/rnet_driver.json``. Needs simgeometry with
``Helix``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simgeometry as sg

from wellbalanced.spatial import (
    EdgeGrid,
    Feed3D,
    ImpedanceConductor3D,
    InductiveField3D,
    inductive3d,
)
from wellbalanced.spatial.cutcell import CutCells

HERE = Path(__file__).resolve().parent
PATH = HERE / "rnet_driver.json"
CACHE = HERE / ".rnet_cache"  # the cut cells, by conductor and grid (not committed)
COPPER = 5.8e7
GEOMETRY = [""]  # the driver's parameters, part of the cache's key
# COCG's relative residual: the slow part of the residual is the gauge's (gradients),
# which leaves the curl and the impedance unchanged (the coax of test_feed3d.py: 1e-4
# gives the resistance to 2e-7 of the direct solve, the driver coil at 4 mm 1e-5 and
# 1e-8 the same to 4 digits)
TOLERANCE = 1e-4


def cached_cut_cells(original):
    """``cut_cells`` with the results kept on disk: they take minutes on these grids."""

    def cut_cells(grid, shapes, **options):
        key = hashlib.sha1(
            repr(
                (
                    GEOMETRY[0],
                    tuple(grid.lower),
                    tuple(grid.upper),
                    grid.cells,
                    [type(shape).__name__ for shape in shapes],
                    options,
                )
            ).encode()
        ).hexdigest()[:16]
        path = CACHE / f"{key}.npz"
        if path.exists():
            data = np.load(path)
            return CutCells(data["edge_air"], data["face_air"], data["face_width"])
        start = time.perf_counter()
        cells = original(grid, shapes, **options)
        print(f"  cut cells in {time.perf_counter() - start:.0f} s", flush=True)
        CACHE.mkdir(exist_ok=True)
        np.savez(path, **asdict(cells))
        return cells

    return cut_cells


inductive3d.cut_cells_of = cached_cut_cells(inductive3d.cut_cells_of)


@dataclass(frozen=True)
class Driver:
    """The driver's geometry [m] and the frequency [Hz]; ``*`` in the module: assumed."""

    plasma_radius: float = 0.1185
    length: float = 0.17
    shield_wall: float = 0.008
    slits: int = 24
    slit_width: float = 0.002
    slit_margin: float = 0.010
    coil_radius: float = 0.1385
    turns: int = 6
    pitch: float = 0.02
    wire_radius: float = 0.004
    lead_radius: float = 0.155  # the axial return bar's distance from the axis
    lead_side: float = 0.010  # the leads' square cross-section
    feed_gap: float = 0.010
    ems_distance: float | None = None  # from the coil's centreline
    ems_wall: float = 0.008
    box_half: float = 0.17
    box_end: float = 0.22
    frequency: float = 1.0e6


def coil_shapes(d: Driver) -> tuple[list, Feed3D]:
    """The helix, its two radial leads and the return bar with its gap; the feed."""
    start = 0.5 * (d.length - d.pitch * d.turns)
    end = start + d.pitch * d.turns
    helix = sg.Helix((0, 0, start), d.coil_radius, d.pitch, d.turns, d.wire_radius)
    s = d.lead_side
    radial = d.lead_radius + s / 2 - d.coil_radius
    centre = d.coil_radius + radial / 2
    leads = [
        sg.Cuboid((centre, 0, z), (radial, s, s)) for z in (start, end)
    ]  # from the helix's ends (at +x) out to the bar
    middle = 0.5 * (start + end)
    bar = sg.Cuboid((d.lead_radius, 0, middle), (s, s, end - start + s)) - sg.Cuboid(
        (d.lead_radius, 0, middle), (2 * s, 2 * s, d.feed_gap)
    )
    reach = d.feed_gap / 2 + s / 2
    feed = Feed3D(
        (d.lead_radius, 0, middle - reach),
        (d.lead_radius, 0, middle + reach),
        name="feed",
    )
    return [helix, *leads, bar], feed


def shield_shape(d: Driver):
    """The slotted tube on the back plate (z = 0), closed slits."""
    inner, outer = d.plasma_radius, d.plasma_radius + d.shield_wall
    shape = sg.Tube((0, 0, 0), (0, 0, d.length), inner, outer)
    slit = sg.Cuboid(
        (0.5 * (inner + outer), 0, d.length / 2),
        (3 * d.shield_wall, d.slit_width, d.length - 2 * d.slit_margin),
    )
    return shape - sg.PolarRepeat(slit, d.slits)


def ems_shapes(d: Driver) -> list:
    """The EMS tube around the coil and four axial support bars to the back plate."""
    inner = d.coil_radius + d.ems_distance
    tube = sg.Tube((0, 0, 0), (0, 0, d.length), inner, inner + d.ems_wall)
    side = d.ems_wall
    supports = [
        sg.Cuboid((0, 0, d.length / 2), (side, side, d.length)).translated(
            (
                (inner + side / 2) * np.cos(np.pi / 4 + k * np.pi / 2),
                (inner + side / 2) * np.sin(np.pi / 4 + k * np.pi / 2),
                0.0,
            )
        )
        for k in range(4)
    ]
    return [tube, *supports]


def model(d: Driver, h: float, variant: str) -> InductiveField3D:
    """The field of the driver ``variant`` on a grid of spacing ``h``."""
    cells = round(2 * d.box_half / h)
    grid = EdgeGrid(
        (-d.box_half, -d.box_half, 0.0),
        (d.box_half, d.box_half, d.box_end),
        (cells, cells, round(d.box_end / h)),
    )
    coil, feed = coil_shapes(d)
    conductors = [ImpedanceConductor3D(coil, COPPER, "coil")]
    if variant in ("bug", "elise"):
        conductors.append(ImpedanceConductor3D([shield_shape(d)], COPPER, "shield"))
    if variant == "elise":
        conductors.append(ImpedanceConductor3D(ems_shapes(d), COPPER, "ems"))
    return InductiveField3D(
        grid,
        frequency=d.frequency,
        feeds=[feed],
        impedance_conductors=conductors,
        cut_cell_samples=16,
    )


def run(variant: str, h: float) -> dict:
    """One row: R_net and its parts, the inductance, the cost."""
    d = Driver(ems_distance=0.04) if variant == "elise" else Driver()
    if variant == "elise":
        d = Driver(ems_distance=0.04, box_half=0.20)
    GEOMETRY[0] = repr(asdict(d))
    setup = time.perf_counter()
    field = model(d, h, variant)
    setup = time.perf_counter() - setup
    start = time.perf_counter()
    solution = field.solve(
        1.0, solver="iterative", tolerance=TOLERANCE, max_iterations=4000
    )
    seconds = time.perf_counter() - start
    parts = {name: 2 * p for name, p in solution.conductor_losses.items()}  # Ohm at 1 A
    row = {
        "variant": variant,
        "h": h,
        "r_net": solution.resistance,
        "resistances": parts,
        "inductance": solution.inductance,
        "power_balance": solution.loss / solution.source_power.real - 1,
        "unknowns": int(field.free_edges.sum()),
        "iterations": field.last_iterative.iterations,
        "setup_seconds": setup,
        "seconds": seconds,
        "tolerance": TOLERANCE,
        "driver": asdict(d),
    }
    print(
        f"{variant} h={1e3 * h:.2f} mm: R_net {row['r_net']:.4f} Ohm"
        f" ({', '.join(f'{k} {v:.4f}' for k, v in parts.items())}),"
        f" L {1e6 * row['inductance']:.3f} uH, {row['unknowns']} unknowns,"
        f" {row['iterations']} iterations, {setup:.0f} + {seconds:.0f} s",
        flush=True,
    )
    return row


def main(arguments: list[str]) -> None:
    """Run one variant on the given grids and merge its rows into the JSON file."""
    variant = arguments[0] if arguments else "bug"
    spacings = [
        float(v) * 1e-3
        for v in (arguments[1] if len(arguments) > 1 else "5").split(",")
    ]
    new = [run(variant, h) for h in spacings]
    old = json.loads(PATH.read_text())["results"] if PATH.exists() else []
    keys = {(r["variant"], r["h"]) for r in new}
    kept = [r for r in old if (r["variant"], r["h"]) not in keys]
    rows = sorted(kept + new, key=lambda r: (r["variant"], -r["h"]))
    PATH.write_text(json.dumps({"results": rows}, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
