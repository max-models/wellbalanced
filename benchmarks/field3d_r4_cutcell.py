"""The slotted cylinder again: surface-impedance conductors on cut cells against staircases.

Step R4's slotted-cylinder test (``field3d_r4.py``, ``slotted_cylinder_case.CASE``) with
the tube as an :class:`~wellbalanced.spatial.ImpedanceConductor3D` (Leontovich's
``E_t = Z_s H x n`` on its surface), its cells cut conformally (simgeometry's
``yee_fractions``) or staircased, against NGSolve with the same surface impedance on
curved meshes fitted to the tube and its slits (``slotted_cylinder_ngsolve_sibc.py``),
which isolates the discretisation; and against NGSolve's volume conductor
(``slotted_cylinder_ngsolve_sibc.py volume2d`` and ``slotted_cylinder_ngsolve.py``), the
model error of the surface impedance. Parts:

- ``tube2d``: the infinite tube (z-periodic quarter, two cells long), no slits or
  through slits, at 50 kHz, 500 kHz and 5 MHz (skin depths 2.25, 0.71 and 0.225 mm in
  the 6 mm wall); also the staircased volume conductor of ``field3d_r4.py`` at 500 kHz.
- ``tube3d``: the finite tube (one eighth), no, closed or through slits, at 500 kHz
  and 50 kHz; the iterative solver from 24 cells on.

The grids are those on which the coil's 45 mm half-side and 80 mm half-length fall on
nodes (cells along 60 mm a multiple of 4 or 12): elsewhere the staircased coil changes
its area, and with it the field, by up to 20 % (the 3.33 and 2 mm grids of
``field3d_r4.py`` have this error). Run
``python benchmarks/field3d_r4_cutcell.py PART [CELLS]`` (``OMP_NUM_THREADS=2``, e.g.
``tube3d 12,24,36``, or ``tube3d 48 5e5 "cut cells"`` for frequencies and methods); rows
are added to ``benchmarks/field3d_r4_cutcell.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from wellbalanced.spatial import (
    Conductor3D,
    EdgeGrid,
    ImpedanceConductor3D,
    InductiveField3D,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from field3d_r4 import coil, tube  # noqa: E402
from slotted_cylinder_case import CASE  # noqa: E402

D, N = "dirichlet", "natural"
PATH = HERE / "field3d_r4_cutcell.json"


def references(name: str) -> list[dict]:
    """NGSolve's rows of ``slotted_cylinder_ngsolve_{name}.json`` (empty if not run)."""
    path = HERE / f"slotted_cylinder_ngsolve_{name}.json"
    return json.loads(path.read_text())["results"] if path.exists() else []


def best(rows: list[dict], slits: str, frequency: float) -> dict | None:
    """The finest mesh and highest order of a variant."""
    rows = [r for r in rows if r["slits"] == slits and r["frequency"] == frequency]
    return max(rows, key=lambda r: (r["mesh"], r["order"])) if rows else None


def run(model: InductiveField3D, solver: str) -> tuple[object, float]:
    """The solution at 1 A and the seconds the solve took."""
    start = time.perf_counter()
    solution = model.solve(1.0, solver=solver)
    return solution, time.perf_counter() - start


def add(rows: list[dict], row: dict, against: dict[str, dict | None]) -> None:
    """Append ``row`` with its errors against the references and print it."""
    for key, ref in against.items():
        if ref is not None:
            row[f"loss_error_{key}"] = row["loss"] / ref["loss"] - 1
            row[f"inductance_error_{key}"] = row["inductance"] / ref["inductance"] - 1
    rows.append(row)
    errors = " ".join(
        f"{k[11:]} {v:+.5f}" for k, v in row.items() if k.startswith("loss_error_")
    )
    print(
        f"{row['part']} {row['slits']} {row['frequency']:.0e} Hz {row['method']}"
        f" h={1e3 * row['h']:.3f} mm: loss {row['loss']:.5e} ({errors})"
        f" {row['unknowns']} unknowns, {row['seconds']:.0f} s",
        flush=True,
    )


def tube2d(cells_list: list[int]) -> list[dict]:
    """The z-periodic quarter: per metre of the whole cross-section."""
    c = CASE
    sibc, volume = references("sibc_2d"), references("volume2d")
    rows: list[dict] = []
    for slits in ("none", "through"):
        for frequency in (5e4, 5e5, 5e6):
            against = {
                "sibc": best(sibc, slits, frequency),
                "volume": best(volume, slits, frequency),
            }
            for cells in cells_list:
                h = c["box"] / cells
                grid = EdgeGrid(
                    (0, 0, 0),
                    (c["box"], c["box"], 2 * h),
                    (cells, cells, 2),
                    (D, D, "periodic"),
                )
                scale = 4 / (2 * h)
                methods = ["cut cells", "staircase"]
                if frequency == 5e5:
                    methods.append("volume")
                for method in methods:
                    shape = tube(slits, 1.0)
                    if method == "volume":
                        model = InductiveField3D(
                            grid,
                            frequency=frequency,
                            coils=[coil(1.0)],
                            conductors=[Conductor3D([shape], c["conductivity"])],
                            regularization=c["regularization"],
                        )
                    else:
                        model = InductiveField3D(
                            grid,
                            frequency=frequency,
                            coils=[coil(1.0)],
                            impedance_conductors=[
                                ImpedanceConductor3D([shape], c["conductivity"])
                            ],
                            cut_cells=method == "cut cells",
                            regularization=c["regularization"],
                        )
                    solution, seconds = run(model, "direct")
                    row = {
                        "part": "tube2d",
                        "slits": slits,
                        "frequency": frequency,
                        "method": method,
                        "cells": cells,
                        "h": h,
                        "loss": scale * solution.loss,
                        "inductance": scale * solution.inductance,
                        "unknowns": int(model.free_edges.sum()),
                        "seconds": seconds,
                    }
                    add(rows, row, against)
    return rows


def tube3d(
    cells_list: list[int],
    frequencies: tuple[float, ...] = (5e5, 5e4),
    methods: tuple[str, ...] = ("cut cells", "staircase"),
) -> list[dict]:
    """The finite tube, one eighth, both kinds of cells."""
    c = CASE
    sibc, volume = references("sibc_3d"), references("3d")
    rows: list[dict] = []
    for cells in cells_list:
        h = c["box"] / cells
        grid = EdgeGrid(
            (0, 0, 0),
            (c["box"], c["box"], c["box_z"]),
            (cells, cells, round(c["box_z"] / h)),
            (D, D, (N, D)),
        )
        solver = "direct" if cells < 24 else "iterative"
        for frequency in frequencies:
            for slits in ("none", "closed", "through"):
                against = {
                    "sibc": best(sibc, slits, frequency),
                    "volume": best(volume, slits, frequency),
                }
                for method in methods:
                    setup = time.perf_counter()
                    model = InductiveField3D(
                        grid,
                        frequency=frequency,
                        coils=[coil(c["coil_half_length"])],
                        impedance_conductors=[
                            ImpedanceConductor3D(
                                [tube(slits, c["tube_half_length"])], c["conductivity"]
                            )
                        ],
                        cut_cells=method == "cut cells",
                        regularization=c["regularization"],
                    )
                    setup = time.perf_counter() - setup
                    solution, seconds = run(model, solver)
                    row = {
                        "part": "tube3d",
                        "slits": slits,
                        "frequency": frequency,
                        "method": method,
                        "cells": cells,
                        "h": h,
                        "loss": solution.loss,
                        "inductance": solution.inductance,
                        "unknowns": int(model.free_edges.sum()),
                        "solver": solver,
                        "iterations": getattr(model, "last_iterative", None)
                        and model.last_iterative.iterations,
                        "setup_seconds": setup,
                        "seconds": seconds,
                    }
                    add(rows, row, against)
    return rows


PARTS = {"tube2d": tube2d, "tube3d": tube3d}
DEFAULT_CELLS = {"tube2d": [12, 24, 36, 48, 96], "tube3d": [12, 24, 36]}


def main(arguments: list[str]) -> None:
    """Run one part on the given grids and merge its rows into the JSON file."""
    part = arguments[0] if arguments else "tube2d"
    cells = (
        [int(n) for n in arguments[1].split(",")]
        if len(arguments) > 1
        else DEFAULT_CELLS[part]
    )
    options = {}
    if len(arguments) > 2:  # tube3d: the frequencies [Hz] and the methods
        options["frequencies"] = tuple(float(f) for f in arguments[2].split(","))
    if len(arguments) > 3:
        options["methods"] = tuple(arguments[3].split(","))
    new = PARTS[part](cells, **options)
    old = json.loads(PATH.read_text())["results"] if PATH.exists() else []
    keys = {
        (r["part"], r["slits"], r["frequency"], r["method"], r["cells"]) for r in new
    }
    kept = [
        r
        for r in old
        if (r["part"], r["slits"], r["frequency"], r["method"], r["cells"]) not in keys
    ]
    PATH.write_text(
        json.dumps({"case": CASE, "results": kept + new}, indent=1, default=float)
        + "\n"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
