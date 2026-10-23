"""The evidence for the 3D field's discretisation: staircased edges against references.

The three tests of step R4 of the RF-driver plan with
:class:`~wellbalanced.spatial.InductiveField3D` (edges of a uniform Cartesian grid,
conductors staircased: an edge conducts if its midpoint is inside), under grid
refinement:

- ``solenoid``: a finite solenoid (length 0.21 m, radius 0.05 m) against Nagaoka's
  coefficient, in boxes of growing size with perfectly conducting (``A x n = 0``) and
  magnetic (``H x n = 0``) walls, which bracket the free-space inductance.
- ``driver``: a rectangular loop of growing length around a conducting column against
  the z-invariant field (the z-periodic grid, equal to ``InductiveField2D``'s to the
  gauge's ``eps``, ``tests/unit/test_inductive3d.py``): the loss per metre in the
  middle and in total.
- ``tube2d``: an infinite tube in an infinite solenoid (z-periodic), without slits
  against the Bessel solution and with four slits along its length against NGSolve
  (``slotted_cylinder_ngsolve.py 2d``); midpoint and sub-cell conductances
  (``samples=3``).
- ``tube3d``: the finite tube with four slits, none, closed or through, against NGSolve
  (``slotted_cylinder_ngsolve.py 3d``).

Run ``python benchmarks/field3d_r4.py [parts]`` with wellbalanced and simgeometry
installed (``OMP_NUM_THREADS=2``; ``tube3d`` takes about 15 minutes, the rest a few
minutes). It writes ``benchmarks/field3d_r4.json`` and prints the tables of
``docs/src/content/docs/development/field-3d.md``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import simgeometry as sg
from scipy.constants import mu_0 as MU_0

from wellbalanced.driver import nagaoka_coefficient
from wellbalanced.spatial import Coil3D, Conductor3D, EdgeGrid, InductiveField3D
from wellbalanced.tests.unit.test_inductive3d import tube_in_box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slotted_cylinder_case import CASE  # noqa: E402

D, N = "dirichlet", "natural"


def timed(model: InductiveField3D):
    """The solution at 1 A and the seconds it took."""
    start = time.perf_counter()
    solution = model.solve(1.0)
    return solution, time.perf_counter() - start


# --- (1) the solenoid ---------------------------------------------------------------


def solenoid() -> list[dict]:
    """One eighth of a solenoid in boxes 2 to 5 radii wide, both kinds of walls."""
    radius, h, turns, half_length = 0.05, 0.01, 100.0, 0.105
    shape = sg.Cylinder((0, 0, -half_length), (0, 0, half_length), radius)
    rows = []
    for scale in (2, 3, 4, 5):
        cells = (5 * scale, 5 * scale, round((half_length + (scale - 1) * radius) / h))
        upper = tuple(n * h for n in cells)
        row: dict = {"box_radii": scale, "cells": cells}
        for name, wall in (("conducting", D), ("magnetic", N)):
            grid = EdgeGrid((0, 0, 0), upper, cells, ((D, wall), (D, wall), (N, wall)))
            model = InductiveField3D(
                grid, frequency=1e3, coils=[Coil3D(shape, (0, 0, 1), turns)]
            )
            solution, seconds = timed(model)
            row[name] = 8 * solution.inductance
            row["seconds"] = row.get("seconds", 0.0) + seconds
        # the coil the grid sees: its z-faces (each plane of nodes) and their planes
        faces = (grid.face_axes() == 2) & grid.face_mask(shape)
        planes = np.unique(grid.face_centres()[2][faces]).size
        area = 4 * faces.sum() / planes * h * h
        length = (2 * planes - 1) * h
        row["nagaoka"] = (
            nagaoka_coefficient(2 * np.sqrt(area / np.pi), length)
            * MU_0
            * turns**2
            * area
            * length
        )
        row["area_ratio"] = area / (np.pi * radius**2)
        rows.append(row)
        print(
            f"solenoid box {scale} radii: conducting {row['conducting'] / row['nagaoka'] - 1:+.4f}"
            f" magnetic {row['magnetic'] / row['nagaoka'] - 1:+.4f} ({row['seconds']:.1f} s)",
            flush=True,
        )
    return rows


# --- (3) the z-invariant driver ---------------------------------------------------------


def driver() -> list[dict]:
    """A loop around a conducting column (a quarter-eighth), against its z-invariant field."""
    h, width_x, width_y = 0.01, 0.15, 0.10
    d, w, radius, frequency = 0.06, 0.055, 0.0413, 1e6
    sigma = 128.24 - 40.29j  # a cold plasma's, n_e = 1e17 m^-3, nu = 2e7 1/s, 1 MHz

    def model(half_length: float | None) -> InductiveField3D:
        periodic = half_length is None
        z_upper = 2 * h if periodic else half_length + 0.1
        cells = (round(width_x / h), round(width_y / h), round(z_upper / h))
        z_boundary = "periodic" if periodic else D
        grid = EdgeGrid(
            (0, 0, 0), (width_x, width_y, z_upper), cells, (D, (N, D), z_boundary)
        )
        length = 10.0 if periodic else 2 * half_length
        loop = Coil3D(
            sg.Cuboid((0, 0, 0), (2 * d, 2 * w, length)), (0, 1, 0), 1 / (2 * w)
        )

        def conductivity(p):
            column = (p[0] ** 2 + p[1] ** 2 < radius**2) & (np.abs(p[2]) < length / 2)
            return np.where(column, sigma, 0.0)

        return InductiveField3D(
            grid, frequency=frequency, coils=[loop], conductivity=conductivity
        )

    reference, _ = timed(model(None))
    per_metre = reference.loss / reference.grid.upper[2]
    rows = []
    for half_length in (0.105, 0.205, 0.405, 0.805):
        solution, seconds = timed(model(half_length))
        grid = solution.grid
        density = (
            0.5
            * solution.omega**2
            * np.real(solution.edge_conductance)
            * np.abs(solution.potential) ** 2
        )
        middle = np.abs(grid.edge_midpoints()[2]) < 0.05
        row = {
            "half_length": half_length,
            "total_per_metre": solution.loss / half_length / per_metre - 1,
            "middle_per_metre": density[middle].sum() / 0.05 / per_metre - 1,
            "seconds": seconds,
        }
        rows.append(row)
        print(
            f"driver half-length {half_length}: total/length {row['total_per_metre']:+.4f}"
            f" middle {row['middle_per_metre']:+.4f} ({seconds:.1f} s)",
            flush=True,
        )
    return rows


# --- (2) the slotted cylinder ---------------------------------------------------------


def tube(slits: str, z_half: float) -> object:
    """The tube of ``CASE``, ``2 z_half`` long, with its slits."""
    c = CASE
    shape = sg.Tube(
        (0, 0, -z_half), (0, 0, z_half), c["inner_radius"], c["outer_radius"]
    )
    if slits != "none":
        reach = 4 * z_half if slits == "through" else c["slit_half_length"]
        w, big = c["slit_width"], 4 * c["box"]
        shape = (
            shape
            - sg.Cuboid((0, 0, 0), (big, w, 2 * reach))
            - sg.Cuboid((0, 0, 0), (w, big, 2 * reach))
        )
    return shape


def coil(z_half: float) -> Coil3D:
    """The square solenoid of ``CASE``, ``2 z_half`` long."""
    s = CASE["coil_half_side"]
    return Coil3D(
        sg.Cuboid((0, 0, 0), (2 * s, 2 * s, 2 * z_half)),
        (0, 0, 1),
        CASE["turns_per_length"],
    )


def reference(name: str, slits: str, frequency: float) -> dict | None:
    """NGSolve's result on its finest mesh and highest order, if it has been run."""
    path = HERE / f"slotted_cylinder_ngsolve_{name}.json"
    if not path.exists():
        return None
    rows = [
        r
        for r in json.loads(path.read_text())["results"]
        if r["slits"] == slits and r["frequency"] == frequency
    ]
    return max(rows, key=lambda r: (r["mesh"], r["order"]))


def tube2d() -> list[dict]:
    """The infinite tube, a quarter of the cross-section, z-periodic, two cells long."""
    c = CASE
    rows = []
    for slits in ("none", "through"):
        for frequency in c["frequencies"]:
            if slits == "none":
                loss, impedance = tube_in_box(
                    frequency,
                    c["conductivity"],
                    c["inner_radius"],
                    c["outer_radius"],
                    c["turns_per_length"],
                    (2 * c["coil_half_side"]) ** 2,
                    (2 * c["box"]) ** 2,
                )
                ref = {
                    "loss": loss,
                    "inductance": impedance.imag / (2 * np.pi * frequency),
                    "source": "Bessel",
                }
            else:
                ref = reference("2d", slits, frequency)
                if ref is None:
                    continue
                ref = {**ref, "source": "NGSolve"}
            for cells in (24, 48, 96, 192):
                for samples in (1, 3):
                    h = c["box"] / cells
                    grid = EdgeGrid(
                        (0, 0, 0),
                        (c["box"], c["box"], 2 * h),
                        (cells, cells, 2),
                        (D, D, "periodic"),
                    )
                    model = InductiveField3D(
                        grid,
                        frequency=frequency,
                        coils=[coil(1.0)],
                        conductors=[Conductor3D([tube(slits, 1.0)], c["conductivity"])],
                        samples=samples,
                        regularization=c["regularization"],
                    )
                    solution, seconds = timed(model)
                    loss = 4 * solution.loss / (2 * h)
                    inductance = 4 * solution.inductance / (2 * h)
                    row = {
                        "slits": slits,
                        "frequency": frequency,
                        "cells": cells,
                        "h": h,
                        "samples": samples,
                        "loss": loss,
                        "inductance": inductance,
                        "loss_error": loss / ref["loss"] - 1,
                        "inductance_error": inductance / ref["inductance"] - 1,
                        "reference": ref["source"],
                        "unknowns": int(grid.free_edges.sum()),
                        "seconds": seconds,
                    }
                    rows.append(row)
                    print(
                        f"tube2d {slits} {frequency:.0f} Hz h={1e3 * h:.3f} mm samples={samples}:"
                        f" loss {row['loss_error']:+.4f} L {row['inductance_error']:+.5f}"
                        f" ({seconds:.1f} s)",
                        flush=True,
                    )
    return rows


def tube3d() -> list[dict]:
    """The finite tube, one eighth, with none, closed or through slits."""
    c = CASE
    rows = []
    for cells in (12, 18, 24, 30):
        h = c["box"] / cells
        nz = round(c["box_z"] / h)
        grid = EdgeGrid(
            (0, 0, 0),
            (c["box"], c["box"], c["box_z"]),
            (cells, cells, nz),
            (D, D, (N, D)),
        )
        for slits in ("none", "closed", "through"):
            for frequency in c["frequencies"]:
                model = InductiveField3D(
                    grid,
                    frequency=frequency,
                    coils=[coil(c["coil_half_length"])],
                    conductors=[
                        Conductor3D(
                            [tube(slits, c["tube_half_length"])], c["conductivity"]
                        )
                    ],
                    regularization=c["regularization"],
                )
                solution, seconds = timed(model)
                ref = reference("3d", slits, frequency)
                row = {
                    "slits": slits,
                    "frequency": frequency,
                    "cells": cells,
                    "h": h,
                    "loss": solution.loss,
                    "inductance": solution.inductance,
                    "unknowns": int(grid.free_edges.sum()),
                    "seconds": seconds,
                }
                if ref is not None:
                    row["loss_error"] = solution.loss / ref["loss"] - 1
                    row["inductance_error"] = (
                        solution.inductance / ref["inductance"] - 1
                    )
                rows.append(row)
                print(
                    f"tube3d {slits} {frequency:.0f} Hz h={1e3 * h:.2f} mm:"
                    f" loss {row['loss']:.4e} ({row.get('loss_error', np.nan):+.4f})"
                    f" L {row['inductance']:.4e} ({row.get('inductance_error', np.nan):+.4f})"
                    f" {row['unknowns']} unknowns, {seconds:.0f} s",
                    flush=True,
                )
    return rows


PARTS = {"solenoid": solenoid, "driver": driver, "tube2d": tube2d, "tube3d": tube3d}


def main(parts: list[str]) -> None:
    """Run the parts and add them to ``field3d_r4.json``."""
    path = HERE / "field3d_r4.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for part in parts:
        results[part] = PARTS[part]()
        path.write_text(json.dumps(results, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or list(PARTS))
