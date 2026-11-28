"""MHD3D on all ranks against the same run on one rank (run under ``mpirun``).

Each rank builds the hierarchy twice: over the world communicator (the patches spread
by ``placement``) and on its own (``COMM_SELF``). Both take the same regrid and steps;
the level-0 state must agree to round-off and the integrals stay those of the start.
Prints ``MPI_JOB_OK``.
"""

from __future__ import annotations

import sys

import numpy as np
from maybempi import MPI
from patchwork import AMRHierarchy

from carafe.mhd3d import CONSERVED_3D, MHD3D

WORLD = MPI.COMM_WORLD


def run(comm, placement):
    mesh = AMRHierarchy(
        (8, 8, 8),
        bounds=((0, 1), (0, 1), (0, 1)),
        periodic=(True, True, True),
        comm=comm,
        placement=placement,
    )
    model = MHD3D(mesh, resistivity=1e-3)
    k = 2 * np.pi
    model.set_state(
        1.0,
        (lambda x, y, z: 0.1 * np.sin(k * y), lambda x, y, z: 0.1 * np.sin(k * z), 0.0),
        1.0,
        vector_potential=(
            lambda x, y, z: 0.1 * np.cos(k * z) / k,
            0.0,
            lambda x, y, z: 0.1 * np.cos(k * x) / k,
        ),
    )
    start = model.totals()
    tags = {0: np.zeros((8, 8, 8), dtype=bool)}
    tags[0][2:6, 1:5, 3:7] = True
    model.regrid(tags, max_patch_size=4)
    for _ in range(3):
        model.step()
    end = model.totals()
    drift = max(abs(end[name] - start[name]) for name in CONSERVED_3D)
    state = model.dataset()
    keys = ("density", "pressure", "magnetic_field_x", "magnetic_field_z")
    return (
        np.stack([state[key].values for key in keys]),
        drift,
        model.composite_divergence(),
    )


def main() -> None:
    placement = sys.argv[1] if len(sys.argv) > 1 else "morton"
    spread, drift, divergence = run(WORLD, placement)
    alone, _, _ = run(MPI.COMM_SELF, placement)
    difference = float(np.max(np.abs(spread - alone)))
    if WORLD.Get_rank() == 0:
        print(f"difference {difference:.2e}, drift {drift:.2e}, div {divergence:.2e}")
        if difference < 1e-12 and drift < 1e-13 and divergence < 1e-13:
            print("MPI_JOB_OK")


if __name__ == "__main__":
    main()
