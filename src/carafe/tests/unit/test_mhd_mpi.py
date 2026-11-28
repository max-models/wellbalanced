"""MHD2D and MHD3D on several ranks give what one rank gives (under ``mpirun``)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

JOB = Path(__file__).with_name("_mpi_mhd_job.py")
JOB_3D = Path(__file__).with_name("_mpi_mhd3d_job.py")
MPIRUN = shutil.which("mpirun")

pytestmark = pytest.mark.skipif(
    MPIRUN is None
    or importlib.util.find_spec("mpi4py") is None
    or importlib.util.find_spec("patchwork") is None,
    reason="needs mpirun, mpi4py and patchwork",
)


@pytest.mark.parametrize(
    ("job", "ranks", "placement"),
    [(JOB, 2, "all"), (JOB, 3, "morton"), (JOB_3D, 2, "all"), (JOB_3D, 3, "morton")],
    ids=["2d-all", "2d-morton", "3d-all", "3d-morton"],
)
def test_mhd_on_several_ranks_gives_what_one_rank_gives(job, ranks, placement) -> None:
    env = os.environ.copy()
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    result = subprocess.run(
        [
            MPIRUN,
            "-n",
            str(ranks),
            "--oversubscribe",
            sys.executable,
            str(job),
            placement,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env=env,
    )
    assert "MPI_JOB_OK" in result.stdout, result.stdout[-2000:] + result.stderr[-2000:]
