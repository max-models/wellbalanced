"""Reference eddy losses of a slotted cylinder: NGSolve, Nedelec elements on fitted meshes.

The benchmark of the 3D field's discretisation (RF-driver plan, step R4): a conducting
tube with four axial slits in the uniform axial field of a square solenoid, inside a
perfectly conducting box, solved with Nedelec elements of order 2 and 3 on curved
meshes fitted to the tube and its slits; the problems of ``benchmarks/field3d_r4.py``
(the shared geometry is ``slotted_cylinder_case.CASE``):

- ``3d``: the finite tube, one eighth, ``x, y, z >= 0``, with ``A x n = 0`` on the
  planes ``x = 0`` and ``y = 0`` (the azimuthal current crosses them at right angles)
  and ``H x n = 0`` on ``z = 0`` (the midplane); slits ``"none"``, ``"closed"`` (ending
  10 mm before the tube's ends) or ``"through"`` (cutting the tube into four).
- ``2d``: the infinite tube in the infinite solenoid (z-invariant, a quarter of the
  cross-section), with no slits or with slits along its whole length: the in-plane
  vector potential in 2D.

The coil is the same ``J = curl T`` (the load vector ``int T . curl v`` over the coil's
box) and the gauge the same small regularisation as in wellbalanced.

NGSolve is a benchmark, not a dependency of wellbalanced: run this in a separate
environment (``pip install ngsolve``), e.g.::

    python -m venv ../.external/fem-venv && ../.external/fem-venv/bin/pip install ngsolve
    ../.external/fem-venv/bin/python benchmarks/slotted_cylinder_ngsolve.py [3d] [2d]

It writes ``benchmarks/slotted_cylinder_ngsolve_3d.json`` (the loss and inductance of
the eighth) and ``..._2d.json`` (of the whole cross-section, per metre) for every slit
variant, frequency, mesh and order.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import netgen.occ as occ
import ngsolve as ng
import numpy as np
from ngsolve.krylovspace import CGSolver

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slotted_cylinder_case import CASE, MU_0  # noqa: E402


def geometry(slits: str, maxh_tube: float, order: int) -> ng.Mesh:
    """The meshed eighth: tube (with slits), the coil's box and the air around.

    ``maxh_tube`` in the tube, twice that in the coil's box; curved to ``order``.
    """
    c = CASE
    domain = occ.Box((0, 0, 0), (c["box"], c["box"], c["box_z"]))
    tube = occ.Cylinder(
        occ.Pnt(0, 0, -c["tube_half_length"]),
        occ.Z,
        r=c["outer_radius"],
        h=2 * c["tube_half_length"],
    ) - occ.Cylinder(
        occ.Pnt(0, 0, -c["tube_half_length"]),
        occ.Z,
        r=c["inner_radius"],
        h=2 * c["tube_half_length"],
    )
    tube = tube * domain
    if slits != "none":
        half_w = 0.5 * c["slit_width"]
        z_slit = (
            c["tube_half_length"] + 1.0 if slits == "through" else c["slit_half_length"]
        )
        big = 2 * c["box"]
        across_x = occ.Box((0, -half_w, -z_slit), (big, half_w, z_slit))  # phi = 0
        across_y = occ.Box((-half_w, 0, -z_slit), (half_w, big, z_slit))  # phi = 90
        tube = tube - across_x - across_y
    coil = occ.Box(
        (0, 0, 0), (c["coil_half_side"], c["coil_half_side"], c["coil_half_length"])
    )
    inner = coil - tube
    outer = domain - coil
    tube.mat("tube")
    inner.mat("coil")
    outer.mat("air")
    tube.maxh = maxh_tube
    inner.maxh = 2 * maxh_tube
    shape = occ.Glue([tube, inner, outer])
    for face in shape.faces:
        centre = face.center
        if abs(centre.x) < 1e-9:
            face.name = "sym"
        elif abs(centre.y) < 1e-9:
            face.name = "sym"
        elif abs(centre.z) < 1e-9:
            face.name = "mid"
        elif (
            abs(centre.x - c["box"]) < 1e-9
            or abs(centre.y - c["box"]) < 1e-9
            or abs(centre.z - c["box_z"]) < 1e-9
        ):
            face.name = "outer"
    geo = occ.OCCGeometry(shape)
    mesh = ng.Mesh(geo.GenerateMesh(maxh=c["maxh_air"]))
    mesh.Curve(order)
    return mesh


def solve(mesh: ng.Mesh, frequency: float, order: int) -> dict:
    """Loss and coil impedance of the eighth at 1 A.

    The space keeps its gradients: in the conductor ``E = -i omega A`` has a gradient
    part (the surface charges), which ``nograds=True`` would cut off (it gave losses a
    few per cent off, changing with the mesh). Static condensation, BDDC and CG for
    the complex-symmetric matrix (``conjugate=False``).
    """
    c = CASE
    omega = 2 * np.pi * frequency
    sigma = mesh.MaterialCF({"tube": c["conductivity"]}, default=0.0)
    fes = ng.HCurl(mesh, order=order, dirichlet="sym|outer", complex=True)
    u, v = fes.TnT()
    nu = 1.0 / MU_0
    size = max(c["box"], c["box_z"])
    gauge = c["regularization"] * nu / size**2
    a = ng.BilinearForm(fes, symmetric=True, condense=True)
    a += nu * ng.curl(u) * ng.curl(v) * ng.dx
    a += (1j * omega * sigma + gauge) * u * v * ng.dx
    preconditioner = ng.Preconditioner(a, "bddc")
    t = c["turns_per_length"] * ng.CF((0, 0, 1))
    f = ng.LinearForm(fes)
    f += t * ng.curl(v) * ng.dx("tube|coil")
    start = time.perf_counter()
    a.Assemble()
    f.Assemble()
    gfu = ng.GridFunction(fes)
    solver = CGSolver(
        a.mat, preconditioner.mat, conjugate=False, tol=1e-10, maxiter=5000
    )
    f.vec.data += a.harmonic_extension_trans * f.vec
    gfu.vec.data = solver * f.vec
    gfu.vec.data += a.harmonic_extension * gfu.vec
    gfu.vec.data += a.inner_solve * f.vec
    elapsed = time.perf_counter() - start
    loss = ng.Integrate(
        0.5 * omega**2 * sigma * ng.InnerProduct(gfu, gfu),
        mesh,
        definedon=mesh.Materials("tube"),
    )
    # S = 1/2 i omega int A . conj(J) = 1/2 i omega int B . T (T real), I = 1 A
    flux = ng.Integrate(ng.curl(gfu) * t, mesh, definedon=mesh.Materials("tube|coil"))
    impedance = 1j * omega * complex(flux)
    return {
        "loss": float(np.real(loss)),
        "resistance": float(impedance.real),
        "inductance": float(impedance.imag / omega),
        "ndof": int(fes.ndof),
        "elements": mesh.ne,
        "iterations": solver.iterations,
        "seconds": elapsed,
    }


def geometry_2d(slits: str, maxh_tube: float, order: int) -> ng.Mesh:
    """The meshed quarter of the cross-section of the z-invariant problem."""
    c = CASE
    domain = occ.Rectangle(c["box"], c["box"]).Face()
    ring = (
        occ.Circle((0, 0), c["outer_radius"]).Face()
        - occ.Circle((0, 0), c["inner_radius"]).Face()
    )
    tube = ring * domain
    if slits == "through":
        half_w, big = 0.5 * c["slit_width"], 2 * c["box"]
        tube = (
            tube
            - occ.MoveTo(0, -half_w).Rectangle(big, 2 * half_w).Face()
            - occ.MoveTo(-half_w, 0).Rectangle(2 * half_w, big).Face()
        )
    coil = occ.Rectangle(c["coil_half_side"], c["coil_half_side"]).Face()
    inner = coil - tube
    outer = domain - coil
    tube.faces.name = "tube"
    inner.faces.name = "coil"
    outer.faces.name = "air"
    tube.faces.maxh = maxh_tube
    inner.faces.maxh = 2 * maxh_tube
    shape = occ.Glue([tube, inner, outer])
    for edge in shape.edges:
        p = edge.center
        if min(abs(p.x), abs(p.y), abs(p.x - c["box"]), abs(p.y - c["box"])) < 1e-9:
            edge.name = "wall"
    mesh = ng.Mesh(occ.OCCGeometry(shape, dim=2).GenerateMesh(maxh=c["maxh_air"]))
    mesh.Curve(order)
    return mesh


def solve_2d(mesh: ng.Mesh, frequency: float, order: int) -> dict:
    """Loss [W/m] and coil impedance per metre of the whole cross-section at 1 A."""
    c = CASE
    omega = 2 * np.pi * frequency
    sigma = mesh.MaterialCF({"tube": c["conductivity"]}, default=0.0)
    fes = ng.HCurl(mesh, order=order, dirichlet="wall", complex=True)
    u, v = fes.TnT()
    nu = 1.0 / MU_0
    gauge = c["regularization"] * nu / max(c["box"], c["box_z"]) ** 2
    a = ng.BilinearForm(fes, symmetric=True)
    a += nu * ng.curl(u) * ng.curl(v) * ng.dx
    a += (1j * omega * sigma + gauge) * u * v * ng.dx
    t = c["turns_per_length"]
    f = ng.LinearForm(fes)
    f += t * ng.curl(v) * ng.dx("tube|coil")
    a.Assemble()
    f.Assemble()
    gfu = ng.GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
    loss = ng.Integrate(
        0.5 * omega**2 * sigma * ng.InnerProduct(gfu, gfu),
        mesh,
        definedon=mesh.Materials("tube"),
    )
    flux = ng.Integrate(ng.curl(gfu) * t, mesh, definedon=mesh.Materials("tube|coil"))
    impedance = 4j * omega * complex(flux)  # the four quarters
    return {
        "loss": 4 * float(np.real(loss)),
        "resistance": float(impedance.real),
        "inductance": float(impedance.imag / omega),
        "ndof": int(fes.ndof),
    }


def main(parts: list[str]) -> None:
    """Run the 3D and 2D references and write their JSON files."""
    ng.SetNumThreads(2)
    t = CASE["thickness"]
    if "3d" in parts:
        results = []
        # (level, maxh in the tube, orders): the last is the reference
        meshes = [(1, t / 2, (2, 3)), (2, t / 3, (3,))]
        for slits in ("none", "closed", "through"):
            for level, maxh, orders in meshes:
                for order in orders:
                    mesh = geometry(slits, maxh, order)
                    for frequency in CASE["frequencies"]:
                        row = {
                            "slits": slits,
                            "frequency": frequency,
                            "mesh": level,
                            "maxh_tube": maxh,
                            "order": order,
                            **solve(mesh, frequency, order),
                        }
                        results.append(row)
                        print(json.dumps(row), flush=True)
        (HERE / "slotted_cylinder_ngsolve_3d.json").write_text(
            json.dumps({"case": CASE, "results": results}, indent=1) + "\n"
        )
    if "2d" in parts:
        results = []
        for slits in ("none", "through"):
            for level, maxh, order in ((1, t / 2, 3), (2, t / 4, 4), (3, t / 8, 4)):
                mesh = geometry_2d(slits, maxh, order)
                for frequency in CASE["frequencies"]:
                    row = {
                        "slits": slits,
                        "frequency": frequency,
                        "mesh": level,
                        "maxh_tube": maxh,
                        "order": order,
                        **solve_2d(mesh, frequency, order),
                    }
                    results.append(row)
                    print(json.dumps(row), flush=True)
        (HERE / "slotted_cylinder_ngsolve_2d.json").write_text(
            json.dumps({"case": CASE, "results": results}, indent=1) + "\n"
        )


if __name__ == "__main__":
    main(sys.argv[1:] or ["3d", "2d"])
