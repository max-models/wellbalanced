"""Reference losses of the slotted cylinder as a surface-impedance boundary: NGSolve.

The same geometry as ``slotted_cylinder_ngsolve.py`` (``slotted_cylinder_case.CASE``),
with the tube removed from the mesh and Leontovich's condition ``E_t = Z_s H x n`` on its
surface, ``Z_s = (1 + i) / (sigma delta)``: the model of wellbalanced's
``ImpedanceConductor3D``, so that the comparison isolates the discretisation (cut cells
or staircase against Nedelec elements on curved meshes fitted to the tube and its
slits). Weak form, ``n`` the surface normal::

    int nu curl A . curl v + int_surface (i omega / Z_s) (n x A) . (n x v) = int J . v

The coil's current is the sheet on the side faces of its box (``J = T x n``), so the
source does not touch the removed tube. Also ``volume2d``: the z-invariant problem with
the tube as a volume conductor at the surface-impedance frequencies, a boundary-layer
mesh resolving the skin depth, for the model error of the surface impedance.

Run in the NGSolve environment of ``slotted_cylinder_ngsolve.py``::

    ../.external/fem-venv/bin/python benchmarks/slotted_cylinder_ngsolve_sibc.py [3d] [2d] [volume2d]

It writes ``slotted_cylinder_ngsolve_sibc_3d.json``, ``..._sibc_2d.json`` and
``..._volume2d.json``.
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

#: The surface-impedance frequencies: skin depths 2.25, 0.71 and 0.225 mm in the
#: 6 mm wall.
FREQUENCIES = [5.0e4, 5.0e5, 5.0e6]


def surface_impedance(frequency: float) -> complex:
    """``Z_s = (1 + i) sqrt(omega mu0 / (2 sigma))`` [Ohm]."""
    omega = 2 * np.pi * frequency
    return (1 + 1j) * np.sqrt(omega * MU_0 / (2 * CASE["conductivity"]))


def tube_3d(slits: str):
    """The tube of the eighth, with its slits, as an OCC solid."""
    c = CASE
    z0, length = -c["tube_half_length"], 2 * c["tube_half_length"]
    tube = occ.Cylinder(
        occ.Pnt(0, 0, z0), occ.Z, r=c["outer_radius"], h=length
    ) - occ.Cylinder(occ.Pnt(0, 0, z0), occ.Z, r=c["inner_radius"], h=length)
    if slits != "none":
        half_w = 0.5 * c["slit_width"]
        z_slit = (
            c["tube_half_length"] + 1.0 if slits == "through" else c["slit_half_length"]
        )
        big = 2 * c["box"]
        tube = (
            tube
            - occ.Box((0, -half_w, -z_slit), (big, half_w, z_slit))
            - occ.Box((-half_w, 0, -z_slit), (half_w, big, z_slit))
        )
    return tube


def geometry_3d(slits: str, maxh_metal: float, order: int) -> ng.Mesh:
    """The air of the eighth: the box minus the tube, the coil's box marked."""
    c = CASE
    domain = occ.Box((0, 0, 0), (c["box"], c["box"], c["box_z"]))
    air = domain - tube_3d(slits)
    s = c["coil_half_side"]
    coil_box = occ.Box((0, 0, 0), (s, s, c["coil_half_length"]))
    inner = air * coil_box
    outer = air - coil_box
    inner.mat("coil")
    outer.mat("air")
    shape = occ.Glue([inner, outer])
    for face in shape.faces:
        p = face.center
        if abs(p.x) < 1e-9 or abs(p.y) < 1e-9:
            face.name = "sym"
        elif abs(p.z) < 1e-9:
            face.name = "mid"
        elif (
            abs(p.x - c["box"]) < 1e-9
            or abs(p.y - c["box"]) < 1e-9
            or abs(p.z - c["box_z"]) < 1e-9
        ):
            face.name = "outer"
        elif abs(p.x - s) < 1e-9 and p.z < c["coil_half_length"]:
            face.name = "coilx"
        elif abs(p.y - s) < 1e-9 and p.z < c["coil_half_length"]:
            face.name = "coily"
        elif abs(p.z - c["coil_half_length"]) < 1e-9:
            face.name = "coiltop"
        else:
            face.name = "metal"
            face.maxh = maxh_metal
    mesh = ng.Mesh(occ.OCCGeometry(shape).GenerateMesh(maxh=c["maxh_air"]))
    mesh.Curve(order)
    return mesh


def solve_3d(mesh: ng.Mesh, frequency: float, order: int) -> dict:
    """Loss and coil impedance of the eighth at 1 A, the tube a surface impedance."""
    c = CASE
    omega = 2 * np.pi * frequency
    impedance_s = surface_impedance(frequency)
    fes = ng.HCurl(mesh, order=order, dirichlet="sym|outer", complex=True)
    u, v = fes.TnT()
    n = ng.specialcf.normal(3)
    nu = 1.0 / MU_0
    gauge = c["regularization"] * nu / max(c["box"], c["box_z"]) ** 2
    a = ng.BilinearForm(fes, symmetric=True, condense=True)
    a += nu * ng.curl(u) * ng.curl(v) * ng.dx
    a += gauge * u * v * ng.dx
    a += (
        (1j * omega / impedance_s)
        * ng.Cross(n, u.Trace())
        * ng.Cross(n, v.Trace())
        * ng.ds("metal")
    )
    preconditioner = ng.Preconditioner(a, "bddc")
    turns = c["turns_per_length"]
    f = ng.LinearForm(fes)
    f += turns * v.Trace()[1] * ng.ds("coilx")
    f += -turns * v.Trace()[0] * ng.ds("coily")
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
    # Z = i omega int A . J (I = 1 A), the coil's sheet on its side faces
    work = ng.Integrate(
        turns * gfu[1], mesh, ng.BND, definedon=mesh.Boundaries("coilx")
    ) - ng.Integrate(turns * gfu[0], mesh, ng.BND, definedon=mesh.Boundaries("coily"))
    impedance = 1j * omega * complex(work)
    return {
        # the gauge is real: the coil's power is the loss on the surface
        "loss": 0.5 * float(impedance.real),
        "resistance": float(impedance.real),
        "inductance": float(impedance.imag / omega),
        "ndof": int(fes.ndof),
        "elements": mesh.ne,
        "iterations": solver.iterations,
        "seconds": elapsed,
    }


def ring_2d(slits: str):
    """The quarter's square and the ring of the tube (with its slits), as OCC faces."""
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
    return domain, tube


def _name_edges_2d(shape, metal_maxh: float | None):
    c = CASE
    s = c["coil_half_side"]
    for edge in shape.edges:
        p = edge.center
        if min(abs(p.x), abs(p.y), abs(p.x - c["box"]), abs(p.y - c["box"])) < 1e-9:
            edge.name = "wall"
        elif abs(p.x - s) < 1e-9:
            edge.name = "coilx"
        elif abs(p.y - s) < 1e-9:
            edge.name = "coily"
        else:
            edge.name = "metal"
            if metal_maxh is not None:
                edge.maxh = metal_maxh


def geometry_2d_sibc(slits: str, maxh_metal: float, order: int) -> ng.Mesh:
    """The air of the quarter cross-section, the coil's square marked."""
    c = CASE
    domain, tube = ring_2d(slits)
    air = domain - tube
    s = c["coil_half_side"]
    coil = occ.Rectangle(s, s).Face()
    inner, outer = air * coil, air - coil
    inner.faces.name = "coil"
    outer.faces.name = "air"
    shape = occ.Glue([inner, outer])
    _name_edges_2d(shape, maxh_metal)
    mesh = ng.Mesh(occ.OCCGeometry(shape, dim=2).GenerateMesh(maxh=c["maxh_air"]))
    mesh.Curve(order)
    return mesh


def solve_2d_sibc(mesh: ng.Mesh, frequency: float, order: int) -> dict:
    """Loss [W/m] and impedance per metre of the whole cross-section at 1 A."""
    c = CASE
    omega = 2 * np.pi * frequency
    impedance_s = surface_impedance(frequency)
    fes = ng.HCurl(mesh, order=order, dirichlet="wall", complex=True)
    u, v = fes.TnT()
    t = ng.specialcf.tangential(2)
    nu = 1.0 / MU_0
    gauge = c["regularization"] * nu / max(c["box"], c["box_z"]) ** 2
    a = ng.BilinearForm(fes, symmetric=True)
    a += nu * ng.curl(u) * ng.curl(v) * ng.dx
    a += gauge * u * v * ng.dx
    a += (1j * omega / impedance_s) * (u.Trace() * t) * (v.Trace() * t) * ng.ds("metal")
    turns = c["turns_per_length"]
    f = ng.LinearForm(fes)
    f += turns * v.Trace()[1] * ng.ds("coilx")
    f += -turns * v.Trace()[0] * ng.ds("coily")
    a.Assemble()
    f.Assemble()
    gfu = ng.GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
    work = ng.Integrate(
        turns * gfu[1], mesh, ng.BND, definedon=mesh.Boundaries("coilx")
    ) - ng.Integrate(turns * gfu[0], mesh, ng.BND, definedon=mesh.Boundaries("coily"))
    impedance = 4j * omega * complex(work)
    return {
        "loss": 0.5 * float(impedance.real),
        "resistance": float(impedance.real),
        "inductance": float(impedance.imag / omega),
        "ndof": int(fes.ndof),
    }


def geometry_2d_volume(slits: str, maxh_skin: float, order: int) -> ng.Mesh:
    """The quarter cross-section with the tube meshed, ``maxh_skin`` on its surface."""
    c = CASE
    domain, tube = ring_2d(slits)
    s = c["coil_half_side"]
    coil = occ.Rectangle(s, s).Face()
    inner, outer = coil - tube, domain - coil
    tube.faces.name = "tube"
    inner.faces.name = "coil"
    outer.faces.name = "air"
    tube.faces.maxh = 4 * maxh_skin
    shape = occ.Glue([tube, inner, outer])
    _name_edges_2d(shape, maxh_skin)
    mesh = ng.Mesh(occ.OCCGeometry(shape, dim=2).GenerateMesh(maxh=c["maxh_air"]))
    mesh.Curve(order)
    return mesh


def solve_2d_volume(mesh: ng.Mesh, frequency: float, order: int) -> dict:
    """The volume conductor's loss [W/m] and impedance per metre, whole section."""
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
    turns = c["turns_per_length"]
    f = ng.LinearForm(fes)
    f += turns * v.Trace()[1] * ng.ds("coilx")
    f += -turns * v.Trace()[0] * ng.ds("coily")
    a.Assemble()
    f.Assemble()
    gfu = ng.GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
    loss = ng.Integrate(
        0.5 * omega**2 * sigma * ng.InnerProduct(gfu, ng.Conj(gfu)),
        mesh,
        definedon=mesh.Materials("tube"),
    )
    work = ng.Integrate(
        turns * gfu[1], mesh, ng.BND, definedon=mesh.Boundaries("coilx")
    ) - ng.Integrate(turns * gfu[0], mesh, ng.BND, definedon=mesh.Boundaries("coily"))
    impedance = 4j * omega * complex(work)
    return {
        "loss": 4 * float(np.real(loss)),
        "resistance": float(impedance.real),
        "inductance": float(impedance.imag / omega),
        "ndof": int(fes.ndof),
    }


def main(parts: list[str]) -> None:
    """Run the references and write their JSON files."""
    ng.SetNumThreads(2)
    t = CASE["thickness"]
    if "2d" in parts:
        results = []
        for slits in ("none", "through"):
            for level, maxh, order in ((1, t / 3, 3), (2, t / 6, 4), (3, t / 12, 4)):
                mesh = geometry_2d_sibc(slits, maxh, order)
                for frequency in FREQUENCIES:
                    row = {
                        "slits": slits,
                        "frequency": frequency,
                        "mesh": level,
                        "maxh_metal": maxh,
                        "order": order,
                        **solve_2d_sibc(mesh, frequency, order),
                    }
                    results.append(row)
                    print(json.dumps(row), flush=True)
        (HERE / "slotted_cylinder_ngsolve_sibc_2d.json").write_text(
            json.dumps({"case": CASE, "results": results}, indent=1) + "\n"
        )
    if "volume2d" in parts:
        results = []
        for slits in ("none", "through"):
            for frequency in FREQUENCIES:
                delta = np.sqrt(
                    2 / (2 * np.pi * frequency * MU_0 * CASE["conductivity"])
                )
                for level, factor, order in ((1, 1.0, 4), (2, 0.5, 5)):
                    mesh = geometry_2d_volume(slits, factor * delta, order)
                    row = {
                        "slits": slits,
                        "frequency": frequency,
                        "mesh": level,
                        "maxh_skin": factor * delta,
                        "order": order,
                        **solve_2d_volume(mesh, frequency, order),
                    }
                    results.append(row)
                    print(json.dumps(row), flush=True)
        (HERE / "slotted_cylinder_ngsolve_volume2d.json").write_text(
            json.dumps({"case": CASE, "results": results}, indent=1) + "\n"
        )
    if "3d" in parts:
        results = []
        # (level, maxh on the tube's surface, order): the last is the reference
        meshes = [(1, t / 3, 2), (1, t / 3, 3), (2, t / 4.5, 3)]
        for slits in ("none", "closed", "through"):
            for level, maxh, order in meshes:
                mesh = geometry_3d(slits, maxh, order)
                for frequency in FREQUENCIES[:2]:
                    row = {
                        "slits": slits,
                        "frequency": frequency,
                        "mesh": level,
                        "maxh_metal": maxh,
                        "order": order,
                        **solve_3d(mesh, frequency, order),
                    }
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    # written as it goes: the 3D references take hours
                    (HERE / "slotted_cylinder_ngsolve_sibc_3d.json").write_text(
                        json.dumps({"case": CASE, "results": results}, indent=1) + "\n"
                    )


if __name__ == "__main__":
    main(sys.argv[1:] or ["2d", "volume2d", "3d"])
