r"""MHD in three dimensions on a patchwork hierarchy.

`MHD3D` solves the equations of ideal magnetohydrodynamics in a box ``(x, y, z)``, in
Heaviside–Lorentz units with :math:`\mu_0 = 1`, the same equations as `carafe.mhd.MHD2D`
without its plane symmetry:

.. math::

    \partial_t\rho + \nabla\cdot(\rho\mathbf u) = 0, \qquad
    \partial_t(\rho\mathbf u) + \nabla\cdot\big(\rho\mathbf u\mathbf u
        + p_T - \mathbf B\mathbf B\big) = 0,

    \partial_t E + \nabla\cdot\big((E + p_T)\mathbf u
        - (\mathbf u\cdot\mathbf B)\mathbf B\big) = 0, \qquad
    \partial_t\mathbf B = -\nabla\times\mathbf E, \quad
    \mathbf E = -\mathbf u\times\mathbf B + \eta\mathbf J,

with :math:`p_T = p + |\mathbf B|^2/2`, :math:`E = p/(\gamma - 1) + \rho|\mathbf u|^2/2
+ |\mathbf B|^2/2` and, with a uniform resistivity :math:`\eta`, the energy flux
:math:`\eta\,\mathbf J\times\mathbf B`. A uniform gravity :math:`\mathbf g` adds
:math:`\rho\mathbf g` to the momentum and :math:`\rho\mathbf u\cdot\mathbf g` to the energy.

**The mesh is patchwork's**: the density, the momentum and the energy are cell-centered
conserved fields of a 3D `patchwork.AMRHierarchy`; refined patches take the same time
step, with refluxing at coarse–fine faces (`patchwork.FluxRegister`).

**The magnetic field** is the curl of a vector potential on the cell edges plus a
uniform mean field :math:`\mathbf B_0`: :math:`A_x` on the x-edges, :math:`A_y` on the
y-edges, :math:`A_z` on the z-edges (patchwork's edge-centered patch fields). The flux
through a face is the circulation of :math:`\mathbf A` around it, so the divergence of
every cell is zero to round-off, and the potential advances with the electric field on
the edges, :math:`\partial_t\mathbf A = -\mathbf E`: the 3D form of the constrained
transport of `MHD2D`, whose :math:`A_z` on the nodes is the z-edge component here. The
restriction of the potential (the mean of the fine edges along a coarse edge) makes the
flux through a coarse face the sum of the fine fluxes, so the divergence is zero on the
composite mesh too, and across regrids, which interpolate the potential.

**Numerics** as in `MHD2D`: MUSCL reconstruction of the primitive variables, the HLLD
Riemann solver along each axis (the ``mhd_sweep`` kernel, the two transverse axes run as
one), the electric field on the edges by the upwind CT-contact average of the face fluxes
(Gardiner and Stone 2008; the ``ct_contact`` kernel on each plane normal to the edges),
Heun's method (SSP-RK2) and refluxing. The resistive electric field and energy flux are
explicit (:math:`\mathbf J` on the edges from the face fields), within the diffusion
limit of the finest cells.

**Boundaries.** Periodic axes are those the hierarchy declares periodic; the faces of the
others are outflow (zero-gradient cells, the potential extrapolated linearly).

**Not yet.** Walls, heat conduction, viscosity, graded axes and r–z; the fields of a patch
are not split between ranks (place patches with ``placement="morton"`` or
``"balanced"``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from methodcard import ConvergenceProblem, method
from scope_profiler import profile

from carafe._profiling import _PROFILE_TAGS
from carafe.datasets import variable
from carafe.kernels.ct_contact import kernel as ct_contact
from carafe.kernels.mhd_sweep import LIMITER_CODES
from carafe.kernels.mhd_sweep import kernel as mhd_sweep
from carafe.limiters import LIMITERS
from carafe.mhd import HALO, _outflow_cells, _outflow_nodes

if TYPE_CHECKING:
    from patchwork import AMRHierarchy, Patch

__all__ = ["CONSERVED_3D", "MHD3D"]

#: The cell-centered conserved fields of `MHD3D`, in this order.
CONSERVED_3D = ("density", "momentum_x", "momentum_y", "momentum_z", "energy")
#: The components of the vector potential and their edges.
_POTENTIAL = (("ax", "x-edge"), ("ay", "y-edge"), ("az", "z-edge"))

Value = float | np.ndarray | Callable[..., Any]


def _cyclic(axis: int) -> tuple[int, int, int]:
    """The axis and the two after it, cyclically: (x, y, z), (y, z, x), (z, x, y)."""
    return axis, (axis + 1) % 3, (axis + 2) % 3


def _along(array: np.ndarray, axis: int, start: int, stop: int) -> np.ndarray:
    """``array[start:stop]`` along ``axis``."""
    index = [slice(None)] * array.ndim
    index[axis] = slice(start, stop)
    return array[tuple(index)]


def _pair_mean(array: np.ndarray, axis: int, start: int, count: int) -> np.ndarray:
    """The means of points ``k`` and ``k + 1`` along ``axis``, ``k = start .. start + count - 1``."""
    return 0.5 * (
        _along(array, axis, start, start + count)
        + _along(array, axis, start + 1, start + 1 + count)
    )


def _field_at_face(
    face: np.ndarray, normal: int, node: int, shape: tuple[int, ...]
) -> np.ndarray:
    """A haloed face field (nodes along ``node``) at a block's faces normal to ``normal``.

    The mean of the four faces around each: over the cells on both sides along
    ``normal`` and the nodes on both sides along ``node``.
    """
    h = HALO
    face = _pair_mean(face, normal, h - 1, shape[normal] + 1)  # cells to nodes
    face = _pair_mean(face, node, h, shape[node])  # nodes to cells
    other = 3 - normal - node
    return _along(face, other, h, h + shape[other])


def _fast_speed(rho, p, bn, bt1, bt2, gamma):
    """The fast magnetosonic speed along the normal."""
    a2 = gamma * p / rho
    b2 = (bn**2 + bt1**2 + bt2**2) / rho
    root = np.sqrt(np.maximum((a2 + b2) ** 2 - 4.0 * a2 * bn**2 / rho, 0.0))
    return np.sqrt(0.5 * (a2 + b2 + root))


def _alfven_problem() -> ConvergenceProblem:
    """The circularly polarized Alfvén wave in 3D after one period."""

    def error(h: float) -> float:
        cells = round(1.0 / h)
        model, exact = MHD3D.alfven_wave(cells, limiter="none")
        model.run(model.period)
        by = model.dataset()["magnetic_field_y"].values
        return float(np.mean(np.abs(by - exact["magnetic_field_y"].values)))

    return ConvergenceProblem(error, steps=[1 / 8, 1 / 16])


@method(
    equations={
        "mass": r"\partial_t\rho + \nabla\cdot(\rho\mathbf u) = 0",
        "momentum": r"\partial_t(\rho\mathbf u) + \nabla\cdot\big(\rho\mathbf u\mathbf u + p_T\mathbf I - \mathbf B\mathbf B\big) = \rho\mathbf g",
        "energy": r"\partial_t E + \nabla\cdot\big((E + p_T)\mathbf u - (\mathbf u\cdot\mathbf B)\mathbf B + \eta\mathbf J\times\mathbf B\big) = \rho\mathbf u\cdot\mathbf g",
        "field": r"\mathbf B = \mathbf B_0 + \nabla\times\mathbf A, \qquad \partial_t\mathbf A = -\mathbf E = \mathbf u\times\mathbf B - \eta\nabla\times\mathbf B \quad \text{on the edges}",
        "ct_contact": r"E_c = \tfrac14\big(\textstyle\sum_\text{faces} E_c^f\big) + \tfrac14\big(\textstyle\sum_\text{4 half-faces} (E_c^f - E_c^\text{cell})_\text{upwind}\big)",
    },
    order=2,
    assumptions=(
        r"Heaviside–Lorentz units ($\mu_0 = 1$); uniform resistivity $\eta$ and gravity $\mathbf g$, explicit",
        r"the vector potential on the edges: zero discrete divergence per cell and on the composite mesh, to round-off",
        r"MUSCL on the primitives, HLLD, CT-contact electric field on the edges, Heun (SSP-RK2); refluxing at coarse–fine faces",
        r"Cartesian, uniform levels, one time step for all levels; periodic or outflow faces",
        r"the order is that of a smooth Alfvén wave with the unlimited reconstruction",
    ),
    references=(
        "[Gardiner and Stone, J. Comput. Phys. 227, 4123 (2008)](https://doi.org/10.1016/j.jcp.2007.12.017)",
        "[Miyoshi and Kusano, J. Comput. Phys. 208, 315 (2005)](https://doi.org/10.1016/j.jcp.2005.02.017)",
    ),
    problem=_alfven_problem,
)
class MHD3D:
    """MHD on the patches of a 3D patchwork hierarchy; see the module docstring.

    Args:
        hierarchy: The 3D Cartesian hierarchy the fields live on. Its periodic axes
            are periodic; the faces of the others are outflow faces.
        gamma: The adiabatic index.
        mean_field: The uniform field ``(B0x, B0y, B0z)`` added to the curl of the
            vector potential.
        resistivity: The uniform resistivity ``eta`` (a magnetic diffusivity); 0 for
            ideal MHD.
        limiter: The slope limiter of the reconstruction (`carafe.limiters`):
            ``"minmod"``, ``"vanleer"`` (default), ``"mc"`` or ``"none"``.
        cfl: The Courant number.
        gravity: The uniform acceleration ``(g_x, g_y, g_z)``.
        prefix: The prefix of the names of the fields the model allocates on the
            hierarchy (``prefix + "density"``, ..., ``prefix + "ax"``, ...).
    """

    def __init__(
        self,
        hierarchy: AMRHierarchy,
        *,
        gamma: float = 5.0 / 3.0,
        mean_field: tuple[float, float, float] = (0.0, 0.0, 0.0),
        resistivity: float = 0.0,
        limiter: str = "vanleer",
        cfl: float = 0.3,
        gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        prefix: str = "mhd_",
    ) -> None:
        """Allocate the fields on ``hierarchy``; set the state with `set_state`."""
        if hierarchy.ndim != 3:
            raise ValueError("MHD3D needs a 3D hierarchy")
        if hierarchy.cylindrical or hierarchy.graded:
            raise NotImplementedError(
                "MHD3D is Cartesian on uniform levels: no graded axes"
            )
        if limiter not in LIMITERS:
            raise ValueError(f"limiter is one of {sorted(LIMITERS)}")
        if not 0.0 < cfl <= 1.0:
            raise ValueError("cfl must be in (0, 1]")
        if gamma <= 1.0:
            raise ValueError("gamma must be larger than 1")
        if not resistivity >= 0.0:
            raise ValueError("resistivity must be non-negative")
        if len(mean_field) != 3 or len(gravity) != 3:
            raise ValueError("mean_field and gravity have three components")
        self.hierarchy = hierarchy
        self.gamma = float(gamma)
        self.mean_field = tuple(float(b) for b in mean_field)
        self.resistivity = float(resistivity)
        self.limiter = limiter
        self.cfl = float(cfl)
        self.gravity = tuple(float(g) for g in gravity)
        self.names = {name: prefix + name for name in CONSERVED_3D}
        self.potential = tuple(prefix + name for name, _ in _POTENTIAL)
        for name in CONSERVED_3D:
            hierarchy.allocate_field(
                self.names[name], halo=HALO, split=None, conservative=True
            )
        for name, (_, centering) in zip(self.potential, _POTENTIAL, strict=True):
            hierarchy.allocate_field(name, halo=HALO, split=None, centering=centering)
        from patchwork import FluxRegister

        self.periodic = tuple(hierarchy.periodic)
        self.register = FluxRegister(hierarchy)
        self.time = 0.0
        self.steps = 0
        #: The period of the wave `alfven_wave` sets up (NaN otherwise).
        self.period = float("nan")
        self._ready = False

    # Construction of common cases ---------------------------------------------#
    @classmethod
    def alfven_wave(
        cls, cells: int, *, amplitude: float = 0.1, **options: Any
    ) -> tuple[MHD3D, xr.Dataset]:
        """A circularly polarized Alfvén wave oblique to all three axes, and its exact state.

        A periodic box of ``2 cells x cells x cells`` cells over ``[0, 2] x [0, 1] x [0,
        1]``; the phase is ``2 pi (x / 2 + y + z)``, so the wave travels along ``(1, 2,
        2) / 3`` with wavelength ``2 / 3``. Density 1, pressure 0.1, a parallel field of
        1 (the Alfvén speed is 1) and a perpendicular field and velocity of
        ``amplitude``. The exact state after a whole number of periods (`period`) is the
        initial one, which the second item returns.
        """
        from patchwork import AMRHierarchy

        hierarchy = AMRHierarchy(
            (2 * cells, cells, cells),
            ((0.0, 2.0), (0.0, 1.0), (0.0, 1.0)),
            periodic=(True, True, True),
        )
        wavevector = 2.0 * np.pi * np.array([0.5, 1.0, 1.0])
        k = float(np.linalg.norm(wavevector))
        n = wavevector / k
        e1 = np.cross(n, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)

        def phase(x, y, z):
            return wavevector[0] * x + wavevector[1] * y + wavevector[2] * z

        def perpendicular(x, y, z):
            # a Beltrami field: curl of (a / k) (sin e1 + cos e2) is a (sin e1 + cos e2)
            s, c = np.sin(phase(x, y, z)), np.cos(phase(x, y, z))
            return [amplitude * (s * e1[i] + c * e2[i]) for i in range(3)]

        def velocity(x, y, z):  # a wave along +n: v_perp = -B_perp / sqrt(rho)
            return [-b for b in perpendicular(x, y, z)]

        def potential(i):
            return lambda x, y, z: perpendicular(x, y, z)[i] / k

        model = cls(hierarchy, mean_field=tuple(n), **options)
        model.set_state(
            1.0,
            velocity,
            0.1,
            vector_potential=(potential(0), potential(1), potential(2)),
        )
        model.period = 2.0 / 3.0
        exact = model.dataset()
        return model, exact

    @classmethod
    def orszag_tang(cls, cells: int, layers: int = 4, **options: Any) -> MHD3D:
        """The Orszag–Tang vortex of `MHD2D.orszag_tang`, uniform along ``z``.

        A periodic box of ``cells x cells x layers`` cells over ``[0, 1]^2 x [0, layers
        / cells]``. Nothing depends on ``z``, so it must stay the 2D solution.
        """
        from patchwork import AMRHierarchy

        hierarchy = AMRHierarchy(
            (cells, cells, layers),
            ((0.0, 1.0), (0.0, 1.0), (0.0, layers / cells)),
            periodic=(True, True, True),
        )
        model = cls(hierarchy, **options)
        b0 = 1.0 / np.sqrt(4.0 * np.pi)
        model.set_state(
            25.0 / (36.0 * np.pi),
            lambda x, y, z: (
                -np.sin(2 * np.pi * y),
                np.sin(2 * np.pi * x),
                0.0 * x,
            ),
            5.0 / (12.0 * np.pi),
            vector_potential=(
                0.0,
                0.0,
                lambda x, y, z: (
                    b0
                    * (
                        np.cos(4 * np.pi * x) / (4 * np.pi)
                        + np.cos(2 * np.pi * y) / (2 * np.pi)
                    )
                ),
            ),
        )
        return model

    @classmethod
    def blast_wave(cls, cells: int, **options: Any) -> MHD3D:
        """The magnetized blast wave in 3D, ``cells x cells x 1.5 cells`` cells.

        Pressure 10 inside radius 0.1 and 0.1 outside, density 1, in a periodic box
        ``[-0.5, 0.5]^2 x [-0.75, 0.75]`` with a uniform field of 1 along ``(1, 1, 1) /
        sqrt 3``; ``gamma = 5/3`` (Gardiner and Stone 2008). The blast expands along the
        field; the reference time is 0.2. The limiter is ``"minmod"`` by default.
        """
        from patchwork import AMRHierarchy

        hierarchy = AMRHierarchy(
            (cells, cells, (3 * cells) // 2),
            ((-0.5, 0.5), (-0.5, 0.5), (-0.75, 0.75)),
            periodic=(True, True, True),
        )
        field = 1.0 / np.sqrt(3.0)
        options = {"limiter": "minmod", **options}
        model = cls(hierarchy, mean_field=(field, field, field), **options)
        model.set_state(
            1.0,
            (0.0, 0.0, 0.0),
            lambda x, y, z: np.where(x**2 + y**2 + z**2 < 0.01, 10.0, 0.1),
        )
        return model

    # State ------------------------------------------------------------------#
    @staticmethod
    def _evaluate(value: Value, *coordinates: np.ndarray) -> np.ndarray:
        shape = coordinates[0].shape
        if callable(value):
            return np.broadcast_to(np.asarray(value(*coordinates), dtype=float), shape)
        return np.broadcast_to(np.asarray(value, dtype=float), shape)

    def _face_fields(
        self, potential: tuple[np.ndarray, ...], spacing: tuple[float, ...]
    ) -> list[np.ndarray]:
        """The field on the faces normal to each axis from the edge potential.

        ``B_a = B0_a + d_b A_c - d_c A_b`` for the cyclic ``(a, b, c)``: the circulation
        of ``A`` around each face over its area. The arrays may carry halos (the same
        width on all of them).
        """
        faces = []
        for a in range(3):
            _, b, c = _cyclic(a)
            faces.append(
                np.diff(potential[c], axis=b) / spacing[b]
                - np.diff(potential[b], axis=c) / spacing[c]
                + self.mean_field[a]
            )
        return faces

    @staticmethod
    def _cell_fields(faces: list[np.ndarray]) -> list[np.ndarray]:
        """The cell-centered field, the mean of the two faces of each cell along its axis."""
        return [
            0.5 * (_along(f, a, 0, f.shape[a] - 1) + _along(f, a, 1, f.shape[a]))
            for a, f in enumerate(faces)
        ]

    def set_state(
        self,
        density: Value,
        velocity: Value | tuple[Value, Value, Value],
        pressure: Value,
        *,
        vector_potential: tuple[Value, Value, Value] = (0.0, 0.0, 0.0),
    ) -> None:
        """Set the state on every patch.

        Each argument is a number or a function of the coordinates ``(x, y, z)``: of
        the cell centres, of the edge midpoints for the three components of
        ``vector_potential``. ``velocity`` is a function returning the three
        components, or a tuple of three values. The field is ``mean_field`` plus the
        curl of ``vector_potential``.
        """
        hierarchy = self.hierarchy
        gamma = self.gamma
        for patch in hierarchy.local_patches:
            x, y, z = patch.local_coordinates(hierarchy, self.names["density"])
            rho = self._evaluate(density, x, y, z)
            if callable(velocity):
                u, v, w = (np.broadcast_to(c, x.shape) for c in velocity(x, y, z))
            elif isinstance(velocity, tuple | list) and len(velocity) == 3:
                u, v, w = (self._evaluate(c, x, y, z) for c in velocity)
            else:
                raise ValueError("velocity is a function or three values")
            p = self._evaluate(pressure, x, y, z)
            if np.any(rho <= 0.0) or np.any(p <= 0.0):
                raise ValueError("the density and the pressure must be positive")
            potential = []
            for name, value in zip(self.potential, vector_potential, strict=True):
                at = patch.local_coordinates(hierarchy, name)
                patch.field(name).local[...] = self._evaluate(value, *at)
                potential.append(np.asarray(patch.field(name).local))
            bx, by, bz = self._cell_fields(
                self._face_fields(tuple(potential), hierarchy.spacing(patch.level))
            )
            values = {
                "density": rho,
                "momentum_x": rho * u,
                "momentum_y": rho * v,
                "momentum_z": rho * w,
                "energy": p / (gamma - 1.0)
                + 0.5 * rho * (u**2 + v**2 + w**2)
                + 0.5 * (bx**2 + by**2 + bz**2),
            }
            for name, value in values.items():
                patch.field(self.names[name]).local[...] = value
        for name in (*self.names.values(), *self.potential):
            hierarchy.restrict(name)
        self.time = 0.0
        self.steps = 0
        self._ready = True

    def _check_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("set the state first (set_state)")

    # Halos and boundaries ---------------------------------------------------#
    @profile("carafe.mhd3d_halos", tags=_PROFILE_TAGS)
    def _fill_halos(self) -> None:
        """Halos from the hierarchy, then the outflow faces of the domain."""
        hierarchy = self.hierarchy
        for name in self.names.values():
            hierarchy.exchange_halos(name)
        for name in self.potential:
            hierarchy.exchange_halos(name, interpolation="quadratic")
        for patch in hierarchy.local_patches:
            shape = hierarchy.level_shape(patch.level)
            for axis in range(3):
                if self.periodic[axis]:
                    continue
                lower = patch.box[axis][0] == 0
                upper = patch.box[axis][1] == shape[axis]
                if not (lower or upper):
                    continue
                for name in self.names.values():
                    _outflow_cells(
                        patch.field(name).local_with_halos, axis, lower, upper
                    )
                for name in self.potential:
                    _outflow_nodes(
                        patch.field(name).local_with_halos, axis, lower, upper
                    )

    # The right-hand side -------------------------------------------------------#
    def _primitive(
        self, patch: Patch
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """The primitive variables of a haloed block, its face fields and cell fields.

        ``prim`` is ``(8, X, Y, Z)``: density, the three velocities, the pressure and the
        three field components (cell means of the faces).
        """
        spacing = self.hierarchy.spacing(patch.level)
        potential = tuple(
            np.asarray(patch.field(name).local_with_halos) for name in self.potential
        )
        faces = self._face_fields(potential, spacing)
        cells = self._cell_fields(faces)
        rho, mx, my, mz, energy = (
            np.asarray(patch.field(self.names[name]).local_with_halos)
            for name in CONSERVED_3D
        )
        u, v, w = mx / rho, my / rho, mz / rho
        bx, by, bz = cells
        p = (self.gamma - 1.0) * (
            energy - 0.5 * rho * (u**2 + v**2 + w**2) - 0.5 * (bx**2 + by**2 + bz**2)
        )
        prim = np.stack((rho, u, v, w, p, bx, by, bz))
        h = HALO
        block = (slice(None), *(slice(h, -h),) * 3)
        inner = prim[block]
        if not (
            np.all(np.isfinite(inner)) and np.all(inner[0] > 0) and np.all(inner[4] > 0)
        ):
            raise FloatingPointError(
                f"non-positive density or pressure on the level-{patch.level} patch "
                f"{patch.box} at t = {self.time}"
            )
        return prim, faces, cells

    def _sweep(self, prim: np.ndarray, face: np.ndarray, axis: int) -> np.ndarray:
        """HLLD fluxes through the faces normal to ``axis`` of a haloed block.

        In the frame ``(axis, b, c)`` of `_cyclic`: ``(7, N - 3, N_b, N_c)`` at the
        block faces ``0 .. n`` along ``axis`` and every transverse cell of the haloed
        block, rows density, normal momentum, the momenta along ``b`` and ``c``,
        energy, ``B_b`` and ``B_c``.
        """
        frame = _cyclic(axis)
        rows = [0, *(1 + a for a in frame), 4, *(5 + a for a in frame)]
        moved = prim[rows].transpose(0, *(1 + a for a in frame))
        n, nb, nc = moved.shape[1:]
        lines = np.ascontiguousarray(moved.reshape(8, n, nb * nc))
        normal = np.ascontiguousarray(face.transpose(frame).reshape(n + 1, nb * nc))
        flux = np.empty((7, n - 3, nb * nc))
        code = LIMITER_CODES.index(self.limiter)
        mhd_sweep(lines, normal, 0, code, self.gamma, flux)
        return flux.reshape(7, n - 3, nb, nc)

    def _edge_field(
        self,
        prim: np.ndarray,
        sweeps: list[np.ndarray],
        patch: Patch,
    ) -> list[np.ndarray]:
        """The electric field on the block's edges along each axis, by CT-contact.

        For the edges along ``c`` and ``(a, b) = (c + 1, c + 2)``: the ``a``-sweep's
        flux of ``B_b`` is ``-E_c``, the ``b``-sweep's flux of ``B_a`` is ``+E_c``; the
        ``ct_contact`` kernel averages them on each plane normal to ``c``.
        """
        h = HALO
        fields = []
        for c in range(3):
            _, a, b = _cyclic(c)
            # E_c in the cells, in the frame (c, a, b)
            cell = prim[5 + a] * prim[1 + b] - prim[5 + b] * prim[1 + a]
            frame = (c, a, b)
            cell = np.ascontiguousarray(cell.transpose(frame))
            # the a-sweep (frame (a, b, c)) to (c, row, a, b); the b-sweep (frame (b, c, a))
            fa = np.ascontiguousarray(sweeps[a][[0, 5]].transpose(3, 0, 1, 2))
            fb = np.ascontiguousarray(sweeps[b][[0, 6]].transpose(2, 0, 3, 1))
            na, nb, nc = (patch.shape[i] for i in frame[1:] + frame[:1])
            out = np.empty((nc, na + 1, nb + 1))
            for k in range(nc):
                kc = h + k
                ct_contact(fa[kc, 1], fb[kc, 1], fa[kc, 0], fb[kc, 0], cell[kc], out[k])
            fields.append(out.transpose(np.argsort(frame)))
        return fields

    def _resistive(
        self,
        faces: list[np.ndarray],
        patch: Patch,
        edges: list[np.ndarray],
        fluxes: list[np.ndarray],
    ) -> None:
        """Add ``eta J`` to the edge field and ``eta J x B`` to the energy fluxes.

        ``J_c = d_a B_b - d_b B_a`` on the block's c-edges from the haloed face fields;
        on a face normal to ``s`` the energy flux gains ``eta (J_b B_c - J_c B_b)``,
        with ``J`` the mean of the face's two edges along each direction and ``B`` the
        mean of the four faces around it. ``fluxes`` are the energy fluxes of the
        block faces, in the physical order of the axes.
        """
        h = HALO
        eta = self.resistivity
        spacing = self.hierarchy.spacing(patch.level)
        shape = patch.shape
        current = []
        for c in range(3):
            _, a, b = _cyclic(c)
            parts = []
            for axis, field, sign in ((a, faces[b], 1.0), (b, faces[a], -1.0)):
                derivative = np.diff(field, axis=axis) / spacing[axis]
                index = [slice(h, h + shape[i]) for i in range(3)]
                index[a] = slice(h, h + shape[a] + 1)
                index[b] = slice(h, h + shape[b] + 1)
                index[axis] = slice(h - 1, h + shape[axis])
                parts.append(sign * derivative[tuple(index)])
            current.append(parts[0] + parts[1])
        for c in range(3):
            edges[c] += eta * current[c]
        for s in range(3):
            _, b, c = _cyclic(s)
            # J along b and c at the s-faces: the mean of the face's two edges
            jb = _pair_mean(current[b], c, 0, shape[c])  # b-edges: nodes along s, c
            jc = _pair_mean(current[c], b, 0, shape[b])  # c-edges: nodes along s, b
            fluxes[s] += eta * (
                jb * _field_at_face(faces[c], s, c, shape)
                - jc * _field_at_face(faces[b], s, b, shape)
            )

    @profile("carafe.mhd3d_fluxes", tags=_PROFILE_TAGS)
    def _patch_fluxes(
        self, patch: Patch
    ) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
        """The conserved fluxes of a patch's block faces and the field on its edges.

        The fluxes by field name, one array per axis in the physical order of the axes
        (``(nx + 1, ny, nz)`` for x, ...), as `patchwork.FluxRegister` takes them; the
        electric field on the block's x-, y- and z-edges.
        """
        prim, faces, _ = self._primitive(patch)
        sweeps = [self._sweep(prim, faces[s], s) for s in range(3)]
        edges = self._edge_field(prim, sweeps, patch)
        h = HALO
        shape = patch.shape
        fluxes: dict[str, list[np.ndarray]] = {name: [] for name in CONSERVED_3D}
        for s in range(3):
            frame = _cyclic(s)
            block = sweeps[s][
                :, :, h : h + shape[frame[1]], h : h + shape[frame[2]]
            ].transpose(0, *(1 + i for i in np.argsort(frame)))
            fluxes["density"].append(block[0])
            for row, axis in zip((1, 2, 3), frame, strict=True):
                fluxes[("momentum_x", "momentum_y", "momentum_z")[axis]].append(
                    block[row]
                )
            fluxes["energy"].append(np.array(block[4]))
        if self.resistivity:
            self._resistive(faces, patch, edges, fluxes["energy"])
        return fluxes, edges

    def _advance(self, dt: float, scale: float) -> None:
        """One forward-Euler stage of every patch, recorded with ``scale``."""
        hierarchy = self.hierarchy
        self._fill_halos()
        results = {
            id(patch): self._patch_fluxes(patch) for patch in hierarchy.local_patches
        }
        g = self.gravity
        sources = {}
        if any(g):
            for patch in hierarchy.local_patches:
                rho = np.array(patch.field(self.names["density"]).local)
                momentum = [
                    np.asarray(patch.field(self.names[f"momentum_{a}"]).local)
                    for a in "xyz"
                ]
                work = sum(m * ga for m, ga in zip(momentum, g, strict=True))
                sources[id(patch)] = (rho, np.array(work))
        for name in CONSERVED_3D:
            self.register.update(
                self.names[name],
                lambda patch, name=name: tuple(results[id(patch)][0][name]),
                dt,
                scale=scale,
            )
        for patch in hierarchy.local_patches:
            edges = results[id(patch)][1]
            for name, field in zip(self.potential, edges, strict=True):
                patch.field(name).local[...] -= dt * field
            if any(g):
                rho, work = sources[id(patch)]
                for a, ga in zip("xyz", g, strict=True):
                    if ga:
                        patch.field(self.names[f"momentum_{a}"]).local[...] += (
                            dt * ga * rho
                        )
                patch.field(self.names["energy"]).local[...] += dt * work

    # Time stepping -------------------------------------------------------------#
    @profile("carafe.mhd3d_stable_step", tags=_PROFILE_TAGS)
    def stable_step(self) -> float:
        """The largest stable step: ``cfl`` over the sum of the fast-wave rates of a cell.

        ``sum_a (|u_a| + c_f,a) / dx_a`` over the axes; with resistivity also ``cfl``
        times the explicit diffusion limit ``1 / (2 eta sum_a dx_a^-2)`` of the finest
        cells.
        """
        self._check_ready()
        hierarchy = self.hierarchy
        smallest = np.inf
        for patch in hierarchy.local_patches:
            spacing = hierarchy.spacing(patch.level)
            potential = tuple(
                np.asarray(patch.field(name).local) for name in self.potential
            )
            bx, by, bz = self._cell_fields(self._face_fields(potential, spacing))
            field = (bx, by, bz)
            rho, *momentum, energy = (
                np.asarray(patch.field(self.names[name]).local) for name in CONSERVED_3D
            )
            velocity = [m / rho for m in momentum]
            p = (self.gamma - 1.0) * (
                energy
                - 0.5 * rho * sum(v**2 for v in velocity)
                - 0.5 * sum(b**2 for b in field)
            )
            rate = sum(
                (
                    np.abs(velocity[a])
                    + _fast_speed(
                        rho,
                        p,
                        field[a],
                        field[(a + 1) % 3],
                        field[(a + 2) % 3],
                        self.gamma,
                    )
                )
                / spacing[a]
                for a in range(3)
            )
            smallest = min(smallest, 1.0 / float(np.max(rate)))
            if self.resistivity:
                smallest = min(
                    smallest,
                    0.5 / (self.resistivity * sum(d**-2 for d in spacing)),
                )
        from maybempi import MPI

        return self.cfl * float(hierarchy.comm.allreduce(smallest, op=MPI.MIN))

    @profile("carafe.mhd3d_step", tags=_PROFILE_TAGS)
    def step(self, dt: float | None = None) -> float:
        """Advance by one step of Heun's method (SSP-RK2); return the step taken."""
        self._check_ready()
        if dt is None:
            dt = self.stable_step()
        hierarchy = self.hierarchy
        names = [*self.names.values(), *self.potential]
        start = {
            (id(patch), name): np.array(patch.field(name).local)
            for patch in hierarchy.local_patches
            for name in names
        }
        self._advance(dt, 0.5 * dt)
        for name in self.potential:
            hierarchy.restrict(name)
        self._advance(dt, 0.5 * dt)
        for patch in hierarchy.local_patches:
            for name in names:
                array = patch.field(name).local
                array += start[id(patch), name]
                array *= 0.5
        for name in self.names.values():
            self.register.reflux(name)
        for name in self.potential:
            hierarchy.restrict(name)
        self.time += dt
        self.steps += 1
        return dt

    @profile("carafe.mhd3d_run", tags=_PROFILE_TAGS)
    def run(self, duration: float, *, max_steps: int = 1_000_000) -> int:
        """Advance by ``duration`` (the last step shortened to land on it); return the steps."""
        self._check_ready()
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("duration must be a finite, non-negative time")
        end = self.time + duration
        taken = 0
        while self.time < end * (1.0 - 1e-14) and taken < max_steps:
            dt = min(self.stable_step(), end - self.time)
            self.step(dt)
            taken += 1
        return taken

    # Refinement ----------------------------------------------------------------#
    def current_tags(self, threshold: float, levels: int) -> dict[int, np.ndarray]:
        """Tag the cells where ``|J|`` times the cell size exceeds ``threshold``.

        ``|J|`` is that of the cell-centered field's curl (central differences). Levels
        ``0 .. levels - 1`` that exist are tagged, from the patches this rank holds;
        the result is for `regrid`.
        """
        self._check_ready()
        hierarchy = self.hierarchy
        self._fill_halos()
        h = HALO
        tags = {
            level: np.zeros(hierarchy.level_shape(level), dtype=bool)
            for level in range(min(levels, hierarchy.max_level + 1))
        }
        for patch in hierarchy.local_patches:
            if patch.level not in tags:
                continue
            spacing = hierarchy.spacing(patch.level)
            _, _, cells = self._primitive(patch)
            square = 0.0
            for c in range(3):
                _, a, b = _cyclic(c)
                jc = np.gradient(cells[b], spacing[a], axis=a) - np.gradient(
                    cells[a], spacing[b], axis=b
                )
                square = square + jc**2
            size = np.sqrt(square)[(slice(h, -h),) * 3]
            index = tuple(slice(lo, hi) for lo, hi in patch.box)
            tags[patch.level][index] |= size * max(spacing) > threshold
        return tags

    def regrid(self, tags: dict[int, Any], **options: Any) -> None:
        """Regrid on ``tags`` (`patchwork.AMRHierarchy.regrid`, which takes ``options``).

        The conserved fields keep their integrals, the potential is interpolated and
        restricted (the field stays divergence-free); the flux register is rebuilt.
        """
        self._check_ready()
        from patchwork import FluxRegister

        hierarchy = self.hierarchy.regrid(tags, **options)
        for name in self.potential:
            hierarchy.restrict(name)
        self.hierarchy = hierarchy
        self.register = FluxRegister(hierarchy)

    # Diagnostics ---------------------------------------------------------------#
    def _own_faces(self, patch: Patch) -> list[np.ndarray]:
        """The face fields of a patch's own faces (gathered, collective over its ranks)."""
        potential = tuple(self.hierarchy.gather(patch, name) for name in self.potential)
        return self._face_fields(potential, self.hierarchy.spacing(patch.level))

    def divergence(self) -> float:
        """The largest ``|div B|`` times the cell size over the cells of all patches.

        The net flux out of each cell: zero to round-off by construction.
        """
        self._check_ready()
        largest = 0.0
        for patch in self.hierarchy.local_patches:
            spacing = self.hierarchy.spacing(patch.level)
            potential = tuple(
                np.asarray(patch.field(name).local) for name in self.potential
            )
            faces = self._face_fields(potential, spacing)
            div = sum(np.diff(f, axis=a) / spacing[a] for a, f in enumerate(faces))
            largest = max(largest, float(np.max(np.abs(div))) * min(spacing))
        from maybempi import MPI

        return float(self.hierarchy.comm.allreduce(largest, op=MPI.MAX))

    def composite_divergence(self) -> float:
        """The largest mismatch of a coarse face's flux and the fine fluxes through it.

        Over the coarse faces under every finer patch (those on its boundary are the
        faces of the coarse leaf cells next to it), relative to the largest face
        flux: zero to round-off, so the divergence of the composite mesh is too.
        Collective.
        """
        self._check_ready()
        hierarchy = self.hierarchy
        r = hierarchy.refinement_ratio
        worst, scale = 0.0, 0.0
        for fine in hierarchy.patches:
            if fine.level == 0:
                continue
            spacing = hierarchy.spacing(fine.level)
            fine_faces = self._own_faces(fine)
            for coarse in hierarchy.patches_at_level(fine.level - 1):
                coarse_spacing = hierarchy.spacing(coarse.level)
                coarse_faces = self._own_faces(coarse)
                for a in range(3):
                    _, b, c = _cyclic(a)
                    flux = fine_faces[a] * spacing[b] * spacing[c]
                    scale = max(scale, float(np.max(np.abs(flux))))
                    summed = flux[
                        tuple(
                            slice(None, None, r) if i == a else slice(None)
                            for i in range(3)
                        )
                    ]
                    summed = summed.reshape(
                        [
                            part
                            for i, n in enumerate(summed.shape)
                            for part in ((n, 1) if i == a else (n // r, r))
                        ]
                    ).sum(axis=(1, 3, 5))
                    under = [
                        (lo // r, hi // r + (i == a))
                        for i, (lo, hi) in enumerate(fine.box)
                    ]
                    own = [(lo, hi + (i == a)) for i, (lo, hi) in enumerate(coarse.box)]
                    part = [
                        (max(u0, o0), min(u1, o1))
                        for (u0, u1), (o0, o1) in zip(under, own, strict=True)
                    ]
                    if any(lo >= hi for lo, hi in part):
                        continue
                    coarse_flux = (
                        coarse_faces[a] * coarse_spacing[b] * coarse_spacing[c]
                    )
                    got = coarse_flux[
                        tuple(
                            slice(lo - o, hi - o)
                            for (lo, hi), (o, _) in zip(part, own, strict=True)
                        )
                    ]
                    want = summed[
                        tuple(
                            slice(lo - u, hi - u)
                            for (lo, hi), (u, _) in zip(part, under, strict=True)
                        )
                    ]
                    worst = max(worst, float(np.max(np.abs(got - want))))
        return worst / scale if scale else 0.0

    def totals(self) -> dict[str, float]:
        """Integrals of the conserved fields over the domain, from the finest cells."""
        self._check_ready()
        return {
            name: self.hierarchy.integral(self.names[name]) for name in CONSERVED_3D
        }

    def dataset(self, level: int = 0) -> xr.Dataset:
        """The primitive variables of ``level`` as a dataset over its cell centres.

        The patches of the level are gathered into one array of the level's shape,
        NaN where the level has no patch. Collective.
        """
        self._check_ready()
        hierarchy = self.hierarchy
        shape = hierarchy.level_shape(level)
        spacing = hierarchy.spacing(level)
        axes = [
            hierarchy.lower_bound[a] + (np.arange(shape[a]) + 0.5) * spacing[a]
            for a in range(3)
        ]
        names = (*CONSERVED_3D, "bx", "by", "bz")
        fields = {name: np.full(shape, np.nan) for name in names}
        for patch in hierarchy.patches_at_level(level):
            index = tuple(slice(lo, hi) for lo, hi in patch.box)
            for name in CONSERVED_3D:
                fields[name][index] = hierarchy.gather(patch, self.names[name])
            cells = self._cell_fields(self._own_faces(patch))
            for name, value in zip(("bx", "by", "bz"), cells, strict=True):
                fields[name][index] = value
        rho = fields["density"]
        u, v, w = (fields[f"momentum_{a}"] / rho for a in "xyz")
        b2 = fields["bx"] ** 2 + fields["by"] ** 2 + fields["bz"] ** 2
        p = (self.gamma - 1.0) * (
            fields["energy"] - 0.5 * rho * (u**2 + v**2 + w**2) - 0.5 * b2
        )
        dims = ("x", "y", "z")
        data = {
            "density": variable(dims, rho, r"$\rho$"),
            "velocity_x": variable(dims, u, r"$u_x$"),
            "velocity_y": variable(dims, v, r"$u_y$"),
            "velocity_z": variable(dims, w, r"$u_z$"),
            "pressure": variable(dims, p, r"$p$"),
            "magnetic_field_x": variable(dims, fields["bx"], r"$B_x$"),
            "magnetic_field_y": variable(dims, fields["by"], r"$B_y$"),
            "magnetic_field_z": variable(dims, fields["bz"], r"$B_z$"),
            "magnetic_pressure": variable(dims, 0.5 * b2, r"$|\mathbf{B}|^2/2$"),
        }
        coords = {
            name: variable(name, values, f"${name}$")
            for name, values in zip(dims, axes, strict=True)
        }
        return xr.Dataset(
            data,
            coords=coords,
            attrs={"time": self.time, "level": level, "gamma": self.gamma},
        )
