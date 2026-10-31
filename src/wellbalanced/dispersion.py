"""Local linear theory of the gradient-drift (Simon-Hoh) instability with kinetic ions.

The first instability of a magnetron's ionisation region (and of a Hall thruster's channel) is
the gradient-drift instability: magnetised electrons convected by ``E x B`` along a density
gradient, unmagnetised ions accelerated by the perturbed field. The cold-ion fluid version is
in :func:`wellbalanced.driftfluid2d.boeuf_dispersion_roots` and
:func:`wellbalanced.driftfluid2d.gradient_drift_roots`. This module adds what those cannot
resolve:

- **kinetic ions**: the cold-ion mode has a lab-frame phase speed of a few 100 m/s, inside
  the thermal spread of room-temperature ions, so ion Landau damping and the Doppler shift of
  the ion flow set its real frequency (:class:`KineticIons`);
- **the compressibility of the drifts** in an inhomogeneous ``B``: for the 2D model field
  ``B(x) z`` of the planar simulations the ``E x B`` flow diverges as ``-v_x B'/B``; for a
  curl-free (vacuum) field the factor is 2, which is what a 3D arch adds to the 2D plane
  (``curl_free_field``);
- **electron inertia and collisions** kept exactly in the local momentum balance (the
  ``1 + k^2 rho_e^2`` of Boeuf's relation), with the equilibrium flow shear;
- **Poisson or quasineutral** closure, an ionisation source, and a scan along a 1D profile.

Geometry and signs follow :mod:`wellbalanced.driftfluid2d`: ``x`` across the field (toward
the anode), ``y`` along ``E x B`` for ``E_0 < 0`` and ``B > 0``, ``z`` along ``B``;
perturbations ``exp(i (k_x x + k_y y) - i omega t)``, so ``Im omega > 0`` grows and
``Re omega / k_y`` is the phase speed along ``+y``. Gradient lengths are signed, ``L = f / f'``,
negative when ``f`` falls toward the anode. Temperatures in eV, fields in V/m and T.

Where the two models can be compared (uniform ``B``, no collisions, Poisson) the roots agree
with :func:`~wellbalanced.driftfluid2d.boeuf_dispersion_roots` to better than 0.1 %. With a
``B`` gradient they do not: Boeuf's relation carries the grad-B frequency
``omega_D = -2 k T_e / (B L_B)`` in both its Doppler shift and its drive, while the drift
fluid in the model field ``B(x) z`` gives half of it (``div(E x B / B^2) = -2 v . grad B / B
- E . curl B / B^2``, whose second term cancels half when ``curl B != 0``). At Boeuf's own
point (6 mT, ``L_B = -6`` mm) this is the difference between his ``1e7`` /s and near
marginality here; the 2D PIC with a varied ``L_B`` is the test. The derivation and the
predictions this serves are in the spokes3d transport theory notes
(``resources/spokes3d/transport_theory.md`` of the todo repository, Sec. 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy.special import wofz

__all__ = [
    "ColdIons",
    "KineticIons",
    "LocalDispersion",
    "LocalRoots",
    "Plasma",
    "plasma_dispersion_function",
    "scan_profile",
]

E = ELEMENTARY_CHARGE
EPSILON_0 = 8.8541878128e-12
ARGON_MASS = 39.948 * ATOMIC_MASS - ELECTRON_MASS


def plasma_dispersion_function(zeta: complex | np.ndarray) -> complex | np.ndarray:
    """``Z(zeta) = i sqrt(pi) w(zeta)``, the plasma dispersion function, on the whole plane.

    ``w`` is the Faddeeva function (:func:`scipy.special.wofz`), which is the analytic
    continuation of ``Z`` to ``Im zeta < 0``, so damped roots are found as well as growing
    ones. Beyond ``|zeta| = 30`` the asymptotic series ``-1/zeta (1 + 1/(2 zeta^2) + 3/(4
    zeta^4))`` is used without the ``exp(-zeta^2)`` term of the lower half-plane, which
    overflows there and only matters for modes damped far faster than any of interest.
    """
    zeta = np.asarray(zeta, dtype=complex)
    large = np.abs(zeta) > 30.0
    small, big = np.where(large, 1.0, zeta), np.where(large, zeta, 1.0)
    series = -1.0 / big * (1.0 + 0.5 / big**2 + 0.75 / big**4)
    result = np.where(large, series, 1j * np.sqrt(np.pi) * wofz(small))
    return result if result.ndim else complex(result)


@dataclass(frozen=True)
class Plasma:
    """The local equilibrium: fields, electrons and their gradients.

    ``density`` (m^-3), ``e_field`` ``E_0`` (V/m, signed along ``x``), ``b_field`` (T, along
    ``z``), ``electron_temperature`` (eV); the signed lengths ``density_length`` ``n / n'``,
    ``field_length`` ``B / B'`` and ``e_field_length`` ``E_0 / E_0'`` (``inf`` for uniform);
    ``electron_collision_frequency`` (1/s) and ``ionization_frequency`` (1/s, the source
    ``nu_iz n_e`` of both species). ``curl_free_field`` doubles the ``E x B`` and
    diamagnetic compressibility from ``B / B'``: the vacuum (3D arch) field instead of the
    2D model field ``B(x) z`` (see the module docstring).
    """

    density: float
    e_field: float
    b_field: float
    electron_temperature: float
    density_length: float
    field_length: float = np.inf
    e_field_length: float = np.inf
    electron_collision_frequency: float = 0.0
    ionization_frequency: float = 0.0
    curl_free_field: bool = False

    @property
    def omega_ce(self) -> float:
        """Electron cyclotron frequency (rad/s)."""
        return E * self.b_field / ELECTRON_MASS

    @property
    def debye_length(self) -> float:
        """Electron Debye length (m)."""
        return float(
            np.sqrt(EPSILON_0 * self.electron_temperature / (self.density * E))
        )

    @property
    def exb_velocity(self) -> float:
        """``-E_0 / B``, the ``E x B`` drift along ``y``."""
        return -self.e_field / self.b_field

    @property
    def electron_flow(self) -> tuple[float, float]:
        """Equilibrium electron velocity ``(v_0x, v_0y)``: ``E x B`` + diamagnetic, Pedersen."""
        te = self.electron_temperature  # eV, so T_e / e = te volts
        v0y = -(self.e_field + te / self.density_length) / self.b_field
        v0x = -(self.electron_collision_frequency / self.omega_ce) * v0y
        return v0x, v0y

    @property
    def flow_shear(self) -> float:
        """``d v_0y / dx`` from the gradients of ``B`` and ``E_0``."""
        _, v0y = self.electron_flow
        return -v0y / self.field_length - self.exb_velocity / self.e_field_length


@dataclass(frozen=True)
class ColdIons:
    """Cold fluid ions: ``mass`` (kg), flow ``(v_0x, v_0y)`` (m/s), ``collision_frequency`` (1/s)."""

    mass: float = ARGON_MASS
    flow: tuple[float, float] = (0.0, 0.0)
    collision_frequency: float = 0.0

    def response(
        self, omega: complex, kx: float, ky: float, plasma: Plasma, a_e: complex
    ) -> complex:
        """``delta n_i / n_0`` for ``e delta phi / T_e = 1``, given the electron ``a_e``.

        Linearised continuity and momentum of unmagnetised ions, with the density gradient
        convected by the perturbed velocity and the ionisation source ``nu_iz a_e``.
        """
        te = plasma.electron_temperature * E
        kv = kx * self.flow[0] + ky * self.flow[1]
        denom = omega - kv + 1j * self.collision_frequency
        # delta v_i = (e / m) k delta phi / (omega - k.v_0 + i nu), delta phi = T_e / e
        vx = (te / self.mass) * kx / denom
        vy = (te / self.mass) * ky / denom
        div = 1j * (kx * vx + ky * vy) + vx / plasma.density_length
        return (div - plasma.ionization_frequency * a_e) / (1j * (omega - kv))


@dataclass(frozen=True)
class KineticIons:
    """Maxwellian unmagnetised ions: ``mass`` (kg), ``temperature`` (eV), flow ``(v_0x, v_0y)``.

    ``delta n_i / n_0 = -(e delta phi / T_i) [1 + zeta Z(zeta)]`` with
    ``zeta = (omega - k . v_0) / (sqrt(2) k v_ti)``: the Landau resonance with the ion thermal
    motion and the Doppler shift of the flow, no collisions, no gradient convection.
    """

    mass: float = ARGON_MASS
    temperature: float = 0.026
    flow: tuple[float, float] = (0.0, 0.0)

    def response(
        self, omega: complex, kx: float, ky: float, plasma: Plasma, a_e: complex
    ) -> complex:
        """``delta n_i / n_0`` for ``e delta phi / T_e = 1`` (``a_e`` unused: no source)."""
        del a_e
        k = np.hypot(kx, ky)
        vti = np.sqrt(self.temperature * E / self.mass)
        zeta = (omega - kx * self.flow[0] - ky * self.flow[1]) / (
            np.sqrt(2.0) * k * vti
        )
        ratio = plasma.electron_temperature / self.temperature
        return -ratio * (1.0 + zeta * plasma_dispersion_function(zeta))


@dataclass(frozen=True)
class LocalRoots:
    """Roots at one ``(k_x, k_y)``: ``omega`` sorted by growth rate, the fastest first."""

    kx: float
    ky: float
    omega: np.ndarray

    @property
    def growth_rate(self) -> float:
        """``Im omega`` of the fastest root (1/s); ``nan`` when no root was found."""
        return float(self.omega[0].imag) if self.omega.size else np.nan

    @property
    def phase_speed(self) -> float:
        """``Re omega / k_y`` of the fastest root (m/s, along ``+y``); ``nan`` without roots."""
        return float(self.omega[0].real / self.ky) if self.omega.size else np.nan


@dataclass(frozen=True)
class LocalDispersion:
    """The local dispersion relation: ``plasma`` and electrons, ``ions``, the closure.

    ``quasineutral=False`` closes with Poisson's equation
    ``k^2 lambda_D^2 (e delta phi / T_e) = a_i - a_e``; ``True`` with ``a_i = a_e``.
    """

    plasma: Plasma
    ions: ColdIons | KineticIons = ColdIons()
    quasineutral: bool = False

    def _electron_velocity(
        self,
        omega: complex,
        kx: float,
        ky: float,
        a: complex,
        b_field: float,
        e_field: float,
    ) -> tuple[complex, complex]:
        """``(delta v_x, delta v_y)`` of the electrons from the perpendicular momentum balance.

        ``e delta phi / T_e = 1`` and ``delta n / n_0 = a``; ``b_field`` and ``e_field`` are
        passed explicitly so the compressibility can be taken by differentiating in them.
        Electron inertia with the equilibrium-flow advection and shear, the Lorentz force,
        the isothermal pressure and collisions: a 2 x 2 complex solve.
        """
        p = self.plasma
        te = p.electron_temperature * E
        plasma = Plasma(
            p.density, e_field, b_field, p.electron_temperature, p.density_length,
            p.field_length, p.e_field_length, p.electron_collision_frequency,
        )  # fmt: skip
        v0x, v0y = plasma.electron_flow
        shear = plasma.flow_shear
        omega_ce = E * b_field / ELECTRON_MASS
        nu = p.electron_collision_frequency
        wt = omega - kx * v0x - ky * v0y
        # the pressure force per particle, -grad p / n, linearised with dn = n_0(x) a: the
        # n_0' a of grad(dn) cancels the grad(p_0) dn / n_0^2 term, leaving -i k T a
        # m (-i wt) dv_x = i e k_x dphi - e B dv_y - i k_x T a - m nu dv_x
        # m (-i wt) dv_y + m shear dv_x = i e k_y dphi + e B dv_x - i k_y T a - m nu dv_y
        diag = -1j * wt + nu
        matrix = np.array([[diag, omega_ce], [-omega_ce + shear, diag]], dtype=complex)
        force_x = 1j * kx * te * (1.0 - a)
        force_y = 1j * ky * te * (1.0 - a)
        rhs = np.array([force_x, force_y], dtype=complex) / ELECTRON_MASS
        vx, vy = np.linalg.solve(matrix, rhs)
        return complex(vx), complex(vy)

    def electron_response(
        self, omega: complex, kx: float, ky: float
    ) -> tuple[complex, complex]:
        """``(a_e, delta v_x)``: ``delta n_e / n_0`` for ``e delta phi / T_e = 1``, and the velocity.

        The continuity equation with the divergence of the perturbed flux taken as
        ``(i k_x + 1 / L_n) delta v_x + (d delta v_x / dB) B' + (d delta v_x / dE_0) E_0'``
        along ``x`` and ``i k_y delta v_y`` along ``y``: the parameter derivatives give the
        compressibility of every drift in the 2D model field exactly; ``curl_free_field``
        adds the second ``-v_x / L_B`` of the ``E x B`` and diamagnetic drifts of a vacuum
        field. Since ``delta v`` is linear in ``a``, the divergence is ``D_0 + D_a a`` and
        ``a_e`` follows from ``(-i omega - nu_iz + i k_y v_0y + ...) a + D_0 + D_a a = 0``.
        """
        p = self.plasma
        v0x, v0y = p.electron_flow

        def divergence(a: complex) -> complex:
            vx, vy = self._electron_velocity(omega, kx, ky, a, p.b_field, p.e_field)
            div = (1j * kx + 1.0 / p.density_length) * vx + 1j * ky * vy
            if np.isfinite(p.field_length):
                db = 1e-5 * p.b_field
                vxp, _ = self._electron_velocity(
                    omega, kx, ky, a, p.b_field + db, p.e_field
                )
                vxm, _ = self._electron_velocity(
                    omega, kx, ky, a, p.b_field - db, p.e_field
                )
                div += (vxp - vxm) / (2 * db) * p.b_field / p.field_length
                if p.curl_free_field:
                    te = p.electron_temperature  # volts
                    v_exb = -1j * ky * te / p.b_field
                    v_dia = 1j * ky * te * a / p.b_field
                    div -= (v_exb + v_dia) / p.field_length
            if np.isfinite(p.e_field_length) and p.e_field != 0.0:
                de = 1e-5 * abs(p.e_field)
                vxp, _ = self._electron_velocity(
                    omega, kx, ky, a, p.b_field, p.e_field + de
                )
                vxm, _ = self._electron_velocity(
                    omega, kx, ky, a, p.b_field, p.e_field - de
                )
                div += (vxp - vxm) / (2 * de) * p.e_field / p.e_field_length
            # the equilibrium Pedersen flow convecting the perturbation
            div += (1j * kx + 1.0 / p.density_length) * v0x * a + 1j * ky * v0y * a
            return div

        d0 = divergence(0.0)
        da = divergence(1.0) - d0
        a_e = -d0 / (-1j * omega - p.ionization_frequency + da)
        vx, _ = self._electron_velocity(omega, kx, ky, a_e, p.b_field, p.e_field)
        return a_e, vx

    def residual(self, omega: complex, kx: float, ky: float) -> complex:
        """``a_i - a_e - k^2 lambda_D^2`` (Poisson) or ``a_i - a_e`` (quasineutral); zero at a root."""
        a_e, _ = self.electron_response(omega, kx, ky)
        a_i = self.ions.response(omega, kx, ky, self.plasma, a_e)
        closure = (
            0.0 if self.quasineutral else (kx**2 + ky**2) * self.plasma.debye_length**2
        )
        return a_i - a_e - closure

    def _newton(self, omega: complex, kx: float, ky: float) -> complex | None:
        """Polish one root by Newton's method with a complex-step derivative."""
        scale = abs(ky * self.plasma.exb_velocity) + 1.0
        with np.errstate(all="ignore"):
            for _ in range(60):
                r = self.residual(omega, kx, ky)
                h = 1e-7 * (abs(omega) + scale)
                dr = (
                    self.residual(omega + h, kx, ky) - self.residual(omega - h, kx, ky)
                ) / (2 * h)
                if dr == 0.0 or not (np.isfinite(r) and np.isfinite(dr)):
                    return None
                step = r / dr
                omega = omega - step
                if abs(step) < 1e-10 * (abs(omega) + scale):
                    return omega
        return None

    def roots(
        self,
        kx: float,
        ky: float,
        seeds: np.ndarray | None = None,
        box: tuple[float, float] = (4.0, 2.0),
        grid: tuple[int, int] = (121, 61),
        validity: float = 0.7,
    ) -> LocalRoots:
        """All roots at ``(k_x, k_y)`` in a box of the complex plane, fastest first.

        Seeds are the local minima of ``|residual|`` on a ``grid`` over
        ``Re omega in k_y v_E [-box[0], box[0]]``, ``Im omega in k_y v_E [-box[1], box[1]]``
        and on a log-polar grid around the origin (plus ``seeds`` given), each polished by
        Newton and deduplicated. Roots with ``|omega - k . v_0e| > validity omega_ce`` or
        ``|Im omega| > 0.1 omega_ce`` are dropped: the drift ordering needs the frequency in
        the electron frame and the growth slow against the gyration, and the cold momentum
        balance also has fast electron modes near the hybrid frequencies (and its cyclotron
        resonance at ``|omega - k . v_0e| = omega_ce``) which are not drift physics. The
        ordering degrades as the ratio grows; at Boeuf's point it is already 0.45.
        """
        scale = abs(ky * self.plasma.exb_velocity) + 1.0
        residual = np.vectorize(lambda o: self.residual(o, kx, ky))

        def minima(w: np.ndarray, periodic_axis: int | None) -> list[complex]:
            """Local minima of ``|residual|`` on the grid ``w`` (no wrap-around at the edges)."""
            with np.errstate(all="ignore"):
                r = np.abs(residual(w))
            r = np.where(np.isfinite(r), r, np.inf)
            local = np.ones_like(r, dtype=bool)
            for axis in (0, 1):
                for shift in (1, -1):
                    rolled = np.roll(r, shift, axis=axis)
                    if axis != periodic_axis:
                        edge = [slice(None), slice(None)]
                        edge[axis] = slice(0, 1) if shift == 1 else slice(-1, None)
                        rolled[tuple(edge)] = np.inf
                    local &= r <= rolled
            return list(w[local])

        # a linear grid over the box, and a log-polar grid for the roots near the origin
        re = np.linspace(-box[0], box[0], grid[0]) * scale
        im = np.linspace(-box[1], box[1], grid[1]) * scale
        im = im[im != 0.0] if im.size > 2 else im
        linear = re[None, :] + 1j * im[:, None]
        radius = np.logspace(-5, np.log10(box[0]), 41) * scale
        angle = np.linspace(0.0, 2 * np.pi, 24, endpoint=False) + np.pi / 24
        polar = radius[:, None] * np.exp(1j * angle[None, :])
        candidates = minima(linear, None) + minima(polar, 1)
        candidates += [] if seeds is None else list(seeds)
        # the drift model holds for |omega - k . v_0e| << omega_ce; the momentum balance also
        # has the cold electron-cyclotron resonance there, which is dropped
        v0x, v0y = self.plasma.electron_flow
        limit = validity * self.plasma.omega_ce
        found: list[complex] = []
        for seed in candidates:
            root = self._newton(seed, kx, ky)
            if root is None or abs(self.residual(root, kx, ky)) > 1e-6:
                continue
            if abs(root - kx * v0x - ky * v0y) > limit or abs(root.imag) > limit / 7:
                continue
            if all(abs(root - f) > 1e-6 * (abs(f) + scale) for f in found):
                found.append(root)
        omega = np.array(sorted(found, key=lambda o: -o.imag), dtype=complex)
        return LocalRoots(kx, ky, omega)

    def scan(self, ky: np.ndarray, kx: float = 0.0) -> xr.Dataset:
        """The fastest root over ``k_y``: ``growth_rate``, ``frequency``, ``phase_speed``.

        Each ``k_y`` is seeded with the roots of the previous one, so a branch is followed.
        """
        growth, freq = [], []
        previous: np.ndarray | None = None
        for k in ky:
            roots = self.roots(kx, float(k), seeds=previous)
            previous = roots.omega if roots.omega.size else None
            growth.append(roots.growth_rate if roots.omega.size else np.nan)
            freq.append(roots.omega[0].real if roots.omega.size else np.nan)
        growth_a, freq_a = np.array(growth), np.array(freq)
        return xr.Dataset(
            {
                "growth_rate": ("ky", growth_a, {"units": "1/s"}),
                "frequency": ("ky", freq_a, {"units": "rad/s"}),
                "phase_speed": ("ky", freq_a / np.asarray(ky), {"units": "m/s"}),
            },
            coords={"ky": ("ky", np.asarray(ky), {"units": "1/m"})},
        )


def scan_profile(
    profile: xr.Dataset,
    ky: float,
    ions: ColdIons | KineticIons | None = None,
    electron_collision_frequency: float = 0.0,
    curl_free_field: bool = False,
    quasineutral: bool = False,
) -> xr.Dataset:
    """The fastest local root along a 1D profile at one ``k_y``.

    ``profile`` has ``x`` (m) and the variables ``density`` (m^-3), ``e_field`` (V/m),
    ``b_field`` (T) and ``electron_temperature`` (eV); the gradient lengths come from
    centred differences. ``ions`` default to cold argon. Returns ``growth_rate``,
    ``frequency`` and ``phase_speed`` on ``x``, with the lengths used.
    """
    ions = ColdIons() if ions is None else ions
    x = profile["x"].values
    n = profile["density"].values
    e_field = profile["e_field"].values
    b = profile["b_field"].values
    te = profile["electron_temperature"].values

    def length(f: np.ndarray) -> np.ndarray:
        df = np.gradient(f, x)
        with np.errstate(divide="ignore"):
            return np.where(df != 0.0, f / df, np.inf)

    ln, lb, le = length(n), length(b), length(e_field)
    growth, freq = [], []
    for i in range(x.size):
        plasma = Plasma(
            float(n[i]), float(e_field[i]), float(b[i]), float(te[i]), float(ln[i]),
            float(lb[i]), float(le[i]), electron_collision_frequency, 0.0, curl_free_field,
        )  # fmt: skip
        roots = LocalDispersion(plasma, ions, quasineutral).roots(0.0, ky)
        growth.append(roots.growth_rate if roots.omega.size else np.nan)
        freq.append(roots.omega[0].real if roots.omega.size else np.nan)
    growth_a, freq_a = np.array(growth), np.array(freq)
    return xr.Dataset(
        {
            "growth_rate": ("x", growth_a, {"units": "1/s"}),
            "frequency": ("x", freq_a, {"units": "rad/s"}),
            "phase_speed": ("x", freq_a / ky, {"units": "m/s"}),
            "density_length": ("x", ln, {"units": "m"}),
            "field_length": ("x", lb, {"units": "m"}),
        },
        coords={"x": ("x", x, {"units": "m"})},
    )
