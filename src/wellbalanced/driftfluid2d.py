"""A 2D axial-azimuthal drift-fluid model of a magnetron: gradient-drift instability and spokes.

The plane is ``(x, y)``: ``x`` axial from the cathode (``x = 0``) to the anode
(``x = length_x``), ``y`` azimuthal and periodic, the magnetic field ``B_z(x)`` out
of the plane (the curvature is neglected, as in the 2D PIC benchmarks of Charoy et
al. 2019 and Boeuf 2022/2023). Species:

- **electrons**, inertialess and magnetised, with the full Hall tensor. From
  ``0 = -e (E + v x B) - grad(n T_e)/n - m nu v``:

      Gamma_e = -mu_perp [[1, -h], [h, 1]] (n E + grad(n T_e)),
      h = omega_ce / nu,   mu_perp = e nu / (m (nu^2 + omega_ce^2)),

  so the E x B drift (``h >> 1``: ``mu_perp h = 1/B``), the diamagnetic drift and the
  gradient-B drift (``d_x mu_perp`` in the divergence of the antisymmetric part) all
  come out of the tensor; ``nu = k_en N + alpha omega_ce / 16`` adds a Bohm-like
  anomalous collision frequency when ``bohm > 0``;
- **ions**, cold or isothermal, unmagnetised, with inertia and friction on the gas
  (``dv/dt + v.grad v = e E / M - nu_in v - T_i grad n / (M n)``). Inertia matters:
  the ion response sets the lower-hybrid / ion-sound frequency of the instability,
  which a drift-diffusion ion does not have;
- **neutral gas** ``N``, depleted by ionization and refilled at the rate
  ``refill_rate`` (the gas flow and the sputter wind), the slow variable of an
  ionization wave;
- the **electron energy** ``w = 3/2 n T_e`` with Joule heating, the enthalpy flux of
  Hagelaar and Pitchford (2005) and three loss channels (elastic, ionization,
  excitation), or a fixed ``T_e``.

Closures (what the 1D runaway diagnosis changed; see the changelog): the electron
momentum-transfer rate is a function of ``T_e`` (not a constant), excitation is an explicit
loss channel (replacing a lumped 30 eV per ionization), the fitted rates are held at their
end values outside ``0.5 .. 100 eV`` so no rate grows with a runaway ``T_e``, the local
energy loss is treated implicitly (the explicit update is stiff: the heating of a
weakly magnetised, high-field cell is 1e11 eV/s per electron), and the cathode emits a
*prescribed net electron current* ``cathode_current_density`` as in Boeuf's PIC (a fixed
``gamma`` is an avalanche whenever ``gamma M >= 1``, with ``M`` about 170 here).

Formulation: **full Poisson, semi-implicit**, not quasi-neutral. The cathode
sheath and the plasma-wall boundaries are then part of the solution and no
sheath boundary model is needed, and the discharge is free to leave quasi-neutrality.
The dielectric relaxation through the electron conductivity (the stiff part: the
Hall current does not relax charge but ``mu_perp`` does) is removed by solving, in
every stage, the Poisson equation for the potential at the end of the stage,
``div((eps0 + dt sigma_e) grad phi) = rho - dt div(J_known)`` with the electron
conductivity tensor ``sigma_e = e n mu_perp [[1, -h], [h, 1]]`` (Lapenta's implicit
permittivity). With the same discrete fluxes this makes the Poisson equation of the
new densities hold exactly. The step is then limited by the advection (E x B drift
up to 1e6 m/s at the cathode) and not by ``eps0 / sigma``. Time stepping is the
SSP Runge-Kutta 3 scheme (each stage an implicit-permittivity Euler step).

Discretisation: finite volumes on cells; the E x B advection of the electrons uses
third-order upwind-biased face values (a first-order upwind scheme has a numerical
diffusion ``v dy / 2`` that damps the spoke wavelengths); the ions use first-order
upwind (positive, monotone). Electron boundaries: the cathode emits a prescribed net
current (``cathode_current_density``) or ``gamma`` electrons per incident ion, the anode
absorbs (outflow only); ions leave freely (outflow only). The potential is ``-voltage`` at the cathode and 0 at the
anode.

Verification (``tests/unit/test_driftfluid2d.py``): a y-uniform state stays uniform to
round-off; a passive blob drifts at ``E/B`` to 3e-6; one Euler stage satisfies the
discrete Poisson equation exactly; ``stationary_sources`` hold an equilibrium; every term
of the discrete operator matches its analytic form to under 1 % (electron Hall flux with
a density gradient, ion continuity and momentum); and in a uniform plasma periodic in
``x`` (``periodic_x``, ``field_bc``) the time-domain growth rate and phase speed of an
``x``-uniform mode agree with :func:`uniform_plasma_roots`, the full-Poisson local
dispersion relation, to under 4 % (0.5 % in the eigenvalues of the semi-discrete operator, at
32 cells per wavelength, converging with resolution). What is not verified: the
gradient-drift growth rate in a bounded box. The local WKB relation
:func:`gradient_drift_roots` assumes a mode uniform along ``x``, which electrodes forbid:
the eigenmodes of a box are waves along ``x`` with their own rates, several times
different from the local ones (the quasi-neutral form is also far off, because the
E x B-driven branch sits on a near cancellation that the few-percent Debye correction moves).

Linear stability: :meth:`DriftFluid2D.linear_modes` builds the Jacobian of the discretised operator
about a y-uniform state for one azimuthal mode, x-resolved, so the electrodes, the gradients, the Hall
tensor and the energy equation are in it, and returns its eigenvalues and eigenvectors. Its leading
eigenvalue is meant to agree with :func:`uniform_plasma_roots` for a uniform plasma and with a
time-domain run fitted in its exponential-growth window (tests in ``test_driftfluid2d.py``); the same
comparison on Boeuf's 1D state is the job of the gallery study 03, which has not been run.

Analytic reference: :func:`uniform_plasma_roots` and :func:`gradient_drift_roots` are the
local dispersion relations of this system (without and with a density gradient).

PLACEHOLDERS, to be checked against the sources before any physical claim: the three
argon rate fits (ionization, excitation, momentum transfer: quoted from memory of
Lieberman and Lichtenberg, *Principles of Plasma Discharges and Materials Processing*;
the momentum rate gives 1.3e-13 m^3/s at 3 eV, close to the ``nu_en = 2e7 /s`` that Boeuf
quotes at 0.6 Pa; replace all three with rates from plasmacoll's cross sections), the
ion friction, the single lumped excitation level (11.56 eV) and ``secondary_emission``.

Not yet: electron inertia, an electron-ion or ion-ion collision operator, sputtered
metal (no HiPIMS), the curvature of the race track, a gas temperature equation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import xarray as xr
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE

from wellbalanced.datasets import variable

__all__ = [
    "DriftFluid2D",
    "DriftFluid2DState",
    "LinearModes",
    "ModeFit",
    "Run",
    "SpokeReport",
    "argon_excitation_rate",
    "argon_ionization_rate",
    "argon_momentum_rate",
    "exponential_field",
    "fit_mode",
    "gradient_drift_roots",
    "mode_amplitudes",
    "spoke_report",
    "uniform_plasma_roots",
]

E = ELEMENTARY_CHARGE
EPSILON_0 = 8.8541878128e-12
#: Floor of the densities (m^-3) and of the electron temperature (eV).
N_FLOOR = 1.0e3
TE_FLOOR = 0.1
TE_CEIL = 200.0


#: Electron temperatures (eV) between which the fitted rates are evaluated; outside they
#: are held at the end values, so no rate can run away with a runaway ``T_e``.
RATE_TE_RANGE = (0.5, 100.0)


def _bounded(te: np.ndarray | float) -> np.ndarray | float:
    return np.clip(te, *RATE_TE_RANGE)


def argon_ionization_rate(te: np.ndarray | float) -> np.ndarray | float:
    """Ionization rate coefficient of argon at a Maxwellian ``T_e`` (eV), m^3/s.

    PLACEHOLDER: ``2.34e-14 T_e^0.59 exp(-17.44 / T_e)``, the fit as remembered from
    Lieberman and Lichtenberg's textbook; check it against the plasmacoll cross
    section before use. ``T_e`` is bounded to :data:`RATE_TE_RANGE`.
    """
    te = _bounded(te)
    return 2.34e-14 * te**0.59 * np.exp(-17.44 / te)


def argon_excitation_rate(te: np.ndarray | float) -> np.ndarray | float:
    """Excitation rate coefficient of argon (lumped, ``11.56 eV``), m^3/s.

    PLACEHOLDER: ``2.48e-14 T_e^0.33 exp(-12.78 / T_e)``, as remembered from Lieberman
    and Lichtenberg; check against plasmacoll.
    """
    te = _bounded(te)
    return 2.48e-14 * te**0.33 * np.exp(-12.78 / te)


def argon_momentum_rate(te: np.ndarray | float) -> np.ndarray | float:
    """Electron-argon momentum-transfer rate coefficient at ``T_e`` (eV), m^3/s.

    PLACEHOLDER: ``2.336e-14 T_e^1.609 exp(0.0618 ln(T_e)^2 - 0.1171 ln(T_e)^3)``, as
    remembered from Lieberman and Lichtenberg; it gives 1.3e-13 m^3/s at 3 eV, close to the
    ``nu_en = 2e7 /s`` at 0.6 Pa that Boeuf quotes (``k = 1.4e-13``).
    """
    log = np.log(_bounded(te))
    return 2.336e-14 * np.exp(1.609 * log + 0.0618 * log**2 - 0.1171 * log**3)


def exponential_field(b_max: float, scale: float) -> Callable[[np.ndarray], np.ndarray]:
    """``B(x) = b_max exp(-x / scale)``: the profile of Boeuf (2022), 120 mT and ``0.3 gap``."""

    def field_of_x(x: np.ndarray) -> np.ndarray:
        return b_max * np.exp(-np.asarray(x) / scale)

    return field_of_x


@dataclass
class DriftFluid2DState:
    """The fields on the ``(nx, ny)`` cells.

    Attributes:
        x: Cell centres along ``x`` (m).
        y: Cell centres along ``y`` (m).
        electron_density: ``n_e`` (m^-3).
        ion_density: ``n_i`` (m^-3).
        ion_velocity: ``(v_x, v_y)`` of the ions, shape ``(2, nx, ny)`` (m/s).
        gas_density: The neutral density ``N`` (m^-3).
        electron_temperature: ``T_e`` (eV).
        potential: ``phi`` (V).
        time: Time (s).
    """

    x: np.ndarray
    y: np.ndarray
    electron_density: np.ndarray
    ion_density: np.ndarray
    ion_velocity: np.ndarray
    gas_density: np.ndarray
    electron_temperature: np.ndarray
    potential: np.ndarray
    time: float = 0.0

    def copy(self) -> DriftFluid2DState:
        """A deep copy."""
        return DriftFluid2DState(
            self.x,
            self.y,
            self.electron_density.copy(),
            self.ion_density.copy(),
            self.ion_velocity.copy(),
            self.gas_density.copy(),
            self.electron_temperature.copy(),
            self.potential.copy(),
            self.time,
        )

    def to_dataset(self) -> xr.Dataset:
        """The state as a labelled dataset on ``(x, y)``."""
        dims = ("x", "y")
        data = {
            "electron_density": variable(
                dims, self.electron_density, "electron density", "m^-3"
            ),
            "ion_density": variable(dims, self.ion_density, "ion density", "m^-3"),
            "ion_velocity_x": variable(
                dims, self.ion_velocity[0], "ion velocity along x", "m/s"
            ),
            "ion_velocity_y": variable(
                dims, self.ion_velocity[1], "ion velocity along y", "m/s"
            ),
            "gas_density": variable(dims, self.gas_density, "gas density", "m^-3"),
            "electron_temperature": variable(
                dims, self.electron_temperature, "electron temperature", "eV"
            ),
            "potential": variable(dims, self.potential, "potential", "V"),
        }
        coords = {
            "x": ("x", self.x, {"long_name": "x (axial)", "units": "m"}),
            "y": ("y", self.y, {"long_name": "y (azimuthal)", "units": "m"}),
        }
        return xr.Dataset(data, coords=coords, attrs={"time": self.time})


@dataclass
class ModeFit:
    """The growth and the phase speed of one azimuthal mode over a time window.

    Attributes:
        mode: The azimuthal mode number ``m`` (``k = 2 pi m / length_y``).
        growth_rate: ``gamma`` (1/s) from the log of the amplitude.
        frequency: ``omega`` (rad/s) of ``exp(i (k y - omega t))``; positive moves along ``+y``.
        phase_speed: ``omega / k`` (m/s), along ``+y`` (the E x B direction for the
            geometry of the module).
    """

    mode: int
    growth_rate: float
    frequency: float
    phase_speed: float


@dataclass
class Run:
    """The record of :meth:`DriftFluid2D.run`.

    Attributes:
        times: Recording times (s).
        probe: The recorded field along ``y`` at ``probe_x``, shape ``(nt, ny)``.
        probe_x: The ``x`` of the probe (m), the cell centre nearest the request.
        amplitudes: Complex Fourier coefficients of ``probe / mean`` per mode
            ``0 .. ny // 2``, shape ``(nt, ny // 2 + 1)``.
        state: The final state.
        steps: The number of steps taken.
        length_y: The azimuthal period (m).
        series: Further records, shape ``(nt, ny)``: ``"density_cathode"`` (the cell
            next to the cathode), ``"potential"``, ``"e_y"`` and ``"v_exb_y"`` (the
            azimuthal E x B velocity) at the probe.
    """

    times: np.ndarray
    probe: np.ndarray
    probe_x: float
    amplitudes: np.ndarray
    state: DriftFluid2DState
    steps: int
    length_y: float = 0.0
    series: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dataset(self) -> xr.Dataset:
        """The space-time map of the probe and the mode amplitudes."""
        ny = self.probe.shape[1]
        y = self.state.y
        data = {
            "probe": variable(
                ("time", "y"), self.probe, "electron density at the probe", "m^-3"
            ),
            "mode_amplitude": variable(
                ("time", "mode"), np.abs(self.amplitudes), "|n_m / n_0|"
            ),
        }
        coords = {
            "time": ("time", self.times, {"long_name": "time", "units": "s"}),
            "y": ("y", y, {"long_name": "y (azimuthal)", "units": "m"}),
            "mode": ("mode", np.arange(ny // 2 + 1), {"long_name": "mode number"}),
        }
        return xr.Dataset(
            data, coords=coords, attrs={"probe_x": self.probe_x, "steps": self.steps}
        )


@dataclass
class SpokeReport:
    """What a run says about its azimuthal structure, in the terms of the experiments.

    Attributes:
        mode: The dominant azimuthal mode number ``m >= 1`` (largest time-averaged amplitude).
        wavelength: ``length_y / m`` (m).
        growth_rate: Growth rate (1/s) of that mode over the window.
        phase_velocity: Phase velocity (m/s) along ``+y``.
        exb_velocity: The azimuthal E x B velocity (m/s) at the probe, averaged over the
            window and ``y``.
        relative_direction: ``+1`` if the mode moves along E x B, ``-1`` against it.
        speed_ratio: ``|phase_velocity| / |exb_velocity|``.
        cathode_modulation: Peak-to-peak density modulation next to the cathode,
            ``(max - min) / (2 mean)`` along ``y``, averaged over the window.
        potential_hump: ``max_y(phi) - mean_y(phi)`` at the probe (V), window average.
        azimuthal_field: ``max_y |E_y|`` at the probe (V/m), window average.
        window: The ``(t0, t1)`` (s) of the analysis.
    """

    mode: int
    wavelength: float
    growth_rate: float
    phase_velocity: float
    exb_velocity: float
    relative_direction: int
    speed_ratio: float
    cathode_modulation: float
    potential_hump: float
    azimuthal_field: float
    window: tuple[float, float]


def spoke_report(run: Run, window: tuple[float, float] | None = None) -> SpokeReport:
    """The :class:`SpokeReport` of a run over a time window (the last half by default)."""
    t = run.times
    window = (0.5 * (t[0] + t[-1]), t[-1]) if window is None else window
    select = (t >= window[0]) & (t <= window[1])
    modes = np.arange(1, run.amplitudes.shape[1])
    mean_amp = np.abs(run.amplitudes[select][:, 1:]).mean(axis=0)
    mode = int(modes[np.argmax(mean_amp)])
    fit = fit_mode(t, run.amplitudes, mode, run.length_y, window)
    v_exb = float(run.series["v_exb_y"][select].mean())
    n_c = run.series["density_cathode"][select]
    modulation = float(
        np.mean((n_c.max(axis=1) - n_c.min(axis=1)) / (2.0 * n_c.mean(axis=1)))
    )
    phi = run.series["potential"][select]
    hump = float(np.mean(phi.max(axis=1) - phi.mean(axis=1)))
    e_y = float(np.mean(np.abs(run.series["e_y"][select]).max(axis=1)))
    direction = int(np.sign(fit.phase_speed) * np.sign(v_exb)) if v_exb else 0
    return SpokeReport(
        mode,
        run.length_y / mode,
        fit.growth_rate,
        fit.phase_speed,
        v_exb,
        direction,
        abs(fit.phase_speed) / abs(v_exb) if v_exb else float("nan"),
        modulation,
        hump,
        e_y,
        window,
    )


def mode_amplitudes(values: np.ndarray) -> np.ndarray:
    """Complex azimuthal Fourier coefficients of ``values / mean`` along the last axis.

    Mode ``m`` is the coefficient of ``exp(-i k y)`` over ``ny``: a perturbation
    ``a cos(k y - omega t)`` has ``a / 2 exp(-i omega t)``, the phase falls in time
    for a wave that moves along ``+y``.
    """
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=-1, keepdims=True)
    return np.fft.rfft(values / mean - 1.0, axis=-1) / values.shape[-1]


def fit_mode(
    times: np.ndarray,
    amplitudes: np.ndarray,
    mode: int,
    length_y: float,
    window: tuple[float, float] | None = None,
) -> ModeFit:
    """Fit the growth rate and the phase speed of mode ``mode`` in a time window.

    Args:
        times: The recording times (s).
        amplitudes: :func:`mode_amplitudes` at those times.
        mode: The mode number ``m``.
        length_y: The azimuthal period (m).
        window: ``(t0, t1)`` to fit in; the whole record by default.
    """
    times = np.asarray(times, dtype=float)
    select = np.ones(times.shape, bool)
    if window is not None:
        select = (times >= window[0]) & (times <= window[1])
    if select.sum() < 3:
        raise ValueError("need at least 3 records in the window")
    t = times[select]
    a = amplitudes[select, mode]
    growth = np.polyfit(t, np.log(np.abs(a)), 1)[0]
    omega = -np.polyfit(t, np.unwrap(np.angle(a)), 1)[0]
    k = 2.0 * np.pi * mode / length_y
    return ModeFit(mode, float(growth), float(omega), float(omega / k))


def _plasma_parameters(
    b_field: float, electron_collision_frequency: float, ion_collision_frequency: float
):
    """``(h, mu_perp)`` of the electrons from the field and the collision frequency."""
    omega_ce = E * b_field / ELECTRON_MASS
    nu = electron_collision_frequency
    return omega_ce / nu, E * nu / (ELECTRON_MASS * (nu**2 + omega_ce**2))


def uniform_plasma_roots(
    kx: complex,
    ky: float,
    density: float,
    e_field: float,
    b_field: float,
    electron_temperature: float,
    ion_mass: float,
    ion_collision_frequency: float,
    electron_collision_frequency: float,
) -> np.ndarray:
    """Eigenvalues ``s`` of ``exp(i (k . r) + s t)`` for a uniform plasma, full Poisson.

    Uniform ``n_0``, uniform ``E_0 = E_0 x`` and ``B_z``, constant ``T_e``, magnetised
    inertialess electrons (the model's Hall tensor), cold ions with friction (steady
    drift ``v_0i = e E_0 / (M nu_i)`` along ``x``) and Poisson's equation
    ``eps0 k^2 delta phi = e n_0 (a_i - a_e)``, with ``delta n = n_0 a``:

        s a_e = -(i k.v_0e + mu_perp k^2 T_e) a_e + mu_perp k^2 delta phi,
        s a_i = -i k.v_0i a_i - i k.v,
        s v   = -(nu_i + i k.v_0i) v - i (e / M) k delta phi,

    where ``v_0e = -mu_perp E_0 (1, h)``. (The antisymmetric Hall part of the tensor drops
    out of ``k . R k = k^2``.) Returns the three roots ``s``; ``Re s > 0`` grows and
    ``omega = -Im s`` is the frequency for ``k_y > 0``. For ``k_x = 0`` and a large
    density this tends to the quasi-neutral relation, whose roots near an
    E x B-driven resonance are very sensitive to the few-percent Debye correction.
    """
    h, mu_perp = _plasma_parameters(
        b_field, electron_collision_frequency, ion_collision_frequency
    )
    k = np.array([kx, ky], dtype=complex)
    k2 = k @ k
    v0e = -mu_perp * e_field * np.array([1.0, h])
    v0i = np.array([E * e_field / (ion_mass * ion_collision_frequency), 0.0])
    poisson = E * density / (EPSILON_0 * k2)
    nu_i = ion_collision_frequency
    matrix = np.zeros((4, 4), complex)  # (a_e, a_i, v_x, v_y)
    matrix[0, 0] = -(1j * (k @ v0e) + mu_perp * k2 * electron_temperature)
    matrix[0, 0] -= mu_perp * k2 * poisson
    matrix[0, 1] = mu_perp * k2 * poisson
    matrix[1, 1] = -1j * (k @ v0i)
    matrix[1, 2], matrix[1, 3] = -1j * k[0], -1j * k[1]
    for row, component in ((2, 0), (3, 1)):
        matrix[row, row] = -(nu_i + 1j * (k @ v0i))
        matrix[row, 0] = 1j * (E / ion_mass) * k[component] * poisson
        matrix[row, 1] = -1j * (E / ion_mass) * k[component] * poisson
    return np.linalg.eigvals(matrix)


def gradient_drift_roots(
    k: float,
    gradient_length: float,
    e_field: float,
    b_field: float,
    electron_temperature: float,
    ion_mass: float,
    ion_collision_frequency: float,
    electron_collision_frequency: float,
    density: float | None = None,
) -> np.ndarray:
    """Roots ``omega`` of the local gradient-drift (Simon-Hoh) dispersion relation.

    The system of the module on an equilibrium ``n_0(x) = n_0 exp(-x / L)`` in a
    uniform ``E_0 = E_0 x`` and a uniform ``B_z = B``, perturbed by
    ``delta n = n_0(x) a exp(i (k y - omega t))`` with ``delta phi`` uniform in ``x``
    (WKB, ``k_x = 0``, ``k L`` large), ``T_e`` constant, cold ions with friction
    ``nu_i`` (so with the steady drift ``v_0i = e E_0 / (M nu_i)``). With
    ``s = -i omega``, ``K = i k (mu_perp h / L + i k mu_perp)``, ``v_0e = -mu_perp (E_0 - T_e / L)``
    along ``x`` and ``h v_0e`` along ``y``:

        s a_e = (v_0e / L - i k h v_0e) a_e + K (T_e a_e - delta phi),
        s a_i = (v_0i / L) a_i - i k v,        s v = -nu_i v - i (e / M) k delta phi,

    with ``eps0 k^2 delta phi = e n_0 (a_i - a_e)`` (Poisson) when ``density`` (``n_0``,
    m^-3) is given, and ``a_e = a_i`` (quasi-neutral, two roots) when it is not.
    The electron part is the continuity equation with the Hall tensor (the E x B
    advection, the density gradient and the diamagnetic term in ``T_e``), the ion part
    the unmagnetised ion continuity and momentum. Valid when the mode varies slowly
    along ``x`` on the scale ``L``; ``T_e`` in eV, ``E_0`` in V/m.

    Returns:
        The complex roots ``omega``; ``Im omega > 0`` grows, ``Re omega / k`` is
        the phase speed along ``+y``.
    """
    h, mu_perp = _plasma_parameters(
        b_field, electron_collision_frequency, ion_collision_frequency
    )
    nu_i = ion_collision_frequency
    v0i = E * e_field / (ion_mass * nu_i)
    v0e = -mu_perp * (e_field - electron_temperature / gradient_length)
    big_k = 1j * k * (mu_perp * h / gradient_length + 1j * k * mu_perp)
    if density is not None:
        poisson = E * density / (EPSILON_0 * k**2)
        matrix = np.zeros((3, 3), complex)  # (a_e, a_i, v)
        matrix[0, 0] = v0e / gradient_length - 1j * k * h * v0e
        matrix[0, 0] += big_k * (electron_temperature + poisson)
        matrix[0, 1] = -big_k * poisson
        matrix[1, 1] = v0i / gradient_length
        matrix[1, 2] = -1j * k
        matrix[2, 2] = -nu_i
        matrix[2, 0] = 1j * (E / ion_mass) * k * poisson
        matrix[2, 1] = -1j * (E / ion_mass) * k * poisson
        return 1j * np.linalg.eigvals(matrix)
    m_over = ion_mass / (E * k**2)
    # polynomial in s: -K m (s - v0i/L)(s + nu_i) + (1 - ...) s + const
    quad = -big_k * m_over
    lin = quad * (nu_i - v0i / gradient_length) + 1.0
    const = (
        quad * (-v0i / gradient_length) * nu_i
        - v0e / gradient_length
        + 1j * k * h * v0e
        - big_k * electron_temperature
    )
    s = np.roots([quad, lin, const])
    return 1j * s


def _upwind3(
    n: np.ndarray, positive: np.ndarray, axis: int, periodic_x: bool = False
) -> np.ndarray:
    """Third-order upwind-biased face values; ``positive`` selects the flow direction.

    Along ``axis = 1`` (periodic) face ``j`` lies between cells ``j - 1`` and ``j``. Along
    ``axis = 0`` the result has ``nx + 1`` faces; the faces next to the boundary
    take the central average and the boundary faces the adjacent cell.
    """
    if axis == 1 or periodic_x:
        left = np.roll(n, 1, axis=axis)
        left2 = np.roll(n, 2, axis=axis)
        right = n
        right2 = np.roll(n, -1, axis=axis)
        pos = (-left2 + 5.0 * left + 2.0 * right) / 6.0
        neg = (2.0 * left + 5.0 * right - right2) / 6.0
        if axis == 1:
            return np.where(positive, pos, neg)
        face = np.where(positive[: n.shape[0]], pos, neg)
        return np.vstack([face, face[:1]])
    nx = n.shape[0]
    faces = np.empty((nx + 1, n.shape[1]))
    faces[0] = n[0]
    faces[nx] = n[-1]
    faces[1:nx] = 0.5 * (n[:-1] + n[1:])
    if nx >= 4:
        pos = (-n[:-3] + 5.0 * n[1:-2] + 2.0 * n[2:-1]) / 6.0
        neg = (2.0 * n[1:-2] + 5.0 * n[2:-1] - n[3:]) / 6.0
        # faces 2 .. nx - 2 lie between cells (i - 1, i)
        faces[2 : nx - 1] = np.where(positive[2 : nx - 1], pos, neg)
    return faces


@dataclass
class LinearModes:
    """The eigenmodes of :meth:`DriftFluid2D.linear_modes`, the most unstable first.

    Attributes:
        mode: The azimuthal mode number ``m``.
        wavenumber: ``k = 2 pi m / length_y`` (1/m).
        eigenvalues: ``s = gamma - i omega`` of ``exp(i k y + s t)``.
        profiles: The eigenvectors along ``x`` per unknown (``"ne"``, ``"ni"``, ...), shape
            ``(nx, number of eigenvalues)``.
        x: The cell centres (m).
        matrix: The complex Jacobian (rows and columns: the unknowns in blocks of ``nx``).
    """

    mode: int
    wavenumber: float
    eigenvalues: np.ndarray
    profiles: dict[str, np.ndarray]
    x: np.ndarray
    matrix: np.ndarray

    @property
    def growth_rates(self) -> np.ndarray:
        """``gamma`` of every eigenvalue (1/s)."""
        return self.eigenvalues.real

    @property
    def frequencies(self) -> np.ndarray:
        """``omega = -Im s`` of every eigenvalue (rad/s)."""
        return -self.eigenvalues.imag

    @property
    def phase_speeds(self) -> np.ndarray:
        """``omega / k`` (m/s), positive along ``+y``."""
        return self.frequencies / self.wavenumber

    def leading(self) -> tuple[float, float, float]:
        """``(gamma, omega, phase speed)`` of the most unstable eigenvalue."""
        return (
            float(self.growth_rates[0]),
            float(self.frequencies[0]),
            float(self.phase_speeds[0]),
        )


@dataclass
class DriftFluid2D:
    """The magnetron plane of the module docstring.

    Attributes:
        length_x: The cathode-anode gap (m).
        length_y: The azimuthal period (m).
        nx: Cells along ``x``.
        ny: Cells along ``y`` (at least 4).
        voltage: The discharge voltage (V): the cathode is at ``-voltage``, the anode at 0.
        magnetic_field: ``B_z(x)`` in T: a callable of the cell positions, or a number.
        gas_density: The neutral density at rest ``N_0`` (m^-3).
        electron_temperature: ``T_e`` held fixed (eV), or ``None`` for the energy equation.
        momentum_rate: Electron-neutral momentum-transfer rate coefficient ``k_en`` (m^3/s),
            ``nu_en = k_en N``: a number, or a function of ``T_e`` (eV; the default is the
            PLACEHOLDER :func:`argon_momentum_rate`).
        ion_friction_rate: Ion-neutral momentum-transfer rate coefficient (m^3/s).
            PLACEHOLDER.
        ion_mass_u: The ion mass in atomic mass units (argon).
        ion_temperature: ``T_i`` in eV (also the gas temperature).
        ionization_rate: Rate coefficient (m^3/s) as a function of ``T_e`` (eV).
        excitation_rate: Excitation rate coefficient (m^3/s) as a function of ``T_e``, or
            ``None`` for none; ``excitation_energy_ev`` is the energy it costs (eV), and
            ``ionization_energy_ev`` the energy an ionization costs. Both are channels
            of the electron energy loss, with the elastic loss ``3 (m / M) nu_en (T_e - T_i)``.
        secondary_emission: Electrons emitted per ion hitting the cathode. PLACEHOLDER.
        cathode_current_density: If given (A/m^2), the cathode emits this net electron current
            density, whatever the ion flux, instead of ``secondary_emission`` per ion. This is the
            condition of Boeuf's PIC (``j_e0 = 0.03 A/m^2``, the electrons returning to the cathode
            re-emitted): a fixed ``gamma`` makes the discharge an avalanche whenever
            ``gamma M >= 1`` (``M`` the multiplication), a fixed emitted current gives a
            stable steady state with ``gamma_net = j_e0 / j_ion``.
        secondary_energy_ev: Temperature of the emitted electrons (eV).
        refill_rate: Relaxation rate (1/s) of the gas towards ``N_0`` (the gas flow and
            the sputter wind): the explicit parameter, with ``depletion``, of the neutral dynamics.
        depletion: Fraction of the ionization events that remove a neutral (1: full
            depletion, 0: a frozen gas).
        field_bc: ``None`` for the electrodes at ``-voltage`` and 0 V. A number ``E_0`` (V/m)
            replaces them by a fixed axial field ``E_x = E_0`` at both ends (a pinned
            potential), so that a perturbation can be uniform along ``x``: the setting
            for comparing with the local dispersion relation.
        heating_scale: Factor on the Joule heating of the electron energy (a diagnostic switch).
        floor_added: Running total of the density (summed over cells, m^-3) that the floor
            ``N_FLOOR`` on ``ne``, ``ni`` and ``N`` created in the Euler stages, a diagnostic of
            how much charge the limiter invents; reset it to zero to start counting.
        anode_thermal_flux: Adds the thermal flux of an absorbing wall, ``n v_bar / 2`` with
            ``v_bar = sqrt(8 e T_e / (pi m))`` (Hagelaar et al. 2000, reflection 0), to the drift
            flux of the electrons at the anode, which otherwise only leave at the drift rate; the
            outflow is then the larger of the two, and the density next to the anode falls to
            what carries the current. Tested for Boeuf's case without success: the 1D discharge
            then decays and runs away (no sheath barrier for the wall flux), so it is off.
        diamagnetic: Factor on the electron pressure-gradient (diamagnetic) force in the fluxes
            and the Poisson solve (a diagnostic switch for the linear analysis; 1 is the model).
        periodic_x: Periodic along ``x`` as well, for a uniform plasma in a uniform field
            ``field_bc`` (required) and a uniform ``B``: pure Fourier modes ``(k_x, k_y)``
            exist, which is what the local dispersion relation describes.
        bohm: ``alpha`` of the anomalous frequency ``alpha omega_ce / 16`` (0 off).
        external_sources: Volume sources added to the time derivative of an unknown
            (``"ne"``, ``"ni"``, ``"N"``, ``"vx"``, ``"vy"``, ``"w"``), numbers or arrays
            on the cells, per unit time. :meth:`stationary_sources` returns the ones
            that make a given state stationary.
    """

    length_x: float
    length_y: float
    nx: int
    ny: int
    voltage: float = 260.0
    magnetic_field: Callable[[np.ndarray], np.ndarray] | float = 0.0
    gas_density: float = 1.45e20
    electron_temperature: float | None = 5.0
    momentum_rate: float | Callable[[np.ndarray], np.ndarray] = argon_momentum_rate
    ion_friction_rate: float = 1.0e-15
    ion_mass_u: float = 39.948
    ion_temperature: float = 0.025
    ionization_rate: Callable[[np.ndarray], np.ndarray] = argon_ionization_rate
    excitation_rate: Callable[[np.ndarray], np.ndarray] | None = argon_excitation_rate
    ionization_energy_ev: float = 15.76
    excitation_energy_ev: float = 11.56
    secondary_emission: float = 0.06
    cathode_current_density: float | None = None
    secondary_energy_ev: float = 1.0
    refill_rate: float = 1.0e4
    depletion: float = 1.0
    field_bc: float | None = None
    periodic_x: bool = False
    heating_scale: float = 1.0
    diamagnetic: float = 1.0
    anode_thermal_flux: bool = False
    floor_added: dict = field(default_factory=lambda: {"ne": 0.0, "ni": 0.0, "N": 0.0})
    _rate_override: object = field(default=None, repr=False)
    _lu: object = field(default=None, repr=False)
    bohm: float = 0.0
    external_sources: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build the grid and the sparse operators."""
        if self.nx < 4 or self.ny < 4:
            raise ValueError("need at least 4 cells in each direction")
        if self.periodic_x and self.field_bc is None:
            raise ValueError("periodic_x needs field_bc, the uniform axial field")
        self.dx = self.length_x / self.nx
        self.dy = self.length_y / self.ny
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.xf = np.arange(self.nx + 1) * self.dx
        self.y = (np.arange(self.ny) + 0.5) * self.dy
        self.ion_mass = self.ion_mass_u * ATOMIC_MASS - ELECTRON_MASS
        bfun = self.magnetic_field
        if not callable(bfun):
            value = float(bfun)

            def bfun(x):
                return np.full_like(np.asarray(x, dtype=float), value)

        self._bc = np.asarray(bfun(self.x), dtype=float)
        self._bf = np.asarray(bfun(self.xf), dtype=float)
        self._k0 = 1.0 if callable(self.momentum_rate) else float(self.momentum_rate)
        self._omega_xf = (E * self._bf / ELECTRON_MASS)[:, None]
        self._omega_yf = (E * self._bc / ELECTRON_MASS)[:, None]
        self._build_operators()

    # -- operators ------------------------------------------------------------------

    def _build_operators(self) -> None:
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        ix, iy = sp.identity(nx, format="csr"), sp.identity(ny, format="csr")
        shift = sp.csr_matrix(
            (np.ones(ny), (np.arange(ny), (np.arange(ny) - 1) % ny)), shape=(ny, ny)
        )  # (shift f)_j = f_{j-1}
        shift_up = shift.T.tocsr()  # f_{j+1}
        d_yc = (shift_up - shift) / (2.0 * dy)
        d_yf = (iy - shift) / dy
        a_yf = (iy + shift) / 2.0
        div_y = (shift_up - iy) / dy
        # x: cells -> faces (nx + 1 faces), Dirichlet potentials as affine terms
        rows, cols, vals = [], [], []
        if self.periodic_x:
            return self._build_periodic_x_operators(ix, iy, d_yc, d_yf, a_yf, div_y)
        for f in range(1, nx):
            rows += [f, f]
            cols += [f, f - 1]
            vals += [1.0 / dx, -1.0 / dx]
        if self.field_bc is None:
            rows += [0, nx]
            cols += [0, nx - 1]
            vals += [2.0 / dx, -2.0 / dx]
        d_xf = sp.csr_matrix((vals, (rows, cols)), shape=(nx + 1, nx))
        rows, cols, vals = [], [], []
        for f in range(1, nx):
            rows += [f, f]
            cols += [f, f - 1]
            vals += [0.5, 0.5]
        rows += [0, nx]
        cols += [0, nx - 1]
        vals += [1.0, 1.0]
        a_xf = sp.csr_matrix((vals, (rows, cols)), shape=(nx + 1, nx))
        div_x = sp.csr_matrix(
            (
                np.r_[np.full(nx, -1.0 / dx), np.full(nx, 1.0 / dx)],
                (
                    np.r_[np.arange(nx), np.arange(nx)],
                    np.r_[np.arange(nx), np.arange(nx) + 1],
                ),
            ),
            shape=(nx, nx + 1),
        )
        # x-derivative at the cell centres with Dirichlet ghosts (potential)
        rows, cols, vals = [], [], []
        for i in range(1, nx - 1):
            rows += [i, i]
            cols += [i + 1, i - 1]
            vals += [0.5 / dx, -0.5 / dx]
        rows += [0, 0, nx - 1, nx - 1]
        cols += [1, 0, nx - 1, nx - 2]
        if self.field_bc is None:
            vals += [0.5 / dx, 0.5 / dx, -0.5 / dx, -0.5 / dx]
        else:  # fixed field: the ghost cell extends the potential linearly
            vals += [0.5 / dx, -0.5 / dx, 0.5 / dx, -0.5 / dx]
        d_xc = sp.csr_matrix((vals, (rows, cols)), shape=(nx, nx))
        mask = np.ones(nx + 1)
        mask[[0, nx]] = 0.0
        self._gxf = sp.kron(d_xf, iy, format="csr")
        self._gyxf_masked = sp.diags(np.repeat(mask, ny)) @ sp.kron(
            a_xf, d_yc, format="csr"
        )
        self._gyf = sp.kron(ix, d_yf, format="csr")
        self._gxyf = sp.kron(d_xc, a_yf, format="csr")
        self._dxc = sp.kron(d_xc, iy, format="csr")
        self._dyc = sp.kron(ix, d_yc, format="csr")
        self._divx = sp.kron(div_x, iy, format="csr")
        self._divy = sp.kron(ix, div_y, format="csr")
        self._pyxf = sp.kron(a_xf, d_yc, format="csr")  # d_y p averaged to x-faces
        self._axf = sp.kron(a_xf, iy, format="csr")
        self._ayf = sp.kron(ix, a_yf, format="csr")
        self._shape_xf = (nx + 1, ny)
        self._build_poisson_pattern()

    def _build_periodic_x_operators(self, ix, iy, d_yc, d_yf, a_yf, div_y) -> None:
        """The operators of a domain periodic in ``x``: face ``nx`` repeats face 0."""
        nx, ny, dx = self.nx, self.ny, self.dx
        faces = np.arange(nx + 1)
        right, left = faces % nx, (faces - 1) % nx
        d_xf = sp.csr_matrix(
            (
                np.r_[np.full(nx + 1, 1.0 / dx), np.full(nx + 1, -1.0 / dx)],
                (np.r_[faces, faces], np.r_[right, left]),
            ),
            shape=(nx + 1, nx),
        )
        a_xf = sp.csr_matrix(
            (
                np.full(2 * (nx + 1), 0.5),
                (np.r_[faces, faces], np.r_[right, left]),
            ),
            shape=(nx + 1, nx),
        )
        cells = np.arange(nx)
        d_xc = sp.csr_matrix(
            (
                np.r_[np.full(nx, 0.5 / dx), np.full(nx, -0.5 / dx)],
                (np.r_[cells, cells], np.r_[(cells + 1) % nx, (cells - 1) % nx]),
            ),
            shape=(nx, nx),
        )
        div_x = sp.csr_matrix(
            (
                np.r_[np.full(nx, -1.0 / dx), np.full(nx, 1.0 / dx)],
                (np.r_[cells, cells], np.r_[cells, cells + 1]),
            ),
            shape=(nx, nx + 1),
        )
        self._gxf = sp.kron(d_xf, iy, format="csr")
        self._gyxf_masked = sp.kron(a_xf, d_yc, format="csr")
        self._gyf = sp.kron(ix, d_yf, format="csr")
        self._gxyf = sp.kron(d_xc, a_yf, format="csr")
        self._dxc = sp.kron(d_xc, iy, format="csr")
        self._dyc = sp.kron(ix, d_yc, format="csr")
        self._divx = sp.kron(div_x, iy, format="csr")
        self._divy = sp.kron(ix, div_y, format="csr")
        self._pyxf = sp.kron(a_xf, d_yc, format="csr")
        self._axf = sp.kron(a_xf, iy, format="csr")
        self._ayf = sp.kron(ix, a_yf, format="csr")
        self._shape_xf = (nx + 1, ny)
        self._build_poisson_pattern()

    def _build_poisson_pattern(self) -> None:
        """Fix the sparsity of the Poisson matrix: its entries are linear in the face coefficients.

        ``A = -(DIVX Cxx GXF + DIVX Cxy MGY + DIVY Cyy GYF + DIVY Cyx GXYF)`` with diagonal
        ``C``: ``data = B @ concat(cxx, cxy, cyy, cyx)`` for a fixed ``B``.
        """
        terms = [
            (self._divx, self._gxf),
            (self._divx, self._gyxf_masked),
            (self._divy, self._gyf),
            (self._divy, self._gxyf),
        ]
        sizes = [t[0].shape[1] for t in terms]
        offsets = np.r_[0, np.cumsum(sizes)]
        ncell = self.nx * self.ny
        entries = []  # (row, col, coefficient index, value)
        for k, (div, grad) in enumerate(terms):
            div_csc = div.tocsc()
            grad_csr = grad.tocsr()
            for face in range(div.shape[1]):
                drows = div_csc.indices[div_csc.indptr[face] : div_csc.indptr[face + 1]]
                dvals = div_csc.data[div_csc.indptr[face] : div_csc.indptr[face + 1]]
                lo, hi = grad_csr.indptr[face], grad_csr.indptr[face + 1]
                for r, dv in zip(drows, dvals, strict=True):
                    for c, gv in zip(
                        grad_csr.indices[lo:hi], grad_csr.data[lo:hi], strict=True
                    ):
                        entries.append((r, c, offsets[k] + face, -dv * gv))
        rows, cols, idx, vals = (np.array(x) for x in zip(*entries, strict=True))
        key = rows * ncell + cols
        unique, position = np.unique(key, return_inverse=True)
        self._pat_rows = unique // ncell
        self._pat_cols = unique % ncell
        self._pat_b = sp.csr_matrix(
            (vals, (position, idx)), shape=(len(unique), offsets[-1])
        )

    def _affine(self) -> tuple[np.ndarray, np.ndarray]:
        """Constant parts of the face gradients of the potential from the boundary values."""
        if self.periodic_x:
            bx = np.full(self.nx + 1, -self.field_bc)
            bc = np.full(self.nx, -self.field_bc)
            return np.repeat(bx, self.ny), np.repeat(bc, self.ny)
        if self.field_bc is not None:
            bx = np.zeros(self.nx + 1)
            bx[[0, -1]] = -self.field_bc
            bc = np.zeros(self.nx)
            bc[[0, -1]] = -0.5 * self.field_bc
            return np.repeat(bx, self.ny), np.repeat(bc, self.ny)
        phi_c, phi_a = -self.voltage, 0.0
        bx = np.zeros(self.nx + 1)
        bx[0] = -2.0 * phi_c / self.dx
        bx[-1] = 2.0 * phi_a / self.dx
        bc = np.zeros(self.nx)
        # ghost = 2 phi_b - phi_edge, so d phi/dx(edge cell) gets phi_b / dx with the sign of the side
        bc[0] = -phi_c / self.dx
        bc[-1] = phi_a / self.dx
        return np.repeat(bx, self.ny), np.repeat(bc, self.ny)

    # -- states ---------------------------------------------------------------------

    def initial_state(
        self,
        density: float | np.ndarray = 1.0e15,
        electron_temperature: float | None = None,
        gas_density: float | np.ndarray | None = None,
        potential: np.ndarray | None = None,
    ) -> DriftFluid2DState:
        """A quasi-neutral plasma of ``density`` (a number or an ``(nx, ny)`` array).

        The potential is the vacuum one (linear between the electrodes) unless given.
        """
        shape = (self.nx, self.ny)
        n = np.broadcast_to(np.asarray(density, dtype=float), shape).copy()
        te = electron_temperature
        if te is None:
            te = self.electron_temperature if self.electron_temperature else 3.0
        gas = self.gas_density if gas_density is None else gas_density
        if potential is None and self.field_bc is not None:
            potential = np.broadcast_to(
                (-self.field_bc * self.x)[:, None], shape
            ).copy()
        if potential is None:
            potential = np.broadcast_to(
                (-self.voltage * (1.0 - self.x / self.length_x))[:, None], shape
            ).copy()
        return DriftFluid2DState(
            self.x,
            self.y,
            n,
            n.copy(),
            np.zeros((2, *shape)),
            np.broadcast_to(np.asarray(gas, dtype=float), shape).copy(),
            np.broadcast_to(np.asarray(te, dtype=float), shape).copy(),
            np.asarray(potential, dtype=float).copy(),
        )

    # -- transport coefficients -------------------------------------------------------

    def _collisions(self, gas_xf, gas_yf):
        """``(mu_perp, h)`` on the x- and y-faces."""
        out = []
        for gas, omega in ((gas_xf, self._omega_xf), (gas_yf, self._omega_yf)):
            nu = np.maximum(self._k0 * gas + self._bohm_nu(omega), 1.0)
            out.append((E * nu / (ELECTRON_MASS * (nu**2 + omega**2)), omega / nu))
        return out

    def _collision_gas(self, gas: np.ndarray, te: np.ndarray) -> np.ndarray:
        """The gas density times ``k_en(T_e) / k_0``: what the collision frequency scales with."""
        if callable(self.momentum_rate):
            return gas * self.momentum_rate(te)
        return gas

    def _energy_loss(self, ne: np.ndarray, te: np.ndarray, gas: np.ndarray, gas_c):
        """Electron energy loss rate (eV m^-3 s^-1): elastic, ionization and excitation."""
        elastic = (
            3.0
            * ELECTRON_MASS
            / self.ion_mass
            * self._k0
            * gas_c
            * ne
            * (te - self.ion_temperature)
        )
        loss = elastic + self.ionization_energy_ev * self.ionization_rate(te) * gas * ne
        if self.excitation_rate is not None:
            loss = (
                loss + self.excitation_energy_ev * self.excitation_rate(te) * gas * ne
            )
        return loss

    def _bohm_nu(self, omega):
        return self.bohm * np.abs(omega) / 16.0

    def _dx_cell(self, f: np.ndarray) -> np.ndarray:
        """Central ``d f / dx`` on cells, one-sided half-steps at the ends."""
        if self.periodic_x:
            return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * self.dx)
        d = np.empty_like(f)
        d[1:-1] = (f[2:] - f[:-2]) / (2.0 * self.dx)
        d[0] = (f[1] - f[0]) / (2.0 * self.dx)
        d[-1] = (f[-1] - f[-2]) / (2.0 * self.dx)
        return d

    def _fields(self, phi: np.ndarray):
        """The electric field on the faces and the cells from the potential ``(nx, ny)``."""
        flat = phi.ravel()
        bx, bc = self._affine()
        shape_xf = self._shape_xf
        ex_xf = -(self._gxf @ flat + bx).reshape(shape_xf)
        ey_xf = -(self._gyxf_masked @ flat).reshape(shape_xf)
        ey_yf = -(self._gyf @ flat).reshape(self.nx, self.ny)
        ex_yf = -(self._gxyf @ flat + bc).reshape(self.nx, self.ny)
        ecx = -(self._dxc @ flat + bc).reshape(self.nx, self.ny)
        ecy = -(self._dyc @ flat).reshape(self.nx, self.ny)
        return ex_xf, ey_xf, ex_yf, ey_yf, ecx, ecy

    def _face_gas(self, gas: np.ndarray):
        gas_xf = (self._axf @ gas.ravel()).reshape(self._shape_xf)
        gas_yf = 0.5 * (gas + np.roll(gas, 1, axis=1))
        return gas_xf, gas_yf

    def _pressure_gradients(self, p: np.ndarray):
        """``grad p`` at the x-faces ``(dx, dy)`` and the y-faces ``(dx, dy)``."""
        nx = self.nx
        dpx_xf = np.empty(self._shape_xf)
        if self.periodic_x:
            dpx_xf[:nx] = (p - np.roll(p, 1, axis=0)) / self.dx
            dpx_xf[nx] = dpx_xf[0]
        else:
            dpx_xf[1:nx] = (p[1:] - p[:-1]) / self.dx
            dpx_xf[0] = (p[1] - p[0]) / self.dx
            dpx_xf[nx] = (p[-1] - p[-2]) / self.dx
        dpy_xf = (self._pyxf @ p.ravel()).reshape(self._shape_xf)
        dpy_yf = (p - np.roll(p, 1, axis=1)) / self.dy
        d = self._dx_cell(p)
        dpx_yf = 0.5 * (d + np.roll(d, 1, axis=1))
        scale = self.diamagnetic
        return scale * dpx_xf, scale * dpy_xf, scale * dpx_yf, scale * dpy_yf

    def _face_densities(self, n, mu_xf, h_xf, mu_yf, h_yf, fields):
        """Upwind-biased densities on the faces, by the sign of the E x B-like drift."""
        ex_xf, ey_xf, ex_yf, ey_yf = fields[:4]
        vn_x = -mu_xf * (ex_xf - h_xf * ey_xf)
        vn_y = -mu_yf * (ey_yf + h_yf * ex_yf)
        n_xf = np.maximum(_upwind3(n, vn_x > 0.0, 0, self.periodic_x), 0.0)
        n_yf = np.maximum(_upwind3(n, vn_y > 0.0, 1), 0.0)
        return n_xf, n_yf

    def _electron_like_flux(self, n, te, fields, gas, scale=1.0, lagged=None):
        """``Gamma = -scale mu_perp [[1, -h], [h, 1]] (n E + grad(n T_e))`` on the faces."""
        ex_xf, ey_xf, ex_yf, ey_yf = fields[:4]
        gas_xf, gas_yf = self._face_gas(gas)
        (mu_xf, h_xf), (mu_yf, h_yf) = self._collisions(gas_xf, gas_yf)
        n_xf, n_yf = self._face_densities(
            n, mu_xf, h_xf, mu_yf, h_yf, lagged if lagged is not None else fields
        )
        dpx_xf, dpy_xf, dpx_yf, dpy_yf = self._pressure_gradients(n * te)
        fx_xf = n_xf * ex_xf + dpx_xf
        fy_xf = n_xf * ey_xf + dpy_xf
        fy_yf = n_yf * ey_yf + dpy_yf
        fx_yf = n_yf * ex_yf + dpx_yf
        gam_x = -scale * mu_xf * (fx_xf - h_xf * fy_xf)
        gam_y = -scale * mu_yf * (fy_yf + h_yf * fx_yf)
        return (
            gam_x,
            gam_y,
            (n_xf, n_yf, mu_xf, h_xf, mu_yf, h_yf, dpx_xf, dpy_xf, dpx_yf, dpy_yf),
        )

    def _ion_fluxes(self, ni, vel):
        """Upwind ion fluxes on the faces; outflow only through the electrodes."""
        vx, vy = vel
        nx = self.nx
        vxf = np.empty(self._shape_xf)
        if self.periodic_x:
            vxf[:nx] = 0.5 * (vx + np.roll(vx, 1, axis=0))
            vxf[nx] = vxf[0]
            nxf = np.where(vxf[:nx] > 0.0, np.roll(ni, 1, axis=0), ni)
            nxf = np.vstack([nxf, nxf[:1]])
            vyf = 0.5 * (vy + np.roll(vy, 1, axis=1))
            nyf = np.where(vyf > 0.0, np.roll(ni, 1, axis=1), ni)
            return nxf * vxf, nyf * vyf
        vxf[1:nx] = 0.5 * (vx[1:] + vx[:-1])
        vxf[0], vxf[nx] = vx[0], vx[-1]
        nxf = np.where(vxf > 0.0, np.vstack([ni[:1], ni]), np.vstack([ni, ni[-1:]]))
        gam_x = nxf * vxf
        gam_x[0] = ni[0] * np.minimum(vx[0], 0.0)
        gam_x[nx] = ni[-1] * np.maximum(vx[-1], 0.0)
        vyf = 0.5 * (vy + np.roll(vy, 1, axis=1))
        nyf = np.where(vyf > 0.0, np.roll(ni, 1, axis=1), ni)
        return gam_x, nyf * vyf

    # -- the potential ----------------------------------------------------------------

    def _solve_potential(self, ne, ni, te, gas, dtau, gam_ix, gam_iy, lagged):
        """Implicit-permittivity Poisson solve for the potential at the end of an Euler step."""
        nx, ny = self.nx, self.ny
        gas_xf, gas_yf = self._face_gas(gas)
        (mu_xf, h_xf), (mu_yf, h_yf) = self._collisions(gas_xf, gas_yf)
        n_xf, n_yf = self._face_densities(ne, mu_xf, h_xf, mu_yf, h_yf, lagged)
        dpx_xf, dpy_xf, dpx_yf, dpy_yf = self._pressure_gradients(ne * te)
        sig_x = E * n_xf * mu_xf
        sig_y = E * n_yf * mu_yf
        # no conduction through the electrodes: their electron flux is explicit
        if not self.periodic_x:
            sig_x[[0, nx]] = 0.0
        sxx = EPSILON_0 + dtau * sig_x
        sxy = -dtau * sig_x * h_xf
        if not self.periodic_x:
            sxy[[0, nx]] = 0.0
        syy = EPSILON_0 + dtau * sig_y
        syx = dtau * sig_y * h_yf
        # known current: ions, electron pressure part, and the electrode boundary fluxes
        jk_x = E * gam_ix + E * mu_xf * (dpx_xf - h_xf * dpy_xf)
        jk_y = E * gam_iy + E * mu_yf * (dpy_yf + h_yf * dpx_yf)
        if not self.periodic_x:
            anode = self._anode_flux(ne, te, gas, lagged[0][-1], lagged[1][-1])
            jk_x[0] = E * gam_ix[0] - E * self._cathode_flux(gam_ix[0])
            jk_x[nx] = E * gam_ix[nx] - E * anode
        div_jk = self._divx @ jk_x.ravel() + self._divy @ jk_y.ravel()
        rho = E * (ni - ne).ravel()
        src_e = self.external_sources.get("ne", 0.0)
        src_i = self.external_sources.get("ni", 0.0)
        rho = rho + dtau * E * np.ravel(np.broadcast_to(src_i - src_e, ne.shape))
        bx, bc = self._affine()
        coefficients = np.concatenate(
            [sxx.ravel(), sxy.ravel(), syy.ravel(), syx.ravel()]
        )
        matrix = sp.csc_matrix(
            (self._pat_b @ coefficients, (self._pat_rows, self._pat_cols)),
            shape=(nx * ny, nx * ny),
        )
        rhs = (
            rho
            - dtau * div_jk
            + self._divx @ (sxx.ravel() * bx)
            + self._divy @ (syx.ravel() * bc)
        )
        if self.field_bc is not None:  # pure Neumann: pin the potential of cell 0
            matrix = matrix.tolil()
            matrix[0, 0] = matrix[0, 0] * (1.0 + 1e8)
            matrix = matrix.tocsc()
            rhs[0] = 0.0
        return self._solve_linear(matrix, rhs).reshape(nx, ny)

    def _cathode_flux(self, ion_flux: np.ndarray) -> np.ndarray:
        """Electron flux into the domain at the cathode, for the ion flux ``ion_flux`` (< 0)."""
        if self.cathode_current_density is not None:
            return np.full_like(ion_flux, self.cathode_current_density / E)
        return -self.secondary_emission * ion_flux

    def _solve_linear(self, matrix, rhs: np.ndarray) -> np.ndarray:
        """Solve ``matrix x = rhs``, refining with the last LU factors while they still work.

        The matrix changes by a few percent per stage, so the previous factorisation is a
        very good preconditioner; the iteration runs to a relative residual of 1e-12
        (the solution is as exact as a fresh factorisation) and refactorises when it
        does not get there in a few sweeps.
        """
        scale = np.linalg.norm(rhs) + 1e-300
        if self._lu is not None:
            x = self._lu.solve(rhs)
            for _ in range(10):
                residual = rhs - matrix @ x
                if np.linalg.norm(residual) <= 1e-12 * scale:
                    return x
                x = x + self._lu.solve(residual)
        self._lu = spla.splu(matrix, permc_spec="COLAMD")
        return self._lu.solve(rhs)

    def _anode_flux(self, ne, te, gas, ex, ey):
        """Electron flux out through the anode face: the drift of the adjacent cell, outflow only.

        With ``anode_thermal_flux`` the thermal flux of an absorbing wall is added.
        ``ex``, ``ey`` are the field components on the anode face (length ``ny``).
        """
        gas_xf, gas_yf = self._face_gas(gas)
        (mu_xf, h_xf), _ = self._collisions(gas_xf, gas_yf)
        dpx_xf, dpy_xf, _, _ = self._pressure_gradients(ne * te)
        fx = ne[-1] * ex + dpx_xf[-1]
        fy = ne[-1] * ey + dpy_xf[-1]
        flux = -mu_xf[-1] * (fx - h_xf[-1] * fy)
        if self.anode_thermal_flux:
            mean_speed = np.sqrt(8.0 * E * te[-1] / (np.pi * ELECTRON_MASS))
            flux = flux + 0.5 * ne[-1] * mean_speed
        return np.maximum(flux, 0.0)

    # -- the right-hand side -----------------------------------------------------------

    def _rhs(self, u: dict, phi_prev: np.ndarray, dtau: float):
        """Time derivatives of the unknowns and the new potential for an Euler step ``dtau``."""
        ne, ni, vx, vy, gas = u["ne"], u["ni"], u["vx"], u["vy"], u["N"]
        energy = "w" in u
        te = (
            np.clip(u["w"] / (1.5 * np.maximum(ne, N_FLOOR)), TE_FLOOR, TE_CEIL)
            if energy
            else np.full_like(ne, self.electron_temperature)
        )
        gas_c = self._collision_gas(gas, te)
        prev = self._fields(phi_prev)
        gam_ix, gam_iy = self._ion_fluxes(ni, (vx, vy))
        phi = self._solve_potential(ne, ni, te, gas_c, dtau, gam_ix, gam_iy, prev)
        fields = self._fields(phi)
        gam_x, gam_y, _ = self._electron_like_flux(ne, te, fields, gas_c, lagged=prev)
        if not self.periodic_x:
            gam_x[0] = self._cathode_flux(gam_ix[0])
            # explicit in the field, as in the Poisson solve, so the discrete charge
            # balance is exact
            gam_x[-1] = self._anode_flux(ne, te, gas_c, prev[0][-1], prev[1][-1])
        ecx, ecy = fields[4], fields[5]
        rate = self.ionization_rate(te) * gas * ne
        if self._rate_override is not None:
            rate = self._rate_override
        div_e = (self._divx @ gam_x.ravel() + self._divy @ gam_y.ravel()).reshape(
            ne.shape
        )
        div_i = (self._divx @ gam_ix.ravel() + self._divy @ gam_iy.ravel()).reshape(
            ne.shape
        )
        du = {"ne": -div_e + rate, "ni": -div_i + rate}
        du["N"] = -self.depletion * rate + self.refill_rate * (self.gas_density - gas)
        # ion momentum: field, friction, pressure, upwind advection
        ni_safe = np.maximum(ni, N_FLOOR)
        nu_i = self.ion_friction_rate * gas
        c = E / self.ion_mass
        grad_nx = self._dx_cell(ni)
        grad_ny = (np.roll(ni, -1, axis=1) - np.roll(ni, 1, axis=1)) / (2.0 * self.dy)
        t_over_m = self.ion_temperature * E / self.ion_mass
        adv_x = self._advect(vx, vx, vy)
        adv_y = self._advect(vy, vx, vy)
        # semi-implicit friction: dv/dt = F - nu v, evaluated for the Euler step dtau
        du["vx"] = c * ecx - t_over_m * grad_nx / ni_safe - adv_x - nu_i * vx
        du["vy"] = c * ecy - t_over_m * grad_ny / ni_safe - adv_y - nu_i * vy
        if energy:
            q_x, q_y, _ = self._electron_like_flux(
                ne * te * 1.5, te, fields, gas_c, scale=5.0 / 3.0, lagged=prev
            )
            if not self.periodic_x:
                q_x[0] = 2.0 * self.secondary_energy_ev * gam_x[0]
                q_x[-1] = 2.0 * te[-1].mean() * np.maximum(gam_x[-1], 0.0)
            div_q = (self._divx @ q_x.ravel() + self._divy @ q_y.ravel()).reshape(
                ne.shape
            )
            gcx = 0.5 * (gam_x[:-1] + gam_x[1:])
            gcy = 0.5 * (gam_y + np.roll(gam_y, -1, axis=1))
            joule = -(gcx * ecx + gcy * ecy)
            loss = self._energy_loss(ne, te, gas, gas_c)
            du["w"] = -div_q + self.heating_scale * joule - loss
            # d(loss)/dw for the implicit treatment of the loss in :meth:`_euler`
            te_up = te * 1.001
            dloss = (
                self._energy_loss(ne, te_up, gas, self._collision_gas(gas, te_up))
                - loss
            ) / (1.5 * np.maximum(ne, N_FLOOR) * 0.001 * te)
            du["_dloss"] = np.maximum(dloss, 0.0)
        for key, source in self.external_sources.items():
            if key in du:
                du[key] = du[key] + source
        return du, phi

    def linear_modes(
        self,
        state: DriftFluid2DState,
        mode: int,
        frozen: tuple[str, ...] = (),
        epsilon: float = 1e-6,
    ) -> LinearModes:
        """Linear stability of a y-uniform state to the azimuthal mode ``mode``, x-resolved.

        The Jacobian of the semi-discrete operator (the time derivative of every unknown,
        with the potential from Poisson's equation) about ``state``, restricted to
        perturbations ``exp(i k y)``, ``k = 2 pi mode / length_y``: one column per unknown and
        cell, by finite differences with a cosine and a sine perturbation (the operator is
        real and invariant along ``y``, so the response to ``exp(i k y)`` stays in that
        mode). The electrodes, the gradients, the Hall tensor and the energy equation are
        all in it; nothing is x-uniform or local. The state is made stationary by
        sources of its own residual (as :meth:`stationary_sources`); ``external_sources``
        are restored afterwards.

        ``frozen`` switches off responses, to see what each does: ``"temperature"`` (``T_e``
        held at its value, no energy perturbation), ``"ionization"`` (the ionization source held
        at its base value), ``"gas"`` (the neutral density not perturbed).

        With ``s = gamma - i omega`` the eigenvalues, ``exp(i k y + s t)``, ``gamma > 0``
        grows and ``omega / k > 0`` moves along ``+y`` (the E x B direction when ``E_x < 0``
        and ``B_z > 0``).
        """
        ny, nx = self.ny, self.nx
        for name in ("electron_density", "ion_density", "electron_temperature"):
            value = getattr(state, name)
            if np.abs(value - value[:, :1]).max() > 1e-9 * np.abs(value).max():
                raise ValueError(f"the state must be uniform along y ({name})")
        k = 2.0 * np.pi * mode / self.length_y
        saved = (self.external_sources, self.electron_temperature, self._rate_override)
        try:
            if "temperature" in frozen and self.electron_temperature is None:
                self.electron_temperature = state.electron_temperature.copy()
            if "ionization" in frozen:
                self._rate_override = None
                self._rate_override = (
                    self.ionization_rate(state.electron_temperature)
                    * state.gas_density
                    * state.electron_density
                )
            self.external_sources = {}
            u0 = self._pack(state)
            phi0 = state.potential

            def evaluate(u):
                # the anode flux and the upwind direction use the field of the previous
                # stage: take the one the state itself gives, so the lag is consistent
                _, phi = self._rhs(u, phi0, 0.0)
                return self._rhs(u, phi, 0.0)[0]

            du0 = evaluate(u0)
            self.external_sources = {k_: -v for k_, v in du0.items() if k_ in u0}
            base = evaluate(u0)
            keys = [name for name in u0 if not (name == "N" and "gas" in frozen)]
            cos, sin = np.cos(k * self.y), np.sin(k * self.y)
            phase = np.exp(-1j * k * self.y)
            matrix = np.zeros((len(keys) * nx, len(keys) * nx), complex)
            for column_key, key in enumerate(keys):
                scale = np.abs(u0[key][:, 0]) * epsilon
                scale = np.where(scale > 0.0, scale, epsilon)
                if key in ("vx", "vy"):
                    scale = np.full(nx, epsilon * (np.abs(u0[key]).max() + 1.0))
                for i in range(nx):
                    responses = []
                    for pattern in (cos, sin):
                        u = dict(u0)
                        u[key] = u0[key].copy()
                        u[key][i, :] += scale[i] * pattern
                        responses.append(evaluate(u))
                    for row_key, name in enumerate(keys):
                        response = (
                            (responses[0][name] - base[name])
                            + 1j * (responses[1][name] - base[name])
                        ) / scale[i]
                        matrix[
                            row_key * nx : (row_key + 1) * nx, column_key * nx + i
                        ] = (response @ phase) / ny
        finally:
            self.external_sources, self.electron_temperature, self._rate_override = (
                saved
            )
        values, vectors = np.linalg.eig(matrix)
        order = np.argsort(-values.real)
        profiles = {
            name: vectors[row * nx : (row + 1) * nx][:, order]
            for row, name in enumerate(keys)
        }
        return LinearModes(
            mode, k, values[order], profiles, self.x.copy(), matrix=matrix
        )

    def stationary_sources(self, state: DriftFluid2DState) -> dict[str, np.ndarray]:
        """The external sources that make ``state`` stationary (well-balanced equilibria).

        Sets ``external_sources`` to ``-d u / d t`` of the state without them and returns
        it, so that the state is a steady state of the discrete equations and only a
        perturbation evolves; the way to test linear growth on an equilibrium that
        the physical sources would not hold.
        """
        self.external_sources = {}
        u = self._pack(state)
        du, _ = self._rhs(u, state.potential, 0.0)
        self.external_sources = {key: -value for key, value in du.items() if key in u}
        return self.external_sources

    def _advect(self, f, vx, vy):
        """``v . grad f`` by first-order upwind; zero gradient at the electrodes."""
        if self.periodic_x:
            fm, fp = np.roll(f, 1, axis=0), np.roll(f, -1, axis=0)
        else:
            fm = np.vstack([f[:1], f[:-1]])
            fp = np.vstack([f[1:], f[-1:]])
        dfx = np.where(vx > 0.0, (f - fm) / self.dx, (fp - f) / self.dx)
        dfy = np.where(
            vy > 0.0,
            (f - np.roll(f, 1, axis=1)) / self.dy,
            (np.roll(f, -1, axis=1) - f) / self.dy,
        )
        return vx * dfx + vy * dfy

    # -- time stepping -----------------------------------------------------------------

    def _pack(self, state: DriftFluid2DState) -> dict:
        u = {
            "ne": state.electron_density.copy(),
            "ni": state.ion_density.copy(),
            "vx": state.ion_velocity[0].copy(),
            "vy": state.ion_velocity[1].copy(),
            "N": state.gas_density.copy(),
        }
        if self.electron_temperature is None:
            u["w"] = 1.5 * state.electron_density * state.electron_temperature
        return u

    def _unpack(self, u: dict, phi: np.ndarray, time: float) -> DriftFluid2DState:
        if "w" in u:
            te = np.clip(
                u["w"] / (1.5 * np.maximum(u["ne"], N_FLOOR)), TE_FLOOR, TE_CEIL
            )
        else:
            te = np.full_like(u["ne"], self.electron_temperature)
        return DriftFluid2DState(
            self.x,
            self.y,
            u["ne"],
            u["ni"],
            np.array([u["vx"], u["vy"]]),
            u["N"],
            te,
            phi,
            time,
        )

    def _euler(self, u, phi_prev, dt):
        du, phi = self._rhs(u, phi_prev, dt)
        out = {}
        for key, value in u.items():
            if key == "w":
                # backward Euler on the (stiff, local) energy loss, linearised in w
                new = value + dt * du["w"] / (1.0 + dt * du["_dloss"])
                new = np.maximum(new, 1.5 * TE_FLOOR * N_FLOOR)
            else:
                new = value + dt * du[key]
                if key in ("ne", "ni", "N"):
                    clipped = np.maximum(new, N_FLOOR)
                    self.floor_added[key] += float((clipped - new).sum())
                    new = clipped
            out[key] = new
        return out, phi

    def step(self, state: DriftFluid2DState, dt: float) -> DriftFluid2DState:
        """Advance by ``dt`` with SSP-RK3; the potential of the result is that of the last stage."""
        u0 = self._pack(state)
        phi = state.potential
        u1, phi = self._euler(u0, phi, dt)
        u2, phi = self._euler(u1, phi, dt)
        u2 = {k: 0.75 * u0[k] + 0.25 * u2[k] for k in u0}
        u3, phi = self._euler(u2, phi, dt)
        u3 = {k: u0[k] / 3.0 + 2.0 / 3.0 * u3[k] for k in u0}
        return self._unpack(u3, phi, state.time + dt)

    def suggest_timestep(self, state: DriftFluid2DState, cfl: float = 0.4) -> float:
        """A step with Courant number ``cfl``: electron drift, ions, and diffusion.

        The diffusive limit is that of the (explicit) energy and density diffusion with
        ``D = (5/3) mu_perp T_e``, which becomes the binding one where ``B`` is small
        and the electrons are hot.
        """
        fields = self._fields(state.potential)
        gas_c = self._collision_gas(state.gas_density, state.electron_temperature)
        gas_xf, gas_yf = self._face_gas(gas_c)
        (mu_xf, h_xf), (mu_yf, h_yf) = self._collisions(gas_xf, gas_yf)
        vx = np.abs(mu_xf * (fields[0] - h_xf * fields[1]))
        vy = np.abs(mu_yf * (fields[3] + h_yf * fields[2]))
        speed_x = max(vx.max(), np.abs(state.ion_velocity[0]).max())
        speed_y = max(vy.max(), np.abs(state.ion_velocity[1]).max())
        te = np.max(state.electron_temperature)
        sound = np.sqrt(E * (te + self.ion_temperature) / self.ion_mass)
        step = cfl / ((speed_x + sound) / self.dx + (speed_y + sound) / self.dy)
        mu_max = max(mu_xf.max(), mu_yf.max())
        # cells without electrons (the T_e of w / n is meaningless there) do not set the limit
        populated = state.electron_density > 1e-3 * state.electron_density.max()
        diffusivity = 5.0 / 3.0 * mu_max * state.electron_temperature[populated].max()
        diffusive = 0.5 * cfl / (diffusivity * (1.0 / self.dx**2 + 1.0 / self.dy**2))
        return min(step, diffusive)

    def run(
        self,
        state: DriftFluid2DState,
        t_end: float,
        dt: float | None = None,
        cfl: float = 0.4,
        record_every: float | None = None,
        probe_x: float | None = None,
        callback: Callable[[DriftFluid2DState], None] | None = None,
    ) -> Run:
        """Advance ``state`` by ``t_end`` seconds and record the electron density along ``y``.

        Args:
            state: The initial state.
            t_end: The duration (s).
            dt: A fixed step (s); by default the step is re-chosen from ``cfl``
                every 20 steps.
            cfl: The Courant number of the chosen step.
            record_every: The recording interval (s); ``t_end / 200`` by default.
            probe_x: Where along ``x`` to record (m); mid-gap by default.
            callback: Called with the state at every recording.
        """
        record_every = t_end / 200.0 if record_every is None else record_every
        probe_x = 0.5 * self.length_x if probe_x is None else probe_x
        index = int(np.argmin(np.abs(self.x - probe_x)))
        times, probe = [], []
        series: dict[str, list] = {
            "density_cathode": [],
            "potential": [],
            "e_y": [],
            "v_exb_y": [],
        }

        def record(st: DriftFluid2DState) -> None:
            times.append(st.time)
            probe.append(st.electron_density[index].copy())
            fields = self._fields(st.potential)
            series["density_cathode"].append(st.electron_density[0].copy())
            series["potential"].append(st.potential[index].copy())
            series["e_y"].append(fields[5][index].copy())
            series["v_exb_y"].append(-fields[4][index] / self._bc[index])

        next_record = state.time
        t_stop = state.time + t_end
        steps = 0
        step_size = dt
        while state.time < t_stop - 1e-15 * t_end:
            if state.time >= next_record - 1e-12 * record_every:
                record(state)
                next_record += record_every
                if callback is not None:
                    callback(state)
            if dt is None and steps % 20 == 0:
                step_size = self.suggest_timestep(state, cfl)
            h = min(step_size, t_stop - state.time, next_record - state.time)
            h = max(h, 1e-3 * step_size)
            state = self.step(state, h)
            steps += 1
            if not np.all(np.isfinite(state.potential)):
                raise FloatingPointError(f"the run diverged at t = {state.time:.3e} s")
        record(state)
        probe_arr = np.array(probe)
        return Run(
            np.array(times),
            probe_arr,
            float(self.x[index]),
            mode_amplitudes(probe_arr),
            state,
            steps,
            self.length_y,
            {key: np.array(value) for key, value in series.items()},
        )

    # -- diagnostics ---------------------------------------------------------------------

    def exb_velocity(self, state: DriftFluid2DState) -> np.ndarray:
        """The cell-centre E x B drift ``(E x B) / B^2``, shape ``(2, nx, ny)`` (m/s)."""
        ecx, ecy = self._fields(state.potential)[4:6]
        b = self._bc[:, None]
        return np.array([ecy / b, -ecx / b])

    def electric_field(self, state: DriftFluid2DState) -> np.ndarray:
        """The cell-centre electric field ``(E_x, E_y)``, shape ``(2, nx, ny)`` (V/m)."""
        return np.array(self._fields(state.potential)[4:6])

    def electron_flux(self, state: DriftFluid2DState):
        """The electron flux on the x-faces ``(nx + 1, ny)`` and the y-faces ``(nx, ny)`` (m^-2 s^-1)."""
        fields = self._fields(state.potential)
        gx, gy, _ = self._electron_like_flux(
            state.electron_density,
            state.electron_temperature,
            fields,
            self._collision_gas(state.gas_density, state.electron_temperature),
        )
        return gx, gy

    def charge_residual(self, state: DriftFluid2DState) -> float:
        """Largest ``|eps0 div E - rho| / (e n)``: how far the state is from Poisson's equation."""
        ex_xf, _, _, ey_yf, _, _ = self._fields(state.potential)
        div_e = (self._divx @ ex_xf.ravel() + self._divy @ ey_yf.ravel()).reshape(
            state.potential.shape
        )
        rho = E * (state.ion_density - state.electron_density)
        scale = E * np.maximum(state.electron_density, N_FLOOR)
        return float(np.max(np.abs(EPSILON_0 * div_e - rho) / scale))
