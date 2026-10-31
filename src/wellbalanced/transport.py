"""Cross-field electron transport from the moments of a 2D ``(x, y)`` simulation.

The plane is the one of the axial-azimuthal simulations (Charoy's benchmark, Boeuf's
magnetron): ``x`` across the field toward the anode, ``y`` the periodic ``E x B``
direction, ``B = B_z(x) z``. Averaging the ``y`` component of the momentum equation of a
species over ``y`` (``<.>``, periodic so ``<d_y .> = 0``) in flux form gives an exact
identity for its ``y``-averaged flux across ``B``, ``Gamma_x = <n u_x>``::

    q B_z <n u_x> = q <n E_y> - m nu <n u_y> - m d_t <n u_y> - m d_x <n v_x v_y>

(``<n E_y> = <dn dE_y>`` since ``<E_y> = 0``): the fluctuation-driven ``E x B`` flux,
the collisional (Pedersen) flux, the inertia and the divergence of the full second
moment, which is the pressure tensor over the mass plus the convective ``n u_x u_y``.
:func:`momentum_budget` evaluates every term from the moments a PIC run deposits
(picnic's ``MomentRecorder``) and returns them as fluxes on ``(t, x)``, with the
residual of the identity, so that the budget is checked rather than assumed, and the
``E x B`` term split into a spoke band and the rest (an exact Parseval split of the
correlator). :func:`loop_identity` checks ``<n E_y> = <phi d_y n>``, the identity
behind the spoke-front picture of the transport (the flux as the correlation of the
potential with the azimuthal density gradient), and :func:`mode_fluxes` resolves the
``E x B`` flux by azimuthal mode with the phase between density and potential.

Input: a dataset with dims ``t``, ``x``, ``y`` (``y`` periodic, uniform) holding
``density``, ``flux_x``, ``flux_y``, ``moment_xy`` (``n <v_x v_y>``) of one species,
``e_field_y``, ``potential`` and ``b_field_z`` (``x`` or ``(x, y)``); the species'
``mass`` and ``charge`` as arguments or dataset attributes. Collisions enter through
``collision_frequency``, a number or an array over ``x`` or ``(t, x)``; a collisionless
run has none.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

__all__ = ["loop_identity", "mode_fluxes", "momentum_budget"]

E = ELEMENTARY_CHARGE


def _species(ds: xr.Dataset, species: str | None) -> tuple[xr.Dataset, int]:
    """The moments of one species (``species`` dim dropped) and its index in the dataset."""
    if "species" not in ds.dims:
        return ds, 0
    names = list(ds.coords["species"].values) if "species" in ds.coords else []
    if species is None:
        if ds.sizes["species"] != 1:
            raise ValueError("several species in the dataset: name one")
        return ds.isel(species=0), 0
    if names and species in names:
        return ds.sel(species=species), names.index(species)
    raise KeyError(f"no species {species!r} in the dataset")


def _mass_charge(
    ds: xr.Dataset, index: int, mass: float | None, charge: float | None
) -> tuple[float, float]:
    """``(mass, charge)`` from the arguments or the dataset's attributes (one value or one per species)."""
    if mass is None:
        mass = ds.attrs.get("mass", ELECTRON_MASS)
    if charge is None:
        charge = ds.attrs.get("charge", -E)
    if np.ndim(mass):
        mass = np.asarray(mass)[index]
    if np.ndim(charge):
        charge = np.asarray(charge)[index]
    return float(mass), float(charge)


def _b_field(ds: xr.Dataset) -> xr.DataArray:
    b = ds["b_field_z"]
    return b.mean("y") if "y" in b.dims else b


def _correlation(
    a: xr.DataArray, b: xr.DataArray, modes: slice | None = None
) -> xr.DataArray:
    """``<a b>`` over ``y`` from the spectra (Parseval), optionally over a band of modes.

    The band ``modes`` is a slice of the non-negative mode numbers ``m``; ``m = 0`` is
    the product of the means, the others count twice (``+m`` and ``-m``).
    """
    n = a.sizes["y"]
    fa = np.fft.rfft(a.values, axis=a.dims.index("y"))
    fb = np.fft.rfft(b.values, axis=b.dims.index("y"))
    product = (fa * np.conj(fb)).real / n**2
    axis = a.dims.index("y")
    product = np.moveaxis(product, axis, -1)
    weights = np.full(product.shape[-1], 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0  # the Nyquist mode appears once
    if modes is not None:
        keep = np.zeros_like(weights)
        keep[modes] = 1.0
        weights = weights * keep
    out = (product * weights).sum(axis=-1)
    dims = tuple(d for d in a.dims if d != "y")
    return xr.DataArray(
        out, dims=dims, coords={d: a.coords[d] for d in dims if d in a.coords}
    )


def momentum_budget(
    ds: xr.Dataset,
    *,
    species: str | None = None,
    mass: float | None = None,
    charge: float | None = None,
    collision_frequency: float | xr.DataArray = 0.0,
    spoke_modes: int | None = None,
) -> xr.Dataset:
    """The terms of the ``y``-averaged momentum identity as fluxes across ``B``.

    Every variable is a particle flux density along ``x`` (m^-2 s^-1) on ``(t, x)``:
    ``flux`` (``<n u_x>`` measured), ``exb`` (``<n E_y> / B``), ``collisional``
    (``-m nu <n u_y> / (q B)``), ``inertial`` (``-m d_t <n u_y> / (q B)``), ``stress``
    (``-m d_x <n v_x v_y> / (q B)``, split into ``pressure`` and ``convective`` with
    ``<n u_x u_y>``), their ``sum`` and the ``residual`` ``flux - sum`` (zero for an
    exact momentum balance up to the deposit and the leapfrog stagger). With
    ``spoke_modes`` the ``E x B`` term is also given as ``exb_spoke`` (modes
    ``|m| <= spoke_modes``) and ``exb_rest``. ``hall_parameter`` is
    ``|<n u_y>| / |<n u_x>|``, the effective Hall parameter (Bohm: 16).
    """
    ds, index = _species(ds, species)
    mass, charge = _mass_charge(ds, index, mass, charge)
    b = _b_field(ds)
    n, nux, nuy = ds["density"], ds["flux_x"], ds["flux_y"]
    flux = nux.mean("y")
    momentum = nuy.mean("y")
    exb = _correlation(n, ds["e_field_y"]) / b
    factor = -mass / (charge * b)
    nu = collision_frequency
    if isinstance(nu, xr.DataArray):
        nu = nu.broadcast_like(momentum) if set(nu.dims) <= set(momentum.dims) else nu
    collisional = factor * nu * momentum
    inertial = (
        factor * momentum.differentiate("t")
        if ds.sizes.get("t", 1) > 1
        else xr.zeros_like(momentum)
    )
    second = ds["moment_xy"].mean("y")
    with np.errstate(divide="ignore", invalid="ignore"):
        convective_point = (nux * nuy / n).where(n > 0, 0.0)
    convective = factor * convective_point.mean("y").differentiate("x")
    pressure = factor * (second - convective_point.mean("y")).differentiate("x")
    stress = pressure + convective
    total = exb + collisional + inertial + stress
    out = xr.Dataset(
        {
            "flux": flux,
            "exb": exb,
            "collisional": collisional,
            "inertial": inertial,
            "stress": stress,
            "pressure": pressure,
            "convective": convective,
            "sum": total,
            "residual": flux - total,
            "hall_parameter": np.abs(momentum) / np.abs(flux).where(np.abs(flux) > 0),
        }
    )
    if spoke_modes is not None:
        spoke = _correlation(n, ds["e_field_y"], slice(0, spoke_modes + 1)) / b
        out["exb_spoke"] = spoke
        out["exb_rest"] = exb - spoke
    for name in out.data_vars:
        if name != "hall_parameter":
            out[name].attrs["units"] = "1/(m^2 s)"
    out.attrs = {"mass": mass, "charge": charge}
    return out.transpose("t", "x", ...) if "t" in out.dims else out


def loop_identity(
    ds: xr.Dataset, *, species: str | None = None, derivative: str = "centred"
) -> xr.Dataset:
    """``<n E_y> / B`` against ``<phi d_y n> / B``: the loop identity of the spoke front.

    Both are the ``E x B`` flux across ``B`` (m^-2 s^-1) on ``(t, x)``, equal for a
    periodic ``y`` with ``E_y = -d_y phi`` when ``d_y n`` is taken with the same
    discrete derivative as the field: ``derivative="centred"`` (the second-order
    difference of a PIC field solver, exact to rounding then) or ``"spectral"`` (exact
    for smooth fields; for a field differenced on the grid it differs by the factor
    ``sin(k dy) / (k dy)`` of the short waves, which is what the ``difference`` then
    measures).
    """
    ds, _ = _species(ds, species)
    b = _b_field(ds)
    n, phi = ds["density"], ds["potential"]
    from_field = _correlation(n, ds["e_field_y"]) / b
    dy = float(n["y"][1] - n["y"][0])
    if derivative == "centred":
        dn = (n.roll(y=-1) - n.roll(y=1)) / (2 * dy)
    elif derivative == "spectral":
        k = 2 * np.pi * np.fft.rfftfreq(n.sizes["y"], dy)
        axis = n.dims.index("y")
        spectrum = np.fft.rfft(n.values, axis=axis)
        shape = [1] * spectrum.ndim
        shape[axis] = k.size
        values = np.fft.irfft(
            1j * k.reshape(shape) * spectrum, n=n.sizes["y"], axis=axis
        )
        dn = xr.DataArray(values, dims=n.dims, coords=n.coords)
    else:
        raise ValueError("derivative is 'centred' or 'spectral'")
    from_potential = _correlation(phi, dn) / b
    out = xr.Dataset(
        {
            "exb_from_field": from_field,
            "exb_from_potential": from_potential,
            "difference": from_field - from_potential,
        }
    )
    return out.transpose("t", "x", ...) if "t" in out.dims else out


def mode_fluxes(
    ds: xr.Dataset, *, species: str | None = None, modes: int = 8
) -> xr.Dataset:
    """The ``E x B`` flux by azimuthal mode and the phase between density and potential.

    ``exb_mode`` (``m = 1 .. modes``, m^-2 s^-1 on ``(t, x, m)``) sums to the ``E x B``
    term of :func:`momentum_budget` without ``m = 0``; ``phase`` is ``arg(phi_m n_m*)``
    (rad), the angle by which the potential pattern is shifted against the density
    pattern (``n ~ cos(k y)``, ``phi ~ cos(k y + phase)``: the potential maximum sits
    ``phase / k`` before the density maximum along ``y``), which sets the sign and size
    of each mode's flux, ``|n_m| |phi_m| k_m sin(phase) / (2 B)`` for the ``E x B``
    part; ``density_amplitude`` and ``potential_amplitude`` are the amplitudes of the
    modes (half the peak-to-peak).
    """
    ds, _ = _species(ds, species)
    b = _b_field(ds)
    n, phi, ey = ds["density"], ds["potential"], ds["e_field_y"]
    axis = n.dims.index("y")
    ny = n.sizes["y"]
    fn = (
        np.moveaxis(np.fft.rfft(n.values, axis=axis), axis, -1)[..., 1 : modes + 1] / ny
    )
    fphi = (
        np.moveaxis(np.fft.rfft(phi.values, axis=axis), axis, -1)[..., 1 : modes + 1]
        / ny
    )
    fey = (
        np.moveaxis(np.fft.rfft(ey.values, axis=axis), axis, -1)[..., 1 : modes + 1]
        / ny
    )
    dims = tuple(d for d in n.dims if d != "y") + ("m",)
    coords = {d: n.coords[d] for d in n.dims if d != "y" and d in n.coords}
    coords["m"] = np.arange(1, modes + 1)
    bb = b.values[..., None] if b.ndim else b.values
    return xr.Dataset(
        {
            "exb_mode": (
                dims,
                2 * (fn * np.conj(fey)).real / bb,
                {"units": "1/(m^2 s)"},
            ),
            "phase": (dims, np.angle(fphi * np.conj(fn)), {"units": "rad"}),
            "density_amplitude": (dims, 2 * np.abs(fn), {"units": "1/m^3"}),
            "potential_amplitude": (dims, 2 * np.abs(fphi), {"units": "V"}),
        },
        coords=coords,
    )
