"""The transport budget of a model spoke.

A density pattern ``n = n_0(x) (1 + 0.3 cos(k y))`` and a potential pattern shifted against
it by a phase, in the plane of the axial-azimuthal simulations, built so that the electron
momentum balance holds exactly; :mod:`wellbalanced.transport` then recovers the flux across
``B`` term by term: the ``E x B`` flux of the correlated fluctuations (all of it in the
spoke band here), the collisional flux, the stress term, and a residual at the level of
the finite differences. The second panel shows the loop identity, ``<n E_y> = <phi d_y n>``,
and the flux the single mode carries as a function of the phase between the potential and
the density: zero when the two patterns are aligned, largest at a quarter wavelength.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

from wellbalanced.transport import loop_identity, mode_fluxes, momentum_budget

B, NU, LY, LX = 0.02, 2e7, 0.08, 0.02


def spoke(phase: float) -> xr.Dataset:
    """Electron moments of a mode-2 spoke with the potential shifted by ``phase``."""
    x = np.linspace(0.0, LX, 41)
    y = np.arange(128) * LY / 128
    k = 2 * np.pi * 2 / LY
    xx, yy = np.meshgrid(x, y, indexing="ij")
    n = 2e15 * np.exp(-(((xx - 0.012) / 0.006) ** 2)) * (1 + 0.3 * np.cos(k * yy))
    phi = 5.0 * np.cos(k * yy + phase)
    ey = 5.0 * k * np.sin(k * yy + phase)
    uy = 5e3 / B * np.ones_like(xx)  # the E x B drift of E_x = -5 kV/m
    stress = 2e26 * np.sin(np.pi * xx / LX)  # n <v_x v_y>
    d_stress = 2e26 * np.pi / LX * np.cos(np.pi * xx / LX)
    factor = ELECTRON_MASS / (ELEMENTARY_CHARGE * B)
    nux = n * ey / B + factor * (NU * n * uy + d_stress)  # the identity, pointwise
    ds = xr.Dataset(
        {
            "density": (("x", "y"), n),
            "flux_x": (("x", "y"), nux),
            "flux_y": (("x", "y"), n * uy),
            "moment_xy": (("x", "y"), stress),
            "e_field_y": (("x", "y"), ey),
            "potential": (("x", "y"), phi),
            "b_field_z": ("x", np.full(x.size, B)),
        },
        coords={"x": x, "y": y},
    ).expand_dims(t=[0.0])
    ds.attrs = {"mass": ELECTRON_MASS, "charge": -ELEMENTARY_CHARGE}
    return ds


fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

budget = momentum_budget(spoke(0.6), collision_frequency=NU, spoke_modes=4).isel(t=0)
for name in ("flux", "exb_spoke", "collisional", "stress", "residual"):
    axes[0].plot(budget["x"] * 1e3, budget[name] * ELEMENTARY_CHARGE, label=name)
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("electron flux across B, e Γ_x [A/m²]")
axes[0].set_title("the budget at a phase of 0.6 rad")
axes[0].legend(fontsize=8)

phases = np.linspace(-np.pi, np.pi, 25)
carried, identity = [], []
for phase in phases:
    ds = spoke(phase)
    carried.append(
        float(
            mode_fluxes(ds, modes=2)["exb_mode"]
            .isel(t=0)
            .sel(m=2, x=0.012, method="nearest")
        )
    )
    identity.append(
        float(
            loop_identity(ds)["exb_from_potential"]
            .isel(t=0)
            .sel(x=0.012, method="nearest")
        )
    )
axes[1].plot(phases, np.array(carried) * ELEMENTARY_CHARGE, label="mode 2 flux")
axes[1].plot(
    phases, np.array(identity) * ELEMENTARY_CHARGE, "k--", label="⟨φ ∂_y n⟩ / B"
)
axes[1].set_xlabel("phase of the potential against the density [rad]")
axes[1].set_ylabel("e Γ_x at x = 12 mm [A/m²]")
axes[1].set_title("flux against the phase shift")
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("transport_budget.png", dpi=150)
print(
    budget[["flux", "exb", "collisional", "stress", "residual"]].isel(x=24).to_pandas()
)
