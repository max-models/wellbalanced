"""Gradient-drift instability with kinetic ions.

The local dispersion relation of :mod:`wellbalanced.dispersion` at the parameters of the
ionisation region of a DC magnetron, in three views:

1. growth rate and phase speed against the wavenumber, cold against kinetic ions: the fast
   mode sits in the cold limit (``zeta ~ 25 i``), ion kinetics only matter near marginality;
2. the compressibility of the drifts: the growth rate against the field gradient length for
   the 2D model field ``B(x) z`` (``1 / L_n - 1 / L_B``) and a curl-free field
   (``1 / L_n - 2 / L_B``), which is stable for ``L_n > L_B / 2``;
3. the comparison with Boeuf's relation (:func:`wellbalanced.driftfluid2d.boeuf_dispersion_roots`)
   at his point: the grad-B terms decide, and the 2D PIC with a varied ``L_B`` is the test.
"""

import matplotlib.pyplot as plt
import numpy as np

from wellbalanced.dispersion import ColdIons, KineticIons, LocalDispersion, Plasma
from wellbalanced.driftfluid2d import boeuf_dispersion_roots

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

# 1. k scan, cold against kinetic ions
plasma = Plasma(1e16, -1e4, 0.05, 3.0, -3e-3)
ky = np.linspace(100.0, 1500.0, 15)
for ions, label in (
    (ColdIons(), "cold ions"),
    (KineticIons(temperature=0.026), "300 K ions"),
):
    scan = LocalDispersion(plasma, ions, quasineutral=True).scan(ky)
    axes[0].plot(ky, scan["growth_rate"] * 1e-6, label=label)
axes[0].set(
    xlabel="$k_y$ (1/m)", ylabel="growth rate (1/µs)", title="cold against kinetic"
)
axes[0].legend()

# 2. compressibility against the field gradient length
lengths = np.linspace(4e-3, 30e-3, 14)
for curl_free, label in (
    (False, "model field $B(x)\\hat z$"),
    (True, "curl-free field"),
):
    growth = [
        LocalDispersion(
            Plasma(1e16, -1e4, 0.05, 3.0, -3e-3, -length, curl_free_field=curl_free),
            ColdIons(),
            quasineutral=True,
        )
        .roots(0.0, 393.0)
        .growth_rate
        for length in lengths
    ]
    # a stable case has no growing root (nan): drawn as zero
    axes[1].plot(lengths * 1e3, np.nan_to_num(np.array(growth)) * 1e-6, label=label)
axes[1].axvline(6.0, color="gray", ls=":", label="$L_B = 2 L_n$")
axes[1].set(xlabel="$|L_B|$ (mm)", ylabel="growth rate (1/µs)", title="compressibility")
axes[1].legend()

# 3. Boeuf's point: his relation against this one
wavelengths = np.linspace(0.004, 0.06, 30)
his, mine = [], []
for wavelength in wavelengths:
    k = 2 * np.pi / wavelength
    roots = boeuf_dispersion_roots(k, 0.8e15, -2200.0, 6e-3, 11.4, -2.2e-3, -6e-3, 2e7)
    his.append(max(roots.imag.max(), 0.0))
    point = Plasma(0.8e15, -2200.0, 6e-3, 11.4, -2.2e-3, -6e-3, np.inf, 2e7)
    mine.append(LocalDispersion(point, ColdIons()).roots(0.0, k).growth_rate)
mine = np.maximum(np.nan_to_num(np.array(mine)), 0.0)  # stable: no growing root
axes[2].plot(wavelengths * 1e2, np.array(his) * 1e-6, label="Boeuf, Eq. (1)")
axes[2].plot(wavelengths * 1e2, mine * 1e-6, label="drift fluid, model field")
axes[2].set(
    xlabel="wavelength (cm)", ylabel="growth rate (1/µs)", title="Boeuf's point"
)
axes[2].legend()

fig.tight_layout()
fig.savefig("magnetron_dispersion.png", dpi=150)
print("saved magnetron_dispersion.png")
