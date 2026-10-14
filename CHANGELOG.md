# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `GlobalModel`: volume-averaged balances of every species, the electron energy and
  optionally the gas temperature; Bohm wall losses with an edge-to-centre ratio and
  magnetic cusps, neutral wall reactions, feed, pumping and volume sources; steady states
  by pseudo-time integration and a root finder; continuation scans.
- `Fluid1D`: drift-diffusion of every species (Scharfetter-Gummel), electron energy and
  Poisson's equation, fully implicit; walls, prescribed densities or fluxes, changeable
  between steps.
- `Species` and `Reaction`, with rates from plasmacoll cross sections, rate tables,
  constants or functions.
- `wellbalanced.neutralizer`: a beam-driven plasma neutralizer; the beam's charge
  states from plasmacoll on gas and plasma targets, its sources (stripped electrons,
  captures, gas ionization) for a global model of the channel, gas flow at the gas
  temperature, a fixed point; illustrative deuterium placeholders that warn.
- `DriftFluid2D` (`wellbalanced.driftfluid2d`): a 2D axial-azimuthal drift-fluid model of a
  magnetron (magnetised electrons with the Hall tensor, inertial ions, neutral depletion
  and refill, electron energy, semi-implicit Poisson, SSP-RK3) for the gradient-drift
  instability and spokes; well-balanced `external_sources`, the local dispersion relation
  `gradient_drift_roots`, and spoke diagnostics (`spoke_report`, `fit_mode`).
- `Magnetron` (`wellbalanced.magnetron`): a global model of the ionization region of a
  (HiPIMS) magnetron with Ar, a sputtered metal (table of Al, Cr, Cu, Ti, Au, Mo, Nb),
  ionization from the Lieberman and Lichtenberg argon fits and the Lotz approximation,
  sputtering and self-sputtering by the Bohm ion flux to the target, argon refill and
  sputter wind; driven by the discharge current and voltage (dc or pulse), with a fit of
  the heating fraction, `current_scan` and pulses to a periodic state, and the spoke
  diagnostics (rarefaction, metal to argon ratio, metal share of the ion flux,
  self-sputtering share). Yields, geometry and heating are placeholders that warn.
- `GlobalModel`: a `VolumeSource` rate that depends on the temperatures, `power` as a
  function of the time, `evolve` (the balances integrated in time, with breakpoints at
  the edges of a pulse) and `thermalize_electrons` (no cooling below the gas in an
  afterglow).
- `to_dataset()` on every result and `stack` for scans: labelled xarray datasets.
- Examples (an argon global model, a 1D argon discharge, a neutralizer scan), plotted with
  plasma-plots, and a tutorial.

### Removed

- The template's console entry point, example and tutorial.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
