# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `RFDriver` (`wellbalanced.driver`): a 0D RF driver, the `GlobalModel` of its plasma
  coupled to a transformer model of the coil (Piejak, Godyak and Alexandrovich 1992;
  Lieberman and Lichtenberg sec. 12.2) with the cold collisional plasma's complex skin
  depth, so the collisional and the electron-inertia limits are one expression; Nagaoka's
  coefficient for the short coil; the exact uniform column in a long solenoid
  (`solenoid_impedance`) as an alternative coupling. The coil current, coil voltage, RF
  power and RF power transfer efficiency for an absorbed power, and scans over the power,
  frequency or geometry in seconds. `illustrative_hydrogen_driver` (BUG geometry, H2 at
  0.3 Pa) runs on placeholder rates and warns. Example `rf_driver_0d.py`.
- `InductiveField2D` (`wellbalanced.inductive`): the magnetoquasistatic `A_z` of coil
  windings, grounded or floating perfect conductors and a prescribed cold plasma
  (`sigma = n_e e^2 / (m_e (nu + i omega))`) in a 2D Cartesian cut, on stretched
  tensor-product grids; time-harmonic (`solve`, one sparse LU per frequency) and
  time-dependent (`time_stepper`, trapezoidal, its steady state the harmonic solution at
  the modified frequency); fields, absorbed and delivered power (equal to round-off), the
  coil current for a power, the plasma resistance and the RF power transfer efficiency.
  Example `rf_driver_inductive_field_2d.py` on the geometry of Zielke et al. (NIBS'22).
  simgeometry is now a dependency (the shapes of windings and conductors).
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
  and refill, the electron energy with elastic, ionization and excitation losses
  (implicit), full Poisson with an implicit permittivity, SSP-RK3) for the gradient-drift
  instability and spokes. Well-balanced `external_sources`, a prescribed cathode
  current (`cathode_current_density`, Boeuf's condition: a fixed `gamma` avalanches when
  `gamma M >= 1`), energy-dependent argon rates bounded to 0.5-100 eV (placeholders),
  `periodic_x`/`field_bc` for uniform-plasma tests against `uniform_plasma_roots` (the
  full-Poisson local dispersion relation) and `gradient_drift_roots`, and spoke
  diagnostics (`spoke_report`, `fit_mode`), and `linear_modes`: the x-resolved Jacobian of the
  discretised operator about a y-uniform state, per azimuthal mode, with switches that freeze the
  T_e, ionization or gas response (checked against the dispersion relation and a time-domain run).
  Diagnostic switches: `diamagnetic`, `heating_scale`, `anode_thermal_flux` (off: it does not
  settle), and `floor_added`, the density the floors create.
- `GasDynamics` (`wellbalanced.gasdynamics`): a two-species (argon and sputtered metal)
  compressible neutral gas in 1D or 2D (axial and periodic azimuthal): MUSCL and HLLC with
  SSP-RK3, the diffusion of the metal through the argon and optional heat conduction
  applied implicitly, boundaries (walls, a fill-gas reservoir, outflow, periodic) that emit
  sputtered atoms with a given energy, return neutralized ions, take up sticking metal and
  exchange heat with the wall, volume sources, loss frequencies (ionization), exchange with
  the plasma ions and cooling by unresolved walls, all plain inputs a plasma model can
  update between steps. A budget of what boundaries and sources bring in closes the
  conservation to rounding. The continuum limit: at 1 Pa the Knudsen number is of order 1.
- `to_dataset()` on every result and `stack` for scans: labelled xarray datasets.
- Examples (an argon global model, a 1D argon discharge, a neutralizer scan), plotted with
  plasma-plots, and a tutorial.
- Opt-in scope-profiler regions (`wellbalanced.<Class>.<method>`) on the solves, time
  steps and runs of every model, tagged `wellbalanced`: off unless an application enables
  the tag (`setup(enable_tags=["wellbalanced"])`, `SCOPE_PROFILER_ENABLE_TAGS` or
  `scope-profiler run --enable-tag`). scope-profiler is a new dependency.

### Removed

- The template's console entry point, example and tutorial.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
