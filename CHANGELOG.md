# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- The neutralizer's gas along the channel (step 3 of the neutralizer plan): `GasProfile`, a
  free-molecular density profile for a feed (it scales with the inflow and `1 / v_mean`),
  from any arrays (e.g. picnic's `NeutralParticleTransport` with diffuse walls) or
  `GasProfile.knudsen` (Knudsen diffusion from the feed to both ends, Marshak's
  `Gamma = n v_mean / 2` at the exits). `Neutralizer(gas_profile=...)` keeps its shape and
  balances its mean: pumping speed from the profile, the beam across its line density, the
  sources on the gas with the fractions weighted with it. In the illustrative channel the
  profile holds 3.5 times the gas of the end conductances for the same feed;
  `NeutralParticleTransport` in 3D has the same peak as the Knudsen model and an 8% lower mean.
  The neutral fraction depends on the line density alone (the same to 1e-3 for a flat and a
  peaked profile of equal line density).
- `Neutralizer.local_plasma`: the plasma of every axial slice in its own gas and beam (the
  limit of no axial transport), with densities, `T_e` and plasma potential along the channel;
  `NeutralizerResult.gas_profile`, `.line_density` and `.plasma_potential`;
  `GlobalModel.plasma_potential` (presheath plus sheath, lowered by net negative sources).
- Example `neutralizer_gas_profile.py`: the neutral fraction, ionization degree and plasma
  potential against the gas flow for the uniform gas, the Knudsen profile and the profile of
  `NeutralParticleTransport` (`examples/data/neutralizer_gas_npt.csv`); against the beam
  current; and along the channel.
- The 0D/1D consistency test of the neutralizer plasma: across the channel, with the gas
  uniform and the beam filling the channel, a `Fluid1D` (momentum ions, electron energy,
  resolved sheaths, the beam's `volume_sources`) averaged over the width against a
  `GlobalModel` of the same slab with the same reactions and beam sources, its `h` taken
  from the 1D wall flux (0.61): the electron temperature within 0.1% (test: 1%), the
  densities within 3% (test: 5%), the remainder the presheath drop (about 0.7 T_e where the
  global model counts T_e / 2).
- `Fluid2D` (`wellbalanced.spatial.fluid`, step R3.3 of the RF-driver plan): the driver
  plasma as a multi-species fluid on the cells of the field's grid, in r–z or in a cut.
  Ion continuity and momentum (inertia, born at rest, drag on the gas from the mobility
  or from a constant mean free path), drift-diffusion electrons with their energy (rates
  from plasmacoll, `nu` from the momentum-transfer reactions), a quasi-neutral potential
  with sheath walls (Bohm ions, Boltzmann-reflected electrons) or Fluid1D's Poisson
  closure; heated by the field's `1/2 Re(J E*)` and pushed by its RF Lorentz force. Finite
  volumes on the faces, implicit, Newton with a coloured finite-difference Jacobian; on a
  `CellGrid` or a single-level `PatchGrid` (refined patches wait for patchwork's
  refluxing). Checked: particles and energy over a step to round-off, a slab against
  Fluid1D to 1e-10, ambipolar diffusion in a slab and a cylinder against their closed
  forms (first order), the collisionless slab with uniform ionization (`n_s = n_0 / 2`),
  patches against the tensor grid, a driver's steady state, and the coupled column
  against the global model (with the fluid's edge-to-mean ratio: `n_e` within 6 %, the
  coil current within 3 %). Placeholder rates and ion mobilities.
- `fluid_plasma` and `FluidPlasma`: the field–fluid fixed point at a given absorbed power,
  the plasma model of `DriverCoupling` (with relaxation and a ramp of the force).
- `InductiveSolution.lorentz_force` (the RF-averaged `-(e/2) Re(u x B*)` per electron) and
  `.electron_velocity_amplitude`; `CellGrid.transport_faces` and
  `PatchGrid.transport_faces` (`TransportFaces`), the faces for finite-volume transport;
  `scharfetter_gummel` shared by Fluid1D and Fluid2D.
- Example `rf_driver_fluid2d.py`: the BUG driver in r–z as a fluid in its field at 25 kW
  (106 A, eta 88 %), against the global model in the same field (124 A, 84 %) and with the
  fluid's edge ratio (`n_e` within 4 %), and the size of the RF Lorentz force (about 17
  kV/m against an ambipolar 0.7 kV/m); with the full force the fixed point does not
  converge yet.
- `InductiveField3D` (step R4 of the RF-driver plan): the time-harmonic
  magnetoquasistatic field in 3D, `curl (nu curl A) + i omega sigma A = J_coil`, with `A`
  on the edges of a uniform Cartesian grid (`EdgeGrid`, finite integration: the curl and
  gradient as incidence matrices, `C G = 0` exactly). Coils as `J = curl T`
  (`Coil3D`, divergence-free on the grid by construction: solenoids, rectangular loops);
  conductors staircased from simgeometry shapes and Booleans (`Conductor3D`, an edge
  conducts if its midpoint is inside, or a sub-cell series/parallel average with
  `samples`), and a background conductivity (a cold plasma's); Dirichlet (`A x n = 0`),
  natural (`H x n = 0`, symmetry planes) and periodic boundaries; the gauge a small real
  regularisation (`O(eps)` in the curl, no loss); one SuperLU per frequency in a geometric
  nested-dissection order. The loss per conductor, the coil's impedance and `B`. Checked
  against a long solenoid in a box (exact for the staircased cross-section), the
  z-invariant `InductiveField2D` (to `1e-7`), symmetry planes, the gauge's `eps` and the
  Bessel solution of a tube in a solenoid (first-order convergence of the staircase).
- `benchmarks/`: the R4 evidence (`field3d_r4.py`), a finite solenoid against Nagaoka, a
  long loop against the z-invariant field and a slotted cylinder against NGSolve
  (`slotted_cylinder_ngsolve.py`, run in a separate environment; NGSolve is a benchmark,
  not a dependency); the docs page "3D field discretisation" with the recommendation.

- The ion momentum equation in `Fluid1D` (`FluidSpecies(..., momentum=True)`): the velocity
  on the faces between cells as an unknown, the electric and pressure force as the
  Scharfetter-Gummel flux times the friction `e / (M mu)` (so without inertia it is the
  drift-diffusion scheme), upwind `Gamma^2 / n`, losses taking their momentum, and a wall
  that takes the outflow of the momentum balance over its half cell. Checked: drift-diffusion
  recovered with strong friction (2e-4 in the bulk), and a collisionless presheath against
  its closed form, the ions reaching the Bohm speed at the sheath edge with half the central
  density and `T_e ln 2` below it: the density profile within 4% at a Debye length of
  7e-3 of the size and 1.3% at 2e-3, the potential drop within 3% and 2%.
  `FluidState.velocity` gives the mean velocity on the faces.
- `Boundary.wall(thermal_share=...)`: the share of `v_th n` a wall takes, 1/2 (Hagelaar et
  al. 2000, the default) or 1/4 (a full Maxwellian, as kinetic sheath theory and the global
  model assume for the Boltzmann electrons of a resolved sheath). The floating sheath of the
  presheath test is `T_e ln(s v_e / c_s)` within 5% for both: 4.8 T_e with 1/4 against 4.67
  from kinetic theory for Ar+, and `T_e ln 2` more with 1/2.
- Neutralizer beam sources as functions of the local beam: `Neutralizer.beam_sources(flux,
  fractions, energy)` gives the stripping, capture and gas-ionization sources where the beam
  has that particle flux, those charge-state fractions and that energy (numbers, or arrays
  across the channel). `BeamCrossSections` holds the cross sections as functions of the beam
  energy (e.g. plasmacoll `CrossSection`s), with the gas ionization optionally a
  `SecondaryElectrons` such as `plasmacoll.RuddIonization`, whose mean secondary energy then
  feeds the electrons (`BeamData.secondary_energy`, per beam state).
- `Fluid1D(volume_sources=...)`: the global model's `VolumeSource`s in 1D, with rates per
  unit density of fluid or fixed species, numbers or arrays on the cells; their electron
  energy appears as `sources` in `energy_balance`.
- `PatchGrid`: the leaf cells of a patchwork hierarchy as the grid of `wellbalanced.spatial`,
  with `CellGrid`'s interface (fields over the leaf cells, shape `(n,)`): `refined` (boxes or
  a function per level), `from_edges` (about as fine as a tensor grid, for comparisons),
  `refined_where` (one level finer where a field is large, e.g. the skin layer).
  `InductiveField2D(grid=...)`, the cold and viscous solvers, grounded and floating conductors,
  the curl, `DriverCoupling` and `PlasmaRegion` run on it; every closed-form check of the
  field models runs on both grids. The r–z BUG example on patches: a third of the cells, the
  same coil currents and efficiencies, four times faster. patchwork comes with the `spatial`
  extra, as `patchwork-amr` (its name on PyPI); the patch-grid tests skip without it.
- `CellGrid.dims`, `.locate` and `.wall_cells`; `PlasmaRegion.wall_faces` goes through the
  grid (periodic seams are no walls now).

- `wellbalanced.spatial`: the spatially resolved models in one subpackage, on one cell
  grid (`CellGrid`: edges, centres, cell volumes and face areas, the x–y or r–z metric,
  shape masks and the curl-curl operator; the only place the geometry enters).
- `InductiveField2D(geometry="cylindrical")`: the axisymmetric `A_phi(r, z)`, coil loops,
  grounded conductors, total powers without a depth; checked against the field of a current
  loop (complete elliptic integrals), a long solenoid and a plasma column in it (the
  Bessel solution of `solenoid_impedance`, to 2e-4).
- Viscous electrons (`electron_temperature=`, or `electron_viscosity=`): the electron
  momentum balance with the local collisional viscosity `(2/3) n_e e T_e / nu` of Zielke et
  al. (PSST 31, 035019, 2022), solved with the field as one sparse system, no slip at the
  edge of the plasma; the absorbed power splits into friction and viscous parts. Checked
  against the closed-form skin layer of a slab (a quartic dispersion relation) and, without
  viscosity, against the cold plasma to round-off. The RF Lorentz force waits for `Fluid2D`.
- `PlasmaRegion`: the plasma as a union of shapes on the grid (the driver and expansion T).
- `DriverCoupling` and `global_model_plasma`: the field with a power-dependent plasma, at a
  set absorbed power or at a set generator power (a fixed point, as the integral controller
  of Zielke et al.); the `RFDriver`'s global model on a fixed spatial shape, which in an
  infinite solenoid reproduces `RFDriver(coupling="solenoid")` to 0.2 %.
- Example `rf_driver_rz.py`: the BUG driver in r–z with cold and viscous electrons and the
  global model's plasma at 60 kW from the generator.

### Changed

- `GlobalModel`'s floating walls when the sources add net negative charge (a beam's
  stripped electrons): the sheath drop falls by `T_e ln(Gamma_e / Gamma_+)` and the extra
  electrons cross the presheath and sheath too, so the electrons lose
  `Gamma_e (5/2 T_e + V)` at the walls (they lost only `2 T_e` each beyond the ions'). Found
  by the 0D/1D test; the illustrative neutralizer's electron density falls by 3%, its
  neutral fraction moves by 1e-4.
- `Fluid1D`'s Newton solver: a central-difference Jacobian, the equations' scales held fixed
  during each solve (so a cell's residual depends on its neighbours only and the Jacobian by
  colours is exact), and the Bernoulli function's series to `x^4` near 0 (`1 - x/2` erred
  by `x^2/12`, noise in the residual of a fine mesh). Newton converges quadratically again
  with an energy equation and ion momentum; the `Fluid1D` tests run in a fifth of the time.
- `InductiveField2D`, `InductiveSolution`, `Winding`, `Conductor` moved from
  `wellbalanced.inductive` to `wellbalanced.spatial` (still exported by `wellbalanced`);
  powers are per volume element (`cell_volumes`), `depth` is optional (an r–z solution has
  none).
- simgeometry is an optional dependency (`wellbalanced[spatial]`, and the test and docs
  extras): the spatial models take any shape with `contains_points` and do not import it.

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
