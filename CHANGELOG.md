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
