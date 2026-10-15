# wellbalanced


<!-- README.md is generated from README.qmd: edit the .qmd and run `make readme`. -->

[![Tests](https://github.com/max-models/wellbalanced/actions/workflows/test_pytest.yml/badge.svg)](https://github.com/max-models/wellbalanced/actions/workflows/test_pytest.yml)
[![Static
analysis](https://github.com/max-models/wellbalanced/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/max-models/wellbalanced/actions/workflows/static_analysis.yml)
[![Docs](https://github.com/max-models/wellbalanced/actions/workflows/docs.yml/badge.svg)](https://max-models.github.io/wellbalanced/)
[![codecov](https://codecov.io/gh/max-models/wellbalanced/branch/main/graph/badge.svg)](https://codecov.io/gh/max-models/wellbalanced)
[![PyPI](https://img.shields.io/pypi/v/wellbalanced.png)](https://pypi.org/project/wellbalanced/)
[![Python](https://img.shields.io/pypi/pyversions/wellbalanced.png)](https://pypi.org/project/wellbalanced/)

Global (0D) and 1D fluid models of low-temperature plasmas.

- **Global model** (`GlobalModel`): the density of every species, the
  electron energy and optionally the gas temperature, balanced over the
  volume of a chamber. Reactions, ions leaving through the walls at the
  Bohm speed (with an edge-to-centre ratio) and through magnetic cusps
  (hybrid leak width), neutral wall reactions, gas feed and pumping, and
  volume sources such as a fast beam. Steady states in seconds, and
  scans that continue from one steady state to the next.
- **1D fluid model** (`Fluid1D`): drift-diffusion of every species with
  Scharfetter-Gummel fluxes, the electron energy equation and Poisson’s
  equation, fully implicit. Walls, prescribed densities or fluxes at
  either end, changeable between steps, as the fluid side of a coupling
  to a particle code.
- **2D axial-azimuthal drift fluid of a magnetron** (`DriftFluid2D`): magnetised electrons
  with the Hall tensor, inertial ions, neutral depletion and refill, the electron energy
  and a semi-implicit Poisson solve; the gradient-drift instability and spoke diagnostics
  (mode number, phase velocity against E x B, cathode modulation, potential hump).
- **The neutral gas of a magnetron** (`GasDynamics`,
  `wellbalanced.gasdynamics`): argon and the sputtered metal as one
  compressible gas in 1D or 2D (HLLC, species diffusion, heat conduction),
  with the sputter wind, the return of neutralized ions, ionization sinks,
  wall heat exchange and a refill reservoir as inputs a plasma model can
  update; the rarefaction that spokes depend on. The continuum limit, so
  Knudsen numbers near 1 need a kinetic check.
- **A beam-driven plasma neutralizer** (`wellbalanced.neutralizer`): a
  fast beam’s charge states on the gas and on the plasma it makes,
  coupled to a global model of the channel fed by the beam’s stripped
  and secondary electrons, iterated to a fixed point in seconds.
- **The data of the kinetic codes**: rates come from
  [plasmacoll](https://github.com/max-models/plasmacoll)’s cross
  sections averaged over a Maxwellian, the data its Monte Carlo
  collisions use, so a fluid and a kinetic model of the same plasma
  differ in their physics, not their data.

The name: a *well-balanced* scheme keeps the steady states of its
equations exactly, and steady balances are what this code is for.

Documentation: <https://max-models.github.io/wellbalanced/>

## Example

``` python
from plasmacoll import CrossSection
from plasmacoll.constants import ELECTRON_MASS

from wellbalanced import ELECTRON, Chamber, GlobalModel, Reaction, Species

argon, argon_ion = Species.atomic("Ar", 39.948), Species.atomic("Ar+", 39.948, 1)
ionization = Reaction.from_cross_section(
    ("e", "Ar"),
    ("e", "e", "Ar+"),
    CrossSection.constant(2.0e-20, threshold=15.76),
    ELECTRON_MASS,
    electron_energy_loss=15.76,
)
model = GlobalModel(
    species=[ELECTRON, argon, argon_ion],
    reactions=[ionization],
    chamber=Chamber(volume=1.0e-3, wall_area=0.06, h_factor=0.5),
    power=100.0,
    fixed={"Ar": 1.0e20},
)
state = model.steady_state()
print(state.electron_temperature, state.electron_density, state.power)
```

## Install

With [uv](https://docs.astral.sh/uv/):

``` bash
make install    # uv sync --extra dev, plus the pre-commit hooks
```

Or with pip, in a Python environment (3.10 or newer):

``` bash
pip install -e .
```

The `test`, `docs` and `dev` extras install the test runner, the
documentation tooling and the linters:

``` bash
pip install -e ".[dev]"
```

## Development

Formatting and linting use [ruff](https://docs.astral.sh/ruff/), type
checking [pyright](https://microsoft.github.io/pyright/), run by
[pre-commit](https://pre-commit.com/) and in CI:

``` bash
make lint     # ruff check, ruff format --check, pyright
make test     # pytest with coverage
```

Commit messages follow [Conventional
Commits](https://www.conventionalcommits.org/); see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Build docs

The documentation in `docs/` is an Astro + Starlight site: hand-written
pages, the notebooks in `tutorials/` executed and published as pages,
complete Python scripts in `examples/` published with their plots, and
the API reference generated from the docstrings with
[starlight-pydocs](https://ewels.github.io/starlight-pydocs/). It needs
Node 22 or newer.

``` bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-examples    # run examples/*.py and publish their plots
make docs-dev         # live preview at http://localhost:4321/wellbalanced/
make docs-build       # the static site in docs/dist
```

## Build the README

`README.md` is rendered from `README.qmd` with
[Quarto](https://quarto.org/):

``` bash
make readme
```

## Releases

Before merging a release to `main`, update the version in
`pyproject.toml`, `src/wellbalanced/__init__.py` and `CITATION.cff`
(including its release date), and add the release notes to
`CHANGELOG.md`. The push to `main` creates a GitHub release with a
`vX.Y.Z` tag and publishes the package to PyPI with trusted publishing
(OIDC). The one-time PyPI and GitHub configuration is described in the
[publishing
tutorial](https://max-models.github.io/wellbalanced/development/publishing/).
