"""Hydrogen reaction sets from OpenCR's collisional-radiative models.

OpenCR (github.com/max-models/opencr, the ``cr`` extra) derives effective rate
coefficients from the elementary processes with resolved excited states:
the ionization and recombination of atomic hydrogen through its excited
levels (``hydrogen/atomic/basic``), and the ionization and dissociation of H2
through its vibrational levels (``hydrogen/molecular/basic``), each with the
electron energy loss per event that the radiation of the excited states adds.
:func:`opencr_hydrogen_reactions` turns them into :class:`~wellbalanced.reactions.Reaction`
objects in place of ground-state cross sections and placeholders.

The effective rates depend on the electron density as well as the
temperature, a :class:`~wellbalanced.reactions.Reaction` on the temperature
only: they are taken at ``electron_density``, which should be near the
density of the solution (the dependence is weak below about 1e18 m^-3, where
the excited states are coronal).

The electron energy loss of each driver (H, H+, H2) is a separate pair of
reactions that change no density, ``rate = max(+-L, 0)`` at ``+-1 eV`` per
event, so the energy balance is the CR model's whatever the sign of ``L``
(three-body recombination heats the electrons).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS
from plasmacoll.hydrogen_elastic import ELASTIC_FITS

from wellbalanced.reactions import Reaction

__all__ = [
    "CR_TEMPERATURES",
    "deuterium_neutralizer",
    "hydrogen_driver",
    "opencr_hydrogen_reactions",
]

#: The electron temperatures (eV) of the tables.
CR_TEMPERATURES = np.geomspace(0.3, 100.0, 40)


def _opencr() -> Any:
    try:
        import opencr
    except ImportError as error:  # pragma: no cover - the cr extra is installed in CI
        raise ImportError("OpenCR is needed: pip install 'wellbalanced[cr]'") from error
    return opencr


def _interpolated(temperatures: np.ndarray, values: np.ndarray) -> Callable:
    """``k(T)``, linear in ``ln k`` against ``ln T`` (end values outside)."""
    log_t = np.log(temperatures)
    positive = np.maximum(values, 1e-300)
    log_k = np.log(positive)

    def rate(t):
        x = np.clip(np.log(np.asarray(t, dtype=float)), log_t[0], log_t[-1])
        k = np.exp(np.interp(x, log_t, log_k))
        return np.where(k <= 1e-299, 0.0, k)

    return rate


def _energy_pair(species: str, loss: np.ndarray, name: str) -> list[Reaction]:
    """Two reactions that carry an energy loss coefficient ``L(T)`` (eV m^3/s) of any sign."""
    pairs = []
    for sign, label in ((1.0, "loss"), (-1.0, "gain")):
        part = np.maximum(sign * loss, 0.0)
        if np.any(part > 0.0):
            pairs.append(
                Reaction(
                    ("e", species),
                    ("e", species),
                    _interpolated(CR_TEMPERATURES, part),
                    electron_energy_loss=sign,
                    name=f"{name} ({label})",
                )
            )
    return pairs


def _table(model: Any, densities: dict[str, float], electron_density: float, outputs):
    opencr = _opencr()
    from opencr.tables import effective_rate_table

    conditions = opencr.Conditions(1.0, electron_density, densities=densities)
    table = effective_rate_table(
        model, conditions, outputs, CR_TEMPERATURES, [electron_density]
    )
    return {name: table(name, CR_TEMPERATURES, electron_density) for name in outputs}


def _isotope(name: str, isotope: str) -> str:
    return name if name == "e" else name.replace("H", isotope)


def opencr_hydrogen_reactions(
    electron_density: float = 1.0e17,
    n_max: int = 12,
    dissociation_heating: float = 2.0,
    isotope: str = "H",
) -> list[Reaction]:
    """Return the reactions of a hydrogen plasma (H2, H, H+, H2+, H3+) from OpenCR.

    Electron collisions:

    * ``e + H -> 2e + H+`` and ``e + H+ -> H`` at OpenCR's collisional-radiative
      ionization and recombination coefficients (``hydrogen/atomic/basic``
      with ``n_max`` levels);
    * ``e + H2 -> 2e + H2+``, ``e + H2 -> e + 2H`` and ``e + H2 -> 2e + H+ + H``
      at the effective rates of ``hydrogen/molecular/basic`` (through the
      vibrational levels and the electronic states, a and b to dissociation);
    * the electron energy loss per H, H+ and H2 of the same models;
    * elastic momentum transfer with H and H2 (Park et al. 2022, through plasmacoll);

    and the ion chemistry of ``hydrogen/molecular/ions``: ``H2+ + H2 -> H3+ + H``
    and the dissociative recombination of H2+ and H3+.

    Args:
        electron_density: The ``n_e`` (m^-3) the effective rates are taken at.
        n_max: The levels of the atomic model.
        dissociation_heating: The energy (eV) the gas gains per dissociation
            (the Franck-Condon fragments of the b state, a few eV).
        isotope: ``"H"`` or ``"D"``; for deuterium the species are renamed
            (D, D+, D2, D2+, D3+) and the elastic masses doubled, the rate
            coefficients are hydrogen's (the isotope shifts of the vibrational
            levels are neglected).
    """
    if isotope not in ("H", "D"):
        raise ValueError("isotope must be 'H' or 'D'")
    opencr = _opencr()
    atomic = opencr.load_model("hydrogen/atomic/basic", n_max=n_max)
    h = _table(
        atomic,
        {"H(n=1)": 1.0, "H+": 1.0},
        electron_density,
        [
            "effective_rate:H(n=1)>H+",
            "effective_rate:H+>H(n=1)",
            "electron_cooling:H(n=1)",
            "electron_cooling:H+",
        ],
    )
    molecular = opencr.load_model("hydrogen/molecular/basic")
    h2 = _table(
        molecular,
        {"H2(X,v=0)": 1.0, "H(n=1)": 1.0, "H+": 1.0, "H2+": 1.0},
        electron_density,
        [
            "effective_rate:H2(X,v=0)>H2+",
            "effective_rate:H2(X,v=0)>H(n=1)",
            "effective_rate:H2(X,v=0)>H+",
            "electron_cooling:H2(X,v=0)",
        ],
    )
    ions = opencr.load_model("hydrogen/molecular/ions")
    chemistry = np.array(
        [
            [rates[name] for name in ("conv:0", "dr:H2+", "dr:H3+:3H", "dr:H3+:H2")]
            for rates in (
                ions.rate_coefficients(opencr.Conditions(float(t), electron_density))
                for t in CR_TEMPERATURES
            )
        ]
    )
    dissociative = h2["effective_rate:H2(X,v=0)>H+"]
    dissociation = 0.5 * (h2["effective_rate:H2(X,v=0)>H(n=1)"] - dissociative)

    def rate(values: np.ndarray) -> Callable:
        return _interpolated(CR_TEMPERATURES, values)

    elastic = {
        target: ELASTIC_FITS[target].cross_section("momentum_transfer")
        for target in ("H", "H2")
    }
    reactions = [
        Reaction(("e", "H"), ("e", "e", "H+"), rate(h["effective_rate:H(n=1)>H+"]),
                 name="CR ionization of H"),
        Reaction(("e", "H+"), ("H",), rate(h["effective_rate:H+>H(n=1)"]),
                 name="CR recombination of H+"),
        *_energy_pair("H", h["electron_cooling:H(n=1)"], "CR energy loss on H"),
        *_energy_pair("H+", h["electron_cooling:H+"], "CR energy loss on H+"),
        Reaction(("e", "H2"), ("e", "e", "H2+"), rate(h2["effective_rate:H2(X,v=0)>H2+"]),
                 name="ionization of H2"),
        Reaction(("e", "H2"), ("e", "H", "H"), rate(dissociation),
                 gas_heating=dissociation_heating, name="dissociation of H2"),
        Reaction(("e", "H2"), ("e", "e", "H+", "H"), rate(dissociative),
                 gas_heating=dissociation_heating, name="dissociative ionization of H2"),
        *_energy_pair("H2", h2["electron_cooling:H2(X,v=0)"], "CR energy loss on H2"),
        Reaction.from_cross_section(("e", "H2"), ("e", "H2"), elastic["H2"], ELECTRON_MASS,
                                    elastic_mass=2.016 * ATOMIC_MASS, name="elastic e-H2"),
        Reaction.from_cross_section(("e", "H"), ("e", "H"), elastic["H"], ELECTRON_MASS,
                                    elastic_mass=1.008 * ATOMIC_MASS, name="elastic e-H"),
        Reaction(("H2+", "H2"), ("H3+", "H"), float(chemistry[0, 0]), temperature="gas",
                 name="H3+ formation"),
        Reaction(("e", "H2+"), ("H", "H"), rate(chemistry[:, 1]), gas_heating=5.0,
                 name="dissociative recombination of H2+"),
        Reaction(("e", "H3+"), ("H", "H", "H"), rate(chemistry[:, 2]), gas_heating=4.0,
                 name="dissociative recombination of H3+ to 3H"),
        Reaction(("e", "H3+"), ("H2", "H"), rate(chemistry[:, 3]), gas_heating=4.0,
                 name="dissociative recombination of H3+"),
    ]  # fmt: skip
    if isotope == "H":
        return reactions
    mass = 2.0
    renamed = []
    for r in reactions:
        renamed.append(
            Reaction(
                tuple(_isotope(n, isotope) for n in r.reactants),
                tuple(_isotope(n, isotope) for n in r.products),
                r.rate,
                temperature=r.temperature,
                electron_energy_loss=r.electron_energy_loss,
                gas_heating=r.gas_heating,
                elastic_mass=None if r.elastic_mass is None else mass * r.elastic_mass,
                name=r.name.replace("H", isotope),
            )
        )
    return renamed


def hydrogen_driver(electron_density: float = 1.0e17, **overrides: Any):
    """The BUG driver of :func:`~wellbalanced.driver.illustrative_hydrogen_driver` with OpenCR's rates.

    The geometry, gas and options of the illustrative driver, with the
    reactions of :func:`opencr_hydrogen_reactions` at ``electron_density``
    in place of its placeholder cross sections. ``overrides`` replace fields
    of the :class:`~wellbalanced.driver.RFDriver`.
    """
    from wellbalanced.driver import illustrative_hydrogen_driver

    reactions = opencr_hydrogen_reactions(electron_density)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return illustrative_hydrogen_driver(reactions=reactions, **overrides)


def deuterium_neutralizer(electron_density: float = 1.0e17, **overrides: Any):
    """The neutralizer channel of :func:`~wellbalanced.neutralizer.illustrative_deuterium_neutralizer` with OpenCR's plasma rates.

    The electron collisions with D2, D and their ions are those of
    :func:`opencr_hydrogen_reactions` (``isotope="D"``) at ``electron_density``,
    without D3+ (not a species of the channel). The beam's charge-changing
    cross sections are still the illustrative ones, so the function warns as
    the illustrative channel does.
    """
    from wellbalanced.neutralizer import illustrative_deuterium_neutralizer

    species = {"e", "D2", "D", "D2+", "D+"}
    reactions = [
        r
        for r in opencr_hydrogen_reactions(electron_density, isotope="D")
        if set(r.reactants) | set(r.products) <= species
    ]
    return illustrative_deuterium_neutralizer(reactions=reactions, **overrides)
