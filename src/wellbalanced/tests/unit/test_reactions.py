"""Reactions from plasmacoll processes; the cross sections are synthetic."""

from __future__ import annotations

import pytest
from plasmacoll import CollisionProcess, CrossSection, RateTable, rate_coefficient
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS

from wellbalanced import Reaction

M_AR = 39.948 * ATOMIC_MASS


def test_an_ionization_process_gives_its_rate_and_threshold_loss() -> None:
    step = CrossSection.constant(2e-20, threshold=15.76)
    process = CollisionProcess(
        "ionization", "Ar", step, products={"electron": "e", "ion": "Ar+"}
    )
    reaction = Reaction.from_process(process, ("e", "Ar"), ("e", "e", "Ar+"))
    assert reaction.stoichiometry == {"e": 1, "Ar": -1, "Ar+": 1}
    assert reaction.electron_loss(3.0, 0.03) == 15.76
    assert reaction.coefficient(3.0, 0.03) == pytest.approx(
        rate_coefficient(step, 3.0, ELECTRON_MASS), rel=2e-3
    )


def test_an_elastic_process_transfers_three_m_over_m_of_the_temperature_difference() -> (
    None
):
    process = CollisionProcess("elastic", "Ar", CrossSection.constant(1e-19))
    with pytest.raises(ValueError, match="target_mass"):
        Reaction.from_process(process, ("e", "Ar"), ("e", "Ar"))
    reaction = Reaction.from_process(
        process, ("e", "Ar"), ("e", "Ar"), target_mass=M_AR
    )
    loss = reaction.electron_loss(4.0, 0.025)
    assert loss == pytest.approx(3 * ELECTRON_MASS / M_AR * (4.0 - 0.025))
    assert reaction.gas_gain(4.0, 0.025) == loss
    assert reaction.stoichiometry == {}


def test_rates_from_numbers_tables_and_functions_at_the_right_temperature() -> None:
    table = RateTable.arrhenius(1e-14, 0.0, 10.0)
    by_table = Reaction(("e", "X"), ("e", "e", "X+"), table)
    assert by_table.coefficient(5.0, 0.1) == pytest.approx(table(5.0))
    heavy = Reaction(("X+", "Y"), ("X", "Y+"), lambda t: 1e-15 * t, temperature="gas")
    assert heavy.coefficient(5.0, 0.1) == pytest.approx(1e-16)
    assert Reaction(("A", "A"), ("A2",), 1e-17).stoichiometry == {"A": -2, "A2": 1}
    with pytest.raises(ValueError, match="temperature"):
        Reaction(("e",), ("e",), 1.0, temperature="ion")
