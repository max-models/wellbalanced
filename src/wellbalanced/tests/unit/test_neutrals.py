"""The H and H2 neutral fluid: walls, viscosity, diffusion, conservation, 0D, the driver.

The collision rate coefficients, wall probabilities and transparencies are placeholders
of the right size, and the driver's reactions are those of
:func:`~wellbalanced.driver.illustrative_hydrogen_driver` (synthetic); the 0D limit uses
the Janev, Reiter and Samm fits of plasmacoll for the electron collisions with H2.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from plasmacoll import CrossSection
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE
from plasmacoll.janev import JANEV_FITS
from scipy.integrate import quad, solve_ivp

from wellbalanced import ELECTRON, Reaction, Species, WallReaction
from wellbalanced.driver import illustrative_hydrogen_driver
from wellbalanced.fluid1d import FluidSpecies
from wellbalanced.spatial import (
    CellGrid,
    Collision,
    Fluid2D,
    Inlet,
    NeutralFluid2D,
    NeutralSpecies,
    Opening,
    couple_neutrals,
    hard_sphere,
)
from wellbalanced.spatial.neutrals import ROOM_TEMPERATURE, half_maxwellian_fluxes

E = ELEMENTARY_CHARGE
H2 = Species("H2", 2.016 * ATOMIC_MASS)
H = Species("H", 1.008 * ATOMIC_MASS)
H_ION = Species.atomic("H+", 1.008, 1)
H2_ION = Species.atomic("H2+", 2.016, 1)
MOLECULES = NeutralSpecies(H2, 3.0)
ATOMS = NeutralSpecies(H, 0.0)
TW = ROOM_TEMPERATURE

DISSOCIATION = Reaction.from_cross_section(
    ("e", "H2"),
    ("e", "H", "H"),
    JANEV_FITS["e_h2_triplet_b"].cross_section(),
    ELECTRON_MASS,
    electron_energy_loss=10.0,
    gas_heating=3.0,  # Franck-Condon energy of the pair: a placeholder
    name="dissociation",
)
H2_IONIZATION = Reaction.from_cross_section(
    ("e", "H2"),
    ("e", "e", "H2+"),
    JANEV_FITS["e_h2_ionization"].cross_section(),
    ELECTRON_MASS,
    electron_energy_loss=15.42,
    name="ionization of H2",
)
H_IONIZATION = Reaction.from_cross_section(
    ("e", "H"),
    ("e", "e", "H+"),
    CrossSection.constant(0.7e-20, threshold=13.6),
    ELECTRON_MASS,
    electron_energy_loss=13.6,
    name="ionization of H",
)
ELASTIC = Reaction.from_cross_section(
    ("e", "H2"),
    ("e", "H2"),
    CrossSection.constant(1e-19),
    ELECTRON_MASS,
    elastic_mass=H2.mass,
    name="elastic e-H2",
)
PLASMA = [ELECTRON, H_ION, H2_ION]
K_H_H2 = hard_sphere(2.0e-19, H.mass, H2.mass)
K_CX = 3.0e-15  # H+ + H, twice the charge-exchange rate coefficient: a placeholder
COLLISIONS = [
    Collision("H2", "H2", hard_sphere(2.6e-19, H2.mass, H2.mass)),
    Collision("H", "H2", K_H_H2),
    Collision("H", "H", hard_sphere(1.5e-19, H.mass, H.mass)),
    Collision("H", "H+", K_CX),
]


def test_the_wall_fluxes_are_those_of_a_drifting_half_maxwellian() -> None:
    """Eqs. (26) and (29) of Zielke et al. against a quadrature of the distribution."""
    mass, t, n, internal = H2.mass, 0.05, 1e19, 3.0
    vp = np.sqrt(2 * E * t / mass)
    for s in (-2.0, -0.3, 0.0, 0.7, 2.5):
        u = s * vp

        def g(v: float, u: float = u) -> float:
            return np.exp(-((v - u) ** 2) / vp**2) / (np.sqrt(np.pi) * vp)

        flux = n * quad(lambda v: v * g(v), 0, u + 12 * vp)[0]
        normal = 0.5 * mass * n * quad(lambda v: v**3 * g(v), 0, u + 12 * vp)[0] / E
        energy = normal + flux * (1 + internal / 2) * t
        got = half_maxwellian_fluxes(
            np.array(n), np.array(t), np.array(u), mass, internal
        )
        assert got[0] == pytest.approx(flux, rel=1e-10)
        assert got[1] == pytest.approx(energy, rel=1e-10)
    # at rest: n v / 4 and (2 + z / 2) T per particle
    rest = half_maxwellian_fluxes(np.array(n), np.array(t), np.array(0.0), mass, 3.0)
    mean_speed = np.sqrt(8 * E * t / (np.pi * mass))
    assert rest[0] == pytest.approx(n * mean_speed / 4, rel=1e-12)
    assert rest[1] == pytest.approx(3.5 * t * rest[0], rel=1e-12)


def test_a_shear_wave_decays_at_the_viscosity_of_the_collisions() -> None:
    """``u_y = U sin(k x)`` in a periodic box decays at ``mu k^2 / (m n)``.

    The viscosity is the paper's eq. (21) for the self-collisions,
    ``mu = 2/3 m n T / (m / 2 n K) = 4/3 T / K``; on the grid, ``k^2`` is the discrete
    Laplacian's eigenvalue and backward Euler takes ``1 / (1 + lambda dt)`` per step.
    """
    cells, length = 16, 0.2
    grid = CellGrid(
        np.linspace(0, length, cells + 1),
        [0.0, 0.01],
        boundaries=("periodic", "periodic"),
    )
    k_rate = 6.0e-16
    model = NeutralFluid2D(
        grid, [MOLECULES], collisions=[Collision("H2", "H2", k_rate)]
    )
    n, t, amplitude = 1e21, TW, 1.0
    k = 2 * np.pi / length
    shear = amplitude * np.sin(k * grid.xx)
    u = model.pack({"H2": n}, t, velocities={"H2": (0.0, shear)})
    dx = length / cells
    viscosity = 4.0 / 3.0 * E * t / k_rate
    rate = viscosity * (2 - 2 * np.cos(k * dx)) / dx**2 / (H2.mass * n)
    dt, steps = 0.2 / rate, 5
    for _ in range(steps):
        u, ok = model.step(u, dt)
        assert ok
    state = model.state(u)
    expected = shear / (1 + rate * dt) ** steps
    assert np.max(np.abs(state.velocities["H2"][1] - expected)) < 1e-8 * amplitude
    assert np.max(np.abs(state.velocities["H2"][0])) < 1e-8 * amplitude
    assert np.allclose(state.temperatures["H2"], t, rtol=1e-10)


def test_a_trace_gas_diffuses_at_the_rate_its_friction_gives() -> None:
    """Atoms injected at one wall and absorbed at the other, in a dense fixed H2.

    The atoms' momentum balance, ``grad(n T) = -mu n n_H2 K u``, is Fick's law with
    ``D = T / (mu n_H2 K)``: the flux through the slab is ``-D dn/dx`` in its interior,
    up to the viscous correction ``(lambda / L)^2``.
    """
    length, cells, n_bg, k = 0.2, 40, 1e21, 6.0e-16
    grid = CellGrid(
        np.linspace(0, length, cells + 1),
        [0.0, 0.01],
        boundaries=("dirichlet", "periodic"),
    )
    rate = 1e16  # atoms per second through the 1 cm wide slab, per metre
    model = NeutralFluid2D(
        grid,
        [ATOMS],
        plasma_species=[H2],
        fixed={"H2": n_bg},
        collisions=[Collision("H", "H2", k), Collision("H", "H", 1e-16)],
        openings=[Opening(lambda x, y: x > length - 1e-9, 1.0)],
        inlets=[Inlet(lambda x, y: x < 1e-9, "H", rate)],
    )
    state = model.steady_state(model.pack({"H": 1e17}, TW))
    assert state.converged
    n = state.densities["H"][:, 0]
    reduced = H.mass * H2.mass / (H.mass + H2.mass)
    diffusion = E * TW / (reduced * n_bg * k)
    flux = rate / 0.01
    gradient = np.gradient(n, grid.x)[cells // 4 : 3 * cells // 4]
    assert -diffusion * gradient == pytest.approx(
        np.full(gradient.size, flux), rel=1e-3
    )
    assert abs(model.balance(state)["H"]) < 1e-9 * rate
    budget = model.wall_budget(state)["H"]
    assert budget["openings"] == pytest.approx(rate, rel=1e-9)


@pytest.fixture
def column() -> CellGrid:
    return CellGrid(
        np.linspace(0, 0.12, 6), np.linspace(0, 0.17, 8), geometry="cylindrical"
    )


def nuclei(model: NeutralFluid2D, u: np.ndarray) -> float:
    state = model.state(u)
    return state.total(state.densities["H"]) + 2 * state.total(state.densities["H2"])


@pytest.mark.parametrize("ionization", [False, True])
def test_reflecting_walls_conserve_the_nuclei(column, ionization) -> None:
    """Without wall losses the nuclei change only by the ionized ones, to round-off.

    The plasma is a non-uniform column with ions drifting outwards: dissociation,
    the electrons' elastic heating, charge exchange and transport all move nuclei
    between the species and the cells, none leaves.
    """
    reactions = [DISSOCIATION, ELASTIC] + (
        [H2_IONIZATION, H_IONIZATION] if ionization else []
    )
    model = NeutralFluid2D(
        column, [MOLECULES, ATOMS], reactions, plasma_species=PLASMA,
        collisions=COLLISIONS, accommodation=0.5,
    )  # fmt: skip
    shape = np.cos(0.5 * np.pi * column.xx / 0.12) * np.sin(np.pi * column.yy / 0.17)
    ne = 1e17 + 4e17 * shape
    drift = 2e3 * column.xx / 0.12
    model.set_plasma(
        {"e": ne, "H+": 0.5 * ne, "H2+": 0.5 * ne},
        2.0 + 3.0 * shape,
        ion_velocities={"H+": (drift, 0.0)},
        ion_temperature=0.1,
    )
    u = model.pack({"H2": 7e19 * (1 - 0.3 * shape), "H": 5e18}, TW)
    dt = 2e-5
    for _ in range(3):
        before = nuclei(model, u)
        u, ok = model.step(u, dt)
        assert ok
        state = model.state(u)
        ionized = 0.0
        if ionization:
            ionized = 2 * state.total(state.reaction_rates["ionization of H2"])
            ionized += state.total(state.reaction_rates["ionization of H"])
            assert ionized * dt > 1e-4 * before
        assert nuclei(model, u) - before == pytest.approx(
            -ionized * dt, abs=1e-9 * before
        )
        balance = model.balance(state)
        assert balance["H"] + 2 * balance["H2"] == pytest.approx(
            -ionized, abs=1e-9 * before / dt
        )
    # the gas moved and heated
    assert np.max(np.abs(state.velocities["H"][0])) > 1.0
    assert state.temperatures["H"].max() > 2 * TW


def zero_d_reference(model, ne, te, ti, start, duration):
    """The closed box with a uniform plasma as ODEs, solved by scipy."""
    kd, k2, k1, kel = (
        r.coefficient(te, TW)
        for r in (DISSOCIATION, H2_IONIZATION, H_IONIZATION, ELASTIC)
    )
    m1, m2, mp = H.mass, H2.mass, H_ION.mass
    c2, c1 = MOLECULES.heat_capacity, ATOMS.heat_capacity

    def rhs(_t, y):
        n2, n1, t2, t1 = y
        pair = (m2 * t1 + m1 * t2) / (m1 + m2)
        k = float(K_H_H2(np.array(pair)))
        exchange = 3 * (m1 * m2 / (m1 + m2)) * n1 * n2 * k / (m1 + m2)
        charge = 3 * (m1 * mp / (m1 + mp)) * n1 * ne * K_CX / (m1 + mp) * (ti - t1)
        dn2 = -(kd + k2) * ne * n2
        dn1 = 2 * kd * ne * n2 - k1 * ne * n1
        de2 = (
            -(kd + k2) * ne * n2 * c2 * t2
            + 3 * ELECTRON_MASS / m2 * (te - t2) * kel * ne * n2
            + exchange * (t1 - t2)
        )
        de1 = (
            2 * kd * ne * n2 * c1 * t2  # the atoms are born at the molecules' T
            + DISSOCIATION.gas_heating * kd * ne * n2
            - k1 * ne * n1 * c1 * t1
            + exchange * (t2 - t1)
            + charge
        )
        return [
            dn2,
            dn1,
            (de2 - c2 * t2 * dn2) / (c2 * n2),
            (de1 - c1 * t1 * dn1) / (c1 * n1),
        ]

    run = solve_ivp(rhs, (0, duration), start, method="Radau", rtol=1e-11, atol=1e-30)
    return run.y[:, -1]


def test_the_0d_limit_is_the_ode_solution() -> None:
    """A closed box (symmetry planes all round) with a uniform plasma: the ODEs.

    Densities and temperatures of both species against scipy's solution of the
    same balances: backward Euler's error is first order in the time step (1-2 %
    at 25 us over 1 ms), and Richardson's extrapolation of two steps is within 1e-3.
    """
    grid = CellGrid(
        np.linspace(0, 0.1, 3),
        np.linspace(0, 0.1, 3),
        boundaries=("neumann", "neumann"),
    )
    model = NeutralFluid2D(
        grid,
        [MOLECULES, ATOMS],
        [DISSOCIATION, H2_IONIZATION, H_IONIZATION, ELASTIC],
        plasma_species=PLASMA,
        collisions=COLLISIONS,
    )
    ne, te, ti = 3e17, 5.0, 0.3
    model.set_plasma({"e": ne, "H+": ne, "H2+": 0.0}, te, ion_temperature=ti)
    start = [7e19, 1e19, TW, 0.1]
    duration = 1e-3
    reference = zero_d_reference(model, ne, te, ti, start, duration)
    u0 = model.pack({"H2": start[0], "H": start[1]}, {"H2": start[2], "H": start[3]})
    errors = []
    for dt in (5e-5, 2.5e-5):
        state = model.state(model.advance(u0, duration, dt))
        got = np.array(
            [
                state.densities["H2"],
                state.densities["H"],
                state.temperatures["H2"],
                state.temperatures["H"],
            ]
        )
        assert np.ptp(got, axis=(1, 2)) == pytest.approx(0.0, abs=1e-9 * got.max())
        errors.append(got[:, 0, 0] / reference - 1)
    assert np.max(np.abs(errors[1])) < 0.02
    order = np.log2(np.abs(errors[0]) / np.abs(errors[1]))
    assert np.all((order > 0.8) & (order < 1.2))
    # Richardson's extrapolation removes the first-order error
    assert np.max(np.abs(2 * errors[1] - errors[0])) < 1e-3


# --- the driver -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def driver():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return illustrative_hydrogen_driver(coupling="solenoid", field_factor=1.0)


TRANSPARENCY = 0.3


@pytest.fixture(scope="module")
def driver_models(driver):
    """The BUG driver plasma at 25 kW (uniformly heated) and its gas, fed at the back.

    The gas comes in through the backplate near the axis at the rate that fills the
    driver to 0.3 Pa at the wall temperature without plasma, and leaves through the
    open end (the grid of the paper; here the expansion region) with a transparency
    of 0.3. Atoms recombine at the walls with a probability of 0.1; the ions come
    back as the gas of the driver's global model. All placeholders.
    """
    radius, length = driver.radius, driver.length
    grid = CellGrid(
        np.linspace(0, radius, 5), np.linspace(0, length, 7), geometry="cylindrical"
    )
    tg = driver.model_options["gas_temperature"]
    species = [FluidSpecies(ELECTRON, 1.0)]
    for s in driver.species:
        if s.charge and s.name != "e":
            thermal = np.sqrt(8 * E * tg / (np.pi * s.mass))
            species.append(
                FluidSpecies(s, E * driver.ion_mean_free_path / (s.mass * thermal))
            )
    volume = np.pi * radius**2 * length
    plasma = Fluid2D(
        grid, species, driver.reactions, fixed=driver.fixed, gas_temperature=tg,
        momentum_transfer=driver.momentum_transfer,
        ion_mean_free_path=driver.ion_mean_free_path, heating=25e3 / volume,
    )  # fmt: skip
    shape = np.cos(0.4 * np.pi * grid.xx / radius) * np.sin(
        np.pi * (0.9 * grid.yy / length + 0.05)
    )
    ions = {"H+": 0.25, "H2+": 0.5, "H3+": 0.25}
    start = plasma.steady_state(
        plasma.pack({n: f * 1e18 * shape for n, f in ions.items()}, 6.0, 30.0)
    )
    assert start.converged
    mass = {s.name: s.mass for s in driver.species}
    ion_collisions = [
        Collision(n, ion, 2e-15)
        for n in ("H", "H2")
        for ion in ("H+", "H2+", "H3+")
        if (n, ion) != ("H", "H+")
    ]
    fill = 0.3 / (E * TW)
    mean_speed = np.sqrt(8 * E * TW / (np.pi * mass["H2"]))
    feed = TRANSPARENCY * fill * mean_speed / 4 * np.pi * radius**2
    gas = NeutralFluid2D(
        grid,
        [MOLECULES, ATOMS],
        driver.reactions,
        plasma_species=[s for s in driver.species if s.charge],
        collisions=COLLISIONS + ion_collisions,
        wall_reactions=[WallReaction("H", 0.1, {"H2": 0.5})],
        ion_wall_products=driver.model_options["ion_wall_products"],
        openings=[Opening(lambda r, z: z > length - 1e-9, TRANSPARENCY, "grid")],
        inlets=[Inlet(lambda r, z: (z < 1e-9) & (r < 0.04), "H2", feed)],
    )
    return plasma, start, gas, fill, feed


def test_the_driver_gas_depletes_and_heats_in_the_plasma(driver_models) -> None:
    """The gas in the 25 kW plasma: balances to round-off, depleted, heated."""
    plasma, start, gas, fill, feed = driver_models
    gas.plasma_from(plasma, start)
    state = gas.steady_state(gas.pack({"H2": fill, "H": 0.1 * fill}, TW))
    assert state.converged
    balance = gas.balance(state)
    assert abs(balance["H"]) + abs(balance["H2"]) < 1e-9 * feed
    # the nuclei: fed and made from the ions at the walls, out through all walls
    budget = gas.wall_budget(state)
    out = {n: b["walls"] + b["openings"] - b["inlets"] for n, b in budget.items()}
    sources = {
        n: state.total(sum(s * state.reaction_rates[r.name] for r in gas.reactions
                           for s in [r.stoichiometry.get(n, 0)]))
        for n in ("H", "H2")
    }  # fmt: skip
    for n in ("H", "H2"):
        assert out[n] == pytest.approx(sources[n], rel=1e-9, abs=1e-9 * feed)
    assert budget["H2"]["inlets"] == pytest.approx(feed, rel=1e-12)
    terms = gas.energy_balance(state)
    assert abs(sum(terms.values())) < 1e-9 * terms["reactions"]
    assert terms["reactions"] > 0 and terms["walls"] < 0
    # depleted: fewer particles than the fill; heated: the atoms (Franck-Condon) most
    total = state.densities["H"] + state.densities["H2"]
    assert state.mean(total) < 0.95 * fill
    t_h2, t_h = (state.mean(state.temperatures[n]) for n in ("H2", "H"))
    assert t_h > t_h2 > 1.5 * TW


def test_the_plasma_and_the_gas_reach_a_common_steady_state(driver_models) -> None:
    """The coupled passes converge; the nuclei fed leave through the grid."""
    plasma, start, gas, fill, feed = driver_models
    gas.plasma_from(plasma, start)
    first = gas.steady_state(gas.pack({"H2": fill, "H": 0.1 * fill}, TW))
    coupled = couple_neutrals(plasma, gas, start, first, tolerance=1e-2)
    assert coupled.converged
    assert coupled.history[-1] < 1e-2
    state, ions = coupled.neutrals, coupled.plasma
    budget = gas.wall_budget(state)
    neutrals_out = budget["H"]["openings"] + 2 * budget["H2"]["openings"]
    # the net outflow of neutrals through the grid has the ions that come back
    # there (1 - transparency of them) taken off: with all the ions that reach the
    # grid it is what leaves
    fluxes = plasma.wall_fluxes(ions)
    grid = gas.wall_centres[1] > driver_length(gas) - 1e-9
    nuclei_per_ion = {"H+": 1, "H2+": 2, "H3+": 3}
    ions_out = sum(
        nuclei_per_ion[ion] * np.sum((fluxes[ion] * gas.wall_faces["area"])[grid])
        for ion in nuclei_per_ion
    )
    # to the tolerance of the passes (2 % at a 1 % change of the gas per pass)
    assert neutrals_out + ions_out == pytest.approx(2 * feed, rel=0.05)
    assert ions.converged and state.converged
    assert ions.mean(ions.electron_density) > 1e17


def driver_length(gas: NeutralFluid2D) -> float:
    return float(gas.grid.y_edges[-1])


def test_bad_models_are_refused(column) -> None:
    with pytest.raises(ValueError, match="no charge"):
        NeutralSpecies(H_ION)
    with pytest.raises(ValueError, match="unique"):
        NeutralFluid2D(column, [ATOMS, ATOMS])
    with pytest.raises(ValueError, match="unknown species"):
        NeutralFluid2D(column, [MOLECULES, ATOMS], [DISSOCIATION])
    with pytest.raises(ValueError, match="no neutral"):
        NeutralFluid2D(column, [ATOMS], collisions=[Collision("H2", "H", 1e-16)])
    with pytest.raises(ValueError, match="add up"):
        NeutralFluid2D(
            column,
            [ATOMS],
            wall_reactions=[WallReaction("H", 0.6), WallReaction("H", 0.6)],
        )
    with pytest.raises(ValueError, match="no wall face"):
        NeutralFluid2D(column, [ATOMS], openings=[Opening(lambda r, z: z > 1.0, 0.5)])
    with pytest.raises(ValueError, match="transparency"):
        Opening(lambda r, z: z > 0, 1.5)
    model = NeutralFluid2D(column, [ATOMS], plasma_species=PLASMA)
    with pytest.raises(ValueError, match="one ion wall flux"):
        model.set_plasma({"e": 1e17}, 3.0, ion_wall_flux={"H+": np.ones(3)})
    with pytest.raises(ValueError, match="unknown plasma species"):
        model.set_plasma({"Ar+": 1e17}, 3.0)
