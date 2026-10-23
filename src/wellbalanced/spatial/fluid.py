"""A 2D multi-species fluid of the RF driver plasma, in r–z or in a cut.

Step R3.3 of the RF-driver plan: the plasma that :class:`InductiveField2D` heats, on the
same cells. Every fluid species ``s`` of charge ``z_s`` obeys its continuity equation,
the ions their momentum equation, and the electrons drift and diffuse and carry their
energy:

    dn_s/dt + div Gamma_s = S_s,
    d(n_i v_i)/dt + div(n_i v_i v_i) = (z_i e / M_i) n_i E - (e T_g / M_i) grad n_i
                                        - nu_i n_i v_i - L_i v_i,
    Gamma_e = -mu_e n_e (E - F / e) - grad(D_e n_e),
    dw/dt + div Gamma_w = -Gamma_e . E - P_loss + p_RF / e,

with ``w = 3/2 n_e T_e`` and the electron energy flux of Hagelaar and Pitchford (2005)
(``5/3`` times the electrons' mobility and diffusion, applied to ``w``). The ions are
cold (``T_g``), born at rest, lose the momentum of those destroyed (``L_i``, the rate of
the reactions that remove them) and rub on the gas at ``nu_i = e / (M_i mu_i)``, so
without inertia their flux is the drift-diffusion one, ``z mu n E - mu T_g grad n``; or,
with ``ion_mean_free_path``, at ``nu_i = sqrt(v_th^2 + v_i^2) / lambda_i`` (a constant
mean free path, the drag behind the global model's edge-to-centre ratios);
inertia is what limits their flow to the walls at low pressure (0.3 Pa: the ion mean
free path is a fraction of the driver), where drift-diffusion ions leave about three
times too fast. ``ions="drift-diffusion"`` drops the inertia except for the ions whose
species asks for it (``FluidSpecies(momentum=True)``, as in
:class:`~wellbalanced.Fluid1D`). Two RF terms come from the field: the heating
``p_RF = 1/2 Re(J E*)`` and the RF-averaged Lorentz force ``F = -(e/2) Re(u x B*)`` on an
electron (:attr:`InductiveSolution.lorentz_force`), which pushes the electrons away
from the coil (it does no work: a magnetic force does none, and its work on the drift
comes out of the RF motion, whose power ``p_RF`` already is). The sources come from
:class:`~wellbalanced.reactions.Reaction` objects, rates from plasmacoll, at the local
``T_e``; the gas is a fixed background (``fixed``), as in NIBS'22 and step R3.5's first
form. The electrons' mobility is a constant or ``e / (m_e nu)`` with ``nu`` the rate of
the ``momentum_transfer`` reactions per electron, the same ``nu`` the field takes.

Two closures of the potential:

- ``"quasineutral"`` (the driver: its Debye length, about 20 µm, is not resolved):
  ``n_e = sum_i z_i n_i``, and the potential follows from the conservation of charge,
  ``div sum_s z_s Gamma_s = 0``. The walls are sheath edges: positive ions leave at
  least at the Bohm speed ``sqrt(z e T_e / M)``, negative ions are held, the electrons
  leave with ``1/4 n_e v_e exp(-(phi - phi_w) / T_e)`` and take ``2 T_e + (phi -
  phi_w)`` each (``phi`` in the wall cell, so the sheath and the last half cell are one
  drop). The walls are one conductor: only the total current to them vanishes. This is
  the sheath and Bohm closure of the global model (:class:`~wellbalanced.GlobalModel`)
  resolved in space. The density at the sheath edge is the wall cell's, a first-order
  error of ``u_B h / (2 D)``, small at low pressure.
- ``"poisson"``: :class:`~wellbalanced.Fluid1D`'s equations, Poisson's equation for the
  potential and its absorbing walls (the drift towards the wall plus half the thermal
  flux), the walls grounded, drift-diffusion ions. On a slab it is
  :class:`~wellbalanced.Fluid1D` (the tests compare the two); it needs the Debye length
  resolved.

The form of the ion momentum equation and the wall conditions of Zielke et al.'s model
are in Zielke's thesis, not yet read (the plan's R0); the ones here are the textbook
ones above.

Grid: the plasma's cells of a :class:`~wellbalanced.spatial.CellGrid` (tensor, uniform or
stretched; r–z or the cut) or of a single-level
:class:`~wellbalanced.spatial.PatchGrid`; their faces from ``grid.transport_faces()``.
Conservative transport on refined patches needs patchwork's refluxing (its plan, 4c),
not merged yet, so a multi-level patch grid is refused. The plasma is the
:class:`~wellbalanced.spatial.PlasmaRegion` (or every cell): a face to a cell outside it is
a wall; so is the domain's edge, except a ``"neumann"`` edge (a symmetry plane), a
periodic seam and the axis of an r–z domain.

Discretisation: finite volumes on the faces of the grid, one flux per face, so every
species and the electron energy are conserved to round-off. Electrons (and
drift-diffusion ions) have :class:`~wellbalanced.Fluid1D`'s Scharfetter-Gummel fluxes,
exact for a constant field between two centres; the ions' mass and momentum fluxes are
Rusanov's (first order) with the speed ``sqrt(v_-^2 + v_+^2 + e (T_e + T_g) / M)``, and
their forces are taken at the cells from the gradients of the interior faces (the sheath
drop of a wall face acts on no ion, and the ions leave a wall cell with its momentum).
The work of the field ``Gamma . E`` on a face is
shared between its two cells in proportion to their distances to it. Backward Euler,
every equation implicit, solved by Newton's method in the logarithms of the densities
and of ``w``, the Jacobian by finite differences on a distance-2 colouring of the cells.
:meth:`Fluid2D.steady_state` steps in pseudo-time with a growing step, then solves the
steady equations.

Coupled to the field through :func:`~wellbalanced.spatial.coupling.fluid_plasma`: the field
of the fluid's plasma, the fluid heated by that field, repeated to a fixed point at a
given absorbed power, which :class:`~wellbalanced.spatial.DriverCoupling` turns into the
coil current and the efficiency at a given generator power.

Verification (``tests/unit/test_fluid2d.py``): particles and energy conserved to
round-off over time steps; a slab against :class:`~wellbalanced.Fluid1D` to round-off;
ambipolar diffusion in a slab and in an infinite cylinder against their closed forms
(first order); the collisionless slab with uniform ionization against its closed form
(``n_s = n_0 / 2``); a single-level patch grid against the tensor grid; the steady state
of a driver and its balances against the global model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import xarray as xr
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from scope_profiler import profile

from wellbalanced._profiling import _PROFILE_TAGS
from wellbalanced.datasets import variable
from wellbalanced.fluid1d import FluidSpecies, scharfetter_gummel
from wellbalanced.reactions import Reaction
from wellbalanced.spatial.grid import CellField, CellGrid
from wellbalanced.spatial.regions import PlasmaRegion

if TYPE_CHECKING:
    from wellbalanced.spatial.coupling import PlasmaProfile
    from wellbalanced.spatial.inductive import InductiveSolution
    from wellbalanced.spatial.patches import PatchGrid

__all__ = ["Fluid2D", "Fluid2DState"]

E = ELEMENTARY_CHARGE
EPSILON_0 = 8.8541878128e-12
#: Floor of the electron temperature (eV) and density (m^-3).
TE_FLOOR = 1.0e-3
NE_FLOOR = 1.0
#: Newton: the largest change of a logarithm per iteration, and the tolerance.
MAX_LOG_STEP = 2.0
NEWTON_TOLERANCE = 1.0e-10
#: Newton: the largest change of the potential (in units of the largest ``T_e``) and of
#: an ion velocity (in units of the largest Bohm speed) per iteration.
MAX_POTENTIAL_STEP = 2.0
MAX_VELOCITY_STEP = 0.5

#: The width of the smooth maximum of the ions' outflow speed and the Bohm speed.
BOHM_SMOOTHING = 1.0e-2

Closure = Literal["quasineutral", "poisson"]
IonModel = Literal["momentum", "drift-diffusion"]


def _smooth_max(a: np.ndarray, b: np.ndarray, width: np.ndarray) -> np.ndarray:
    """``max(a, b)``, rounded over ``width`` (differentiable for Newton's method)."""
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + width**2))


@dataclass
class Fluid2DState:
    """The fluid on the grid: fields over the grid's cells, zero outside the plasma.

    Attributes:
        grid: The grid.
        plasma: The plasma cells (a mask of the grid's shape).
        unknowns: The model's vector of unknowns, to continue from.
        densities: Density of every species [m^-3], the electrons included.
        velocities: In-plane velocity ``(v_1, v_2)`` [m/s] of every ion with inertia
            (``(v_x, v_y)`` in a cut, ``(v_r, v_z)`` in r–z).
        electron_temperature: ``T_e`` [eV].
        potential: ``phi`` [V].
        reaction_rates: Volume rate of every reaction [m^-3 s^-1].
        time: Time [s] of the state; ``inf`` for a steady state.
        converged: Whether the last solve converged.
    """

    grid: CellGrid | PatchGrid
    plasma: np.ndarray
    unknowns: np.ndarray
    densities: dict[str, np.ndarray]
    velocities: dict[str, tuple[np.ndarray, np.ndarray]]
    electron_temperature: np.ndarray
    potential: np.ndarray
    reaction_rates: dict[str, np.ndarray]
    time: float = 0.0
    converged: bool = True

    def total(self, values: np.ndarray) -> float:
        """The integral of a field over the plasma (per metre along ``z`` in a cut)."""
        return float(np.sum(values[self.plasma] * self.grid.cell_volumes[self.plasma]))

    @property
    def volume(self) -> float:
        """The plasma's volume [m^3] (per metre along ``z`` in a cut)."""
        return float(np.sum(self.grid.cell_volumes[self.plasma]))

    def mean(self, values: np.ndarray) -> float:
        """The volume average of a field over the plasma."""
        return self.total(values) / self.volume

    @property
    def electron_density(self) -> np.ndarray:
        """``n_e`` [m^-3]."""
        return self.densities["e"]

    def to_dataset(self) -> xr.Dataset:
        """The state as a labelled dataset, fields over the grid's cells."""
        dims = self.grid.dims
        species = list(self.densities)
        reactions = list(self.reaction_rates)
        first, second = self.grid.coordinate_names
        data = {
            "density": variable(
                ("species", *dims),
                [self.densities[s] for s in species],
                "density",
                "m^-3",
            ),
            "electron_temperature": variable(
                dims, self.electron_temperature, "electron temperature", "eV"
            ),
            "potential": variable(dims, self.potential, "potential", "V"),
            "plasma": variable(dims, self.plasma, "plasma cells"),
            "converged": variable((), self.converged, "converged"),
        }
        coords: dict[str, Any] = {
            **self.grid.coordinates(),
            "species": ("species", species, {"long_name": "species"}),
        }
        ions = list(self.velocities)
        if ions:
            for k, axis in enumerate((first, second)):
                data[f"velocity_{axis}"] = variable(
                    ("ion", *dims),
                    [self.velocities[s][k] for s in ions],
                    f"ion velocity v_{axis}",
                    "m/s",
                )
            coords["ion"] = ("ion", ions, {"long_name": "ion"})
        if reactions:
            data["reaction_rate"] = variable(
                ("reaction", *dims),
                [self.reaction_rates[r] for r in reactions],
                "reaction rate",
                "m^-3 s^-1",
            )
            coords["reaction"] = ("reaction", reactions, {"long_name": "reaction"})
        return xr.Dataset(
            data,
            coords=coords,
            attrs={"time": self.time, "geometry": self.grid.geometry},
        )


class Fluid2D:
    """The plasma on the cells of a grid; see the module docstring.

    Args:
        grid: A :class:`CellGrid` or a single-level :class:`PatchGrid` (the field's).
        species: The fluid species, the electrons (``"e"``) included; a species'
            ``mobility`` sets an ion's friction on the gas, ``e / (M mu)``, and
            ``diffusion`` overrides the Einstein relation of the drift-diffusion
            species (``D = mu T_e`` for the electrons, ``mu T_g`` for the rest).
        reactions: The volume reactions; reactants that are not fluid species must be
            in ``fixed``.
        region: Where the plasma is: a :class:`PlasmaRegion`, a mask of the grid's
            shape, or ``None`` for every cell.
        fixed: Densities of species not carried, e.g. the gas [m^-3] (cell fields).
        gas_temperature: ``T_g`` [eV], also the ions' temperature.
        electron_temperature: ``None`` solves the electron energy equation; a number
            holds ``T_e`` [eV] fixed.
        potential: ``"quasineutral"`` (sheath walls) or ``"poisson"`` (Fluid1D's).
        ions: ``"momentum"`` (inertia; the default of the quasi-neutral closure) or
            ``"drift-diffusion"`` (the only form with Poisson's equation); an ion
            whose :class:`~wellbalanced.fluid1d.FluidSpecies` has ``momentum=True``
            has inertia either way.
        heating: The power to the electrons [W/m^3], a cell field: the field's
            ``1/2 Re(J E*)`` (see :meth:`set_field`).
        electron_force: The in-plane force on an electron [N], two cell fields: the
            field's RF-averaged Lorentz force (see :meth:`set_field`).
        sources: External volume sources ``{species: rate}`` [m^-3 s^-1]; ions are
            born at rest.
        momentum_transfer: Names of the reactions whose rates per electron make up the
            electrons' momentum-transfer frequency ``nu``; with them the electron
            mobility is ``e / (m_e nu)`` instead of the species' constant.
        ion_mean_free_path: ``lambda_i`` [m], one or per ion: the ions with inertia rub
            on the gas at ``nu_i = sqrt(v_th^2 + v^2) / lambda_i`` (a constant mean free
            path, charge exchange; ``v_th`` the mean thermal speed at ``T_g``), the drag
            of the global model's edge-to-centre ratios, instead of ``e / (M mu)``.
        wall_potential: The potential of the walls [V].
    """

    def __init__(
        self,
        grid: CellGrid | PatchGrid,
        species: Sequence[FluidSpecies],
        reactions: Sequence[Reaction] = (),
        *,
        region: PlasmaRegion | np.ndarray | None = None,
        fixed: Mapping[str, CellField] | None = None,
        gas_temperature: float = 0.025,
        electron_temperature: float | None = None,
        potential: Closure = "quasineutral",
        ions: IonModel | None = None,
        heating: CellField = 0.0,
        electron_force: tuple[CellField, CellField] | None = None,
        sources: Mapping[str, CellField] | None = None,
        momentum_transfer: Sequence[str] = (),
        ion_mean_free_path: float | Mapping[str, float] | None = None,
        wall_potential: float = 0.0,
    ) -> None:
        """Index the plasma cells, their faces and the unknowns; check the species."""
        if potential not in ("quasineutral", "poisson"):
            raise ValueError(f"unknown potential closure {potential!r}")
        if ions is None:
            ions = "momentum" if potential == "quasineutral" else "drift-diffusion"
        if ions not in ("momentum", "drift-diffusion"):
            raise ValueError(f"unknown ion model {ions!r}")
        flagged = any(s.momentum for s in species)
        if potential == "poisson" and (ions == "momentum" or flagged):
            raise ValueError("ions with inertia need the quasi-neutral closure")
        self.grid = grid
        self.species = tuple(species)
        self.reactions = tuple(reactions)
        self.gas_temperature = float(gas_temperature)
        self.electron_temperature = electron_temperature
        self.closure: Closure = potential
        self.ions: IonModel = ions
        self.wall_potential = float(wall_potential)
        self.momentum_transfer = tuple(momentum_transfer)
        names = [s.name for s in self.species]
        if ion_mean_free_path is None or isinstance(ion_mean_free_path, Mapping):
            self.ion_mean_free_path = dict(ion_mean_free_path or {})
        else:
            self.ion_mean_free_path = {
                s.name: float(ion_mean_free_path)
                for s in self.species
                if s.name != "e" and s.species.charge != 0
            }
        if any(v <= 0.0 for v in self.ion_mean_free_path.values()):
            raise ValueError("ion mean free paths must be > 0")
        if len(set(names)) != len(names):
            raise ValueError("species names must be unique")
        if "e" not in names:
            raise ValueError('the model needs the electrons, a species named "e"')
        self._by_name = {s.name: s for s in self.species}
        fixed = dict(fixed or {})
        known = set(names) | set(fixed)
        by_reaction = {r.name: r for r in self.reactions}
        self._reaction_index = {r.name: k for k, r in enumerate(self.reactions)}
        for reaction in self.reactions:
            unknown = (set(reaction.reactants) | set(reaction.products)) - known
            if unknown:
                raise ValueError(f"{reaction.name}: unknown species {sorted(unknown)}")
        for name in self.momentum_transfer:
            reaction = by_reaction.get(name)
            if reaction is None:
                raise ValueError(f"momentum_transfer: unknown reaction {name!r}")
            if list(reaction.reactants).count("e") != 1:
                raise ValueError(f"{name}: a momentum transfer needs one electron")
        if potential == "quasineutral" and not any(
            s.species.charge > 0 for s in self.species
        ):
            raise ValueError("a quasi-neutral plasma needs a positive ion")

        # the plasma cells and their faces
        if region is None:
            mask = np.ones(grid.shape, dtype=bool)
        elif isinstance(region, PlasmaRegion):
            mask = region.mask(grid)
        else:
            mask = np.asarray(region, dtype=bool).reshape(grid.shape)
        if not mask.any():
            raise ValueError("the plasma region covers no cell")
        self.plasma = mask
        self.cells = np.flatnonzero(mask.ravel())
        self.size = self.cells.size
        self._faces(grid)
        self.volumes = grid.cell_volumes.ravel()[self.cells]

        # the unknowns per cell: log densities, ion velocities, log w, phi
        self._carried = [
            n for n in names if not (n == "e" and potential == "quasineutral")
        ]
        self._inertial = [
            s.name
            for s in self.species
            if s.name != "e"
            and s.species.charge != 0
            and (ions == "momentum" or s.momentum)
        ]
        self._energy = electron_temperature is None
        columns = len(self._carried)
        self._velocity_columns = {}
        for name in self._inertial:
            self._velocity_columns[name] = (columns, columns + 1)
            columns += 2
        self._energy_column = columns
        self._nvar = columns + int(self._energy) + 1
        self._is_log = np.zeros(self._nvar, dtype=bool)
        self._is_log[: len(self._carried)] = True
        if self._energy:
            self._is_log[self._energy_column] = True
        self._fixed = {name: self._on_cells(value) for name, value in fixed.items()}
        self._sources = {
            name: self._on_cells(value) for name, value in (sources or {}).items()
        }
        self.heating: np.ndarray = self._on_cells(heating)
        self.electron_force: tuple[np.ndarray, np.ndarray] = (
            (np.zeros(self.size), np.zeros(self.size))
            if electron_force is None
            else (self._on_cells(electron_force[0]), self._on_cells(electron_force[1]))
        )
        self._colours = self._colouring()
        self._rate_cache: dict[bytes, list[np.ndarray]] = {}
        self._iterations = 0

    # --- geometry ------------------------------------------------------------------

    def _faces(self, grid: CellGrid | PatchGrid) -> None:
        """The interior faces of the plasma and its walls, in plasma-cell indices."""
        faces = grid.transport_faces()
        index = np.full(grid.size, -1)
        index[self.cells] = np.arange(self.size)
        low, high = index[faces.low], index[faces.high]
        inner = (low >= 0) & (high >= 0)
        self._low, self._high = low[inner], high[inner]
        self._axis = faces.axis[inner]
        self._area = faces.area[inner]
        self._delta_low = faces.delta_low[inner]
        self._delta_high = faces.delta_high[inner]
        self._distance = self._delta_low + self._delta_high
        # walls: faces to a cell outside the plasma, and the domain's edges
        to_high = (low >= 0) & (high < 0)  # the wall is above the plasma cell
        to_low = (low < 0) & (high >= 0)
        edge = index[faces.boundary_cell]
        edge_wall = (edge >= 0) & ~faces.boundary_symmetry
        self._wall_cell = np.concatenate([low[to_high], high[to_low], edge[edge_wall]])
        self._wall_axis = np.concatenate(
            [faces.axis[to_high], faces.axis[to_low], faces.boundary_axis[edge_wall]]
        )
        self._wall_sign = np.concatenate(
            [
                np.ones(np.count_nonzero(to_high)),
                -np.ones(np.count_nonzero(to_low)),
                np.where(faces.boundary_side[edge_wall] == 1, 1.0, -1.0),
            ]
        )
        self._wall_area = np.concatenate(
            [faces.area[to_high], faces.area[to_low], faces.boundary_area[edge_wall]]
        )
        self._wall_delta = np.concatenate(
            [
                faces.delta_low[to_high],
                faces.delta_high[to_low],
                faces.boundary_delta[edge_wall],
            ]
        )
        spacing = np.concatenate([self._distance, 2.0 * self._wall_delta])
        self._length_scale = float(np.min(spacing)) if spacing.size else 1.0
        n = self.size
        # the weights of the cell gradients: the areas of a cell's faces per axis
        self._gradient_weight = []
        for axis in (0, 1):
            along = np.where(self._axis == axis, self._area, 0.0)
            total = np.bincount(self._low, along, n) + np.bincount(self._high, along, n)
            self._gradient_weight.append(
                np.where(total > 0.0, 1.0 / np.where(total > 0, total, 1.0), 0.0)
            )

    def _on_cells(self, value: CellField) -> np.ndarray:
        """A cell field (or an array over the plasma cells) on the plasma cells."""
        if not callable(value):
            array = np.asarray(value)
            if array.shape == (self.size,) and self.size != self.grid.size:
                return array.astype(float)
        return self.grid.cell_field(value).ravel()[self.cells]

    def on_grid(self, values: np.ndarray, outside: float = 0.0) -> np.ndarray:
        """Values on the plasma cells as a field over the grid, ``outside`` elsewhere."""
        result = np.full(self.grid.size, outside, dtype=np.asarray(values).dtype)
        result[self.cells] = values
        return result.reshape(self.grid.shape)

    def _colouring(self) -> np.ndarray:
        """A colouring of the cells in which no two of one colour share a neighbour.

        A cell's residual depends on its neighbours across faces, so perturbing all
        cells of one colour at once gives separate columns of the Jacobian.
        """
        n = self.size
        ones = np.ones(self._low.size)
        adjacency = sp.coo_matrix((ones, (self._low, self._high)), shape=(n, n))
        adjacency = (adjacency + adjacency.T + sp.identity(n)).tocsr()
        adjacency.data[:] = 1.0
        square = (adjacency @ adjacency).tocsr()
        colours = np.full(n, -1)
        for cell in range(n):
            neighbours = square.indices[square.indptr[cell] : square.indptr[cell + 1]]
            used = set(colours[neighbours].tolist())
            colour = 0
            while colour in used:
                colour += 1
            colours[cell] = colour
        self._adjacency = adjacency.tocoo()
        return colours

    # --- the unknowns ----------------------------------------------------------------

    def pack(
        self,
        densities: Mapping[str, CellField],
        electron_temperature: CellField = 3.0,
        potential: CellField = 0.0,
        velocities: Mapping[str, tuple[CellField, CellField]] | None = None,
    ) -> np.ndarray:
        """The unknowns of fields (densities by name, ``T_e``, ``phi``, ion velocities).

        Fields are cell fields of the grid or arrays over the plasma cells; in the
        quasi-neutral closure the electron density follows from the ions'. The ions are
        at rest unless ``velocities`` are given.
        """
        u = np.zeros((self.size, self._nvar))
        for k, name in enumerate(self._carried):
            u[:, k] = np.log(np.maximum(self._on_cells(densities[name]), NE_FLOOR))
        for name, (c0, c1) in self._velocity_columns.items():
            if velocities is not None and name in velocities:
                u[:, c0] = self._on_cells(velocities[name][0])
                u[:, c1] = self._on_cells(velocities[name][1])
        if self._energy:
            ne = self._electrons(
                {n: np.exp(u[:, k]) for k, n in enumerate(self._carried)}
            )
            te = self._on_cells(electron_temperature)
            u[:, self._energy_column] = np.log(1.5 * ne * np.maximum(te, TE_FLOOR))
        u[:, -1] = self._on_cells(potential)
        return u.ravel()

    def _electrons(self, densities: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.closure == "poisson":
            return densities["e"]
        charge = sum(
            s.species.charge * densities[s.name]
            for s in self.species
            if s.name != "e" and s.species.charge != 0
        )
        return np.maximum(charge, NE_FLOOR)

    def _unpack(self, unknowns: np.ndarray):
        u = unknowns.reshape(self.size, self._nvar)
        densities = {name: np.exp(u[:, k]) for k, name in enumerate(self._carried)}
        densities["e"] = self._electrons(densities)
        velocities = {
            name: (u[:, c0], u[:, c1])
            for name, (c0, c1) in self._velocity_columns.items()
        }
        if self._energy:
            w = np.exp(u[:, self._energy_column])
            te = np.maximum(w / (1.5 * densities["e"]), TE_FLOOR)
        else:
            te = np.full(self.size, float(self.electron_temperature))  # type: ignore[arg-type]
        return densities, velocities, te, u[:, -1]

    # --- coefficients ------------------------------------------------------------------

    def _coefficients(self, te: np.ndarray) -> list[np.ndarray]:
        """Every reaction's rate coefficient on the cells, remembered for a few ``T_e``.

        Newton's finite differences perturb the ion velocities and the potential at the
        same ``T_e``, so most of their residuals reuse the coefficients.
        """
        key = te.tobytes()
        values = self._rate_cache.get(key)
        if values is None:
            tg = self.gas_temperature
            ones = np.ones(self.size)
            values = [r.coefficients(te, tg) * ones for r in self.reactions]
            if len(self._rate_cache) >= 4:
                self._rate_cache.pop(next(iter(self._rate_cache)))
            self._rate_cache[key] = values
        return values

    def _collision_frequency(self, densities, te: np.ndarray) -> np.ndarray:
        """The electrons' momentum-transfer frequency ``nu`` on the plasma cells."""
        if not self.momentum_transfer:
            mobility = self._by_name["e"].mobility
            return np.full(self.size, E / (ELECTRON_MASS * mobility))
        n_all = {**self._fixed, **densities}
        nu = np.zeros(self.size)
        coefficients = self._coefficients(te)
        for name in self.momentum_transfer:
            k = self._reaction_index[name]
            reaction, rate = self.reactions[k], coefficients[k]
            for reactant in reaction.reactants:
                if reactant != "e":
                    rate = rate * n_all[reactant]
            nu = nu + rate
        return nu

    def _mobility(self, fluid: FluidSpecies, densities, te) -> np.ndarray:
        """The mobility on the plasma cells."""
        if fluid.name == "e" and self.momentum_transfer:
            return E / (ELECTRON_MASS * self._collision_frequency(densities, te))
        return np.full(self.size, fluid.mobility)

    def _temperature(self, fluid: FluidSpecies, te: np.ndarray) -> np.ndarray:
        return te if fluid.name == "e" else np.full(self.size, self.gas_temperature)

    def _ion_friction(self, fluid: FluidSpecies, velocity) -> np.ndarray | float:
        """An ion's collision frequency with the gas, ``nu_i`` [1/s]."""
        mass = fluid.species.mass
        path = self.ion_mean_free_path.get(fluid.name)
        if path is None:
            return E / (mass * fluid.mobility)
        thermal = 8.0 * E * self.gas_temperature / (np.pi * mass)
        return np.sqrt(thermal + velocity[0] ** 2 + velocity[1] ** 2) / path

    def _gradient(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The gradient on the cells: the area-weighted mean of their interior faces'."""
        low, high, n = self._low, self._high, self.size
        face = (values[high] - values[low]) / self._distance * self._area
        result = []
        for axis in (0, 1):
            along = np.where(self._axis == axis, face, 0.0)
            total = np.bincount(low, along, n) + np.bincount(high, along, n)
            result.append(total * self._gradient_weight[axis])
        return result[0], result[1]

    # --- fluxes ------------------------------------------------------------------------

    def _transport(self, densities, velocities, te, phi):
        """Fluxes on the interior faces (low to high) and out through the walls.

        Returns ``(interior, walls, fields, momentum)``: fluxes by species and
        ``"energy"``; the fields ``E`` (interior, along the face's axis) and ``E_out``
        (walls); and the ions' momentum fluxes by species, ``(interior, walls)`` per
        component.
        """
        low, high, distance = self._low, self._high, self._distance
        wc = self._wall_cell
        efield = -(phi[high] - phi[low]) / distance
        sheath = phi[wc] - self.wall_potential  # the drop to the wall
        e_out = sheath / self._wall_delta
        force = self._face_force()
        interior: dict[str, np.ndarray] = {}
        walls: dict[str, np.ndarray] = {}
        momentum: dict[str, Any] = {}
        for fluid in self.species:
            name, z = fluid.name, fluid.species.charge
            n = densities[name]
            if name in velocities:
                interior[name], walls[name], momentum[name] = self._ion_fluxes(
                    fluid, n, velocities[name], te
                )
                continue
            mu_cell = self._mobility(fluid, densities, te)
            t_cell = self._temperature(fluid, te)
            mu = 0.5 * (mu_cell[low] + mu_cell[high])
            if fluid.diffusion is not None:
                d = np.full(low.size, fluid.diffusion)
            else:
                d = mu * 0.5 * (t_cell[low] + t_cell[high])
            field, field_out = efield, e_out
            if name == "e":  # the force acts as the field -F/e on the electrons
                field = efield - force[0] / E
                field_out = e_out - force[1] / E
            interior[name] = scharfetter_gummel(
                z, mu, d, field, n[low], n[high], distance
            )
            walls[name] = self._wall_flux(
                fluid, n[wc], t_cell[wc], mu_cell[wc], field_out, sheath, te[wc]
            )
            if name == "e" and self._energy:
                w = 1.5 * n * te
                interior["energy"] = scharfetter_gummel(
                    z, 5 / 3 * mu, 5 / 3 * d, field, w[low], w[high], distance
                )
                walls["energy"] = self._wall_flux(
                    fluid,
                    w[wc],
                    t_cell[wc],
                    mu_cell[wc],
                    field_out,
                    sheath,
                    te[wc],
                    energy=True,
                    particles=walls["e"],
                )
        return interior, walls, (efield, e_out), momentum

    def _ion_fluxes(self, fluid: FluidSpecies, n, velocity, te):
        """Rusanov fluxes of an ion's mass and momentum, and its Bohm outflow."""
        low, high, axis = self._low, self._high, self._axis
        z, mass = fluid.species.charge, fluid.species.mass
        nl, nh = n[low], n[high]
        vl, vh = [c[low] for c in velocity], [c[high] for c in velocity]
        v_low = np.where(axis == 0, vl[0], vl[1])
        v_high = np.where(axis == 0, vh[0], vh[1])
        sound = E * (z * 0.5 * (te[low] + te[high]) + self.gas_temperature) / mass
        speed = np.sqrt(v_low**2 + v_high**2 + sound)
        flux = 0.5 * (nl * v_low + nh * v_high) - 0.5 * speed * (nh - nl)
        inner = [
            0.5 * (nl * cl * v_low + nh * ch * v_high)
            - 0.5 * speed * (nh * ch - nl * cl)
            for cl, ch in zip(vl, vh, strict=True)
        ]
        wc = self._wall_cell
        bohm = np.sqrt(z * E * te[wc] / mass)
        normal = self._wall_sign * np.where(
            self._wall_axis == 0, velocity[0][wc], velocity[1][wc]
        )
        out_speed = _smooth_max(normal, bohm, BOHM_SMOOTHING * bohm)
        out = n[wc] * out_speed
        # the ions leave with the cell's momentum: their acceleration to the Bohm speed
        # happens in the half cell to the wall, whose force is left out as well
        outer = [out * c[wc] for c in velocity]
        return flux, out, (inner, outer)

    def _wall_flux(
        self,
        fluid: FluidSpecies,
        n: np.ndarray,
        temperature: np.ndarray,
        mu: np.ndarray,
        field_out: np.ndarray,
        sheath: np.ndarray,
        te: np.ndarray,
        *,
        energy: bool = False,
        particles: np.ndarray | None = None,
    ) -> np.ndarray:
        """The flux out through the walls of the density (or energy) ``n`` of their cells."""
        z, mass = fluid.species.charge, fluid.species.mass
        thermal = np.sqrt(8.0 * E * temperature / (np.pi * mass))
        if self.closure == "poisson" or z == 0:
            # Fluid1D's absorbing wall: the drift towards it plus half the thermal flux
            drift = np.maximum(z * mu * field_out, 0.0)
            if energy:
                return 5.0 / 3.0 * drift * n + 2.0 / 3.0 * thermal * n
            return drift * n + 0.5 * thermal * n
        if fluid.name == "e":
            if energy:  # each electron takes 2 T_e and the sheath drop
                assert particles is not None
                return particles * (2.0 * te + sheath)
            return 0.25 * thermal * n * np.exp(-sheath / te)
        if z < 0:
            return np.zeros_like(n)  # negative ions are held by the sheath
        return n * np.sqrt(z * E * te / mass)  # the Bohm speed

    def _face_force(self) -> tuple[np.ndarray, np.ndarray]:
        """The electron force along the interior faces' axes and out through the walls."""
        components = self.electron_force
        low, high = self._low, self._high
        along = np.where(
            self._axis == 0,
            0.5 * (components[0][low] + components[0][high]),
            0.5 * (components[1][low] + components[1][high]),
        )
        wc = self._wall_cell
        out = self._wall_sign * np.where(
            self._wall_axis == 0, components[0][wc], components[1][wc]
        )
        return along, out

    def _divergence(self, interior: np.ndarray, walls: np.ndarray) -> np.ndarray:
        """``div Gamma`` on the cells, from the face fluxes (one per face)."""
        n = self.size
        out = (
            np.bincount(self._low, interior * self._area, n)
            - np.bincount(self._high, interior * self._area, n)
            + np.bincount(self._wall_cell, walls * self._wall_area, n)
        )
        return out / self.volumes

    def _face_work(self, interior: np.ndarray, walls: np.ndarray | None) -> np.ndarray:
        """A power per volume on the faces, shared between their cells, per cell volume.

        ``interior`` and ``walls`` are densities (per volume) on the faces; a face's
        share of a cell is its area times its distance to the cell's centre, so the
        sum over the cells is ``sum_faces A d (interior)``.
        """
        n = self.size
        total = np.bincount(
            self._low, interior * self._area * self._delta_low, n
        ) + np.bincount(self._high, interior * self._area * self._delta_high, n)
        if walls is not None:
            total += np.bincount(
                self._wall_cell, walls * self._wall_area * self._wall_delta, n
            )
        return total / self.volumes

    def _field_work(self, interior, walls, fields) -> np.ndarray:
        """The work of the field on the electrons, ``-Gamma_e . E`` (eV m^-3 s^-1).

        The RF Lorentz force does none: a magnetic force does no work, and its work
        on the drift comes out of the RF motion, whose power is already ``p_RF``.
        """
        efield, e_out = fields
        gamma, gamma_out = interior["e"], walls["e"]
        # in the quasi-neutral closure the sheath energy is in the wall flux
        at_walls = self.closure == "poisson"
        return self._face_work(
            -gamma * efield, -gamma_out * e_out if at_walls else None
        )

    # --- the equations -----------------------------------------------------------------

    def _reactions(self, densities, te):
        """The reactions' rates, the species' sources and destruction, the energy loss."""
        n_all = {**self._fixed, **densities}
        sources = {s.name: np.zeros(self.size) for s in self.species}
        destroyed = {name: np.zeros(self.size) for name in self._inertial}
        for name, value in self._sources.items():
            if name in sources:
                sources[name] = sources[name] + value
        loss = np.zeros(self.size)
        rates: dict[str, np.ndarray] = {}
        tg = self.gas_temperature
        for reaction, rate in zip(self.reactions, self._coefficients(te), strict=True):
            for name in reaction.reactants:
                rate = rate * n_all[name]
            rates[reaction.name] = rates.get(reaction.name, 0.0) + rate
            for name, count in reaction.stoichiometry.items():
                if name in sources:
                    sources[name] = sources[name] + count * rate
                if name in destroyed and count < 0:
                    destroyed[name] = destroyed[name] - count * rate
            loss = loss + reaction.electron_loss(te, tg) * rate  # type: ignore[arg-type]
        return rates, sources, loss, destroyed

    def _residual(
        self, unknowns: np.ndarray, previous: np.ndarray | None, dt: float
    ) -> np.ndarray:
        """Backward Euler residual (``previous is None``: the steady equations)."""
        densities, velocities, te, phi = self._unpack(unknowns)
        interior, walls, fields, momentum = self._transport(
            densities, velocities, te, phi
        )
        _, sources, loss, destroyed = self._reactions(densities, te)
        old = None if previous is None else self._unpack(previous)
        res = np.empty((self.size, self._nvar))
        balance = {}
        for fluid in self.species:
            name = fluid.name
            term = self._divergence(interior[name], walls[name]) - sources[name]
            if old is not None:
                term = term + (densities[name] - old[0][name]) / dt
            balance[name] = term
        for k, name in enumerate(self._carried):
            res[:, k] = balance[name] / max(float(np.max(densities[name])), 1.0)
        field = self._gradient(-phi) if self._inertial else (0.0, 0.0)
        for name in self._inertial:
            fluid = self._by_name[name]
            z, mass = fluid.species.charge, fluid.species.mass
            n, v = densities[name], velocities[name]
            friction = self._ion_friction(fluid, v) * n + destroyed[name]
            pressure = self._gradient(n)
            (inner, outer) = momentum[name]
            scale = max(float(np.max(n)), 1.0) * max(
                float(np.max(np.abs(v[0]))), float(np.max(np.abs(v[1]))), 1.0
            )
            for k, column in enumerate(self._velocity_columns[name]):
                term = (
                    self._divergence(inner[k], outer[k])
                    - z * E / mass * n * field[k]
                    + E * self.gas_temperature / mass * pressure[k]
                    + friction * v[k]
                )
                if old is not None:
                    term = term + (n * v[k] - old[0][name] * old[1][name][k]) / dt
                res[:, column] = term / scale
        if self._energy:
            n = densities["e"]
            w = 1.5 * n * te
            joule = self._field_work(interior, walls, fields)
            term = (
                self._divergence(interior["energy"], walls["energy"])
                - joule
                + loss
                - self.heating / E
            )
            if old is not None:
                term = term + (w - 1.5 * old[0]["e"] * old[2]) / dt
            res[:, self._energy_column] = term / max(float(np.max(w)), 1.0)
        scale = max(float(np.max(densities["e"])), 1.0)
        if self.closure == "quasineutral":
            # the conservation of charge: sum_s z_s (the balance of s)
            current = sum(
                f.species.charge * balance[f.name]
                for f in self.species
                if f.species.charge != 0
            )
            res[:, -1] = current / scale
        else:
            efield, e_out = fields
            div_e = self._divergence(efield, e_out)
            charge = sum(f.species.charge * densities[f.name] for f in self.species)
            res[:, -1] = (div_e - E * charge / EPSILON_0) * self._length_scale**2
        return res.ravel()

    def _jacobian(self, unknowns: np.ndarray, previous, dt) -> sp.csc_matrix:
        """The Jacobian by finite differences, one colour of cells per evaluation."""
        nvar, n = self._nvar, self.size
        base = self._residual(unknowns, previous, dt)
        rows_adj, cols_adj = self._adjacency.row, self._adjacency.col
        rows, cols, vals = [], [], []
        for colour in range(int(self._colours.max()) + 1):
            chosen = self._colours == colour
            # the residual cells each perturbed cell reaches
            reach = chosen[cols_adj]
            target, source = rows_adj[reach], cols_adj[reach]
            for k in range(nvar):
                perturbed = unknowns.copy()
                index = np.flatnonzero(chosen) * nvar + k
                step = 1.0e-7 * np.maximum(1.0, np.abs(unknowns[index]))
                perturbed[index] += step
                change = (self._residual(perturbed, previous, dt) - base).reshape(
                    n, nvar
                )
                steps = np.zeros(n)
                steps[chosen] = step
                block = change[target] / steps[source][:, None]
                for j in range(nvar):
                    rows.append(target * nvar + j)
                    cols.append(source * nvar + k)
                    vals.append(block[:, j])
        size = n * nvar
        return sp.csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(size, size),
        )

    @profile("wellbalanced.Fluid2D.newton", tags=_PROFILE_TAGS)
    def _newton(self, unknowns, previous, dt, max_iterations: int | None = None):
        """Solve the (backward Euler or steady) equations; return the solution and success.

        Newton's method with the finite-difference Jacobian. Once the steps are taken
        whole (no damping, :meth:`_damping`) and shrink at least four times per
        iteration, the Jacobian and its LU are kept for the next iteration. A time step
        gives up after 15 iterations (a shorter one is cheaper), the steady equations
        after 40.
        """
        if max_iterations is None:
            max_iterations = 40 if previous is None else 15
        u = unknowns.copy()
        is_log = np.tile(self._is_log, self.size)
        factor = None
        last = np.inf
        for iteration in range(1, max_iterations + 1):
            self._iterations = iteration
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                residual = self._residual(u, previous, dt)
                if not np.all(np.isfinite(residual)):
                    return u, False
                if factor is None:
                    factor = self._factorized(self._jacobian(u, previous, dt))
                    if factor is None:
                        return u, False
                    last = np.inf
                delta = factor.solve(-residual)
            if not np.all(np.isfinite(delta)):
                return u, False
            size = float(np.max(np.abs(delta) / np.where(is_log, 1.0, 1.0 + np.abs(u))))
            fraction = self._damping(u, delta, is_log)
            if fraction < 1.0 or size > 0.25 * last:
                factor = None  # far from the root: a fresh Jacobian next time
            last = size
            u = u + fraction * delta
            scale = 1.0 + np.abs(u[~is_log])
            if np.max(np.abs(delta[is_log])) < NEWTON_TOLERANCE and np.all(
                np.abs(delta[~is_log]) < NEWTON_TOLERANCE * scale
            ):
                return u, True
        return u, False

    def _damping(self, u: np.ndarray, delta: np.ndarray, is_log: np.ndarray) -> float:
        """The fraction of a Newton step to take.

        A logarithm changes by at most :data:`MAX_LOG_STEP`, the potential by at most
        :data:`MAX_POTENTIAL_STEP` times the largest ``T_e`` and an ion velocity by at
        most :data:`MAX_VELOCITY_STEP` times the largest Bohm speed.
        """
        fraction = 1.0
        largest = float(np.max(np.abs(delta[is_log])))
        if largest > MAX_LOG_STEP:
            fraction = MAX_LOG_STEP / largest
        _, _, te, _ = self._unpack(u)
        t_max = float(np.max(te))
        steps = delta.reshape(self.size, self._nvar)
        potential = float(np.max(np.abs(steps[:, -1])))
        if potential > MAX_POTENTIAL_STEP * t_max:
            fraction = min(fraction, MAX_POTENTIAL_STEP * t_max / potential)
        for name, columns in self._velocity_columns.items():
            mass = self._by_name[name].species.mass
            bohm = np.sqrt(E * t_max / mass)
            velocity = float(np.max(np.abs(steps[:, list(columns)])))
            if velocity > MAX_VELOCITY_STEP * bohm:
                fraction = min(fraction, MAX_VELOCITY_STEP * bohm / velocity)
        return fraction

    @staticmethod
    def _factorized(matrix: sp.csc_matrix):
        """The sparse LU of a Jacobian, or ``None`` if it is singular or not finite."""
        matrix = sp.csc_matrix(matrix)
        if not np.all(np.isfinite(matrix.data)):
            return None
        try:
            return spla.splu(matrix)
        except RuntimeError:
            return None

    # --- running -----------------------------------------------------------------------

    def step(self, unknowns: np.ndarray, dt: float) -> tuple[np.ndarray, bool]:
        """One backward-Euler step of ``dt`` seconds; return the new unknowns and success."""
        return self._newton(unknowns, unknowns, dt)

    def advance(self, unknowns: np.ndarray, duration: float, dt: float) -> np.ndarray:
        """Step ``duration`` seconds with steps of at most ``dt``, halving failed ones."""
        time = 0.0
        u = unknowns
        while time < duration * (1.0 - 1e-12):
            current = min(dt, duration - time)
            new, ok = self.step(u, current)
            if ok:
                u, time = new, time + current
            else:
                dt = 0.5 * current
                if dt < 1e-6 * duration:
                    raise RuntimeError("the time step fell below 1e-6 of the duration")
        return u

    @profile("wellbalanced.Fluid2D.steady_state", tags=_PROFILE_TAGS)
    def steady_state(
        self,
        unknowns: np.ndarray,
        *,
        dt: float = 1.0e-9,
        growth: float = 2.0,
        max_dt: float = 1.0,
        max_steps: int = 400,
    ) -> Fluid2DState:
        """Return the steady state.

        Steps in pseudo-time from ``unknowns`` with a step growing from ``dt`` up to
        ``max_dt`` (by ``growth``, or its square after a step of at most four Newton
        iterations), then solves the steady equations from there. A failed step is
        retried with a quarter of it; below a millionth of the first step the search
        stops and the state is returned with ``converged=False``.
        """
        u = unknowns
        smallest = 1.0e-6 * dt
        for _ in range(max_steps):
            new, ok = self.step(u, dt)
            if not ok:
                dt *= 0.25
                if dt < smallest:
                    return self.state(u, time=np.inf, converged=False)
                continue
            u = new
            if dt >= max_dt:
                break
            # an easy step (a few iterations): grow faster
            dt *= growth**2 if self._iterations <= 4 else growth
        steady, ok = self._newton(u, None, np.inf)
        if ok:
            u = steady
        return self.state(u, time=np.inf, converged=ok)

    def state(
        self, unknowns: np.ndarray, time: float = 0.0, converged: bool = True
    ) -> Fluid2DState:
        """The :class:`Fluid2DState` of a vector of unknowns."""
        densities, velocities, te, phi = self._unpack(unknowns)
        rates = self._reactions(densities, te)[0]
        return Fluid2DState(
            grid=self.grid,
            plasma=self.plasma,
            unknowns=np.array(unknowns, copy=True),
            densities={s.name: self.on_grid(densities[s.name]) for s in self.species},
            velocities={
                name: (self.on_grid(v[0]), self.on_grid(v[1]))
                for name, v in velocities.items()
            },
            electron_temperature=self.on_grid(te),
            potential=self.on_grid(phi),
            reaction_rates={name: self.on_grid(r) for name, r in rates.items()},
            time=time,
            converged=converged,
        )

    # --- the field ---------------------------------------------------------------------

    def set_field(
        self, solution: InductiveSolution, *, lorentz: bool | float = True
    ) -> None:
        """Heat the electrons with a field's ``1/2 Re(J E*)`` and push them with its force.

        ``solution`` is on the same grid (an :class:`InductiveField2D` of this grid,
        scaled to its coil current); ``lorentz`` scales the force (``False``: none).
        """
        if solution.grid.size != self.grid.size:
            raise ValueError("the field and the fluid must be on the same grid")
        self.heating = self._on_cells(np.asarray(solution.power_density))
        factor = float(lorentz)
        force = solution.lorentz_force
        self.electron_force = (
            factor * self._on_cells(force[0]),
            factor * self._on_cells(force[1]),
        )

    def set_gas(
        self,
        densities: Mapping[str, CellField] | None = None,
        temperature: float | None = None,
    ) -> None:
        """Change the fixed species' densities [m^-3] and the gas temperature [eV].

        The gas of a neutral model (:func:`~wellbalanced.spatial.couple_neutrals`):
        ``densities`` replace those of fixed species by name (cell fields).
        """
        for name, value in (densities or {}).items():
            if name not in self._fixed:
                raise ValueError(f"{name!r} is not a fixed species")
            self._fixed[name] = self._on_cells(value)
        if temperature is not None:
            if temperature <= 0.0:
                raise ValueError("the gas temperature must be > 0")
            self.gas_temperature = float(temperature)
        self._rate_cache.clear()  # the gas temperature enters the coefficients

    def collision_frequency(self, state: Fluid2DState) -> np.ndarray:
        """The electrons' momentum-transfer frequency ``nu`` [1/s], zero outside."""
        densities, _, te, _ = self._unpack(state.unknowns)
        return self.on_grid(self._collision_frequency(densities, te))

    def plasma_profile(
        self, state: Fluid2DState, *, viscous: bool = False
    ) -> PlasmaProfile:
        """The plasma of ``state`` for the field: ``n_e``, ``nu`` and (``viscous``) ``T_e``."""
        from wellbalanced.spatial.coupling import PlasmaProfile

        return PlasmaProfile(
            electron_density=state.densities["e"],
            collision_frequency=self.collision_frequency(state),
            electron_temperature=state.electron_temperature if viscous else None,
        )

    # --- balances ----------------------------------------------------------------------

    def _terms(self, state: Fluid2DState):
        densities, velocities, te, phi = self._unpack(state.unknowns)
        transport = self._transport(densities, velocities, te, phi)
        return densities, te, transport, self._reactions(densities, te)

    def balance(self, state: Fluid2DState) -> dict[str, float]:
        """The volume source minus the outflow through the walls of every species [1/s].

        Per metre along ``z`` in a cut; zero at a steady state, and the rate of change
        of the species' total in time. The interior faces cancel exactly.
        """
        _, _, (_, walls, _, _), (_, sources, _, _) = self._terms(state)
        return {
            s.name: float(
                np.sum(sources[s.name] * self.volumes)
                - np.sum(walls[s.name] * self._wall_area)
            )
            for s in self.species
        }

    def energy_balance(self, state: Fluid2DState) -> dict[str, float]:
        """The terms of the electron energy balance, integrated over the plasma [W].

        ``heating`` (the RF power), ``field`` (the work of the field, ``-Gamma_e . E``,
        which drives the ions in the quasi-neutral closure), ``reactions`` (lost in
        collisions, negative) and
        ``walls`` (the energy flux out, negative). Per metre along ``z`` in a cut. Their
        sum is the rate of change of the electrons' energy: zero at a steady state.
        """
        if not self._energy:
            raise ValueError("isothermal electrons have no energy balance")
        _, _, (interior, walls, fields, _), (_, _, loss, _) = self._terms(state)
        joule = self._field_work(interior, walls, fields)
        v = self.volumes
        return {
            "heating": float(np.sum(self.heating * v)),
            "field": E * float(np.sum(joule * v)),
            "reactions": -E * float(np.sum(loss * v)),
            "walls": -E * float(np.sum(walls["energy"] * self._wall_area)),
        }

    def wall_fluxes(self, state: Fluid2DState) -> dict[str, np.ndarray]:
        """The flux of every species out through each wall face [m^-2 s^-1].

        Over the faces of :attr:`wall_faces`; ``"energy"`` the electrons' energy flux
        [eV m^-2 s^-1].
        """
        return self._terms(state)[2][1]

    @property
    def wall_faces(self) -> dict[str, np.ndarray]:
        """The walls' faces: their cell (a grid index), axis, outward sign and area."""
        return {
            "cell": self.cells[self._wall_cell],
            "axis": self._wall_axis,
            "sign": self._wall_sign,
            "area": self._wall_area,
        }
