"""An iterative solver for the edge field: COCG with an auxiliary-space preconditioner.

For the complex symmetric systems ``A a = j`` of
:class:`~wellbalanced.spatial.inductive3d.InductiveField3D` on grids where a sparse LU is
too slow (its cost grows as the square of the unknowns). The preconditioner is Hiptmair
and Xu's auxiliary-space one (SIAM J. Numer. Anal. 45, 2483, 2007; hypre's AMS, Kolev and
Vassilevski, J. Comput. Math. 27, 604, 2009) in its additive form,

    B = S^-1 + G (G^T A_r G)^-1 G^T + P (P^T A_r P)^-1 P^T,

with ``S`` an l1-Jacobi smoother of ``A``, ``G`` the gradient of the nodes inside the
unknowns' region, ``P`` the interpolation of nodal vector fields to the edges
(``(P u)_e = l_e t_e . (u_i + u_j) / 2``, ``l_e`` the edge's length outside the impedance
conductors) and ``A_r`` a real surrogate of ``A`` (the absolute values of the face and
edge coefficients). The two nodal problems are solved approximately by one V-cycle of a
geometric multigrid on the grid's nodes (trilinear interpolation, Galerkin coarse
operators, l1-Jacobi smoothing, a direct solve on the coarsest level): the grid is a
uniform Cartesian one, so no algebraic multigrid is needed. ``B`` is real symmetric, so
the conjugate orthogonal conjugate gradients (COCG, van der Vorst and Melissen, IEEE
Trans. Magn. 26, 706, 1990), the complex-symmetric form of CG, applies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from wellbalanced.spatial.edges import EdgeGrid

__all__ = ["AuxiliarySpacePreconditioner", "IterativeResult", "cocg"]


def _prolongation_1d(n: int, periodic: bool) -> sp.csr_matrix:
    """Linear interpolation from every other node to all ``n`` nodes of an axis."""
    if n <= 3 or (periodic and n % 2):
        return sp.identity(n, format="csr")
    coarse = n // 2 if periodic else (n + 1) // 2
    rows, cols, values = [], [], []
    for i in range(n):
        if i % 2 == 0:
            rows.append(i)
            cols.append(i // 2)
            values.append(1.0)
        else:
            left, right = (i - 1) // 2, (i + 1) // 2
            if periodic:
                right %= coarse
            if right >= coarse:  # the last fine node of an even count
                rows.append(i)
                cols.append(left)
                values.append(1.0)
                continue
            rows += [i, i]
            cols += [left, right]
            values += [0.5, 0.5]
    return sp.csr_matrix((values, (rows, cols)), shape=(n, coarse))


def _l1_diagonal(matrix: sp.spmatrix) -> np.ndarray:
    return np.asarray(abs(matrix).sum(axis=1)).ravel()


class _Multigrid:
    """A geometric V-cycle for a real SPD operator on (a subset of) a node lattice.

    ``components`` copies of the lattice (a nodal vector field), each with the same
    ``active`` nodes, numbered component by component.
    """

    def __init__(
        self,
        matrix: sp.spmatrix,
        lattice: Sequence[int],
        periodic: Sequence[bool],
        active: np.ndarray,
        components: int = 1,
        coarsest: int = 3000,
        sweeps: int = 2,
    ) -> None:
        self.sweeps = sweeps
        self.levels: list[tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix | None]] = []
        matrix = sp.csr_matrix(matrix)
        lattice = tuple(int(n) for n in lattice)
        while True:
            diagonal = _l1_diagonal(matrix)
            diagonal[diagonal == 0] = 1.0
            if matrix.shape[0] <= coarsest:
                break
            factors = [
                _prolongation_1d(n, p) for n, p in zip(lattice, periodic, strict=True)
            ]
            coarse_lattice = tuple(f.shape[1] for f in factors)
            if coarse_lattice == lattice:
                break
            full = sp.kron(sp.kron(factors[0], factors[1]), factors[2], format="csr")
            block = full[active]
            coarse_active = np.flatnonzero(np.asarray(abs(block).sum(axis=0)).ravel())
            block = block[:, coarse_active]
            prolongation = sp.block_diag([block] * components, format="csr")
            self.levels.append((matrix, 1.0 / diagonal, prolongation))
            matrix = sp.csr_matrix(prolongation.T @ matrix @ prolongation)
            lattice, active = coarse_lattice, coarse_active
        self.levels.append((matrix, 1.0 / diagonal, None))
        self.coarse = spla.splu(sp.csc_matrix(matrix))

    def _cycle(self, level: int, rhs: np.ndarray) -> np.ndarray:
        matrix, inverse, prolongation = self.levels[level]
        if prolongation is None:
            return self.coarse.solve(rhs)
        x = np.zeros_like(rhs)
        for _ in range(self.sweeps):
            x += inverse * (rhs - matrix @ x)
        correction = self._cycle(level + 1, prolongation.T @ (rhs - matrix @ x))
        x += prolongation @ correction
        for _ in range(self.sweeps):
            x += inverse * (rhs - matrix @ x)
        return x

    def __call__(self, rhs: np.ndarray) -> np.ndarray:
        """One V-cycle from zero, for the real and imaginary parts."""
        if np.iscomplexobj(rhs):
            return self._cycle(0, rhs.real) + 1j * self._cycle(0, rhs.imag)
        return self._cycle(0, rhs)


class AuxiliarySpacePreconditioner:
    """Hiptmair and Xu's additive auxiliary-space preconditioner on an edge grid.

    Args:
        grid: The edge grid.
        matrix: ``A`` on the unknowns (complex symmetric).
        unknowns: Which edges are unknowns (a mask over all edges).
        face_weights: ``|kappa_f|`` of ``C^T diag(kappa) C``, per face.
        edge_weights: ``|m_e|`` of the edge term ``diag(m)``, per edge.
        edge_lengths: The edges' lengths in ``P`` (outside the impedance conductors).
    """

    def __init__(
        self,
        grid: EdgeGrid,
        matrix: sp.spmatrix,
        unknowns: np.ndarray,
        face_weights: np.ndarray,
        edge_weights: np.ndarray,
        edge_lengths: np.ndarray,
    ) -> None:
        """Build the smoother, the auxiliary operators and their multigrids."""
        self.matrix = sp.csr_matrix(matrix)
        inverse = _l1_diagonal(self.matrix)
        self.smoother = 1.0 / np.where(inverse > 0, inverse, 1.0)
        curl = grid.curl()
        real = (
            curl.T @ sp.diags(np.abs(face_weights)) @ curl
            + sp.diags(np.abs(edge_weights))
        ).tocsr()[unknowns][:, unknowns]
        gradient = grid.gradient().tocsr()
        # the nodes all of whose edges are unknowns: their gradients are in the space
        blocked = np.asarray(abs(gradient[~unknowns]).sum(axis=0)).ravel() > 0
        interior = np.flatnonzero(~blocked)
        self.gradient = gradient[unknowns][:, interior]
        touched = np.flatnonzero(
            np.asarray(abs(gradient[unknowns]).sum(axis=0)).ravel() > 0
        )
        axes = grid.edge_axes()
        half = 0.5 * abs(gradient)
        blocks = [
            sp.diags(np.where(axes == k, edge_lengths, 0.0)) @ half for k in range(3)
        ]
        self.interpolation = sp.hstack(
            [b.tocsr()[unknowns][:, touched] for b in blocks], format="csr"
        )
        periodic = grid.periodic
        self.nodal = _Multigrid(
            self.gradient.T @ real @ self.gradient, grid.nodes, periodic, interior
        )
        self.vector = _Multigrid(
            self.interpolation.T @ real @ self.interpolation,
            grid.nodes,
            periodic,
            touched,
            components=3,
        )

    def __call__(self, residual: np.ndarray) -> np.ndarray:
        """``B r``."""
        result = self.smoother * residual
        result += self.gradient @ self.nodal(self.gradient.T @ residual)
        result += self.interpolation @ self.vector(self.interpolation.T @ residual)
        return result


@dataclass(frozen=True)
class IterativeResult:
    """The outcome of :func:`cocg`.

    Attributes:
        solution: The solution.
        iterations: The iterations taken.
        residual: The final ``|b - A x| / |b|``.
        converged: Whether ``residual <= tolerance``.
    """

    solution: np.ndarray
    iterations: int
    residual: float
    converged: bool


def cocg(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    preconditioner,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 2000,
) -> IterativeResult:
    """COCG for a complex symmetric ``matrix`` with a symmetric ``preconditioner``.

    CG with the bilinear ``x^T y`` in place of ``x^H y``; the relative residual
    ``|b - A x| / |b|`` (Euclidean) stops it.
    """
    b = np.asarray(rhs, dtype=complex)
    norm = float(np.linalg.norm(b))
    x = np.zeros_like(b)
    if norm == 0.0:
        return IterativeResult(x, 0, 0.0, True)
    r = b.copy()
    z = preconditioner(r)
    p = z.copy()
    rho = r @ z
    residual = 1.0
    for iteration in range(1, max_iterations + 1):
        q = matrix @ p
        alpha = rho / (p @ q)
        x += alpha * p
        r -= alpha * q
        residual = float(np.linalg.norm(r)) / norm
        if residual <= tolerance:
            return IterativeResult(x, iteration, residual, True)
        z = preconditioner(r)
        rho, rho_old = r @ z, rho
        p = z + (rho / rho_old) * p
    return IterativeResult(x, max_iterations, residual, False)
