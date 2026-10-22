---
title: 3D field discretisation
description: The evidence for step R4's decision, staircased Cartesian edges against analytic and NGSolve references.
---

Step R4 of the RF-driver plan decides how the 3D inductive field is discretised:
Yee-type edges on Cartesian grids (staircased or cut-cell conductors, patchwork
refinement) or Nédélec elements on fitted meshes. This page gives the evidence from a
first 3D solver with staircased conductors, `InductiveField3D`, measured against
closed forms and against NGSolve on fitted meshes, and ends with a recommendation.
The decision is yours. The numbers come from `benchmarks/field3d_r4.py` (wellbalanced)
and `benchmarks/slotted_cylinder_ngsolve.py` (NGSolve, in a separate environment:
it is a benchmark, not a dependency). Both write JSON files next to the scripts.

## The solver

`InductiveField3D` solves the time-harmonic magnetoquasistatic field
`curl (nu curl A) + i omega sigma A = J_coil` (temporal gauge, `E = -i omega A`) with
`A` on the edges of a uniform Cartesian grid (`EdgeGrid`, finite integration). The curl
and the gradient are incidence matrices, so `C G = 0` holds exactly. Coils are
`J = curl T`, divergence-free on the grid for any shape. A conductor is staircased: an
edge conducts when its midpoint lies inside the simgeometry shape (Booleans give the
slits). `samples=3` instead averages each edge's dual cell over sub-points, in series
along the edge and in parallel across it. Boundaries are `A x n = 0`, `H x n = 0`
(symmetry planes) or periodic. The gauge is a small real regularisation
`eps (nu / l^2) M_1`: on the gradients it is a weak Coulomb gauge. It adds no loss and
shifts the results by `O(eps)` (tested: `1e-6` gives relative changes below `1e-5`). The
solve is one SuperLU factorisation in a geometric nested-dissection order.

## (1) A long solenoid's inductance

- **Periodic solenoid in a conducting box:** exact to `2e-8` for the staircased
  cross-section the grid sees, `L' = mu0 n^2 A_s (1 - A_s / A_box)`. Against the true
  circle the error is the staircased area: −15 %, +1.4 % and +0.4 % at 10, 20 and 40
  cells across the box (in the tests).
- **Finite solenoid against Nagaoka's coefficient:** length 0.21 m, radius 0.05 m,
  `h = 1 cm`, one eighth of the domain. A conducting box underestimates the free-space
  inductance and a magnetic box overestimates it:

| box half-width | conducting walls | magnetic walls |
| --- | --- | --- |
| 2 radii | −14.0 % | +4.3 % |
| 3 radii | −4.7 % | +1.9 % |
| 4 radii | −2.0 % | +0.9 % |
| 5 radii | −1.0 % | +0.6 % |

The two bounds close in on Nagaoka's value, evaluated for the staircased coil the grid
sees.

## (3) The 2D result for a long, z-invariant driver

- **z-periodic driver:** a rectangular loop around a cold-plasma column
  (`sigma = 128 − 40i S/m`). The z-edges reproduce `InductiveField2D` on the matching
  cut to `3e-7` in `A_z`, and the loss and impedance to `4e-8` (the gauge's `eps`; in
  the tests).
- **Finite loop of growing length:** the same loop with its end segments, against the
  z-periodic result.

| loop half-length | loss per metre, middle 10 cm | total loss / length |
| --- | --- | --- |
| 0.105 m | +54 % | +49 % |
| 0.205 m | +12 % | +26 % |
| 0.405 m | +1.5 % | +13 % |
| 0.805 m | +0.6 % | +6.8 % |

The middle section converges to the 2D field. The total keeps a `1/l` term from the end
turns.

## (2) Eddy losses of a slotted cylinder

**The geometry:**

- A tube, radii 30 and 36 mm (6 mm wall), 100 mm long, with four axial slits 4 mm wide.
- It sits inside a square solenoid (90 mm side, 160 mm long, 100 turns per metre) in a
  conducting box 120 × 120 × 200 mm.
- The material is σ = 1 MS/m at 5 kHz (skin depth 7.1 mm, larger than the wall) and at
  50 kHz (2.25 mm, a third of the wall).
- Three slit variants: no slits, `closed` slits that end 10 mm before the tube's ends
  (as in a Faraday shield), and `through` slits that cut the tube into four.

**The references:**

- NGSolve, Nédélec elements of order 3 on curved tetrahedra fitted to the slits, one
  eighth, 0.75 M degrees of freedom. Two meshes agree to 0.12 %.
- In 2D, the Bessel solution (no slits; NGSolve agrees with it to `1e-7`) and NGSolve
  (through slits).

A caveat about the reference: NGSolve's `nograds=True`, the usual setting for
magnetostatics, removes the gradient part that `E` has in a conductor. With it, the
losses came out up to 4 % off and drifted with the mesh. The reference keeps the full
space.

### The slits in the infinite tube (z-periodic, a quarter, staircased)

Relative error of the loss, staircased edges (`samples=1`); in brackets, the sub-cell
average (`samples=3`):

| h | no slits, 5 kHz | no slits, 50 kHz | through slits, 5 kHz | through slits, 50 kHz |
| --- | --- | --- | --- | --- |
| 2.5 mm | +1.8 % (+7.7) | −19.5 % (−10.0) | −30.2 % (−37.5) | −29.1 % (−23.6) |
| 1.25 mm | +1.2 % (+3.4) | −5.5 % (−4.0) | −13.3 % (−20.0) | −8.4 % (−7.6) |
| 0.625 mm | +0.6 % (+1.7) | −1.5 % (−1.4) | −7.7 % (−12.3) | −3.8 % (−4.4) |
| 0.3125 mm | +0.25 % (+0.8) | −0.3 % (−0.5) | −2.6 % (−5.5) | −1.1 % (−1.6) |

- The slits change the loss by a factor 13 at 5 kHz (0.0617 to 0.00465 W/m at 1 A)
  and by +7 % at 50 kHz.
- The inductance is within 1.6 % at every h.
- The loss converges at about first order in `h`. The slit loss at 5 kHz, a small eddy
  current in each segment, needs `h = 0.3 mm`, a twentieth of the wall and a
  thirteenth of the slit, for 3 %.
- The sub-cell average does not help. A diagonal series and parallel conductance
  misses the off-diagonal coupling of a wall at 45° to the grid.

### The finite tube in 3D (one eighth, staircased)

Relative error of the loss against NGSolve. The references are 0.964, 2.831, 0.964,
4.279, 0.0568 and 2.281 mW for the eighth at 1 A, in the order of the columns.

| h | unknowns | none, 5 kHz | none, 50 kHz | closed, 5 kHz | closed, 50 kHz | through, 5 kHz | through, 50 kHz |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 mm | 7 700 | −9.6 % | −44 % | −9.6 % | −63 % | +1435 % | −31 % |
| 3.33 mm | 27 030 | −19 % | −44 % | −25 % | −45 % | −45 % | −56 % |
| 2.5 mm | 65 320 | +2.2 % | −18 % | −0.6 % | −18 % | −27 % | −28 % |
| 2 mm | 129 050 | −13 % | −24 % | −18 % | −27 % | −26 % | −31 % |

- **The slit effect:** in NGSolve, closed slits barely change the loss at 5 kHz
  (−0.02 %) and raise it by 51 % at 50 kHz. Through slits cut it by a factor 17 at
  5 kHz and by 19 % at 50 kHz. The inductance rises by 10 % (closed) and 19 %
  (through) at 5 kHz.
- **At h = 5 mm the slits are invisible.** No edge midpoint falls in the 4 mm slits, so
  every variant gives the unslotted loss (hence +1435 % for through slits).
- **On the affordable grids (h ≥ t/3) the error is 0.6–56 % and not monotone.** It
  depends on how the wall's staircase falls on the grid, so no grid here gives the
  slit effect better than about 20 %. The inductance is within 6 % from h = 3.33 mm.

### An unresolved skin layer (infinite tube, no slits)

Copper at 1 MHz has a skin depth of 66 µm. With δ = 0.71 mm (500 kHz) the loss is
−36 %, −11 % and −2.2 % at h = 1.25, 0.625 and 0.3125 mm. With δ = 0.225 mm (5 MHz) it
is −78 %, −56 % and −24 %. A volume-resolved metal needs `h ≲ δ / 2`, so the shield of a
real driver needs a surface impedance whatever the grid.

## What it costs

- **Yee grid:** the direct solve takes 40–60 s for 65 000 unknowns (24 × 24 × 40 cells)
  and about 2 min for 129 000 (h = 2 mm, a third of the wall). The cost grows as the
  square of the unknowns, so the 3D convergence study of the slotted cylinder stops
  there.
- **NGSolve:** 0.29 M degrees of freedom (order 3, mesh 3 mm in the tube) give the loss
  to 0.12 % in 15–45 s, with BDDC and conjugate gradients.

## Recommendation (evidence for your decision)

**Plain staircasing is not accurate enough for the slit losses at affordable grids.**

- It converges, at first order and not monotonically at coarse h.
- The z-invariant slit loss needs h ≈ t/20 ≈ w/13 for 3 % (t the wall, w the slit).
- A slit narrower than about one cell disappears unless an edge midpoint falls inside
  it. At h = 5 mm the 4 mm slits were not seen at all: `closed` gave the unslotted
  result.
- A real Faraday shield adds a skin depth far below any grid (surface impedance
  needed) and slits of about 1 mm.

**The leaning to Cartesian edges still holds for the plasma, the coil and the air**,
with two additions:

1. **Conformal conductors (cut cells):** edge lengths and dual areas cut by the
   conductor's surface (simgeometry's level sets and cut edges give them), with a
   surface-impedance condition on the cut faces for the metal. This restores
   second-order convergence for curved walls and keeps slits narrower than a cell. The
   cheap alternative tried here, sub-cell averages of the conductivity, does not do it.
2. **Local refinement around the shield and the slits** (patchwork patches of edges).
   The air and the plasma do not need 0.3 mm. They need about the skin depth of the
   plasma (millimetres).

Without both, a uniform staircased grid at shield resolution costs about `10^8` cells
for a driver. This rules out direct solvers and needs an iterative solver with an
auxiliary-space preconditioner on mpiarray and cunumpy.

If R_net against 0.6 Ω (BUG) and 1.28 Ω (ELISE) is needed before cut cells and edge
patches exist, Nédélec elements on a fitted mesh are the faster route for the
conductors. NGSolve reached 0.12 % here at a few hundred thousand degrees of freedom.
The plasma could stay on the Cartesian grid, coupled through the coil's impedance.
Struphy's mapped domains fit the driver's cylinder, but the coil, the EMS and the slits
need multipatch domains there.

## What is left

- Cut-cell (conformal) edges and a surface impedance for the metal, and the same slit
  test with them.
- Edge operators in patchwork. Patchwork's operators today are cell-centred and 2D,
  so they cannot refine the 3D field.
- An iterative solver (auxiliary-space preconditioned) for grids beyond about 10^5
  edges.
- The helical coil, the Faraday shield of BUG and ELISE, and R_net.
