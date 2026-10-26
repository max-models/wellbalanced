---
title: 3D field discretisation
description: The evidence for step R4's decision, staircased and cut-cell Cartesian edges against analytic and NGSolve references.
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

*Note from the second round:* two discretisation errors of the coil, not of the tube,
are in this table. On the 3.33 and 2 mm grids the coil's 45 mm half-side is not on the
grid, and its staircased cross-section changes the field by up to 20 %. And a coil's `T`
was sampled at the face centres, so the faces in the plane of its end (80 mm, on the
grid) counted fully and the coil was half a cell too long (inductance +3.1 % at 5 mm,
+1.6 % at 2.5 mm for the coil alone); `T` is now averaged over the face's dual edge, and
the coil alone converges at second order (−0.47 %, −0.14 %, −0.07 % at 5, 2.5 and
1.67 mm against NGSolve).

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

## Cut cells with a surface impedance

The second round of R4 adds the two things the recommendation below asked for in the
conductors: **cut-cell (conformal) geometry** and a **surface impedance**, and an
iterative solver. The numbers come from `benchmarks/field3d_r4_cutcell.py` and
`benchmarks/slotted_cylinder_ngsolve_sibc.py`.

**The model.** An `ImpedanceConductor3D` drops out of the field: the edges entirely
inside it are not unknowns, and on its surface Leontovich's condition
`E_t = Z_s H x n` holds, `Z_s = (1 + i) / (sigma delta)`. This is right where the skin
depth is small against the wall and its radii of curvature: copper at MHz, the Faraday
shield, the coil.

**The discretisation** (Dey and Mittra's conformal faces, here in the frequency
domain). A cut face's flux `b_f` is the line integral of `A` around its part outside
the metal: the unknown edges' parts outside plus the stretch along the surface, where
`A_t = (i / omega) Z_s H x n`. With `H` tangent to a good conductor that stretch is
`(i Z_s / omega) H_f w_f`, `w_f` the surface's width in the face (its trace divided by
the sine between the surface and the face's normal). So
`H_f = b_f / (mu0 A_f^air - i Z_s w_f / omega)`: only the face reluctances change, the
curl stays the incidence matrix. The losses are `1/2 Re(Z_s) |H_f|^2 w_f L*_f`, equal to
the coil's power to round-off. The geometry comes from simgeometry's new
`yee_fractions` (simgeometry#26): the edge-length and face-area fractions and the traces,
exact for planes, second order for curved surfaces, slits narrower than a cell kept,
walls on grid lines treated like a staircase. `cut_cells=False` staircases the same
model (the metal is the cells whose centre is inside), for comparison.

**Closed forms (tests).** In a z-invariant solenoid `H_z` is uniform in each connected
region of air, which gives the loss of any conductor from its perimeter and area. A
tilted plane wall: exact (to `1e-5`, the gauge). The tube without slits: −0.009 %,
−0.004 %, −0.002 % and −0.001 % at h = 2.5, 1.67, 1.25 and 0.625 mm (second order);
staircased +22 % to +28 % on every grid. A staircase is `4 / pi` too long for a circle,
and the loss is proportional to the surface: it never converges.

### The infinite tube (z-periodic quarter)

Relative error of the loss against NGSolve with the same surface impedance (order 4 on
curved meshes; three meshes agree to `1e-7`). Skin depths 2.25, 0.71 and 0.225 mm in the
6 mm wall; the slits 4 mm.

| variant | 5 mm | 2.5 mm | 1.67 mm | 1.25 mm | 0.625 mm |
| --- | --- | --- | --- | --- | --- |
| none, cut cells (all three frequencies) | +10.6 to +12.8 % | −0.009 % | −0.004 % | −0.002 % | −0.001 % |
| none, staircase | +23 to +24 % | +22 to +23 % | +28 to +30 % | +27 to +28 % | +26 to +28 % |
| through slits, 50 kHz, cut cells | −0.51 % | −0.07 % | +0.22 % | −0.06 % | −0.06 % |
| through slits, 500 kHz, cut cells | −0.53 % | −0.07 % | +0.24 % | −0.07 % | −0.07 % |
| through slits, 5 MHz, cut cells | −0.54 % | −0.07 % | +0.24 % | −0.07 % | −0.07 % |
| through slits, staircase | +8 to +9 % | +19 to +21 % | +27 to +30 % | +23 to +25 % | +23 to +25 % |
| 500 kHz, volume conductor (staircased, `field3d_r4.py`) | −83 / −85 % | −66 / −67 % | −50 / −53 % | −37 / −39 % | −12 / −15 % |

- At h = 5 mm (a cell close to the wall, 6 mm) an edge can reach through the wall and
  couple the bore to the outside: +11 % without slits. From h = t/2.4 on the cut cells
  are within 0.25 %.
- With slits the cut cells settle at −0.07 % of NGSolve, independent of the frequency;
  the remainder is at the slit's corners, where both the field and the trace have
  corners. The inductance is within 0.01 % from h = 2.5 mm.
- A volume conductor at 500 kHz is 12–15 % low even at h = 0.625 mm (δ/h = 1.1).

**The surface impedance's own error** (against NGSolve's volume conductor, boundary-layer
meshes resolving the skin, two meshes agree to `2e-6`):

| | 50 kHz (δ/t = 0.38) | 500 kHz (0.12) | 5 MHz (0.04) |
| --- | --- | --- | --- |
| no slits | +3.7 % | +1.0 % | +0.31 % |
| through slits | +9.2 % | +3.5 % | +1.0 % |

It falls as `δ`: the curvature (`δ / 2R`) without slits, and the slits' corners with
them. At 5 kHz (δ = 7.1 mm, more than the wall) the surface impedance does not apply:
the original 5 kHz and 50 kHz tests of the first round stay volume problems.

### The finite tube in 3D (one eighth)

Loss against NGSolve's surface impedance on fitted meshes, `tube3d` of
`benchmarks/field3d_r4_cutcell.py` (iterative solver from 24 cells on):

| 500 kHz | h = 5 mm | 2.5 mm | 1.67 mm | 1.25 mm |
| --- | --- | --- | --- | --- |
| no slits, cut cells | −3.3 % | +1.1 % | +0.95 % | +0.91 % |
| no slits, staircase | +15 % | +16 % | +23 % | +22 % |
| closed slits, cut cells | −33 % | −5.4 % | −1.2 % | −3.1 % |
| closed slits, staircase | −21 % | +0.4 % | +12 % | +6.9 % |
| through slits, cut cells | +5.2 % | +2.4 % | +2.0 % | +1.2 % |
| through slits, staircase | +24 % | +19 % | +28 % | +23 % |

- The cut cells converge to about 1 % without and with through slits, at first order or
  slower: the remainder is at the tube's and the slits' ends, the 3D corners.
- Closed slits are not monotone (−1.2 % at 1.67 mm, −3.1 % at 1.25 mm): the slit's closed
  ends fall differently on each grid.
- The staircase stays 20–30 % off and does not converge.
- 50 kHz gives the same picture (cut cells +0.7 %, +1.2 % and −2.3 % at 1.25 mm). The
  inductance is within 0.2 % from h = 2.5 mm with cut cells.
- On a quiet machine the iterative solve takes 25–50 iterations and 3–11 s for
  2·10^5 to 5·10^5 unknowns (two threads).

## What it costs

- **Yee grid, direct:** the direct solve takes 40–60 s for 65 000 unknowns (24 × 24 × 40
  cells) and about 2 min for 129 000 (h = 2 mm, a third of the wall), on a quiet machine;
  320 s for 62 000 on a loaded one. The cost grows as the square of the unknowns.
- **Yee grid, iterative** (`solve(solver="iterative")`, second round): COCG with Hiptmair
  and Xu's auxiliary-space preconditioner, the two nodal problems by a geometric
  multigrid V-cycle on the grid's nodes (no algebraic multigrid needed on a Cartesian
  grid), to a relative residual of `1e-8`. 97 iterations at 7 500 unknowns, 38 at 62 000,
  44 at 216 000: 26 s instead of 320 s at 62 000, 2–3 min at 216 000 (a loaded machine,
  two threads). Cut cells' small faces do not slow it down. Agrees with the direct solve
  to `1e-8`.
- **NGSolve:** 0.29 M degrees of freedom (order 3, mesh 3 mm in the tube) give the loss
  to 0.12 % in 15–45 s, with BDDC and conjugate gradients.

## Recommendation (evidence for your decision)

*Second round (cut cells):* for metal with a skin depth below the grid, Cartesian edges
with cut-cell surface-impedance conductors are accurate: 0.01 % on the tube and 0.07 %
with slits narrower than a cell in z-invariant geometry, second order, where the
staircase stays 20–30 % off on every grid; 0.2 % for a sphere in 3D. The 3D corners
(the tube's and the slits' ends) converge at first order, 1–5 % at h = 2.5 mm, so the
Faraday shield's slit ends need the corner treatment above (or local refinement) before
R_net is trusted to a per cent. With the iterative solver the grids of the slotted
cylinder (2·10^5 unknowns) take minutes. The first round's text follows.


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

- Done in the second round: cut-cell surface-impedance conductors, the slit test with
  them, and the iterative solver.
- The 3D corners: the cut cells use only each face's normal `H`, which is second order
  on smooth and z-invariant surfaces but first order, and 1–5 % at h = 2.5 mm, at the
  tube's ends and the slits' ends. A corner-aware weight (or the tangential `H` of the
  neighbouring faces in the surface term) is the next step.
- Volume conductors with cut cells, for metal whose skin depth is comparable to its
  thickness (the 5 kHz case): the sub-cell conductance of the first round does not do
  it.
- Edge operators in patchwork. Patchwork's operators today are cell-centred and 2D,
  so they cannot refine the 3D field.
- The iterative solver on mpiarray and cunumpy (it is all sparse products and a
  multigrid on a Cartesian grid).
- The helical coil, the Faraday shield of BUG and ELISE, and R_net.
