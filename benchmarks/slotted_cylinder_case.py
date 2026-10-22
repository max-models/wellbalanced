"""The slotted-cylinder benchmark's geometry and material, shared by both solvers.

One eighth (``x, y, z >= 0``) of a perfectly conducting box around a square solenoid
(``J = curl T``, ``T`` along ``z`` in the coil's box) and a conducting tube with four
axial slits (at 0, 90, 180 and 270 degrees, the first two cut by the symmetry planes).
Lengths in metres; plain Python so that the NGSolve environment needs nothing else.
"""

MU_0 = 1.25663706212e-6

CASE = {
    "box": 0.060,  # the box's half-width in x and y
    "box_z": 0.100,  # its half-length
    "coil_half_side": 0.045,  # the square solenoid
    "coil_half_length": 0.080,
    "turns_per_length": 100.0,  # turns per metre, at 1 A
    "inner_radius": 0.030,  # the tube
    "outer_radius": 0.036,
    "thickness": 0.006,
    "tube_half_length": 0.050,
    "slit_width": 0.004,
    "slit_half_length": 0.040,  # "closed" slits stop 10 mm before the ends
    "conductivity": 1.0e6,  # S/m: skin depth 7.1 mm at 5 kHz, 2.25 mm at 50 kHz
    "frequencies": [5.0e3, 5.0e4],
    "regularization": 1.0e-6,
    "maxh_air": 0.012,
}
