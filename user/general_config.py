"""Minimal configuration for the Ixcononly research branch.

This branch studies invariant construction only. There are no magnet-optimizer,
objective-function, Powell/CMA, or automatic-a_box settings.

Polynomial coordinate convention:
    [delta, x, y, px, py]
"""

from pathlib import Path
import numpy as np
import sympy as sp


# Fixed accelerator used by every constructor.
LATTICE_FILE = "lattice_config.py"
ANALYSIS_CELLS = 1
CORRECT_CHROMATICITY = True


# Symbolic phase-space variables.
delta, x, y, px, py = sp.symbols("delta x y px py")
VARIABLES = [delta, x, y, px, py]

# Element/Hamiltonian coefficients.
b1, b2, b3, b4, b5 = sp.symbols("b1 b2 b3 b4 b5")
FIELD_SYMBOLS = [b1, b2, b3, b4, b5]

HAMILTONIAN = (
    sp.Rational(1, 2) * (px**2 + py**2) * (1 - delta + delta**2)
    - b1 * x * delta
    + sp.Rational(1, 2) * b1**2 * x**2
    + sp.Rational(1, 2) * b2 * (x**2 - y**2)
    + sp.Rational(1, 3) * b3 * (x**3 - 3 * x * y**2)
    + sp.Rational(1, 4) * b4 * (x**4 - 6 * x**2 * y**2 + y**4)
)


# Polynomial truncation.
ORDER = 8
DELTA_ORDER = 1
N_PLANES = 2

# Used by the weighted a_box construction.
# a_box_y0 automatically replaces y and py by exactly zero and rebuilds a
# reduced (delta,x,px) basis. Keep this normal 5-D box positive.
A_BOX = np.array([0.01, 10e-3, 8e-3, 1e-3, 0.8e-3], dtype=float)

LEAST_SQUARES_TOL = 1.0e-16

# Map-averaging invariant constructors.
# cesaro: c_N = (1/N) sum_{k=0}^{N-1} T^k Sx
CESARO_TERMS = 64

# abel: c_rho = (1-rho) (I-rho*T)^(-1) Sx
# Larger rho approaches the fixed subspace more strongly but makes the
# resolvent increasingly ill-conditioned as rho -> 1.
ABEL_RHO = 0.98


# Paired physical tracking used by compare(name1, name2).
# Coordinates are [xmin, xmax, ymin, ymax] in millimetres.
TRACKING_COORDS_MM = [-15.0, 15.0, -15.0, 15.0]
TRACKING_STEPS = [121, 121]
TRACKING_TURNS = 512
TRACKING_DELTA = 0.0
TRACKING_NUM_INT_STEPS = 10

# None -> infer the number of cells required for 360 degrees.
TRACKING_PHYSICAL_RING_CELLS = None
TRACKING_POOL_SIZE = None

# Relative-Ix denominator floor.
IX_INVARIANCE_NORM_FLOOR_FRACTION = 1.0e-12


# Output.
SAVE_PLOTS = True
SHOW_PLOTS = False
OUTPUT_DIRECTORY = Path("ix_construction_output")
