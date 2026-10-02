"""Research configuration for invariant-construction experiments."""

from pathlib import Path

import numpy as np
import sympy as sp


# Lattice ---------------------------------------------------------------------
LATTICE_FILE = "lattice_config.py"
ANALYSIS_CELLS = 1
CORRECT_CHROMATICITY = True


# Polynomial model ------------------------------------------------------------
delta, x, y, px, py = sp.symbols("delta x y px py")
VARIABLES = [delta, x, y, px, py]

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

ORDER = 8
DELTA_ORDER = 1
N_PLANES = 2
LEAST_SQUARES_TOL = 1.0e-16


# Representation --------------------------------------------------------------
# Every method uses the same full [delta,x,y,px,py] monomial index set.
#
# C controls the scaled polynomial basis e_i = C_i z^alpha_i.
# G is built after C.
#
# C may be:
#   "a_box"      -> original box normalization
#   "fischer"    -> factorial scaling
#   [cδ,cx,cy,cpx,cpy] -> coordinate scaling lifted to all monomials
#   one number per monomial
#   a callable(idx_to_vec) -> full C
#
# G may be:
#   "box"        -> symmetric-box L2 metric using C and A_BOX
#   "coefficient"-> Euclidean metric in the C-scaled basis
#   "fischer"    -> Fischer metric for the current C
#   a matrix or callable(idx_to_vec, C)
A_BOX = np.array([0.01, 10e-3, 8e-3, 1e-3, 0.8e-3], dtype=float)

INVARIANT_OPTIONS = {
    "a_box": {
        "C": "a_box",
        "G": "box",
    },
    "a_box_y0": {
        "C": "a_box",
        "G": "box",
        "horizontal_only": True,
    },
    "hybrid": {
        "C": np.ones(5),
        "G": "coefficient",
    },
    "eigen": {
        "C": np.ones(5),
        "G": "coefficient",
    },
    "graded_ls": {
        "C": "fischer",
        "G": "coefficient",
    },
    "cesaro": {
        "C": np.ones(5),
        "G": "coefficient",
    },
    "abel": {
        "C": np.ones(5),
        "G": "coefficient",
    },
}

CESARO_TERMS = 64
ABEL_RHO = 0.98


# Tracking --------------------------------------------------------------------
TRACKING_COORDS_MM = [-15.0, 15.0, -15.0, 15.0]
TRACKING_STEPS = [121, 121]
TRACKING_TURNS = 512
TRACKING_DELTA = 0.0
TRACKING_NUM_INT_STEPS = 10
TRACKING_PHYSICAL_RING_CELLS = None
TRACKING_POOL_SIZE = None

# Raw turn-by-turn trajectories and FMA values are kept here and reused.
TRACKING_CACHE = Path("ix_construction_output/tracking_cache")

IX_INVARIANCE_NORM_FLOOR_FRACTION = 1.0e-12


# Output ----------------------------------------------------------------------
OUTPUT_DIRECTORY = Path("ix_construction_output")
SAVE_PLOTS = True
SHOW_PLOTS = False
