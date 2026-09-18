"""General user-editable settings for nonlinear analysis and optimization.

Choose the lattice Python file below.  The lattice file may have any filename;
its path is resolved relative to this general_config.py file.
"""
from pathlib import Path
import numpy as np
import sympy as sp

# 1. LATTICE TO USE
# The lattice file can be renamed or placed in a subfolder.
# Examples:
# LATTICE_FILE = "lattice_config.py"
# LATTICE_FILE = "lattices/my_storage_ring.py"
LATTICE_FILE = "lattice_config.py"

# Number of configured cells used to build the nonlinear/invariant map.
# This is NOT the physical full-ring cell count used by FMA / Ix tracking.
ANALYSIS_CELLS = 1

# 2. VARIABLES TO OPTIMIZE
VARY = [     #magnets to edit names.
    "kse1", "kfd2", "kfd3", "ks1", "ks2", "ksd3",
    "ks1s", "ks2s", "ko1", "ko2", "ko3",
]

# 3. SYMBOLIC PHASE-SPACE VARIABLES
# Keep this order unless nonlinear.py is generalized beyond the current model:
#     [delta, x, y, px, py]
delta, x, y, px, py = sp.symbols("delta x y px py")
VARIABLES = [delta, x, y, px, py]

# 4. SYMBOLIC ELEMENT / HAMILTONIAN COEFFICIENTS
#     b1 = curvature; b2 = quadrupole strength K; b3 = sextupole strength S; b4 = nonlinear multipole strength O; b5 = reserved fifth coefficient (currently 0 for every lattice element)
b1, b2, b3, b4, b5 = sp.symbols("b1 b2 b3 b4 b5")
FIELD_SYMBOLS = [b1, b2, b3, b4, b5]

# 5. HAMILTONIAN
HAMILTONIAN = (
    sp.Rational(1, 2) * (px**2 + py**2) * (1 - delta + delta**2)
    - b1 * x * delta
    + sp.Rational(1, 2) * b1**2 * x**2
    + sp.Rational(1, 2) * b2 * (x**2 - y**2)
    + sp.Rational(1, 3) * b3 * (x**3 - 3 * x * y**2)
    + sp.Rational(1, 4) * b4 * (x**4 - 6 * x**2 * y**2 + y**4)
)

# 6. POLYNOMIAL SPACE
# m = maximum transverse degree
# d = maximum delta degree
ORDER = 8
DELTA_ORDER = 1
N_PLANES = 2

# 7. NORMALIZATION / PHYSICAL BOX
A_BOX = np.array( #     [delta, x, y, px, py]
    [0.01, 10e-3, 8.0e-3, 1.0e-3, 0.8e-3],
    dtype=float,
)

# 8. NONLINEAR NUMERICAL SETTINGS
LEAST_SQUARES_TOL = 1.0e-14
THIN_MULTIPOLE_LENGTH = 1.0e-8
CACHE_REPEATED_MAGNET_MAPS = True
CHECK_ELEMENT_UPPER_RIGHT = False

# 9. OBJECTIVE
GRADIENT_WEIGHT = 0.10    #how much influence is the derivate part gonna have
INVALID_PENALTY = 1.0e30    #???????????????????

# 10. CHROMATIC CORRECTION
CORRECT_CHROMATICITY = True   #Do you want it to be corrected? ofc you do, but just in case you got the choice.

# 11. OPTIMIZER SETTINGS
CMA_SIGMA = 0.50  #IDK check google
CMA_POPSIZE = 3#2   #
PRINT_EVERY = 1#0
SCALES = {name: 100.0 for name in VARY}
CMA_TIME = 60.0
POWELL_TIME_FRACTION = 0.25

# 12. NONLINEAR / START-END PLOT SETTINGS
PLOT_PLANE = "both"
PLOT_LEVELS = 60
PLOT_GRID_POINTS = 350
PLOT_RMIN = 0.06
PLOT_RMAX = 0.95
PLOT_DELTA = 0.0
PLOT_FOLDER = "nonlinear_plots"

# Physical plotting aperture.
PLOT_X_MAX = 5.0e-3
PLOT_PX_MAX = 1.0e-3
PLOT_Y_MAX = 3.0e-3
PLOT_PY_MAX = 1.0e-3

PLOT_START_END_SLICES = True   #Do you want it to be plot at the beggining and at the end?
SLICE_Y_VALUES = (0.0, 0.1 * PLOT_Y_MAX, 0.2*PLOT_Y_MAX)  #Amount of Ix slices to plot: (list of values of y)
SLICE_X_VALUES = (0.0,) #Amount of Iy slices to plot: (list of values of x)
SLICE_DELTA_VALUES = (0.0, 0.5 * float(A_BOX[0]), float(A_BOX[0])) #  Different values of delta for the slices: (list of values of delta)
SLICE_FROZEN_MOMENTUM = 0.0 #IDK what this is for?

# 13. FREQUENCY MAP ANALYSIS + IX TRACKING
RUN_FMA_START_END = True

FMA_CASE_LABEL = "current_lattice"

# Frequency-map window. Accelerator Toolbox expects these coordinates in mm:
# [xmin, xmax, ymin, ymax].
FMA_COORDS_MM = [-15.0, 15.0, -15.0, 15.0]

# Number of initial conditions in x and y.
# Start small for testing; increase to [60, 60] or [100, 100] for final maps.
FMA_STEPS = [121, 121]

# PyAT tracks 2*TURNS internally: one block for the first tune estimate and
# one block for the second tune estimate.
FMA_TURNS = 256

# Number of integration slices used by PyAT for thick multipoles.
FMA_NUM_INT_STEPS = 10

# Use the same chromatic correction as the native lattice before conversion.
FMA_CORRECT_CHROMATICITY = CORRECT_CHROMATICITY

# FMA / Ix tracking should represent one physical turn of the machine.
# If None, infer the number of identical cells needed for 360 degrees of net bend.
FMA_PHYSICAL_RING_CELLS = None

# Optional common momentum offset. 0.0 means on-momentum FMA.
FMA_DELTA = 0.0

# Parallel tracking. None lets PyAT choose its default process count.
FMA_POOL_SIZE = None
FMA_SHOW_PLOT = True

# Ix invariance tracking uses the same launch grid and total tracking length as
# the FMA diagnostic (2 * FMA_TURNS complete physical-ring turns).
IX_INVARIANCE_NORM_FLOOR_FRACTION = 1.0e-12
IX_INVARIANCE_LOG_MIN = -14.0
IX_INVARIANCE_LOG_MAX = 0.0

# 14. OUTPUTS
OUTPUT_ROOT = Path("optimization_output")
REPORT_FILE = OUTPUT_ROOT / "optimization_report.txt"
FINAL_LATTICE_FILE = OUTPUT_ROOT / "final_lattice.json"
PLOT_ROOT = OUTPUT_ROOT / "slices"
FMA_OUTPUT_DIRECTORY = OUTPUT_ROOT / "FMA"
