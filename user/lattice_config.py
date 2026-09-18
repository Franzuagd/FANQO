"""Editable definition of the accelerator lattice used.
This is the lattice-specific file.  If the lattice, physical parameters,
magnet definitions, parameter dependencies, or linear-model settings change, edit them here.
The computational module linear_lattice.py should not need to be edited.
This file may be renamed; select it with LATTICE_FILE in general_config.py.
"""

def magnet(name, magnet_type, length, angle=0.0, K_value=0.0, S_value=0.0, O_value=0.0):  #Use by configuration
    """Create one magnet in the project's compact list representation."""
    return [name, magnet_type, float(length), float(angle), float(K_value), float(S_value), float(O_value), None, None]

# 1. PHYSICAL PARAMETERS
PARAMETERS = {
    "energy": 3.0,
    "LSD": 0.1,
    "F": 0.8,

    # Linear / geometric variables
    "X1": 3.633167514008421,
    "X2": -4.258277621861492,
    "X3": -2.690860661253351,
    "X4": 2.754505457254375,
    "X5": -3.336431720192718,
    "X6": -1.492176552197721,
    "X7": 2.943728718874649e-3,
    "X8": 6.010287762639232e-1,
    "X9": 5.370545478298007e-1,

    # Sextupole strengths
    "kse1": 0.0,
    "kfd2": 0.0,
    "kfd3": 0.0,
    "ks1": 0.0,
    "ks2": 0.0,
    "ksd3": 0.0,
    "ks1s": 0.0,
    "ks2s": 0.0,
    "ksf1": 42.72177269866247,
    "ksd1": -99.9460644308728,

    # Thin higher-multipole strengths
    "ko1": 0.0,
    "ko2": 0.0,
    "ko3": 0.0,
}

# 2. MAGNET DEFINITIONS
def define_magnets(parameters):
    """Return the unique magnet definitions for the current parameter values."""
    p = parameters
    LSD = p["LSD"]
    F = p["F"]

    return [   #Name/Type/strength
        # Drifts
        magnet("D1", "drift", 2.654400 - LSD),
        magnet("D4", "drift", 0.081240),
        magnet("D11", "drift", 0.063628),
        magnet("D12", "drift", 0.0099526),
        magnet("D5D6", "drift", p["X8"]),
        magnet("D9D10", "drift", p["X9"]),
        # Quadrupoles
        magnet("QF1", "quadrupole", 0.349140, K_value=p["X1"]),
        magnet("QD2", "quadrupole", 0.222950, K_value=p["X2"]),
        magnet("QD3", "quadrupole", 0.194780, K_value=p["X3"]),
        magnet("QF4", "quadrupole", 0.224580, K_value=p["X4"]),
        magnet("QD5", "quadrupole", 0.210950, K_value=p["X5"]),
        magnet("QF7", "quadrupole", 0.020986, K_value=p["X6"]),
        # Sextupoles
        magnet("SE1", "sextupole", LSD, S_value=p["kse1"]),
        magnet("FD2", "sextupole", 0.094502, S_value=p["kfd2"]),
        magnet("FD3", "sextupole", p["X7"], S_value=p["kfd3"]),
        magnet("S1", "sextupole", LSD, S_value=p["ks1"]),
        magnet("S2", "sextupole", LSD, S_value=p["ks2"]),
        magnet("SD3", "sextupole", 0.010176, S_value=p["ksd3"]),
        magnet("S1S", "sextupole", 0.002964, S_value=p["ks1s"]),
        magnet("S2S", "sextupole", 0.172130, S_value=p["ks2s"]),
        magnet("SF1", "sextupole", 0.220440, S_value=p["ksf1"]),
        magnet("SD1", "sextupole", LSD, S_value=p["ksd1"]),
        # Thin higher multipoles. O is the integrated nonlinear strength.
        magnet("O1", "multipole", 0.0, O_value=p["ko1"]),
        magnet("O2", "multipole", 0.0, O_value=p["ko2"]),
        magnet("O3", "multipole", 0.0, O_value=p["ko3"]),
        # Bending / combined-function magnets
        magnet("DQ6", "bending", 0.275390, angle=-0.73179259 * F, K_value=2.692600),
        magnet("A1", "bending", 0.075497, angle=0.0021719 * F),
        magnet("A2", "bending", 0.384040, angle=0.53380 * F),
        magnet("A3", "bending", 0.001995, angle=0.00032534 * F),
        magnet("A4", "bending", 0.913400, angle=2.0382 * F),
        magnet("A5", "bending", 0.152490, angle=0.93133 * F),
        magnet("B1", "bending", 0.400570, angle=0.63294 * F),
        magnet("B2", "bending", 0.563170, angle=1.1254 * F),
        magnet("B3", "bending", 0.362720, angle=1.1741 * F),
        magnet("B4", "bending", 0.285610, angle=1.4465 * F),
        magnet("B5", "bending", 0.240960, angle=0.58358 * F),
        magnet("B1S", "bending", 0.015767, angle=0.080780 * F),
        magnet("B2S", "bending", 0.001644, angle=-0.00041155 * F),
        magnet("B3S", "bending", 0.212550, angle=1.7586 * F),
        magnet("DQ1S", "bending", 0.257080, angle=0.81690 * F, K_value=-5.135300),
        magnet("ABQ1", "bending", 0.215990, angle=-0.60542 * F, K_value=6.191000),
    ]


# 3. CELL / LATTICE DEFINITION
DA1 = ["A1", "A2", "A3", "A4", "A5"]
IDA1 = DA1[::-1]

DBA = [
    "D1", "SE1", "QF1", "FD2", "QD2", "FD3",
    *IDA1,
    "D4", "QD3", "SD1", "O2", "D5D6", "S1", "QF4", "SF1",
    "O1", "QF4", "S2", "D9D10", "O3", "SD1", "QD5", "D11",
    "B1", "B2", "B3", "B4", "B5", "D12", "QF7", "SD3", "DQ6",
]

CELA = [
    "S1S", "ABQ1", "S2S", "DQ1S", "B1S", "B2S", "B3S",
    "B2S", "B1S", "DQ1S", "S2S", "ABQ1", "S1S",
]

CELL_NAMES = DBA + CELA + CELA + CELA + DBA[::-1]

# The number of cells used for the nonlinear/invariant calculation is selected
# in general_config.py through ANALYSIS_CELLS.

# 4. PARAMETER DEPENDENCIES  (Errors here are important to avoid)

LINEAR_VARIABLES = { #Variables that if changed, then we need to recompute the linear part.
    "energy", "LSD", "F", "X1", "X2", "X3", "X4", "X5", "X6", 
    "X7", "X8", "X9",
} 

CHROMATIC_VARIABLES = { #Variables that if changed, we need to recompute the chromatic sextupoles.
    "kse1", "kfd2", "kfd3", "ks1", "ks2", "ksd3", "ks1s", "ks2s",
    "ksf1", "ksd1",
}

NONLINEAR_VARIABLES = {"ko1", "ko2", "ko3"}  #Variables that if changed, they only changed the non-linear transport matrix.

PARAMETER_MAP = {  #Which magnets do each variable affects. (name,parameter affected)
    "energy": [],
    "X1": [("QF1", "K")],
    "X2": [("QD2", "K")],
    "X3": [("QD3", "K")],
    "X4": [("QF4", "K")],
    "X5": [("QD5", "K")],
    "X6": [("QF7", "K")],
    "X7": [("FD3", "LENGTH")],
    "X8": [("D5D6", "LENGTH")],
    "X9": [("D9D10", "LENGTH")],
    "kse1": [("SE1", "S")],
    "kfd2": [("FD2", "S")],
    "kfd3": [("FD3", "S")],
    "ks1": [("S1", "S")],
    "ks2": [("S2", "S")],
    "ksd3": [("SD3", "S")],
    "ks1s": [("S1S", "S")],
    "ks2s": [("S2S", "S")],
    "ksf1": [("SF1", "S")],
    "ksd1": [("SD1", "S")],
    "ko1": [("O1", "O")],
    "ko2": [("O2", "O")],
    "ko3": [("O3", "O")],
    "LSD": [
        ("D1", "LENGTH"), ("SE1", "LENGTH"), ("S1", "LENGTH"),
        ("S2", "LENGTH"), ("SD1", "LENGTH"),
    ],
    "F": [
        (name, "ANGLE")
        for name in (
            "DQ6", "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3",
            "B4", "B5", "B1S", "B2S", "B3S", "DQ1S", "ABQ1",
        )
    ],
}

CORRECTION_PARAMETER_MAP = {  #SExtupoles used for chromatic correction
    "SF1": "ksf1",
    "SD1": "ksd1",
}

# 5. LINEAR-MODEL SETTINGS
ENERGY_PARAMETER = "energy"
REPETITIONS = 1  #what is this used for?
STEP = 0.01 #subdivition used for linear computations
CHROMATIC_FAMILY1 = "SF1"  #Chromatic term 1   #Maybe I can make this more general.
CHROMATIC_FAMILY2 = "SD1"  #Chromatic term 2
TARGET_CHROM_X = 0.0
TARGET_CHROM_Y = 0.0