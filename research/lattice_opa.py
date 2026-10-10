"""FANQO lattice configuration converted from the supplied OPA lattice.

Source OPA ring:
    CELL : DBA, CELA, CELA, CELA, IDBA;
    RING : 20*CELL;

Only elements that actually occur in CELL are included below.  The uploaded
OPA file also defines DBA2/CELB/CELC/CELD and their magnets, but those segments
are not referenced by CELL or RING and therefore do not affect this lattice.
"""


def magnet(name, magnet_type, length, angle=0.0, K_value=0.0, S_value=0.0, O_value=0.0):
    """Create one magnet in FANQO's compact list representation."""
    return [
        name,
        magnet_type,
        float(length),
        float(angle),
        float(K_value),
        float(S_value),
        float(O_value),
        None,
        None,
    ]


# =============================================================================
# 1. PHYSICAL PARAMETERS
# =============================================================================

PARAMETERS = {
    "energy": 3.0,
    "LSD": 0.1,
    "F": 0.8,

    # Linear / geometric variables from the OPA file.
    "X1": 3.633167514008421,
    "X2": -4.258277621861492,
    "X3": -2.690860661253351,
    "X4": 2.754505457254375,
    "X5": -3.336431720192718,
    "X6": -1.492176552197721,
    "X7": 2.943728718874649e-3,
    "X8": 6.010287762639232e-1,
    "X9": 5.370545478298007e-1,

    # OPA sextupole K values after evaluating the explicit '*3' factors.
    "kse1": -1.261801314388578e1 * 3.0,
    "kfd2": 1.659756190762369e1 * 3.0,
    "kfd3": -2.595859148256768e2 * 3.0,
    "ks1": 3.215833056411174e0 * 3.0,
    "ks2": 1.283119806798643e1 * 3.0,
    "ksd3": 4.205796825865830e1 * 3.0,
    "ks1s": -1.209262491460888e2 * 3.0,
    "ks2s": 1.533075802893096e1 * 3.0,
    "ksf1": 8.974352327835229e0 * 3.0,
    "ksd1": -3.620170883693414e1 * 3.0,

    # OPA thin N=4 multipoles after evaluating the explicit '*1e-8' factors.
    "ko1": 5.336735998852785e9 * 1.0e-8,
    "ko2": -2.535776991395063e10 * 1.0e-8,
    "ko3": 2.055525746016869e9 * 1.0e-8,
}


# =============================================================================
# 2. MAGNET DEFINITIONS
# =============================================================================


def define_magnets(parameters):
    """Return the unique magnet definitions for the current parameter values."""
    p = parameters
    LSD = p["LSD"]
    F = p["F"]

    return [
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

        # Sextupoles used by the selected CELL.
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

        # OPA N=4 Multipole elements are thin.  Their K is already integrated.
        magnet("O1", "multipole", 0.0, O_value=p["ko1"]),
        magnet("O2", "multipole", 0.0, O_value=p["ko2"]),
        magnet("O3", "multipole", 0.0, O_value=p["ko3"]),

        # Bending / combined-function magnets.  OPA T is in degrees.
        magnet("DQ6", "bending", 0.275390, angle=-7.317925899999995e-1 * F, K_value=2.692600),
        magnet("A1", "bending", 0.075497, angle=2.1719e-3 * F),
        magnet("A2", "bending", 0.384040, angle=5.3380e-1 * F),
        magnet("A3", "bending", 0.001995, angle=3.2534e-4 * F),
        magnet("A4", "bending", 0.913400, angle=2.0382e0 * F),
        magnet("A5", "bending", 0.152490, angle=9.3133e-1 * F),
        magnet("B1", "bending", 0.400570, angle=6.3294e-1 * F),
        magnet("B2", "bending", 0.563170, angle=1.1254e0 * F),
        magnet("B3", "bending", 0.362720, angle=1.1741e0 * F),
        magnet("B4", "bending", 0.285610, angle=1.4465e0 * F),
        magnet("B5", "bending", 0.240960, angle=5.8358e-1 * F),
        magnet("B1S", "bending", 0.015767, angle=8.0780e-2 * F),
        magnet("B2S", "bending", 0.001644, angle=-4.1155e-4 * F),
        magnet("B3S", "bending", 0.212550, angle=1.7586e0 * F),
        magnet("DQ1S", "bending", 0.257080, angle=8.1690e-1 * F, K_value=-5.135300),
        magnet("ABQ1", "bending", 0.215990, angle=-6.0542e-1 * F, K_value=6.191000),
    ]


# =============================================================================
# 3. CELL / LATTICE DEFINITION
# =============================================================================

DA1 = ["A1", "A2", "A3", "A4", "A5"]
IDA1 = DA1[::-1]

DBA = [
    "D1", "SE1", "QF1", "FD2", "QD2", "FD3",
    *IDA1,
    "D4", "QD3", "SD1", "O2", "D5D6", "S1", "QF4", "SF1",
    "O1", "QF4", "S2", "D9D10", "O3", "SD1", "QD5", "D11",
    "B1", "B2", "B3", "B4", "B5", "D12", "QF7", "SD3", "DQ6",
]

# OPA: CELA : S1S, ABQ1, S2S, DQ1S, B1S, B2S, B3S,
#             B2S, B1S, DQ1S, S2S, ABQ1, S1S;
CELA = [
    "S1S", "ABQ1", "S2S", "DQ1S", "B1S", "B2S", "B3S",
    "B2S", "B1S", "DQ1S", "S2S", "ABQ1", "S1S",
]

# OPA: IDBA = -DBA and CELL = DBA, CELA, CELA, CELA, IDBA.
# In the current FANQO representation the reversed segment is represented by
# reversing the ordered names.  All bend edge angles in the supplied OPA file
# are zero, so no additional entrance/exit edge swap is required here.
IDBA = DBA[::-1]
CELL_NAMES = DBA + CELA + CELA + CELA + IDBA

# With F=0.8 this configured cell bends by exactly 18 degrees, so FANQO's
# physical-ring inference gives 20 cells for FMA / tracking, matching OPA:
#     RING : 20*CELL;


# =============================================================================
# 4. PARAMETER DEPENDENCIES
# =============================================================================

LINEAR_VARIABLES = {
    "energy", "LSD", "F", "X1", "X2", "X3", "X4", "X5", "X6",
    "X7", "X8", "X9",
}

CHROMATIC_VARIABLES = {
    "kse1", "kfd2", "kfd3", "ks1", "ks2", "ksd3", "ks1s", "ks2s",
    "ksf1", "ksd1",
}

NONLINEAR_VARIABLES = {"ko1", "ko2", "ko3"}

PARAMETER_MAP = {
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
        ("D1", "LENGTH"),
        ("SE1", "LENGTH"),
        ("S1", "LENGTH"),
        ("S2", "LENGTH"),
        ("SD1", "LENGTH"),
    ],
    "F": [
        (name, "ANGLE")
        for name in (
            "DQ6", "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3",
            "B4", "B5", "B1S", "B2S", "B3S", "DQ1S", "ABQ1",
        )
    ],
}


# =============================================================================
# 5. CHROMATIC CORRECTION / LINEAR MODEL
# =============================================================================

CORRECTION_PARAMETER_MAP = {
    "SF1": "ksf1",
    "SD1": "ksd1",
}

ENERGY_PARAMETER = "energy"
REPETITIONS = 1
STEP = 0.01
CHROMATIC_FAMILY1 = "SF1"
CHROMATIC_FAMILY2 = "SD1"
TARGET_CHROM_X = 0.0
TARGET_CHROM_Y = 0.0
