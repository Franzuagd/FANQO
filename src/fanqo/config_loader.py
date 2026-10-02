"""Load the small configuration surface used by Ixcononly."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import hashlib
import sys


_REQUIRED_LATTICE_SETTINGS = (
    "PARAMETERS",
    "define_magnets",
    "CELL_NAMES",
    "CORRECTION_PARAMETER_MAP",
    "ENERGY_PARAMETER",
    "REPETITIONS",
    "STEP",
    "CHROMATIC_FAMILY1",
    "CHROMATIC_FAMILY2",
    "TARGET_CHROM_X",
    "TARGET_CHROM_Y",
)

_REQUIRED_GENERAL_SETTINGS = (
    "LATTICE_FILE",
    "ANALYSIS_CELLS",
    "VARIABLES",
    "FIELD_SYMBOLS",
    "HAMILTONIAN",
    "ORDER",
    "DELTA_ORDER",
    "N_PLANES",
    "A_BOX",
    "LEAST_SQUARES_TOL",
    "CORRECT_CHROMATICITY",
    "TRACKING_COORDS_MM",
    "TRACKING_STEPS",
    "TRACKING_TURNS",
    "TRACKING_NUM_INT_STEPS",
    "TRACKING_DELTA",
    "IX_INVARIANCE_NORM_FLOOR_FRACTION",
    "OUTPUT_DIRECTORY",
    "SHOW_PLOTS",
)


def resolve_config_path(file_name, *, relative_to=None):
    path = Path(file_name).expanduser()
    if not path.suffix:
        path = path.with_suffix(".py")
    if not path.is_absolute():
        base = Path.cwd() if relative_to is None else Path(relative_to).expanduser().resolve()
        if base.is_file() or base.suffix:
            base = base.parent
        path = base / path
    return path.resolve()


def load_python_file(file_name, *, relative_to=None, module_prefix="user_config", reload=False):
    path = resolve_config_path(file_name, relative_to=relative_to)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file was not found: {path}")

    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    module_name = f"_{module_prefix}_{path.stem}_{digest}"
    existing = sys.modules.get(module_name)
    if existing is not None and not reload:
        return existing
    if existing is not None:
        sys.modules.pop(module_name, None)

    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python configuration: {path}")

    module = module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def validate_lattice_config(lattice_cfg):
    missing = [name for name in _REQUIRED_LATTICE_SETTINGS if not hasattr(lattice_cfg, name)]
    if missing:
        source = getattr(lattice_cfg, "__file__", "<unknown>")
        raise AttributeError(
            f"Invalid lattice configuration '{source}'.\nMissing required setting(s):\n    "
            + "\n    ".join(missing)
        )
    if not callable(lattice_cfg.define_magnets):
        raise TypeError("Lattice setting 'define_magnets' must be callable.")
    if not isinstance(lattice_cfg.PARAMETERS, dict):
        raise TypeError("Lattice setting 'PARAMETERS' must be a dictionary.")
    if not lattice_cfg.CELL_NAMES:
        raise ValueError("Lattice setting 'CELL_NAMES' cannot be empty.")
    return lattice_cfg


def load_lattice_config(file_name, *, relative_to=None, reload=False):
    return validate_lattice_config(
        load_python_file(
            file_name,
            relative_to=relative_to,
            module_prefix="lattice",
            reload=reload,
        )
    )


def load_selected_lattice(general_cfg, *, reload=False):
    return load_lattice_config(
        general_cfg.LATTICE_FILE,
        relative_to=general_cfg.__file__,
        reload=reload,
    )


def analysis_ring_names(general_cfg, lattice_cfg):
    cells = int(general_cfg.ANALYSIS_CELLS)
    if cells < 1 or cells != general_cfg.ANALYSIS_CELLS:
        raise ValueError("ANALYSIS_CELLS must be a positive integer.")
    return list(lattice_cfg.CELL_NAMES) * cells


def validate_general_config(general_cfg):
    missing = [name for name in _REQUIRED_GENERAL_SETTINGS if not hasattr(general_cfg, name)]
    if missing:
        source = getattr(general_cfg, "__file__", "<unknown>")
        raise AttributeError(
            f"Invalid general configuration '{source}'.\nMissing required setting(s):\n    "
            + "\n    ".join(missing)
        )

    if int(general_cfg.ANALYSIS_CELLS) < 1:
        raise ValueError("ANALYSIS_CELLS must be a positive integer.")
    if int(general_cfg.ORDER) < 2:
        raise ValueError("ORDER must be at least 2.")
    if int(general_cfg.DELTA_ORDER) < 0:
        raise ValueError("DELTA_ORDER must be nonnegative.")

    box = list(general_cfg.A_BOX)
    if len(box) != 5:
        raise ValueError("A_BOX must contain [delta, x, y, px, py].")
    if any(float(v) <= 0.0 for v in box):
        raise ValueError(
            "A_BOX entries must be positive on Ixcononly. "
            "Use method='a_box_y0' for the horizontal-only coefficient construction."
        )

    if len(general_cfg.TRACKING_COORDS_MM) != 4:
        raise ValueError("TRACKING_COORDS_MM must be [xmin, xmax, ymin, ymax].")
    if len(general_cfg.TRACKING_STEPS) != 2:
        raise ValueError("TRACKING_STEPS must be [nx, ny].")
    if int(general_cfg.TRACKING_TURNS) < 1:
        raise ValueError("TRACKING_TURNS must be positive.")

    deltas = getattr(
        general_cfg,
        "TRACKING_DELTAS",
        (general_cfg.TRACKING_DELTA,),
    )
    try:
        deltas = tuple(float(value) for value in deltas)
    except TypeError as exc:
        raise TypeError("TRACKING_DELTAS must be an iterable of numbers.") from exc
    if not deltas:
        raise ValueError("TRACKING_DELTAS cannot be empty.")
    if any(not __import__("math").isfinite(value) for value in deltas):
        raise ValueError("TRACKING_DELTAS entries must be finite.")
    return general_cfg


def load_general_config(file_name="general_config.py", *, relative_to=None, reload=False):
    return validate_general_config(
        load_python_file(
            file_name,
            relative_to=relative_to,
            module_prefix="general",
            reload=reload,
        )
    )
