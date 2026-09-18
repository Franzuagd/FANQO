"""Load and validate user-selected Python configuration files."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import hashlib
import sys


_REQUIRED_LATTICE_SETTINGS = (
    "PARAMETERS",
    "define_magnets",
    "CELL_NAMES",
    "LINEAR_VARIABLES",
    "CHROMATIC_VARIABLES",
    "PARAMETER_MAP",
    "CORRECTION_PARAMETER_MAP",
    "ENERGY_PARAMETER",
    "REPETITIONS",
    "STEP",
    "CHROMATIC_FAMILY1",
    "CHROMATIC_FAMILY2",
    "TARGET_CHROM_X",
    "TARGET_CHROM_Y",
)


def resolve_config_path(file_name, *, relative_to=None):
    """Resolve a Python config path, optionally relative to another file."""
    path = Path(file_name).expanduser()

    if not path.suffix:
        path = path.with_suffix(".py")

    if not path.is_absolute():
        if relative_to is None:
            base = Path.cwd()
        else:
            base = Path(relative_to).expanduser().resolve()
            if base.is_file() or base.suffix:
                base = base.parent
        path = base / path

    return path.resolve()


def load_python_file(file_name, *, relative_to=None, module_prefix="user_config", reload=False):
    """Load an arbitrary Python file and return it as a module object."""
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
    """Raise a clear error when a selected lattice file is incomplete."""
    missing = [
        name for name in _REQUIRED_LATTICE_SETTINGS
        if not hasattr(lattice_cfg, name)
    ]
    if missing:
        source = getattr(lattice_cfg, "__file__", "<unknown>")
        formatted = "\n    ".join(missing)
        raise AttributeError(
            f"Invalid lattice configuration '{source}'.\n"
            f"Missing required setting(s):\n    {formatted}"
        )

    if not callable(lattice_cfg.define_magnets):
        raise TypeError("Lattice setting 'define_magnets' must be callable.")

    if not isinstance(lattice_cfg.PARAMETERS, dict):
        raise TypeError("Lattice setting 'PARAMETERS' must be a dictionary.")

    if not lattice_cfg.CELL_NAMES:
        raise ValueError("Lattice setting 'CELL_NAMES' cannot be empty.")

    return lattice_cfg


def load_lattice_config(file_name, *, relative_to=None, reload=False):
    """Load and validate one user-selected lattice Python file."""
    lattice_cfg = load_python_file(
        file_name,
        relative_to=relative_to,
        module_prefix="lattice",
        reload=reload,
    )
    return validate_lattice_config(lattice_cfg)


def load_selected_lattice(general_cfg, *, reload=False):
    """Load LATTICE_FILE selected by a general configuration module."""
    if not hasattr(general_cfg, "LATTICE_FILE"):
        raise AttributeError("General configuration is missing LATTICE_FILE.")

    return load_lattice_config(
        general_cfg.LATTICE_FILE,
        relative_to=general_cfg.__file__,
        reload=reload,
    )


def analysis_ring_names(general_cfg, lattice_cfg):
    """Return the ring used for nonlinear/invariant analysis."""
    cells = int(general_cfg.ANALYSIS_CELLS)
    if cells < 1 or cells != general_cfg.ANALYSIS_CELLS:
        raise ValueError("ANALYSIS_CELLS must be a positive integer.")

    return list(lattice_cfg.CELL_NAMES) * cells

_REQUIRED_GENERAL_SETTINGS = (
    "LATTICE_FILE","ANALYSIS_CELLS","VARY","VARIABLES","FIELD_SYMBOLS",
    "HAMILTONIAN","ORDER","DELTA_ORDER","N_PLANES","A_BOX",
    "LEAST_SQUARES_TOL","GRADIENT_WEIGHT","INVALID_PENALTY",
    "CORRECT_CHROMATICITY","CMA_SIGMA","CMA_POPSIZE","PRINT_EVERY",
    "SCALES","CMA_TIME","POWELL_TIME_FRACTION",
)
def validate_general_config(general_cfg):
    missing=[name for name in _REQUIRED_GENERAL_SETTINGS if not hasattr(general_cfg,name)]
    if missing:
        source=getattr(general_cfg,"__file__","<unknown>")
        raise AttributeError(
            f"Invalid general configuration '{source}'.\nMissing required setting(s):\n    "
            + "\n    ".join(missing)
        )
    if int(general_cfg.ANALYSIS_CELLS)<1:
        raise ValueError("ANALYSIS_CELLS must be a positive integer.")
    return general_cfg
def load_general_config(file_name="general_config.py", *, relative_to=None, reload=False):
    cfg=load_python_file(file_name, relative_to=relative_to, module_prefix="general", reload=reload)
    return validate_general_config(cfg)
