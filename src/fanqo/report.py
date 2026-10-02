"""Small text report describing one research run.

This module contains reporting only.  Numerical code does not print or write
human-facing reports.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np


def _line(name, value):
    return f"{name}: {value}"


def write_run_report(
    path,
    config,
    context,
    *,
    methods,
    comparisons,
    tracking_directory=None,
):
    """Write a compact record of what was selected for a run."""
    lines = [
        "FANQO invariant-construction run",
        _line("time", datetime.now().isoformat(timespec="seconds")),
        _line("config", getattr(config, "__file__", "")),
        _line("lattice_file", config.LATTICE_FILE),
        _line("analysis_cells", int(config.ANALYSIS_CELLS)),
        _line("order", int(config.ORDER)),
        _line("delta_order", int(config.DELTA_ORDER)),
        _line("chromatic_correction", bool(config.CORRECT_CHROMATICITY)),
        _line("least_squares_tol", float(config.LEAST_SQUARES_TOL)),
        _line("a_box", np.asarray(config.A_BOX, dtype=float).tolist()),
        _line("methods", tuple(methods)),
        _line("comparisons", tuple(tuple(pair) for pair in comparisons)),
        _line("tracking_coords_mm", list(map(float, config.TRACKING_COORDS_MM))),
        _line("tracking_steps", list(map(int, config.TRACKING_STEPS))),
        _line("tracking_turns", int(config.TRACKING_TURNS)),
        _line("tracking_delta", float(config.TRACKING_DELTA)),
        _line("tracking_num_int_steps", int(config.TRACKING_NUM_INT_STEPS)),
        _line(
            "tracking_physical_ring_cells",
            getattr(config, "TRACKING_PHYSICAL_RING_CELLS", None),
        ),
        _line("tracking_cache", tracking_directory),
        _line(
            "cesaro_terms",
            getattr(config, "CESARO_TERMS", None),
        ),
        _line(
            "abel_rho",
            getattr(config, "ABEL_RHO", None),
        ),
        _line(
            "method_options",
            getattr(config, "INVARIANT_OPTIONS", {}),
        ),
        _line("corrected_parameters", context.get("parameters", {})),
        "",
    ]

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
