"""Thin research API for the Ixcononly branch.

No numerical routine prints.  The user runner selects constructors and plots;
this module only connects configuration, nonlinear construction, cached
tracking, and plotting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .state import STATE
from .config_loader import (
    load_general_config,
    load_selected_lattice,
    analysis_ring_names,
)
from .core import linear as lin
from .core import nonlinear as nl
from . import plotting
from . import tracking as tracking_module
from .report import write_run_report


METHODS = tuple(nl.CONSTRUCTORS)
_ALIASES = {
    "ls": "a_box",
    "weighted_ls": "a_box",
    "ls_y0": "a_box_y0",
    "y0": "a_box_y0",
    "cartesian_ls": "hybrid",
    "graded": "graded_ls",
    "block_ls": "graded_ls",
    "mean_ergodic": "cesaro",
    "abel_average": "abel",
}


def _method(name):
    method = _ALIASES.get(str(name).lower(), str(name).lower())
    if method not in nl.CONSTRUCTORS:
        raise ValueError(f"Unknown invariant method {name!r}.")
    return method


def available_methods():
    return tuple(nl.CONSTRUCTORS)


def _cfg():
    if STATE.config is None:
        raise RuntimeError("No experiment is loaded.")
    return STATE.config


def _context():
    if STATE.context is None:
        raise RuntimeError("No lattice is loaded.")
    return STATE.context


def _config_directory():
    return Path(_cfg().__file__).resolve().parent


def _resolve(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = _config_directory() / path
    return path.resolve()


def load(config_file="general_config.py", *, force=False):
    """Load the fixed lattice and linear optics used by all constructors."""
    cfg = load_general_config(config_file, reload=bool(force))
    lattice_cfg = load_selected_lattice(cfg, reload=bool(force))

    magnets, lattice, data, correction, parameters = lin.prepare_lattice(
        parameters=dict(lattice_cfg.PARAMETERS),
        ring_names=analysis_ring_names(cfg, lattice_cfg),
        magnet_builder=lattice_cfg.define_magnets,
        energy_parameter=lattice_cfg.ENERGY_PARAMETER,
        correction_parameter_map=lattice_cfg.CORRECTION_PARAMETER_MAP,
        correct_chromatic=bool(cfg.CORRECT_CHROMATICITY),
        family1=lattice_cfg.CHROMATIC_FAMILY1,
        family2=lattice_cfg.CHROMATIC_FAMILY2,
        target_chrom_x=lattice_cfg.TARGET_CHROM_X,
        target_chrom_y=lattice_cfg.TARGET_CHROM_Y,
        repetitions=lattice_cfg.REPETITIONS,
        step=lattice_cfg.STEP,
    )

    STATE.config = cfg
    STATE.config_path = str(Path(cfg.__file__).resolve())
    STATE.lattice_config = lattice_cfg
    STATE.context = {
        "magnets": magnets,
        "lattice": lattice,
        "data": data,
        "correction": correction,
        "parameters": dict(parameters),
    }
    STATE.invariants.clear()
    STATE.source = "loaded"
    return STATE.context


def status():
    """Small machine-readable configuration summary."""
    cfg = _cfg()
    return {
        "config": STATE.config_path,
        "lattice_file": cfg.LATTICE_FILE,
        "analysis_cells": int(cfg.ANALYSIS_CELLS),
        "order": int(cfg.ORDER),
        "delta_order": int(cfg.DELTA_ORDER),
        "chromatic_correction": bool(cfg.CORRECT_CHROMATICITY),
        "available_methods": available_methods(),
        "constructed_methods": tuple(STATE.invariants),
    }


def _method_options(method):
    cfg = _cfg()
    all_options = getattr(cfg, "INVARIANT_OPTIONS", {})
    return dict(all_options.get(method, {}))


def _state_for_method(method):
    cfg = _cfg()
    state = nl.initialize_nonlinear_for_method(
        _context()["data"],
        m=int(cfg.ORDER),
        d=int(cfg.DELTA_ORDER),
        hamiltonian=cfg.HAMILTONIAN,
        a_box=np.asarray(cfg.A_BOX, dtype=float),
        variables=tuple(cfg.VARIABLES),
        field_symbols=tuple(cfg.FIELD_SYMBOLS),
        n=int(cfg.N_PLANES),
        invariant_construction=method,
        options=_method_options(method),
    )

    if method == "cesaro":
        state["cesaro_terms"] = int(getattr(cfg, "CESARO_TERMS", 64))
    if method == "abel":
        state["abel_rho"] = float(getattr(cfg, "ABEL_RHO", 0.98))
    return state


def construct(method="a_box", *, force=False):
    """Construct and cache one Ix."""
    method = _method(method)
    if method in STATE.invariants and not force:
        return STATE.invariants[method]

    cfg = _cfg()
    state = _state_for_method(method)
    Ix, details, transfer = nl.construct_ix(
        _context()["lattice"],
        _context()["data"],
        state,
        tol=float(cfg.LEAST_SQUARES_TOL),
        cache=True,
    )

    result = {
        "method": method,
        "Ix": np.asarray(Ix, dtype=float),
        "state": state,
        "details": dict(details),
        "transfer": np.asarray(transfer, dtype=float),
    }
    if "Iy" in details:
        result["Iy"] = np.asarray(details.pop("Iy"), dtype=float)
    STATE.invariants[method] = result
    return result


def clear_cache():
    STATE.invariants.clear()


def coefficients(method="a_box", *, physical=True):
    result = construct(method)
    values = np.asarray(result["Ix"], dtype=float).copy()
    if physical:
        values = nl.physical_coefficients(values, result["state"])
    return values


def polynomial(method="a_box"):
    result = construct(method)
    return nl.vector_to_poly(
        coefficients(method, physical=True),
        result["state"]["monomial_basis"],
    )


def construction_details(method="a_box"):
    return dict(construct(method)["details"])


def tracking_deltas():
    """Momentum offsets selected for validation."""
    cfg = _cfg()
    values = getattr(cfg, "TRACKING_DELTAS", (cfg.TRACKING_DELTA,))
    return tuple(float(value) for value in values)


def tracking_directory(delta=None):
    """Cache directory for one momentum offset.

    Passing no delta returns the common tracking-cache root.  Passing a delta
    returns a dedicated subdirectory, so on- and off-momentum runs never
    overwrite one another.
    """
    root = tracking_module.default_cache_directory(_cfg(), _config_directory())
    if delta is None:
        return root
    return root / tracking_module.delta_label(float(delta))


def track(*, force=False, cache_directory=None, delta=None):
    """Track one momentum offset, or reuse its existing physical cache."""
    cfg = _cfg()
    effective_delta = (
        float(cfg.TRACKING_DELTA)
        if delta is None
        else float(delta)
    )
    folder = (
        tracking_directory(effective_delta)
        if cache_directory is None
        else _resolve(cache_directory)
    )
    return tracking_module.track(
        cfg,
        STATE.lattice_config,
        _context(),
        folder,
        force=force,
        delta=effective_delta,
    )


def load_tracking(cache_directory=None, *, delta=None):
    cfg = _cfg()
    effective_delta = (
        float(cfg.TRACKING_DELTA)
        if delta is None
        else float(delta)
    )
    folder = (
        tracking_directory(effective_delta)
        if cache_directory is None
        else _resolve(cache_directory)
    )
    return tracking_module.load_tracking(folder)


def _tracking_object(value=None):
    if value is None:
        return track(force=False)
    if isinstance(value, dict):
        return value
    return tracking_module.load_tracking(value)


def invariance(method, *, tracking=None, force=False):
    """Evaluate one Ix on cached trajectories; this never launches particles."""
    result = construct(method)
    data = _tracking_object(tracking)
    return tracking_module.invariance_metrics(
        result,
        data,
        floor_fraction=float(_cfg().IX_INVARIANCE_NORM_FLOOR_FRACTION),
        force=force,
    )


def plot_invariance(
    method,
    *,
    tracking=None,
    output_path=None,
    force_metrics=False,
    show=None,
):
    method = _method(method)
    data = _tracking_object(tracking)
    metrics = invariance(method, tracking=data, force=force_metrics)
    cfg = _cfg()
    delta = float(data["metadata"]["tracking"]["delta"])
    delta_name = tracking_module.delta_label(delta)
    if output_path is None:
        output_path = (
            _resolve(cfg.OUTPUT_DIRECTORY)
            / delta_name
            / "invariance"
            / f"{method}.png"
        )
    if show is None:
        show = bool(cfg.SHOW_PLOTS)
    return plotting.plot_invariance_map(
        data,
        metrics,
        method,
        output_path,
        show=show,
        vmin=float(getattr(cfg, "IX_INVARIANCE_LOG_MIN", -14.0)),
        vmax=float(getattr(cfg, "IX_INVARIANCE_LOG_MAX", 0.0)),
        delta=delta,
    )


def plot_comparison(
    name1,
    name2,
    *,
    tracking=None,
    output_path=None,
    force_metrics=False,
    show=None,
):
    method1 = _method(name1)
    method2 = _method(name2)
    data = _tracking_object(tracking)
    metrics1 = invariance(method1, tracking=data, force=force_metrics)
    metrics2 = invariance(method2, tracking=data, force=force_metrics)

    cfg = _cfg()
    delta = float(data["metadata"]["tracking"]["delta"])
    delta_name = tracking_module.delta_label(delta)
    if output_path is None:
        output_path = (
            _resolve(cfg.OUTPUT_DIRECTORY)
            / delta_name
            / "comparisons"
            / f"{method1}_vs_{method2}.png"
        )
    if show is None:
        show = bool(cfg.SHOW_PLOTS)

    return plotting.plot_comparison_map(
        data,
        metrics1,
        metrics2,
        method1,
        method2,
        output_path,
        show=show,
        delta=delta,
    )


def compare(name1, name2, **kwargs):
    """Compatibility alias for plot_comparison."""
    return plot_comparison(name1, name2, **kwargs)


def write_report(
    *,
    methods,
    comparisons,
    invariance_plots=(),
    tracking=None,
    force_tracking=None,
    force_metrics=None,
    output_path=None,
):
    """Write the compact text record for one run."""
    cfg = _cfg()
    if output_path is None:
        output_path = _resolve(cfg.OUTPUT_DIRECTORY) / "run_configuration.txt"

    tracking_dir = None
    if tracking is not None:
        if isinstance(tracking, dict):
            tracking_dir = tracking.get("directory")
        else:
            tracking_dir = tracking

    return write_run_report(
        output_path,
        cfg,
        _context(),
        methods=methods,
        comparisons=comparisons,
        invariance_plots=invariance_plots,
        tracking_directory=tracking_dir,
        force_tracking=force_tracking,
        force_metrics=force_metrics,
    )
