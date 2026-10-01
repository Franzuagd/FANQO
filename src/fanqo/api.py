"""Public API for the Ixcononly invariant-construction laboratory.

This branch intentionally has no magnet optimizer and no a_box optimizer.
A lattice is loaded once, then different Ix constructions are built on the same
fixed machine and compared by paired full-ring tracking.
"""

from __future__ import annotations

from pathlib import Path
import csv
import importlib.util
import math
import re

import numpy as np

from .state import STATE
from .config_loader import (
    load_general_config,
    load_selected_lattice,
    analysis_ring_names,
)
from .core import linear as lin
from .core import nonlinear as nl
from .plotting import get_pyplot


METHODS = ("a_box", "hybrid", "eigen", "graded_ls", "cesaro", "abel", "a_box_y0")
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
    if method not in METHODS:
        raise ValueError(
            f"Unknown invariant method {name!r}. Choose from {METHODS}."
        )
    return method


def available_methods():
    """Return the invariant constructors available on this branch."""
    return METHODS


def _cfg():
    if STATE.config is None:
        raise RuntimeError("No experiment is loaded. Run load() first.")
    return STATE.config


def _require_context():
    if STATE.context is None:
        raise RuntimeError("No lattice is loaded. Run load() first.")
    return STATE.context


def _config_dir():
    return Path(_cfg().__file__).resolve().parent


def _resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (_config_dir() / path).resolve()


def load(config_file="general_config.py", *, force=False):
    """Load one fixed lattice used by every invariant construction."""
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
    """Return a compact description of the current invariant experiment."""
    cfg = _cfg()
    context = _require_context()
    return {
        "config": STATE.config_path,
        "analysis_cells": int(cfg.ANALYSIS_CELLS),
        "order": int(cfg.ORDER),
        "delta_order": int(cfg.DELTA_ORDER),
        "a_box": np.asarray(cfg.A_BOX, dtype=float).copy(),
        "chromatic_correction": bool(cfg.CORRECT_CHROMATICITY),
        "methods": METHODS,
        "constructed": tuple(sorted(STATE.invariants)),
        "parameter_count": len(context["parameters"]),
    }


def _state_for_method(method):
    cfg = _cfg()
    context = _require_context()
    state = nl.initialize_nonlinear_for_method(
        context["data"],
        m=int(cfg.ORDER),
        d=int(cfg.DELTA_ORDER),
        hamiltonian=cfg.HAMILTONIAN,
        a_box=np.asarray(cfg.A_BOX, dtype=float),
        variables=tuple(cfg.VARIABLES),
        field_symbols=tuple(cfg.FIELD_SYMBOLS),
        n=int(cfg.N_PLANES),
        invariant_construction=method,
    )

    # Optional controls for the averaging constructors.  getattr keeps older
    # user configs valid.
    if method == "cesaro":
        terms = int(getattr(cfg, "CESARO_TERMS", 64))
        if terms < 1:
            raise ValueError("CESARO_TERMS must be a positive integer.")
        state["cesaro_terms"] = terms

    if method == "abel":
        rho = float(getattr(cfg, "ABEL_RHO", 0.98))
        if not 0.0 < rho < 1.0:
            raise ValueError("ABEL_RHO must satisfy 0 < ABEL_RHO < 1.")
        state["abel_rho"] = rho

    return state


def construct(method="a_box", *, force=False):
    """Construct and cache Ix for one method on the fixed loaded lattice."""
    method = _method(method)
    if method in STATE.invariants and not force:
        return STATE.invariants[method]

    cfg = _cfg()
    context = _require_context()
    state = _state_for_method(method)
    Ix, details, transfer = nl.construct_ix(
        context["lattice"],
        context["data"],
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
    STATE.invariants[method] = result
    return result


def clear_cache():
    """Forget constructed invariants without rebuilding the lattice."""
    STATE.invariants.clear()


def coefficients(method="a_box", *, physical=True):
    """Return Ix coefficients; physical=True multiplies by the basis scale C."""
    result = construct(method)
    coeff = np.asarray(result["Ix"], dtype=float).copy()
    if physical:
        coeff *= np.asarray(result["state"]["C"], dtype=float)
    return coeff


def polynomial(method="a_box"):
    """Return the symbolic physical Ix polynomial for one construction."""
    result = construct(method)
    physical = coefficients(method, physical=True)
    return nl.vector_to_poly(physical, result["state"]["monomial_basis"])


def construction_details(method="a_box"):
    """Return numerical construction diagnostics for one method."""
    return dict(construct(method)["details"])


def _ix_callable(result):
    Ix = np.asarray(result["Ix"], dtype=float)
    state = result["state"]
    physical = Ix * np.asarray(state["C"], dtype=float)
    powers = tuple(state["idx_to_vec"][k] for k in range(len(physical)))

    def evaluate(delta, x, y, px, py):
        variables = (
            np.asarray(delta),
            np.asarray(x),
            np.asarray(y),
            np.asarray(px),
            np.asarray(py),
        )
        out = np.zeros(np.broadcast_shapes(*(v.shape for v in variables)), dtype=float)
        variables = tuple(np.broadcast_to(v, out.shape) for v in variables)
        for coeff, exponents in zip(physical, powers):
            if coeff == 0.0:
                continue
            term = float(coeff)
            for values, power in zip(variables, exponents):
                if power:
                    term = term * values ** int(power)
            out = out + term
        return out

    return evaluate


# =============================================================================
# PHYSICAL FULL-RING TRACKING
# =============================================================================

def _infer_physical_ring_cells(parameters):
    lc = STATE.lattice_config
    magnets = lc.define_magnets(parameters)
    by_name = {lin.magnet_field(e, "NAME"): e for e in magnets}
    bend = 0.0
    for name in lc.CELL_NAMES:
        elem = by_name[name]
        if lin.magnet_field(elem, "TYPE") == "bending":
            bend += float(lin.magnet_field(elem, "ANGLE"))
    if abs(bend) < 1e-12:
        raise ValueError(
            "Configured cell has zero net bend; set TRACKING_PHYSICAL_RING_CELLS."
        )
    cells_float = 360.0 / abs(bend)
    cells = int(round(cells_float))
    if cells < 1 or not np.isclose(cells_float, cells, rtol=0, atol=1e-7):
        raise ValueError(
            f"Cell bend {bend:.12g} deg does not make an integer 360-degree ring."
        )
    return cells, bend


def _prepare_physical_ring(parameters):
    cfg = _cfg()
    lc = STATE.lattice_config
    configured = getattr(cfg, "TRACKING_PHYSICAL_RING_CELLS", None)
    if configured is None:
        n_cells, cell_bend = _infer_physical_ring_cells(parameters)
    else:
        n_cells = int(configured)
        if n_cells < 1:
            raise ValueError("TRACKING_PHYSICAL_RING_CELLS must be positive.")
        _, cell_bend = _infer_physical_ring_cells(parameters)

    ring_names = list(lc.CELL_NAMES) * n_cells
    magnets, lattice, data, correction, p = lin.prepare_lattice(
        parameters=dict(parameters),
        ring_names=ring_names,
        magnet_builder=lc.define_magnets,
        energy_parameter=lc.ENERGY_PARAMETER,
        correction_parameter_map=lc.CORRECTION_PARAMETER_MAP,
        correct_chromatic=bool(cfg.CORRECT_CHROMATICITY),
        family1=lc.CHROMATIC_FAMILY1,
        family2=lc.CHROMATIC_FAMILY2,
        target_chrom_x=lc.TARGET_CHROM_X,
        target_chrom_y=lc.TARGET_CHROM_Y,
        repetitions=1,
        step=lc.STEP,
    )
    total_bend = sum(
        float(lin.magnet_field(e, "ANGLE"))
        for e in lattice
        if lin.magnet_field(e, "TYPE") == "bending"
    )
    return {
        "magnets": magnets,
        "lattice": lattice,
        "data": data,
        "correction": correction,
        "parameters": p,
        "n_cells": n_cells,
        "cell_bend_deg": cell_bend,
        "total_bend_deg": total_bend,
    }


def _at_element(elem, at, nsteps):
    f = lin.magnet_field
    name = str(f(elem, "NAME"))
    typ = str(f(elem, "TYPE")).lower()
    L = float(f(elem, "LENGTH"))
    angle = float(f(elem, "ANGLE"))
    K = float(f(elem, "K"))
    S = float(f(elem, "S"))
    O = float(f(elem, "O"))

    if typ == "drift":
        return at.Drift(name, L)
    if typ == "quadrupole":
        if S or O:
            b = np.array([0.0, K, S, O])
            return at.Multipole(
                name, L, np.zeros_like(b), b, NumIntSteps=nsteps
            )
        return at.Quadrupole(name, L, k=K, NumIntSteps=nsteps)
    if typ == "sextupole":
        if K or O:
            b = np.array([0.0, K, S, O])
            return at.Multipole(
                name, L, np.zeros_like(b), b, NumIntSteps=nsteps
            )
        return at.Sextupole(name, L, h=S, NumIntSteps=nsteps)
    if typ == "multipole":
        b = np.array([0.0, K, S, O])
        a = np.zeros_like(b)
        return (
            at.ThinMultipole(name, a, b)
            if L == 0.0
            else at.Multipole(name, L, a, b, NumIntSteps=nsteps)
        )
    if typ == "bending":
        if S or O:
            raise NotImplementedError(f"Bend {name} contains S or O.")
        return at.Dipole(
            name,
            L,
            bending_angle=math.radians(angle),
            k=K,
            NumIntSteps=nsteps,
        )
    raise ValueError(f"Unknown magnet type {typ!r}: {name}")


def _tracking_ring():
    if importlib.util.find_spec("at") is None:
        raise ImportError(
            'Tracking comparison requires Accelerator Toolbox. Install with: '
            'python -m pip install -e ".[tracking]"'
        )

    import at
    from at.physics import find_orbit

    cfg = _cfg()
    context = _require_context()
    native = _prepare_physical_ring(context["parameters"])
    if not np.isclose(abs(native["total_bend_deg"]), 360.0, atol=1e-6):
        raise ValueError(
            f"Physical tracking ring bends {native['total_bend_deg']} deg, not 360."
        )

    elements = [
        _at_element(elem, at, int(cfg.TRACKING_NUM_INT_STEPS))
        for elem in native["lattice"]
    ]
    energy = float(native["parameters"][STATE.lattice_config.ENERGY_PARAMETER]) * 1e9
    ring = at.Lattice(elements, name="Ix_constructor_comparison", energy=energy)
    orbit, _ = find_orbit(ring)
    return ring, np.asarray(orbit, dtype=float).reshape(6), native


def _grid(force_y0=False):
    cfg = _cfg()
    xmin, xmax, ymin, ymax = map(float, cfg.TRACKING_COORDS_MM)
    nx, ny = map(int, cfg.TRACKING_STEPS)
    if nx < 1 or ny < 1:
        raise ValueError("TRACKING_STEPS entries must be positive.")

    xs = np.linspace(min(xmin, xmax), max(xmin, xmax), nx)
    if force_y0:
        ys = np.array([0.0])
    else:
        ys = np.linspace(min(ymin, ymax), max(ymin, ymax), ny)
    return xs, ys


def _initial_row(xs_mm, y_mm, orbit, delta):
    z = np.repeat(np.asarray(orbit, dtype=float).reshape(1, 6), len(xs_mm), axis=0)
    z[:, 4] += float(delta)
    z[:, 0] += 1.0e-3 * np.asarray(xs_mm, dtype=float)
    z[:, 2] += 1.0e-3 * float(y_mm)
    return z


def _eval_ix(fn, coordinates):
    c = np.asarray(coordinates, dtype=float)
    return fn(c[4], c[0], c[2], c[1], c[3])


def _initial_scales(fn1, fn2, xs, ys, orbit, delta):
    values1 = []
    values2 = []
    for y in ys:
        z0 = _initial_row(xs, y, orbit, delta)
        values1.append(np.asarray(_eval_ix(fn1, z0.T), dtype=float).reshape(-1))
        values2.append(np.asarray(_eval_ix(fn2, z0.T), dtype=float).reshape(-1))

    def scale(blocks):
        vals = np.concatenate(blocks) if blocks else np.array([1.0])
        value = float(np.nanmax(np.abs(vals)))
        return value if np.isfinite(value) and value > 0.0 else 1.0

    return scale(values1), scale(values2)


def _drift_metric(fn, trajectory, ix0, floor):
    values = np.asarray(_eval_ix(fn, trajectory), dtype=float).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        return np.nan, np.nan, np.nan

    # trajectory[:,0] is the initial condition. Drift rates start at turn 1.
    d = values[1:] - float(ix0)
    denom = max(abs(float(ix0)), float(floor))
    relative = np.abs(d) / denom
    turn = np.arange(1, len(relative) + 1, dtype=float)
    rate = relative / turn
    D = float(np.max(rate))
    return (
        D,
        float(np.log10(max(D, 1e-300))),
        float(np.sqrt(np.mean(relative**2))),
    )


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def compare(name1, name2, *, output_directory=None, save=None, show=None):
    """Compare two Ix constructions on identical physical trajectories.

    The plotted score is

        score = log10(D_name2) - log10(D_name1).

    Therefore:
        score > 0  -> name1 has smaller drift -> RED
        score < 0  -> name2 has smaller drift -> BLUE

    If either method is a_box_y0, tracking is automatically restricted to the
    exact y0=0 launch slice, because that reduced invariant is not defined as a
    full y-dependent construction.
    """
    method1 = _method(name1)
    method2 = _method(name2)
    result1 = construct(method1)
    result2 = construct(method2)

    force_y0 = bool(
        result1["state"].get("horizontal_slice_only", False)
        or result2["state"].get("horizontal_slice_only", False)
    )

    cfg = _cfg()
    save = bool(cfg.SAVE_PLOTS if save is None else save)
    show = bool(cfg.SHOW_PLOTS if show is None else show)
    turns = int(cfg.TRACKING_TURNS)
    delta = float(cfg.TRACKING_DELTA)
    pool_size = getattr(cfg, "TRACKING_POOL_SIZE", None)
    floor_fraction = float(cfg.IX_INVARIANCE_NORM_FLOOR_FRACTION)

    ring, orbit, native = _tracking_ring()
    xs, ys = _grid(force_y0=force_y0)
    fn1 = _ix_callable(result1)
    fn2 = _ix_callable(result2)
    scale1, scale2 = _initial_scales(fn1, fn2, xs, ys, orbit, delta)
    floor1 = floor_fraction * scale1
    floor2 = floor_fraction * scale2

    from at.tracking import patpass

    rows = []
    for y in ys:
        z0 = _initial_row(xs, y, orbit, delta)
        ix01 = np.asarray(_eval_ix(fn1, z0.T), dtype=float).reshape(-1)
        ix02 = np.asarray(_eval_ix(fn2, z0.T), dtype=float).reshape(-1)

        kwargs = {"losses": True}
        if pool_size is not None:
            kwargs["pool_size"] = int(pool_size)

        tracked, loss = patpass(
            ring,
            np.asfortranarray(z0.T),
            turns,
            **kwargs,
        )
        tracked = np.asarray(tracked, dtype=float)
        if tracked.ndim == 4:
            tracks = tracked[:, :, 0, :]
        elif tracked.ndim == 3:
            tracks = tracked
        else:
            raise RuntimeError(
                f"Unexpected patpass result shape {tracked.shape}."
            )

        lost = np.asarray(
            loss.get("islost", np.zeros(len(xs), bool)),
            dtype=bool,
        )

        for i, x in enumerate(xs):
            part = tracks[:, i, :]
            finite = np.all(np.isfinite(part), axis=0)
            completed = (
                int(np.flatnonzero(~finite)[0])
                if not np.all(finite)
                else turns
            )
            completed = max(0, min(completed, turns))
            survived = bool(completed == turns and not lost[i])
            trajectory = np.concatenate(
                (z0[i].reshape(6, 1), part[:, :completed]),
                axis=1,
            )

            D1, log1, rms1 = _drift_metric(
                fn1, trajectory, ix01[i], floor1
            )
            D2, log2, rms2 = _drift_metric(
                fn2, trajectory, ix02[i], floor2
            )
            valid1 = bool(survived and np.isfinite(log1))
            valid2 = bool(survived and np.isfinite(log2))
            score = float(log2 - log1) if valid1 and valid2 else np.nan

            if valid1 and valid2:
                # log10 is strictly increasing: positive score must mean
                # D1 < D2, i.e. method1 really is the red/better method.
                if score > 0.0 and not (D1 < D2):
                    raise AssertionError("Red/blue comparison sign is inconsistent.")
                if score < 0.0 and not (D2 < D1):
                    raise AssertionError("Red/blue comparison sign is inconsistent.")
                if score > 0.0:
                    winner = method1
                elif score < 0.0:
                    winner = method2
                else:
                    winner = "tie"
            elif valid1:
                winner = method1
            elif valid2:
                winner = method2
            else:
                winner = "invalid"

            rows.append({
                "x_mm": float(x),
                "y_mm": float(y),
                "survived": survived,
                f"D_{method1}": D1,
                f"log10D_{method1}": log1,
                f"rms_{method1}": rms1,
                f"D_{method2}": D2,
                f"log10D_{method2}": log2,
                f"rms_{method2}": rms2,
                "score": score,
                "winner": winner,
            })

    output_root = _resolve(
        cfg.OUTPUT_DIRECTORY if output_directory is None else output_directory
    )
    folder = output_root / f"{_safe_name(method1)}_vs_{_safe_name(method2)}"
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "comparison.csv"

    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    both = [r for r in rows if np.isfinite(r["score"])]
    scores = np.asarray([r["score"] for r in both], dtype=float)
    stats = {
        "method1": method1,
        "method2": method2,
        "red_means": method1,
        "blue_means": method2,
        "horizontal_slice_only": force_y0,
        "physical_ring_cells": int(native["n_cells"]),
        "turns": turns,
        "valid_both": int(len(both)),
        "method1_better": int(np.count_nonzero(scores > 0.0)),
        "method2_better": int(np.count_nonzero(scores < 0.0)),
        "ties": int(np.count_nonzero(scores == 0.0)),
        "median_score": float(np.median(scores)) if scores.size else np.nan,
        "fraction_method1_better": (
            float(np.mean(scores > 0.0)) if scores.size else np.nan
        ),
        "fraction_method2_better": (
            float(np.mean(scores < 0.0)) if scores.size else np.nan
        ),
    }

    plot_path = folder / "comparison.png" if save else None
    if save or show:
        plt = get_pyplot(show)
        fig, ax = plt.subplots(figsize=(9.0, 6.8))

        valid = [r for r in rows if np.isfinite(r["score"])]
        invalid = [r for r in rows if not np.isfinite(r["score"])]
        if valid:
            values = np.asarray([r["score"] for r in valid], dtype=float)
            vmax = float(np.max(np.abs(values)))
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
            xv = np.asarray([r["x_mm"] for r in valid], dtype=float)
            yv = np.asarray([r["y_mm"] for r in valid], dtype=float)
            scatter = ax.scatter(
                xv,
                yv,
                c=values,
                cmap="bwr",
                vmin=-vmax,
                vmax=vmax,
                marker="s",
                s=(120 if force_y0 else 34),
                linewidths=0,
            )
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(
                f"log10(D_{method2}) - log10(D_{method1})\n"
                f"red = {method1} better, blue = {method2} better"
            )

        if invalid and not force_y0:
            xi = np.asarray([r["x_mm"] for r in invalid], dtype=float)
            yi = np.asarray([r["y_mm"] for r in invalid], dtype=float)
            ax.scatter(
                xi,
                yi,
                marker="s",
                s=20,
                color="0.88",
                linewidths=0,
                label="invalid/lost",
                zorder=0,
            )
            ax.legend(loc="best")

        ax.set_xlabel(r"$x_0$ [mm]")
        if force_y0:
            ax.set_ylabel(r"$y_0=0$")
            ax.set_ylim(-0.75, 0.75)
            ax.set_yticks([0.0])
        else:
            ax.set_ylabel(r"$y_0$ [mm]")
            ax.set_aspect("equal", adjustable="box")

        ax.set_title(
            f"{method1} vs {method2}: tracked Ix quality\n"
            f"red = {method1} better | blue = {method2} better | "
            f"{native['n_cells']} cells | {turns} turns"
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        if save:
            fig.savefig(plot_path, dpi=240)
        if show:
            plt.show()
        plt.close(fig)

    return {
        "stats": stats,
        "rows": rows,
        "csv_path": csv_path,
        "plot_path": plot_path,
        "output_directory": folder,
    }
