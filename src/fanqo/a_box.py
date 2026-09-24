"""One-time physical calibration of FANQO's nonlinear normalization box.

This module contains the relatively long a_box calibration workflow so the
public API stays small and readable.  Users should normally call
``fanqo.optimize_a_box()``, not functions in this module directly.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json

import numpy as np

from .state import STATE
from .core import optimization as opt
from .core import objective_functions as obj
from .plotting import get_pyplot


def _api():
    """Import the stateful API lazily to avoid an import cycle."""
    from . import api
    return api


# =============================================================================
# QUICK A_BOX SELECTION FROM FIXED PHYSICAL TRAJECTORIES
# =============================================================================

def _surviving_tracking_a_box_seed(
    trajectories,
    tracking_meta,
    current_a_box,
    *,
    quantile,
    margin,
    min_fraction,
    optimize_mask,
):
    """Propose a_box from the envelope of particles that survive all turns."""
    survivors = [
        track
        for track, meta in zip(trajectories, tracking_meta)
        if bool(meta["survived"])
    ]
    if not survivors:
        raise RuntimeError(
            "No particle survived the full a_box calibration tracking interval. "
            "Reduce A_BOX_TRACKING_COORDS_MM or A_BOX_TRACKING_TURNS."
        )

    points = np.concatenate(survivors, axis=1)
    # FANQO polynomial order: [delta, x, y, px, py].
    at_rows = (4, 0, 2, 1, 3)
    envelope = np.asarray(
        [
            np.quantile(np.abs(points[row]), float(quantile))
            for row in at_rows
        ],
        dtype=float,
    )

    current = np.asarray(current_a_box, dtype=float)
    mask = np.asarray(optimize_mask, dtype=bool)
    if current.shape != (5,) or mask.shape != (5,):
        raise ValueError("a_box and A_BOX_OPTIMIZE_MASK must both have five entries.")

    seed = current.copy()
    for i in range(5):
        if not mask[i]:
            continue
        proposed = float(margin) * float(envelope[i])
        fallback = float(min_fraction) * float(current[i])
        if not np.isfinite(proposed) or proposed <= 0.0:
            proposed = float(current[i])
        seed[i] = max(proposed, fallback)

    return seed, envelope, len(survivors)


def _plot_saved_ix_tracking(
    Ix,
    state,
    trajectories,
    initial,
    orbit,
    native,
    x_values,
    turns,
    delta,
    path,
    *,
    title,
    save=True,
    show=False,
):
    """Plot Ix contours against already-saved physical trajectories."""
    api = _api()
    cfg = api._cfg()
    fn = api._make_invariant_callable(Ix, state)
    qmax = max(
        float(cfg.PLOT_X_MAX),
        1.10 * float(np.max(np.abs(x_values))),
    )
    pmax = float(cfg.PLOT_PX_MAX)
    ngrid = int(getattr(cfg, "POINCARE_GRID_POINTS", cfg.PLOT_GRID_POINTS))

    q = np.linspace(-qmax, qmax, ngrid)
    p = np.linspace(-pmax, pmax, ngrid)
    Q, P = np.meshgrid(q, p, indexing="xy")
    Z = fn(
        orbit[4] + delta,
        orbit[0] + Q,
        orbit[2],
        orbit[1] + P,
        orbit[3],
    )
    levels = np.unique(np.asarray(
        [
            float(fn(row[4], row[0], row[2], row[1], row[3]))
            for row in initial
        ],
        dtype=float,
    ))
    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    levels = levels[(levels > zmin) & (levels < zmax)]
    if levels.size == 0:
        return None

    plt = get_pyplot(show)
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.contour(Q, P, Z, levels=np.sort(levels), linewidths=1.0)
    for trajectory in trajectories:
        ax.scatter(
            trajectory[0] - orbit[0],
            trajectory[1] - orbit[1],
            s=8,
            alpha=0.65,
        )
    ax.set_xlabel(r"$x-x_c$ [m]")
    ax.set_ylabel(r"$p_x-p_{x,c}$")
    ax.set_title(
        f"{title}\n{native['n_cells']} cells | {turns} turns | delta={delta:g}"
    )
    ax.set_xlim(-qmax, qmax)
    ax.set_ylim(-pmax, pmax)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = Path(path)
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=240)
    if show:
        plt.show()
    plt.close(fig)
    return path if save else None


def optimize_a_box(
    *,
    coords_mm=None,
    steps=None,
    turns=None,
    delta=None,
    make_active=True,
):
    """Calibrate a_box once from survivor tracking plus a short CMA-ES search.

    Physical particles are tracked exactly once.  Their surviving trajectories
    define a physically scaled seed box.  A short CMA-ES search then changes
    only a_box while scoring how nearly Ix is conserved on those same fixed
    trajectories.  The winning box is rebuilt fully once and can then be used
    for the complete magnet optimization.
    """
    api = _api()
    context = api._require_context("optimize_a_box()")
    cfg = api._cfg()

    if context["settings"].get("invariant_construction", "a_box") != "a_box":
        raise ValueError(
            "optimize_a_box() is only meaningful when "
            "INVARIANT_CONSTRUCTION='a_box'."
        )

    if importlib.util.find_spec("at") is None:
        raise ImportError(
            'optimize_a_box() requires tracking support. Install with: '
            'python -m pip install -e ".[tracking]"'
        )
    if importlib.util.find_spec("cma") is None:
        raise ImportError("optimize_a_box() requires cma.")

    current_a_box = np.asarray(context["state"]["a_box"], dtype=float)
    if current_a_box.shape != (5,):
        raise ValueError("The current FANQO model expects a five-entry a_box.")

    coords_mm = list(
        getattr(cfg, "A_BOX_TRACKING_COORDS_MM", cfg.FMA_COORDS_MM)
        if coords_mm is None else coords_mm
    )
    steps = list(
        getattr(cfg, "A_BOX_TRACKING_STEPS", [10, 10])
        if steps is None else steps
    )
    turns = int(
        getattr(cfg, "A_BOX_TRACKING_TURNS", 128)
        if turns is None else turns
    )
    delta = float(
        getattr(cfg, "A_BOX_TRACKING_DELTA", 0.0)
        if delta is None else delta
    )
    pool_size = getattr(cfg, "FMA_POOL_SIZE", None)

    xs_mm, ys_mm = api._fma_grid(coords_mm, steps)
    launch_pairs = [(float(x), float(y)) for y in ys_mm for x in xs_mm]

    ring, orbit, native = api._physical_tracking_ring(context)
    initial = np.repeat(orbit.reshape(1, 6), len(launch_pairs), axis=0)
    initial[:, 4] += delta
    initial[:, 0] += 1.0e-3 * np.asarray([p[0] for p in launch_pairs])
    initial[:, 2] += 1.0e-3 * np.asarray([p[1] for p in launch_pairs])

    trajectories, tracking_meta = api._track_initial_conditions(
        ring,
        initial,
        turns,
        pool_size=pool_size,
    )

    output_dir = Path(
        api._resolve(getattr(cfg, "A_BOX_OUTPUT_DIRECTORY", api._output_root() / "a_box"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_path = output_dir / "a_box_fixed_tracking.npz"
    np.savez_compressed(
        tracking_path,
        launch_pairs_mm=np.asarray(launch_pairs, dtype=float),
        orbit=orbit,
        delta=np.asarray([delta]),
        survived=np.asarray(
            [bool(meta["survived"]) for meta in tracking_meta],
            dtype=bool,
        ),
        **{f"track_{i:04d}": track for i, track in enumerate(trajectories)},
    )

    optimize_mask = np.asarray(
        getattr(cfg, "A_BOX_OPTIMIZE_MASK", [False, True, True, True, True]),
        dtype=bool,
    )
    seed, survivor_envelope, survivor_count = _surviving_tracking_a_box_seed(
        trajectories,
        tracking_meta,
        current_a_box,
        quantile=float(getattr(cfg, "A_BOX_SEED_QUANTILE", 0.98)),
        margin=float(getattr(cfg, "A_BOX_SEED_MARGIN", 1.15)),
        min_fraction=float(getattr(cfg, "A_BOX_SEED_MIN_FRACTION", 0.20)),
        optimize_mask=optimize_mask,
    )

    # Only complete survivors are used to fit the invariant box.
    survivor_trajectories = [
        track
        for track, meta in zip(trajectories, tracking_meta)
        if bool(meta["survived"])
    ]

    tol = float(cfg.LEAST_SQUARES_TOL)
    floor_fraction = float(
        getattr(cfg, "A_BOX_INVARIANCE_FLOOR_FRACTION", 1.0e-8)
    )

    # Expensive nonlinear transfer is constructed once in the current basis.
    reference_transfer, _, _ = opt.nonlinear_transfer(context, tol)

    history = []
    penalty = float(getattr(cfg, "INVALID_PENALTY", 1.0e30))

    def score_box(a_box):
        try:
            data = opt.a_box_objective_data(
                context,
                reference_transfer,
                np.asarray(a_box, dtype=float),
                survivor_trajectories,
                tol,
            )
            score, diagnostics = obj.tracked_ix_invariance(
                data,
                floor_fraction=floor_fraction,
            )
            score = float(score)
            if not np.isfinite(score):
                score = penalty
                diagnostics = {}
        except (
            ValueError,
            KeyError,
            OverflowError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ):
            score = penalty
            diagnostics = {}

        history.append({
            "a_box": np.asarray(a_box, dtype=float).copy(),
            "score": float(score),
        })
        return float(score), diagnostics

    seed_score, seed_diagnostics = score_box(seed)
    best_a_box = seed.copy()
    best_score = float(seed_score)
    best_diagnostics = dict(seed_diagnostics)

    active = np.flatnonzero(optimize_mask)
    if active.size:
        import cma

        factor_bounds = tuple(
            getattr(cfg, "A_BOX_CMA_FACTOR_BOUNDS", (0.5, 2.0))
        )
        if len(factor_bounds) != 2:
            raise ValueError("A_BOX_CMA_FACTOR_BOUNDS must contain [min_factor, max_factor].")
        lower_factor, upper_factor = map(float, factor_bounds)
        if not (0.0 < lower_factor < upper_factor):
            raise ValueError("A_BOX_CMA_FACTOR_BOUNDS must be positive and increasing.")

        lower = np.log(lower_factor)
        upper = np.log(upper_factor)
        sigma = float(getattr(cfg, "A_BOX_CMA_SIGMA", 0.25))
        popsize = int(getattr(cfg, "A_BOX_CMA_POPSIZE", 6))
        max_evals = int(getattr(cfg, "A_BOX_CMA_MAX_EVALS", 30))

        options = {
            "verb_disp": 0,
            "verbose": -9,
            "popsize": popsize,
            "maxfevals": max_evals,
            "bounds": [
                [lower] * len(active),
                [upper] * len(active),
            ],
        }
        es = cma.CMAEvolutionStrategy(
            np.zeros(len(active), dtype=float),
            sigma,
            options,
        )

        while not es.stop():
            solutions = es.ask()
            values = []
            for u in solutions:
                candidate = seed.copy()
                candidate[active] = seed[active] * np.exp(
                    np.asarray(u, dtype=float)
                )
                score, diagnostics = score_box(candidate)
                values.append(score)
                if score < best_score:
                    best_score = float(score)
                    best_a_box = candidate.copy()
                    best_diagnostics = dict(diagnostics)
            es.tell(solutions, values)

    save_plot, show_plot = api._plot_flags()
    seed_plot_path = None
    best_plot_path = None
    if save_plot or show_plot:
        # A 2-D Ix contour plot is only directly comparable to trajectories
        # launched on the horizontal slice nearest y=0. The full 2-D launch
        # grid is still used for the survivor seed and the objective.
        y_launch_mm = np.asarray([p[1] for p in launch_pairs], dtype=float)
        y0_mm = float(ys_mm[np.argmin(np.abs(ys_mm))])
        plot_indices = np.flatnonzero(np.isclose(y_launch_mm, y0_mm))
        plot_trajectories = [trajectories[i] for i in plot_indices]
        plot_initial = initial[plot_indices]
        x_launch_m = 1.0e-3 * np.asarray(
            [launch_pairs[i][0] for i in plot_indices],
            dtype=float,
        )

        # IMPORTANT: context is still in the original reference basis here.
        seed_data = opt.a_box_objective_data(
            context,
            reference_transfer,
            seed,
            survivor_trajectories,
            tol,
        )
        best_data = opt.a_box_objective_data(
            context,
            reference_transfer,
            best_a_box,
            survivor_trajectories,
            tol,
        )
        seed_plot_target = output_dir / "a_box_seed_poincare.png"
        best_plot_target = output_dir / "a_box_best_poincare.png"
        seed_plot_path = _plot_saved_ix_tracking(
            seed_data["Ix"],
            seed_data["state"],
            plot_trajectories,
            plot_initial,
            orbit,
            native,
            x_launch_m,
            turns,
            delta,
            seed_plot_target,
            title="Seed a_box: Ix contours vs. fixed tracking",
            save=save_plot,
            show=show_plot,
        )
        best_plot_path = _plot_saved_ix_tracking(
            best_data["Ix"],
            best_data["state"],
            plot_trajectories,
            plot_initial,
            orbit,
            native,
            x_launch_m,
            turns,
            delta,
            best_plot_target,
            title="Optimized a_box: Ix contours vs. fixed tracking",
            save=save_plot,
            show=show_plot,
        )

    # Build the winning nonlinear state exactly once for the subsequent main
    # optimization. No future magnet candidate changes a_box.
    if make_active:
        opt.set_context_a_box(context, best_a_box)
        configured_ix = bool(getattr(cfg, "COMPUTE_IX", True))
        configured_iy = bool(getattr(cfg, "COMPUTE_IY", False))
        active_invariants = opt.compute_requested_invariants(
            context,
            tol,
            compute_ix=configured_ix,
            compute_iy=configured_iy,
        )
        active_invariants.update({
            "objective": best_score,
            "objective_name": "tracked_ix_invariance",
            "selected_a_box": best_a_box.copy(),
            "a_box_seed": seed.copy(),
            "survivor_count": survivor_count,
            **best_diagnostics,
        })
        api._set_invariants(active_invariants)
        STATE.source = "a_box_calibrated"

    summary = {
        "initial_a_box": current_a_box.copy(),
        "survivor_seed_a_box": seed.copy(),
        "survivor_envelope": survivor_envelope.copy(),
        "survivor_count": int(survivor_count),
        "total_particles": int(len(trajectories)),
        "best_a_box": best_a_box.copy(),
        "seed_score": float(seed_score),
        "best_score": float(best_score),
        "history": history,
        "tracking": tracking_meta,
        "tracking_path": tracking_path,
        "seed_plot_path": seed_plot_path,
        "best_plot_path": best_plot_path,
        "turns": turns,
        "delta": delta,
        "coords_mm": coords_mm,
        "steps": steps,
        "make_active": bool(make_active),
        "native": native,
    }

    summary_path = output_dir / "a_box_calibration.json"
    payload = {
        "initial_a_box": current_a_box.tolist(),
        "survivor_seed_a_box": seed.tolist(),
        "survivor_envelope": survivor_envelope.tolist(),
        "survivor_count": int(survivor_count),
        "total_particles": int(len(trajectories)),
        "best_a_box": best_a_box.tolist(),
        "seed_score": float(seed_score),
        "best_score": float(best_score),
        "history": [
            {"a_box": item["a_box"].tolist(), "score": item["score"]}
            for item in history
        ],
        "tracking_path": str(tracking_path),
        "seed_plot_path": (
            None if seed_plot_path is None else str(seed_plot_path)
        ),
        "best_plot_path": (
            None if best_plot_path is None else str(best_plot_path)
        ),
        "turns": turns,
        "delta": delta,
        "coords_mm": list(map(float, coords_mm)),
        "steps": list(map(int, steps)),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary["summary_path"] = summary_path
    STATE.a_box_result = summary
    return summary


