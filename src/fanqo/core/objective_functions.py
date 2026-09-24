"""Objective functions available to FANQO optimizers.

The optimization engine is objective-agnostic.  Each objective receives a
dictionary of precomputed data and declares the quantities it needs through a
``requires`` attribute.  This lets FANQO skip expensive calculations such as
Iy when an objective only uses Ix.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew

from . import nonlinear as nl


def horizontal_invariant_shape(data, *, gradient_weight=0.10):
    """Current FANQO horizontal quasi-invariant objective."""
    objective, value_norm, gradient_norm = nl.horizontal_shape_objective(
        data["Ix"],
        data["Sx"],
        data["state"],
        gradient_weight=float(gradient_weight),
    )
    return float(objective), {
        "value_norm": float(value_norm),
        "gradient_norm": float(gradient_norm),
    }


horizontal_invariant_shape.requires = {"Ix", "Sx"}


def advisor_fluctuation_index(
    data,
    *,
    x_range=2.5e-3,
    x_points=21,
    y_range=0.7e-3,
    y_points=11,
    delta_values=(-3.4e-2,),
    momentum_weight=0.7,
):
    """Advisor-code fluctuation objective, independent of Ix construction.

    The active legacy objective samples px=py=0 on an x-y grid and measures

        g = Ix_x^2 + Ix_y^2
            + (w Ix_px)^2 + (w Ix_py)^2

    for the nonlinear part of Ix. For each delta slice the score is

        sqrt(mean(g)^2 + std(g)^2 + skew(g)^2),

    and the worst delta slice is returned.

    FANQO reproduces that criterion using coefficient-space derivative
    operators, so it works for both a_box and advisor_eigen constructions.
    """
    Ix = np.asarray(data["Ix"], dtype=float).reshape(-1)
    state = data["state"]
    size = len(state["idx_to_vec"])
    if Ix.size != size:
        raise ValueError(f"Ix must have length {size}, received {Ix.size}.")

    x_range = float(x_range)
    y_range = float(y_range)
    x_points = int(x_points)
    y_points = int(y_points)
    momentum_weight = float(momentum_weight)
    deltas = np.asarray(delta_values, dtype=float).reshape(-1)

    if x_range <= 0.0 or y_range < 0.0:
        raise ValueError("x_range must be positive and y_range nonnegative.")
    if x_points < 4 or y_points < 1:
        raise ValueError("x_points must be >=4 and y_points >=1.")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("delta_values must be finite.")
    if not np.isfinite(momentum_weight) or momentum_weight < 0.0:
        raise ValueError("momentum_weight must be finite and nonnegative.")

    # The advisor derivative routines have their pure delta=0 quadratic
    # Courant-Snyder contribution commented out. Reproduce that generically:
    # remove only the delta=0 transverse sector of degree <= 2. Delta-dependent
    # terms remain, as they do in the legacy expressions.
    nonlinear_part = Ix.copy()
    for k, powers in state["idx_to_vec"].items():
        delta_degree = int(powers[0])
        transverse_degree = int(sum(powers[1:]))
        if delta_degree == 0 and transverse_degree <= 2:
            nonlinear_part[k] = 0.0

    derivatives = {
        "x": state["D_x"] @ nonlinear_part,
        "y": state["D_y"] @ nonlinear_part,
        "px": state["D_px"] @ nonlinear_part,
        "py": state["D_py"] @ nonlinear_part,
    }

    # Match the legacy x grid construction: two linspaces meeting at zero.
    half = int(x_points / 2)
    xs = np.unique(np.concatenate((
        np.linspace(-x_range, 0.0, half),
        np.linspace(0.0, x_range, half),
    )))
    ys = np.linspace(0.0, y_range, y_points)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    flat_x = X.reshape(-1)
    flat_y = Y.reshape(-1)
    zeros = np.zeros_like(flat_x)

    slice_scores = []
    slice_details = []
    for delta in deltas:
        coordinates = np.vstack((
            flat_x,       # x
            zeros,        # px
            flat_y,       # y
            zeros,        # py
            np.full_like(flat_x, float(delta)),
            zeros,        # ct, unused by polynomial evaluator
        ))

        dx = _evaluate_invariant_vector(derivatives["x"], state, coordinates)
        dy = _evaluate_invariant_vector(derivatives["y"], state, coordinates)
        dpx = _evaluate_invariant_vector(derivatives["px"], state, coordinates)
        dpy = _evaluate_invariant_vector(derivatives["py"], state, coordinates)

        values = (
            dx**2
            + dy**2
            + (momentum_weight * dpx)**2
            + (momentum_weight * dpy)**2
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("inf"), {"reason": "no finite advisor objective samples"}

        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        skew_value = float(skew(values))
        if not np.isfinite(skew_value):
            # The legacy implementation becomes NaN here and is rejected by
            # its outer objective. Preserve that invalid-candidate behavior.
            return float("inf"), {
                "reason": "undefined advisor skewness",
                "delta": float(delta),
            }

        score = float(np.sqrt(
            mean_value**2 + std_value**2 + skew_value**2
        ))
        slice_scores.append(score)
        slice_details.append({
            "delta": float(delta),
            "mean": mean_value,
            "std": std_value,
            "skew": skew_value,
            "score": score,
        })

    objective = float(np.max(slice_scores))
    return objective, {
        "slice_scores": slice_details,
        "x_samples": int(len(xs)),
        "y_samples": int(len(ys)),
        "momentum_weight": momentum_weight,
    }


advisor_fluctuation_index.requires = {"Ix"}


def _evaluate_invariant_vector(I, state, coordinates):
    """Evaluate an invariant vector on AT coordinates [x, px, y, py, delta, ct]."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] != 6:
        raise ValueError("Each trajectory must have shape (6, N).")

    I = np.asarray(I, dtype=float)
    C = np.asarray(state["C"], dtype=float)
    if len(I) != len(C):
        raise ValueError("Invariant vector and normalization C have different sizes.")

    variables = (
        coordinates[4],
        coordinates[0],
        coordinates[2],
        coordinates[1],
        coordinates[3],
    )

    result = np.zeros(coordinates.shape[1], dtype=float)
    idx_to_vec = state["idx_to_vec"]
    for k, coeff in enumerate(I):
        physical_coeff = float(coeff) * float(C[k])
        if physical_coeff == 0.0:
            continue
        term = physical_coeff
        for values, power in zip(variables, idx_to_vec[k]):
            if power:
                term = term * values ** int(power)
        result += term
    return result


def tracked_ix_invariance(data, *, floor_fraction=1.0e-8):
    """RMS relative Ix excursion on a fixed set of physical trajectories."""
    trajectories = data["trajectories"]
    if not trajectories:
        raise ValueError("At least one tracked trajectory is required.")

    floor_fraction = float(floor_fraction)
    if not np.isfinite(floor_fraction) or floor_fraction <= 0.0:
        raise ValueError("floor_fraction must be positive and finite.")

    values_by_track = []
    global_scale = 0.0
    for trajectory in trajectories:
        values = _evaluate_invariant_vector(data["Ix"], data["state"], trajectory)
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        values_by_track.append(values)
        global_scale = max(global_scale, float(np.max(np.abs(values))))

    if not values_by_track:
        raise ValueError("No trajectory contains at least two finite Ix evaluations.")

    if global_scale == 0.0 or not np.isfinite(global_scale):
        global_scale = 1.0
    absolute_floor = floor_fraction * global_scale

    per_track_rms = []
    per_track_max = []
    count = 0
    for values in values_by_track:
        reference = float(values[0])
        denominator = max(abs(reference), absolute_floor)
        relative = (values[1:] - reference) / denominator
        relative = relative[np.isfinite(relative)]
        if relative.size == 0:
            continue
        per_track_rms.append(float(np.sqrt(np.mean(relative ** 2))))
        per_track_max.append(float(np.max(np.abs(relative))))
        count += int(relative.size)

    if not per_track_rms:
        raise ValueError("No finite relative Ix excursions were produced.")

    score = float(np.sqrt(np.mean(np.asarray(per_track_rms, dtype=float) ** 2)))
    return score, {
        "trajectory_rms": per_track_rms,
        "trajectory_max": per_track_max,
        "max_relative_excursion": float(max(per_track_max)),
        "normalization_floor": float(absolute_floor),
        "tracked_samples": int(count),
    }


tracked_ix_invariance.requires = {"Ix", "trajectories"}
