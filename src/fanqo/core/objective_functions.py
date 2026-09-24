"""Objective functions available to FANQO optimizers.

The optimization engine is objective-agnostic.  Each objective receives a
dictionary of precomputed data and declares the quantities it needs through a
``requires`` attribute.  This lets FANQO skip expensive calculations such as
Iy when an objective only uses Ix.
"""

from __future__ import annotations

import numpy as np

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
