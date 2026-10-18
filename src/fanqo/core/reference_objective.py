"""Recovered advisor derivative/fluctuation objective; no tracking."""
import numpy as np
from scipy.stats import skew


def _evaluate_invariant_vector(
    I, state, coordinates
):
    """Evaluate an invariant vector on AT coordinates [x, px, y, py, delta, ct]."""
    coordinates = np.asarray(
        coordinates, dtype=float
    )
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] != 6
    ):
        raise ValueError(
            "Each trajectory must have shape (6, N)."
        )

    I = np.asarray(I, dtype=float)
    C = np.asarray(
        state["C"], dtype=float
    )
    if len(I) != len(C):
        raise ValueError(
            "Invariant vector and normalization C have different sizes."
        )

    variables = (
        coordinates[4],
        coordinates[0],
        coordinates[2],
        coordinates[1],
        coordinates[3],
    )

    result = np.zeros(
        coordinates.shape[1], dtype=float
    )
    idx_to_vec = state["idx_to_vec"]
    for k, coeff in enumerate(I):
        physical_coeff = (
            float(coeff) * float(C[k])
        )
        if physical_coeff == 0.0:
            continue
        term = physical_coeff
        for values, power in zip(
            variables, idx_to_vec[k]
        ):
            if power:
                term = term * values ** int(power)
        result += term
    return result


def reference_fluctuation_index(
    data,
    *,
    x_range=2.5e-3,
    x_points=21,
    y_range=0.7e-3,
    y_points=11,
    delta_values=(-3.4e-2,),
    momentum_weight=0.7,
):
    """Reference fluctuation objective based on sampled derivatives.

    On the x-y grid with px=py=0, form

        g = Ix_x^2 + Ix_y^2 + (w Ix_px)^2 + (w Ix_py)^2

    after removing the pure delta=0 quadratic sector. Each delta slice is scored
    with mean, standard deviation, and skewness of g; the worst slice is used.

    It only requires Ix, so it works with either invariant construction.
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
            flat_x,
            zeros,
            flat_y,
            zeros,
            np.full_like(flat_x, float(delta)),
            zeros,
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
            return float("inf"), {"reason": "no finite fluctuation samples"}

        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        # A constant field has no asymmetry. scipy.stats.skew returns NaN
        # for zero variance, which previously penalized even h=0 as invalid.
        skew_value = 0.0 if std_value == 0.0 else float(skew(values))
        if not np.isfinite(skew_value):
            return float("inf"), {
                "reason": "undefined fluctuation skewness",
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
