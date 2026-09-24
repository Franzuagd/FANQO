"""Objective functions available to FANQO optimizers.

The optimization engine is objective-agnostic. Each objective receives a
precomputed data dictionary and declares its required quantities through a
``requires`` attribute.

The objectives adapted from the 2-D synthetic experiment are evaluated on a
family of horizontal slices

    Ix(delta, x, y, px, py=frozen_py),

and aggregate the per-slice scores with a conservative worst-slice default.
This preserves FANQO's full 5-D invariant while applying the original 2-D
geometry where it is mathematically defined.
"""

from __future__ import annotations

from functools import wraps

import numpy as np
from scipy.stats import skew

from . import nonlinear as nl


# =============================================================================
# 1. EXISTING FANQO OBJECTIVES
# =============================================================================


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
    """Legacy fluctuation-index objective, independent of Ix construction."""
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
        skew_value = float(skew(values))
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


reference_fluctuation_index.requires = {"Ix"}


# Backward-compatible public name. Other repository files already import it,
# and this task is intentionally restricted to this module.
def advisor_fluctuation_index(data, **kwargs):
    return reference_fluctuation_index(data, **kwargs)


advisor_fluctuation_index.requires = {"Ix"}


# =============================================================================
# 2. OBJECTIVE CONFIGURATION HELPER
# =============================================================================


def configured_objective(objective, /, **settings):
    """Return an objective with fixed settings and preserved requirements."""
    requirements = set(getattr(objective, "requires", ()))
    if not requirements:
        raise ValueError("The objective must declare a non-empty .requires set.")

    @wraps(objective)
    def wrapped(data):
        return objective(data, **settings)

    wrapped.requires = requirements
    return wrapped


# =============================================================================
# 3. COMMON MULTI-SLICE POLYNOMIAL GEOMETRY
# =============================================================================


def _physical_box(state):
    box = np.asarray(state.get("a_box"), dtype=float).reshape(-1)
    if box.size != 5 or np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("state['a_box'] must contain five positive finite half-widths.")
    return box


def _slice_values(state, y_values=None, delta_values=None):
    """Resolve physical y/delta slices without changing external config."""
    box = _physical_box(state)
    if y_values is None:
        y_values = (0.0, 0.05 * box[2], 0.10 * box[2])
    if delta_values is None:
        delta_values = (0.0, 0.50 * box[0], 1.00 * box[0])

    ys = np.asarray(y_values, dtype=float).reshape(-1)
    deltas = np.asarray(delta_values, dtype=float).reshape(-1)
    if ys.size == 0 or deltas.size == 0:
        raise ValueError("y_values and delta_values must be non-empty.")
    if np.any(~np.isfinite(ys)) or np.any(~np.isfinite(deltas)):
        raise ValueError("y_values and delta_values must be finite.")
    return ys, deltas


def _horizontal_limits(state, x_max=None, px_max=None):
    box = _physical_box(state)
    x_max = 0.50 * box[1] if x_max is None else float(x_max)
    px_max = box[3] if px_max is None else float(px_max)
    if (
        not np.isfinite(x_max)
        or not np.isfinite(px_max)
        or x_max <= 0.0
        or px_max <= 0.0
    ):
        raise ValueError("x_max and px_max must be positive and finite.")
    return x_max, px_max


def _horizontal_cs(state):
    cs0 = np.asarray(state["linear_cs0"], dtype=float).reshape(-1)
    if cs0.size < 3:
        raise ValueError("state['linear_cs0'] does not contain horizontal Twiss data.")
    beta, alpha, gamma = map(float, cs0[:3])
    if beta <= 0.0 or not np.all(np.isfinite([beta, alpha, gamma])):
        raise ValueError("Invalid horizontal Courant-Snyder parameters.")
    return beta, alpha, gamma


def _pad_quadratic_reference(Ix, Sx, state):
    Ix = np.asarray(Ix, dtype=float).reshape(-1)
    Sx = np.asarray(Sx, dtype=float).reshape(-1)
    q = int(state["quad_size"])
    if Sx.size != q:
        raise ValueError(f"Sx must have length {q}, received {Sx.size}.")
    if Ix.size != len(state["idx_to_vec"]):
        raise ValueError("Ix length does not match the monomial basis.")
    full = np.zeros_like(Ix)
    full[:q] = Sx
    return full


def _slice_polynomial_coefficients(vector, state, *, y, delta, py):
    """Freeze (delta,y,py) and collapse a FANQO vector to c[i,j] x^i px^j."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    C = np.asarray(state["C"], dtype=float).reshape(-1)
    if vector.size != C.size:
        raise ValueError("Coefficient vector and state['C'] must have the same length.")

    max_x = max(int(p[1]) for p in state["idx_to_vec"].values())
    max_px = max(int(p[3]) for p in state["idx_to_vec"].values())
    coeff = np.zeros((max_x + 1, max_px + 1), dtype=float)

    y = float(y)
    delta = float(delta)
    py = float(py)
    physical = vector * C
    for k, value in enumerate(physical):
        if value == 0.0:
            continue
        pd, pxpow, pypow, ppx, ppypow = map(
            int, state["idx_to_vec"][k]
        )
        if ppypow and py == 0.0:
            continue
        if pypow and y == 0.0:
            continue
        if pd and delta == 0.0:
            continue

        frozen = value
        if pd:
            frozen *= delta**pd
        if pypow:
            frozen *= y**pypow
        if ppypow:
            frozen *= py**ppypow
        coeff[pxpow, ppx] += frozen
    return coeff


def _poly2d(coeff, x, px):
    return np.polynomial.polynomial.polyval2d(x, px, coeff)


def _differentiate_2d(coeff, axis):
    coeff = np.asarray(coeff, dtype=float)
    out = np.zeros_like(coeff)
    if axis == 0:
        powers = np.arange(1, coeff.shape[0], dtype=float)[:, None]
        out[:-1, :] = powers * coeff[1:, :]
    elif axis == 1:
        powers = np.arange(1, coeff.shape[1], dtype=float)[None, :]
        out[:, :-1] = powers * coeff[:, 1:]
    else:
        raise ValueError("axis must be 0 (x) or 1 (px).")
    return out


def _slice_bundle(Ix, state, *, y, delta, py, Sx=None):
    I_coeff = _slice_polynomial_coefficients(
        Ix, state, y=y, delta=delta, py=py
    )
    I_centered = I_coeff.copy()
    I_offset = float(I_centered[0, 0])
    I_centered[0, 0] = 0.0

    bundle = {
        "I": I_centered,
        "I_offset": I_offset,
        "Ix_dx": _differentiate_2d(I_coeff, 0),
        "Ix_dpx": _differentiate_2d(I_coeff, 1),
    }

    if Sx is not None:
        S_full = _pad_quadratic_reference(Ix, Sx, state)
        h_vector = np.asarray(Ix, dtype=float) - S_full
        h_coeff = _slice_polynomial_coefficients(
            h_vector, state, y=y, delta=delta, py=py
        )
        h_coeff = h_coeff.copy()
        h_coeff[0, 0] = 0.0
        bundle.update({
            "h": h_coeff,
            "h_dx": _differentiate_2d(h_coeff, 0),
            "h_dpx": _differentiate_2d(h_coeff, 1),
        })
    return bundle


def _cs_values(x, px, state):
    beta, alpha, gamma = _horizontal_cs(state)
    S = gamma * x**2 + 2.0 * alpha * x * px + beta * px**2
    dS_dx = 2.0 * gamma * x + 2.0 * alpha * px
    dS_dpx = 2.0 * alpha * x + 2.0 * beta * px
    return S, dS_dx, dS_dpx


def _aggregate_slice_scores(scores, mode="worst"):
    values = np.asarray(scores, dtype=float)
    if values.size == 0 or np.any(~np.isfinite(values)):
        return float("inf")
    mode = str(mode).lower()
    if mode == "worst":
        return float(np.max(values))
    if mode == "mean":
        return float(np.mean(values))
    if mode == "rms":
        return float(np.sqrt(np.mean(values**2)))
    raise ValueError("aggregation must be 'worst', 'mean', or 'rms'.")


def _slice_records(ys, deltas):
    for delta in deltas:
        for y in ys:
            yield float(y), float(delta)


# =============================================================================
# 4. OBJECTIVE 1 -- WEIGHTED INTEGRAL DEFORMATION
# =============================================================================


def integral_objective(
    data,
    *,
    y_values=None,
    delta_values=None,
    frozen_py=0.0,
    aggregation="worst",
    x_max=None,
    px_max=None,
    quadrature_points=30,
    lambda_h=1.0,
    lambda_normal=0.20,
    weight_mode="custom",
    weight_strength=2.0,
):
    """Weighted deformation objective over multiple horizontal FANQO slices."""
    Ix, Sx, state = data["Ix"], data["Sx"], data["state"]
    ys, deltas = _slice_values(state, y_values, delta_values)
    x_max, px_max = _horizontal_limits(state, x_max, px_max)
    n = int(quadrature_points)
    if n < 4:
        raise ValueError("quadrature_points must be >= 4.")

    nodes, weights = np.polynomial.legendre.leggauss(n)
    x = x_max * nodes
    px = px_max * nodes
    wx = x_max * weights
    wpx = px_max * weights
    X, PX = np.meshgrid(x, px, indexing="xy")
    quad = np.outer(wpx, wx)
    S, dS_dx, dS_dpx = _cs_values(X, PX, state)
    Smax = max(float(np.max(S)), 1e-300)
    Sn = np.clip(S / Smax, 0.0, None)

    mode = str(weight_mode).lower()
    if mode == "uniform":
        w = np.ones_like(S)
    elif mode == "core":
        w = np.exp(-float(weight_strength) * Sn)
    elif mode == "edge":
        w = 1.0 + float(weight_strength) * Sn
    elif mode == "custom":
        # Dimensionless translation of the synthetic 1/(S+0.1) choice.
        w = 1.0 / (Sn + 0.1)
    else:
        raise ValueError(
            "weight_mode must be 'uniform', 'core', 'edge', or 'custom'."
        )

    W = quad * w
    Z = float(np.sum(W))
    if not np.isfinite(Z) or Z <= 0.0:
        raise ValueError("Total quadrature weight must be positive and finite.")

    gradS_norm = np.sqrt(dS_dx**2 + dS_dpx**2)
    slices = []
    for y, delta in _slice_records(ys, deltas):
        bundle = _slice_bundle(
            Ix, state, y=y, delta=delta, py=frozen_py, Sx=Sx
        )
        h = _poly2d(bundle["h"], X, PX)
        dh_dx = _poly2d(bundle["h_dx"], X, PX)
        dh_dpx = _poly2d(bundle["h_dpx"], X, PX)

        numerator = dh_dx * dS_dx + dh_dpx * dS_dpx
        D_normal = np.where(
            gradS_norm > 1e-14,
            numerator / np.where(
                gradS_norm > 1e-14, gradS_norm, 1.0
            ),
            0.0,
        )

        norm_h = float(
            np.sqrt(max(float(np.sum(W * h**2) / Z), 0.0))
        )
        norm_normal = float(
            np.sqrt(max(float(np.sum(W * D_normal**2) / Z), 0.0))
        )
        score = (
            float(lambda_h) * norm_h
            + float(lambda_normal) * norm_normal
        )
        slices.append({
            "y": y,
            "delta": delta,
            "score": float(score),
            "norm_h": norm_h,
            "norm_normal": norm_normal,
        })

    objective = _aggregate_slice_scores(
        [s["score"] for s in slices], aggregation
    )
    return objective, {
        "aggregation": aggregation,
        "slice_scores": slices,
        "x_max": x_max,
        "px_max": px_max,
        "quadrature_points": n,
    }


integral_objective.requires = {"Ix", "Sx"}


# =============================================================================
# 5. OBJECTIVE 2 -- OUTWARD DEFORMATION GROWTH
# =============================================================================


def mesh_objective(
    data,
    *,
    y_values=None,
    delta_values=None,
    frozen_py=0.0,
    aggregation="worst",
    x_max=None,
    px_max=None,
    mesh_points=75,
    stability_weight_power=3.0,
    stability_weight_floor=0.02,
    penalty_power=1.0,
):
    """Penalize positive outward growth of h^2 on multiple horizontal slices."""
    Ix, Sx, state = data["Ix"], data["Sx"], data["state"]
    ys, deltas = _slice_values(state, y_values, delta_values)
    x_max, px_max = _horizontal_limits(state, x_max, px_max)
    n = int(mesh_points)
    if n < 5:
        raise ValueError("mesh_points must be >= 5.")

    x = np.linspace(-x_max, x_max, n)
    px = np.linspace(-px_max, px_max, n)
    X, PX = np.meshgrid(x, px, indexing="xy")
    S, dS_dx, dS_dpx = _cs_values(X, PX, state)
    Smax = max(float(np.max(S)), 1e-300)
    normalized_S = np.clip(S / Smax, 0.0, 1.0)
    W = float(stability_weight_floor) + (
        1.0 - float(stability_weight_floor)
    ) * normalized_S ** float(stability_weight_power)
    Z = float(np.sum(W))
    gradS_sq = dS_dx**2 + dS_dpx**2

    slices = []
    for y, delta in _slice_records(ys, deltas):
        bundle = _slice_bundle(
            Ix, state, y=y, delta=delta, py=frozen_py, Sx=Sx
        )
        h = _poly2d(bundle["h"], X, PX)
        dh_dx = _poly2d(bundle["h_dx"], X, PX)
        dh_dpx = _poly2d(bundle["h_dpx"], X, PX)

        numerator = 2.0 * h * (
            dh_dx * dS_dx + dh_dpx * dS_dpx
        )
        growth = np.zeros_like(S)
        mask = gradS_sq > 1e-14
        growth[mask] = numerator[mask] / gradS_sq[mask]
        positive = np.maximum(growth, 0.0)
        penalty = positive ** float(penalty_power)
        score = float(np.sum(W * penalty) / Z)

        dI_dx = _poly2d(bundle["Ix_dx"], X, PX)
        dI_dpx = _poly2d(bundle["Ix_dpx"], X, PX)
        slope = np.ones_like(S)
        slope[mask] = (
            dI_dx[mask] * dS_dx[mask]
            + dI_dpx[mask] * dS_dpx[mask]
        ) / gradS_sq[mask]

        slices.append({
            "y": y,
            "delta": delta,
            "score": score,
            "positive_fraction": float(np.mean(positive > 0.0)),
            "max_growth": float(np.max(positive)),
            "min_dI_dS": float(np.min(slope)),
        })

    objective = _aggregate_slice_scores(
        [s["score"] for s in slices], aggregation
    )
    return objective, {
        "aggregation": aggregation,
        "slice_scores": slices,
        "x_max": x_max,
        "px_max": px_max,
        "mesh_points": n,
    }


mesh_objective.requires = {"Ix", "Sx"}


# =============================================================================
# 6. SHARED ORIGIN-CONNECTED CONTOUR GEOMETRY
# =============================================================================


def _build_contour_geometry(
    state, *, x_max, px_max, angles, radial_points
):
    beta, alpha, _ = _horizontal_cs(state)
    theta = np.linspace(
        0.0, 2.0 * np.pi, int(angles), endpoint=False
    )
    sqrt_beta = np.sqrt(beta)
    dx_dr = sqrt_beta * np.cos(theta)
    dpx_dr = (
        np.sin(theta) - alpha * np.cos(theta)
    ) / sqrt_beta

    tiny = 1e-14
    rx = np.where(
        np.abs(dx_dr) > tiny,
        x_max / np.abs(dx_dr),
        np.inf,
    )
    rp = np.where(
        np.abs(dpx_dr) > tiny,
        px_max / np.abs(dpx_dr),
        np.inf,
    )
    r_box = np.minimum(rx, rp)

    fraction = np.linspace(
        0.0, 1.0, int(radial_points)
    )[:, None]
    R = fraction * r_box[None, :]
    X = R * dx_dr[None, :]
    PX = R * dpx_dr[None, :]
    return {
        "theta": theta,
        "dx_dr": dx_dr,
        "dpx_dr": dpx_dr,
        "r_box": r_box,
        "R": R,
        "X": X,
        "PX": PX,
        "box_inscribed_radius": float(np.min(r_box)),
    }


def _contour_components(
    I_coeff,
    geometry,
    *,
    min_radial_slope=0.02,
    level_fraction=0.97,
    size_weight=1.00,
    distortion_weight=0.20,
    roughness_weight=0.05,
    fold_mean_weight=0.20,
    fold_worst_weight=0.80,
    invalid_weight=5.00,
):
    R = geometry["R"]
    X = geometry["X"]
    PX = geometry["PX"]
    dx_dr = geometry["dx_dr"][None, :]
    dpx_dr = geometry["dpx_dr"][None, :]
    n_angles = len(geometry["theta"])

    I = _poly2d(I_coeff, X, PX)
    dI_dx = _poly2d(
        _differentiate_2d(I_coeff, 0), X, PX
    )
    dI_dpx = _poly2d(
        _differentiate_2d(I_coeff, 1), X, PX
    )
    dI_dr = dI_dx * dx_dr + dI_dpx * dpx_dr

    slope = np.ones_like(I)
    mask = R > 1e-12
    slope[mask] = dI_dr[mask] / (2.0 * R[mask])

    finite = np.isfinite(I) & np.isfinite(slope)
    good = finite & (slope > float(min_radial_slope))
    good[0, :] = True
    prefix_good = np.logical_and.accumulate(good, axis=0)
    prefix_count = np.sum(prefix_good, axis=0)
    last_good = np.maximum(
        prefix_count - 1, 0
    ).astype(int)

    angle_index = np.arange(n_angles)
    terminal_I = I[last_good, angle_index]
    valid_terminal = (
        np.isfinite(terminal_I)
        & (last_good >= 1)
        & (terminal_I > 0.0)
    )
    valid_fraction = float(np.mean(valid_terminal))

    violation = np.maximum(
        float(min_radial_slope) - slope, 0.0
    )
    violation = np.where(
        np.isfinite(violation), violation, 1e6
    )
    bounded = violation / (1.0 + violation)
    fold_mean = float(
        np.mean(bounded[1:, :] ** 2)
    )
    fold_worst = float(
        np.max(bounded[1:, :] ** 2)
    )

    if not np.all(valid_terminal):
        return {
            "J_contour": float(
                invalid_weight * (2.0 - valid_fraction)
            ),
            "safe_level": np.nan,
            "safe_radius": 0.0,
            "mean_radius": 0.0,
            "max_radius": 0.0,
            "distortion": np.inf,
            "roughness": np.inf,
            "fold_mean": fold_mean,
            "fold_worst": fold_worst,
            "valid_fraction": valid_fraction,
            "r_contour": np.full(n_angles, np.nan),
        }

    raw_level = float(np.min(terminal_I))
    safe_level = float(level_fraction) * raw_level
    r_contour = np.full(
        n_angles, np.nan, dtype=float
    )

    for j in range(n_angles):
        kmax = int(last_good[j])
        Ij = I[: kmax + 1, j]
        Rj = R[: kmax + 1, j]
        k = int(np.searchsorted(
            Ij, safe_level, side="left"
        ))
        if k <= 0:
            r_contour[j] = 0.0
        elif k > kmax:
            continue
        else:
            I0, I1 = Ij[k - 1], Ij[k]
            r0, r1 = Rj[k - 1], Rj[k]
            if I1 <= I0:
                continue
            t = (safe_level - I0) / (I1 - I0)
            r_contour[j] = r0 + t * (r1 - r0)

    closed = (
        np.isfinite(r_contour)
        & (r_contour > 0.0)
    )
    valid_fraction = min(
        valid_fraction, float(np.mean(closed))
    )
    if not np.all(closed):
        return {
            "J_contour": float(
                invalid_weight * (2.0 - valid_fraction)
            ),
            "safe_level": safe_level,
            "safe_radius": 0.0,
            "mean_radius": 0.0,
            "max_radius": 0.0,
            "distortion": np.inf,
            "roughness": np.inf,
            "fold_mean": fold_mean,
            "fold_worst": fold_worst,
            "valid_fraction": valid_fraction,
            "r_contour": r_contour,
        }

    safe_radius = float(np.min(r_contour))
    mean_radius = float(np.mean(r_contour))
    max_radius = float(np.max(r_contour))
    distortion = float(
        (max_radius - safe_radius)
        / max(mean_radius, 1e-12)
    )
    dr = np.roll(r_contour, -1) - r_contour
    roughness = float(
        np.sqrt(np.mean(dr**2))
        / max(mean_radius, 1e-12)
    )

    radius_scale = (
        np.sqrt(float(level_fraction))
        * geometry["box_inscribed_radius"]
    )
    size_penalty = (
        radius_scale
        / max(safe_radius, 1e-12)
    )
    J = (
        float(size_weight) * size_penalty
        + float(distortion_weight) * distortion
        + float(roughness_weight) * roughness
        + float(fold_mean_weight) * fold_mean
        + float(fold_worst_weight) * fold_worst
        + float(invalid_weight)
        * (1.0 - valid_fraction)
    )
    return {
        "J_contour": float(J),
        "safe_level": safe_level,
        "safe_radius": safe_radius,
        "mean_radius": mean_radius,
        "max_radius": max_radius,
        "distortion": distortion,
        "roughness": roughness,
        "fold_mean": fold_mean,
        "fold_worst": fold_worst,
        "valid_fraction": valid_fraction,
        "r_contour": r_contour,
    }


def _safe_contour_area(r_contour):
    r = np.asarray(r_contour, dtype=float)
    if (
        r.size == 0
        or np.any(~np.isfinite(r))
        or np.any(r <= 0.0)
    ):
        return 0.0
    dtheta = 2.0 * np.pi / r.size
    return float(
        0.5 * dtheta * np.sum(r**2)
    )


def _contours_for_slices(
    Ix,
    state,
    ys,
    deltas,
    *,
    frozen_py,
    geometry,
    contour_kwargs,
):
    records = []
    for y, delta in _slice_records(ys, deltas):
        bundle = _slice_bundle(
            Ix,
            state,
            y=y,
            delta=delta,
            py=frozen_py,
        )
        contour = _contour_components(
            bundle["I"],
            geometry,
            **contour_kwargs,
        )
        contour.update({
            "y": y,
            "delta": delta,
            "I_coeff": bundle["I"],
        })
        records.append(contour)
    return records


# =============================================================================
# 7. OBJECTIVE 3 -- ORIGIN-CONNECTED SAFE CONTOUR
# =============================================================================


def contour_objective(
    data,
    *,
    y_values=None,
    delta_values=None,
    frozen_py=0.0,
    aggregation="worst",
    x_max=None,
    px_max=None,
    angles=120,
    radial_points=140,
    level_fraction=0.97,
    min_radial_slope=0.02,
    size_weight=1.00,
    distortion_weight=0.20,
    roughness_weight=0.05,
    fold_mean_weight=0.20,
    fold_worst_weight=0.80,
    invalid_weight=5.00,
):
    """Largest origin-connected monotone contour on each y/delta slice."""
    Ix, state = data["Ix"], data["state"]
    ys, deltas = _slice_values(
        state, y_values, delta_values
    )
    x_max, px_max = _horizontal_limits(
        state, x_max, px_max
    )
    geometry = _build_contour_geometry(
        state,
        x_max=x_max,
        px_max=px_max,
        angles=angles,
        radial_points=radial_points,
    )
    kwargs = {
        "min_radial_slope": min_radial_slope,
        "level_fraction": level_fraction,
        "size_weight": size_weight,
        "distortion_weight": distortion_weight,
        "roughness_weight": roughness_weight,
        "fold_mean_weight": fold_mean_weight,
        "fold_worst_weight": fold_worst_weight,
        "invalid_weight": invalid_weight,
    }
    slices = _contours_for_slices(
        Ix,
        state,
        ys,
        deltas,
        frozen_py=frozen_py,
        geometry=geometry,
        contour_kwargs=kwargs,
    )
    objective = _aggregate_slice_scores(
        [s["J_contour"] for s in slices],
        aggregation,
    )
    diagnostics = [
        {
            k: v
            for k, v in s.items()
            if k not in {"I_coeff", "r_contour"}
        }
        for s in slices
    ]
    return objective, {
        "aggregation": aggregation,
        "slice_scores": diagnostics,
        "worst_safe_radius": float(
            min(s["safe_radius"] for s in slices)
        ),
    }


contour_objective.requires = {"Ix"}


# =============================================================================
# 8. OBJECTIVE 4 -- SAFE BARRIER AREA
# =============================================================================


def barrier_objective(
    data,
    *,
    y_values=None,
    delta_values=None,
    frozen_py=0.0,
    aggregation="worst",
    x_max=None,
    px_max=None,
    angles=120,
    radial_points=140,
    level_fraction=0.97,
    min_radial_slope=0.02,
    invalid_cost=1e6,
):
    """Maximize the safe origin-connected horizontal phase-space area."""
    Ix, state = data["Ix"], data["state"]
    ys, deltas = _slice_values(
        state, y_values, delta_values
    )
    x_max, px_max = _horizontal_limits(
        state, x_max, px_max
    )
    geometry = _build_contour_geometry(
        state,
        x_max=x_max,
        px_max=px_max,
        angles=angles,
        radial_points=radial_points,
    )
    slices = _contours_for_slices(
        Ix,
        state,
        ys,
        deltas,
        frozen_py=frozen_py,
        geometry=geometry,
        contour_kwargs={
            "min_radial_slope": min_radial_slope,
            "level_fraction": level_fraction,
        },
    )

    reference_radius = (
        np.sqrt(float(level_fraction))
        * geometry["box_inscribed_radius"]
    )
    reference_area = float(
        np.pi * reference_radius**2
    )

    details = []
    for contour in slices:
        area = _safe_contour_area(
            contour["r_contour"]
        )
        valid = (
            contour["valid_fraction"]
            >= 1.0 - 1e-12
            and area > 0.0
        )
        score = (
            reference_area / area
            if valid
            else float(invalid_cost)
        )
        details.append({
            "y": contour["y"],
            "delta": contour["delta"],
            "score": float(score),
            "safe_area": float(area),
            "reference_safe_area": reference_area,
            "safe_radius": float(
                contour["safe_radius"]
            ),
            "valid_fraction": float(
                contour["valid_fraction"]
            ),
        })

    objective = _aggregate_slice_scores(
        [d["score"] for d in details],
        aggregation,
    )
    return objective, {
        "aggregation": aggregation,
        "slice_scores": details,
        "worst_safe_area": float(
            min(d["safe_area"] for d in details)
        ),
    }


barrier_objective.requires = {"Ix"}


# =============================================================================
# 9. OBJECTIVES 5-7 -- ACTUAL ONE-TURN NORMAL TRANSPORT
# =============================================================================


def _upper_tail_cvar(values, tail_fraction):
    values = np.asarray(
        values, dtype=float
    ).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.inf

    tail_fraction = float(np.clip(
        tail_fraction,
        1.0 / values.size,
        1.0,
    ))
    count = max(
        1,
        int(np.ceil(
            tail_fraction * values.size
        )),
    )
    if count >= values.size:
        return float(np.mean(values))

    tail = np.partition(
        values,
        values.size - count,
    )[-count:]
    return float(np.mean(tail))


def _transport_slice_components(
    Ix,
    transfer,
    state,
    contour,
    geometry,
    *,
    frozen_py,
    annulus_inner,
    annulus_layers,
    cvar_tail,
    transport_eps,
    invalid_cost,
    level_fraction,
):
    r_contour = np.asarray(
        contour["r_contour"], dtype=float
    )
    valid = (
        contour["valid_fraction"]
        >= 1.0 - 1e-12
        and r_contour.size
        == len(geometry["theta"])
        and np.all(np.isfinite(r_contour))
        and np.all(r_contour > 0.0)
    )
    reference_radius = (
        np.sqrt(float(level_fraction))
        * geometry["box_inscribed_radius"]
    )
    reference_area = float(
        np.pi * reference_radius**2
    )

    transfer = np.asarray(
        transfer, dtype=float
    )
    invariant = np.asarray(
        Ix, dtype=float
    )
    defect_vector = (
        transfer @ invariant - invariant
    )
    defect_coeff = _slice_polynomial_coefficients(
        defect_vector,
        state,
        y=contour["y"],
        delta=contour["delta"],
        py=frozen_py,
    )

    if not valid:
        return {
            "safe_area": 0.0,
            "reference_safe_area": reference_area,
            "transport_cvar": np.inf,
            "transport_mean": np.inf,
            "transport_max": np.inf,
            "J_barrier": float(invalid_cost),
            "J_flux": float(invalid_cost),
            "J_escape": float(invalid_cost),
            "J_ultimate": float(invalid_cost),
        }

    safe_area = _safe_contour_area(
        r_contour
    )
    safe_radius = float(
        contour["safe_radius"]
    )
    fractions = np.linspace(
        float(annulus_inner),
        1.0,
        int(annulus_layers),
    )[:, None]
    R = fractions * r_contour[None, :]
    X = R * geometry["dx_dr"][None, :]
    PX = R * geometry["dpx_dr"][None, :]

    defect = _poly2d(
        defect_coeff, X, PX
    )
    I_coeff = contour["I_coeff"]
    dI_dx = _poly2d(
        _differentiate_2d(I_coeff, 0),
        X,
        PX,
    )
    dI_dpx = _poly2d(
        _differentiate_2d(I_coeff, 1),
        X,
        PX,
    )
    grad_norm = np.sqrt(
        dI_dx**2 + dI_dpx**2
    )
    delta_perp = np.abs(defect) / (
        grad_norm + float(transport_eps)
    )

    transport_cvar = _upper_tail_cvar(
        delta_perp, cvar_tail
    )
    transport_mean = float(
        np.mean(delta_perp)
    )
    transport_max = float(
        np.max(delta_perp)
    )
    area_ratio = (
        reference_area
        / max(
            safe_area,
            float(transport_eps),
        )
    )

    J_barrier = float(area_ratio)
    J_flux = float(transport_cvar)
    J_escape = float(
        transport_cvar
        / max(
            safe_radius,
            float(transport_eps),
        )
    )
    J_ultimate = float(
        J_escape * np.sqrt(area_ratio)
    )
    return {
        "safe_area": safe_area,
        "reference_safe_area": reference_area,
        "transport_cvar": transport_cvar,
        "transport_mean": transport_mean,
        "transport_max": transport_max,
        "J_barrier": J_barrier,
        "J_flux": J_flux,
        "J_escape": J_escape,
        "J_ultimate": J_ultimate,
    }


def _transport_objective(
    data,
    mode,
    *,
    y_values=None,
    delta_values=None,
    frozen_py=0.0,
    aggregation="worst",
    x_max=None,
    px_max=None,
    angles=120,
    radial_points=140,
    level_fraction=0.97,
    min_radial_slope=0.02,
    annulus_inner=0.90,
    annulus_layers=6,
    cvar_tail=0.01,
    transport_eps=1e-12,
    invalid_cost=1e6,
):
    Ix = data["Ix"]
    transfer = data["transfer"]
    state = data["state"]
    ys, deltas = _slice_values(
        state, y_values, delta_values
    )
    x_max, px_max = _horizontal_limits(
        state, x_max, px_max
    )
    geometry = _build_contour_geometry(
        state,
        x_max=x_max,
        px_max=px_max,
        angles=angles,
        radial_points=radial_points,
    )
    contours = _contours_for_slices(
        Ix,
        state,
        ys,
        deltas,
        frozen_py=frozen_py,
        geometry=geometry,
        contour_kwargs={
            "min_radial_slope": min_radial_slope,
            "level_fraction": level_fraction,
        },
    )

    key = {
        "flux": "J_flux",
        "escape": "J_escape",
        "ultimate": "J_ultimate",
    }[mode]
    details = []
    for contour in contours:
        transport = _transport_slice_components(
            Ix,
            transfer,
            state,
            contour,
            geometry,
            frozen_py=frozen_py,
            annulus_inner=annulus_inner,
            annulus_layers=annulus_layers,
            cvar_tail=cvar_tail,
            transport_eps=transport_eps,
            invalid_cost=invalid_cost,
            level_fraction=level_fraction,
        )
        details.append({
            "y": contour["y"],
            "delta": contour["delta"],
            "score": float(
                transport[key]
            ),
            "safe_radius": float(
                contour["safe_radius"]
            ),
            "valid_fraction": float(
                contour["valid_fraction"]
            ),
            **transport,
        })

    objective = _aggregate_slice_scores(
        [d["score"] for d in details],
        aggregation,
    )
    return objective, {
        "aggregation": aggregation,
        "slice_scores": details,
        "mode": mode,
    }


def flux_objective(data, **kwargs):
    """Worst-slice CVaR of normal one-turn transport."""
    return _transport_objective(
        data, "flux", **kwargs
    )


def escape_objective(data, **kwargs):
    """Worst-slice transport divided by the weakest safe radius."""
    return _transport_objective(
        data, "escape", **kwargs
    )


def ultimate_objective(data, **kwargs):
    """Worst-slice escape cost additionally penalized by small safe area."""
    return _transport_objective(
        data, "ultimate", **kwargs
    )


flux_objective.requires = {"Ix", "transfer"}
escape_objective.requires = {"Ix", "transfer"}
ultimate_objective.requires = {"Ix", "transfer"}


# =============================================================================
# 10. TRACKING-BASED A_BOX OBJECTIVE
# =============================================================================


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


def tracked_ix_invariance(
    data, *, floor_fraction=1.0e-8
):
    """RMS relative Ix excursion on a fixed set of physical trajectories."""
    trajectories = data["trajectories"]
    if not trajectories:
        raise ValueError(
            "At least one tracked trajectory is required."
        )

    floor_fraction = float(floor_fraction)
    if (
        not np.isfinite(floor_fraction)
        or floor_fraction <= 0.0
    ):
        raise ValueError(
            "floor_fraction must be positive and finite."
        )

    values_by_track = []
    global_scale = 0.0
    for trajectory in trajectories:
        values = _evaluate_invariant_vector(
            data["Ix"],
            data["state"],
            trajectory,
        )
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        values_by_track.append(values)
        global_scale = max(
            global_scale,
            float(np.max(np.abs(values))),
        )

    if not values_by_track:
        raise ValueError(
            "No trajectory contains at least two finite Ix evaluations."
        )

    if (
        global_scale == 0.0
        or not np.isfinite(global_scale)
    ):
        global_scale = 1.0
    absolute_floor = (
        floor_fraction * global_scale
    )

    per_track_rms = []
    per_track_max = []
    count = 0
    for values in values_by_track:
        reference = float(values[0])
        denominator = max(
            abs(reference),
            absolute_floor,
        )
        relative = (
            values[1:] - reference
        ) / denominator
        relative = relative[
            np.isfinite(relative)
        ]
        if relative.size == 0:
            continue
        per_track_rms.append(
            float(np.sqrt(
                np.mean(relative ** 2)
            ))
        )
        per_track_max.append(
            float(np.max(np.abs(relative)))
        )
        count += int(relative.size)

    if not per_track_rms:
        raise ValueError(
            "No finite relative Ix excursions were produced."
        )

    score = float(np.sqrt(np.mean(
        np.asarray(
            per_track_rms,
            dtype=float,
        ) ** 2
    )))
    return score, {
        "trajectory_rms": per_track_rms,
        "trajectory_max": per_track_max,
        "max_relative_excursion": float(
            max(per_track_max)
        ),
        "normalization_floor": float(
            absolute_floor
        ),
        "tracked_samples": int(count),
    }


tracked_ix_invariance.requires = {
    "Ix",
    "trajectories",
}
