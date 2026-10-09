"""Coupled construction of approximate horizontal/vertical polynomial actions.

The shared quadratic normal-form kernel span(Sx, Sy) supplies admissible
quadratic cross-action corrections. The coefficients of Sx in Ix and Sy in Iy
remain fixed at one, preventing a trivial overall amplitude collapse.

The Poisson bracket uses (x,px),(y,py) in FANQO's [delta,x,y,px,py]
ordering. Products outside the finite polynomial basis are truncated.
"""
import numpy as np
from scipy.optimize import least_squares


def _whitener(metric):
    """Return W with W.T @ W = metric, allowing numerical PSD matrices."""
    g = np.asarray(metric, dtype=float)
    g = (g + g.T) / 2
    try:
        return np.linalg.cholesky(g).T
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(g)
        threshold = max(1.0, np.max(np.abs(values))) * 1e-12
        if np.min(values) < -threshold:
            raise ValueError("Coefficient metric has a negative eigenvalue.")
        return np.sqrt(np.maximum(values, 0.0))[:, None] * vectors.T


def _bracket_indices(state):
    """Sparse tensor triples (k,i,j,v): {e_i,e_j} += v e_k."""
    powers = state["idx_to_vec"]
    lookup = state["vec_to_idx"]
    c = np.asarray(state["C"], dtype=float)
    ks, ii, jj, vv = [], [], [], []
    for i, a in powers.items():
        for j, b in powers.items():
            for q, p in ((1, 3), (2, 4)):
                term = int(a[q]) * int(b[p]) - int(a[p]) * int(b[q])
                if not term:
                    continue
                exponent = tuple(int(x) + int(y) -
                                 (1 if t == q or t == p else 0)
                                 for t, (x, y) in enumerate(zip(a, b)))
                k = lookup.get(exponent)
                if k is not None:
                    ks.append(k)
                    ii.append(i)
                    jj.append(j)
                    vv.append(float(term) * c[i] * c[j] / c[k])
    return (np.asarray(ks, dtype=int), np.asarray(ii, dtype=int),
            np.asarray(jj, dtype=int), np.asarray(vv, dtype=float))


def construct(transfer, tnn, tnq, data, state, tol):
    """Return optimized Ix and diagnostics; store optimized Iy in details."""
    # Import at call time to avoid a circular dependency with nonlinear.py.
    from . import nonlinear as nl

    options = state["method_options"]
    lb = float(options.get("COUPLED_BRACKET_WEIGHT", 1.0))
    ls = float(options.get("COUPLED_QUADRATIC_WEIGHT", 10.0))
    lh = float(options.get("COUPLED_NONLINEAR_WEIGHT", 1e-3))
    if not np.all(np.isfinite((lb, ls, lh))) or min(lb, ls, lh) < 0:
        raise ValueError("Coupled penalty weights must be finite and nonnegative.")

    sx, sy = nl.quadratic_invariants(data, state)
    if sy is None:
        raise ValueError("Coupled invariants require the full x/y polynomial basis.")
    q = state["quad_size"]
    n = state["nonquad_size"]
    size = q + n
    d = np.eye(n) - np.asarray(tnn, dtype=float)
    t = np.asarray(tnq, dtype=float)
    wn = _whitener(state["Gnn"])
    w = _whitener(state["G"])

    sx_full = np.r_[sx, np.zeros(n)]
    sy_full = np.r_[sy, np.zeros(n)]
    sx_norm = float(np.linalg.norm(w @ sx_full))
    sy_norm = float(np.linalg.norm(w @ sy_full))
    if min(sx_norm, sy_norm) < 1e-14:
        raise ValueError("Courant-Snyder norm is too small to normalize.")

    # Quadratic normal-form corrections: delta Sx = cx Sy,
    # delta Sy = cy Sx. Their leading actions retain unit normalization.
    ix0, _ = nl.least_squares_ix(tnn, tnq, sx, state, tol=tol)
    iy0, _ = nl.least_squares_ix(tnn, tnq, sy, state, tol=tol)
    initial = np.r_[ix0[q:], iy0[q:], 0., 0.]
    triples = _bracket_indices(state)
    k, i, j, v = triples

    def bracket(ix, iy):
        return np.bincount(k, weights=v * ix[i] * iy[j], minlength=size)

    def unpack(z):
        hx, hy = z[:n], z[n:2*n]
        cx, cy = z[-2:]
        qx = sx + cx * sy
        qy = sy + cy * sx
        return hx, hy, cx, cy, np.r_[qx, hx], np.r_[qy, hy]

    def residual(z):
        hx, hy, cx, cy, ix, iy = unpack(z)
        pieces = [wn @ (d @ hx - t @ ix[:q]),
                  wn @ (d @ hy - t @ iy[:q])]
        if lb:
            pieces.append(np.sqrt(lb) * (w @ bracket(ix, iy)))
        if ls:
            pieces.extend((np.sqrt(ls) * cx * (w @ sy_full) / sx_norm,
                           np.sqrt(ls) * cy * (w @ sx_full) / sy_norm))
        if lh:
            pieces.extend((np.sqrt(lh) * (wn @ hx) / sx_norm,
                           np.sqrt(lh) * (wn @ hy) / sy_norm))
        return np.concatenate(pieces)

    result = least_squares(
        residual, initial, method="trf", jac="2-point",
        max_nfev=int(options.get("COUPLED_MAX_NFEV", 40)),
        ftol=float(options.get("COUPLED_FTOL", 1e-8)),
        xtol=float(options.get("COUPLED_XTOL", 1e-8)),
        gtol=float(options.get("COUPLED_GTOL", 1e-8)),
    )
    hx, hy, cx, cy, ix, iy = unpack(result.x)
    rx = d @ hx - t @ ix[:q]
    ry = d @ hy - t @ iy[:q]
    b = bracket(ix, iy)
    details = {
        "Iy": iy,
        "quadratic_mixing_x": float(cx),
        "quadratic_mixing_y": float(cy),
        "homological_x": float(np.linalg.norm(wn @ rx)),
        "homological_y": float(np.linalg.norm(wn @ ry)),
        "poisson_bracket": float(np.linalg.norm(w @ b)),
        "relative_quadratic_x": float(abs(cx) * np.linalg.norm(w @ sy_full) / sx_norm),
        "relative_quadratic_y": float(abs(cy) * np.linalg.norm(w @ sx_full) / sy_norm),
        "relative_nonlinear_x": float(np.linalg.norm(wn @ hx) / sx_norm),
        "relative_nonlinear_y": float(np.linalg.norm(wn @ hy) / sy_norm),
        "objective": float(np.dot(result.fun, result.fun)),
        "solver_status": int(result.status),
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "nfev": int(result.nfev),
        "optimality": float(result.optimality),
        "bracket_terms": int(len(v)),
        "bracket_truncated_to_basis": True,
        "nullity_D": int(n - np.linalg.matrix_rank(d)) if n <= 200 else None,
        "nullity_D_note": "Skipped for large matrix" if n > 200 else "Numerical rank diagnostic",
        "weights": {"bracket": lb, "quadratic": ls, "nonlinear": lh},
    }
    return ix, details
