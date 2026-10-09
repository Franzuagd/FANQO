"""Coupled construction of approximate horizontal/vertical polynomial actions.

The shared quadratic normal-form kernel span(Sx, Sy) supplies admissible
quadratic cross-action corrections. The coefficients of Sx in Ix and Sy in Iy
remain fixed at one, preventing a trivial overall amplitude collapse.

The Poisson bracket uses (x,px),(y,py) in FANQO's [delta,x,y,px,py]
ordering. Products outside the finite polynomial basis are truncated.
"""
import numpy as np
from scipy.optimize import minimize


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
    """Optimize the coupled sum of squared residuals with an analytic gradient.

    L-BFGS-B avoids an impractically large dense finite-difference Jacobian
    at FANQO's usual polynomial order. The quadratic terms can be fixed exactly.
    """
    from . import nonlinear as nl

    options = state["method_options"]
    lb = float(options.get("COUPLED_BRACKET_WEIGHT", 1.0))
    ls = float(options.get("COUPLED_QUADRATIC_WEIGHT", 10.0))
    lh = float(options.get("COUPLED_NONLINEAR_WEIGHT", 1e-3))
    if not np.all(np.isfinite((lb, ls, lh))) or min(lb, ls, lh) < 0:
        raise ValueError("Coupled penalty weights must be finite and nonnegative.")

    sx, sy = nl.quadratic_invariants(data, state)
    if sy is None:
        raise ValueError("Coupled construction requires both CS actions.")
    q = int(state["quad_size"])
    n = int(state["nonquad_size"])
    size = q + n
    D = np.eye(n) - np.asarray(tnn, dtype=float)
    T = np.asarray(tnq, dtype=float)
    Gnn = np.asarray(state["Gnn"], dtype=float)
    G = np.asarray(state["G"], dtype=float)

    sx_full = np.r_[sx, np.zeros(n)]
    sy_full = np.r_[sy, np.zeros(n)]
    sx_sq = float(sx_full @ G @ sx_full)
    sy_sq = float(sy_full @ G @ sy_full)
    if min(sx_sq, sy_sq) <= 1e-28:
        raise ValueError("Courant-Snyder norm is too small to normalize.")

    fixed = bool(options.get("COUPLED_FIX_QUADRATIC", False))
    ix0, _ = nl.least_squares_ix(tnn, tnq, sx, state, tol=tol)
    iy0, _ = nl.least_squares_ix(tnn, tnq, sy, state, tol=tol)
    z0 = np.r_[ix0[q:], iy0[q:]] if fixed else np.r_[ix0[q:], iy0[q:], 0., 0.]

    k, i, j, v = _bracket_indices(state)
    Ty, Tx = T @ sy, T @ sx

    def unpack(z):
        hx, hy = z[:n], z[n:2*n]
        cx, cy = (0., 0.) if fixed else z[-2:]
        ix = np.r_[sx + cx * sy, hx]
        iy = np.r_[sy + cy * sx, hy]
        return hx, hy, float(cx), float(cy), ix, iy

    def value_gradient(z):
        hx, hy, cx, cy, ix, iy = unpack(z)
        rx = D @ hx - T @ ix[:q]
        ry = D @ hy - T @ iy[:q]
        grx = Gnn @ rx
        gry = Gnn @ ry
        ghx = Gnn @ hx
        ghy = Gnn @ hy
        b = np.bincount(k, weights=v * ix[i] * iy[j], minlength=size)
        gb = G @ b
        fx = float(rx @ grx)
        fy = float(ry @ gry)
        fb = float(b @ gb)
        fs = cx*cx*sy_sq/sx_sq + cy*cy*sx_sq/sy_sq
        fh = float(hx @ ghx)/sx_sq + float(hy @ ghy)/sy_sq
        f = fx + fy + lb*fb + ls*fs + lh*fh

        dx = 2.0 * (D.T @ grx + lh*ghx/sx_sq)
        dy = 2.0 * (D.T @ gry + lh*ghy/sy_sq)
        if lb:
            # Derivatives of b_k = sum_ij B_kij Ix_i Iy_j.
            bix = np.bincount(
                i, weights=v * gb[k] * iy[j], minlength=size
            )
            biy = np.bincount(
                j, weights=v * gb[k] * ix[i], minlength=size
            )
            dx += 2.0 * lb * bix[q:]
            dy += 2.0 * lb * biy[q:]
        else:
            bix = np.zeros(size)
            biy = np.zeros(size)

        gradient = np.r_[dx, dy]
        if not fixed:
            dcx = 2.0 * (-Ty @ grx + lb * (sy @ bix[:q])
                          + ls * cx * sy_sq / sx_sq)
            dcy = 2.0 * (-Tx @ gry + lb * (sx @ biy[:q])
                          + ls * cy * sx_sq / sy_sq)
            gradient = np.r_[gradient, dcx, dcy]
        return f, gradient

    result = minimize(
        value_gradient, z0, jac=True, method="L-BFGS-B",
        options={
            "maxiter": int(options.get("COUPLED_MAX_ITER",
                                         options.get("COUPLED_MAX_NFEV", 200))),
            "ftol": float(options.get("COUPLED_FTOL", 1e-12)),
            "gtol": float(options.get("COUPLED_GTOL", 1e-8)),
            "maxls": 30,
        },
    )

    hx, hy, cx, cy, ix, iy = unpack(result.x)
    rx = D @ hx - T @ ix[:q]
    ry = D @ hy - T @ iy[:q]
    b = np.bincount(k, weights=v * ix[i] * iy[j], minlength=size)
    details = {
        "Iy": iy,
        "quadratic_fixed": fixed,
        "quadratic_mixing_x": cx,
        "quadratic_mixing_y": cy,
        "homological_x": float(np.sqrt(max(rx @ Gnn @ rx, 0.))),
        "homological_y": float(np.sqrt(max(ry @ Gnn @ ry, 0.))),
        "poisson_bracket": float(np.sqrt(max(b @ G @ b, 0.))),
        "relative_quadratic_x": float(abs(cx) * np.sqrt(sy_sq/sx_sq)),
        "relative_quadratic_y": float(abs(cy) * np.sqrt(sx_sq/sy_sq)),
        "relative_nonlinear_x": float(np.sqrt(max(hx @ Gnn @ hx, 0.)/sx_sq)),
        "relative_nonlinear_y": float(np.sqrt(max(hy @ Gnn @ hy, 0.)/sy_sq)),
        "objective": float(result.fun),
        "solver_status": int(result.status),
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "optimality": float(np.linalg.norm(result.jac, ord=np.inf)),
        "bracket_terms": int(len(v)),
        "bracket_truncated_to_basis": True,
        "nullity_D": None,
        "nullity_D_note": "Large SVD omitted; run separate rank diagnostic",
        "weights": {"bracket": lb, "quadratic": ls, "nonlinear": lh},
    }
    return ix, details
