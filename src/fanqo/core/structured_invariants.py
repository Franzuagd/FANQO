"""Experimental pair constructors; no changes to the legacy eigen/LS methods.

Graded coupled: small linear weighted solves, including delta-linear terms.
Canonical graded: a composition of shared Lie generators, with transverse
order >= 2. This restriction makes rectangular (m,d) bracket truncation safe,
but excludes dispersion-like delta*x terms. Both new methods fix pure delta
coefficients to zero and keep the on-momentum CS quadratic sectors exact.
"""
import math
import numpy as np
from scipy.linalg import block_diag

PAIR_METHODS = (
    "a_box_coupled_fixed", "a_box_coupled_regularized",
    "graded_coupled", "canonical_graded",
)


def least_squares_seed(tnn, tnq, seed, state, tol=1e-14):
    D = np.eye(len(tnn)) - tnn
    W = np.linalg.cholesky(state["Gnn"]).T
    h, _, rank, _ = np.linalg.lstsq(W @ D, W @ (tnq @ seed), rcond=tol)
    return np.r_[seed, h], {"rank": int(rank)}


def bracket_tensor(state):
    """Cache the *physical* bracket in the stored e_i=C_i*z^alpha basis."""
    if "_pair_tensor" not in state:
        from .coupled_invariants import _bracket_indices
        state["_pair_tensor"] = _bracket_indices(state)
    return state["_pair_tensor"]


def bracket(left, right, state):
    k, i, j, v = bracket_tensor(state)
    return np.bincount(k, weights=v * left[i] * right[j], minlength=len(left))


def bracket_matrix(fixed, state, *, fixed_left=True):
    """Columns are {fixed,e_j}, or {e_j,fixed}."""
    k, i, j, v = bracket_tensor(state)
    N = len(fixed)
    values = v * (fixed[i] if fixed_left else fixed[j])
    columns = j if fixed_left else i
    return np.bincount(k*N+columns, weights=values,
                       minlength=N*N).reshape(N, N)


def seeds(data, state):
    from .nonlinear import quadratic_invariants
    sx, sy = quadratic_invariants(data, state)
    if sy is None:
        raise ValueError("Pair construction requires both transverse planes.")
    n = state["nonquad_size"]
    return np.r_[sx, np.zeros(n)], np.r_[sy, np.zeros(n)]


def grade_groups(state):
    groups = {}
    for i, a in state["idx_to_vec"].items():
        groups.setdefault((int(a[0]), sum(a[1:])), []).append(i)
    return [(g, np.array(ids, dtype=int)) for g, ids in
            sorted(groups.items(), key=lambda kv: (sum(kv[0]), kv[0][0]))]


def _weights(state):
    o = state.get("method_options", {})
    lb = float(o.get("STRUCTURED_BRACKET_WEIGHT", 1.0))
    ridge = float(o.get("STRUCTURED_RIDGE", 1e-10))
    if not np.isfinite([lb, ridge]).all() or min(lb, ridge) < 0:
        raise ValueError("Structured weights must be finite and nonnegative.")
    return lb, ridge


def _norm(c, G):
    return float(np.sqrt(max(float(c @ G @ c), 0.0)))


def pair_diagnostics(transfer, ix, iy, sx, sy, state):
    G = state["G"]
    nx, ny = _norm(sx, G), _norm(sy, G)
    b = bracket(ix, iy, state)
    # Bracket scale has action units for the canonical coordinates used here.
    bs = math.sqrt(nx*ny)
    q = state["quad_size"]
    return {
        "homological_x": _norm(transfer @ ix-ix, G),
        "homological_y": _norm(transfer @ iy-iy, G),
        "relative_homological_x": _norm(transfer @ ix-ix, G)/nx,
        "relative_homological_y": _norm(transfer @ iy-iy, G)/ny,
        "poisson_bracket": _norm(b, G),
        "relative_poisson_bracket": _norm(b, G)/bs,
        "quadratic_error_x": _norm(ix[:q]-sx[:q], G[:q, :q]),
        "quadratic_error_y": _norm(iy[:q]-sy[:q], G[:q, :q]),
        "central_coefficient_norm_x": float(np.linalg.norm([
            ix[i] for i,a in state["idx_to_vec"].items() if a[0] and not sum(a[1:])])),
        "central_coefficient_norm_y": float(np.linalg.norm([
            iy[i] for i,a in state["idx_to_vec"].items() if a[0] and not sum(a[1:])])),
    }


def _check_structure(transfer, state):
    if state.get("horizontal_slice_only", False):
        raise ValueError("Structured pairs require a full two-plane basis.")
    if state["d"] not in (0, 1):
        raise ValueError("Structured constructors currently support d=0 or d=1.")
    if state["m"] < 3:
        raise ValueError("Structured constructors require m>=3.")
    groups = grade_groups(state)
    order = np.empty(len(transfer), dtype=int)
    for t, (_, rows) in enumerate(groups):
        order[rows] = t
    forbidden = order[:, None] < order[None, :]
    leakage = float(np.max(np.abs(transfer[forbidden]), initial=0.0))
    return groups, leakage


def _solve(A, b, regularizer, ridge, tol):
    if ridge:
        A = np.vstack((A, math.sqrt(ridge)*regularizer))
        b = np.r_[b, np.zeros(regularizer.shape[0])]
    h, _, rank, singular = np.linalg.lstsq(A, b, rcond=tol)
    return h, int(rank), float(np.linalg.norm(A @ h-b))


def construct_graded(transfer, data, state, tol=1e-14):
    """Soft coupled bracket least squares at successive homogeneous blocks.

    Each block is linear. Earlier coefficients are frozen, so this is NOT
    the minimizer of the global quartic coupled objective. The bracket is
    penalized, not asserted identically zero.
    """
    groups, leakage = _check_structure(transfer, state)
    if leakage > 1e-10*max(1., np.max(np.abs(transfer))):
        return _construct_full_residual(transfer, data, state, tol, canonical=False)
    sx, sy = seeds(data, state)
    ix, iy = sx.copy(), sy.copy()
    R = np.asarray(transfer)-np.eye(len(transfer))
    G = state["G"]
    nx, ny = _norm(sx, G), _norm(sy, G)
    bs = math.sqrt(nx*ny)
    lb, ridge = _weights(state)
    blocks = []
    for (dd, td), rows in groups:
        if (dd == 0 and td <= 2) or td == 0:
            continue
        W = np.linalg.cholesky(G[np.ix_(rows, rows)]).T
        T = R[np.ix_(rows, rows)]
        n = len(rows)
        Ax, Ay = W @ T/nx, W @ T/ny
        A = block_diag(Ax, Ay)
        b = -np.r_[W @ (R[rows] @ ix)/nx, W @ (R[rows] @ iy)/ny]
        if lb:
            Cx = bracket_matrix(iy, state, fixed_left=False)[np.ix_(rows, rows)]
            Cy = bracket_matrix(ix, state, fixed_left=True)[np.ix_(rows, rows)]
            A = np.vstack((A, math.sqrt(lb)*W @ np.hstack((Cx, Cy))/bs))
            b = np.r_[b, -math.sqrt(lb)*W @ bracket(ix, iy, state)[rows]/bs]
        regularizer = block_diag(W/nx, W/ny)
        h, rank, error = _solve(A, b, regularizer, ridge, tol)
        ix[rows], iy[rows] = h[:n], h[n:]
        blocks.append({"grade": [dd, td], "unknowns": 2*n,
                       "rank": rank, "weighted_residual": error})
    return ix, iy, {"method": "graded_coupled", "blocks": blocks,
                    "triangular_leakage": leakage, "ridge": ridge,
                    "bracket_weight": lb, "bracket_exact_by_construction": False,
                    **pair_diagnostics(transfer, ix, iy, sx, sy, state)}


def lie_transform(coefficients, generator, state):
    """Finite exact jet exponential exp({.,chi}) on the safe subalgebra.

    Generators must have transverse degree >=2 and total degree >=3.
    Then they never decrease transverse degree and strictly raise total
    degree: no discarded rectangular-cutoff term can return.
    """
    nz = np.flatnonzero(generator)
    if not len(nz):
        return coefficients.copy()
    grades = [state["idx_to_vec"][i] for i in nz]
    if any(sum(a[1:]) < 2 or sum(a) < 3 for a in grades):
        raise ValueError("Canonical generator must have transverse order >=2 and total order >=3.")
    step = min(sum(a)-2 for a in grades)
    minimum = min((sum(state["idx_to_vec"][i])
                   for i in np.flatnonzero(coefficients)), default=0)
    terms = (state["m"]+state["d"]-minimum)//step
    out = coefficients.copy()
    term = coefficients.copy()
    for k in range(1, terms+1):
        term = bracket(term, generator, state)/k
        out += term
    if not np.isfinite(out).all():
        raise FloatingPointError("Non-finite canonical jet; reduce domain/order or increase ridge.")
    return out


def construct_canonical(transfer, data, state, tol=1e-14):
    """Fit a sequence of shared generators by grade, enforcing involution.

    At the active grade, increments are {Sx,chi_r}, {Sy,chi_r}; fit their
    homological residuals linearly, then apply the complete finite Lie jet
    to BOTH invariants. This keeps all induced higher-order corrections.
    """
    groups, leakage = _check_structure(transfer, state)
    if leakage > 1e-10*max(1., np.max(np.abs(transfer))):
        return _construct_full_residual(transfer, data, state, tol, canonical=True)
    sx, sy = seeds(data, state)
    ix, iy = sx.copy(), sy.copy()
    N = len(ix)
    R = np.asarray(transfer)-np.eye(N)
    G = state["G"]
    nx, ny = _norm(sx, G), _norm(sy, G)
    _, ridge = _weights(state)
    Bx = bracket_matrix(sx, state)
    By = bracket_matrix(sy, state)
    blocks, generators = [], []
    for (dd, td), rows in groups:
        if td < 2 or dd+td < 3:
            continue
        W = np.linalg.cholesky(G[np.ix_(rows, rows)]).T
        Cx, Cy = Bx[:, rows], By[:, rows]
        A = np.vstack((W @ (R[rows] @ Cx)/nx, W @ (R[rows] @ Cy)/ny))
        b = -np.r_[W @ (R[rows] @ ix)/nx, W @ (R[rows] @ iy)/ny]
        reg = np.vstack((W @ Cx[rows]/nx, W @ Cy[rows]/ny))
        h, rank, error = _solve(A, b, reg, ridge, tol)
        chi = np.zeros(N)
        chi[rows] = h
        ix, iy = lie_transform(ix, chi, state), lie_transform(iy, chi, state)
        generators.append(chi)
        blocks.append({"grade": [dd, td], "unknowns": len(rows),
                       "rank": rank, "weighted_residual": error})
    return ix, iy, {"method": "canonical_graded", "blocks": blocks,
                    "triangular_leakage": leakage, "ridge": ridge,
                    "bracket_exact_by_construction": True,
                    "dispersion_linear_terms_included": False,
                    "generator_count": len(generators),
                    "canonical_generators": generators,
                    **pair_diagnostics(transfer, ix, iy, sx, sy, state)}


def construct_pair(transfer, data, state, tol=1e-14):
    method = state["invariant_construction"]
    if method == "graded_coupled":
        return construct_graded(transfer, data, state, tol)
    if method == "canonical_graded":
        return construct_canonical(transfer, data, state, tol)
    if method in PAIR_METHODS[:2]:
        from .coupled_invariants import construct
        q = state["quad_size"]
        ix, details = construct(transfer, transfer[q:, q:], transfer[q:, :q],
                                data, state, tol)
        details = dict(details)
        iy = details.pop("Iy")
        return ix, iy, {"method": method, **details}
    raise ValueError(f"Unknown pair construction: {method}")


def _construct_full_residual(transfer, data, state, tol, *, canonical):
    """Non-triangular fallback: fit every residual row simultaneously.

    This deliberately costs more than the graded path. No map entry is
    discarded. Canonical fitting uses one shared finite Lie exponential;
    the other fit uses the exact bilinear bracket, not a frozen linearization.
    """
    from scipy.optimize import least_squares
    sx, sy = seeds(data, state)
    R = np.asarray(transfer)-np.eye(len(sx))
    W = np.linalg.cholesky(state['G']).T
    nx, ny = _norm(sx, state['G']), _norm(sy, state['G'])
    lb, ridge = _weights(state)
    ids = np.array([i for i,a in state['idx_to_vec'].items()
                    if sum(a[1:]) >= 2 and sum(a) >= 3], dtype=int) if canonical else np.array([
                        i for i,a in state['idx_to_vec'].items()
                        if sum(a[1:]) > 0 and not (a[0] == 0 and sum(a[1:]) <= 2)], dtype=int)
    def unpack(v):
        if canonical:
            chi = np.zeros(len(sx)); chi[ids] = v
            return lie_transform(sx, chi, state), lie_transform(sy, chi, state)
        ix, iy = sx.copy(), sy.copy()
        ix[ids] += v[:len(ids)]; iy[ids] += v[len(ids):]
        return ix, iy
    def residual(v):
        ix, iy = unpack(v)
        parts = [W @ (R @ ix)/nx, W @ (R @ iy)/ny]
        if not canonical and lb:
            parts.append(math.sqrt(lb)*W @ bracket(ix, iy, state)/math.sqrt(nx*ny))
        if ridge:
            parts.extend([math.sqrt(ridge)*W @ (ix-sx)/nx,
                          math.sqrt(ridge)*W @ (iy-sy)/ny])
        return np.concatenate(parts)
    fit = least_squares(residual, np.zeros(len(ids)*(1 if canonical else 2)),
                        max_nfev=int(state.get('method_options', {}).get('STRUCTURED_MAX_NFEV', 200)))
    ix, iy = unpack(fit.x)
    return ix, iy, dict(method='canonical_graded' if canonical else 'graded_coupled',
                       fitting_strategy='full_residual_nontriangular',
                       solver_success=bool(fit.success), solver_message=str(fit.message),
                       nfev=int(fit.nfev), bracket_exact_by_construction=canonical,
                       **pair_diagnostics(transfer, ix, iy, sx, sy, state))
