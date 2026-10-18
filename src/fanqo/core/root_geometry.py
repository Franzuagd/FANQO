"""Horizontal action--angle roots and constrained, tracking-free refinement.

J is FANQO's S=gamma*x^2+2*alpha*x*px+beta*px^2 (twice canonical action).
All geometry here is at delta=0, y=py=0. Polynomial storage is unchanged;
the radial--Fourier representation is evaluated exactly from that storage.
"""
import numpy as np
from scipy.optimize import minimize

from .action_a_box import solve as action_solve


def from_action_angle(action, phi, beta, alpha):
    """Return q,p with phi=atan2(-(alpha*q+beta*p),q)."""
    action, phi = np.broadcast_arrays(np.asarray(action, float), np.asarray(phi, float))
    if beta <= 0 or np.any(action < 0):
        raise ValueError('Actions must be nonnegative and beta positive.')
    radius = np.sqrt(action)
    q = np.sqrt(beta) * radius * np.cos(phi)
    p = -radius * (np.sin(phi) + alpha*np.cos(phi)) / np.sqrt(beta)
    return q, p


def to_action_angle(q, p, beta, alpha):
    """Return FANQO action and wrapped angle; angle at the origin is NaN."""
    X = np.asarray(q, float) / np.sqrt(beta)
    P = alpha*X + np.sqrt(beta)*np.asarray(p, float)
    action = X*X + P*P
    phi = np.mod(np.arctan2(-P, X), 2*np.pi)
    return action, np.where(action > 0, phi, np.nan)


def radial_basis(state, angles, action_scale):
    """B[a,k,h]: stored horizontal term h at angle a times u**k.

    u=sqrt(J/action_scale); nonhorizontal and chromatic terms vanish exactly.
    Only the surviving term indices are returned, keeping tensors small.
    """
    angles = np.asarray(angles, float)
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError('Angles must be a finite 1D array.')
    if not np.isfinite(action_scale) or action_scale <= 0:
        raise ValueError('action_scale must be positive.')
    powers = np.array([state['idx_to_vec'][i] for i in range(len(state['C']))])
    indices = np.flatnonzero(np.all(powers[:, [0, 2, 4]] == 0, axis=1))
    degrees = powers[indices, 1] + powers[indices, 3]
    bx, ax = state['linear_cs0'][:2]
    x, px = from_action_angle(action_scale, angles, bx, ax)
    basis = np.zeros((len(angles), int(degrees.max())+1, len(indices)))
    for j, (i, degree) in enumerate(zip(indices, degrees)):
        basis[:, degree, j] = (state['C'][i] * x**powers[i, 1]
                               * px**powers[i, 3])
    return indices, basis


def positive_roots(coefficients, *, imaginary_tolerance=1e-7):
    """All distinct nonnegative u roots, scaled polynomial in ascending order.

    Exact zero leading coefficients are removed; small nonzero ones are kept
    so remote branches are not silently discarded. None means identically zero.
    """
    c = np.asarray(coefficients, float)
    if not np.isfinite(c).all():
        return np.empty(0)
    nz = np.flatnonzero(c != 0)
    if not len(nz):
        return None
    c = c[:nz[-1]+1] / np.max(np.abs(c))
    roots = np.polynomial.polynomial.polyroots(c)
    real = roots.real[np.abs(roots.imag) <= imaginary_tolerance*np.maximum(1, np.abs(roots.real))]
    real = np.sort(real[real >= -imaginary_tolerance])
    unique = []
    for r in np.maximum(real, 0):
        if not unique or abs(r-unique[-1]) > 1e-6*max(1, abs(r)):
            unique.append(float(r))
    return np.asarray(unique)


def trace_radial_branch(radial, launch_action, action_scale, *, slope_floor=1e-7,
                        max_action=None, max_step_fraction=.5):
    """Continue roots from the known launch root at the first angle.

    Angles must include the repeated endpoint (normally 0 and 2*pi). Missing,
    degenerate, ill-conditioned or discontinuous steps terminate the branch;
    they are never bridged. This is algebraic continuation, not tracking.
    All root lists remain available even when continuation terminates.
    """
    radial = np.asarray(radial, float) / action_scale
    u0 = np.sqrt(launch_action/action_scale)
    level = np.polynomial.polynomial.polyval(u0, radial[0])
    polynomials = radial.copy(); polynomials[:, 0] -= level
    all_roots = [positive_roots(c) for c in polynomials]
    values = np.full(len(radial), np.nan)
    slopes = values.copy(); previous = u0
    status = 'complete'
    for a, roots in enumerate(all_roots):
        if roots is None:
            status = 'degenerate_polynomial'; break
        if len(roots) == 0:
            status = 'missing_positive_root'; break
        r = float(roots[np.argmin(abs(roots-previous))])
        derivative = np.polynomial.polynomial.polyval(
            r, np.polynomial.polynomial.polyder(radial[a]))
        slope = derivative/(2*r) if r > 0 else np.nan
        J = action_scale*r*r
        if a == 0 and not np.isclose(r, u0, rtol=1e-5, atol=1e-8):
            status = 'launch_root_not_resolved'; break
        if not np.isfinite(slope) or abs(slope) <= slope_floor:
            status = 'multiple_or_ill_conditioned_root'; break
        if a and np.sign(slope) != np.sign(slopes[a-1]):
            status = 'branch_orientation_changed'; break
        if a and abs(J-values[a-1]) > max_step_fraction*max(launch_action, values[a-1]):
            status = 'unresolved_branch_step'; break
        if max_action is not None and J > max_action:
            status = 'outside_search_domain'; break
        values[a] = J; slopes[a] = slope; previous = r
    if status == 'complete' and not np.isclose(values[-1], launch_action, rtol=1e-5, atol=1e-12*action_scale):
        status = 'not_closed'
    complete = status == 'complete'
    # The endpoint duplicates the first angle and must not be counted twice.
    g = values[:-1]
    stats = {}
    if complete:
        mean = float(np.mean(g)); variance = float(np.mean((g-mean)**2))
        C = level*action_scale
        stats = dict(mean=mean, std=np.sqrt(variance), bias=mean-C,
                     mean_absolute_deviation=float(np.mean(abs(g-C))),
                     rms_deviation=float(np.sqrt(np.mean((g-C)**2))),
                     relative_launch_rms=float(np.sqrt(np.mean((g/launch_action-1)**2))),
                     max_action=float(np.max(g)), min_abs_slope=float(np.min(abs(slopes))))
    return dict(action=values, slopes=slopes, level=float(level*action_scale),
                all_roots=[None if r is None else action_scale*r*r for r in all_roots],
                complete=complete, status=status, statistics=stats)


def root_curves(invariant, state, launch_actions, angles, action_scale, **kwargs):
    """Curves C=P(J_launch,phi=0), independent of tracking samples."""
    indices, basis = radial_basis(state, angles, action_scale)
    radial = basis @ np.asarray(invariant)[indices]
    return [trace_radial_branch(radial, J, action_scale, **kwargs) for J in launch_actions]


def solve(transfer, seed, state, tol=1e-14):
    """Refine horizontal root geometry, constrained by action-domain residual.

    The CS seed block stays fixed. Weak algebraic directions in the entire
    tail allow coupled/chromatic compensation for a horizontal geometry edit.
    The objective itself remains at delta=0, Jy=0. No tracking import.
    """
    initial, seed_details = action_solve(transfer, seed, state, tol)
    options = state.get('method_options', {})
    scale = float(state['action_box']['Jx_max'])
    launches = scale*np.asarray(options.get('ROOT_ACTION_FRACTIONS', (.05, .15, .3, .5, .7)), float)
    n_angles = int(options.get('ROOT_ANGLES', 64))
    maxiter = int(options.get('ROOT_MAX_ITER', 60))
    factor = float(options.get('ROOT_RESIDUAL_FACTOR', 1.1))
    floor = float(options.get('ROOT_RESIDUAL_FLOOR', 1e-8))
    directions = int(options.get('ROOT_DIRECTIONS', 12))
    weak_modes = int(options.get('ROOT_WEAK_MODES', 64))
    if (n_angles < 8 or maxiter < 0 or factor < 1 or floor <= 0
            or len(launches) == 0 or not np.isfinite(launches).all() or np.any(launches <= 0)
            or not np.isfinite([factor, floor]).all() or directions < 1 or weak_modes < 1):
        raise ValueError('Invalid ROOT_* refinement options.')
    angles = np.linspace(0, 2*np.pi, n_angles+1)
    indices, basis = radial_basis(state, angles, scale)
    q = state['quad_size']
    local_free = np.flatnonzero(indices >= q)
    free = np.arange(q, len(initial))
    radial0 = basis @ initial[indices]
    curves0 = [trace_radial_branch(radial0, J, scale, max_action=4*scale) for J in launches]
    fit = [i for i, c in enumerate(curves0) if c['complete']]
    W = state['action_weight']; R = np.asarray(transfer)-np.eye(len(transfer))
    weighted = W @ R
    seed_norm = max(float(np.linalg.norm(W @ np.r_[seed, np.zeros(len(initial)-len(seed))])), np.finfo(float).tiny)
    e0 = weighted @ initial / seed_norm
    limit = max(factor*np.linalg.norm(e0), floor)
    details = dict(method='root_geometry', seed_method='action_a_box', seed_details=seed_details,
                   geometry_slice='delta=0, Jy=0 (phi_y undefined)',
                   action_convention=state['action_convention'],
                   objective='mean squared (g_C(phi)/J_launch-1), C=P(J_launch,0)',
                   residual_initial=float(np.linalg.norm(e0)), residual_limit=float(limit),
                   fit_launch_actions=launches[fit].tolist(),
                   excluded_launches=[dict(action=float(J), status=c['status'])
                                      for J, c in zip(launches, curves0) if not c['complete']],
                   accepted=False, root_angles=n_angles, max_iterations=maxiter,
                   residual_factor=factor, residual_floor=floor,
                   adjustable_coefficients=int(len(free)))
    if not fit or not len(free):
        details.update(status='no_complete_seed_branches' if not fit else 'no_free_horizontal_terms',
                       residual_final=float(np.linalg.norm(e0)))
        return initial, details
    # First retain weak full-tail algebraic directions. A purely horizontal
    # edit may need vertical/chromatic compensation to stay approximately
    # invariant. Normalize columns by their RMS in the action metric.
    column_scale = seed_norm/np.sqrt(np.diag(state['G'])[free])
    A = weighted[:, free]*column_scale / seed_norm
    _, singular, vh = np.linalg.svd(A, full_matrices=False)
    count = min(weak_modes, len(free))
    steps = np.minimum(1., limit/(np.sqrt(count)*np.maximum(singular[-count:], 1e-30)))
    B = column_scale[:, None]*vh[-count:].T*steps
    horizontal_rows = indices[local_free]-q
    # Rank directions by visible root-geometry changes across fitting radii,
    # rather than spending optimizer variables on invisible chromatic modes.
    visible = []
    for J in launches[fit]:
        u = np.sqrt(J/scale)
        phi_values = np.einsum('akh,k->ah', basis[:, :, local_free],
                               u**np.arange(basis.shape[1]))
        visible.append(phi_values @ B[horizontal_rows]/J)
    _, visible_singular, rotation = np.linalg.svd(np.vstack(visible), full_matrices=False)
    keep = min(directions, int(np.sum(visible_singular > max(visible_singular[0]*1e-12, 1e-30))))
    if keep == 0:
        details.update(status='no_visible_algebraic_directions', residual_final=float(np.linalg.norm(e0)))
        return initial, details
    B = B @ rotation[:keep].T
    tangent = np.einsum('akh,hv->akv', basis[:, :, local_free], B[horizontal_rows])/scale
    E = weighted[:, free] @ B / seed_norm
    details.update(optimized_directions=keep, weak_modes=count)
    best = {'score': np.inf, 'z': np.zeros(keep)}
    cache = {}

    def evaluate(z):
        if 'z' in cache and np.array_equal(cache['z'], z):
            return cache['value']
        radial = radial0/scale + tangent @ z
        residuals = []; gradients = []; valid = True
        for k in fit:
            curve = trace_radial_branch(radial*scale, launches[k], scale, max_action=4*scale)
            if not curve['complete']:
                valid = False; break
            u = np.sqrt(curve['action']/scale)
            u0 = np.sqrt(launches[k]/scale)
            powers = np.arange(radial.shape[1])
            dC = (u0**powers) @ tangent[0]
            dg = np.empty((n_angles, len(z)))
            for a, r in enumerate(u[:-1]):
                dP = (r**powers) @ tangent[a]
                derivative = np.polynomial.polynomial.polyval(r, np.polynomial.polynomial.polyder(radial[a]))
                dg[a] = 2*scale*r*(dC-dP)/derivative/launches[k]
            residuals.extend(curve['action'][:-1]/launches[k]-1)
            gradients.extend(dg)
        if valid:
            residuals = np.asarray(residuals); gradients = np.asarray(gradients)
            score = float(np.mean(residuals**2))
            gradient = 2*residuals @ gradients/len(residuals)
            feasible = np.linalg.norm(e0+E@z) <= limit*(1+1e-8)
            if feasible and score < best['score']:
                best.update(score=score, z=np.array(z, copy=True))
        else:
            score = 1e6; gradient = np.zeros_like(z)
        value = score, gradient
        cache.update(z=np.array(z, copy=True), value=value)
        return value

    z0 = np.zeros(keep); initial_score = evaluate(z0)[0]
    constraint = dict(type='ineq',
        fun=lambda z: 1-np.dot(e0+E@z, e0+E@z)/limit**2,
        jac=lambda z: -2*(e0+E@z) @ E/limit**2)
    result = minimize(lambda z: evaluate(z)[0], z0, jac=lambda z: evaluate(z)[1],
                      method='SLSQP', bounds=[(-2., 2.)]*keep,
                      constraints=[constraint], options=dict(maxiter=maxiter, ftol=1e-12))
    evaluate(result.x)
    candidate = initial.copy(); candidate[free] += B @ best['z']
    accepted = best['score'] < initial_score-max(1e-14, 1e-8*initial_score)
    if not accepted:
        candidate = initial
    final_norm = float(np.linalg.norm(weighted @ candidate)/seed_norm)
    # Independent postcondition; do not return a numerically infeasible fit.
    if final_norm > limit*(1+1e-7):
        candidate = initial; accepted = False; final_norm = float(np.linalg.norm(e0))
    # Check the actual returned polynomial on a denser, independent angular
    # grid. A fitting-grid improvement alone does not justify accepting it.
    validation = {}
    if accepted:
        dense_angles = np.linspace(0, 2*np.pi, 2*n_angles+1)
        initial_dense = root_curves(initial, state, launches[fit], dense_angles, scale,
                                    max_action=4*scale)
        candidate_dense = root_curves(candidate, state, launches[fit], dense_angles, scale,
                                      max_action=4*scale)
        complete = all(c['complete'] for c in initial_dense+candidate_dense)
        if complete:
            before = float(np.mean([c['statistics']['relative_launch_rms']**2 for c in initial_dense]))
            after = float(np.mean([c['statistics']['relative_launch_rms']**2 for c in candidate_dense]))
            validation.update(validation_objective_initial=before, validation_objective_final=after,
                              validation_angles=2*n_angles)
        if not complete or after >= before-max(1e-14, 1e-8*before):
            candidate = initial; accepted = False; final_norm = float(np.linalg.norm(e0))
    details.update(status='refined' if accepted else 'seed_retained', accepted=bool(accepted),
                   objective_initial=initial_score,
                   objective_final=best['score'] if accepted else initial_score,
                   residual_final=final_norm, optimizer_success=bool(result.success),
                   optimizer_message=str(result.message), iterations=int(result.nit), **validation)
    return candidate, details
