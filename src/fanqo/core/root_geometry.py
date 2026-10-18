"""Horizontal action--angle roots; construction is direct radial--Fourier LS.

J is FANQO's S=gamma*x^2+2*alpha*x*px+beta*px^2 (twice canonical action).
All geometry here is at delta=0, y=py=0. Polynomial storage is unchanged;
the radial--Fourier representation is evaluated exactly from that storage.
"""
import numpy as np



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


def all_level_roots(invariant, state, levels, angles, action_scale, *, radial_coefficients=None):
    """Every nonnegative real root for each fixed C at all sampled angles.

    No continuation, nearest-root selection, slope filter or amplitude cutoff.
    The repeated endpoint is evaluated from the same coefficients as phi=0.
    """
    levels = np.asarray(levels, float)
    if levels.ndim != 1 or not len(levels) or not np.isfinite(levels).all():
        raise ValueError('Levels must be a nonempty finite 1D array.')
    if radial_coefficients is None:
        indices, basis = radial_basis(state, angles, action_scale)
        radial = (basis @ np.asarray(invariant)[indices])/action_scale
    else:
        # Evaluate the new ansatz directly in sqrt(J), without converting it
        # back into Cartesian monomials for the root polynomial.
        radial = np.zeros((len(angles), state['m']+1))
        ratio = action_scale/state['action_box']['Jx_max']
        for coefficient, label in zip(radial_coefficients, state['radial_labels']):
            delta_degree, i, j, n, kind_x, m, kind_y = label
            if delta_degree or j:
                continue
            angular = np.cos(n*angles) if kind_x == 'c' else np.sin(n*angles)
            radial[:, i] += coefficient*ratio**(i/2)*angular/action_scale
    if np.isclose(angles[-1]-angles[0], 2*np.pi, rtol=0, atol=1e-12):
        radial[-1] = radial[0]
    output = []
    for level in levels:
        polynomial = radial.copy(); polynomial[:, 0] -= level/action_scale
        rows = [positive_roots(c) for c in polynomial]
        output.append(dict(level=float(level), all_roots=[None if r is None else action_scale*r*r for r in rows]))
    return output


def solve(transfer, seed, state, tol=1e-14):
    """Compatibility alias: direct radial--Fourier LS, no root refinement."""
    from .action_angle_ls import solve as direct_solve
    return direct_solve(transfer, seed, state, tol)
