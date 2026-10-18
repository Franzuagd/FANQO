"""Discrete geometric quality of radial--Fourier quasi-invariant levels.

No tracking, coefficient fitting or magnet optimization occurs here. Actions
use FANQO's J=S convention (twice canonical action). See docs/action_geometry.md.
"""
import numpy as np

from .action_roots import trace_radial_branch


def action_geometry_objective(coefficients, state, points, *, weights=None,
                              angles=64, amplitude_scale=.5, slope_scale=1.,
                              derivative_weight=.1):
    """Return loss=1-weighted_mean(Q) for rows (delta,Jy,phi_y,Jx0).

    ``coefficients`` are explicit radial--Fourier coefficients from
    action_angle_ls.solve, NOT the returned Cartesian storage vector. Pass
    state['radial_last_coefficients'] after constructing Ix. The launch angle
    is phi_x=0; Jy,phi_y,delta are held fixed on each slice.

    Q=exp[-(max|J/J0-1|/amplitude_scale)^2
          -derivative_weight*mean(d^2/(slope_scale^2+d^2))], d=d(J/J0)/dphi.

    Selects a regular radial graph through the launch, not disconnected roots.
    Folded/missing/unresolved graphs get Q=0 with an explicit status; this does
    NOT certify an unbounded component or physical instability. Finite angular
    sampling cannot certify completeness, connectedness or absence of poles.
    No action cutoff is used. No inputs/state are mutated.
    """
    p = np.asarray(points, dtype=float)
    c = np.asarray(coefficients, dtype=float)
    if p.ndim != 2 or p.shape[1] != 4 or not len(p) or not np.isfinite(p).all():
        raise ValueError('points must be finite nonempty rows (delta,Jy,phi_y,Jx0).')
    if np.any(p[:, 1] < 0) or np.any(p[:, 3] <= 0):
        raise ValueError('Require Jy>=0 and Jx0>0.')
    labels = state['radial_labels']; box = state['action_box']
    if c.shape != (len(labels),) or not np.isfinite(c).all():
        raise ValueError('Supply one finite radial--Fourier coefficient per label.')
    if not isinstance(angles, (int, np.integer)) or angles < 8:
        raise ValueError('angles must be an integer >=8.')
    if not (np.isfinite(amplitude_scale) and amplitude_scale > 0
            and np.isfinite(slope_scale) and slope_scale > 0
            and np.isfinite(derivative_weight) and derivative_weight >= 0):
        raise ValueError('Require positive scales and nonnegative derivative_weight.')
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(p),) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('weights must be finite positive values, one per point.')
    w = w/np.max(w); w = w/np.sum(w)
    phi = np.linspace(0., 2*np.pi, angles+1)
    # P/J0 in r=sqrt(J/J0), with launch r=1. Analytic phi derivative.
    radial = np.zeros((len(p), angles+1, state['m']+1))
    angular_derivative = np.zeros_like(radial)
    for coefficient, (dd, i, j, n, kx, m, ky) in zip(c, labels):
        if coefficient == 0:
            continue
        fy = np.cos(m*p[:, 2]) if ky == 'c' else np.sin(m*p[:, 2])
        factor = (coefficient/p[:, 3]*(p[:, 0]/box['delta_max'])**dd
                  *(p[:, 3]/box['Jx_max'])**(i/2)
                  *(p[:, 1]/box['Jy_max'])**(j/2)*fy)
        fx = np.cos(n*phi) if kx == 'c' else np.sin(n*phi)
        dfx = -n*np.sin(n*phi) if kx == 'c' else n*np.cos(n*phi)
        radial[:, :, i] += factor[:, None]*fx
        angular_derivative[:, :, i] += factor[:, None]*dfx
    radial[:, -1] = radial[:, 0]
    angular_derivative[:, -1] = angular_derivative[:, 0]
    scores = np.zeros(len(p)); excursions = np.full(len(p), np.nan)
    slope_penalties = np.full(len(p), np.nan)
    actions = np.full((len(p), angles), np.nan)
    derivatives = np.full_like(actions, np.nan)
    levels = np.empty(len(p)); statuses = []
    for q in range(len(p)):
        curve = trace_radial_branch(radial[q], 1., 1., slope_floor=1e-12)
        levels[q] = curve['level']*p[q, 3]
        statuses.append(curve['status'])
        actions[q] = curve['action'][:-1]*p[q, 3]
        if not curve['complete']:
            continue
        u = curve['action'][:-1]; r = np.sqrt(u)
        numerator = np.polynomial.polynomial.polyval(
            r, angular_derivative[q, :-1].T, tensor=False)
        # slopes is d(P/J0)/du, so d(u)/dphi=-P_phi/P_u.
        d = -numerator/curve['slopes'][:-1]
        derivatives[q] = d
        excursions[q] = np.max(np.abs(u-1))
        # hypot avoids overflow when |d| is very large near a vertical tangent.
        slope_penalties[q] = np.mean((d/np.hypot(slope_scale, d))**2)
        with np.errstate(over='ignore'):
            scores[q] = np.exp(-(excursions[q]/amplitude_scale)**2
                               -derivative_weight*slope_penalties[q])
    quality = float(np.clip(w @ scores, 0., 1.))
    return dict(loss=1.-quality, quality=quality, scores=scores,
                points=p.copy(), weights=w, angles=phi[:-1], actions=actions,
                normalized_derivatives=derivatives, levels=levels,
                max_relative_excursion=excursions, derivative_penalty=slope_penalties,
                status=statuses, resolved=np.array([s == 'complete' for s in statuses]),
                settings=dict(angles=angles, amplitude_scale=amplitude_scale,
                              slope_scale=slope_scale, derivative_weight=derivative_weight))
