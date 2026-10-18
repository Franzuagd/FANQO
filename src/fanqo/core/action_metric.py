"""Analytic polynomial metric on Courant--Snyder action disks.

FANQO uses Jx=Sx=gamma*x**2+2*alpha*x*px+beta*px**2 (no 1/2).
The normalized measure is uniform in Jx, Jy, both angles and delta.
The stored polynomial basis and Taylor projection are unchanged. All moments
are analytic; no tracking or Monte Carlo sampling enters construction.
"""
from math import comb, factorial, fsum

import numpy as np


def disk_moment(a, b, radius_squared):
    """Mean X**a P**b on a uniform disk X**2+P**2 <= radius_squared."""
    if a % 2 or b % 2:
        return 0.0
    i, j = a // 2, b // 2
    return (radius_squared ** (i+j) * factorial(2*i) * factorial(2*j)
            / (4 ** (i+j) * factorial(i) * factorial(j) * factorial(i+j+1)))


def ellipse_moment(a, b, beta, alpha, action_max):
    """Mean x**a px**b; x=sqrt(beta) X, px=(P-alpha X)/sqrt(beta)."""
    if (a+b) % 2:
        return 0.0
    return beta ** ((a-b)/2) * fsum(
        comb(b, t) * (-alpha) ** (b-t)
        * disk_moment(a+b-t, t, action_max) for t in range(b+1))


def action_gram(state, action_box, cs0=None):
    """Exact Gram matrix for the stored e_i=C_i*monomial_i basis."""
    cs = np.asarray(state['linear_cs0'] if cs0 is None else cs0, dtype=float)
    bx, ax, gx, by, ay, gy = cs
    if (not np.isfinite(cs).all() or bx <= 0 or by <= 0
            or not np.allclose([bx*gx-ax*ax, by*gy-ay*ay], 1., rtol=1e-7, atol=1e-9)):
        raise ValueError('Action metric needs positive, normalized uncoupled CS parameters.')
    try:
        bounds = {k: float(action_box[k]) for k in ('Jx_max', 'Jy_max', 'delta_max')}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('ACTION_BOX needs Jx_max, Jy_max and delta_max.') from exc
    if not all(np.isfinite(v) and v > 0 for v in bounds.values()):
        raise ValueError('All ACTION_BOX bounds must be positive and finite.')
    if state.get('horizontal_slice_only', False):
        raise ValueError('The action metric requires a full two-plane basis (positive A_BOX widths).')
    powers = np.array([state['idx_to_vec'][i] for i in range(len(state['C']))], dtype=int)
    sums = powers[:, None, :] + powers[None, :, :]
    max_power = int(sums.max())
    mx = np.zeros((max_power+1, max_power+1))
    my = mx.copy()
    for a in range(max_power+1):
        for b in range(max_power+1-a):
            mx[a, b] = ellipse_moment(a, b, bx, ax, bounds['Jx_max'])
            my[a, b] = ellipse_moment(a, b, by, ay, bounds['Jy_max'])
    dd = sums[:, :, 0]
    md = np.where(dd % 2 == 0, bounds['delta_max'] ** dd / (dd+1), 0.)
    raw = md * mx[sums[:,:,1], sums[:,:,3]] * my[sums[:,:,2], sums[:,:,4]]
    G = raw * state['C'][:, None] * state['C'][None, :]
    return (G + G.T) / 2, bounds


def attach_metric(state, action_box, cs0=None):
    """Refresh the domain when the lattice's Twiss parameters change."""
    G, bounds = action_gram(state, action_box, cs0)
    q = state['quad_size']
    # Validate positive definiteness once; reuse the residual weighting factor.
    W = np.linalg.cholesky(G).T
    state.update(G=G, Gqq=G[:q,:q], Gnn=G[q:,q:], action_box=bounds,
                 action_weight=W, metric='uniform_CS_action_disks',
                 action_convention='J=S=gamma*q^2+2*alpha*q*p+beta*p^2 (no 1/2)')
    if cs0 is not None:
        state['linear_cs0'] = np.asarray(cs0, dtype=float).copy()
    return state
