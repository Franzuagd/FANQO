"""Direct linear least squares in smooth radial--Fourier polynomials.

Basis: (delta/dmax)^d (Jx/Jxmax)^(i/2) (Jy/Jymax)^(j/2)
       F_n(phi_x) F_m(phi_y), with i-n and j-m even/nonnegative.
No root statistics, nonlinear optimizer, tracking or fitted trajectories enter
construction. D maps radial--Fourier coefficients into FANQO stored ones.
"""
from math import comb

import numpy as np
from .action_metric import attach_metric


def _multiply(a, b):
    result = {}
    for (i, j), x in a.items():
        for (k, l), y in b.items():
            key = (i+k, j+l)
            result[key] = result.get(key, 0)+x*y
    return result


def plane_modes(degree, beta, alpha, action_max):
    """Homogeneous radial Fourier functions as physical (q,p) polynomials."""
    gamma = (1+alpha*alpha)/beta
    action = {(2, 0):gamma, (1, 1):2*alpha, (0, 2):beta}
    modes = []
    for n in range(degree % 2, degree+1, 2):
        # Q-iP=sqrt(J)*exp(i phi), where P=alpha*Q+sqrt(beta)*p.
        z = {(n-k, k):comb(n, k)*((1-1j*alpha)/np.sqrt(beta))**(n-k)
             *(-1j*np.sqrt(beta))**k for k in range(n+1)}
        for _ in range((degree-n)//2):
            z = _multiply(z, action)
        for kind in ('c',) if n == 0 else ('c', 's'):
            poly = {p:float((v.real if kind == 'c' else v.imag)/action_max**(degree/2))
                    for p, v in z.items()}
            modes.append((n, kind, {p:v for p, v in poly.items() if v != 0}))
    return modes


def prepare_basis(state):
    """Build the invertible basis change D and analytic action/Fourier Gram."""
    box = state['action_box']; cs = state['linear_cs0']; size = len(state['C'])
    groups = {}
    for row in range(size):
        p = state['idx_to_vec'][row]
        groups.setdefault((p[0], sum(p[1:])), []).append(row)
    mx = {i:plane_modes(i, cs[0], cs[1], box['Jx_max']) for i in range(state['m']+1)}
    my = {i:plane_modes(i, cs[3], cs[4], box['Jy_max']) for i in range(state['m']+1)}
    D = np.zeros((size, size)); labels = [None]*size
    for (delta_degree, degree), rows in groups.items():
        columns = iter(rows)
        for i in range(degree+1):
            j = degree-i
            for n, kind_x, px in mx[i]:
                for m, kind_y, py in my[j]:
                    column = next(columns)
                    labels[column] = (delta_degree, i, j, n, kind_x, m, kind_y)
                    for (a, b), vx in px.items():
                        for (c, d), vy in py.items():
                            row = state['vec_to_idx'][(delta_degree, a, c, b, d)]
                            D[row, column] += vx*vy/box['delta_max']**delta_degree/state['C'][row]
    G = np.zeros((size, size))
    for a, x in enumerate(labels):
        da, i, j, n, kx, m, ky = x
        for b in range(a+1):
            db, ib, jb, nb, kxb, mb, kyb = labels[b]
            if (n, kx, m, ky) != (nb, kxb, mb, kyb) or (da+db) % 2:
                continue
            angular = (1. if n == 0 else .5)*(1. if m == 0 else .5)
            G[a, b] = G[b, a] = angular/((da+db+1)*((i+ib)/2+1)*((j+jb)/2+1))
    state.update(radial_transform=D, radial_labels=labels, radial_gram=G,
                 radial_weight=np.linalg.cholesky(G).T,
                 radial_groups=list(groups.values()))
    return state


def attach(state, action_box, cs0=None):
    """Refresh both the physical metric and the optics-dependent Fourier basis."""
    attach_metric(state, action_box, cs0)
    return prepare_basis(state)


def transform_transport(transfer, state):
    """T_J=D^{-1} T D, solved degree block by degree block for conditioning."""
    D = state['radial_transform']; rhs = np.asarray(transfer) @ D
    result = np.empty_like(rhs)
    for rows in state['radial_groups']:
        result[rows] = np.linalg.solve(D[np.ix_(rows, rows)], rhs[rows])
    return result


def solve(transfer, seed, state, tol=1e-14):
    """Fix P2=Jx (or Jy); solve all remaining radial coefficients by SVD LS."""
    if 'radial_transform' not in state:
        prepare_basis(state)
    D = state['radial_transform']; q = state['quad_size']
    Tj = transform_transport(transfer, state)
    residual_operator = Tj-np.eye(len(Tj))
    seed_j = np.linalg.solve(D[:q, :q], seed)
    W = state['radial_weight']
    A = W @ residual_operator[:, q:]
    b = -W @ (residual_operator[:, :q] @ seed_j)
    tail, _, rank, singular = np.linalg.lstsq(A, b, rcond=tol)
    coeff_j = np.r_[seed_j, tail]
    invariant = D @ coeff_j
    invariant[:q] = seed  # enforce the prescribed physical block exactly
    residual = W @ (residual_operator @ coeff_j)
    seed_norm = np.linalg.norm(W @ np.r_[seed_j, np.zeros(len(tail))])
    state['radial_last_coefficients'] = coeff_j
    return invariant, dict(method='action_angle_ls', solver='linear SVD least squares',
        objective='projected invariance residual in radial-Fourier/action L2',
        quadratic_seed='fixed Courant-Snyder at delta=0', action_box=dict(state['action_box']),
        action_convention=state['action_convention'], rank=int(rank), unknowns=len(tail),
        rcond=float(tol), residual_norm=float(np.linalg.norm(residual)),
        relative_residual=float(np.linalg.norm(residual)/max(seed_norm, np.finfo(float).tiny)),
        largest_singular_value=float(singular[0]) if len(singular) else 0.,
        smallest_retained_singular_value=float(singular[rank-1]) if rank else None,
        transport_similarity_error=float(np.linalg.norm(D @ Tj-np.asarray(transfer) @ D)
                                         /max(np.linalg.norm(np.asarray(transfer) @ D), 1e-300)))
