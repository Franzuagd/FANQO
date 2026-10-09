"""Algebraic regression tests for simultaneous Ix/Iy construction."""
import numpy as np

from fanqo.core.coupled_invariants import _bracket_indices, _whitener


def _tiny_state():
    # Full powers for x,y,px,py quadratic monomials (plus constant).
    powers = [(0,0,0,0,0), (0,1,0,0,0), (0,0,1,0,0),
              (0,0,0,1,0), (0,0,0,0,1), (0,2,0,0,0),
              (0,0,2,0,0), (0,0,0,2,0), (0,0,0,0,2)]
    return {"idx_to_vec": dict(enumerate(powers)),
            "vec_to_idx": {p:i for i,p in enumerate(powers)},
            "C": np.ones(len(powers))}


def test_bracket_canonical_coordinates():
    state = _tiny_state()
    k,i,j,v = _bracket_indices(state)
    def pb(left, right):
        return np.bincount(k, weights=v*left[i]*right[j],
                           minlength=len(state["C"]))
    basis = np.eye(len(state["C"]))
    assert np.allclose(pb(basis[1], basis[3]), basis[0])
    assert np.allclose(pb(basis[3], basis[1]), -basis[0])
    assert np.allclose(pb(basis[2], basis[4]), basis[0])
    assert np.allclose(pb(basis[1], basis[4]), 0)


def test_whitener():
    g = np.array([[2.0, 0.5],[0.5, 3.0]])
    w = _whitener(g)
    assert np.allclose(w.T @ w, g)


def _toy_problem(fixed):
    from fanqo.core import nonlinear as nl
    idx, lookup = nl.monomial_index_maps(3, 0, n=2)
    q = nl.quadratic_size(2)
    n = len(idx) - q
    sx = np.zeros(q)
    sy = np.zeros(q)
    for powers in ((0, 2, 0, 0, 0), (0, 0, 0, 2, 0)):
        sx[lookup[powers]] = 1.0
    for powers in ((0, 0, 2, 0, 0), (0, 0, 0, 0, 2)):
        sy[lookup[powers]] = 1.0
    state = {
        "idx_to_vec": idx, "vec_to_idx": lookup, "C": np.ones(q+n),
        "G": np.eye(q+n), "Gnn": np.eye(n),
        "quad_size": q, "nonquad_size": n,
        "method_options": {
            "COUPLED_FIX_QUADRATIC": fixed,
            "COUPLED_BRACKET_WEIGHT": 1.0,
            "COUPLED_QUADRATIC_WEIGHT": 2.0,
            "COUPLED_NONLINEAR_WEIGHT": 0.01,
            "COUPLED_MAX_ITER": 120,
        },
    }
    tnn = np.eye(n) * 0.4
    tnq = np.zeros((n, q))
    tnq[0, lookup[(0, 2, 0, 0, 0)]] = 0.12
    tnq[1, lookup[(0, 0, 2, 0, 0)]] = 0.06
    return state, sx, sy, tnn, tnq


def test_coupled_fixed_quadratic_and_analytic_gradient(monkeypatch):
    from fanqo.core import coupled_invariants as ci
    from fanqo.core import nonlinear as nl

    state, sx, sy, tnn, tnq = _toy_problem(True)
    monkeypatch.setattr(nl, "quadratic_invariants", lambda data, state: (sx, sy))
    scipy_minimize = ci.minimize

    def audited_minimize(fun, x0, **kwargs):
        # Check all analytic gradient components against centered differences.
        z = x0.copy()
        z[0] += 0.07
        z[len(z)//2] -= 0.04
        _, grad = fun(z)
        numeric = np.zeros_like(z)
        for p in range(len(z)):
            h = 1e-6
            right, _ = fun(z + h * np.eye(1, len(z), p)[0])
            left, _ = fun(z - h * np.eye(1, len(z), p)[0])
            numeric[p] = (right-left)/(2*h)
        assert np.allclose(grad, numeric, atol=1e-7, rtol=1e-4)
        return scipy_minimize(fun, x0, **kwargs)

    monkeypatch.setattr(ci, "minimize", audited_minimize)
    ix, details = ci.construct(None, tnn, tnq, None, state, 1e-14)
    assert np.array_equal(ix[:state["quad_size"]], sx)
    assert np.array_equal(details["Iy"][:state["quad_size"]], sy)
    assert details["quadratic_fixed"] is True
    assert details["relative_quadratic_x"] == 0
    assert details["relative_quadratic_y"] == 0
    assert np.isfinite(details["objective"])


def test_coupled_free_quadratic_and_analytic_gradient(monkeypatch):
    from fanqo.core import coupled_invariants as ci
    from fanqo.core import nonlinear as nl

    state, sx, sy, tnn, tnq = _toy_problem(False)
    monkeypatch.setattr(nl, "quadratic_invariants", lambda data, state: (sx, sy))
    original = ci.minimize

    def audited(fun, z0, **kwargs):
        z = z0.copy()
        z[0] += 0.07
        z[-2:] = [0.03, -0.04]
        _, grad = fun(z)
        for pos in (0, len(z)//2, len(z)-2, len(z)-1):
            eps = 1e-6
            dz = np.zeros_like(z)
            dz[pos] = eps
            fd = (fun(z+dz)[0] - fun(z-dz)[0])/(2*eps)
            assert np.isclose(grad[pos], fd, atol=1e-7, rtol=1e-4)
        return original(fun, z0, **kwargs)

    monkeypatch.setattr(ci, "minimize", audited)
    ix, details = ci.construct(None, tnn, tnq, None, state, 1e-14)
    assert details["quadratic_fixed"] is False
    assert np.isfinite(details["objective"])
    assert len(ix) == len(details["Iy"]) == len(state["idx_to_vec"])
