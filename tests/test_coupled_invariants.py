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
