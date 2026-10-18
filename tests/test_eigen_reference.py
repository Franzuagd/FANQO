"""Independent checks of eigen normalization, ranking and advisor objective."""
import numpy as np
import pytest
import sympy as sp

from fanqo.core import nonlinear as nl, eigen_reference as eigen
from fanqo.core.reference_objective import reference_fluctuation_index


@pytest.fixture(scope='module')
def state():
    v = sp.symbols('delta x y px py'); f = sp.symbols('b1 b2 b3 b4 b5')
    _, x, y, px, py = v
    cs = np.array([2., .3, 1.09/2, 3., -.2, 1.04/3])
    st = nl.initialize_nonlinear([cs], 3, 1, (x*x+px*px+y*y+py*py)/2,
                                 np.ones(5), v, f)
    st['D_y'] = nl.build_derivative_matrix(st, 2)
    st['D_py'] = nl.build_derivative_matrix(st, 4)
    return st


def seed(st, plane='x'):
    sx, sy = nl.quadratic_invariants([st['linear_cs0']], st)
    c = np.zeros(len(st['C'])); c[:len(sx)] = sx if plane == 'x' else sy
    return c*st['C']


@pytest.mark.parametrize('plane', ['x', 'y'])
def test_normalize_then_compare_all_quadratic_terms(state, plane):
    target = seed(state, plane); contaminated = target.copy()
    other = (0, 0, 2, 0, 0) if plane == 'x' else (0, 2, 0, 0, 0)
    contaminated[state['vec_to_idx'][other]] += 10.
    vectors = np.column_stack((3*contaminated, -7*target))
    result, info = eigen.solve(np.eye(len(target)), state, plane=plane,
        eigensystem=(np.zeros(2), vectors))
    np.testing.assert_allclose(result*state['C'], target, atol=1e-14)
    assert info['quadratic_relative_distance'] < 1e-14
    assert info['reference_candidate_count'] == 2


def test_physical_similarity_preserves_eigenvector_and_cs_normalization(state):
    target = seed(state); target[state['vec_to_idx'][(0, 3, 0, 0, 0)]] = .4
    physical = 2*np.eye(len(target))-np.outer(target, target)/(target@target)
    C = state['C']; stored = physical*C[None, :]/C[:, None]
    result, info = eigen.solve(stored, state)
    np.testing.assert_allclose(result*C, target, atol=2e-13)
    np.testing.assert_allclose(stored@result, result, atol=2e-13)
    assert info['eigenvalue_residual'] < 1e-12


def test_rejects_vectors_without_positive_plane_determinant(state):
    vector = np.zeros((len(state['C']), 1)); vector[0, 0] = 1.
    with pytest.raises(ValueError, match='no normalizable'):
        eigen.solve(np.eye(len(vector)), state, eigensystem=(np.zeros(1), vector))


def test_advisor_objective_matches_analytic_polynomial_derivatives(state):
    # N=3x^3+2delta*y^2+4x^2*px+5y^2*py.
    physical = seed(state)
    for powers, value in [((0,3,0,0,0),3.), ((1,0,2,0,0),2.),
                          ((0,2,0,1,0),4.), ((0,0,2,0,1),5.)]:
        physical[state['vec_to_idx'][powers]] += value
    options = dict(x_range=.4, x_points=9, y_range=.2, y_points=5,
                   delta_values=(-.3,.1), momentum_weight=.7)
    value, details = reference_fluctuation_index({'Ix':physical/state['C'], 'state':state}, **options)
    xs = np.unique(np.r_[np.linspace(-.4,0,4),np.linspace(0,.4,4)])
    X,Y = np.meshgrid(xs,np.linspace(0,.2,5),indexing='xy')
    expected = []
    for delta in options['delta_values']:
        g = (81+.7**2*16)*X**4+16*delta**2*Y**2+.7**2*25*Y**4
        mean = g.mean(); std = g.std(); skew = np.mean((g-mean)**3)/std**3
        expected.append(np.sqrt(mean**2+std**2+skew**2))
    np.testing.assert_allclose([s['score'] for s in details['slice_scores']], expected)
    assert value == pytest.approx(max(expected))


def test_cs_has_zero_reference_fluctuation(state):
    value, _ = reference_fluctuation_index({'Ix':seed(state)/state['C'], 'state':state})
    assert value == 0.


def test_reference_dispatch_uses_eigen_for_both_diagnostics_and_candidates(tmp_path, monkeypatch):
    from pathlib import Path
    from fanqo import api
    from fanqo.core import optimization as opt
    root = Path(__file__).resolve().parents[1]
    text = (root/'user/general_config.py').read_text()
    text += f"\nLATTICE_FILE={str(root/'user/lattice_config.py')!r}\nORDER=3\nDELTA_ORDER=1\n"
    text += "INVARIANT_CONSTRUCTION='eigen'\nOBJECTIVE='reference_fluctuation_index'\n"
    path = tmp_path/'config.py'; path.write_text(text)
    context = api.load(path, force=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('Reference dispatch must not run the LS constructor or tracking.')
    monkeypatch.setattr(nl, 'invariant', forbidden)
    monkeypatch.setattr(api, 'run_fma', forbidden)
    api.compute_invariants()
    details = api.STATE.invariant_details
    candidate = opt.evaluate_candidate(context, [], [], .1, 1e-14, 1e30)
    assert np.isfinite(candidate) and candidate < 1e30
    assert candidate == pytest.approx(details['objective'], rel=1e-10)
    for key in ('Ix', 'Iy'):
        assert details[key+'_construction_details']['method'] == 'eigen'
        assert details[key+'_construction_details']['eigenvalue_residual'] < 1e-8
