import numpy as np
import sympy as sp
import pytest
from types import SimpleNamespace
from scipy.linalg import expm
from fanqo.core import nonlinear as nl, structured_invariants as si


def state_for(d=1):
    variables = sp.symbols('delta x y px py')
    delta, x, y, px, py = variables
    fields = sp.symbols('b1 b2 b3 b4 b5')
    H = .155*(x*x+px*px)+.235*(y*y+py*py)-.03*delta*x+.02*x**3
    data = [np.array([1., 0., 1., 1., 0., 1.])]
    st = nl.initialize_nonlinear_for_method(data, 4, d, H, np.ones(5), variables, fields,
        invariant_construction='graded_coupled', method_options={'STRUCTURED_RIDGE': 0.})
    return data, st


def test_delta_is_outer_layer_not_part_of_transverse_degree():
    _, st = state_for(d=3)
    grades = [g for g, _ in si.grade_groups(st)]
    assert grades == sorted(grades)
    j = st['vec_to_idx'][(3, 1, 0, 1, 0)]  # delta**3*x*px has transverse degree 2
    assert any(g == (3, 2) and j in rows for g, rows in si.grade_groups(st))
    assert grades.index((0, 4)) < grades.index((1, 1))


def test_delta_degree_lowering_uses_layered_solver():
    data, st = state_for()
    M = nl.assemble_M(np.asarray(st['H_vec_func'](0, 0, 0, 0, 0), float).ravel(), st['M_basis'])
    T = expm(-M)
    high = st['vec_to_idx'][(0, 2, 0, 1, 0)]
    low = st['vec_to_idx'][(1, 2, 0, 0, 0)]
    assert abs(T[low, high]) > 1e-5
    groups, leak = si._check_structure(T, st)
    assert leak < 1e-12
    ix, iy, details = si.construct_graded(T, data, st)
    assert details['fitting_strategy'] == 'delta_layers_then_transverse_degree'
    assert [tuple(b['grade']) for b in details['blocks']] == sorted(tuple(b['grade']) for b in details['blocks'])
    assert np.isfinite(ix).all() and np.isfinite(iy).all()


def test_eigen_initial_correction_retains_null_component_and_CS_seed():
    data, st = state_for()
    sx, _ = nl.quadratic_invariants(data, st)
    N = len(st['C']); q = st['quad_size']
    # One solvable residual coefficient plus one exactly unobservable gauge.
    T = np.eye(N)
    active, gauge = q+1, q
    T[active, active] = .5
    T[active, st['vec_to_idx'][(0, 2, 0, 0, 0)]] = .1
    physical_eigen = np.zeros(N)
    physical_eigen[gauge] = 7.*st['C'][gauge]
    physical_eigen[active] = 3.*st['C'][active]
    result, details = nl.eigen_initialized_a_box(T, st, sx, eigen_initial=physical_eigen)
    assert np.array_equal(result[:q], sx)
    assert np.isclose(result[gauge], 7.)
    assert details['weighted_residual_after'] < 1e-12
    assert details['weighted_residual_after'] <= details['weighted_residual_before']


def test_a_box_bracket_diagnostic_uses_actual_coefficient_scale():
    data, st = state_for()
    # The legacy tensor still uses epsilon; emulate the extra common CS scale.
    st = dict(st, C=2.*st['C'])
    sx, sy = nl.quadratic_invariants(data, st)
    q, N = st['quad_size'], len(st['C'])
    tnn = np.zeros((N-q, N-q))
    tnq = np.zeros((N-q, q))
    # Force a cross-plane tail with a nonzero bracket against Iy.
    j = st['vec_to_idx'][(0, 1, 1, 0, 1)]
    source = np.flatnonzero(sx)[0]
    tnq[j-q, source] = 1.
    result = nl.invariant(tnn, tnq, sx, sy, st)
    physical_bracket = si.bracket(result[-2], result[-1], st)
    assert np.isclose(result[8], si._norm(physical_bracket, st['G']))


def test_eigen_initial_method_dispatch(monkeypatch):
    data, original = state_for()
    delta, x, y, px, py = original['variables']
    st = nl.initialize_nonlinear_for_method(data, 4, 1,
        .155*(x*x+px*px)+.235*(y*y+py*py),
        np.ones(5), original['variables'], sp.symbols('b1 b2 b3 b4 b5'),
        invariant_construction='a_box_eigen_initial')
    q, N = st['quad_size'], len(st['C'])
    T = np.eye(N); T[q:, q:] *= .5
    monkeypatch.setattr(nl, 'nonlinear_transfer', lambda *a, **k: (T, T[q:, q:], T[q:, :q]))
    monkeypatch.setattr(nl, 'eigen_invariant', lambda transfer, state, plane='x': (np.ones(N), {}))
    ix, iy, details, _ = nl.invariant_vectors([], data, st)
    sx, sy = nl.quadratic_invariants(data, st)
    assert np.array_equal(ix[:q], sx) and np.array_equal(iy[:q], sy)
    assert details['method'] == 'a_box_eigen_initial'
    assert np.allclose(ix[q:], 0., atol=1e-12)


@pytest.mark.parametrize('fixed', [False, True])
def test_coupled_objective_gradient_matches_directional_differences(monkeypatch, fixed):
    from fanqo.core import coupled_invariants as ci
    data, st = state_for()
    st['method_options'] = {'COUPLED_FIX_QUADRATIC': fixed}
    M = nl.assemble_M(np.asarray(st['H_vec_func'](0, 0, 0, 0, 0), float).ravel(), st['M_basis'])
    T = expm(-M); q = st['quad_size']
    rng = np.random.default_rng(571)
    def check(fun, z0, **kwargs):
        z = z0 + .003*rng.normal(size=len(z0))
        value, grad = fun(z)
        for _ in range(3):
            direction = rng.normal(size=len(z)); direction /= np.linalg.norm(direction)
            h = 1e-6
            numerical = (fun(z+h*direction)[0]-fun(z-h*direction)[0])/(2*h)
            assert np.isclose(numerical, grad @ direction, rtol=1e-5, atol=1e-8)
        return SimpleNamespace(x=z, fun=value, jac=grad, status=0, success=True,
                               message='gradient verified', nfev=7, nit=0)
    monkeypatch.setattr(ci, 'minimize', check)
    ci.construct(T, T[q:, q:], T[q:, :q], data, st, 1e-14)
