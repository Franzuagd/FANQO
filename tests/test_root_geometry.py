"""Independent action-angle, branch-selection and constrained-fit checks."""
import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from fanqo import api
from fanqo.core import nonlinear as nl, optimization as opt, root_geometry as rg


@pytest.fixture(scope='module')
def problem():
    v = sp.symbols('delta x y px py'); f = sp.symbols('b1 b2 b3 b4 b5')
    _, x, y, px, py = v
    cs = np.array([2., .4, (1+.4**2)/2, 3., -.3, (1+.3**2)/3])
    state = nl.initialize_nonlinear_for_method([cs], 4, 0, (x*x+y*y+px*px+py*py)/2,
        np.ones(5), v, f, invariant_construction='root_geometry',
        method_options={'ACTION_BOX':dict(Jx_max=1., Jy_max=.7, delta_max=.2),
                        'ROOT_ANGLES':24, 'ROOT_MAX_ITER':30})
    return [cs], state


def test_action_angle_roundtrip_and_quadratic(problem):
    _, st = problem
    cs = st['linear_cs0']; J = np.linspace(.1, 2., 30); phi = np.linspace(.01, 6.2, 30)
    q, p = rg.from_action_angle(J, phi, cs[0], cs[1])
    measured, angles = rg.to_action_angle(q, p, cs[0], cs[1])
    np.testing.assert_allclose(measured, J, rtol=1e-14)
    np.testing.assert_allclose(angles, phi, atol=1e-14)
    np.testing.assert_allclose(cs[2]*q*q+2*cs[1]*q*p+cs[0]*p*p, J)
    assert np.isnan(rg.to_action_angle(0., 0., cs[0], cs[1])[1])


def test_radial_evaluation_matches_independent_polynomial(problem):
    _, st = problem
    rng = np.random.default_rng(2026); c = rng.normal(size=len(st['C']))
    phi = np.linspace(0, 2*np.pi, 43); scale = .8
    indices, basis = rg.radial_basis(st, phi, scale)
    radial = basis @ c[indices]
    J = np.linspace(.01, 1., len(phi)); u = np.sqrt(J/scale)
    actual = np.sum(radial*u[:, None]**np.arange(radial.shape[1]), axis=1)
    q, p = rg.from_action_angle(J, phi, *st['linear_cs0'][:2])
    fn = api._make_ix_callable(c, st)
    np.testing.assert_allclose(actual, fn(0., q, 0., p, 0.), rtol=2e-12, atol=1e-12)


def test_known_ellipse_and_moment_identity():
    phi = np.linspace(0, 2*np.pi, 513); eps = .3; J0 = .2
    radial = np.zeros((len(phi), 3)); radial[:, 2] = 1-eps*np.cos(2*phi)
    curve = rg.trace_radial_branch(radial, J0, 1.)
    C = J0*(1-eps)
    assert curve['complete']
    np.testing.assert_allclose(curve['action'], C/(1-eps*np.cos(2*phi)), rtol=1e-13)
    stats = curve['statistics']
    assert stats['rms_deviation']**2 == pytest.approx(stats['bias']**2+stats['std']**2)


def test_closed_inner_branch_not_contaminated_by_outer_roots():
    phi = np.linspace(0, 2*np.pi, 513); eps = .1; J0 = .2
    radial = np.zeros((len(phi), 5)); radial[:, 2] = 1.; radial[:, 4] = eps*np.cos(4*phi)
    curve = rg.trace_radial_branch(radial, J0, 1.)
    assert curve['complete']
    C = J0+eps*J0**2
    expected = 2*C/(1+np.sqrt(1+4*eps*C*np.cos(4*phi)))
    np.testing.assert_allclose(curve['action'], expected, rtol=1e-12)
    assert any(len(r) == 2 for r in curve['all_roots'])
    assert max(max(r) for r in curve['all_roots']) > 10
    assert curve['statistics']['max_action'] < 1


def test_missing_roots_are_not_filled_or_averaged():
    phi = np.linspace(0, 2*np.pi, 65)
    radial = np.zeros((len(phi), 3)); radial[:, 2] = 1+2*np.cos(phi)
    curve = rg.trace_radial_branch(radial, .2, 1.)
    assert not curve['complete']
    assert np.isnan(curve['action']).any()
    assert curve['statistics'] == {}
    assert rg.positive_roots(np.zeros(4)) is None


def test_constrained_refinement_preserves_seed_and_budget(problem):
    data, st = problem; st = dict(st)
    sx, _ = nl.quadratic_invariants(data, st)
    n = len(st['C']); q = st['quad_size']
    # One algebraic condition leaves a nonunique family of quartic tails.
    # Refinement may improve geometry but cannot escape this residual budget.
    T = np.eye(n)
    free = [st['vec_to_idx'][p] for p in ((0,4,0,0,0), (0,2,0,2,0), (0,0,0,4,0))]
    T[q, free] = [1., .5, .3]
    T[q, st['vec_to_idx'][(0,2,0,0,0)]] += .01
    c, details = rg.solve(T, sx, st)
    np.testing.assert_array_equal(c[:q], sx)
    assert details['residual_final'] <= details['residual_limit']*(1+1e-7)
    assert details['accepted']
    assert details['objective_final'] < details['objective_initial']
    assert details['validation_objective_final'] < details['validation_objective_initial']
    assert np.isfinite(c).all()


def test_public_dispatch_and_no_false_improvement_on_identity(problem, monkeypatch):
    data, st = problem
    n = len(st['C']); q = st['quad_size']; T = np.eye(n)
    monkeypatch.setattr(opt, 'nonlinear_transfer', lambda *args: (T, T[q:, q:], T[q:, :q]))
    result = opt.compute_requested_invariants({'state':st, 'data':data}, 1e-12)
    sx, _ = nl.quadratic_invariants(data, st)
    np.testing.assert_array_equal(result['Ix'][:q], sx)
    assert not result['Ix_construction_details']['accepted']
    with pytest.raises(ValueError, match='Ix only'):
        opt.compute_requested_invariants({'state':st, 'data':data}, 1e-12, compute_iy=True)


def test_geometry_runner_never_calls_tracking(tmp_path, monkeypatch):
    from fanqo.root_comparison import run_root_comparison
    def forbidden(*args, **kwargs):
        raise AssertionError('Tracking called during geometry-only construction')
    monkeypatch.setattr(api, '_physical_tracking_ring', forbidden)
    monkeypatch.setattr(api, '_track_initial_conditions', forbidden)
    config = Path(__file__).resolve().parents[1]/'user'/'general_config.py'
    report = run_root_comparison(config, output=tmp_path, order=3, delta_order=0,
        angles=16, geometry_only=True, root_options={'ROOT_ANGLES':8, 'ROOT_MAX_ITER':2})
    assert report['tracking_used_in_construction'] is False
    assert not list(tmp_path.glob('tracking_*.npz'))
    assert (tmp_path/'algebraic_action_angle.png').is_file()
    assert (tmp_path/'root_geometry'/'Ix_coefficients.npz').is_file()
    assert json.loads((tmp_path/'comparison.json').read_text())['delta'] == 0.
