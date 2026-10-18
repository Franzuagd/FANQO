"""Independent action-angle, all-root and linear-LS checks."""
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
        method_options={'ACTION_BOX':dict(Jx_max=1., Jy_max=.7, delta_max=.2)})
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


def test_public_dispatch_is_linear_ls(problem, monkeypatch):
    data, st = problem
    n=len(st['C']);q=st['quad_size'];T=np.eye(n)
    monkeypatch.setattr(opt,'nonlinear_transfer',lambda *args:(T,T[q:,q:],T[q:,:q]))
    result=opt.compute_requested_invariants({'state':st,'data':data},1e-12)
    sx,_=nl.quadratic_invariants(data,st)
    np.testing.assert_array_equal(result['Ix'][:q],sx)
    assert result['Ix_construction_details']['method']=='action_angle_ls'
    assert result['Ix_construction_details']['solver']=='linear SVD least squares'


def test_all_root_plot_data_retains_both_branches(problem):
    _,st=problem
    c=np.zeros(len(st['C']));v=st['vec_to_idx'];cs=st['linear_cs0']
    # P=J-epsilon*J^2 has two roots for this positive level at every angle.
    J={(0,2,0,0,0):cs[2],(0,1,0,1,0):2*cs[1],(0,0,0,2,0):cs[0]}
    for power,value in J.items():c[v[power]]+=value/st['C'][v[power]]
    for p1,a in J.items():
        for p2,b in J.items():
            power=tuple(x+y for x,y in zip(p1,p2))
            c[v[power]]-=.1*a*b/st['C'][v[power]]
    phi=np.linspace(0,2*np.pi,49)
    data=rg.all_level_roots(c,st,[.2,.4],phi,1.)
    for level,rootset in zip([.2,.4],data):
        expected=np.sort(np.roots([.1,-1,level]))
        assert all(len(row)==2 for row in rootset['all_roots'])
        for row in rootset['all_roots']:np.testing.assert_allclose(row,expected,rtol=1e-12)
        np.testing.assert_array_equal(rootset['all_roots'][0],rootset['all_roots'][-1])


@pytest.mark.parametrize('legacy_options',[None,{'ROOT_MAX_ITER':999,'ROOT_RESIDUAL_FACTOR':42.}])
def test_geometry_runner_never_calls_tracking(tmp_path, monkeypatch, legacy_options):
    from fanqo.root_comparison import run_root_comparison
    def forbidden(*args, **kwargs):
        raise AssertionError('Tracking called during geometry-only construction')
    monkeypatch.setattr(api, '_physical_tracking_ring', forbidden)
    monkeypatch.setattr(api, '_track_initial_conditions', forbidden)
    config = Path(__file__).resolve().parents[1]/'user'/'general_config.py'
    report = run_root_comparison(config, output=tmp_path, order=3, delta_order=0,
        angles=16, geometry_only=True, root_options=legacy_options)
    assert report['tracking_used_in_construction'] is False
    assert report['methods']['action_angle_ls']['solver']=='linear SVD least squares'
    assert report['ignored_legacy_root_options']==(list(legacy_options) if legacy_options is not None else [])
    assert not list(tmp_path.glob('tracking_*.npz'))
    assert (tmp_path/'algebraic_action_angle.png').is_file()
    assert (tmp_path/'action_angle_ls'/'Ix_coefficients.npz').is_file()
    assert json.loads((tmp_path/'comparison.json').read_text())['delta'] == 0.
    # The baseline left in the context must remain the original a_box state.
    assert api.STATE.context['state']['invariant_construction']=='a_box'
    expected=opt.compute_requested_invariants(api.STATE.context,1e-14)['Ix']
    saved=np.load(tmp_path/'a_box'/'Ix_coefficients.npz')['Ix']
    np.testing.assert_allclose(saved,expected,rtol=1e-12,atol=1e-12)


def test_restored_legacy_tracking_metric():
    from fanqo.root_comparison import _legacy_rows
    a=np.zeros((6,4));a[0]=[2.,2.2,1.8,2.6]
    b=np.zeros((6,3));b[0]=[0.,.1,.4]
    meta=[dict(completed_turns=3,survived=True),dict(completed_turns=2,survived=False)]
    rows=_legacy_rows(lambda delta,x,y,px,py:x,[a,b],meta,[(1.,0.),(2.,0.)],.1)
    # First uses |Ix0|=2; second uses floor=.1*max_grid(|Ix0|)=.2.
    assert rows[0,5]==pytest.approx(.3)
    assert rows[0,7]==pytest.approx(.1)
    assert rows[1,5]==pytest.approx(2.)
    assert rows[1,7]==pytest.approx(1.)
    assert rows[1,10]==0
