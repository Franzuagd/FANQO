"""Integration with main's existing load/candidate/diagnostic/optimizer flow."""
from pathlib import Path

import numpy as np
import pytest

from fanqo import api
from fanqo.core import nonlinear as nl, optimization as opt


ROOT = Path(__file__).resolve().parents[1]


def configuration(folder, geometry=True):
    text = (ROOT/'user/general_config.py').read_text()
    text += f"\nLATTICE_FILE={str(ROOT/'user/lattice_config.py')!r}\nORDER=3\nDELTA_ORDER=1\n"
    text += "\nRUN_FMA_START_END=False\nSAVE_PLOTS=False\nSHOW_PLOTS=False\nPLOT_START_END_SLICES=False\n"
    if geometry:
        text += "\nINVARIANT_CONSTRUCTION='action_angle_ls'\nOBJECTIVE='action_geometry'\n"
        text += "ACTION_GEOMETRY_POINTS=np.array([[0.,0.,0.,1e-7],[.002,1e-8,.7,2e-7]])\n"
        text += "ACTION_GEOMETRY_OPTIONS=dict(angles=32)\n"
    else:
        # Old configurations lack all these new settings entirely.
        text += "\nfor name in ('INVARIANT_CONSTRUCTION','OBJECTIVE','ACTION_BOX','ACTION_GEOMETRY_POINTS','ACTION_GEOMETRY_OPTIONS'):\n    globals().pop(name,None)\n"
    path = folder/'config.py'; path.write_text(text)
    return path


@pytest.fixture
def context(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Tracking must not enter invariant/objective construction.')
    monkeypatch.setattr(api, 'run_fma', forbidden)
    return api.load(configuration(tmp_path), force=True)


def test_old_load_and_a_box_objective_unchanged(tmp_path):
    ctx = api.load(configuration(tmp_path, geometry=False), force=True)
    st = ctx['state']; tol = 1e-14
    _, tnn, tnq = opt.nonlinear_transfer(ctx, tol)
    sx, sy = nl.quadratic_invariants(ctx['data'], st)
    legacy = nl.invariant(tnn, tnq, sx, sy, st, tol=tol)
    expected = nl.horizontal_shape_objective(legacy[-2], sx, st, gradient_weight=.1)
    details = opt.full_diagnostics(ctx, .1, tol)
    np.testing.assert_array_equal(details['Ix'], legacy[-2])
    np.testing.assert_array_equal(details['Iy'], legacy[-1])
    assert details['objective'] == expected[0]
    assert details['value_norm'] == expected[1]
    assert details['gradient_norm'] == expected[2]


def test_compute_invariants_and_candidate_use_same_horizontal_score(context):
    ix, iy = api.compute_invariants()
    details = api.STATE.invariant_details
    st = context['state']; sx, _ = nl.quadratic_invariants(context['data'], st)
    np.testing.assert_array_equal(ix[:st['quad_size']], sx)
    assert np.isfinite(iy).all()
    value = opt.evaluate_candidate(context, [], [], .1, 1e-14, 1e30)
    assert value == pytest.approx(details['objective'], abs=1e-12)
    assert 0 <= value <= 1
    # Full diagnostics solved Iy last, but its geometry must still refer to Ix.
    radial = st['radial_transform'] @ st['radial_last_coefficients']
    np.testing.assert_allclose(radial, ix, atol=1e-15, rtol=1e-10)
    assert details['quality'] == pytest.approx(1-value)


def test_failed_candidate_restores_basis_after_optics_refresh(context, monkeypatch):
    api.compute_invariants()
    st = context['state']
    before = {key: st[key].copy() for key in
              ('C', 'G', 'radial_transform', 'radial_gram', 'linear_cs0', 'radial_last_coefficients')}
    def change_then_fail(ctx, values, vary):
        data = list(ctx['data']); data[0] = np.array(data[0], copy=True)
        data[0][0] *= 1.1; data[0][2] = (1+data[0][1]**2)/data[0][0]
        opt._refresh_nonlinear_normalization(ctx['state'], data)
        raise ValueError('Synthetic rejected lattice after refreshing its basis.')
    monkeypatch.setattr(opt, 'apply_candidate', change_then_fail)
    assert opt.evaluate_candidate(context, [], [], .1, 1e-14, 1e30) == 1e30
    for key, value in before.items():
        np.testing.assert_array_equal(st[key], value)


def test_new_optics_basis_represents_new_courant_snyder(context):
    st = context['state']; data = list(context['data'])
    data[0] = np.array(data[0], copy=True)
    data[0][0] *= 1.1; data[0][2] = (1+data[0][1]**2)/data[0][0]
    old = st['radial_transform'].copy()
    opt._refresh_nonlinear_normalization(st, data)
    assert not np.array_equal(old, st['radial_transform'])
    sx, _ = nl.quadratic_invariants(data, st); q = st['quad_size']
    cj = np.linalg.solve(st['radial_transform'][:q, :q], sx)
    mode = st['radial_labels'].index((0, 2, 0, 0, 'c', 0, 'c'))
    expected = np.zeros(q); expected[mode] = st['action_box']['Jx_max']
    np.testing.assert_allclose(cj, expected, atol=1e-18)


def test_existing_optimizer_api_accepts_geometry_objective(context, tmp_path, monkeypatch):
    cfg = api.STATE.config
    for name, value in dict(VARY=['kse1', 'kfd2'], CMA_TIME=.1, CMA_SIGMA=.01,
            CMA_POPSIZE=4, POWELL_TIME_FRACTION=0., OUTPUT_ROOT=tmp_path,
            REPORT_FILE=tmp_path/'report.txt', FINAL_LATTICE_FILE=tmp_path/'lattice.json',
            PLOT_ROOT=tmp_path/'plots').items():
        monkeypatch.setattr(cfg, name, value)
    result = api.optimize(run_start_end_fma=False)
    assert 0 <= result['f_final'] <= result['start_details']['objective']+1e-10
    assert 'geometry' in result['final_details']
    assert 'action_geometry' in (tmp_path/'report.txt').read_text()
    assert (tmp_path/'lattice.json').is_file()


def test_load_rejects_geometry_with_a_box(tmp_path):
    path = configuration(tmp_path)
    path.write_text(path.read_text()+"\nINVARIANT_CONSTRUCTION='a_box'\n")
    with pytest.raises(ValueError, match='action_geometry needs'):
        api.load(path, force=True)
