"""Independent analytic level sets for the tracking-free objective."""
import numpy as np
import pytest

from fanqo.core.action_geometry_objective import action_geometry_objective


def state(labels, degree=4):
    return dict(m=degree, radial_labels=labels,
                action_box=dict(Jx_max=1., Jy_max=1., delta_max=1.))


J = (0, 2, 0, 0, 'c', 0, 'c')


def test_courant_snyder_is_one_on_all_slices():
    p = [[-.1, .2, .7, .3], [0., 0., 0., .8], [.1, .4, 2., .1]]
    result = action_geometry_objective([1.], state([J]), p)
    np.testing.assert_allclose(result['scores'], 1., atol=1e-14)
    assert result['loss'] == pytest.approx(0., abs=1e-14)
    assert result['resolved'].all()


def test_coupled_chromatic_slice_matches_analytic_roots_and_derivative():
    # P=J*(1-epsilon*delta*Jy*cos(2phi_x)*cos(2phi_y)).
    labels = [J, (1, 2, 2, 2, 'c', 2, 'c')]
    p = np.array([[.3, .4, .2, .2], [-.2, .5, .4, .3]])
    result = action_geometry_objective([1., -.7], state(labels), p,
                                       weights=[1., 3.], angles=128)
    e = .7*p[:, 0]*p[:, 1]*np.cos(2*p[:, 2])
    phi = result['angles']; den = 1-e[:, None]*np.cos(2*phi)
    u = (1-e[:, None])/den
    derivative = -(1-e[:, None])*2*e[:, None]*np.sin(2*phi)/den**2
    np.testing.assert_allclose(result['actions'], u*p[:, 3, None], atol=1e-13)
    np.testing.assert_allclose(result['normalized_derivatives'], derivative, atol=1e-13)
    expected = np.exp(-(np.max(abs(u-1), axis=1)/.5)**2
                      -.1*np.mean(derivative**2/(1+derivative**2), axis=1))
    np.testing.assert_allclose(result['scores'], expected, atol=1e-13)
    assert result['loss'] == pytest.approx(1-np.dot([.25, .75], expected))


def test_remote_disconnected_root_is_not_penalized():
    result = action_geometry_objective([1., -.1],
        state([J, (0, 4, 0, 0, 'c', 0, 'c')]), [[0., 0., 0., .2]])
    np.testing.assert_allclose(result['actions'], .2, atol=1e-13)
    assert result['quality'] == pytest.approx(1.)


def test_pole_is_not_skipped_from_average():
    result = action_geometry_objective([1.],
        state([(0, 2, 0, 2, 'c', 0, 'c')]), [[0., 0., 0., .2]])
    assert result['quality'] == 0.
    assert not result['resolved'][0]
    assert result['status'][0] != 'complete'


def test_large_bounded_excursion_scores_near_zero():
    result = action_geometry_objective([1., .95],
        state([J, (0, 2, 0, 2, 'c', 0, 'c')]), [[0., 0., 0., .2]], angles=512)
    assert result['resolved'][0]
    assert result['max_relative_excursion'][0] == pytest.approx(38.)
    assert result['quality'] < 1e-100


@pytest.mark.parametrize('kwargs', [dict(angles=7), dict(amplitude_scale=0),
    dict(slope_scale=-1), dict(derivative_weight=-1), dict(weights=[0.])])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        action_geometry_objective([1.], state([J]), [[0., 0., 0., .2]], **kwargs)
