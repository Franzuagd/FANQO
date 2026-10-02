import numpy as np

from fanqo import tracking


def test_fma_grid_matches_at_interval_convention():
    xs, ys = tracking._fma_grid(
        [-15.0, 15.0, -15.0, 15.0],
        [121, 121],
    )

    assert len(xs) == 122
    assert len(ys) == 122
    assert np.isclose(xs[0], -15.0)
    assert np.isclose(xs[-1], 15.0)
    assert np.isclose(ys[0], -15.0)
    assert np.isclose(ys[-1], 15.0)


def test_initial_row_keeps_at_one_nanometer_offset():
    orbit = np.zeros(6)
    z0 = tracking._initial_row(
        np.array([0.0]),
        0.0,
        orbit,
        0.0,
    )

    assert np.isclose(z0[0, 0], 1.0e-9)
    assert np.isclose(z0[0, 2], 1.0e-9)
