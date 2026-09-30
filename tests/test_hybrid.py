import numpy as np

from fanqo.core import nonlinear as nl


def test_hybrid_is_plain_euclidean_least_squares():
    # A rank-deficient example where G-weighted and Euclidean LS differ.
    # D = I - T_nn
    D = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    tnn = np.eye(2) - D
    tnq = np.array([
        [0.0],
        [1.0],
    ])
    Sx = np.array([1.0])
    state = {
        "nonquad_size": 2,
        "Gnn": np.diag([1.0, 100.0]),
    }

    hybrid, h_details = nl.least_squares_ix(
        tnn, tnq, Sx, state, weighted=False
    )
    weighted, g_details = nl.least_squares_ix(
        tnn, tnq, Sx, state, weighted=True
    )

    # Euclidean solution minimizes h^2 + (h-1)^2 -> h = 1/2.
    assert np.isclose(hybrid[1], 0.5)

    # G weighting minimizes h^2 + 100(h-1)^2 -> h = 100/101.
    assert np.isclose(weighted[1], 100.0 / 101.0)

    assert h_details["weighted"] is False
    assert g_details["weighted"] is True
    assert not np.isclose(hybrid[1], weighted[1])
