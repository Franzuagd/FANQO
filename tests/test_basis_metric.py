import math

import numpy as np

from fanqo.core import nonlinear as nl


def _small_basis():
    idx_to_vec, _ = nl.monomial_index_maps(m=2, d=1, n=2)
    return idx_to_vec


def test_identity_coordinate_C_lifts_to_all_ones():
    idx_to_vec = _small_basis()
    C = nl.build_C(idx_to_vec, np.ones(5))
    assert np.allclose(C, 1.0)


def test_fischer_C_encodes_factorial_metric():
    idx_to_vec = {
        0: [0, 3, 0, 0, 0],
        1: [0, 2, 0, 1, 0],
        2: [1, 1, 0, 1, 0],
    }
    C = nl.C_fischer(idx_to_vec)

    assert np.isclose(C[0], 1.0 / math.sqrt(6.0))
    assert np.isclose(C[1], 1.0 / math.sqrt(2.0))
    assert np.isclose(C[2], 1.0)

    # Euclidean norm in stored coefficients equals the Fischer norm of
    # physical coefficients because physical = C * stored.
    stored = np.array([2.0, 3.0, 4.0])
    physical = C * stored
    fischer = (
        6.0 * physical[0] ** 2
        + 2.0 * physical[1] ** 2
        + 1.0 * physical[2] ** 2
    )
    assert np.isclose(fischer, stored @ stored)


def test_box_G_is_built_from_C_and_a_box():
    idx_to_vec = {
        0: [0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0],
        2: [0, 2, 0, 0, 0],
    }
    a_box = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
    C = nl.C_a_box(idx_to_vec, a_box)
    G = nl.G_box(idx_to_vec, C, a_box)

    # Odd moments vanish.
    assert np.isclose(G[0, 1], 0.0)

    # With C_a_box, every diagonal monomial has unit box-average norm.
    assert np.allclose(np.diag(G), 1.0)
