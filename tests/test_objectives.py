import numpy as np

from fanqo.core.objective_functions import h_reduction


def test_h_reduction_uses_only_nonlinear_tail():
    state = {
        "quad_size": 3,
        "idx_to_vec": {
            0: [0, 0, 0, 0, 0],
            1: [0, 1, 0, 0, 0],
            2: [0, 2, 0, 0, 0],
            3: [0, 3, 0, 0, 0],
            4: [0, 4, 0, 0, 0],
        },
        "Gnn": np.array([
            [2.0, 0.5],
            [0.5, 3.0],
        ]),
    }

    # Changes in the first quad_size coefficients must not affect J_h.
    Ix1 = np.array([1.0, 2.0, 3.0, 4.0, -1.0])
    Ix2 = np.array([9.0, -8.0, 7.0, 4.0, -1.0])

    value1, details1 = h_reduction({"Ix": Ix1, "state": state})
    value2, details2 = h_reduction({"Ix": Ix2, "state": state})

    h = np.array([4.0, -1.0])
    expected = np.sqrt(h @ state["Gnn"] @ h)

    assert np.isclose(value1, expected)
    assert np.isclose(value2, expected)
    assert details1["h_coefficients"] == 2
    assert details2["h_norm"] == value2
