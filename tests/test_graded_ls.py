import numpy as np

from fanqo.core import nonlinear as nl


def test_cs_grade_transform_maps_sx_to_normalized_circle():
    # Physical basis restricted to the horizontal quadratic monomials.
    state = {
        "idx_to_vec": {
            0: [0, 2, 0, 0, 0],
            1: [0, 1, 0, 1, 0],
            2: [0, 0, 0, 2, 0],
        },
        "linear_cs0": np.array([
            4.0,                 # beta_x
            1.5,                 # alpha_x
            (1.0 + 1.5**2) / 4.0,
            1.0, 0.0, 1.0,
        ]),
    }

    C = nl._cs_grade_coefficient_transform(state, np.array([0, 1, 2]))

    beta = 4.0
    alpha = 1.5
    gamma = (1.0 + alpha**2) / beta
    sx_physical = np.array([gamma, 2.0 * alpha, beta])

    sx_normalized = C @ sx_physical

    # Sx = gamma*x^2 + 2 alpha*x*px + beta*px^2
    # becomes X^2 + P_X^2 in normalized coordinates.
    assert np.allclose(sx_normalized, np.array([1.0, 0.0, 1.0]))


def test_fischer_weights_use_multiindex_factorial():
    state = {
        "idx_to_vec": {
            0: [0, 3, 0, 0, 0],
            1: [0, 2, 0, 1, 0],
            2: [2, 1, 0, 1, 0],
        }
    }

    w = nl._fischer_row_weights(state, np.array([0, 1, 2]))

    assert np.allclose(
        w,
        np.array([
            np.sqrt(6.0),  # 3!
            np.sqrt(2.0),  # 2! 1!
            1.0,           # delta degree is a grading label; 1! 1! transversely
        ]),
    )


def test_graded_ls_solves_degree_three_after_fixing_sx():
    # Minimal horizontal toy basis ordered like FANQO's low-order basis:
    # 1, x, px, x^2, x px, px^2, x^3.
    idx_to_vec = {
        0: [0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0],
        2: [0, 0, 0, 1, 0],
        3: [0, 2, 0, 0, 0],
        4: [0, 1, 0, 1, 0],
        5: [0, 0, 0, 2, 0],
        6: [0, 3, 0, 0, 0],
    }
    state = {
        "idx_to_vec": idx_to_vec,
        "vec_to_idx": {tuple(v): k for k, v in idx_to_vec.items()},
        "linear_cs0": np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        "C": np.ones(7),
        "quad_size": 6,
        "nonquad_size": 1,
        "horizontal_slice_only": True,
    }

    # Identity on lower degrees.  At degree 3:
    #   c3 = 0.2*c_x2 + 0.5*c3
    # and Sx has c_x2=1, so c3=0.4.
    T = np.eye(7)
    T[6, 6] = 0.5
    T[6, 3] = 0.2

    data = [np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])]

    Ix, details = nl.graded_least_squares_ix(T, data, state, tol=1e-14)

    assert np.allclose(Ix[:6], np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0]))
    assert np.isclose(Ix[6], 0.4)
    assert details["fixed_quadratic"] == "Sx"
    assert details["a_box_independent"] is True
    assert details["solved_grades"] == (3,)


def test_bigraded_solver_handles_same_total_degree_delta_block():
    # Sx lives in block (delta_degree=0, transverse_degree=2).
    # A dispersive/chromatic same-total-degree contribution can live in
    # (delta_degree=1, transverse_degree=1), and must be solved AFTER Sx.
    idx_to_vec = {
        0: [0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0],
        2: [0, 0, 0, 1, 0],
        3: [0, 2, 0, 0, 0],
        4: [0, 1, 0, 1, 0],
        5: [0, 0, 0, 2, 0],
        6: [1, 1, 0, 0, 0],
        7: [1, 0, 0, 1, 0],
    }
    state = {
        "idx_to_vec": idx_to_vec,
        "vec_to_idx": {tuple(v): k for k, v in idx_to_vec.items()},
        "linear_cs0": np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        "C": np.ones(8),
        "quad_size": 6,
        "nonquad_size": 2,
        "horizontal_slice_only": True,
    }

    T = np.eye(8)
    # delta*x = 0.2*x^2 + 0.5*(delta*x), hence invariant coefficient = 0.4.
    T[6, 6] = 0.5
    T[6, 3] = 0.2

    data = [np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])]
    Ix, details = nl.graded_least_squares_ix(T, data, state, tol=1e-14)

    assert np.isclose(Ix[6], 0.4)
    assert (1, 1) in details["solved_blocks"]
    assert details["grading"] == "(delta_degree, transverse_degree)"
    assert details["metric"] == "Fischer through C"
