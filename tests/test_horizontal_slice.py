import numpy as np
import sympy as sp

from fanqo.core import nonlinear as nl


def test_a_box_y0_uses_full_basis_and_horizontal_coefficient_mask():
    delta, x, y, px, py = sp.symbols("delta x y px py")
    b1, b2, b3, b4, b5 = sp.symbols("b1 b2 b3 b4 b5")

    hamiltonian = (
        px**2 / 2
        + b2 * (x**2 - y**2) / 2
        + b3 * (x**3 - 3*x*y**2) / 3
        + b4 * (x**4 - 6*x**2*y**2 + y**4) / 4
    )

    data = [np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])]
    state = nl.initialize_nonlinear_for_method(
        data=data,
        m=4,
        d=1,
        hamiltonian=hamiltonian,
        a_box=np.array([0.01, 10e-3, 8e-3, 1e-3, 0.8e-3]),
        variables=(delta, x, y, px, py),
        field_symbols=(b1, b2, b3, b4, b5),
        n=2,
        invariant_construction="a_box_y0",
    )

    # Same full basis as every other construction:
    # degree <= 4 in four transverse variables gives C(8,4)=70 terms/layer.
    assert len(state["idx_to_vec"]) == 140
    assert state["quad_size"] == 15
    assert state["nonquad_size"] == 125
    assert (0, 1, 2, 0, 0) in state["vec_to_idx"]

    # The constructor restriction is a coefficient mask, not a different basis.
    positions = nl.horizontal_nonquad_positions(state)
    assert len(positions) > 0
    q = state["quad_size"]
    for position in positions:
        powers = state["idx_to_vec"][q + int(position)]
        assert powers[2] == 0
        assert powers[4] == 0

    assert state["horizontal_only"] is True


def test_horizontal_only_ls_keeps_full_residual_equations():
    # D = I-Tnn. Only h0 is active, but both residual equations matter.
    D = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    tnn = np.eye(2) - D
    tnq = np.array([
        [0.0],
        [1.0],
    ])
    Sx = np.array([1.0])
    state = {
        "nonquad_size": 2,
        "Gnn": np.eye(2),
    }

    Ix, _ = nl.least_squares_ix(
        tnn,
        tnq,
        Sx,
        state,
        weighted=True,
        active_positions=np.array([0]),
    )

    # min_h h^2 + (h-1)^2 gives h=1/2; inactive h1 remains exactly zero.
    assert np.isclose(Ix[1], 0.5)
    assert np.isclose(Ix[2], 0.0)
