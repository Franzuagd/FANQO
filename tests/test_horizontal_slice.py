import numpy as np
import sympy as sp

from fanqo.core import nonlinear as nl


def test_a_box_y_zero_builds_horizontal_only_basis():
    delta, x, y, px, py = sp.symbols("delta x y px py")
    b1, b2, b3, b4, b5 = sp.symbols("b1 b2 b3 b4 b5")

    # Include an explicit x-y coupling term. It must disappear from the reduced
    # y=py=0 polynomial representation because the slice is built algebraically,
    # not by replacing y with a very small nonzero scale.
    hamiltonian = (
        px**2 / 2
        + b2 * x**2 / 2
        + b3 * x**3 / 6
        + b3 * x * y**2
        + b4 * x**4 / 24
    )

    state = nl.load(
        m=4,
        d=1,
        hamiltonian=hamiltonian,
        a_box=np.array([0.01, 10e-3, 0.0, 1e-3, 0.8e-3]),
        variables=(delta, x, y, px, py),
        field_symbols=(b1, b2, b3, b4, b5),
        n=2,
    )

    assert state["horizontal_slice_only"] is True
    assert state["active_variables"] == ("delta", "x", "px")
    assert state["a_box"][2] == 0.0
    assert state["a_box"][4] == 0.0

    # Degree <=4 in two transverse variables has 1+2+3+4+5 = 15 monomials
    # per delta layer. DELTA_ORDER=1 therefore gives 30 coefficients.
    assert len(state["idx_to_vec"]) == 30
    assert state["G"].shape == (30, 30)

    # The delta=0 quadratic block in (x,px) contains 1+2+3 = 6 monomials.
    assert state["quad_size"] == 6
    assert state["nonquad_size"] == 24

    # Public exponent labels remain [delta,x,y,px,py], but y and py powers are
    # identically zero throughout the reduced basis.
    for powers in state["idx_to_vec"].values():
        assert len(powers) == 5
        assert powers[2] == 0
        assert powers[4] == 0

    # Cross-plane Hamiltonian monomials cannot survive the reduced basis.
    assert (0, 1, 2, 0, 0) not in state["vec_to_idx"]
