import numpy as np

from fanqo.core import nonlinear as nl


def _toy_problem():
    # One fixed quadratic coefficient Sx=1 and one nonlinear coefficient h.
    # The nonlinear map is h -> 0.2 + 0.5 h, whose exact fixed point is h=0.4.
    tnn = np.array([[0.5]])
    tnq = np.array([[0.2]])
    Sx = np.array([1.0])
    state = {"nonquad_size": 1}
    return tnn, tnq, Sx, state


def test_cesaro_average_is_map_orbit_mean_and_keeps_sx_exact():
    tnn, tnq, Sx, state = _toy_problem()

    Ix, details = nl.cesaro_invariant(
        tnn, tnq, Sx, state, terms=4
    )

    # h_0=0, h_1=.2, h_2=.3, h_3=.35.
    expected_mean = (0.0 + 0.2 + 0.3 + 0.35) / 4.0
    assert np.array_equal(Ix[:1], Sx)
    assert np.isclose(Ix[1], expected_mean)
    assert details["terms"] == 4
    assert details["fixed_quadratic"] == "Sx"
    assert details["residual_identity_error"] < 1e-14


def test_cesaro_converges_toward_fixed_point():
    tnn, tnq, Sx, state = _toy_problem()

    short, _ = nl.cesaro_invariant(tnn, tnq, Sx, state, terms=4)
    long, _ = nl.cesaro_invariant(tnn, tnq, Sx, state, terms=64)

    assert abs(long[1] - 0.4) < abs(short[1] - 0.4)


def test_abel_resolvent_has_closed_form_and_keeps_sx_exact():
    tnn, tnq, Sx, state = _toy_problem()
    rho = 0.8

    Ix, details = nl.abel_invariant(
        tnn, tnq, Sx, state, rho=rho, tol=1e-16
    )

    expected = rho * 0.2 / (1.0 - rho * 0.5)
    assert np.array_equal(Ix[:1], Sx)
    assert np.isclose(Ix[1], expected)
    assert details["rho"] == rho
    assert details["fixed_quadratic"] == "Sx"
    assert details["resolvent_residual"] < 1e-14
    assert details["residual_identity_error"] < 1e-14


def test_abel_approaches_exact_fixed_point_as_rho_goes_to_one():
    tnn, tnq, Sx, state = _toy_problem()

    a90, _ = nl.abel_invariant(tnn, tnq, Sx, state, rho=0.90)
    a999, _ = nl.abel_invariant(tnn, tnq, Sx, state, rho=0.999)

    assert abs(a999[1] - 0.4) < abs(a90[1] - 0.4)
