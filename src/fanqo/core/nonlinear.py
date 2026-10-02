"""Nonlinear polynomial map and invariant constructions.

This file is intentionally research-oriented.  Every invariant construction
uses the same full monomial index set

    [delta, x, y, px, py]

and differs only through the basis scale C, the metric G, and the constructor
itself.  To add a new invariant, write one small constructor and add it to
CONSTRUCTORS near the bottom of the file.

Convention
----------
The stored basis is

    e_i(z) = C_i z**alpha_i.

If c is a stored coefficient vector, its physical polynomial coefficients are

    c_physical = C * c.

C is therefore the central representation choice.  G is always built after C
and represents a metric in that scaled coefficient basis.
"""

from __future__ import annotations

import math
from math import comb

import numpy as np
import scipy.sparse as sps
import sympy as sp
from scipy.linalg import expm

from . import linear as lin


# =============================================================================
# 1. MONOMIAL BASIS
# =============================================================================

def indexmap(v):
    """Map a nonnegative exponent vector to the graded monomial index."""
    a = len(v)
    s = sum(v)
    index = sum(comb(t + a - 1, a - 1) for t in range(s))
    remaining = s
    for i in range(a - 1):
        for value in range(v[i] + 1, remaining + 1):
            index += comb(remaining - value + (a - i - 2), a - i - 2)
        remaining -= v[i]
    return index


def vectormap(idx, a):
    """Inverse of indexmap for exponent vectors of length a."""
    s = 0
    while True:
        count = comb(s + a - 1, a - 1)
        if idx < count:
            break
        idx -= count
        s += 1

    v = []
    remaining = s
    for i in range(a - 1):
        for value in range(remaining, -1, -1):
            count = comb(remaining - value + (a - i - 2), a - i - 2)
            if count <= idx:
                idx -= count
            else:
                v.append(value)
                remaining -= value
                break
    v.append(remaining)
    return v


def quadratic_size(n=2):
    """Number of delta=0 transverse monomials of degree <= 2."""
    ndim = 2 * n
    return sum(comb(s + ndim - 1, ndim - 1) for s in range(3))


def monomial_index_maps(m, d, n=2):
    """Build the common full basis used by every invariant method."""
    ndim = 2 * n
    layer_size = sum(comb(s + ndim - 1, ndim - 1) for s in range(m + 1))
    idx_to_vec = {}
    vec_to_idx = {}

    for delta_degree in range(d + 1):
        offset = delta_degree * layer_size
        for idx in range(layer_size):
            powers = [delta_degree] + vectormap(idx, ndim)
            idx_to_vec[offset + idx] = powers
            vec_to_idx[tuple(powers)] = offset + idx

    return idx_to_vec, vec_to_idx


def build_monomial_basis(variables, idx_to_vec):
    """Return symbolic physical monomials in the common index order."""
    basis = []
    for i in range(len(idx_to_vec)):
        term = 1
        for variable, power in zip(variables, idx_to_vec[i]):
            if power:
                term *= variable ** int(power)
        basis.append(term)
    return basis


def poly_to_vector(poly, variables, vec_to_idx):
    """Convert a symbolic polynomial to physical monomial coefficients."""
    if not isinstance(poly, sp.Poly):
        poly = sp.Poly(poly, *variables)
    vector = np.zeros(len(vec_to_idx), dtype=object)
    for monom, coefficient in poly.terms():
        if monom in vec_to_idx:
            vector[vec_to_idx[monom]] = coefficient
    return vector


def vector_to_poly(vec, monomial_basis):
    """Convert physical monomial coefficients to a SymPy expression."""
    expression = 0
    for coefficient, basis_term in zip(vec, monomial_basis):
        if coefficient != 0:
            expression += coefficient * basis_term
    return sp.expand(expression)


# =============================================================================
# 2. BASIS SCALE C AND METRIC G
# =============================================================================

def C_identity(idx_to_vec):
    """Unscaled physical monomial basis."""
    return np.ones(len(idx_to_vec), dtype=float)


def C_from_coordinates(idx_to_vec, coordinate_scale):
    """Lift a five-component coordinate scale to a monomial scale C."""
    scale = np.asarray(coordinate_scale, dtype=float)
    if scale.shape != (5,):
        raise ValueError("A coordinate C must have five entries [delta,x,y,px,py].")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("Coordinate C entries must be positive and finite.")

    C = np.ones(len(idx_to_vec), dtype=float)
    for i, powers in idx_to_vec.items():
        for value, power in zip(scale, powers):
            if power:
                C[i] *= float(value) ** int(power)
    return C


def C_a_box(idx_to_vec, a_box):
    """Original a_box monomial normalization written explicitly as C."""
    a = np.asarray(a_box, dtype=float)
    if a.shape != (5,) or not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("A_BOX must contain five positive finite half-widths.")

    C = np.ones(len(idx_to_vec), dtype=float)
    for i, powers in idx_to_vec.items():
        value = 1.0
        for half_width, power in zip(a, powers):
            power = int(power)
            value *= math.sqrt(2 * power + 1) / (float(half_width) ** power)
        C[i] = value
    return C


def C_fischer(idx_to_vec):
    """Scale so Euclidean stored coefficients carry the Fischer factorial norm.

    Delta is a grading parameter; factorials are applied only to transverse
    exponents.  Since physical_coeff = C * stored_coeff, choosing

        C_i = 1 / sqrt(alpha!)

    makes ||stored||_2^2 = sum alpha! |physical_coeff|^2.
    """
    C = np.ones(len(idx_to_vec), dtype=float)
    for i, powers in idx_to_vec.items():
        factorial = 1
        for power in powers[1:]:
            factorial *= math.factorial(int(power))
        C[i] = 1.0 / math.sqrt(float(factorial))
    return C


def build_C(idx_to_vec, specification="identity", *, a_box=None):
    """Build C from a named rule, a five-vector, a full vector, or a callable."""
    if callable(specification):
        C = np.asarray(specification(idx_to_vec, a_box), dtype=float)
    elif isinstance(specification, str):
        key = specification.lower()
        if key in {"identity", "physical", "ones"}:
            C = C_identity(idx_to_vec)
        elif key in {"a_box", "box"}:
            C = C_a_box(idx_to_vec, a_box)
        elif key in {"fischer", "factorial"}:
            C = C_fischer(idx_to_vec)
        else:
            raise ValueError(f"Unknown C specification: {specification!r}")
    else:
        values = np.asarray(specification, dtype=float)
        if values.shape == (5,):
            C = C_from_coordinates(idx_to_vec, values)
        elif values.shape == (len(idx_to_vec),):
            C = values.copy()
        else:
            raise ValueError(
                "C must be a named rule, a five-component coordinate scale, "
                "or one value per monomial."
            )

    if C.shape != (len(idx_to_vec),):
        raise ValueError("C has the wrong size for the polynomial basis.")
    if not np.all(np.isfinite(C)) or np.any(C <= 0.0):
        raise ValueError("Every monomial scale C_i must be positive and finite.")
    return C


def G_identity(C):
    """Euclidean metric in the scaled coefficient basis."""
    return np.eye(len(C), dtype=float)


def G_box(idx_to_vec, C, a_box):
    """L2 metric on the symmetric physical box, expressed in the C-scaled basis.

    The normalized box average is used, so for one coordinate

        E[z^p] = 0                    for odd p,
        E[z^p] = a^p / (p + 1)       for even p.

    Consequently G depends explicitly on C and optionally on A_BOX.
    """
    a = np.asarray(a_box, dtype=float)
    if a.shape != (5,) or np.any(a <= 0.0):
        raise ValueError("G='box' requires a positive five-component A_BOX.")

    size = len(idx_to_vec)
    G = np.zeros((size, size), dtype=float)
    for i in range(size):
        fi = idx_to_vec[i]
        for j in range(i, size):
            fj = idx_to_vec[j]
            value = float(C[i] * C[j])
            for half_width, pi, pj in zip(a, fi, fj):
                power = int(pi) + int(pj)
                if power % 2:
                    value = 0.0
                    break
                value *= float(half_width) ** power / (power + 1.0)
            G[i, j] = value
            G[j, i] = value
    return G


def G_fischer_physical(idx_to_vec, C):
    """Fischer metric for an arbitrary C-scaled representation."""
    diagonal = np.ones(len(idx_to_vec), dtype=float)
    for i, powers in idx_to_vec.items():
        factorial = 1
        for power in powers[1:]:
            factorial *= math.factorial(int(power))
        diagonal[i] = float(factorial) * float(C[i]) ** 2
    return np.diag(diagonal)


def build_G(idx_to_vec, C, specification="coefficient", *, a_box=None):
    """Build G after C.

    A custom metric can be supplied as a matrix or callable.  This function is
    the intended edit point for new research metrics.
    """
    if callable(specification):
        G = np.asarray(specification(idx_to_vec, C, a_box), dtype=float)
    elif isinstance(specification, str):
        key = specification.lower()
        if key in {"coefficient", "identity", "euclidean"}:
            G = G_identity(C)
        elif key in {"box", "a_box"}:
            G = G_box(idx_to_vec, C, a_box)
        elif key in {"fischer", "factorial"}:
            G = G_fischer_physical(idx_to_vec, C)
        else:
            raise ValueError(f"Unknown G specification: {specification!r}")
    else:
        G = np.asarray(specification, dtype=float).copy()

    expected = (len(idx_to_vec), len(idx_to_vec))
    if G.shape != expected:
        raise ValueError(f"G must have shape {expected}, received {G.shape}.")
    return G


# =============================================================================
# 3. LIE MAP IN THE C-SCALED BASIS
# =============================================================================

def hamiltonian_data(variables, hamiltonian, vec_to_idx, field_symbols):
    """Return physical Hamiltonian coefficients and a numerical evaluator."""
    vector = poly_to_vector(hamiltonian, variables, vec_to_idx)
    order = [i for i, value in enumerate(vector) if value != 0]
    symbolic = sp.Matrix([sp.sympify(vector[i]) for i in order])
    evaluator = sp.lambdify(list(field_symbols), symbolic, modules="numpy")
    return order, evaluator


def build_M_basis(idx_to_vec, vec_to_idx, C, hamiltonian_order, n=2):
    """Matrices for unit physical Hamiltonian monomials.

    M(H) f = {f,H}.  The basis is e_j=C_j z**alpha_j, while each Hamiltonian
    basis matrix corresponds to physical H=z**alpha_i.  Therefore

        M_i[k,j] = bracket(alpha_j, alpha_i) C_j / C_k.
    """
    size = len(idx_to_vec)
    matrices = []

    for i in hamiltonian_order:
        hi = idx_to_vec[i]
        M = sps.lil_matrix((size, size), dtype=float)

        for j in range(size):
            fj = idx_to_vec[j]
            base = [int(a) + int(b) for a, b in zip(fj, hi)]

            for plane in range(n):
                q = 1 + plane
                p = 3 + plane
                symplectic = int(fj[q]) * int(hi[p]) - int(fj[p]) * int(hi[q])
                if symplectic == 0:
                    continue

                target = base.copy()
                target[q] -= 1
                target[p] -= 1
                k = vec_to_idx.get(tuple(target))
                if k is not None:
                    M[k, j] += float(symplectic) * float(C[j]) / float(C[k])

        matrices.append(M.toarray())

    return matrices


def assemble_M(h_vec, M_basis):
    """Assemble M(H) from physical Hamiltonian coefficients."""
    M = np.zeros_like(M_basis[0], dtype=float)
    for coefficient, basis_matrix in zip(h_vec, M_basis):
        if coefficient != 0.0:
            M += float(coefficient) * basis_matrix
    return M


def element_hamiltonian_values(elem):
    """Return length and the five Hamiltonian field coefficients."""
    L = float(lin.magnet_field(elem, "LENGTH"))
    if L > 0.0:
        b1 = math.radians(float(lin.magnet_field(elem, "ANGLE"))) / L
    else:
        b1 = 0.0
    return (
        L,
        b1,
        float(lin.magnet_field(elem, "K")),
        float(lin.magnet_field(elem, "S")),
        float(lin.magnet_field(elem, "O")),
        0.0,
    )


def element_transfer(elem, state, tol=1e-14, check_upper_right=False, **_):
    """Polynomial coefficient transfer through one lattice element."""
    size = len(state["idx_to_vec"])
    identity = np.eye(size, dtype=float)
    L = float(lin.magnet_field(elem, "LENGTH"))
    typ = str(lin.magnet_field(elem, "TYPE"))
    O = float(lin.magnet_field(elem, "O"))

    if L == 0.0:
        if typ == "multipole" and O != 0.0:
            T = identity - O * state["M_octupole_unit"]
        else:
            T = identity
    else:
        L, b1, b2, b3, b4, b5 = element_hamiltonian_values(elem)
        h = np.asarray(
            state["H_vec_func"](b1, b2, b3, b4, b5),
            dtype=float,
        ).reshape(-1)
        T = expm(-L * assemble_M(h, state["M_basis"]))

    q = state["quad_size"]
    if check_upper_right and not np.all(np.abs(T[:q, q:]) < tol):
        raise ValueError("Upper-right nonlinear block is not zero.")

    return T, T[:q, :q], T[q:, q:], T[:q, q:], T[q:, :q]


def nonlinear_transfer(
    lattice,
    state,
    tol=1e-14,
    check_upper_right=False,
    cache=True,
    **_,
):
    """One-turn polynomial transfer in the current C-scaled basis."""
    size = len(state["idx_to_vec"])
    T = np.eye(size, dtype=float)
    element_cache = {}

    for elem in lattice:
        key = id(elem)
        if cache and key in element_cache:
            Te = element_cache[key]
        else:
            Te = element_transfer(
                elem,
                state,
                tol=tol,
                check_upper_right=check_upper_right,
            )[0]
            if cache:
                element_cache[key] = Te
        T = Te @ T

    q = state["quad_size"]
    return T, T[q:, q:], T[q:, :q]


# =============================================================================
# 4. STATE
# =============================================================================

DEFAULT_METHOD_OPTIONS = {
    "a_box": {"C": "a_box", "G": "box"},
    "a_box_y0": {"C": "a_box", "G": "box", "horizontal_only": True},
    "hybrid": {"C": [1, 1, 1, 1, 1], "G": "coefficient"},
    "eigen": {"C": [1, 1, 1, 1, 1], "G": "coefficient"},
    "graded_ls": {"C": "fischer", "G": "coefficient"},
    "cesaro": {"C": [1, 1, 1, 1, 1], "G": "coefficient"},
    "abel": {"C": [1, 1, 1, 1, 1], "G": "coefficient"},
}


def build_derivative_matrix(state, variable_index):
    """Derivative operator in the same C-scaled coefficient basis."""
    idx_to_vec = state["idx_to_vec"]
    vec_to_idx = state["vec_to_idx"]
    C = np.asarray(state["C"], dtype=float)
    size = len(idx_to_vec)
    rows, cols, values = [], [], []

    for k in range(size):
        powers = list(idx_to_vec[k])
        power = int(powers[variable_index])
        if power == 0:
            continue
        powers[variable_index] -= 1
        j = vec_to_idx.get(tuple(powers))
        if j is None:
            continue
        rows.append(j)
        cols.append(k)
        values.append(power * C[k] / C[j])

    return sps.csr_matrix((values, (rows, cols)), shape=(size, size))


def initialize_nonlinear_for_method(
    data,
    m,
    d,
    hamiltonian,
    a_box,
    variables,
    field_symbols,
    n=2,
    invariant_construction="a_box",
    options=None,
):
    """Build one research state on the common full monomial basis."""
    method = str(invariant_construction).lower()
    if method not in DEFAULT_METHOD_OPTIONS:
        raise ValueError(f"Unknown invariant construction: {method!r}")

    settings = dict(DEFAULT_METHOD_OPTIONS[method])
    if options:
        settings.update(dict(options))

    idx_to_vec, vec_to_idx = monomial_index_maps(m, d, n=n)
    C = build_C(idx_to_vec, settings.get("C", "identity"), a_box=a_box)
    G = build_G(
        idx_to_vec,
        C,
        settings.get("G", "coefficient"),
        a_box=a_box,
    )

    order, H_vec_func = hamiltonian_data(
        variables, hamiltonian, vec_to_idx, field_symbols
    )
    M_basis = build_M_basis(idx_to_vec, vec_to_idx, C, order, n=n)

    zero_fields = np.zeros(len(field_symbols), dtype=float)
    octupole_fields = zero_fields.copy()
    octupole_fields[3] = 1.0
    h0 = np.asarray(H_vec_func(*zero_fields), dtype=float).reshape(-1)
    h4 = np.asarray(H_vec_func(*octupole_fields), dtype=float).reshape(-1) - h0
    M_octupole_unit = assemble_M(h4, M_basis)

    q = quadratic_size(n)
    state = {
        "m": int(m),
        "d": int(d),
        "n": int(n),
        "variables": tuple(variables),
        "field_symbols": tuple(field_symbols),
        "hamiltonian": sp.expand(hamiltonian),
        "idx_to_vec": idx_to_vec,
        "vec_to_idx": vec_to_idx,
        "monomial_basis": build_monomial_basis(variables, idx_to_vec),
        "C": C,
        "G": G,
        "Gqq": G[:q, :q],
        "Gnn": G[q:, q:],
        "quad_size": q,
        "nonquad_size": len(idx_to_vec) - q,
        "H_vec_func": H_vec_func,
        "M_basis": M_basis,
        "M_octupole_unit": M_octupole_unit,
        "linear_cs0": np.asarray(lin.linear_data(data, "CS0"), dtype=float).copy(),
        "a_box": np.asarray(a_box, dtype=float).copy(),
        "invariant_construction": method,
        "horizontal_only": bool(settings.get("horizontal_only", False)),
        "method_options": settings,
    }
    state["D_x"] = build_derivative_matrix(state, 1)
    state["D_y"] = build_derivative_matrix(state, 2)
    state["D_px"] = build_derivative_matrix(state, 3)
    state["D_py"] = build_derivative_matrix(state, 4)
    return state


def initialize_nonlinear(data, m, d, hamiltonian, a_box, variables, field_symbols, n=2):
    """Compatibility wrapper for the a_box representation."""
    return initialize_nonlinear_for_method(
        data, m, d, hamiltonian, a_box, variables, field_symbols,
        n=n, invariant_construction="a_box"
    )


def initialize_nonlinear_eigen(data, m, d, hamiltonian, a_box, variables, field_symbols, n=2):
    """Compatibility wrapper for the unscaled eigen representation."""
    return initialize_nonlinear_for_method(
        data, m, d, hamiltonian, a_box, variables, field_symbols,
        n=n, invariant_construction="eigen"
    )


def load(m, d, hamiltonian, a_box, variables, field_symbols, n=2):
    """Compatibility alias returning the full a_box state without linear optics.

    New research code should use initialize_nonlinear_for_method.  This wrapper
    uses a neutral linear CS vector only because older tests called load
    directly.
    """
    fake_data = [np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])]
    return initialize_nonlinear_for_method(
        fake_data, m, d, hamiltonian, a_box, variables, field_symbols,
        n=n, invariant_construction="a_box"
    )


def load_eigen(m, d, hamiltonian, a_box, variables, field_symbols, n=2):
    """Compatibility alias for the eigen representation."""
    fake_data = [np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])]
    return initialize_nonlinear_for_method(
        fake_data, m, d, hamiltonian, a_box, variables, field_symbols,
        n=n, invariant_construction="eigen"
    )


# =============================================================================
# 5. COURANT-SNYDER SEED
# =============================================================================

def quadratic_invariants(data, state):
    """Stored coefficient vectors for the physical Courant-Snyder invariants."""
    bx, ax, gx, by, ay, gy = np.asarray(
        lin.linear_data(data, "CS0"), dtype=float
    )
    v = state["vec_to_idx"]
    C = np.asarray(state["C"], dtype=float)
    q = state["quad_size"]

    Sx = np.zeros(q, dtype=float)
    for powers, physical in (
        ((0, 2, 0, 0, 0), gx),
        ((0, 1, 0, 1, 0), 2.0 * ax),
        ((0, 0, 0, 2, 0), bx),
    ):
        i = v[powers]
        Sx[i] = physical / C[i]

    if state.get("horizontal_slice_only", False):
        return Sx, None

    Sy = np.zeros(q, dtype=float)
    for powers, physical in (
        ((0, 0, 2, 0, 0), gy),
        ((0, 0, 1, 0, 1), 2.0 * ay),
        ((0, 0, 0, 0, 2), by),
    ):
        i = v[powers]
        Sy[i] = physical / C[i]

    return Sx, Sy


def horizontal_nonquad_positions(state):
    """Positions in the nonlinear block containing no y or py powers."""
    q = state["quad_size"]
    positions = []
    for global_index in range(q, len(state["idx_to_vec"])):
        powers = state["idx_to_vec"][global_index]
        if int(powers[2]) == 0 and int(powers[4]) == 0:
            positions.append(global_index - q)
    return np.asarray(positions, dtype=int)


# =============================================================================
# 6. INVARIANT CONSTRUCTORS
# =============================================================================

def least_squares_ix(
    tnn,
    tnq,
    Sx,
    state,
    tol=1e-14,
    *,
    weighted=True,
    active_positions=None,
):
    """Fixed-Sx least squares, optionally restricted to a coefficient subspace."""
    D = np.eye(state["nonquad_size"]) - np.asarray(tnn, dtype=float)
    U = np.asarray(tnq, dtype=float) @ np.asarray(Sx, dtype=float)

    if active_positions is None:
        active = np.arange(state["nonquad_size"], dtype=int)
    else:
        active = np.asarray(active_positions, dtype=int)

    # Restrict the unknown polynomial coefficients, not the invariance
    # equations.  This is the scientifically useful interpretation for
    # a_box_y0: coefficients containing y or py are fixed to zero, while the
    # residual is still measured in the complete 5-D polynomial space.
    Ds = D[:, active]
    Us = U

    if weighted:
        Gs = np.asarray(state["Gnn"], dtype=float)
        L = np.linalg.cholesky(Gs)
        hs, _, rank, singular = np.linalg.lstsq(
            L.T @ Ds,
            L.T @ Us,
            rcond=tol,
        )
    else:
        hs, _, rank, singular = np.linalg.lstsq(Ds, Us, rcond=tol)

    h = np.zeros(state["nonquad_size"], dtype=float)
    h[active] = hs
    residual = U - D @ h

    details = {
        "weighted": bool(weighted),
        "active_coefficients": int(len(active)),
        "rank": int(rank),
        "residual": float(np.linalg.norm(residual)),
    }
    if singular.size:
        details["min_relative_singular"] = (
            float(singular[-1] / singular[0]) if singular[0] > 0 else float("nan")
        )
    return np.concatenate((np.asarray(Sx, dtype=float), h)), details


def _eigenvalue_cutoff(minimum_positive):
    value = abs(float(minimum_positive))
    if 0.0 < value < 1.0e-12:
        return 1.0e-10
    if value < 1.0e-8:
        return 1.0e-8
    if value < 1.0e-4:
        return 1.0e-4
    return 1.0e-1


def eigen_invariant(transfer, state, *, plane="x", imag_tol=1e-12):
    """Near-fixed eigenvector of T, normalized by its quadratic determinant."""
    transfer = np.asarray(transfer, dtype=float)
    v = state["vec_to_idx"]

    if plane == "x":
        iq2, iqp, ip2 = v[(0, 2, 0, 0, 0)], v[(0, 1, 0, 1, 0)], v[(0, 0, 0, 2, 0)]
    else:
        iq2, iqp, ip2 = v[(0, 0, 2, 0, 0)], v[(0, 0, 1, 0, 1)], v[(0, 0, 0, 0, 2)]

    values, vectors = np.linalg.eig(transfer - np.eye(len(transfer)))
    real = np.abs(np.imag(values)) <= imag_tol
    positive = np.real(values[real])
    positive = positive[positive > 0.0]
    if positive.size == 0:
        raise ValueError("No positive real near-fixed eigenvalue was found.")

    cutoff = _eigenvalue_cutoff(np.min(positive))
    candidates = []

    for k, value in enumerate(values):
        if abs(float(np.imag(value))) > imag_tol:
            continue
        if not 0.0 < float(np.real(value)) < cutoff:
            continue

        w = np.real(vectors[:, k]).astype(float)
        determinant = w[iq2] * w[ip2] - (0.5 * w[iqp]) ** 2
        if determinant <= 0.0 or not np.isfinite(determinant):
            continue
        sign = np.sign(w[ip2]) or 1.0
        w *= float(sign) / math.sqrt(float(determinant))
        residual = transfer @ w - w
        candidates.append((
            abs(float(residual[iqp])),
            abs(float(np.real(value))),
            w,
        ))

    if not candidates:
        raise ValueError("No normalizable near-fixed eigenvector was found.")

    candidates.sort(key=lambda item: (item[0], item[1]))
    selection, eigen_residual, vector = candidates[0]
    return vector, {
        "selection_residual": float(selection),
        "eigenvalue_residual": float(eigen_residual),
        "candidate_count": int(len(candidates)),
        "cutoff": float(cutoff),
    }


def _polynomial_bidegree(powers):
    return int(powers[0]), int(sum(int(p) for p in powers[1:]))


def _grade_index_map(state):
    groups = {}
    for i, powers in state["idx_to_vec"].items():
        groups.setdefault(_polynomial_bidegree(powers), []).append(int(i))
    return {block: np.asarray(indices, dtype=int) for block, indices in groups.items()}


def _graded_block_order(block):
    delta_degree, transverse_degree = block
    return (
        int(delta_degree + transverse_degree),
        int(delta_degree),
        int(transverse_degree),
    )


def _fischer_row_weights(state, indices):
    """Compatibility helper: sqrt(alpha!) for transverse exponents."""
    weights = []
    for i in indices:
        factorial = 1
        for power in state["idx_to_vec"][int(i)][1:]:
            factorial *= math.factorial(int(power))
        weights.append(math.sqrt(float(factorial)))
    return np.asarray(weights, dtype=float)


def _cs_grade_coefficient_transform(state, indices):
    """Compatibility research helper for a CS-normalized homogeneous block."""
    indices = np.asarray(indices, dtype=int)
    exponents = [tuple(state["idx_to_vec"][int(i)]) for i in indices]
    lookup = {powers: i for i, powers in enumerate(exponents)}
    bx, ax, _, by, ay, _ = np.asarray(state["linear_cs0"], dtype=float)
    sbx, sby = math.sqrt(bx), math.sqrt(by)
    transform = np.zeros((len(indices), len(indices)), dtype=float)

    for column, powers in enumerate(exponents):
        dd, xp, yp, pxp, pyp = map(int, powers)
        for Pxp in range(pxp + 1):
            xf = pxp - Pxp
            cx = (
                sbx ** xp * math.comb(pxp, Pxp) * (-ax) ** xf / sbx ** pxp
            )
            Xp = xp + xf
            for Pyp in range(pyp + 1):
                yf = pyp - Pyp
                cy = (
                    sby ** yp * math.comb(pyp, Pyp) * (-ay) ** yf / sby ** pyp
                )
                Yp = yp + yf
                row = lookup[(dd, Xp, Yp, Pxp, Pyp)]
                transform[row, column] += cx * cy
    return transform


def graded_least_squares_ix(transfer, data, state, tol=1e-14):
    """Bi-graded fixed-Sx solve with the original CS/Fischer mathematics.

    C is only the stored-basis representation.  Before the graded solve, the
    transfer and seed are converted back to physical monomial coefficients.
    Each homogeneous block is then transformed to Courant-Snyder normalized
    coordinates and solved with the factorial Fischer residual norm, exactly as
    in the earlier working graded_ls implementation.

    This is important: a diagonal factorial C alone is not equivalent to the
    Courant-Snyder transformation when alpha != 0.  The CS transform must
    remain explicit.
    """
    transfer = np.asarray(transfer, dtype=float)
    scale = np.asarray(state["C"], dtype=float)
    size = len(state["idx_to_vec"])

    if transfer.shape != (size, size):
        raise ValueError("Transfer shape does not match the polynomial basis.")

    # stored coefficients c satisfy physical_coefficients = C*c.
    # Therefore T_physical = C*T_stored*C^{-1}.
    transfer_physical = (
        scale[:, None] * transfer / scale[None, :]
    )

    groups = _grade_index_map(state)
    blocks = tuple(sorted(groups, key=_graded_block_order))
    if not blocks:
        raise ValueError("graded_ls found an empty polynomial basis.")

    Sx_stored, _ = quadratic_invariants(data, state)
    q = len(Sx_stored)

    fixed_physical = np.zeros(size, dtype=float)
    fixed_physical[:q] = scale[:q] * np.asarray(Sx_stored, dtype=float)

    transforms = {}
    inverse_transforms = {}
    coefficients_normalized = {}
    fixed_blocks = {(0, 0), (0, 1), (0, 2)}

    for block in blocks:
        indices = groups[block]
        transform = _cs_grade_coefficient_transform(state, indices)
        transforms[block] = transform
        inverse_transforms[block] = np.linalg.inv(transform)

        if block in fixed_blocks:
            coefficients_normalized[block] = (
                transform @ fixed_physical[indices]
            )

    invariant_physical = fixed_physical.copy()
    solved_blocks = []
    block_details = []

    for block in blocks:
        if block in fixed_blocks:
            continue

        rows = groups[block]
        Cb = transforms[block]
        Cb_inv = inverse_transforms[block]

        Tbb_physical = transfer_physical[np.ix_(rows, rows)]
        Tbb_normalized = Cb @ Tbb_physical @ Cb_inv
        D = np.eye(len(rows), dtype=float) - Tbb_normalized

        U = np.zeros(len(rows), dtype=float)
        for previous_block in blocks:
            if _graded_block_order(previous_block) >= _graded_block_order(block):
                break

            previous_coefficients = coefficients_normalized.get(previous_block)
            if previous_coefficients is None or not np.any(previous_coefficients):
                continue

            columns = groups[previous_block]
            Tba_physical = transfer_physical[np.ix_(rows, columns)]
            Tba_normalized = (
                Cb
                @ Tba_physical
                @ inverse_transforms[previous_block]
            )
            U += Tba_normalized @ previous_coefficients

        weights = _fischer_row_weights(state, rows)
        weighted_D = weights[:, None] * D
        weighted_U = weights * U

        solution, _, rank, singular_values = np.linalg.lstsq(
            weighted_D,
            weighted_U,
            rcond=tol,
        )
        coefficients_normalized[block] = solution
        invariant_physical[rows] = Cb_inv @ solution
        solved_blocks.append(block)

        residual = U - D @ solution
        fischer_residual = float(np.linalg.norm(weights * residual))
        fischer_rhs = float(np.linalg.norm(weighted_U))

        block_details.append({
            "delta_degree": int(block[0]),
            "transverse_degree": int(block[1]),
            "total_degree": int(block[0] + block[1]),
            "size": int(len(rows)),
            "rank": int(rank),
            "fischer_residual": fischer_residual,
            "fischer_rhs_norm": fischer_rhs,
            "relative_residual": (
                fischer_residual / fischer_rhs
                if fischer_rhs > 0.0 else 0.0
            ),
            "min_relative_singular": (
                float(singular_values[-1] / singular_values[0])
                if singular_values.size and singular_values[0] > 0.0
                else float("nan")
            ),
        })

    invariant_stored = invariant_physical / scale
    # Preserve the exact stored Courant-Snyder seed, without roundoff from
    # physical->stored conversion.
    invariant_stored[:q] = Sx_stored

    return invariant_stored, {
        "fixed_quadratic": "Sx",
        "grading": "(delta_degree, transverse_degree)",
        "block_order": "(total_degree, delta_degree)",
        "coordinate_system": "Courant-Snyder normalized transverse coordinates",
        "metric": "Fischer factorial metric: alpha!",
        "solved_blocks": tuple(solved_blocks),
        "solved_grades": tuple(sorted({a + b for a, b in solved_blocks})),
        "grade_details": tuple(block_details),
    }


def cesaro_invariant(tnn, tnq, Sx, state, terms=64):
    """Cesaro average of the map orbit of the fixed Courant-Snyder seed."""
    terms = int(terms)
    forcing = np.asarray(tnq) @ np.asarray(Sx)
    h = np.zeros(state["nonquad_size"], dtype=float)
    total = np.zeros_like(h)

    for _ in range(terms):
        total += h
        h = forcing + np.asarray(tnn) @ h
        if not np.all(np.isfinite(h)):
            raise FloatingPointError("Cesaro iterates became non-finite.")

    mean = total / float(terms)
    residual = forcing + np.asarray(tnn) @ mean - mean
    return np.concatenate((Sx, mean)), {
        "fixed_quadratic": "Sx",
        "terms": terms,
        "residual": float(np.linalg.norm(residual)),
    }


def abel_invariant(tnn, tnq, Sx, state, rho=0.98, tol=1e-14):
    """Abel/resolvent average with the Courant-Snyder block kept exact."""
    rho = float(rho)
    forcing = np.asarray(tnq) @ np.asarray(Sx)
    A = np.eye(state["nonquad_size"]) - rho * np.asarray(tnn)
    b = rho * forcing

    try:
        h = np.linalg.solve(A, b)
        solver = "solve"
    except np.linalg.LinAlgError:
        h, _, _, _ = np.linalg.lstsq(A, b, rcond=tol)
        solver = "lstsq"

    return np.concatenate((Sx, h)), {
        "fixed_quadratic": "Sx",
        "rho": rho,
        "solver": solver,
        "residual": float(np.linalg.norm(A @ h - b)),
    }


def construct_a_box(transfer, tnn, tnq, data, state, tol):
    Sx, _ = quadratic_invariants(data, state)
    return least_squares_ix(tnn, tnq, Sx, state, tol=tol, weighted=True)


def construct_a_box_y0(transfer, tnn, tnq, data, state, tol):
    Sx, _ = quadratic_invariants(data, state)
    return least_squares_ix(
        tnn,
        tnq,
        Sx,
        state,
        tol=tol,
        weighted=True,
        active_positions=horizontal_nonquad_positions(state),
    )


def construct_hybrid(transfer, tnn, tnq, data, state, tol):
    Sx, _ = quadratic_invariants(data, state)
    return least_squares_ix(tnn, tnq, Sx, state, tol=tol, weighted=False)


def construct_eigen(transfer, tnn, tnq, data, state, tol):
    return eigen_invariant(transfer, state, plane="x")


def construct_graded_ls(transfer, tnn, tnq, data, state, tol):
    return graded_least_squares_ix(transfer, data, state, tol=tol)


def construct_cesaro(transfer, tnn, tnq, data, state, tol):
    Sx, _ = quadratic_invariants(data, state)
    return cesaro_invariant(
        tnn, tnq, Sx, state, terms=int(state.get("cesaro_terms", 64))
    )


def construct_abel(transfer, tnn, tnq, data, state, tol):
    Sx, _ = quadratic_invariants(data, state)
    return abel_invariant(
        tnn, tnq, Sx, state,
        rho=float(state.get("abel_rho", 0.98)),
        tol=tol,
    )


CONSTRUCTORS = {
    "a_box": construct_a_box,
    "hybrid": construct_hybrid,
    "eigen": construct_eigen,
    "graded_ls": construct_graded_ls,
    "cesaro": construct_cesaro,
    "abel": construct_abel,
    "a_box_y0": construct_a_box_y0,
}


def construct_ix(lattice, data, state, tol=1e-14, cache=True):
    """Construct Ix with the research constructor selected in state."""
    transfer, tnn, tnq = nonlinear_transfer(
        lattice, state, tol=tol, cache=cache
    )
    method = state["invariant_construction"]
    constructor = CONSTRUCTORS[method]
    Ix, details = constructor(transfer, tnn, tnq, data, state, tol)
    details = {"method": method, **dict(details)}
    return np.asarray(Ix, dtype=float), details, transfer


# =============================================================================
# 7. EVALUATION
# =============================================================================

def physical_coefficients(coefficients, state):
    return np.asarray(coefficients, dtype=float) * np.asarray(state["C"], dtype=float)


def evaluate_physical_coefficients(coefficients, idx_to_vec, coordinates):
    """Evaluate physical polynomial coefficients on AT-order coordinates.

    coordinates has first dimension [x,px,y,py,delta,ct].
    """
    z = np.asarray(coordinates, dtype=float)
    values = (
        z[4],
        z[0],
        z[2],
        z[1],
        z[3],
    )
    out = np.zeros(np.broadcast_shapes(*(np.asarray(v).shape for v in values)))
    values = tuple(np.broadcast_to(v, out.shape) for v in values)

    for coefficient, powers in zip(coefficients, idx_to_vec.values()):
        if coefficient == 0.0:
            continue
        term = float(coefficient)
        for value, power in zip(values, powers):
            if power:
                term = term * value ** int(power)
        out = out + term
    return out


def evaluate_invariant(Ix, state, coordinates):
    """Evaluate an invariant coefficient vector on AT-order coordinates."""
    return evaluate_physical_coefficients(
        physical_coefficients(Ix, state),
        state["idx_to_vec"],
        coordinates,
    )


def eval_plane(I, state, Q, P, plane="x", delta0=0.0, frozen_q0=0.0, frozen_p0=0.0):
    """Compatibility helper for two-dimensional invariant sections."""
    Q = np.asarray(Q, dtype=float)
    P = np.asarray(P, dtype=float)
    shape = np.broadcast_shapes(Q.shape, P.shape)
    Q, P = np.broadcast_to(Q, shape), np.broadcast_to(P, shape)
    coordinates = np.zeros((6,) + shape, dtype=float)
    coordinates[4] = float(delta0)

    if plane.lower() == "x":
        coordinates[0], coordinates[1] = Q, P
        coordinates[2], coordinates[3] = float(frozen_q0), float(frozen_p0)
    elif plane.lower() == "y":
        coordinates[2], coordinates[3] = Q, P
        coordinates[0], coordinates[1] = float(frozen_q0), float(frozen_p0)
    else:
        raise ValueError("plane must be 'x' or 'y'.")

    return evaluate_invariant(I, state, coordinates)


def Non_linear_Transfer(lattice, H_vec, M_basis):
    """Deprecated compatibility name."""
    raise RuntimeError("Use nonlinear_transfer(lattice, state).")
