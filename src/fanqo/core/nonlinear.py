"""General nonlinear polynomial/invariant machinery.

This module contains only imports and reusable functions.  It has no machine-
specific Hamiltonian, polynomial order, phase-space box, plotting settings, or
executable main().  Those choices belong in general_config.py and the runner.

The linear accelerator itself is supplied by linear_lattice.py and the lattice file selected by general_config.py.
"""

import math
import os
from datetime import datetime
from math import comb

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import sympy as sp
from scipy.linalg import expm

from . import linear as lin


def poly_to_vector(poly, variables, vec_to_idx):
    """Convert a symbolic polynomial to the coefficient vector of the basis."""
    if not isinstance(poly, sp.Poly):
        poly = sp.Poly(poly, *variables)

    vec = np.zeros(len(vec_to_idx), dtype=object)
    for monom, coeff in poly.terms():
        if monom in vec_to_idx:
            vec[vec_to_idx[monom]] = coeff
    return vec


def build_monomial_basis(variables, idx_to_vec):
    """Build physical monomials in the same order as idx_to_vec."""
    basis = []
    for idx in range(len(idx_to_vec)):
        powers = idx_to_vec[idx]
        term = 1
        for variable, power in zip(variables, powers):
            if power:
                term *= variable ** power
        basis.append(term)
    return basis


def vector_to_poly(vec, monomial_basis):
    """Convert a coefficient vector in a physical monomial basis to SymPy."""
    expr = 0
    for coeff, basis in zip(vec, monomial_basis):
        if coeff != 0:
            expr += coeff * basis
    return sp.expand(expr)


def hamiltonian_dict(variables, hamiltonian, vec_to_idx):
    """Return the nonzero Hamiltonian coefficients keyed by basis index."""
    h_vec = poly_to_vector(hamiltonian, variables, vec_to_idx)
    return {i: h_vec[i] for i in range(len(h_vec)) if h_vec[i] != 0}


def hamiltonean_dict(variables, hamiltonian, vec_to_idx):
    """Backward-compatible spelling of hamiltonian_dict()."""
    return hamiltonian_dict(variables, hamiltonian, vec_to_idx)


def indexmap(v):
    """Map a nonnegative exponent vector to the graded monomial index."""
    a = len(v)
    s = sum(v)
    index = 0

    for t in range(s):
        index += comb(t + a - 1, a - 1)

    remaining_sum = s
    for i in range(a - 1):
        for val in range(v[i] + 1, remaining_sum + 1):
            index += comb(remaining_sum - val + (a - i - 2), a - i - 2)
        remaining_sum -= v[i]

    return index


def vectormap(idx, a):
    """Inverse of indexmap() for exponent vectors of length a."""
    s = 0
    while True:
        count = comb(s + a - 1, a - 1)
        if idx < count:
            break
        idx -= count
        s += 1

    v = []
    remaining_sum = s
    for i in range(a - 1):
        for val in range(remaining_sum, -1, -1):
            count = comb(remaining_sum - val + (a - i - 2), a - i - 2)
            if count <= idx:
                idx -= count
            else:
                v.append(val)
                remaining_sum -= val
                break

    v.append(remaining_sum)
    return v


def quadratic_size(n=2):
    """Number of transverse monomials of degree <= 2 for n canonical planes."""
    ndim = 2 * n
    return sum(comb(s + ndim - 1, ndim - 1) for s in range(3))


def load(m, d, hamiltonian, a_box, variables, field_symbols, n=2):
    """Build every symbolic/numerical object needed by the nonlinear model.

    Parameters
    ----------
    m : int
        Maximum transverse polynomial degree.
    d : int
        Maximum delta degree.
    hamiltonian : sympy expression
        User-specified Hamiltonian from general_config.py.
    a_box : sequence
        Physical half-widths ordered exactly like ``variables``.
    variables : sequence of sympy.Symbol
        Polynomial variables, expected as [delta, x, y, px, py] for n=2.
    field_symbols : sequence of sympy.Symbol
        Element coefficients used by the Hamiltonian, expected to receive
        [curvature, K, S, O, fifth-order coefficient].
    """
    if n != 2:
        raise NotImplementedError("The current block structure assumes n=2 (x and y planes).")
    if m < 2:
        raise ValueError("m must be at least 2 because the invariant contains a quadratic block.")
    if d < 0:
        raise ValueError("d must be nonnegative.")
    if len(variables) != 5:
        raise ValueError("The current model requires variables=[delta, x, y, px, py].")
    if len(field_symbols) != 5:
        raise ValueError("field_symbols must contain exactly five symbols [b1,b2,b3,b4,b5].")

    ndim = 2 * n
    idx_to_vec = {}
    vec_to_idx = {}

    layer_size = sum(comb(s + ndim - 1, ndim - 1) for s in range(m + 1))

    for delta_degree in range(d + 1):
        offset = delta_degree * layer_size
        for idx in range(layer_size):
            v = [delta_degree] + vectormap(idx, ndim)
            idx_to_vec[offset + idx] = v
            vec_to_idx[tuple(v)] = offset + idx

    size = len(idx_to_vec)
    dim = len(idx_to_vec[0])
    bracket_pairs = {}
    keys = list(idx_to_vec)

    for i in keys:
        fi = idx_to_vec[i]
        for j in keys:
            fj = idx_to_vec[j]
            base = [fi[t] + fj[t] for t in range(dim)]

            for plane in range(n):
                q_idx = 1 + plane
                p_idx = 3 + plane
                if base[q_idx] == 0 or base[p_idx] == 0:
                    continue

                cand = base.copy()
                cand[q_idx] -= 1
                cand[p_idx] -= 1
                k = vec_to_idx.get(tuple(cand))
                if k is not None:
                    bracket_pairs.setdefault((k, i), []).append((j, plane))

    h_dict = hamiltonian_dict(variables, hamiltonian, vec_to_idx)

    G = np.zeros((size, size), dtype=float)
    for i in range(size):
        fi = idx_to_vec[i]
        for j in range(i, size):
            fj = idx_to_vec[j]
            val = 1.0
            for l in range(dim):
                s_ij = fi[l] + fj[l]
                if s_ij % 2:
                    val = 0.0
                    break
                val *= np.sqrt((2 * fi[l] + 1) * (2 * fj[l] + 1)) / (s_ij + 1)
            G[i, j] = val
            G[j, i] = val

    if np.isscalar(a_box):
        a_box = np.full(dim, float(a_box))
    else:
        a_box = np.asarray(a_box, dtype=float)

    if len(a_box) != dim:
        raise ValueError(f"a_box must have {dim} entries, received {len(a_box)}.")
    if np.any(a_box <= 0.0):
        raise ValueError("Every a_box entry must be positive.")

    epsilon = np.zeros(size, dtype=float)
    for i in range(size):
        fi = idx_to_vec[i]
        val = 1.0
        for l in range(dim):
            val *= np.sqrt((2 * fi[l] + 1) / (a_box[l] ** (2 * fi[l])))
        epsilon[i] = val

    B = [sps.lil_matrix((size, size), dtype=float) for _ in range(size)]
    for (k, i), pairs in bracket_pairs.items():
        fi = idx_to_vec[i]
        for j, plane in pairs:
            fj = idx_to_vec[j]
            q_idx = 1 + plane
            p_idx = 3 + plane
            sympl = fi[q_idx] * fj[p_idx] - fi[p_idx] * fj[q_idx]
            if sympl:
                B[k][i, j] += sympl * epsilon[i] * epsilon[j] / epsilon[k]

    B = [matrix.tocsr() for matrix in B]
    order = sorted(h_dict)
    h_vec = sp.Matrix([sp.sympify(h_dict[j]) for j in order])
    h_vec_func = sp.lambdify(list(field_symbols), h_vec, modules="numpy")

    # B[k][i, j] stores the coefficient of {e_i, e_j}.
    #
    # For transport we define the Hamiltonian Lie matrix by
    #
    #       M(H) f = {f, H}.
    #
    # The row selected below is i = Hamiltonian basis index, so B gives
    # {H, f}.  The minus sign converts it to {f, H}.  With this convention
    # coefficient transport is dc/ds = -M(H)c and therefore T = exp(-L M).
    M_basis = []
    for i in order:
        Mi = sps.lil_matrix((size, size), dtype=float)
        for k in range(size):
            row = B[k].getrow(i)
            if row.nnz:
                for j, val in zip(row.indices, row.data):
                    Mi[k, j] = -val / epsilon[i]
        M_basis.append(Mi.toarray())

    # Isolate the unit integrated-octupole Hamiltonian contribution.  Evaluating
    # H at b4=1 alone would also include field-independent kinetic terms.
    zero_fields = np.zeros(len(field_symbols), dtype=float)
    octupole_fields = zero_fields.copy()
    octupole_fields[3] = 1.0
    h_zero = np.asarray(h_vec_func(*zero_fields), dtype=float).reshape(-1)
    h_oct = (
        np.asarray(h_vec_func(*octupole_fields), dtype=float).reshape(-1)
        - h_zero
    )

    M_octupole_unit = np.zeros(M_basis[0].shape, dtype=float)
    for coeff, basis_matrix in zip(h_oct, M_basis):
        if coeff != 0.0:
            M_octupole_unit += float(coeff) * basis_matrix

    return {
        "m": int(m),
        "d": int(d),
        "n": int(n),
        "variables": tuple(variables),
        "field_symbols": tuple(field_symbols),
        "hamiltonian": sp.expand(hamiltonian),
        "idx_to_vec": idx_to_vec,
        "vec_to_idx": vec_to_idx,
        "bracket_pairs": bracket_pairs,
        "G": G,
        "epsilon": epsilon,
        "B": B,
        "H_vec_func": h_vec_func,
        "M_basis": M_basis,
        "M_octupole_unit": M_octupole_unit,
        "order": order,
        "H_dict": h_dict,
        "quad_size": quadratic_size(n),
        "nonquad_size": size - quadratic_size(n),
        "monomial_basis": build_monomial_basis(variables, idx_to_vec),
        "a_box": a_box.copy(),
    }


def assemble_M(h_vec, M_basis):
    """Assemble the Lie-generator matrix from Hamiltonian coefficients."""
    if not M_basis:
        raise ValueError("M_basis is empty.")
    M = np.zeros(M_basis[0].shape, dtype=float)
    for coeff, basis_matrix in zip(h_vec, M_basis):
        if coeff != 0:
            M += float(coeff) * basis_matrix
    return M


def element_hamiltonian_values(elem):
    """Translate one thick lattice element to Hamiltonian coefficients.

    The returned length is always the physical length.  A zero-length
    multipole is handled separately by :func:`element_transfer` as an
    integrated kick; no artificial thin-element length is introduced.
    """
    physical_length = float(lin.magnet_field(elem, "LENGTH"))

    if physical_length > 0.0:
        curvature = math.radians(
            float(lin.magnet_field(elem, "ANGLE"))
        ) / physical_length
    else:
        curvature = 0.0

    return (
        physical_length,
        curvature,
        float(lin.magnet_field(elem, "K")),
        float(lin.magnet_field(elem, "S")),
        float(lin.magnet_field(elem, "O")),
        0.0,
    )


def element_transfer(
    elem,
    state,
    tol=1e-14,
    check_upper_right=False,
    **_legacy_kwargs,
):
    """Construct the nonlinear polynomial transfer matrix of one element.

    Sign convention:
        M(H) f = {f, H},
        dc/ds = -M(H)c,
        T_thick = exp(-L M(H)).

    For a zero-length octupole, O is already the integrated strength.  The
    integrated transport generator is ML = -O*M_octupole_unit and the kick is
    applied directly as T = I + ML.  No fictitious length enters the map.

    Extra keyword arguments are ignored only for compatibility with older
    runners; they do not affect the physics.
    """
    M_basis = state["M_basis"]
    if not M_basis:
        raise ValueError("M_basis is empty.")

    size = M_basis[0].shape[0]
    identity = np.eye(size)

    element_type = lin.magnet_field(elem, "TYPE")
    physical_length = float(lin.magnet_field(elem, "LENGTH"))
    O = float(lin.magnet_field(elem, "O"))

    if physical_length == 0.0:
        if element_type == "multipole" and O != 0.0:
            # O is already integrated.  Do not divide by, multiply by,
            # or invent a thin-element length.
            ML = -O * state["M_octupole_unit"]
            tmatrix = identity + ML
        else:
            tmatrix = identity
    else:
        L, b1, b2, b3, b4, b5 = element_hamiltonian_values(elem)
        h_vec = np.asarray(
            state["H_vec_func"](b1, b2, b3, b4, b5),
            dtype=float,
        ).reshape(-1)
        M = assemble_M(h_vec, M_basis)
        tmatrix = expm(-L * M)

    q = state["quad_size"]
    Mqq = tmatrix[:q, :q]
    Mqn = tmatrix[:q, q:]
    Mnq = tmatrix[q:, :q]
    Mnn = tmatrix[q:, q:]

    if check_upper_right and not np.all(np.abs(Mqn) < tol):
        raise ValueError(
            f"Upper-right nonlinear block is not zero within tolerance {tol}."
        )

    return tmatrix, Mqq, Mnn, Mqn, Mnq


def nonlinear_transfer(lattice, state, tol=1e-14, check_upper_right=False, cache=True, **_legacy_kwargs):
    """Construct the nonlinear transfer matrix of an ordered lattice.

    When cache=True, repeated magnet families reuse their already-computed
    element transfer matrix.  This is important for a full ring with many
    repeated occurrences of the same magnet objects.
    """
    size = len(state["idx_to_vec"])
    transfer = np.eye(size)
    element_cache = {}

    for elem in lattice:
        key = id(elem) if cache else None
        if cache and key in element_cache:
            tmatrix = element_cache[key]
        else:
            tmatrix = element_transfer(
                elem,
                state,
                tol=tol,
                check_upper_right=check_upper_right,
            )[0]
            if cache:
                element_cache[key] = tmatrix
        transfer = tmatrix @ transfer

    q = state["quad_size"]
    return transfer, transfer[q:, q:], transfer[q:, :q]


def Non_linear_Transfer(lattice, H_vec, M_basis):
    """Deprecated compatibility wrapper.

    New code should use nonlinear_transfer(lattice, state).  This wrapper is
    retained only for older callers that explicitly pass H_vec/M_basis.
    """
    raise RuntimeError(
        "Non_linear_Transfer no longer uses module globals. Use nonlinear_transfer(lattice, state)."
    )



def build_derivative_matrix(state, variable_index):
    """Build the sparse coefficient-space derivative matrix for one variable.

    The invariant vectors stored by this module are coefficients of the scaled
    physical basis

        C_k z**alpha_k.

    Therefore, if alpha_k[variable_index] = r and differentiation maps basis
    index k to j, then

        (D_variable)_[j,k] = r * C_k / C_j.

    With this convention ``D @ coeffs`` is the coefficient vector of the
    physical derivative in the same scaled basis.  The matrix has at most one
    nonzero entry per column, so applying it is very cheap.
    """
    if "C" not in state:
        raise KeyError("state must contain C before derivative matrices are built.")

    idx_to_vec = state["idx_to_vec"]
    vec_to_idx = state["vec_to_idx"]
    C = np.asarray(state["C"], dtype=float)
    size = len(idx_to_vec)

    if not 0 <= int(variable_index) < len(idx_to_vec[0]):
        raise ValueError("variable_index is outside the polynomial exponent vector.")

    rows = []
    cols = []
    values = []

    for k in range(size):
        powers = list(idx_to_vec[k])
        power = powers[variable_index]
        if power == 0:
            continue

        powers[variable_index] -= 1
        j = vec_to_idx.get(tuple(powers))
        if j is None:
            continue

        rows.append(j)
        cols.append(k)
        values.append(float(power) * C[k] / C[j])

    return sps.csr_matrix((values, (rows, cols)), shape=(size, size), dtype=float)

def initialize_nonlinear(data, m, d, hamiltonian, a_box, variables, field_symbols, n=2):
    """Build nonlinear state and attach the linear-lattice normalization."""
    state = load(m, d, hamiltonian, a_box, variables, field_symbols, n=n)
    cs0 = np.asarray(lin.linear_data(data, "CS0"), dtype=float)
    bx0, ax0, gx0, _, _, _ = cs0

    vec_to_idx = state["vec_to_idx"]
    epsilon = state["epsilon"]

    idx_x2 = vec_to_idx[(0, 2, 0, 0, 0)]
    idx_xpx = vec_to_idx[(0, 1, 0, 1, 0)]
    idx_px2 = vec_to_idx[(0, 0, 0, 2, 0)]

    normalization_arg = (
        bx0 * gx0 / (epsilon[idx_px2] * epsilon[idx_x2])
        - ax0**2 / (epsilon[idx_xpx] ** 2)
    )
    if normalization_arg <= 0.0:
        raise ValueError(
            "Invariant normalization is not positive for this lattice/a_box: "
            f"{normalization_arg}."
        )

    C = epsilon * math.sqrt(normalization_arg)
    q = state["quad_size"]

    state = dict(state)
    state["C"] = C
    state["Gqq"] = state["G"][:q, :q]
    state["Gnn"] = state["G"][q:, q:]
    state["linear_cs0"] = cs0.copy()

    # Fast coefficient-space derivatives used by the shape/stability objective.
    # Variable order is [delta, x, y, px, py].
    state["D_x"] = build_derivative_matrix(state, 1)
    state["D_px"] = build_derivative_matrix(state, 3)

    return state


def quadratic_invariants(data, state):
    """Build the normalized horizontal and vertical Courant-Snyder vectors."""
    bx0, ax0, gx0, by0, ay0, gy0 = np.asarray(lin.linear_data(data, "CS0"), dtype=float)
    vec_to_idx = state["vec_to_idx"]
    C = state["C"]
    q = state["quad_size"]

    idx_x2 = vec_to_idx[(0, 2, 0, 0, 0)]
    idx_xpx = vec_to_idx[(0, 1, 0, 1, 0)]
    idx_px2 = vec_to_idx[(0, 0, 0, 2, 0)]
    idx_y2 = vec_to_idx[(0, 0, 2, 0, 0)]
    idx_ypy = vec_to_idx[(0, 0, 1, 0, 1)]
    idx_py2 = vec_to_idx[(0, 0, 0, 0, 2)]

    Sx = np.zeros(q, dtype=float)
    Sx[idx_x2] = gx0 / C[idx_x2]
    Sx[idx_xpx] = 2.0 * ax0 / C[idx_xpx]
    Sx[idx_px2] = bx0 / C[idx_px2]

    Sy = np.zeros(q, dtype=float)
    Sy[idx_y2] = gy0 / C[idx_y2]
    Sy[idx_ypy] = 2.0 * ay0 / C[idx_ypy]
    Sy[idx_py2] = by0 / C[idx_py2]
    return Sx, Sy



def horizontal_shape_objective(Ix, Sx, state, gradient_weight=0.1):
    """Measure how far Ix departs from Sx in value and horizontal gradient.

    This is intended as a fast optimization objective built directly from the
    polynomial coefficient vector, without grids, plotting, SymPy, tracking,
    or searches for critical points.

    Let

        N = Ix - Sx,

    where Sx is padded with zeros outside the quadratic block.  The objective is

        J = sqrt( ||N||_G^2
                  + lambda * [ a_x^2 ||dN/dx||_G^2
                               + a_px^2 ||dN/dpx||_G^2 ] ).

    The factors a_x and a_px make the derivative contribution comparable in the
    normalized physical box.  When ``gradient_weight=0`` this reduces exactly
    to the current ||Ix-Sx||_G objective.

    Returns
    -------
    objective : float
        Combined value + gradient objective.
    value_norm : float
        ||Ix-Sx||_G.
    gradient_norm : float
        sqrt(a_x^2||dN/dx||_G^2 + a_px^2||dN/dpx||_G^2).
    """
    if gradient_weight < 0.0:
        raise ValueError("gradient_weight must be nonnegative.")

    Ix = np.asarray(Ix, dtype=float).reshape(-1)
    Sx = np.asarray(Sx, dtype=float).reshape(-1)
    size = len(state["idx_to_vec"])
    q = state["quad_size"]

    if Ix.size != size:
        raise ValueError(f"Ix must have length {size}, received {Ix.size}.")
    if Sx.size != q:
        raise ValueError(f"Sx must have length {q}, received {Sx.size}.")

    # Ix contains Sx exactly in its quadratic block by construction.  Writing
    # the subtraction explicitly keeps this function valid if that assumption
    # is relaxed later.
    N = Ix.copy()
    N[:q] -= Sx

    G = state["G"]
    Dx = state["D_x"]
    Dpx = state["D_px"]
    a_box = np.asarray(state["a_box"], dtype=float)
    ax = float(a_box[1])
    apx = float(a_box[3])

    value_sq = float(N @ G @ N)

    Nx = Dx @ N
    Npx = Dpx @ N
    gradient_sq = float(
        ax**2 * (Nx @ G @ Nx)
        + apx**2 * (Npx @ G @ Npx)
    )

    # Protect against tiny negative values from floating-point roundoff.
    value_sq = max(value_sq, 0.0)
    gradient_sq = max(gradient_sq, 0.0)
    total_sq = value_sq + float(gradient_weight) * gradient_sq

    return (
        math.sqrt(max(total_sq, 0.0)),
        math.sqrt(value_sq),
        math.sqrt(gradient_sq),
    )

def invariant(tnn, tnq, Sx, Sy, state, tol=1e-14):
    """Solve the weighted least-squares nonlinear invariant problem."""
    Gnn = state["Gnn"]
    B = state["B"]
    G = state["G"]
    nonquad_size = state["nonquad_size"]

    Ux = tnq @ Sx
    Uy = tnq @ Sy
    D = np.eye(nonquad_size, dtype=float) - tnn

    Lg = np.linalg.cholesky(Gnn)
    Aw = Lg.T @ D
    bx_ls = Lg.T @ Ux
    by_ls = Lg.T @ Uy

    hx, *_ = np.linalg.lstsq(Aw, bx_ls, rcond=tol)
    hy, *_ = np.linalg.lstsq(Aw, by_ls, rcond=tol)

    residual_x = Ux - D @ hx
    residual_y = Uy - D @ hy

    nhx = np.sqrt(max(hx @ Gnn @ hx, 0.0))
    nhy = np.sqrt(max(hy @ Gnn @ hy, 0.0))
    err_x = np.sqrt(max(residual_x.T @ Gnn @ residual_x, 0.0))
    err_y = np.sqrt(max(residual_y.T @ Gnn @ residual_y, 0.0))
    norm_Ux_G = np.sqrt(max(Ux.T @ Gnn @ Ux, 0.0))
    norm_Uy_G = np.sqrt(max(Uy.T @ Gnn @ Uy, 0.0))
    rel_err_x = err_x / max(norm_Ux_G, 1e-300)
    rel_err_y = err_y / max(norm_Uy_G, 1e-300)

    Ix = np.concatenate((Sx, hx))
    Iy = np.concatenate((Sy, hy))

    bracket_vec = np.zeros(len(Ix), dtype=float)
    for k in range(len(Ix)):
        if B[k].nnz:
            bracket_vec[k] = Ix @ (B[k] @ Iy)

    bracket_sq = bracket_vec.T @ G @ bracket_vec
    if bracket_sq < 0.0 and abs(bracket_sq) < 1e-25:
        bracket_sq = 0.0
    n_bracketed = np.sqrt(max(bracket_sq, 0.0))

    return (
        nhx, nhy, err_x, err_y, norm_Ux_G, norm_Uy_G,
        rel_err_x, rel_err_y, n_bracketed, Ix, Iy,
    )


def invariant_vectors(lattice, data, state, tol=1e-14, cache=True, **_legacy_kwargs):
    """Run transfer construction + least-squares solve using prepared state."""
    Sx, Sy = quadratic_invariants(data, state)
    transfer, tnn, tnq = nonlinear_transfer(
        lattice,
        state,
        tol=tol,
        cache=cache,
    )
    result = invariant(tnn, tnq, Sx, Sy, state, tol=tol)
    return result[-2], result[-1], result, transfer


def invariant_polynomial(lattice, data, state, tol=1e-14, cache=True, **_legacy_kwargs):
    """Return horizontal/vertical invariant polynomials in physical variables."""
    Ix, Iy, result, transfer = invariant_vectors(
        lattice,
        data,
        state,
        tol=tol,
        cache=cache,
    )
    Ix_pol = vector_to_poly(np.asarray(Ix) * state["C"], state["monomial_basis"])
    Iy_pol = vector_to_poly(np.asarray(Iy) * state["C"], state["monomial_basis"])
    return Ix_pol, Iy_pol, result, transfer


def eval_plane(I, state, Q, P, plane="x", delta0=0.0, frozen_q0=0.0, frozen_p0=0.0):
    """Evaluate an invariant on an x-px or y-py transverse section.

    Parameters
    ----------
    I : array-like
        Invariant coefficient vector.
    state : dict
        Nonlinear state returned by initialize_nonlinear().
    Q, P : array-like
        Coordinates of the active plotting plane.
    plane : {"x", "y"}
        Active plane. For "x", the variables are (x, px) and the frozen plane
        is (y, py). For "y", the variables are (y, py) and the frozen plane
        is (x, px).
    delta0 : float, optional
        Fixed delta value.
    frozen_q0 : float, optional
        Fixed coordinate of the inactive transverse plane (y for plane="x",
        x for plane="y").
    frozen_p0 : float, optional
        Fixed momentum of the inactive transverse plane (py for plane="x",
        px for plane="y").
    """
    plane = plane.lower()
    if plane == "x":
        q, p = 1, 3
        frozen_q, frozen_p = 2, 4
    elif plane == "y":
        q, p = 2, 4
        frozen_q, frozen_p = 1, 3
    else:
        raise ValueError("plane must be 'x' or 'y'.")

    Z = np.zeros_like(Q, dtype=float)
    idx_to_vec = state["idx_to_vec"]
    eps = state["C"]

    for k, c in enumerate(I):
        if c == 0:
            continue

        v = idx_to_vec[k]
        term = float(c) * float(eps[k])

        if v[0] != 0:
            term *= delta0 ** v[0]
        if term == 0:
            continue

        if v[frozen_q] != 0:
            term *= frozen_q0 ** v[frozen_q]
        if term == 0:
            continue

        if v[frozen_p] != 0:
            term *= frozen_p0 ** v[frozen_p]
        if term == 0:
            continue

        if v[q] != 0:
            term *= Q ** v[q]
        if v[p] != 0:
            term *= P ** v[p]
        Z += term

    return Z


def _format_value_for_filename(value):
    """Compact, filename-safe formatting for floating values."""
    s = f"{float(value):.6g}"
    s = s.replace("+", "")
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def plot_invariant_section(
    Ix,
    Iy,
    state,
    plane="both",
    levels=25,
    grid_points=350,
    rmin=0.06,
    rmax=0.95,
    delta0=0.0,
    folder="nonlinear_plots",
    x_max=5e-3,
    px_max=1e-3,
    y_max=2e-3,
    py_max=1e-3,
    frozen_q0=0.0,
    frozen_p0=0.0,
):
    """Save invariant contour plots and return the generated paths/data.

    For plane="x", the plot is in (x, px) with fixed (y, py) =
    (frozen_q0, frozen_p0).
    For plane="y", the plot is in (y, py) with fixed (x, px) =
    (frozen_q0, frozen_p0).
    """
    plane = plane.lower()
    if plane == "both":
        return {
            "x": plot_invariant_section(
                Ix, Iy, state, "x", levels, grid_points, rmin, rmax, delta0,
                folder, x_max, px_max, y_max, py_max, frozen_q0, frozen_p0,
            ),
            "y": plot_invariant_section(
                Ix, Iy, state, "y", levels, grid_points, rmin, rmax, delta0,
                folder, x_max, px_max, y_max, py_max, frozen_q0, frozen_p0,
            ),
        }

    if plane == "x":
        I = Ix
        qmax, pmax = x_max, px_max
        qlab, plab = "x", "px"
        title = f"Ix on y={frozen_q0:g}, py={frozen_p0:g}, delta={delta0:g}"
        frozen_name_q, frozen_name_p = "y", "py"
    elif plane == "y":
        I = Iy
        qmax, pmax = y_max, py_max
        qlab, plab = "y", "py"
        title = f"Iy on x={frozen_q0:g}, px={frozen_p0:g}, delta={delta0:g}"
        frozen_name_q, frozen_name_p = "x", "px"
    else:
        raise ValueError("plane must be 'x', 'y', or 'both'.")

    qvals = np.linspace(-qmax, qmax, grid_points)
    pvals = np.linspace(-pmax, pmax, grid_points)
    Q, P = np.meshgrid(qvals, pvals, indexing="xy")
    Z = eval_plane(I, state, Q, P, plane, delta0, frozen_q0, frozen_p0)

    if isinstance(levels, int):
        level_count = 2 * levels
        theta = np.linspace(0.0, 2.0 * np.pi, 300, endpoint=False)
        radii = np.linspace(rmin, rmax, level_count)
        lev = []
        for r in radii:
            Qr = r * qmax * np.cos(theta)
            Pr = r * pmax * np.sin(theta)
            vals = eval_plane(I, state, Qr, Pr, plane, delta0, frozen_q0, frozen_p0)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                lev.append(float(np.median(vals)))
        lev = np.unique(np.round(np.sort(np.asarray(lev)), 14))
        zmin, zmax = np.nanmin(Z), np.nanmax(Z)
        lev = lev[(lev > zmin) & (lev < zmax)]
        if len(lev) < 2:
            lev = np.linspace(zmin, zmax, level_count + 2)[1:-1]
    else:
        lev = np.asarray(levels, dtype=float)

    os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.contour(Q, P, Z, levels=lev, linewidths=0.45)
    plt.xlabel(qlab)
    plt.ylabel(plab)
    plt.title(title)
    plt.xlim(-qmax, qmax)
    plt.ylim(-pmax, pmax)
    plt.grid(True)
    plt.tight_layout()

    filename = (
        datetime.now().strftime("%Y%m%d_%H%M%S_")
        + f"invariant_section_{plane}_"
        + f"{frozen_name_q}_{_format_value_for_filename(frozen_q0)}_"
        + f"{frozen_name_p}_{_format_value_for_filename(frozen_p0)}_"
        + f"delta_{_format_value_for_filename(delta0)}.png"
    )
    path = os.path.join(folder, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return Q, P, Z, lev, path


def plot_invariant_slices(
    Ix,
    Iy,
    state,
    y_values=(0.0, 0.5e-1, 1e-1),
    x_values=(0.0, 0.5e-1, 1e-1),
    delta_values=(0.0, 0.005, 0.01),
    frozen_momentum=0.0,
    levels=25,
    grid_points=350,
    rmin=0.06,
    rmax=0.95,
    folder="nonlinear_plots",
    x_max=5e-3,
    px_max=1e-3,
    y_max=2e-3,
    py_max=1e-3,
):
    """Create a folder and save a batch of Ix/Iy section plots.

    Generated plots
    ---------------
    - Ix on the x-px plane for each fixed y in y_values, with py fixed to 0.
    - Iy on the y-py plane for each fixed x in x_values, with px fixed to 0.
    - Each of the above for every delta in delta_values.

    Returns
    -------
    dict
        Dictionary with the output folder and saved plot metadata.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = os.path.join(folder, f"invariant_slices_{timestamp}")
    os.makedirs(out_folder, exist_ok=True)

    results = {
        "folder": out_folder,
        "Ix": {},
        "Iy": {},
    }

    for delta0 in delta_values:
        results["Ix"][delta0] = {}
        for y0 in y_values:
            results["Ix"][delta0][y0] = plot_invariant_section(
                Ix,
                Iy,
                state,
                plane="x",
                levels=levels,
                grid_points=grid_points,
                rmin=rmin,
                rmax=rmax,
                delta0=delta0,
                folder=out_folder,
                x_max=x_max,
                px_max=px_max,
                y_max=y_max,
                py_max=py_max,
                frozen_q0=y0,
                frozen_p0=frozen_momentum,
            )

        results["Iy"][delta0] = {}
        for x0 in x_values:
            results["Iy"][delta0][x0] = plot_invariant_section(
                Ix,
                Iy,
                state,
                plane="y",
                levels=levels,
                grid_points=grid_points,
                rmin=rmin,
                rmax=rmax,
                delta0=delta0,
                folder=out_folder,
                x_max=x_max,
                px_max=px_max,
                y_max=y_max,
                py_max=py_max,
                frozen_q0=x0,
                frozen_p0=frozen_momentum,
            )

    return results

def check_nonlinear_state(state):
    """Return basic internal consistency checks for a prepared nonlinear state."""
    idx_to_vec = state["idx_to_vec"]
    vec_to_idx = state["vec_to_idx"]
    inverse_error = sum(vec_to_idx.get(tuple(v), -1) != i for i, v in idx_to_vec.items())
    G = state["G"]
    gram_symmetry = float(np.linalg.norm(G - G.T))
    gram_min_eigenvalue = float(np.linalg.eigvalsh(G).min())
    basis_shape_ok = all(M.shape == G.shape for M in state["M_basis"])
    h_basis_match = len(state["order"]) == len(state["M_basis"])
    return {
        "index_inverse_errors": int(inverse_error),
        "gram_symmetry_error": gram_symmetry,
        "gram_min_eigenvalue": gram_min_eigenvalue,
        "basis_shapes_ok": bool(basis_shape_ok),
        "hamiltonian_basis_count_ok": bool(h_basis_match),
    }


def transfer_checks(transfer, state):
    """Return block-structure checks for a completed nonlinear transfer matrix."""
    q = state["quad_size"]
    upper_right = transfer[:q, q:]
    return {
        "transfer_shape": tuple(transfer.shape),
        "upper_right_norm": float(np.linalg.norm(upper_right)),
        "transfer_finite": bool(np.all(np.isfinite(transfer))),
    }
