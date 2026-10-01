"""Nonlinear polynomial maps and quasi-invariant mathematics.

Reading guide
-------------
This module is easiest to understand in five layers:

1. Polynomial representation
   idx_to_vec maps coefficient index -> [delta,x,y,px,py] exponent vector.
   vec_to_idx is the inverse lookup.

2. Lie algebra
   The Poisson bracket is precomputed in the truncated monomial basis and used
   to build Hamiltonian generator matrices.

3. Element and lattice maps
   element_transfer() constructs one polynomial map; nonlinear_transfer()
   multiplies element maps in physical lattice order.

4. Invariant construction
   FANQO supports two constructions with the same monomial ordering:
     a_box : weighted least-squares continuation of Courant-Snyder;
     eigen : near-fixed eigenvector of the one-turn transfer.

5. Evaluation / plotting / checks
   Derivative matrices, polynomial reconstruction, 2-D sections, and numerical
   consistency checks operate on the common invariant-vector representation.

Important coordinate convention
-------------------------------
Polynomial exponents always use
    [delta, x, y, px, py]

This differs from Accelerator Toolbox tracking order
    [x, px, y, py, delta, ct]

Machine-specific choices remain in the user configuration; this module contains
only reusable mathematics.
"""

import math
import os
from datetime import datetime
from math import comb

import numpy as np
import scipy.sparse as sps
import sympy as sp
from scipy.linalg import expm

from . import linear as lin
from ..plotting import get_pyplot


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
    """Build the scaled a_box polynomial representation.

    Conceptually this function performs four jobs:

    1. enumerate the truncated monomial basis;
    2. build the Gram matrix G and diagonal monomial scale epsilon;
    3. precompute the Poisson-bracket tensor in that scaled basis;
    4. convert the symbolic Hamiltonian into reusable Lie-generator matrices.

    The output is a dictionary because the same basis data must be reused for
    thousands of objective evaluations during an optimization.

    Parameters
    ----------

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

    # A_BOX[2] == 0 is an explicit request for the invariant submanifold
    # y = py = 0.  In that mode FANQO builds a genuinely reduced horizontal
    # polynomial space rather than approximating the slice with a tiny y box.
    #
    # The public exponent convention remains [delta, x, y, px, py], but every
    # retained monomial has y_degree = py_degree = 0.  This keeps downstream
    # labeling/evaluation compatible while making G, the Poisson algebra, and
    # the least-squares problem depend only on (delta, x, px).
    if np.isscalar(a_box):
        a_box = np.full(len(variables), float(a_box))
    else:
        a_box = np.asarray(a_box, dtype=float)

    if len(a_box) != len(variables):
        raise ValueError(
            f"a_box must have {len(variables)} entries, received {len(a_box)}."
        )
    if not np.all(np.isfinite(a_box)):
        raise ValueError("Every a_box entry must be finite.")

    horizontal_slice_only = bool(a_box[2] == 0.0)
    if horizontal_slice_only:
        if a_box[0] <= 0.0 or a_box[1] <= 0.0 or a_box[3] <= 0.0:
            raise ValueError(
                "With A_BOX[2]=0, delta, x, and px half-widths must remain positive."
            )
        if a_box[4] < 0.0:
            raise ValueError("The py half-width cannot be negative.")
        a_box = a_box.copy()
        a_box[2] = 0.0
        a_box[4] = 0.0
    elif np.any(a_box <= 0.0):
        raise ValueError(
            "Every a_box entry must be positive, except A_BOX[2]=0 which "
            "activates the horizontal y=py=0 LS mode."
        )

    # Basis bookkeeping. Every coefficient index k still uses the public
    # [delta,x,y,px,py] exponent vector.
    idx_to_vec = {}
    vec_to_idx = {}

    if horizontal_slice_only:
        active_transverse_dim = 2  # x, px
        layer_size = sum(
            comb(s + active_transverse_dim - 1, active_transverse_dim - 1)
            for s in range(m + 1)
        )
        for delta_degree in range(d + 1):
            offset = delta_degree * layer_size
            for idx in range(layer_size):
                x_degree, px_degree = vectormap(idx, active_transverse_dim)
                v = [delta_degree, x_degree, 0, px_degree, 0]
                idx_to_vec[offset + idx] = v
                vec_to_idx[tuple(v)] = offset + idx
    else:
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

            for plane in range(1 if horizontal_slice_only else n):
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

    # G is the coefficient-space representation of the polynomial inner
    # product on the normalized symmetric box. Odd total powers integrate to
    # zero, which explains the parity test below.
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

    # epsilon rescales each physical monomial using the chosen a_box. This is
    # what makes coefficients from very different physical powers comparable.
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
        "quad_size": quadratic_size(1 if horizontal_slice_only else n),
        "nonquad_size": size - quadratic_size(1 if horizontal_slice_only else n),
        "monomial_basis": build_monomial_basis(variables, idx_to_vec),
        "a_box": a_box.copy(),
        "horizontal_slice_only": horizontal_slice_only,
        "active_variables": (
            ("delta", "x", "px")
            if horizontal_slice_only
            else ("delta", "x", "y", "px", "py")
        ),
        "invariant_construction": "a_box",
        "transport_sign": -1.0,
        "coordinate_scale": np.asarray(a_box, dtype=float).copy(),
    }


def load_eigen(m, d, hamiltonian, a_box, variables, field_symbols, n=2):
    """Build the unscaled eigenvector representation on the same monomial order.

    The exponent dictionaries are intentionally identical to load(). The
    difference is mathematical representation, not basis notation:

      * coefficient scale C = 1;
      * epsilon = 1;
      * G is the physical-monomial Gram matrix on a unit symmetric box;
      * M(H)f = {H,f};
      * thick transport is exp(+L M).

    Keeping the same idx_to_vec / vec_to_idx interface is what lets every
    downstream objective, plot, and report consume Ix without caring how the
    invariant was constructed.

    """
    dim = len(variables)
    a_box = np.asarray(a_box, dtype=float)
    if a_box.shape != (dim,):
        raise ValueError(f"a_box must have {dim} entries, received {len(a_box)}.")
    if not np.all(np.isfinite(a_box)) or np.any(a_box <= 0.0):
        raise ValueError("Every a_box entry must be positive and finite.")

    base = load(
        m, d, hamiltonian, np.ones(dim, dtype=float),
        variables, field_symbols, n=n,
    )

    idx_to_vec = base["idx_to_vec"]
    size = len(idx_to_vec)

    G = np.zeros((size, size), dtype=float)
    for i in range(size):
        fi = idx_to_vec[i]
        for j in range(i, size):
            fj = idx_to_vec[j]
            value = 1.0
            for axis in range(dim):
                power = fi[axis] + fj[axis]
                if power % 2:
                    value = 0.0
                    break
                value *= 1.0 / (power + 1.0)
            G[i, j] = value
            G[j, i] = value

    epsilon = np.ones(size, dtype=float)

    B = [sps.lil_matrix((size, size), dtype=float) for _ in range(size)]
    for (k, i), pairs in base["bracket_pairs"].items():
        fi = idx_to_vec[i]
        for j, plane in pairs:
            fj = idx_to_vec[j]
            q_idx = 1 + plane
            p_idx = 3 + plane
            sympl = fi[q_idx] * fj[p_idx] - fi[p_idx] * fj[q_idx]
            if sympl:
                B[k][i, j] += float(sympl)
    B = [matrix.tocsr() for matrix in B]

    # B[k][i,j] is the coefficient of {e_i,e_j}; selecting the Hamiltonian
    # row therefore builds M(H)f={H,f}. There is intentionally no minus sign.
    M_basis = []
    for i in base["order"]:
        Mi = sps.lil_matrix((size, size), dtype=float)
        for k in range(size):
            row = B[k].getrow(i)
            if row.nnz:
                for j, value in zip(row.indices, row.data):
                    Mi[k, j] = float(value)
        M_basis.append(Mi.toarray())

    zero_fields = np.zeros(len(field_symbols), dtype=float)
    octupole_fields = zero_fields.copy()
    octupole_fields[3] = 1.0
    h_zero = np.asarray(
        base["H_vec_func"](*zero_fields), dtype=float
    ).reshape(-1)
    h_oct = (
        np.asarray(base["H_vec_func"](*octupole_fields), dtype=float).reshape(-1)
        - h_zero
    )
    M_octupole_unit = np.zeros(M_basis[0].shape, dtype=float)
    for coeff, basis_matrix in zip(h_oct, M_basis):
        if coeff != 0.0:
            M_octupole_unit += float(coeff) * basis_matrix

    state = dict(base)
    state.update({
        "G": G,
        "epsilon": epsilon,
        "B": B,
        "M_basis": M_basis,
        "M_octupole_unit": M_octupole_unit,
        "a_box": np.asarray(a_box, dtype=float).copy(),
        "invariant_construction": "eigen",
        "transport_sign": 1.0,
        "coordinate_scale": np.ones(dim, dtype=float),
    })
    return state


def assemble_M(h_vec, M_basis):
    """Assemble M(H) by linearly combining precomputed basis generators.

    h_vec contains the Hamiltonian coefficients for one element. M_basis holds
    the matrix representation associated with each Hamiltonian basis monomial.
    The expensive bracket algebra is therefore done once in load()/load_eigen(),
    not once per magnet and not once per optimizer candidate.
    """
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

    Sign convention is selected by the nonlinear state:
        a_box : M(H)f={f,H}, T=exp(-L M)
        eigen : M(H)f={H,f}, T=exp(+L M)

    The two lines use opposite definitions of the Lie operator, so the opposite
    exponential signs are part of the convention rather than two different
    physical Hamiltonian flows.

    For a zero-length octupole, O is already the integrated strength.  The
    integrated transport generator uses the same state-dependent sign and the kick is
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
            transport_sign = float(state.get("transport_sign", -1.0))
            ML = transport_sign * O * state["M_octupole_unit"]
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
        transport_sign = float(state.get("transport_sign", -1.0))
        tmatrix = expm(transport_sign * L * M)

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
    state["D_y"] = build_derivative_matrix(state, 2)
    state["D_px"] = build_derivative_matrix(state, 3)
    state["D_py"] = build_derivative_matrix(state, 4)

    return state


def initialize_nonlinear_eigen(
    data,
    m,
    d,
    hamiltonian,
    a_box,
    variables,
    field_symbols,
    n=2,
):
    """Initialize the reference eigenvector construction in physical monomials."""
    state = load_eigen(
        m, d, hamiltonian, a_box, variables, field_symbols, n=n
    )
    size = len(state["idx_to_vec"])
    q = state["quad_size"]

    state = dict(state)
    # Requested coordinate C=(1,1,1,1,1). Downstream FANQO stores one scale
    # per monomial, so the compatible coefficient scale is the identity.
    state["C"] = np.ones(size, dtype=float)
    state["coordinate_scale"] = np.ones(len(variables), dtype=float)
    state["Gqq"] = state["G"][:q, :q]
    state["Gnn"] = state["G"][q:, q:]
    state["linear_cs0"] = np.asarray(
        lin.linear_data(data, "CS0"), dtype=float
    ).copy()
    state["D_x"] = build_derivative_matrix(state, 1)
    state["D_y"] = build_derivative_matrix(state, 2)
    state["D_px"] = build_derivative_matrix(state, 3)
    state["D_py"] = build_derivative_matrix(state, 4)
    return state


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
):
    """Build the polynomial representation used by one Ix constructor.

    Methods
    -------
    a_box
        Full 5-D scaled monomial basis and G-weighted least squares.
    a_box_y0
        Exact invariant submanifold y=py=0. The basis itself is reduced to
        (delta, x, px) before G and the Lie matrices are built.
    hybrid
        Same unscaled physical monomial representation used by eigen, but Ix is
        obtained later with ordinary Euclidean least squares (no G weighting).
    eigen
        Full unscaled physical monomial representation and near-fixed
        eigenvector construction.
    graded_ls
        A_BOX-independent block-by-block continuation of Sx. The invariant is
        solved recursively by total polynomial degree in Courant-Snyder
        normalized coordinates using the Fischer coefficient norm.
    """
    method = str(invariant_construction).lower()

    if method == "a_box":
        state = initialize_nonlinear(
            data, m, d, hamiltonian, a_box, variables, field_symbols, n=n
        )
        state["invariant_construction"] = "a_box"
        return state

    if method == "a_box_y0":
        reduced_box = np.asarray(a_box, dtype=float).copy()
        if reduced_box.shape != (5,):
            raise ValueError("a_box_y0 expects A_BOX=[delta,x,y,px,py].")
        reduced_box[2] = 0.0
        reduced_box[4] = 0.0
        state = initialize_nonlinear(
            data, m, d, hamiltonian, reduced_box, variables, field_symbols, n=n
        )
        state["invariant_construction"] = "a_box_y0"
        state["horizontal_slice_only"] = True
        state["active_variables"] = ("delta", "x", "px")
        return state

    if method in {"eigen", "hybrid", "graded_ls"}:
        # These methods all use the same unscaled physical monomial transfer.
        # load_eigen() internally constructs that representation with unit
        # scaling, so the numerical value of A_BOX does not enter their map.
        state = initialize_nonlinear_eigen(
            data, m, d, hamiltonian, a_box, variables, field_symbols, n=n
        )
        state["invariant_construction"] = method
        state["a_box_independent"] = method in {"hybrid", "eigen", "graded_ls"}
        return state

    raise ValueError(
        "method must be one of: 'a_box', 'a_box_y0', 'hybrid', 'eigen', "
        "'graded_ls'."
    )


def quadratic_invariants(data, state):
    """Embed the linear Courant-Snyder invariants in the quadratic basis.

    Horizontal:
        Sx = gamma_x x^2 + 2 alpha_x x px + beta_x px^2

    Vertical is analogous. Division by C converts physical polynomial
    coefficients into FANQO's stored coefficient representation.
    """
    bx0, ax0, gx0, by0, ay0, gy0 = np.asarray(lin.linear_data(data, "CS0"), dtype=float)
    vec_to_idx = state["vec_to_idx"]
    C = state["C"]
    q = state["quad_size"]

    idx_x2 = vec_to_idx[(0, 2, 0, 0, 0)]
    idx_xpx = vec_to_idx[(0, 1, 0, 1, 0)]
    idx_px2 = vec_to_idx[(0, 0, 0, 2, 0)]
    Sx = np.zeros(q, dtype=float)
    Sx[idx_x2] = gx0 / C[idx_x2]
    Sx[idx_xpx] = 2.0 * ax0 / C[idx_xpx]
    Sx[idx_px2] = bx0 / C[idx_px2]

    if state.get("horizontal_slice_only", False):
        return Sx, None

    idx_y2 = vec_to_idx[(0, 0, 2, 0, 0)]
    idx_ypy = vec_to_idx[(0, 0, 1, 0, 1)]
    idx_py2 = vec_to_idx[(0, 0, 0, 0, 2)]
    Sy = np.zeros(q, dtype=float)
    Sy[idx_y2] = gy0 / C[idx_y2]
    Sy[idx_ypy] = 2.0 * ay0 / C[idx_ypy]
    Sy[idx_py2] = by0 / C[idx_py2]
    return Sx, Sy



def _eigenvalue_cutoff(minimum_positive):
    """Reproduce the adaptive eigenvalue window used in the reference implementation."""
    value = abs(float(minimum_positive))
    if 0.0 < value < 1.0e-12:
        return 1.0e-10
    if value < 1.0e-8:
        return 1.0e-8
    if value < 1.0e-4:
        return 1.0e-4
    return 1.0e-1


def eigen_invariant(
    transfer,
    state,
    *,
    plane="x",
    imag_tol=1.0e-12,
):
    """Construct a near-invariant eigenvector of the one-turn transfer.

    The ideal invariant coefficient vector satisfies T c = c, equivalently
    (T-I)c = 0. Numerically we diagonalize T-I, retain near-zero real positive
    eigenvalues, normalize candidate quadratic blocks with

        c_q2*c_p2 - (c_qp/2)^2 = 1,

    and select the candidate with the implemented mixed-quadratic residual
    criterion. The normalization removes the arbitrary eigenvector amplitude.

    """
    transfer = np.asarray(transfer, dtype=float)
    size = len(state["idx_to_vec"])
    if transfer.shape != (size, size):
        raise ValueError("Transfer shape does not match the polynomial basis.")

    plane = str(plane).lower()
    vec_to_idx = state["vec_to_idx"]
    if plane == "x":
        i_q2 = vec_to_idx[(0, 2, 0, 0, 0)]
        i_qp = vec_to_idx[(0, 1, 0, 1, 0)]
        i_p2 = vec_to_idx[(0, 0, 0, 2, 0)]
    elif plane == "y":
        i_q2 = vec_to_idx[(0, 0, 2, 0, 0)]
        i_qp = vec_to_idx[(0, 0, 1, 0, 1)]
        i_p2 = vec_to_idx[(0, 0, 0, 0, 2)]
    else:
        raise ValueError("plane must be 'x' or 'y'.")

    eigenvalues, eigenvectors = np.linalg.eig(
        transfer - np.eye(size, dtype=float)
    )
    real_mask = np.abs(np.imag(eigenvalues)) <= float(imag_tol)
    real_values = np.real(eigenvalues[real_mask])
    positive = real_values[real_values > 0.0]
    if positive.size == 0:
        raise ValueError(
            "Eigen construction found no positive real eigenvalue of T-I."
        )

    cutoff = _eigenvalue_cutoff(np.min(positive))
    candidate_indices = [
        k for k, value in enumerate(eigenvalues)
        if abs(float(np.imag(value))) <= float(imag_tol)
        and 0.0 < float(np.real(value)) < cutoff
    ]
    if not candidate_indices:
        raise ValueError(
            "Eigen construction found no eigenvalue inside its adaptive window."
        )

    candidates = []
    for k in candidate_indices:
        w = np.real(eigenvectors[:, k]).astype(float, copy=True)
        determinant = w[i_q2] * w[i_p2] - (0.5 * w[i_qp]) ** 2
        if not np.isfinite(determinant) or determinant <= 0.0:
            continue

        sign = float(np.sign(w[i_p2]))
        if sign == 0.0:
            sign = 1.0
        w *= sign / math.sqrt(determinant)

        residual = transfer @ w - w
        selection_residual = abs(float(residual[i_qp]))
        if not np.isfinite(selection_residual):
            continue
        candidates.append((
            selection_residual,
            abs(float(np.real(eigenvalues[k]))),
            w,
        ))

    if not candidates:
        raise ValueError(
            "Eigen construction found no normalizable invariant."
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    selection_residual, eigen_residual, invariant_vector = candidates[0]
    return invariant_vector, {
        "reference_selection_residual": float(selection_residual),
        "eigenvalue_residual": float(eigen_residual),
        "reference_candidate_count": int(len(candidates)),
        "eigen_cutoff": float(cutoff),
    }


def least_squares_ix(tnn, tnq, Sx, state, tol=1e-14, *, weighted=True):
    """Construct Ix with a fixed Courant-Snyder quadratic block.

    weighted=True is the original a_box construction:

        min_h ||(I-T_nn)h - T_nq Sx||_{G_nn}.

    weighted=False is the hybrid construction in the unscaled physical
    monomial basis used by eigen:

        min_h ||(I-T_nn)h - T_nq Sx||_2.

    The hybrid intentionally does not use G or a Cholesky factor.
    """
    D = np.eye(state["nonquad_size"], dtype=float) - np.asarray(tnn, dtype=float)
    U = np.asarray(tnq, dtype=float) @ np.asarray(Sx, dtype=float)

    if weighted:
        Gnn = np.asarray(state["Gnn"], dtype=float)
        Lg = np.linalg.cholesky(Gnn)
        h, *_ = np.linalg.lstsq(Lg.T @ D, Lg.T @ U, rcond=tol)
    else:
        h, *_ = np.linalg.lstsq(D, U, rcond=tol)

    residual = U - D @ h
    details = {
        "weighted": bool(weighted),
        "euclidean_residual": float(np.linalg.norm(residual)),
        "euclidean_rhs_norm": float(np.linalg.norm(U)),
    }
    if weighted:
        Gnn = np.asarray(state["Gnn"], dtype=float)
        details["G_residual"] = float(
            np.sqrt(max(residual @ Gnn @ residual, 0.0))
        )
        details["G_rhs_norm"] = float(
            np.sqrt(max(U @ Gnn @ U, 0.0))
        )

    return np.concatenate((np.asarray(Sx, dtype=float), h)), details



def _polynomial_grade(exponents):
    """Total polynomial degree used by the graded invariant construction.

    Delta counts as one polynomial degree, so with DELTA_ORDER=1 the basis is
    naturally bi-graded but can be traversed by total degree.  The transfer
    generated by FANQO's truncated Hamiltonian is lower triangular in this
    ordering: a degree-r output coefficient depends only on coefficients of
    degree <= r.
    """
    return int(sum(int(power) for power in exponents))


def _grade_index_map(state):
    """Return {total_degree: coefficient_indices} for the current basis."""
    groups = {}
    for index, exponents in state["idx_to_vec"].items():
        groups.setdefault(_polynomial_grade(exponents), []).append(int(index))
    return {grade: np.asarray(indices, dtype=int)
            for grade, indices in sorted(groups.items())}


def _cs_grade_coefficient_transform(state, indices):
    """Map physical coefficients to Courant-Snyder normalized coefficients.

    The normalized canonical coordinates are defined by

        x  = sqrt(beta_x) X
        px = (P_X - alpha_x X) / sqrt(beta_x)

        y  = sqrt(beta_y) Y
        py = (P_Y - alpha_y Y) / sqrt(beta_y)

    so the quadratic Courant-Snyder forms become

        Sx = X^2 + P_X^2,
        Sy = Y^2 + P_Y^2.

    For one homogeneous grade this function returns C such that

        c_normalized = C @ c_physical.

    Because the coordinate change is linear, it preserves total polynomial
    degree and therefore acts independently inside each graded block.
    """
    indices = np.asarray(indices, dtype=int)
    exponents = [tuple(int(v) for v in state["idx_to_vec"][int(i)])
                 for i in indices]
    local_index = {powers: i for i, powers in enumerate(exponents)}

    bx, ax, _, by, ay, _ = np.asarray(
        state["linear_cs0"], dtype=float
    )
    if bx <= 0.0 or by <= 0.0:
        raise ValueError(
            "Courant-Snyder beta functions must be positive for graded_ls."
        )

    sbx = math.sqrt(float(bx))
    sby = math.sqrt(float(by))
    transform = np.zeros((len(indices), len(indices)), dtype=float)

    # Columns correspond to physical-basis coefficients.  Each physical
    # monomial is expanded in normalized variables and accumulated by row.
    for column, powers in enumerate(exponents):
        d_power, x_power, y_power, px_power, py_power = powers

        for px_norm_power in range(px_power + 1):
            x_from_px = px_power - px_norm_power
            coeff_x = (
                (sbx ** x_power)
                * math.comb(px_power, px_norm_power)
                * ((-float(ax)) ** x_from_px)
                / (sbx ** px_power)
            )
            X_power = x_power + x_from_px

            for py_norm_power in range(py_power + 1):
                y_from_py = py_power - py_norm_power
                coeff_y = (
                    (sby ** y_power)
                    * math.comb(py_power, py_norm_power)
                    * ((-float(ay)) ** y_from_py)
                    / (sby ** py_power)
                )
                Y_power = y_power + y_from_py

                normalized_powers = (
                    d_power,
                    X_power,
                    Y_power,
                    px_norm_power,
                    py_norm_power,
                )
                row = local_index.get(normalized_powers)
                if row is None:
                    raise ValueError(
                        "The polynomial grade is not closed under the "
                        "Courant-Snyder coordinate transform."
                    )
                transform[row, column] += coeff_x * coeff_y

    return transform


def _fischer_row_weights(state, indices):
    """Diagonal row weights for the Fischer inner product.

    In normalized coordinates the Fischer product satisfies

        <z^alpha, z^beta>_F = alpha! delta_(alpha,beta).

    Therefore weighting a coefficient residual by sqrt(alpha!) converts the
    Fischer norm into an ordinary Euclidean vector norm.  No physical box or
    amplitude scale enters this definition.
    """
    weights = np.ones(len(indices), dtype=float)
    for row, index in enumerate(np.asarray(indices, dtype=int)):
        powers = state["idx_to_vec"][int(index)]
        factorial_product = 1
        for power in powers:
            factorial_product *= math.factorial(int(power))
        weights[row] = math.sqrt(float(factorial_product))
    return weights


def graded_least_squares_ix(transfer, data, state, tol=1e-14):
    """Construct Ix recursively by polynomial degree with Sx fixed exactly.

    Write

        Ix = Sx + I3 + I4 + ...,

    and use the lower-triangular structure of the coefficient transfer matrix.
    If c_r denotes the coefficients of total degree r, the degree-r invariance
    equation is

        (I - T_rr) c_r = sum_(s<r) T_rs c_s.

    The lower-degree coefficients are frozen before the next block is solved.
    Each block is expressed in Courant-Snyder normalized coordinates and its
    residual is minimized in the Fischer norm.  This removes the A_BOX
    dependence while preserving the exact quadratic Courant-Snyder part.

    Delta is counted as one polynomial degree.  With DELTA_ORDER=1 this means,
    for example, delta*x^2 is solved together with ordinary cubic monomials.
    """
    transfer = np.asarray(transfer, dtype=float)
    size = len(state["idx_to_vec"])
    if transfer.shape != (size, size):
        raise ValueError("Transfer shape does not match the polynomial basis.")

    groups = _grade_index_map(state)
    grades = tuple(sorted(groups))
    if not grades or max(grades) < 3:
        raise ValueError("graded_ls requires polynomial terms of degree >= 3.")

    # Fix exactly the physical Courant-Snyder quadratic invariant.  All other
    # coefficients of total degree <= 2 remain zero.
    Sx, _ = quadratic_invariants(data, state)
    fixed_physical = np.zeros(size, dtype=float)
    fixed_physical[: len(Sx)] = np.asarray(Sx, dtype=float)

    transforms = {}
    inverse_transforms = {}
    coefficients_normalized = {}

    for grade in grades:
        indices = groups[grade]
        C = _cs_grade_coefficient_transform(state, indices)
        transforms[grade] = C
        inverse_transforms[grade] = np.linalg.inv(C)

        if grade <= 2:
            coefficients_normalized[grade] = (
                C @ fixed_physical[indices]
            )

    invariant_physical = fixed_physical.copy()
    grade_details = []

    for grade in grades:
        if grade <= 2:
            continue

        rows = groups[grade]
        Cg = transforms[grade]
        Cg_inv = inverse_transforms[grade]

        Tgg_physical = transfer[np.ix_(rows, rows)]
        Tgg_normalized = Cg @ Tgg_physical @ Cg_inv
        D = np.eye(len(rows), dtype=float) - Tgg_normalized

        U = np.zeros(len(rows), dtype=float)
        for lower_grade in grades:
            if lower_grade >= grade:
                break

            lower_coefficients = coefficients_normalized.get(lower_grade)
            if lower_coefficients is None or not np.any(lower_coefficients):
                continue

            columns = groups[lower_grade]
            Tgl_physical = transfer[np.ix_(rows, columns)]
            Tgl_normalized = (
                Cg
                @ Tgl_physical
                @ inverse_transforms[lower_grade]
            )
            U += Tgl_normalized @ lower_coefficients

        weights = _fischer_row_weights(state, rows)
        weighted_D = weights[:, None] * D
        weighted_U = weights * U

        solution, _, rank, singular_values = np.linalg.lstsq(
            weighted_D,
            weighted_U,
            rcond=tol,
        )
        coefficients_normalized[grade] = solution
        invariant_physical[rows] = Cg_inv @ solution

        residual = U - D @ solution
        fischer_residual = float(np.linalg.norm(weights * residual))
        fischer_rhs = float(np.linalg.norm(weighted_U))

        if singular_values.size and singular_values[0] > 0.0:
            min_relative_singular = float(
                singular_values[-1] / singular_values[0]
            )
        else:
            min_relative_singular = float("nan")

        grade_details.append({
            "degree": int(grade),
            "size": int(len(rows)),
            "rank": int(rank),
            "fischer_residual": fischer_residual,
            "fischer_rhs_norm": fischer_rhs,
            "relative_residual": (
                fischer_residual / fischer_rhs
                if fischer_rhs > 0.0 else 0.0
            ),
            "min_relative_singular": min_relative_singular,
        })

    # The unscaled representation used by graded_ls has C=1, so the assembled
    # vector is already in physical monomial coefficients.
    q = len(Sx)
    if not np.array_equal(
        invariant_physical[:q],
        fixed_physical[:q],
    ):
        raise AssertionError("graded_ls changed the fixed Courant-Snyder block.")

    return invariant_physical, {
        "method": "graded_ls",
        "a_box_independent": True,
        "fixed_quadratic": "Sx",
        "grading": "total_degree(delta,x,y,px,py)",
        "coordinate_system": "Courant-Snyder normalized",
        "metric": "Fischer",
        "solved_grades": tuple(
            int(grade) for grade in grades if grade >= 3
        ),
        "grade_details": tuple(grade_details),
    }


def construct_ix(lattice, data, state, tol=1e-14, cache=True):
    """Construct the horizontal invariant with the method stored in state."""
    transfer, tnn, tnq = nonlinear_transfer(
        lattice, state, tol=tol, cache=cache
    )
    method = str(state.get("invariant_construction", "a_box")).lower()

    if method == "eigen":
        Ix, details = eigen_invariant(transfer, state, plane="x")
        details = {"method": "eigen", **details}
        return Ix, details, transfer

    if method == "graded_ls":
        Ix, details = graded_least_squares_ix(
            transfer, data, state, tol=tol
        )
        return Ix, details, transfer

    Sx, _ = quadratic_invariants(data, state)
    if method in {"a_box", "a_box_y0"}:
        Ix, details = least_squares_ix(
            tnn, tnq, Sx, state, tol=tol, weighted=True
        )
    elif method == "hybrid":
        Ix, details = least_squares_ix(
            tnn, tnq, Sx, state, tol=tol, weighted=False
        )
    else:
        raise ValueError(f"Unknown invariant construction: {method!r}")

    details.update({
        "method": method,
        "horizontal_slice_only": bool(
            state.get("horizontal_slice_only", False)
        ),
    })
    return Ix, details, transfer


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
    save=True,
    show=False,
):
    """Save invariant contour plots and return the generated paths/data.

    For plane="x", the plot is in (x, px) with fixed (y, py) =
    (frozen_q0, frozen_p0).
    For plane="y", the plot is in (y, py) with fixed (x, px) =
    (frozen_q0, frozen_p0).
    """
    plane = plane.lower()
    if state.get("horizontal_slice_only", False):
        if plane in {"y", "both"}:
            raise ValueError(
                "A_BOX[2]=0 activates horizontal-only LS mode; Iy/y-plane plots "
                "are not defined."
            )
        if abs(float(frozen_q0)) > 0.0 or abs(float(frozen_p0)) > 0.0:
            raise ValueError(
                "Horizontal-only LS mode is defined on y=py=0; use "
                "frozen_q0=0 and frozen_p0=0."
            )

    if plane == "both":
        return {
            "x": plot_invariant_section(
                Ix, Iy, state, "x", levels, grid_points, rmin, rmax, delta0,
                folder, x_max, px_max, y_max, py_max, frozen_q0, frozen_p0, save, show,
            ),
            "y": plot_invariant_section(
                Ix, Iy, state, "y", levels, grid_points, rmin, rmax, delta0,
                folder, x_max, px_max, y_max, py_max, frozen_q0, frozen_p0, save, show,
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

    path = None
    if save:
        os.makedirs(folder, exist_ok=True)
        filename = (
            datetime.now().strftime("%Y%m%d_%H%M%S_")
            + f"invariant_section_{plane}_"
            + f"{frozen_name_q}_{_format_value_for_filename(frozen_q0)}_"
            + f"{frozen_name_p}_{_format_value_for_filename(frozen_p0)}_"
            + f"delta_{_format_value_for_filename(delta0)}.png"
        )
        path = os.path.join(folder, filename)

    if save or show:
        plt = get_pyplot(show)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contour(Q, P, Z, levels=lev, linewidths=0.45)
        ax.set_xlabel(qlab)
        ax.set_ylabel(plab)
        ax.set_title(title)
        ax.set_xlim(-qmax, qmax)
        ax.set_ylim(-pmax, pmax)
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
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
    save=True,
    show=False,
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
    if save:
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
                save=save,
                show=show,
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
                save=save,
                show=show,
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
