"""Reference eigenvector selection in physical monomial coefficients.

Reuse main's transport by exact diagonal similarity; keep its map construction
and storage unchanged. Eigenvalues here belong to T-I, hence target zero.
"""
import math
import numpy as np


def decompose(transfer, state):
    """Diagonalize the physical coefficient operator once for both planes."""
    scale = np.asarray(state['C'], float)
    transfer = np.asarray(transfer, float)
    size = len(scale)
    if transfer.shape != (size, size) or not np.isfinite(transfer).all():
        raise ValueError('Transfer must be finite and match the polynomial basis.')
    if not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError('Polynomial coefficient scales must be finite and nonzero.')
    physical = scale[:, None]*transfer/scale[None, :]
    return np.linalg.eig(physical-np.eye(size))


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


def solve(
    transfer,
    state,
    *,
    plane="x",
    imag_tol=1.0e-12,
    eigensystem=None,
):
    """Select a near-unit eigenvector by distance of its quadratic part to Sx.

    For plane='y' the target is Sy. Each call uses ONLY its requested seed.
    Candidate vectors are normalized to positive requested-plane determinant
    one. Rank by Euclidean distance of ALL TEN physical delta=0 quadratic
    coefficients to that seed (other-plane and cross-plane targets are zero).
    No mixing of eigenvectors or fitting to the other CS seed is performed.
    Exact unit eigenvalues and small deviations of either sign are included.
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

    eigenvalues, eigenvectors = decompose(transfer, state) if eigensystem is None else eigensystem
    real_mask = np.abs(np.imag(eigenvalues)) <= float(imag_tol)
    real_values = np.real(eigenvalues[real_mask])
    nonzero = np.abs(real_values[np.abs(real_values) > 0.0])
    cutoff = _eigenvalue_cutoff(np.min(nonzero)) if nonzero.size else 1e-10
    candidate_indices = [
        k for k, value in enumerate(eigenvalues)
        if abs(float(np.imag(value))) <= float(imag_tol)
        and abs(float(np.real(value))) <= cutoff
    ]
    if not candidate_indices:
        raise ValueError(
            "Eigen construction found no eigenvalue inside its adaptive window."
        )

    quad = np.array([i for i,a in state['idx_to_vec'].items()
                     if a[0] == 0 and sum(a[1:]) == 2], dtype=int)
    target = np.zeros(size)
    offset = 0 if plane == 'x' else 3
    beta, alpha, gamma = np.asarray(state['linear_cs0'])[offset:offset+3]
    target[[i_q2, i_qp, i_p2]] = [gamma, 2*alpha, beta]
    target_norm = np.linalg.norm(target[quad])
    candidates = []
    for k in candidate_indices:
        vector = eigenvectors[:, k]
        pivot = int(np.argmax(np.abs(vector)))
        vector = vector * np.exp(-1j*np.angle(vector[pivot]))
        if np.linalg.norm(vector.imag) > 1e-8 * max(np.linalg.norm(vector.real), 1e-300):
            continue
        w = vector.real.astype(float, copy=True)
        physical = w.copy()
        determinant = physical[i_q2] * physical[i_p2] - (0.5 * physical[i_qp]) ** 2
        if not np.isfinite(determinant) or determinant <= 0.0:
            continue

        sign = float(np.sign(physical[i_p2]))
        if sign == 0.0:
            sign = 1.0
        w *= sign / math.sqrt(determinant)

        selection_residual = np.linalg.norm((w-target)[quad]) / target_norm
        if not np.isfinite(selection_residual):
            continue
        candidates.append((
            selection_residual,
            abs(float(np.real(eigenvalues[k]))),
            w,
            float(np.real(eigenvalues[k])),
        ))

    if not candidates:
        raise ValueError(
            "Eigen construction found no normalizable invariant."
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    selection_residual, eigen_residual, invariant_vector, shift = candidates[0]
    # Convert the selected physical polynomial back to main's stored basis.
    invariant_vector = invariant_vector / state['C']
    return invariant_vector, {
        'method': 'eigen',
        'normalization': 'positive requested-plane quadratic determinant one',
        "selection": "closest_full_quadratic_to_requested_CS_seed",
        "target_seed": 'Sx' if plane == 'x' else 'Sy',
        "quadratic_relative_distance": float(selection_residual),
        "selection_metric": "Euclidean physical monomial coefficients, all delta=0 degree-2 terms",
        "eigenvalue_residual": float(eigen_residual),
        "eigenvalue": 1.+shift,
        "reference_candidate_count": int(len(candidates)),
        "eigen_cutoff": float(cutoff),
    }
