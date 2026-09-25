"""Optimization engine connecting lattice edits, invariants, and objectives.

The central object in this module is the context dictionary. It stores the
ordered lattice, current linear optics, scalar parameters, nonlinear polynomial
state, reusable element-map cache, and settings needed to update the experiment.

One objective evaluation follows this chain:

    candidate parameters
      -> apply_candidate()
      -> dependency-aware linear/chromatic update
      -> nonlinear_transfer()
      -> construct requested invariant(s)
      -> call Fobj(data)
      -> return a finite score or INVALID_PENALTY

CMA-ES explores globally for the configured wall-clock budget; Powell may then
refine the best point locally. Invalid candidates are rolled back so they never
corrupt the active context.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import scipy.sparse as sps
from scipy.optimize import minimize

from . import linear as lin
from . import nonlinear as nl


# =============================================================================
# 1. SETUP / UPDATE
# =============================================================================


def initial_vector(vary, parameters):
    return np.asarray([parameters[name] for name in vary], dtype=float)


def create_context(
    parameters,
    *,
    ring_names,
    magnet_builder,
    energy_parameter,
    correction_parameter_map,
    correct_chromatic,
    family1,
    family2,
    target_chrom_x,
    target_chrom_y,
    repetitions,
    step,
    linear_variables,
    chromatic_variables,
    parameter_map,
    m,
    d,
    hamiltonian,
    a_box,
    variables,
    field_symbols,
    n_planes=2,
    invariant_construction="a_box",
):
    # Build a self-consistent linear lattice first. When chromatic
    # correction is enabled, corrected family strengths are written into p.
    magnets, lattice, data, correction, p = lin.prepare_lattice(
        parameters=parameters,
        ring_names=ring_names,
        magnet_builder=magnet_builder,
        energy_parameter=energy_parameter,
        correction_parameter_map=correction_parameter_map,
        correct_chromatic=correct_chromatic,
        family1=family1,
        family2=family2,
        target_chrom_x=target_chrom_x,
        target_chrom_y=target_chrom_y,
        repetitions=repetitions,
        step=step,
    )

    state = nl.initialize_nonlinear_for_method(
        data,
        m=m,
        d=d,
        hamiltonian=hamiltonian,
        a_box=a_box,
        variables=variables,
        field_symbols=field_symbols,
        n=n_planes,
        invariant_construction=invariant_construction,
    )

    # The context joins linear and nonlinear state. It is mutated in place
    # during optimization, so failed candidates must restore a snapshot.
    return {
        "magnets": magnets,
        "lattice": lattice,
        "data": data,
        "correction": correction,
        "parameters": dict(p),
        "state": state,
        "map_cache": {},
        "settings": {
            "magnet_builder": magnet_builder,
            "energy_parameter": energy_parameter,
            "correction_parameter_map": correction_parameter_map,
            "correct_chromatic": correct_chromatic,
            "family1": family1,
            "family2": family2,
            "target_chrom_x": target_chrom_x,
            "target_chrom_y": target_chrom_y,
            "repetitions": repetitions,
            "step": step,
            "linear_variables": set(linear_variables),
            "chromatic_variables": set(chromatic_variables),
            "parameter_map": parameter_map,
            # Keep nonlinear model settings so a_box can be rebuilt without
            # reconstructing the physical lattice.
            "m": int(m),
            "d": int(d),
            "hamiltonian": hamiltonian,
            "a_box": np.asarray(a_box, dtype=float).copy(),
            "variables": tuple(variables),
            "field_symbols": tuple(field_symbols),
            "n_planes": int(n_planes),
            "invariant_construction": str(invariant_construction).lower(),
        },
    }


def _build_nonlinear_state(context, a_box):
    settings = context["settings"]
    return nl.initialize_nonlinear_for_method(
        context["data"],
        m=settings["m"],
        d=settings["d"],
        hamiltonian=settings["hamiltonian"],
        a_box=np.asarray(a_box, dtype=float),
        variables=settings["variables"],
        field_symbols=settings["field_symbols"],
        n=settings["n_planes"],
        invariant_construction=settings["invariant_construction"],
    )


def context_with_a_box(context, a_box):
    """Return an analysis context with the same lattice and a fresh a_box."""
    if context["settings"].get("invariant_construction", "a_box") != "a_box":
        raise ValueError(
            "a_box recalibration is only defined for INVARIANT_CONSTRUCTION='a_box'."
        )
    a_box = np.asarray(a_box, dtype=float)
    expected = len(context["settings"]["variables"])
    if a_box.shape != (expected,):
        raise ValueError(f"a_box must have shape ({expected},).")
    if not np.all(np.isfinite(a_box)) or np.any(a_box <= 0.0):
        raise ValueError("Every a_box half-width must be positive and finite.")

    settings = dict(context["settings"])
    settings["a_box"] = a_box.copy()
    temporary = {
        "magnets": context["magnets"],
        "lattice": context["lattice"],
        "data": context["data"],
        "correction": context["correction"],
        "parameters": dict(context["parameters"]),
        "state": None,
        "map_cache": {},
        "settings": settings,
    }
    temporary["state"] = _build_nonlinear_state(temporary, a_box)
    return temporary


def set_context_a_box(context, a_box):
    """Make a new a_box the active nonlinear normalization."""
    rebuilt = context_with_a_box(context, a_box)
    context["state"] = rebuilt["state"]
    context["settings"]["a_box"] = np.asarray(a_box, dtype=float).copy()
    context["map_cache"].clear()
    return context


def _epsilon_for_a_box(state, a_box):
    """Recompute diagonal monomial scaling for the same truncated basis."""
    a_box = np.asarray(a_box, dtype=float)
    dim = len(state["idx_to_vec"][0])
    if a_box.shape != (dim,):
        raise ValueError(f"a_box must have shape ({dim},).")
    if not np.all(np.isfinite(a_box)) or np.any(a_box <= 0.0):
        raise ValueError("Every a_box half-width must be positive and finite.")

    epsilon = np.zeros(len(state["idx_to_vec"]), dtype=float)
    for i, powers in state["idx_to_vec"].items():
        value = 1.0
        for axis, power in enumerate(powers):
            value *= np.sqrt((2 * power + 1) / (a_box[axis] ** (2 * power)))
        epsilon[i] = value
    return epsilon


def rescaled_state_for_a_box(reference_state, data, a_box):
    """Build the a_box-dependent normalization without rebuilding the Lie basis."""
    state = dict(reference_state)
    epsilon = _epsilon_for_a_box(reference_state, a_box)
    cs0 = np.asarray(lin.linear_data(data, "CS0"), dtype=float)
    bx0, ax0, gx0, _, _, _ = cs0
    vec_to_idx = reference_state["vec_to_idx"]

    ix2 = vec_to_idx[(0, 2, 0, 0, 0)]
    ixpx = vec_to_idx[(0, 1, 0, 1, 0)]
    ipx2 = vec_to_idx[(0, 0, 0, 2, 0)]
    arg = (
        bx0 * gx0 / (epsilon[ipx2] * epsilon[ix2])
        - ax0**2 / epsilon[ixpx]**2
    )
    if arg <= 0.0:
        raise ValueError("Nonlinear normalization is not positive for this a_box.")

    state["epsilon"] = epsilon
    state["C"] = epsilon * math.sqrt(arg)
    state["a_box"] = np.asarray(a_box, dtype=float).copy()
    state["coordinate_scale"] = np.asarray(a_box, dtype=float).copy()
    state["linear_cs0"] = cs0.copy()
    state["D_x"] = nl.build_derivative_matrix(state, 1)
    state["D_y"] = nl.build_derivative_matrix(state, 2)
    state["D_px"] = nl.build_derivative_matrix(state, 3)
    state["D_py"] = nl.build_derivative_matrix(state, 4)
    return state


def rescale_transfer_for_a_box(transfer, reference_state, candidate_state):
    """Change transfer representation under a diagonal monomial rescaling.

    The truncated physical monomial space is unchanged. If D_old and D_new are
    the diagonal epsilon scalings, then

        T_new = D_new^{-1} D_old T_old D_old^{-1} D_new.

    This avoids rebuilding every element Lie map during the short a_box search.
    """
    transfer = np.asarray(transfer, dtype=float)
    old_eps = np.asarray(reference_state["epsilon"], dtype=float)
    new_eps = np.asarray(candidate_state["epsilon"], dtype=float)
    if transfer.shape != (len(old_eps), len(old_eps)):
        raise ValueError("Transfer shape does not match nonlinear basis size.")

    ratio = old_eps / new_eps
    return ratio[:, None] * transfer / ratio[None, :]


def a_box_objective_data(
    context,
    reference_transfer,
    a_box,
    trajectories,
    tol,
):
    """Construct Ix for one a_box from a fixed reference transfer."""
    candidate_state = rescaled_state_for_a_box(
        context["state"],
        context["data"],
        a_box,
    )
    transfer = rescale_transfer_for_a_box(
        reference_transfer,
        context["state"],
        candidate_state,
    )
    q = candidate_state["quad_size"]
    tnn = transfer[q:, q:]
    tnq = transfer[q:, :q]
    Sx, _ = nl.quadratic_invariants(context["data"], candidate_state)
    Ix = _solve_invariant(Sx, tnn, tnq, candidate_state, tol)
    return {
        "Ix": Ix,
        "Sx": Sx,
        "state": candidate_state,
        "context": context,
        "transfer": transfer,
        "tnn": tnn,
        "tnq": tnq,
        "trajectories": trajectories,
    }


def _refresh_nonlinear_normalization(state, data):
    """Update only the part of the nonlinear state that depends on CS0."""
    cs0 = np.asarray(lin.linear_data(data, "CS0"), dtype=float)
    if state.get("invariant_construction", "a_box") == "eigen":
        state["linear_cs0"] = cs0.copy()
        # C is the identity in eigen mode, so coefficient derivatives do not
        # change when the linear Twiss parameters change.
        return
    bx0, ax0, gx0, _, _, _ = cs0

    vec_to_idx = state["vec_to_idx"]
    epsilon = np.asarray(state["epsilon"], dtype=float)

    ix2 = vec_to_idx[(0, 2, 0, 0, 0)]
    ixpx = vec_to_idx[(0, 1, 0, 1, 0)]
    ipx2 = vec_to_idx[(0, 0, 0, 2, 0)]

    arg = (
        bx0 * gx0 / (epsilon[ipx2] * epsilon[ix2])
        - ax0**2 / epsilon[ixpx]**2
    )
    if arg <= 0.0:
        raise ValueError("Nonlinear normalization is not positive.")

    state["C"] = epsilon * math.sqrt(arg)
    state["linear_cs0"] = cs0.copy()
    state["D_x"] = nl.build_derivative_matrix(state, 1)
    state["D_y"] = nl.build_derivative_matrix(state, 2)
    state["D_px"] = nl.build_derivative_matrix(state, 3)
    state["D_py"] = nl.build_derivative_matrix(state, 4)


def apply_candidate(context, v, vary):
    """Apply one optimizer vector while preserving a consistent experiment.

    The scalar parameter dictionary is edited first. linear.update_linear()
    repeats only the linear/chromatic work affected by those names. If the
    underlying linear optics changed, the nonlinear normalization/derivative
    operators are refreshed before the next invariant is constructed.

    A snapshot is restored if any step fails.
    """
    v = np.asarray(v, dtype=float)
    if len(v) != len(vary):
        raise ValueError(f"Expected {len(vary)} variables, received {len(v)}.")
    if not np.all(np.isfinite(v)):
        raise ValueError("Candidate values must be finite.")

    p = dict(context["parameters"])
    edited = []
    for name, value in zip(vary, v):
        value = float(value)
        if value != float(p[name]):
            edited.append(name)
        p[name] = value

    if not edited:
        return

    settings = context["settings"]
    linear_changed = bool(set(edited) & settings["linear_variables"])
    snapshot = _snapshot_mutable_context(context)

    try:
        lattice, data, correction, p = lin.update_linear(
            lattice=context["lattice"],
            data=context["data"],
            parameters=p,
            edited_variables=edited,
            correct_chromatic=settings["correct_chromatic"],
            family1=settings["family1"],
            family2=settings["family2"],
            target_chrom_x=settings["target_chrom_x"],
            target_chrom_y=settings["target_chrom_y"],
            repetitions=settings["repetitions"],
            step=settings["step"],
            linear_variables=settings["linear_variables"],
            chromatic_variables=settings["chromatic_variables"],
            parameter_map=settings["parameter_map"],
            magnet_builder=settings["magnet_builder"],
            correction_parameter_map=settings["correction_parameter_map"],
            energy_parameter=settings["energy_parameter"],
        )

        context["lattice"] = lattice
        context["data"] = data
        context["parameters"] = dict(p)
        if correction is not None:
            context["correction"] = correction

        if linear_changed:
            _refresh_nonlinear_normalization(context["state"], data)
    except Exception:
        _restore_mutable_context(context, snapshot)
        raise


# =============================================================================
# 2. NONLINEAR TRANSFER / OBJECTIVE
# =============================================================================


def _magnet_signature(elem):
    return (
        lin.magnet_field(elem, "TYPE"),
        float(lin.magnet_field(elem, "LENGTH")),
        float(lin.magnet_field(elem, "ANGLE")),
        float(lin.magnet_field(elem, "K")),
        float(lin.magnet_field(elem, "S")),
        float(lin.magnet_field(elem, "O")),
    )


def nonlinear_transfer(context, tol):
    """Build the ordered polynomial transfer while reusing unchanged maps.

    The cache key is the complete physical element signature
    (TYPE, LENGTH, ANGLE, K, S, O). Any changed strength or geometry therefore
    invalidates only the maps that actually changed.
    """
    state = context["state"]
    transfer = np.eye(len(state["idx_to_vec"]), dtype=float)
    cache = context["map_cache"]
    active_cache = {}

    for elem in context["lattice"]:
        key = _magnet_signature(elem)
        cached = active_cache.get(key)
        if cached is None:
            cached = cache.get(key)
            if cached is None:
                tmatrix = nl.element_transfer(elem, state, tol=tol)[0]
                cached = sps.csr_matrix(tmatrix)
            else:
                tmatrix = cached.toarray()
            active_cache[key] = cached
        else:
            tmatrix = cached.toarray()
        transfer = tmatrix @ transfer

    cache.clear()
    cache.update(active_cache)
    q = state["quad_size"]
    return transfer, transfer[q:, q:], transfer[q:, :q]


def _solve_invariant(S, tnn, tnq, state, tol):
    """Solve one weighted least-squares quasi-invariant plane.

    For c=[S;h], periodicity T c = c gives

        (I - T_nn) h = T_nq S.

    Multiplying by a Cholesky factor of G_nn turns the weighted polynomial norm
    into an ordinary Euclidean least-squares problem.
    """
    Gnn = state["Gnn"]
    D = np.eye(state["nonquad_size"], dtype=float) - tnn
    U = tnq @ S

    Lg = np.linalg.cholesky(Gnn)
    h, *_ = np.linalg.lstsq(Lg.T @ D, Lg.T @ U, rcond=tol)
    return np.concatenate((S, h))


def objective_requirements(Fobj):
    requirements = getattr(Fobj, "requires", None)
    if requirements is None:
        raise ValueError(
            f"Objective {getattr(Fobj, '__name__', Fobj)!r} must declare a .requires set."
        )
    return set(requirements)


def prepare_objective_data(
    context,
    Fobj,
    tol,
    *,
    compute_ix=True,
    compute_iy=False,
    extra_data=None,
):
    """Compute only the expensive quantities declared by Fobj.requires.

    Examples:
      {Ix}          -> build the transfer and requested Ix;
      {Ix,Sx}       -> also provide the Courant-Snyder reference;
      {Ix,transfer} -> keep the full T for transport-defect objectives.

    This dependency contract keeps objective functions independent of the
    invariant constructor. The objective asks for Ix; this function decides
    whether Ix comes from least squares or from the eigen construction.
    """
    requirements = objective_requirements(Fobj)
    extra_data = {} if extra_data is None else dict(extra_data)

    supported = {
        "Ix", "Iy", "Sx", "Sy", "transfer", "tnn", "tnq",
        "state", "context",
    } | set(extra_data)
    unknown = requirements - supported
    if unknown:
        raise ValueError(
            "Unsupported objective requirement(s): " + ", ".join(sorted(unknown))
        )

    if "Ix" in requirements and not compute_ix:
        raise ValueError("The selected objective requires Ix but COMPUTE_IX is False.")
    if "Iy" in requirements and not compute_iy:
        raise ValueError("The selected objective requires Iy but COMPUTE_IY is False.")

    state = context["state"]
    result = {"state": state, "context": context, **extra_data}

    method = state.get("invariant_construction", "a_box")
    need_quadratic = bool(requirements & {"Sx", "Sy"})
    if method == "a_box":
        need_quadratic = need_quadratic or bool(requirements & {"Ix", "Iy"})
    Sx = Sy = None
    if need_quadratic:
        Sx, Sy = nl.quadratic_invariants(context["data"], state)
        if "Sx" in requirements or (method == "a_box" and "Ix" in requirements):
            result["Sx"] = Sx
        if "Sy" in requirements or (method == "a_box" and "Iy" in requirements):
            result["Sy"] = Sy

    need_transfer = bool(requirements & {"Ix", "Iy", "transfer", "tnn", "tnq"})
    if need_transfer:
        transfer, tnn, tnq = nonlinear_transfer(context, tol)
        if "transfer" in requirements:
            result["transfer"] = transfer
        if "tnn" in requirements:
            result["tnn"] = tnn
        if "tnq" in requirements:
            result["tnq"] = tnq
        method = state.get("invariant_construction", "a_box")
        if "Ix" in requirements:
            if method == "eigen":
                result["Ix"], result["Ix_construction_details"] = (
                    nl.eigen_invariant(transfer, state, plane="x")
                )
            else:
                result["Ix"] = _solve_invariant(Sx, tnn, tnq, state, tol)
        if "Iy" in requirements:
            if method == "eigen":
                result["Iy"], result["Iy_construction_details"] = (
                    nl.eigen_invariant(transfer, state, plane="y")
                )
            else:
                result["Iy"] = _solve_invariant(Sy, tnn, tnq, state, tol)

    return result


def compute_requested_invariants(context, tol, *, compute_ix=True, compute_iy=False):
    """Compute selected invariant planes without evaluating an objective."""
    class _InvariantRequest:
        requires = set()

    request = _InvariantRequest()
    if compute_ix:
        request.requires.add("Ix")
    if compute_iy:
        request.requires.add("Iy")
    if not request.requires:
        return {}

    data = prepare_objective_data(
        context,
        request,
        tol,
        compute_ix=compute_ix,
        compute_iy=compute_iy,
    )
    return {key: data[key] for key in ("Ix", "Iy") if key in data}


def _call_objective(Fobj, data, objective_kwargs=None):
    kwargs = {} if objective_kwargs is None else dict(objective_kwargs)
    output = Fobj(data, **kwargs)
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
        value, diagnostics = output
    else:
        value, diagnostics = output, {}
    return float(value), dict(diagnostics)

def _copy_value(value):
    try:
        return value.copy()
    except AttributeError:
        return value


def _snapshot_mutable_context(context):
    magnets = []
    for elem in lin.unique_magnets(context["lattice"]):
        magnets.append((elem, [_copy_value(value) for value in elem]))

    state = context["state"]
    state_values = {
        key: _copy_value(state[key])
        for key in ("C", "linear_cs0", "D_x", "D_y", "D_px", "D_py")
        if key in state
    }
    return {
        "magnets": magnets,
        "data": [_copy_value(value) for value in context["data"]],
        "parameters": dict(context["parameters"]),
        "correction": _copy_value(context["correction"]),
        "state": state_values,
        "map_cache": dict(context["map_cache"]),
    }


def _restore_mutable_context(context, snapshot):
    for elem, saved in snapshot["magnets"]:
        elem[:] = [_copy_value(value) for value in saved]
    context["data"] = [_copy_value(value) for value in snapshot["data"]]
    context["parameters"] = dict(snapshot["parameters"])
    context["correction"] = _copy_value(snapshot["correction"])
    for key, value in snapshot["state"].items():
        context["state"][key] = _copy_value(value)
    context["map_cache"].clear()
    context["map_cache"].update(snapshot["map_cache"])


def evaluate_candidate(
    context,
    v,
    vary,
    Fobj,
    *,
    compute_ix,
    compute_iy,
    objective_kwargs,
    tol,
    invalid_penalty,
):
    """Evaluate one optimizer candidate without corrupting valid state.

    The context is snapshotted before mutation. A finite candidate returns its
    objective value. Numerical or physical failures restore the snapshot and
    return the configured penalty so the search can continue.
    """
    snapshot = _snapshot_mutable_context(context)
    try:
        apply_candidate(context, v, vary)
        data = prepare_objective_data(
            context,
            Fobj,
            tol,
            compute_ix=compute_ix,
            compute_iy=compute_iy,
        )
        objective, _ = _call_objective(Fobj, data, objective_kwargs)
        if not np.isfinite(objective):
            _restore_mutable_context(context, snapshot)
            return float(invalid_penalty)
        return float(objective)
    except (ValueError, KeyError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
        _restore_mutable_context(context, snapshot)
        return float(invalid_penalty)


def full_diagnostics(
    context,
    Fobj,
    tol,
    *,
    compute_ix=True,
    compute_iy=False,
    objective_kwargs=None,
    extra_data=None,
):
    """Evaluate an objective and retain only configured invariant planes."""
    data = prepare_objective_data(
        context,
        Fobj,
        tol,
        compute_ix=compute_ix,
        compute_iy=compute_iy,
        extra_data=extra_data,
    )
    objective, diagnostics = _call_objective(Fobj, data, objective_kwargs)
    details = {
        "objective": float(objective),
        "objective_name": getattr(Fobj, "__name__", Fobj.__class__.__name__),
        **diagnostics,
    }
    for key in (
        "Ix", "Iy", "Sx", "Sy", "transfer",
        "Ix_construction_details", "Iy_construction_details",
    ):
        if key in data:
            details[key] = data[key]

    # Keep the configured public invariant state available at start/end without
    # forcing those extra solves inside every candidate evaluation.
    missing_ix = bool(compute_ix and "Ix" not in details)
    missing_iy = bool(compute_iy and "Iy" not in details)
    if missing_ix or missing_iy:
        configured = compute_requested_invariants(
            context,
            tol,
            compute_ix=missing_ix,
            compute_iy=missing_iy,
        )
        details.update(configured)
    return details


# =============================================================================
# 3. PLOTS / SNAPSHOTS / SAVE
# =============================================================================


def plot_slices(details, state, folder, settings):
    """Plot whichever invariant planes are present in details."""
    folder = Path(folder)
    save = bool(settings.get("save", True))
    show = bool(settings.get("show", False))
    if save:
        folder.mkdir(parents=True, exist_ok=True)

    results = {"folder": str(folder), "Ix": {}, "Iy": {}}
    Ix = details.get("Ix")
    Iy = details.get("Iy")

    if Ix is not None:
        for delta0 in settings["delta_values"]:
            results["Ix"][delta0] = {}
            for y0 in settings["y_values"]:
                results["Ix"][delta0][y0] = nl.plot_invariant_section(
                    Ix, None, state, plane="x",
                    levels=settings["levels"],
                    grid_points=settings["grid_points"],
                    rmin=settings["rmin"], rmax=settings["rmax"],
                    delta0=delta0, folder=str(folder),
                    x_max=settings["x_max"], px_max=settings["px_max"],
                    y_max=settings["y_max"], py_max=settings["py_max"],
                    frozen_q0=y0,
                    frozen_p0=settings["frozen_momentum"],
                    save=save, show=show,
                )

    if Iy is not None:
        for delta0 in settings["delta_values"]:
            results["Iy"][delta0] = {}
            for x0 in settings["x_values"]:
                results["Iy"][delta0][x0] = nl.plot_invariant_section(
                    None, Iy, state, plane="y",
                    levels=settings["levels"],
                    grid_points=settings["grid_points"],
                    rmin=settings["rmin"], rmax=settings["rmax"],
                    delta0=delta0, folder=str(folder),
                    x_max=settings["x_max"], px_max=settings["px_max"],
                    y_max=settings["y_max"], py_max=settings["py_max"],
                    frozen_q0=x0,
                    frozen_p0=settings["frozen_momentum"],
                    save=save, show=show,
                )
    return results

def magnet_snapshot(lattice):
    result = {}
    for elem in lin.unique_magnets(lattice):
        name = str(lin.magnet_field(elem, "NAME"))
        result[name] = {
            "type": str(lin.magnet_field(elem, "TYPE")),
            "length": float(lin.magnet_field(elem, "LENGTH")),
            "angle": float(lin.magnet_field(elem, "ANGLE")),
            "K": float(lin.magnet_field(elem, "K")),
            "S": float(lin.magnet_field(elem, "S")),
            "O": float(lin.magnet_field(elem, "O")),
        }
    return result


def save_final_lattice(file_name, context):
    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    magnets = []
    for elem in lin.unique_magnets(context["lattice"]):
        magnets.append({
            "name": lin.magnet_field(elem, "NAME"),
            "type": lin.magnet_field(elem, "TYPE"),
            "length": float(lin.magnet_field(elem, "LENGTH")),
            "angle": float(lin.magnet_field(elem, "ANGLE")),
            "K": float(lin.magnet_field(elem, "K")),
            "S": float(lin.magnet_field(elem, "S")),
            "O": float(lin.magnet_field(elem, "O")),
        })

    payload = {
        "parameters": context["parameters"],
        "chromatic_correction": context["correction"],
        "magnets": magnets,
        "ring": [lin.magnet_field(elem, "NAME") for elem in context["lattice"]],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# =============================================================================
# 4. CMA-ES + POWELL
# =============================================================================


def hybrid_optimize(
    context,
    v0,
    vary,
    *,
    Fobj,
    compute_ix,
    compute_iy,
    objective_kwargs,
    tol,
    invalid_penalty,
    sigma,
    scales,
    cma_time,
    popsize,
    print_every,
    powell_time_fraction,
    plot_start_end_slices,
    plot_root,
    slice_settings,
):
    """Run start diagnostics, CMA-ES, Powell, and final diagnostics.

    The search budget is wall-clock based but is not a hard process deadline:
    one invariant/objective evaluation is always allowed to finish.

    CMA search scales use abs(v0) for nonzero parameters. Parameters starting at
    zero use the user SCALES entry so they still have a meaningful step size.
    """
    try:
        import cma
    except ImportError as exc:
        raise ImportError("Install CMA-ES with: pip install cma") from exc

    requirements = objective_requirements(Fobj)
    if "Ix" in requirements and not compute_ix:
        raise ValueError("The selected objective requires Ix but COMPUTE_IX is False.")
    if "Iy" in requirements and not compute_iy:
        raise ValueError("The selected objective requires Iy but COMPUTE_IY is False.")

    v0 = np.asarray(v0, dtype=float)

    cma_scales = np.abs(v0).copy()
    for i, name in enumerate(vary):
        if cma_scales[i] == 0.0:
            cma_scales[i] = float(scales[name])
        if not np.isfinite(cma_scales[i]) or cma_scales[i] <= 0.0:
            raise ValueError(f"Scale for {name} must be a positive finite value.")

    cma_time = float(cma_time)
    powell_time_fraction = float(powell_time_fraction)
    if not np.isfinite(cma_time) or cma_time <= 0.0:
        raise ValueError("cma_time must be a positive finite value.")
    if not np.isfinite(powell_time_fraction) or powell_time_fraction < 0.0:
        raise ValueError("powell_time_fraction must be a non-negative finite value.")

    start_parameters = dict(context["parameters"])
    start_snapshot = magnet_snapshot(context["lattice"])
    start_correction = context["correction"]
    start_details = full_diagnostics(
        context, Fobj, tol,
        compute_ix=compute_ix,
        compute_iy=compute_iy,
        objective_kwargs=objective_kwargs,
    )

    start_plots = None
    if plot_start_end_slices and ("Ix" in start_details or "Iy" in start_details):
        start_plots = plot_slices(
            start_details,
            context["state"],
            Path(plot_root) / "start",
            slice_settings,
        )

    best_x = v0.copy()
    best_f = float(start_details["objective"])

    def objective(x):
        return evaluate_candidate(
            context,
            x,
            vary,
            Fobj,
            compute_ix=compute_ix,
            compute_iy=compute_iy,
            objective_kwargs=objective_kwargs,
            tol=tol,
            invalid_penalty=invalid_penalty,
        )

    options = {"verb_disp": 0}
    if popsize is not None:
        options["popsize"] = int(popsize)

    cma_v0 = v0 / cma_scales
    es = cma.CMAEvolutionStrategy(cma_v0, sigma, options)

    cma_start = time.monotonic()
    iteration = 0
    cma_finished = False
    while not es.stop() and not cma_finished:
        if time.monotonic() - cma_start >= cma_time:
            break

        solutions = es.ask()
        values = []
        used_solutions = []
        for scaled_x in solutions:
            if time.monotonic() - cma_start >= cma_time:
                cma_finished = True
                break
            x = np.asarray(scaled_x, dtype=float) * cma_scales
            value = objective(x)
            used_solutions.append(scaled_x)
            values.append(value)
            if value < best_f:
                best_f = float(value)
                best_x = x.copy()

        if cma_finished:
            break
        es.tell(used_solutions, values)
        iteration += 1
        if print_every and iteration % int(print_every) == 0:
            elapsed = time.monotonic() - cma_start
            print(
                f"[CMA-ES] {iteration}   {elapsed:.1f}/{cma_time:.1f} s   "
                f"best J = {best_f:.6e}"
            )

    powell_time = powell_time_fraction * cma_time
    powell_start = time.monotonic()
    powell_best_x = best_x.copy()
    powell_best_f = best_f

    class _PowellTimeLimit(Exception):
        pass

    def powell_objective(x):
        nonlocal powell_best_x, powell_best_f
        if time.monotonic() - powell_start >= powell_time:
            raise _PowellTimeLimit
        value = objective(x)
        if value < powell_best_f:
            powell_best_f = float(value)
            powell_best_x = np.asarray(x, dtype=float).copy()
        if time.monotonic() - powell_start >= powell_time:
            raise _PowellTimeLimit
        return value

    if powell_time > 0.0:
        try:
            res = minimize(
                powell_objective,
                best_x,
                method="Powell",
                options={"disp": False},
            )
            if float(res.fun) < powell_best_f:
                powell_best_f = float(res.fun)
                powell_best_x = np.asarray(res.x, dtype=float).copy()
        except _PowellTimeLimit:
            pass

    x_final = powell_best_x
    apply_candidate(context, x_final, vary)
    final_details = full_diagnostics(
        context, Fobj, tol,
        compute_ix=compute_ix,
        compute_iy=compute_iy,
        objective_kwargs=objective_kwargs,
    )
    final_parameters = dict(context["parameters"])
    final_snapshot = magnet_snapshot(context["lattice"])
    final_correction = context["correction"]

    final_plots = None
    if plot_start_end_slices and ("Ix" in final_details or "Iy" in final_details):
        final_plots = plot_slices(
            final_details,
            context["state"],
            Path(plot_root) / "end",
            slice_settings,
        )

    return {
        "objective_function": getattr(Fobj, "__name__", Fobj.__class__.__name__),
        "x_final": np.asarray(x_final, dtype=float),
        "f_final": float(final_details["objective"]),
        "start_details": start_details,
        "final_details": final_details,
        "start_parameters": start_parameters,
        "final_parameters": final_parameters,
        "start_snapshot": start_snapshot,
        "final_snapshot": final_snapshot,
        "start_correction": start_correction,
        "final_correction": final_correction,
        "start_plots": start_plots,
        "final_plots": final_plots,
        "context": context,
    }
