"""All constructors on one fixed lattice; m=8, d=1. No magnet optimization.

Install the corrected branch, place this file by your general_config.py, run:
    python run_Ix_comb.py
Defaults use your existing FMA grid and turn count, at seven delta values.
For a smaller check: python run_Ix_comb.py --steps 12 12 --turns 256
All data and plots stay in the local output directory.
"""
from pathlib import Path
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import argparse
import csv
import json
import tempfile
import time
import numpy as np
import sympy as sp
import fanqo as fq
from fanqo import api
from fanqo.core import nonlinear as nl, linear as lin, optimization as opt
from fanqo.core import structured_invariants as si

METHODS = ("eigen", "a_box", "a_box_eigen_initial", "a_box_coupled_fixed",
           "a_box_coupled_regularized", "graded_coupled", "canonical_graded")
DELTAS = (0., -.01, .01, -.02, .02, -.03, .03)


def json_value(value):
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items() if k != "canonical_generators"}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2), encoding="utf-8")


def audit_M(state, cfg):
    """Independent monomial differentiation of the supplied Hamiltonian.

    Compare every retained coefficient, for every column, to the PHYSICAL
    coefficient transport generator {H,f}. This checks indexing, scaling,
    delta exponents and signs rather than just comparing two implementations.
    """
    results = []
    for fields in ((0., 0., 0., 0., 0.), (.07, .3, -.8, 1.7, 0.)):
        H = sp.Poly(cfg.HAMILTONIAN.subs(dict(zip(cfg.FIELD_SYMBOLS, fields))), *cfg.VARIABLES)
        expected = np.zeros((len(state["C"]), len(state["C"])))
        for a, c in H.terms():
            for j, b in state["idx_to_vec"].items():
                for q, p in ((1, 3), (2, 4)):
                    v = a[q]*b[p]-a[p]*b[q]
                    if not v:
                        continue
                    power = tuple(a[t]+b[t]-(t == q)-(t == p) for t in range(5))
                    i = state["vec_to_idx"].get(power)
                    if i is not None:
                        expected[i, j] += float(c)*v
        hv = np.asarray(state["H_vec_func"](*fields), float).ravel()
        actual = state["transport_sign"]*nl.assemble_M(hv, state["M_basis"])
        C = state["C"]
        actual = C[:, None]*actual/C[None, :]
        error = float(np.max(np.abs(actual-expected)))
        scale = max(1., float(np.max(np.abs(expected))))
        results.append({"fields": fields, "max_abs_error": error, "relative_error": error/scale})
        if error > 1e-10*scale:
            raise AssertionError("Hamiltonian matrix M failed independent differentiation.")
    return results


def audit_thin(state):
    delta, x, y, px, py = state["variables"]
    O = 2.
    element = ["audit_O", "multipole", 0., 0., 0., 0., O, None, None]
    T = nl.element_transfer(element, state)[0]
    errors = []
    for before, after in ((px**2, (px+O*(x**3-3*x*y**2))**2),
                          (py**2, (py-O*(3*x**2*y-y**3))**2)):
        c = np.asarray(nl.poly_to_vector(before, state["variables"], state["vec_to_idx"]), float)/state["C"]
        expected = np.asarray(nl.poly_to_vector(sp.expand(after), state["variables"], state["vec_to_idx"]), float)
        errors.append(float(np.max(np.abs((T @ c)*state["C"]-expected))))
    if max(errors) > 1e-10:
        raise AssertionError("Thin octupole does not match exact polynomial substitution.")
    return errors


def construct_all(cfg, context, methods, output):
    state_a = context["state"]
    if state_a.get("horizontal_slice_only", False):
        raise ValueError("All-method comparison requires positive two-plane A_BOX widths.")
    state_p = dict(state_a, C=state_a["epsilon"].copy())
    start = time.perf_counter()
    T, tnn, tnq = opt.nonlinear_transfer(context, cfg.LEAST_SQUARES_TOL)
    map_time = time.perf_counter()-start
    start = time.perf_counter()
    state_e = nl.initialize_nonlinear_eigen(
        context["data"], 8, 1, cfg.HAMILTONIAN, cfg.A_BOX,
        cfg.VARIABLES, cfg.FIELD_SYMBOLS, n=cfg.N_PLANES)
    independent_eigen_map = nl.nonlinear_transfer(context["lattice"], state_e, tol=cfg.LEAST_SQUARES_TOL)[0]
    eigen_map_time = time.perf_counter()-start
    C = state_a["C"]
    # Preserve the original eigen workflow. Audit numerical basis dependence
    # explicitly instead of replacing the user's reference constructor.
    Te = independent_eigen_map
    difference = Te-C[:, None]*T/C[None, :]
    similarity = float(np.linalg.norm(difference)/max(np.linalg.norm(independent_eigen_map), 1e-300))
    audit = {"M_a_box": audit_M(state_a, cfg), "M_eigen": audit_M(state_e, cfg),
             "thin_octupole_errors": audit_thin(state_a),
             "map_similarity_relative_error": similarity,
             "map_similarity_box_relative_error": float(np.linalg.norm(independent_eigen_map/C[:, None]*C[None, :]-T)/max(np.linalg.norm(T), 1e-300)),
             "comparison_map": "original native maps retained; M independently checked in each basis; finite precision map discrepancy reported",
             "graded_strategy": "delta outer layer; transverse degree inside each layer",
             "triangular_leakage": si._check_structure(T, state_p)[1],
             "truncation_limit": "M is the correctly projected generator. Its exponential need not equal the exact physical jet when discarded higher transverse terms can return through delta-linear terms."}
    save_json(output/"matrix_audit.json", audit)
    print(f"M audit passed. Map similarity error: physical={similarity:.3g}, "
          f"box={audit['map_similarity_box_relative_error']:.3g}", flush=True)
    constructors = {}
    timing = []
    start = time.perf_counter()
    ex, dx = nl.eigen_invariant(Te, state_e, plane="x")
    ey, dy = nl.eigen_invariant(Te, state_e, plane="y")
    eigen_seconds = time.perf_counter()-start
    sx_a, sy_a = nl.quadratic_invariants(context["data"], state_a)
    sx_p, sy_p = si.seeds(context["data"], state_p)
    for method in methods:
        start = time.perf_counter()
        if method == "eigen":
            ix, iy, st = ex, ey, state_e
            details = {"Ix_details": dx, "Iy_details": dy}
        elif method == "a_box":
            result = nl.invariant(tnn, tnq, sx_a, sy_a, state_a, cfg.LEAST_SQUARES_TOL)
            ix, iy, st = result[-2], result[-1], state_a
            details = {"weighted_residual_x": result[2], "weighted_residual_y": result[3]}
        elif method == "a_box_eigen_initial":
            ix, ix_details = nl.eigen_initialized_a_box(T, state_a, sx_a,
                eigen_initial=ex, plane="x", tol=cfg.LEAST_SQUARES_TOL)
            iy, iy_details = nl.eigen_initialized_a_box(T, state_a, sy_a,
                eigen_initial=ey, plane="y", tol=cfg.LEAST_SQUARES_TOL)
            st = state_a
            details = {"Ix_details": ix_details, "Iy_details": iy_details}
        else:
            st = dict(state_p, invariant_construction=method,
                      method_options=dict(getattr(cfg, "INVARIANT_OPTIONS", {}).get(method, {})))
            if method.startswith("a_box_coupled"):
                st["method_options"]["COUPLED_FIX_QUADRATIC"] = method.endswith("fixed")
            ix, iy, details = si.construct_pair(T, context["data"], st, cfg.LEAST_SQUARES_TOL)
        seconds = eigen_seconds if method == "eigen" else time.perf_counter()-start
        if not np.isfinite(ix).all() or not np.isfinite(iy).all():
            raise FloatingPointError(f"Nonfinite coefficients for {method}")
        physical_x, physical_y = ix*st["C"], iy*st["C"]
        q = st["quad_size"]
        own_T = Te if method == "eigen" else T
        own_sx, own_sy = nl.quadratic_invariants(context["data"], st)
        diagnostics = si.pair_diagnostics(own_T, ix, iy,
            np.r_[own_sx, np.zeros(len(ix)-q)], np.r_[own_sy, np.zeros(len(iy)-q)], st)
        if method in ("a_box", "a_box_eigen_initial", "a_box_coupled_fixed", "graded_coupled", "canonical_graded"):
            if not (np.allclose(ix[:q], own_sx, atol=1e-13, rtol=1e-12) and
                    np.allclose(iy[:q], own_sy, atol=1e-13, rtol=1e-12)):
                raise AssertionError(f"{method} changed its prescribed quadratic seed")
        if method == "canonical_graded" and diagnostics["relative_poisson_bracket"] > 1e-7:
            raise AssertionError("Canonical constructor failed its projected zero bracket.")
        constructors[method] = {"physical_x": physical_x, "physical_y": physical_y,
                                "details": details, "diagnostics": diagnostics}
        np.savez_compressed(output/f"{method}_coefficients.npz", Ix=physical_x, Iy=physical_y,
                            powers=np.array(list(st["idx_to_vec"].values()), int))
        timing.append({"method": method, "constructor_seconds": seconds,
                       "eigen_initial_seconds": eigen_seconds if method == "a_box_eigen_initial" else 0.,
                       "total_constructor_seconds": seconds+(eigen_seconds if method == "a_box_eigen_initial" else 0.)})
        print(f"{method}: {seconds:.3f} s; residual={diagnostics['relative_homological_x']:.3g}; "
              f"strategy={details.get('fitting_strategy', 'native')}", flush=True)
    save_json(output/"constructors.json", constructors)
    with (output/"construction_times.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=timing[0]); writer.writeheader(); writer.writerows(timing)
    return constructors, state_e, {"a_box_map_seconds": map_time, "eigen_setup_and_map_seconds": eigen_map_time}


def evaluate_all(coefficients, powers, coordinates):
    """Shared monomial features; chunk the trajectory to bound memory usage."""
    coords = np.asarray(coordinates, float)[[4, 0, 2, 1, 3]]
    shape = coords.shape[1:]
    flat = coords.reshape(5, -1)
    result = np.empty((len(coefficients), flat.shape[1]))
    for start in range(0, flat.shape[1], 2048):
        z = flat[:, start:start+2048]
        tables = [np.stack([z[axis]**k for k in range(int(powers[:, axis].max())+1)]) for axis in range(5)]
        features = np.ones((len(powers), z.shape[1]))
        for axis in range(5):
            features *= tables[axis][powers[:, axis]]
        result[:, start:start+z.shape[1]] = coefficients @ features
    return result.reshape((len(coefficients),)+shape)


def map_plot(values, xs, ys, path, title, label, *, vmin, vmax, cmap="viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    xx, yy = np.meshgrid(xs, ys)
    image = ax.pcolormesh(xx, yy, np.asarray(values).reshape(len(ys), len(xs)),
                         shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    if not np.isfinite(values).any():
        ax.text(.5, .5, "No finite results on this grid", transform=ax.transAxes,
                ha="center", va="center")
    fig.colorbar(image, ax=ax).set_label(label)
    ax.set(xlabel=r"$x_0$ [mm]", ylabel=r"$y_0$ [mm]", title=title)
    ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def track_delta(cfg, native, ring, orbit, constructors, state, delta, output):
    import at
    from at.physics.frequency_maps import fmap_parallel_track
    name = f"delta_{delta:+.3f}"
    root = output/name; root.mkdir(parents=True, exist_ok=True)
    coords = list(cfg.FMA_COORDS_MM); steps = list(cfg.FMA_STEPS); turns = int(cfg.FMA_TURNS)
    xs, ys = api._fma_grid(coords, steps)
    # FMA is a property of the fixed lattice, so calculate it once per delta.
    offset = np.zeros(6); offset[4] = delta
    start = time.perf_counter()
    fmap, loss = fmap_parallel_track(ring, coords=coords, steps=steps, turns=turns,
        orbit=orbit, add_offset6D=offset, verbose=False, lossmap=True,
        pool_size=cfg.FMA_POOL_SIZE if cfg.FMA_POOL_SIZE is not None else 1)
    np.savetxt(root/"fma.csv", fmap, delimiter=",", comments="",
               header="x_mm,y_mm,nux,nuy,dnux,dnuy,log10_tune_diffusion")
    methods = list(constructors)
    coeff = np.stack([constructors[m]["physical_x"] for m in methods])
    powers = np.array(list(state["idx_to_vec"].values()), int)
    initial_rows = [api._initial_row(xs, y, orbit, delta) for y in ys]
    initial_values = np.stack([evaluate_all(coeff, powers, z.T) for z in initial_rows], axis=1)
    floors = cfg.IX_INVARIANCE_NORM_FLOOR_FRACTION*np.max(np.abs(initial_values), axis=(1, 2))
    floors = np.maximum(floors, np.finfo(float).tiny)
    tables = {m: [] for m in methods}
    sample_curves = {m: [] for m in methods}
    nturns = 2*turns
    for row, (y, z0) in enumerate(zip(ys, initial_rows)):
        tracked, tracking_info = at.lattice_pass(ring, np.asfortranarray(z0.T.copy()), nturns, losses=True)
        tracks = np.asarray(tracked)[:, :, 0, :]
        # Common physical trajectories and survivor mask for ALL methods.
        lost = np.asarray(tracking_info.get("islost", np.zeros(len(xs), bool)), dtype=bool)
        finite = np.isfinite(tracks).all(axis=0)
        completed = np.where(finite.all(axis=1), nturns, np.argmax(~finite, axis=1))
        valid = (completed == nturns) & ~lost
        safe_tracks = tracks.copy(); safe_tracks[:, ~valid] = 0.
        values = evaluate_all(coeff, powers, safe_tracks)
        initial = initial_values[:, row, :]
        den = np.maximum(np.abs(initial), floors[:, None])
        drift = np.abs(values-initial[:, :, None])
        rates = drift/den[:, :, None]/np.arange(1, nturns+1)
        for mi, method in enumerate(methods):
            for i, x in enumerate(xs):
                if valid[i]:
                    maxabs = float(drift[mi, i].max())
                    maxrel = float((drift[mi, i]/den[mi, i]).max())
                    rms = float(np.sqrt(np.mean((drift[mi, i]/den[mi, i])**2)))
                    rate = float(rates[mi, i].max())
                    lograte = float(np.log10(max(rate, 1e-300)))
                    final = float(values[mi, i, -1])
                    if len(sample_curves[method]) < 6 and abs(y) < max(abs(ys).max()/3, 1e-6):
                        sample_curves[method].append((float(x), float(y), values[mi, i].copy(), float(initial[mi, i])))
                else:
                    final = maxabs = maxrel = rms = rate = lograte = np.nan
                tables[method].append([x, y, initial[mi, i], final, maxabs, maxrel,
                                      rms, rate, lograte, int(completed[i]), float(valid[i])])
        print(f"  delta={delta:+.3f}, row {row+1}/{len(ys)}", flush=True)
    arrays = {m: np.asarray(v, float) for m, v in tables.items()}
    # One fixed scale for every difference plot, independent of constructor.
    differences = {}
    reference = arrays["eigen"]
    for method, a in arrays.items():
        if method == "eigen":
            continue
        valid = (a[:, 10] > .5) & (reference[:, 10] > .5) & np.isfinite(a[:, 8]) & np.isfinite(reference[:, 8])
        diff = np.full(len(a), np.nan)
        diff[valid] = reference[valid, 8]-a[valid, 8]
        differences[method] = diff
    stats = {}
    import matplotlib.pyplot as plt
    for method, a in arrays.items():
        dest = root/method; dest.mkdir(exist_ok=True)
        api._save_csvs(fmap, a, dest, method)
        # Individual FMA plot; physically identical for each constructor.
        diffusion = np.full(len(xs)*len(ys), np.nan)
        for r in fmap:
            ix = int(np.argmin(abs(xs-r[0]))); iy = int(np.argmin(abs(ys-r[1])))
            diffusion[iy*len(xs)+ix] = r[6]
        map_plot(diffusion, xs, ys, dest/"frequency_map.png",
            f"FMA | {method} | delta={delta:+.3f}", r"$\log_{10}$ tune diffusion", vmin=-10, vmax=-2)
        valid = (a[:, 10] > .5) & np.isfinite(a[:, 8])
        shown = np.where(valid, a[:, 8], np.nan)
        map_plot(shown, xs, ys, dest/"Ix_invariance.png",
            f"{method} | delta={delta:+.3f} | {nturns} ring turns",
            r"$\log_{10}\max_k |I_x(k)-I_x(0)|/[k\max(|I_x(0)|,\epsilon_I)]$",
            vmin=cfg.IX_INVARIANCE_LOG_MIN, vmax=cfg.IX_INVARIANCE_LOG_MAX)
        fig, ax = plt.subplots(figsize=(8, 5))
        for x, y, vals, v0 in sample_curves[method]:
            ax.plot(np.arange(nturns+1), np.r_[v0, vals], label=f"x={x:.2f}, y={y:.2f} mm")
        ax.set(xlabel="Full-ring turn", ylabel=r"$I_x$", title=f"{method} | delta={delta:+.3f}")
        if sample_curves[method]:
            ax.legend(fontsize=8)
        ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(dest/"Ix_vs_turn.png", dpi=180); plt.close(fig)
        if method in differences:
            diff = differences[method]
            # Positive = eigen's error is larger = OTHER method is better.
            # RdBu_r maps positive to red, negative to blue.
            map_plot(diff, xs, ys, dest/"difference_vs_eigen.png",
                f"{method} vs eigen | delta={delta:+.3f}\nRed: {method} better; blue: eigen better",
                r"$\log_{10}(E_{\rm eigen}/E_{\rm method})$", vmin=-3., vmax=3., cmap="RdBu_r")
            np.savetxt(dest/"difference_vs_eigen.csv", np.c_[a[:, :2], diff],
                delimiter=",", comments="", header="x_mm,y_mm,log10_eigen_error_over_method_error")
        stats[method] = {"launched": len(a), "survived": int(valid.sum()),
                         "median_log_relative_rate": float(np.median(a[valid, 8])) if valid.any() else None}
        if method in differences:
            diff = differences[method]; ok = np.isfinite(diff)
            stats[method]["fraction_better_than_eigen"] = float(np.mean(diff[ok] > 0)) if ok.any() else None
    save_json(root/"summary.json", stats)
    print(f"delta={delta:+.3f}: finished in {time.perf_counter()-start:.1f} s", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("general_config.py")))
    parser.add_argument("--output", default="Ix_comb_results")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--deltas", nargs="+", type=float, default=DELTAS)
    parser.add_argument("--steps", nargs=2, type=int)
    parser.add_argument("--turns", type=int)
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    if "eigen" not in args.methods:
        parser.error("Include eigen as the common reference.")
    if not hasattr(nl, "eigen_initialized_a_box"):
        raise RuntimeError("Install the corrected Ix_comb branch before running this file.")
    config = Path(args.config).resolve(); output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    # Execute the original config with its original __file__ so runpy-based
    # configs and relative lattice paths keep working. Override only method
    # and the explicitly requested orders before creating the context.
    runtime_text = ("from pathlib import Path\n"
        f"_source = {str(config)!r}\n"
        "_settings = {'__file__': _source, '__name__': '_original_Ix_config'}\n"
        "exec(compile(Path(_source).read_bytes(), _source, 'exec'), _settings)\n"
        "globals().update({k:v for k,v in _settings.items() if not k.startswith('__')})\n"
        "ORDER=8\nDELTA_ORDER=1\nINVARIANT_CONSTRUCTION='a_box'\n")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="_Ix_comb_", dir=config.parent, delete=False) as f:
        runtime = Path(f.name); f.write(runtime_text)
    try:
        context = fq.load(runtime, force=True); cfg = api._cfg()
        if args.steps:
            cfg.FMA_STEPS = args.steps
        if args.turns:
            cfg.FMA_TURNS = args.turns
        if int(cfg.FMA_TURNS) < 64 or min(cfg.FMA_STEPS) < 1:
            parser.error("FMA requires at least 64 turns/window and positive steps.")
        constructors, state, timings = construct_all(cfg, context, args.methods, output)
        report = {"source_config": str(config), "order": 8, "delta_order": 1,
            "methods": args.methods, "deltas": args.deltas,
            "parameters": context["parameters"], "FMA_STEPS": cfg.FMA_STEPS,
            "FMA_COORDS_MM": cfg.FMA_COORDS_MM, "FMA_TURNS_PER_WINDOW": cfg.FMA_TURNS,
            "Ix_tracking_turns": 2*cfg.FMA_TURNS, "map_timing": timings,
            "metric": "existing FANQO maximum relative drift per turn, with its normalization floor",
            "difference_sign": "log10(E_eigen/E_method): positive/red=other better; negative/blue=eigen better",
            "launch_convention": "original production on-momentum closed orbit plus delta offset",
            "FMA_scope": "same fixed lattice for all methods; shared tracking, no magnet optimization", "results": {}}
        save_json(output/"report.json", report)
        if not args.timing_only:
            import at
            native = api._prepare_physical_ring(context["parameters"])
            # Do not silently compare a differently corrected physical machine.
            if any(not np.isclose(native["parameters"][k], v, rtol=1e-10, atol=1e-10)
                   for k, v in context["parameters"].items() if isinstance(v, (int, float, np.number))):
                raise ValueError("Analysis and tracking chromatic corrections differ; use matching correction settings.")
            ring = api._build_at_ring(native, at); orbit, _ = at.find_orbit4(ring)
            for delta in args.deltas:
                report["results"][str(delta)] = track_delta(cfg, native, ring, orbit, constructors, state, delta, output)
                save_json(output/"report.json", report)
        print(f"Saved local results: {output}", flush=True)
    finally:
        runtime.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
