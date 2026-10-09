"""Compare eigen Ix with coupled Ix/Iy, first fixed then free quadratic actions.

Run from the user directory:
    python run_coupled_comparison.py

Install FANQO editable first: pip install -e ..
Tracking data are cached and reused for all three methods.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import fanqo as fq


CONFIG = str(Path(__file__).resolve().with_name("general_config.py"))
METHODS = (
    "eigen",
    "a_box_coupled_fixed",
    "a_box_coupled_regularized",
)
FORCE_TRACKING = False
FORCE_METRICS = False


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def main():
    fq.load(CONFIG, force=True)
    # Public API handles default output paths for maps and comparisons.
    output = Path(__file__).resolve().parent / "coupled_comparison_output"
    output.mkdir(parents=True, exist_ok=True)

    results = {}
    for method in METHODS:
        print(f"Constructing {method} ...", flush=True)
        result = fq.construct(method, force=True)
        results[method] = result
        details = result["details"]
        print(f"  construction diagnostics: {details}", flush=True)
        if details.get("solver_success") is False:
            print("  WARNING: coupled solve did not converge: "
                  + str(details.get("solver_message")), flush=True)
        if "Iy" in result:
            np.save(output / f"{method}_Iy.npy", result["Iy"])
        np.save(output / f"{method}_Ix.npy", result["Ix"])

    diagnostics = {method: _serializable(result["details"])
                   for method, result in results.items()}
    (output / "construction_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    for delta in fq.tracking_deltas():
        print(f"Tracking at delta={delta:+.5f}", flush=True)
        trajectory = fq.track(delta=delta, force=FORCE_TRACKING)
        folder = output / ("delta_" + f"{delta:+.5f}".replace("+", "p").replace("-", "m"))
        folder.mkdir(exist_ok=True)
        metrics = {}
        for method in METHODS:
            metric = fq.invariance(method, tracking=trajectory,
                                   force=FORCE_METRICS)
            metrics[method] = metric
            fq.plot_invariance(
                method, tracking=trajectory,
                output_path=folder / f"{method}_invariance.png",
                force_metrics=FORCE_METRICS, show=False,
            )
        for candidate in METHODS[1:]:
            # Color convention from FANQO:
            # red = candidate (method1) has lower Ix drift;
            # blue = eigen (method2) has lower Ix drift.
            comparison = fq.plot_comparison(
                candidate, "eigen", tracking=trajectory,
                output_path=folder / f"{candidate}_vs_eigen.png",
                force_metrics=FORCE_METRICS, show=False,
            )
            print(f"  {candidate} vs eigen: "
                  f"{comparison['name1_better']} coupled wins, "
                  f"{comparison['name2_better']} eigen wins, "
                  f"{comparison['ties']} ties", flush=True)
        # Direct fixed-vs-free comparison, same tracking trajectories.
        fq.plot_comparison(
            "a_box_coupled_regularized", "a_box_coupled_fixed",
            tracking=trajectory,
            output_path=folder / "free_vs_fixed.png",
            force_metrics=FORCE_METRICS, show=False,
        )
        summary = {}
        for method, metric in metrics.items():
            values = np.asarray(metric["log10D"], dtype=float)
            valid = np.asarray(metric["valid"], dtype=bool) & np.isfinite(values)
            summary[method] = {
                "valid_particles": int(valid.sum()),
                "median_log10_relative_Ix_drift":
                    float(np.median(values[valid])) if valid.any() else None,
            }
        (folder / "tracking_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    print(f"Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
