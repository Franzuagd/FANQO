"""48-hour FANQO validation campaign.

This script exercises the full optimization matrix:

    9 optimization objectives x 2 invariant constructions = 18 cases

Every optimization case:
  * starts from a freshly reloaded original lattice,
  * uses the same polynomial truncation for a fair comparison,
  * receives a dynamically allocated wall-clock CMA budget,
  * writes into an isolated output directory,
  * saves final linear/invariant/Poincare plots and reports.

Expensive setup is deliberately shared:
  * the original-lattice FMA is run once,
  * a_box is calibrated once on the original lattice and reused by every
    a_box-construction case,
  * the original invariant/Poincare plots are generated once per construction.

The target campaign duration is 48 hours. The scheduler reserves time for
plots/reports and redistributes unused time among remaining cases. Exact wall
time can still differ slightly because a single objective evaluation cannot be
interrupted halfway through.

Run from the user directory, for example:

    python master_run_48h.py

The generated _master_runtime_config.py is intentionally kept beside this file
so a failed case can be reproduced easily.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

import fanqo as fq
from fanqo.core.objective_functions import (
    barrier_objective,
    configured_objective,
    contour_objective,
    escape_objective,
    flux_objective,
    horizontal_invariant_shape,
    integral_objective,
    mesh_objective,
    reference_fluctuation_index,
    ultimate_objective,
)


# =============================================================================
# 1. CAMPAIGN SETTINGS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

BASE_CONFIG = SCRIPT_DIR / "general_config.py"
RUNTIME_CONFIG = SCRIPT_DIR / "_master_runtime_config.py"
MASTER_ROOT = SCRIPT_DIR / "master_48h_output"

TOTAL_WALL_HOURS = 48.0
FINAL_BUFFER_MINUTES = 45.0
OPTIMIZER_SHARE_OF_CASE = 0.80

# m=6,d=1 keeps the two invariant constructions directly comparable while
# making repeated dense eigendecompositions practical in a two-day campaign.
MASTER_ORDER = 6
MASTER_DELTA_ORDER = 1

MASTER_CMA_SIGMA = 0.50
MASTER_CMA_POPSIZE = 4
MASTER_POWELL_FRACTION = 0.15
MASTER_PRINT_EVERY = 5
MIN_CMA_SECONDS = 5.0 * 60.0

# Common physical slices used by all multi-slice objectives and invariant plots.
# Keeping these fixed across constructions makes the comparison meaningful.
MASTER_Y_VALUES = (0.0, 0.30e-3, 0.60e-3)
MASTER_DELTA_VALUES = (0.0, 0.005, 0.010)
MASTER_FROZEN_PY = 0.0

MASTER_X_MAX = 5.0e-3
MASTER_PX_MAX = 1.0e-3
MASTER_Y_MAX = 3.0e-3
MASTER_PY_MAX = 1.0e-3

# Objective discretization. These are intentionally lighter than a final paper
# scan because every objective is optimized twice in this validation campaign.
MASTER_QUADRATURE_POINTS = 24
MASTER_MESH_POINTS = 61
MASTER_CONTOUR_ANGLES = 72
MASTER_CONTOUR_RADIAL_POINTS = 96

# Tracking plots.
MASTER_POINCARE_TURNS = 128
MASTER_POINCARE_X_VALUES = (2.0e-3, 4.0e-3, 6.0e-3, 8.0e-3)

# FMA is only a baseline diagnostic in this matrix.
MASTER_FMA_STEPS = (80, 80)
MASTER_FMA_TURNS = 256

# One-time a_box calibration.
MASTER_ABOX_TRACKING_STEPS = (12, 12)
MASTER_ABOX_TRACKING_TURNS = 128
MASTER_ABOX_CMA_MAX_EVALS = 24

MASTER_RANDOM_SEED = 20260924

STATE_FILE = MASTER_ROOT / "campaign_state.json"
SUMMARY_CSV = MASTER_ROOT / "campaign_summary.csv"


# =============================================================================
# 2. OBJECTIVE MATRIX
# =============================================================================

COMMON_SLICE_SETTINGS = dict(
    y_values=MASTER_Y_VALUES,
    delta_values=MASTER_DELTA_VALUES,
    frozen_py=MASTER_FROZEN_PY,
    aggregation="worst",
    x_max=MASTER_X_MAX,
    px_max=MASTER_PX_MAX,
)


def build_objectives():
    """Return all objectives that can be passed directly to fq.optimize()."""
    return [
        (
            "horizontal_shape",
            horizontal_invariant_shape,
        ),
        (
            "reference_fluctuation",
            reference_fluctuation_index,
        ),
        (
            "integral",
            configured_objective(
                integral_objective,
                **COMMON_SLICE_SETTINGS,
                quadrature_points=MASTER_QUADRATURE_POINTS,
                lambda_h=1.0,
                lambda_normal=0.20,
                weight_mode="custom",
            ),
        ),
        (
            "mesh",
            configured_objective(
                mesh_objective,
                **COMMON_SLICE_SETTINGS,
                mesh_points=MASTER_MESH_POINTS,
                stability_weight_power=3.0,
                stability_weight_floor=0.02,
                penalty_power=1.0,
            ),
        ),
        (
            "contour",
            configured_objective(
                contour_objective,
                **COMMON_SLICE_SETTINGS,
                angles=MASTER_CONTOUR_ANGLES,
                radial_points=MASTER_CONTOUR_RADIAL_POINTS,
                level_fraction=0.97,
                min_radial_slope=0.02,
            ),
        ),
        (
            "barrier",
            configured_objective(
                barrier_objective,
                **COMMON_SLICE_SETTINGS,
                angles=MASTER_CONTOUR_ANGLES,
                radial_points=MASTER_CONTOUR_RADIAL_POINTS,
                level_fraction=0.97,
                min_radial_slope=0.02,
            ),
        ),
        (
            "flux",
            configured_objective(
                flux_objective,
                **COMMON_SLICE_SETTINGS,
                angles=MASTER_CONTOUR_ANGLES,
                radial_points=MASTER_CONTOUR_RADIAL_POINTS,
                level_fraction=0.97,
                min_radial_slope=0.02,
                annulus_inner=0.90,
                annulus_layers=6,
                cvar_tail=0.01,
            ),
        ),
        (
            "escape",
            configured_objective(
                escape_objective,
                **COMMON_SLICE_SETTINGS,
                angles=MASTER_CONTOUR_ANGLES,
                radial_points=MASTER_CONTOUR_RADIAL_POINTS,
                level_fraction=0.97,
                min_radial_slope=0.02,
                annulus_inner=0.90,
                annulus_layers=6,
                cvar_tail=0.01,
            ),
        ),
        (
            "ultimate",
            configured_objective(
                ultimate_objective,
                **COMMON_SLICE_SETTINGS,
                angles=MASTER_CONTOUR_ANGLES,
                radial_points=MASTER_CONTOUR_RADIAL_POINTS,
                level_fraction=0.97,
                min_radial_slope=0.02,
                annulus_inner=0.90,
                annulus_layers=6,
                cvar_tail=0.01,
            ),
        ),
    ]


CONSTRUCTIONS = ("a_box", "eigen")


def build_cases():
    """Interleave constructions so partial campaigns still contain pairs."""
    cases = []
    for objective_name, objective in build_objectives():
        for construction in CONSTRUCTIONS:
            cases.append({
                "id": f"{objective_name}__{construction}",
                "objective_name": objective_name,
                "objective": objective,
                "construction": construction,
            })
    return cases


# =============================================================================
# 3. PERSISTENT CAMPAIGN STATE
# =============================================================================


def _fresh_state():
    return {
        "started_epoch": time.time(),
        "target_wall_hours": TOTAL_WALL_HOURS,
        "baseline_fma_done": False,
        "baseline_constructions_done": [],
        "best_a_box": None,
        "a_box_calibration_done": False,
        "cases": {},
    }


def load_campaign_state():
    MASTER_ROOT.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    state = _fresh_state()
    save_campaign_state(state)
    return state


def save_campaign_state(state):
    MASTER_ROOT.mkdir(parents=True, exist_ok=True)
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(STATE_FILE)
    write_summary_csv(state)


def write_summary_csv(state):
    rows = []
    for case_id, info in state.get("cases", {}).items():
        rows.append({
            "case_id": case_id,
            "construction": info.get("construction"),
            "objective": info.get("objective"),
            "status": info.get("status"),
            "cma_time_seconds": info.get("cma_time_seconds"),
            "elapsed_seconds": info.get("elapsed_seconds"),
            "start_objective": info.get("start_objective"),
            "final_objective": info.get("final_objective"),
            "output_directory": info.get("output_directory"),
            "error": info.get("error"),
        })

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "construction",
                "objective",
                "status",
                "cma_time_seconds",
                "elapsed_seconds",
                "start_objective",
                "final_objective",
                "output_directory",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def campaign_deadline(state):
    return float(state["started_epoch"]) + TOTAL_WALL_HOURS * 3600.0


# =============================================================================
# 4. GENERATED RUNTIME CONFIGURATION
# =============================================================================


def write_runtime_config(
    *,
    output_root,
    construction,
    a_box,
    cma_time,
    fma_enabled=False,
    case_label="case",
):
    """Generate a temporary config beside general_config.py.

    Importing the base config keeps the actual lattice selection, Hamiltonian,
    variable list, magnet map, chromatic correction settings, and search scales.
    Only campaign-specific settings are overridden here.
    """
    output_root = Path(output_root).resolve()
    a_box = np.asarray(a_box, dtype=float).reshape(5)

    text = f'''# Auto-generated by master_run_48h.py
from general_config import *
from pathlib import Path
import numpy as np

ORDER = {MASTER_ORDER}
DELTA_ORDER = {MASTER_DELTA_ORDER}
INVARIANT_CONSTRUCTION = {construction!r}

# a_box is calibrated once externally by the master runner and then frozen.
A_BOX_MODE = "fixed"
A_BOX = np.array({a_box.tolist()!r}, dtype=float)

COMPUTE_IX = True
COMPUTE_IY = False

CMA_SIGMA = {MASTER_CMA_SIGMA!r}
CMA_POPSIZE = {MASTER_CMA_POPSIZE}
CMA_TIME = {float(cma_time)!r}
POWELL_TIME_FRACTION = {MASTER_POWELL_FRACTION!r}
PRINT_EVERY = {MASTER_PRINT_EVERY}

SAVE_PLOTS = True
SHOW_PLOTS = False
PLOT_START_END_SLICES = False
PLOT_X_MAX = {MASTER_X_MAX!r}
PLOT_PX_MAX = {MASTER_PX_MAX!r}
PLOT_Y_MAX = {MASTER_Y_MAX!r}
PLOT_PY_MAX = {MASTER_PY_MAX!r}
PLOT_GRID_POINTS = 300

SLICE_Y_VALUES = {tuple(MASTER_Y_VALUES)!r}
SLICE_X_VALUES = (0.0,)
SLICE_DELTA_VALUES = {tuple(MASTER_DELTA_VALUES)!r}
SLICE_FROZEN_MOMENTUM = {MASTER_FROZEN_PY!r}

# The legacy/reference fluctuation objective is configured here because the
# public optimize() API already forwards these values by name.
REFERENCE_OBJECTIVE_X_RANGE = 2.5e-3
REFERENCE_OBJECTIVE_X_POINTS = 21
REFERENCE_OBJECTIVE_Y_RANGE = 0.7e-3
REFERENCE_OBJECTIVE_Y_POINTS = 11
REFERENCE_OBJECTIVE_DELTA_VALUES = {tuple(MASTER_DELTA_VALUES)!r}
REFERENCE_OBJECTIVE_MOMENTUM_WEIGHT = 0.7

RUN_FMA_START_END = False
FMA_CASE_LABEL = {case_label!r}
FMA_STEPS = {list(MASTER_FMA_STEPS)!r}
FMA_TURNS = {MASTER_FMA_TURNS}
FMA_SAVE_PLOT = True
FMA_SHOW_PLOT = False

POINCARE_X_VALUES = {list(MASTER_POINCARE_X_VALUES)!r}
POINCARE_Y_VALUES = []
POINCARE_DELTA = 0.0
POINCARE_TURNS = {MASTER_POINCARE_TURNS}
POINCARE_GRID_POINTS = 300

A_BOX_TRACKING_STEPS = {list(MASTER_ABOX_TRACKING_STEPS)!r}
A_BOX_TRACKING_TURNS = {MASTER_ABOX_TRACKING_TURNS}
A_BOX_CMA_MAX_EVALS = {MASTER_ABOX_CMA_MAX_EVALS}

OUTPUT_ROOT = Path({str(output_root)!r})
REPORT_FILE = OUTPUT_ROOT / "reports" / "optimization_report.txt"
FINAL_LATTICE_FILE = OUTPUT_ROOT / "final_lattice.json"
PLOT_ROOT = OUTPUT_ROOT / "slices"
FMA_OUTPUT_DIRECTORY = OUTPUT_ROOT / "FMA"
POINCARE_OUTPUT_DIRECTORY = OUTPUT_ROOT / "poincare"
A_BOX_OUTPUT_DIRECTORY = OUTPUT_ROOT / "a_box"
'''
    RUNTIME_CONFIG.write_text(text, encoding="utf-8")
    return RUNTIME_CONFIG


def snapshot_runtime_config(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        RUNTIME_CONFIG,
        output_root / "runtime_config_snapshot.py",
    )


# =============================================================================
# 5. ONE-TIME BASELINE + A_BOX CALIBRATION
# =============================================================================


def base_a_box_from_config():
    # Load once only to read the untouched user configuration value.
    write_runtime_config(
        output_root=MASTER_ROOT / "00_bootstrap",
        construction="a_box",
        a_box=np.array([0.01, 10e-3, 8e-3, 1e-3, 0.8e-3]),
        cma_time=60.0,
        fma_enabled=False,
        case_label="bootstrap",
    )
    fq.load(str(RUNTIME_CONFIG), force=True)
    return np.asarray(fq.api._cfg().A_BOX, dtype=float).copy()


def run_baseline_fma_and_calibrate(state, initial_a_box):
    baseline_root = MASTER_ROOT / "00_baseline"

    if not state.get("baseline_fma_done", False):
        print("\n" + "=" * 88)
        print("ONE-TIME BASELINE: LINEAR + INVARIANT + FMA")
        print("=" * 88)
        write_runtime_config(
            output_root=baseline_root,
            construction="a_box",
            a_box=initial_a_box,
            cma_time=60.0,
            case_label="original_lattice",
        )
        snapshot_runtime_config(baseline_root)
        fq.load(str(RUNTIME_CONFIG), force=True)
        fq.linear_summary()
        fq.plot_linear(
            file_name=baseline_root / "plots" / "linear_original.png"
        )
        fq.compute_invariants()
        fq.plot_invariant(
            folder=baseline_root / "plots" / "invariant_original"
        )
        fq.plot_invariant_tracking(
            "x",
            file_name=baseline_root / "plots" / "poincare_original.png",
            turns=MASTER_POINCARE_TURNS,
            save=True,
            show=False,
        )
        fq.run_fma(stage="baseline", quick=False)
        fq.write_tracking_report(
            "baseline",
            file_name=baseline_root / "reports" / "tracking_baseline.txt",
        )
        fq.write_linear_report(
            file_name=baseline_root / "reports" / "linear_baseline.txt"
        )
        state["baseline_fma_done"] = True
        save_campaign_state(state)

    if not state.get("a_box_calibration_done", False):
        print("\n" + "=" * 88)
        print("ONE-TIME A_BOX CALIBRATION ON ORIGINAL LATTICE")
        print("=" * 88)
        calibration_root = MASTER_ROOT / "01_a_box_calibration"
        write_runtime_config(
            output_root=calibration_root,
            construction="a_box",
            a_box=initial_a_box,
            cma_time=60.0,
            case_label="a_box_calibration",
        )
        snapshot_runtime_config(calibration_root)
        fq.load(str(RUNTIME_CONFIG), force=True)
        result = fq.optimize_a_box(make_active=True)
        best = np.asarray(result["best_a_box"], dtype=float)
        state["best_a_box"] = best.tolist()
        state["a_box_calibration_done"] = True
        save_campaign_state(state)

    return np.asarray(state["best_a_box"], dtype=float)


def run_constructor_baselines(state, best_a_box, initial_a_box):
    done = set(state.get("baseline_constructions_done", []))

    for construction in CONSTRUCTIONS:
        if construction in done:
            continue

        print("\n" + "=" * 88)
        print(f"BASELINE INVARIANT PLOTS: {construction}")
        print("=" * 88)

        root = MASTER_ROOT / "02_constructor_baselines" / construction
        box = best_a_box if construction == "a_box" else initial_a_box
        write_runtime_config(
            output_root=root,
            construction=construction,
            a_box=box,
            cma_time=60.0,
            case_label=f"baseline_{construction}",
        )
        snapshot_runtime_config(root)
        fq.load(str(RUNTIME_CONFIG), force=True)
        fq.compute_invariants()
        fq.nonlinear_checks()
        fq.plot_invariant(folder=root / "plots" / "invariant")
        fq.plot_invariant_tracking(
            "x",
            file_name=root / "plots" / "poincare.png",
            turns=MASTER_POINCARE_TURNS,
            save=True,
            show=False,
        )
        fq.write_invariant_report(
            file_name=root / "reports" / "invariant_report.txt"
        )

        done.add(construction)
        state["baseline_constructions_done"] = sorted(done)
        save_campaign_state(state)


# =============================================================================
# 6. DYNAMIC 48-HOUR SCHEDULER
# =============================================================================


def cma_budget_for_next_case(state, cases_left):
    """Allocate the remaining campaign wall time among unfinished cases."""
    final_buffer = FINAL_BUFFER_MINUTES * 60.0
    remaining = (
        campaign_deadline(state)
        - time.time()
        - final_buffer
    )
    if remaining <= 0.0:
        return MIN_CMA_SECONDS

    per_case_wall = remaining / max(int(cases_left), 1)
    optimizer_wall = (
        OPTIMIZER_SHARE_OF_CASE * per_case_wall
    )

    # hybrid_optimize spends CMA_TIME in CMA and
    # POWELL_TIME_FRACTION*CMA_TIME in Powell.
    cma_seconds = (
        optimizer_wall
        / (1.0 + MASTER_POWELL_FRACTION)
    )
    return max(
        MIN_CMA_SECONDS,
        float(cma_seconds),
    )


def format_hours(seconds):
    return float(seconds) / 3600.0


# =============================================================================
# 7. INDIVIDUAL CASE
# =============================================================================


def run_case(
    *,
    case,
    case_index,
    total_cases,
    cma_seconds,
    best_a_box,
    initial_a_box,
):
    construction = case["construction"]
    objective_name = case["objective_name"]
    objective = case["objective"]
    case_id = case["id"]

    output_root = (
        MASTER_ROOT
        / "runs"
        / construction
        / objective_name
    )
    box = (
        best_a_box
        if construction == "a_box"
        else initial_a_box
    )

    print("\n" + "=" * 88)
    print(
        f"CASE {case_index + 1:02d}/{total_cases:02d}: "
        f"{objective_name} x {construction}"
    )
    print(
        f"CMA budget: {format_hours(cma_seconds):.3f} h  |  "
        f"Powell: {MASTER_POWELL_FRACTION * 100:.1f}% of CMA"
    )
    print("=" * 88)

    # This fresh load is the guarantee that every case starts from the original
    # lattice rather than from the preceding optimized solution.
    write_runtime_config(
        output_root=output_root,
        construction=construction,
        a_box=box,
        cma_time=cma_seconds,
        case_label=case_id,
    )
    snapshot_runtime_config(output_root)
    fq.load(str(RUNTIME_CONFIG), force=True)

    np.random.seed(
        MASTER_RANDOM_SEED + int(case_index)
    )

    started = time.monotonic()
    result = fq.optimize(
        objective,
        quick=False,
        run_start_end_fma=False,
    )

    # Final plotting/diagnostics. FMA is intentionally omitted here because
    # the baseline FMA was already run once for this validation campaign.
    plot_dir = output_root / "plots"
    fq.plot_linear(
        file_name=plot_dir / "linear_final.png"
    )
    fq.plot_invariant(
        folder=plot_dir / "invariant_final"
    )
    fq.plot_invariant_tracking(
        "x",
        file_name=plot_dir / "poincare_final.png",
        turns=MASTER_POINCARE_TURNS,
        save=True,
        show=False,
    )
    fq.write_invariant_report(
        file_name=output_root / "reports" / "invariant_final.txt"
    )
    fq.write_optimization_report(
        file_name=output_root / "reports" / "optimization_report.txt",
        result=result,
    )
    fq.write_full_report(
        file_name=output_root / "reports" / "full_report.txt"
    )

    elapsed = time.monotonic() - started
    return {
        "construction": construction,
        "objective": objective_name,
        "status": "completed",
        "cma_time_seconds": float(cma_seconds),
        "elapsed_seconds": float(elapsed),
        "start_objective": float(
            result["start_details"]["objective"]
        ),
        "final_objective": float(
            result["final_details"]["objective"]
        ),
        "output_directory": str(output_root),
        "error": None,
    }


# =============================================================================
# 8. MAIN CAMPAIGN
# =============================================================================


def main():
    MASTER_ROOT.mkdir(parents=True, exist_ok=True)
    state = load_campaign_state()
    cases = build_cases()

    print("=" * 88)
    print("FANQO 48-HOUR MASTER VALIDATION")
    print("=" * 88)
    print(f"Output root          : {MASTER_ROOT}")
    print(f"Polynomial space     : m={MASTER_ORDER}, d={MASTER_DELTA_ORDER}")
    print(f"Invariant methods    : {CONSTRUCTIONS}")
    print(f"Objective functions  : {len(build_objectives())}")
    print(f"Optimization cases   : {len(cases)}")
    print(f"Target wall time [h] : {TOTAL_WALL_HOURS}")
    print("=" * 88)

    # Read the untouched user-configured box before the campaign changes
    # anything. This is also the compatibility box for eigen-mode plotting.
    initial_a_box = base_a_box_from_config()

    best_a_box = run_baseline_fma_and_calibrate(
        state,
        initial_a_box,
    )
    print("Calibrated a_box:", best_a_box)

    run_constructor_baselines(
        state,
        best_a_box,
        initial_a_box,
    )

    completed = {
        case_id
        for case_id, info in state.get("cases", {}).items()
        if info.get("status") == "completed"
    }

    for index, case in enumerate(cases):
        case_id = case["id"]
        if case_id in completed:
            print(f"Skipping completed case: {case_id}")
            continue

        unfinished = [
            c for c in cases
            if c["id"] not in completed
        ]
        cma_seconds = cma_budget_for_next_case(
            state,
            len(unfinished),
        )

        case_started = time.monotonic()
        try:
            info = run_case(
                case=case,
                case_index=index,
                total_cases=len(cases),
                cma_seconds=cma_seconds,
                best_a_box=best_a_box,
                initial_a_box=initial_a_box,
            )
            completed.add(case_id)
        except Exception as exc:
            elapsed = time.monotonic() - case_started
            error_text = (
                f"{type(exc).__name__}: {exc}"
            )
            print("\nCASE FAILED:", case_id)
            print(error_text)
            traceback.print_exc()
            info = {
                "construction": case["construction"],
                "objective": case["objective_name"],
                "status": "failed",
                "cma_time_seconds": float(cma_seconds),
                "elapsed_seconds": float(elapsed),
                "start_objective": None,
                "final_objective": None,
                "output_directory": str(
                    MASTER_ROOT
                    / "runs"
                    / case["construction"]
                    / case["objective_name"]
                ),
                "error": error_text,
            }

        state.setdefault("cases", {})[case_id] = info
        save_campaign_state(state)

        elapsed_campaign = (
            time.time() - float(state["started_epoch"])
        )
        remaining = max(
            campaign_deadline(state) - time.time(),
            0.0,
        )
        print(
            f"Campaign elapsed: {format_hours(elapsed_campaign):.2f} h | "
            f"remaining target: {format_hours(remaining):.2f} h"
        )

    print("\n" + "=" * 88)
    print("MASTER CAMPAIGN FINISHED")
    print("=" * 88)
    print("State file :", STATE_FILE)
    print("Summary CSV:", SUMMARY_CSV)
    print("Best a_box :", state.get("best_a_box"))

    completed_count = sum(
        info.get("status") == "completed"
        for info in state.get("cases", {}).values()
    )
    failed_count = sum(
        info.get("status") == "failed"
        for info in state.get("cases", {}).values()
    )
    print(f"Completed   : {completed_count}/{len(cases)}")
    print(f"Failed      : {failed_count}/{len(cases)}")


if __name__ == "__main__":
    main()
