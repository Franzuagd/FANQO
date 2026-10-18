"""Optimize eigen with the original main workflow.

Requires FANQO branch action-geometry-main-0.3. Place beside general_config.py
and its lattice file. Uses configured CMA/Powell time budgets, without quick mode.
"""
from importlib.util import find_spec
from pathlib import Path
from fanqo import config_loader
import fanqo as fq

def main():
    config = Path(__file__).resolve().with_name("general_config.py")
    cfg = config_loader.load_general_config(config, reload=True)
    # Override only this run's method/objective; never edit the config file.
    cfg.INVARIANT_CONSTRUCTION = 'eigen'
    cfg.OBJECTIVE = 'reference_fluctuation_index'
    # Separate all outputs so the two optimizations cannot overwrite each other.
    cfg.OUTPUT_ROOT = Path(cfg.OUTPUT_ROOT) / 'eigen'
    cfg.REPORT_FILE = cfg.OUTPUT_ROOT / "optimization_report.txt"
    cfg.FINAL_LATTICE_FILE = cfg.OUTPUT_ROOT / "final_lattice.json"
    cfg.PLOT_ROOT = cfg.OUTPUT_ROOT / "slices"
    cfg.FMA_OUTPUT_DIRECTORY = cfg.OUTPUT_ROOT / "FMA"
    # load() reuses the config module just prepared above (no force reload).
    fq.load(config)
    print("Construction:", cfg.INVARIANT_CONSTRUCTION, " Objective:", cfg.OBJECTIVE)
    print("CMA seconds:", cfg.CMA_TIME, " Powell fraction:", cfg.POWELL_TIME_FRACTION)
    fq.status()

    fq.linear_summary()
    linear_plot = fq.plot_linear()
    linear_report = fq.write_linear_report()

    fq.compute_invariants()
    fq.nonlinear_checks()
    initial_plots = fq.plot_invariant(folder=cfg.OUTPUT_ROOT / "initial_invariants")
    invariant_report = fq.write_invariant_report()

    tracking_available = find_spec("at") is not None
    do_tracking = bool(cfg.RUN_FMA_START_END) and tracking_available
    if cfg.RUN_FMA_START_END and not tracking_available:
        print("PyAT is unavailable: start/end FMA is skipped, as in the original runner.")
    result = fq.optimize(quick=False, run_start_end_fma=do_tracking)

    # The optimized machine is now the active in-memory state.
    fq.status()
    final_plots = fq.plot_invariant(folder=cfg.OUTPUT_ROOT / "final_invariants")
    optimization_report = fq.write_optimization_report()

    if do_tracking:
        fq.write_tracking_report("start")
        fq.write_tracking_report("end")

    full_report = fq.write_full_report()

    print("\neigen outputs")
    print("-" * 72)
    print("Linear plot        :", linear_plot)
    print("Linear report      :", linear_report)
    print("Invariant report   :", invariant_report)
    print("Optimization report:", optimization_report)
    print("Full report        :", full_report)
    print("Initial plots      :", initial_plots["folder"])
    print("Final plots        :", final_plots["folder"])
    print("Final objective    :", result["final_details"]["objective"])
    return result

if __name__ == "__main__":
    main()
