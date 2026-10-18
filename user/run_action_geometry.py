"""Optimize action_geometry with the original main workflow.

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
    cfg.INVARIANT_CONSTRUCTION = 'action_angle_ls'
    cfg.OBJECTIVE = 'action_geometry'
    # Fallbacks let an older main configuration use the new method too.
    import numpy as np
    cfg.ACTION_BOX = getattr(cfg, 'ACTION_BOX',
        dict(Jx_max=4.3645e-6, Jy_max=1.83e-5, delta_max=.01))
    box = cfg.ACTION_BOX
    cfg.ACTION_GEOMETRY_POINTS = getattr(cfg, 'ACTION_GEOMETRY_POINTS', np.array([
        (dd, jy, fy, f*box['Jx_max'])
        for dd in (-.5*box['delta_max'], 0., .5*box['delta_max'])
        for jy, fy in ((0., 0.), (.1*box['Jy_max'], 0.), (.1*box['Jy_max'], np.pi/2))
        for f in (.1, .3, .5)
    ]))
    cfg.ACTION_GEOMETRY_OPTIONS = getattr(cfg, 'ACTION_GEOMETRY_OPTIONS',
        dict(angles=64, amplitude_scale=.5, slope_scale=1., derivative_weight=.1))
    # Separate all outputs so the two optimizations cannot overwrite each other.
    cfg.OUTPUT_ROOT = Path(cfg.OUTPUT_ROOT) / 'action_geometry'
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

    print("\naction_geometry outputs")
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
