"""Small end-to-end FANQO tutorial.

Read this file as the public workflow, not as part of the numerical core:

    load -> inspect linear lattice -> compute Ix -> choose Fobj
         -> optimize -> inspect optimized state -> write reports

Use quick=True here for a smoke test. Long research campaigns belong in
master_run_48h.py or a dedicated experiment runner.
"""
from importlib.util import find_spec
import fanqo as fq
from fanqo.core.objective_functions import (
    horizontal_invariant_shape,
    reference_fluctuation_index,
)

def main():
    # 1. Build a fresh context from general_config.py + lattice_config.py.
    fq.load("general_config.py")
    fq.status()

    # 2. Inspect the linear optics before doing nonlinear work.
    fq.linear_summary()
    linear_plot = fq.plot_linear()
    linear_report = fq.write_linear_report()

    # 3. Build Ix/Iy using the construction selected in general_config.py.
    fq.compute_invariants()
    fq.nonlinear_checks()
    initial_plots = fq.plot_invariant()
    invariant_report = fq.write_invariant_report()

    # 4. Objective choice is independent of invariant construction.
    tracking_available = find_spec("at") is not None
    # Choose the objective independently from INVARIANT_CONSTRUCTION:
    Fobj = horizontal_invariant_shape
    # Fobj = reference_fluctuation_index

    # If INVARIANT_CONSTRUCTION="a_box" and A_BOX_MODE="auto", FANQO
    # calibrates a_box once before the magnet optimization. eigen does
    # not use that calibration.
    # 5. optimize() updates the active context to the winning lattice.
    result = fq.optimize(
        Fobj,
        quick=True,
        run_start_end_fma=tracking_available,
    )

    # Optional validation/manual calibration tools:
    # fq.plot_invariant_tracking("x")
    # a_box_result = fq.optimize_a_box()  # can be run manually whenever desired

    # 6. From here onward the public API refers to the optimized machine until
    #    fq.load() is called again.
    fq.status()
    final_plots = fq.plot_invariant()
    optimization_report = fq.write_optimization_report()

    if tracking_available:
        fq.write_tracking_report("start")
        fq.write_tracking_report("end")

    full_report = fq.write_full_report()

    print("\nTutorial outputs")
    print("-" * 72)
    print("Linear plot        :", linear_plot)
    print("Linear report      :", linear_report)
    print("Invariant report   :", invariant_report)
    print("Optimization report:", optimization_report)
    print("Full report        :", full_report)
    print("Initial plots      :", initial_plots["folder"])
    print("Final plots        :", final_plots["folder"])
    print("Final objective    :", result["final_details"]["objective"])

if __name__ == "__main__":
    main()
