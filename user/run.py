"""End-to-end tutorial for FANQO."""
from importlib.util import find_spec
import fanqo as fq
from fanqo.core.objective_functions import horizontal_invariant_shape

def main():
    fq.load("general_config.py")
    fq.status()

    fq.linear_summary()
    linear_plot = fq.plot_linear()
    linear_report = fq.write_linear_report()

    fq.compute_invariants()
    fq.nonlinear_checks()
    initial_plots = fq.plot_invariant()
    invariant_report = fq.write_invariant_report()

    tracking_available = find_spec("at") is not None
    # The optimizer now receives the objective explicitly.
    result = fq.optimize(
        horizontal_invariant_shape,
        quick=True,
        run_start_end_fma=tracking_available,
    )

    # Optional validation tools:
    # fq.plot_invariant_tracking("x")
    # a_box_result = fq.optimize_a_box()

    # The optimized machine is now the active in-memory state.
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
