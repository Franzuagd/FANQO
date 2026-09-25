"""FANQO public package surface.

Numerical implementation lives in fanqo.api and fanqo.core. Re-exporting the
user-facing functions here keeps normal experiment scripts simple:

    import fanqo as fq
    fq.load(...)
    fq.compute_invariants()
    fq.optimize(Fobj)
"""

from .api import (
    load,
    status,
    current_parameters,
    current_lattice,
    set_parameters,
    linear_summary,
    linear_checks,
    plot_linear,
    write_linear_report,
    compute_invariants,
    nonlinear_checks,
    get_Ix,
    get_Iy,
    invariant_polynomial,
    ix_callable,
    iy_callable,
    plot_invariant,
    write_invariant_report,
    run_fma,
    plot_invariant_tracking,
    optimize_a_box,
    write_tracking_report,
    optimize,
    save_current_lattice,
    write_optimization_report,
    write_full_report,
)

__all__ = [name for name in globals() if not name.startswith("_")]
__version__ = "0.3.0.dev0"
