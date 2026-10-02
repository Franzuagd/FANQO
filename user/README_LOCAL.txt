Ixcononly research workflow

1. Edit general_config.py for the mathematical representation:
       C, G, polynomial order, Hamiltonian, tracking grid.

2. Edit run.py only to select:
       INVARIANCE_PLOTS
       COMPARISON_PLOTS
       FORCE_TRACKING
       FORCE_METRICS

3. Run:
       python run.py

The code does not print progress.  The selected configuration is written to:
       ix_construction_output/run_configuration.txt

Physical tracking is cached once in:
       ix_construction_output/tracking_cache/

The cache contains turn-by-turn coordinates and FMA values.  New invariant
plots and pairwise comparisons reuse those trajectories.

To add an invariant construction, edit src/fanqo/core/nonlinear.py:
       - write construct_<name>(...)
       - add it to CONSTRUCTORS
       - choose default C and G in DEFAULT_METHOD_OPTIONS
