"""Small research runner.

Edit only the selections below.  Numerical modules do not print.  The run is
recorded in ix_construction_output/run_configuration.txt.
"""

import fanqo as fq


CONFIG = "general_config.py"

# One-method Ix invariance maps.
INVARIANCE_PLOTS = (
    "a_box",
    "graded_ls",
    "eigen",
)

# Pairwise red/blue maps.
COMPARISON_PLOTS = (
    ("a_box", "graded_ls"),
    ("graded_ls", "eigen"),
)

# False reuses the saved physical trajectories when the tracking configuration
# and lattice parameters are unchanged.
FORCE_TRACKING = False

# False reuses saved Ix drift values when the invariant coefficients are unchanged.
FORCE_METRICS = False


def main():
    fq.load(CONFIG, force=True)

    selected = set(INVARIANCE_PLOTS)
    for name1, name2 in COMPARISON_PLOTS:
        selected.add(name1)
        selected.add(name2)

    methods = tuple(name for name in fq.available_methods() if name in selected)
    for method in methods:
        fq.construct(method)

    tracking = fq.track(force=FORCE_TRACKING)

    for method in INVARIANCE_PLOTS:
        fq.plot_invariance(
            method,
            tracking=tracking,
            force_metrics=FORCE_METRICS,
        )

    for name1, name2 in COMPARISON_PLOTS:
        fq.plot_comparison(
            name1,
            name2,
            tracking=tracking,
            force_metrics=FORCE_METRICS,
        )

    fq.write_report(
        methods=methods,
        comparisons=COMPARISON_PLOTS,
        tracking=tracking,
    )


if __name__ == "__main__":
    main()
