IxOptCompare-0.3

The experiment is intentionally fixed to four cases:

    a_box + h_reduction
    eigen + h_reduction
    a_box + advisor_derivative
    eigen + advisor_derivative

Edit general_config.py for the lattice/optimizer budget.
Edit run.py only if you want to change QUICK or disable start/end FMA.

A_BOX is fixed. There is no a_box calibration.

Run:

    python run.py

Results are written under:

    optimization_compare/<construction>__<objective>/

and the compact matrix summary is:

    optimization_compare/comparison_summary.json
