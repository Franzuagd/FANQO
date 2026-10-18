FANQO LOCAL STARTER FILES

These files are intentionally separate from the FANQO GitHub repository.

1. Create a working folder anywhere on your computer.
2. Copy general_config.py, lattice_config.py, and run.py into it.
3. Edit general_config.py for analysis/optimization settings.
4. Edit lattice_config.py for your actual accelerator lattice.
5. Activate the conda environment where FANQO is installed.
6. From this folder run:

   python run.py

Your generated plots/reports stay in this local working folder.

TWO OBJECTIVE COMPARISON (action-geometry-main-0.3 package)
------------------------------------------------------
Place run_a_box.py and run_action_geometry.py beside the same general_config.py.
Run each independently:

   python run_a_box.py
   python run_action_geometry.py

Both follow the original run.py workflow and respect your configured CMA_TIME
and POWELL_TIME_FRACTION (quick mode is disabled). Each starts from the lattice
selected in general_config.py. They select their method/objective in memory;
your configuration file is not modified. Outputs go to OUTPUT_ROOT/a_box and
OUTPUT_ROOT/action_geometry. FMA is performed if RUN_FMA_START_END=True and
PyAT is installed; otherwise optimization and the native plots/reports still run.

Third reference: run_eigen.py selects eigen + reference_fluctuation_index
(the advisor derivative/fluctuation objective) and saves to OUTPUT_ROOT/eigen.
It follows the same full-budget workflow. REFERENCE_OBJECTIVE_* settings in
general_config.py control its grid; missing settings use the prior defaults.

PLOT DISPLAY MODE
-----------------
In general_config.py:
    SAVE_PLOTS = True
    SHOW_PLOTS = False
saves figures without opening GUI windows. This is the recommended mode for long runs.

