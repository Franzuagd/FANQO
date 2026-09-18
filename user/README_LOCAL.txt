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

PLOT DISPLAY MODE
-----------------
In general_config.py:
    SAVE_PLOTS = True
    SHOW_PLOTS = False
saves figures without opening GUI windows. This is the recommended mode for long runs.

