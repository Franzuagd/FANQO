# FANQO

**FANQO — Franzua's Accelerator Nonlinear Quasi-Invariant Optimizer**

Python research package for linear optics, nonlinear polynomial quasi-invariants,
CMA-ES/Powell optimization, frequency-map analysis (FMA), and full-ring tracking
of the horizontal invariant `Ix`.

## Repository structure

This repository contains the reusable FANQO library and a `user/` starter folder.
The files under `user/` are templates that users copy to their own working directory
and edit for their accelerator and run settings.

```text
.
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── fanqo/
│       ├── __init__.py
│       ├── api.py
│       ├── a_box.py
│       ├── config_loader.py
│       ├── state.py
│       └── core/
│           ├── __init__.py
│           ├── linear.py
│           ├── nonlinear.py
│           ├── objective_functions.py
│           └── optimization.py
├── tests/
└── user/
    ├── general_config.py
    ├── lattice_config.py
    └── run.py
```

The local working directory used by a researcher normally contains files such as:

```text
my_fanqo_experiment/
├── general_config.py
├── lattice_config.py
└── run.py
```

The files in `user/` are examples. Copy them outside the FANQO repository before
editing them for a real experiment.

## Install with Anaconda

```bash
conda create -n fanqo python=3.12 -y
conda activate fanqo
python -m pip install --upgrade pip
```

From a local clone of this repository:

```bash
python -m pip install -e ".[dev]"
```

For FMA and Ix tracking with Accelerator Toolbox:

```bash
python -m pip install -e ".[tracking,dev]"
```

## Basic import

```python
import fanqo
print(fanqo.__version__)
```

## Typical local workflow

From a directory containing your own `general_config.py` and
`lattice_config.py`:

```python
import fanqo as fq
from fanqo.core.objective_functions import horizontal_invariant_shape

fq.load("general_config.py")
fq.status()

fq.linear_summary()
fq.plot_linear()
fq.write_linear_report()

fq.compute_invariants()
fq.plot_invariant()
fq.write_invariant_report()

result = fq.optimize(horizontal_invariant_shape)
```

After `optimize()`, the optimized lattice and whichever invariant planes are enabled in `general_config.py` are the active in-memory state.


## Choosing a_box

The user can keep a fixed normalization box:

```python
A_BOX_MODE = "fixed"
A_BOX = np.array([0.01, 10e-3, 8e-3, 1e-3, 0.8e-3])
```

or ask FANQO to calibrate it once before the main magnet optimization:

```python
A_BOX_MODE = "auto"
```

In automatic mode FANQO performs one FMA-like physical tracking grid, keeps the complete survivor trajectories, proposes a seed `a_box` from their phase-space envelope, and runs a short CMA-ES search in `log(a_box)`. Every CMA candidate is scored using the same saved trajectories, so no extra particle tracking is performed. The winning `a_box` is then rebuilt once and remains fixed throughout the full magnet optimization.

The same calibration can be requested manually at any time:

```python
result = fq.optimize_a_box()
print(result["best_a_box"])
print(result["best_score"])
```

## FMA and Ix tracking

With the tracking extra installed:

```python
fq.compute_invariants()
diagnostic = fq.run_fma()
fq.write_tracking_report()
```

The invariant may be computed from one or several analysis cells. Tracking is
performed on an inferred physical 360-degree ring, and Ix drift is measured once
per completed full-ring turn.

## Reports

```python
fq.write_linear_report()
fq.write_invariant_report()
fq.write_optimization_report()
fq.write_tracking_report("start")
fq.write_tracking_report("end")
fq.write_full_report()
```

## Tests

```bash
python -m pip install -e ".[dev]"
pytest -q
```

The repository tests check the installed public package interface. Full
machine-specific numerical smoke tests should be run from the user's local
working files.

## Install directly from GitHub

After replacing `USER` with the repository owner:

```bash
python -m pip install "git+https://github.com/USER/fanqo.git"
```

or, for an exact tagged release:

```bash
python -m pip install "git+https://github.com/USER/fanqo.git@v0.2.0"
```

## Plot display and saving

The starter `general_config.py` separates saving figures from displaying them:

```python
SAVE_PLOTS = True
SHOW_PLOTS = False
```

This mode saves all requested figures without opening GUI windows and uses a
non-interactive Matplotlib backend, which is recommended for optimization, FMA,
remote sessions, and long runs. Set `SHOW_PLOTS = True` when interactive windows
are desired.
