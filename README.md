# FANQO

**FANQO — Franzua's Accelerator Nonlinear Quasi-Invariant Optimizer**

Python research package for linear optics, nonlinear polynomial quasi-invariants,
CMA-ES/Powell optimization, frequency-map analysis (FMA), and full-ring tracking
of the horizontal invariant `Ix`.

## Repository structure

This repository contains the reusable FANQO library itself. Machine-specific
configuration and runner files are intentionally kept outside the repository so
that each user maintains their own local experiment files.

```text
.
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── fanqo/
│       ├── __init__.py
│       ├── api.py
│       ├── config_loader.py
│       ├── state.py
│       └── core/
│           ├── __init__.py
│           ├── linear.py
│           ├── nonlinear.py
│           └── optimization.py
└── tests/
```

The local working directory used by a researcher normally contains files such as:

```text
my_fanqo_experiment/
├── general_config.py
├── lattice_config.py
└── run.py
```

Those files are user-editable and should not be committed to the FANQO library
repository. A separate `fanqo_user_starter.zip` can be distributed for this
purpose.

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

fq.load("general_config.py")
fq.status()

fq.linear_summary()
fq.plot_linear()
fq.write_linear_report()

fq.compute_invariants()
fq.plot_invariant()
fq.write_invariant_report()

result = fq.optimize()
```

After `optimize()`, the optimized lattice and final `Ix`/`Iy` are the active
in-memory state.

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
