# FANQO — a_box vs eigen optimization experiment

This branch starts from `Development-0.3` / FANQO `0.3.0.dev0` and is
deliberately narrow.  It tests whether the invariant construction used inside
the magnet optimizer changes the final physical machine quality.

## Invariant constructions

Only the original two constructors are used:

```text
a_box
eigen
```

`a_box` uses the fixed `A_BOX` from `user/general_config.py`.  There is no
automatic or manual a_box calibration in this branch.

## Optimization objectives

The comparison runner uses exactly two objectives.

### 1. h_reduction

For the coefficient split

```text
Ix = [quadratic block ; h]
```

the score is

```text
J_h = sqrt(h^T G_nn h).
```

For `a_box` this is the usual nonlinear-tail reduction after the fixed
Courant-Snyder block.  For `eigen` the same coefficient split is used, so its
free quadratic normalization is not directly penalized.

The raw value of `J_h` is representation-dependent.  It is an optimizer
objective, not the final cross-method quality metric.

### 2. advisor_derivative

The runner maps this label to the existing
`reference_fluctuation_index` from the 0.3.dev code.  It removes the pure
delta=0 quadratic sector, forms sampled derivatives

```text
g = Ix_x^2 + Ix_y^2 + (w Ix_px)^2 + (w Ix_py)^2,
```

and scores each momentum slice from the mean, standard deviation, and skewness
of `g`.  The worst configured delta slice is used.

## Four-run protocol

`user/run.py` always reloads `lattice_config.PARAMETERS` before each case:

```text
a_box + h_reduction
eigen + h_reduction
a_box + advisor_derivative
eigen + advisor_derivative
```

Each case has an independent output directory under:

```text
user/optimization_compare/
```

The common scientific comparison is the physical start/end FMA and tracked-Ix
diagnostics, not the absolute optimizer objective value.

## Run

```bash
python -m pip install --upgrade --force-reinstall "git+https://github.com/Franzuagd/FANQO.git@IxOptCompare-0.3"
cd user
python run.py
```

Set `QUICK=True` in `user/run.py` only for a smoke test.  With
`QUICK=False`, each of the four cases uses the optimizer budget from
`general_config.py`.

## Main research settings

Edit `user/general_config.py` for:

```text
A_BOX
VARY
CMA_SIGMA
CMA_POPSIZE
CMA_TIME
POWELL_TIME_FRACTION
REFERENCE_OBJECTIVE_*
FMA_*
```

The linear lattice implementation is unchanged from `Development-0.3`.


## Experimental structured pair constructors

`graded_coupled` and `canonical_graded` add fast degree-by-degree pair construction and a shared Lie construction with zero projected bracket. The expensive `a_box_coupled_fixed` and `a_box_coupled_regularized` baselines are also available. Existing defaults are unchanged.

Run `python user/run_structured_comparison.py --small` for a short comparison against eigen, or omit `--small` for the m=8 experiment. See [methods, limitations, timings and local run instructions](docs/structured_comparison.md).
