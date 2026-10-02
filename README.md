# FANQO — Ixcononly

Research branch for comparing horizontal nonlinear invariant constructions on a
fixed accelerator lattice.

This branch has no magnet optimization.  The intended workflow is:

1. build the nonlinear polynomial map;
2. construct several candidate invariants;
3. track the physical grid once;
4. reuse the saved trajectories for every invariant-quality and comparison plot.

## Common polynomial representation

Every method uses exactly the same full monomial index set

```text
[delta, x, y, px, py]
```

up to the configured `ORDER` and `DELTA_ORDER`.

The stored basis is

```text
e_i(z) = C_i z^alpha_i
```

and physical coefficients are

```text
c_physical = C * c_stored.
```

`C` is the main representation choice.  `G` is built after `C`.

The defaults are visible in `user/general_config.py`:

- `a_box`: C from A_BOX, G is the box L2 metric;
- `a_box_y0`: same full basis and same C/G, but coefficients containing y or py are constrained to zero;
- `hybrid`, `eigen`, `cesaro`, `abel`: C=(1,1,1,1,1), coefficient metric;
- `graded_ls`: factorial storage C, followed by the explicit
  Courant-Snyder-normalized factorial Fischer graded solve.

The y0 method therefore still returns a full-length polynomial vector.  Its
vertical and mixed coefficients are zero, so the resulting Ix is well-defined
for arbitrary y even though the construction itself uses only horizontal
monomials.

## Adding a construction

The main edit point is:

```text
src/fanqo/core/nonlinear.py
```

A new construction normally needs only:

1. a `construct_<name>(...)` function;
2. one entry in `CONSTRUCTORS`;
3. one default C/G entry in `DEFAULT_METHOD_OPTIONS`.

The monomial basis, lattice map, coefficient conversion, and tracking code do
not need to change.

## Tracking cache

Physical AT tracking is separated from plotting and invariant construction.

```text
ix_construction_output/tracking_cache/
```

contains the turn-by-turn coordinates, survival data, and FMA tune/diffusion
values.  The launch grid, 1 nm zero-avoidance offset, split-window frequency
analysis, and Ix drift definition follow the working FANQO
Development-0.3/main implementation.  FMA is post-processed from the same saved
512-turn trajectories, so plotting additional invariants does not track the
particles again.

Invariant drift is the established FANQO quantity

```text
D = max_n |Ix(n)-Ix(0)| / (max(|Ix(0)|, floor) * n)
```

and the individual invariance map uses the fixed log10 range configured by
`IX_INVARIANCE_LOG_MIN/MAX`.

## Plots

This branch intentionally has only two plot types:

- one-method tracked Ix invariance map;
- pairwise comparison map
  `log10(D_name2) - log10(D_name1)`.

For comparison maps:

```text
positive / red  -> name1 has smaller Ix drift
negative / blue -> name2 has smaller Ix drift
```

## Runner

Edit only the selections at the top of:

```text
user/run.py
```

The runner does not print progress.  A compact description of the selected run
is written to:

```text
ix_construction_output/run_configuration.txt
```

## Install

```bash
python -m pip install --upgrade --force-reinstall "git+https://github.com/Franzuagd/FANQO.git@Ixcononly"
```

For a local editable checkout:

```bash
python -m pip install -e ".[tracking,dev]"
pytest -q
```
