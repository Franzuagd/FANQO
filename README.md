# FANQO — Ixcononly

Experimental branch for studying **horizontal nonlinear invariant construction**
on a fixed accelerator lattice.

This branch intentionally removes magnet optimization, objective campaigns,
automatic `a_box` calibration, CMA-ES, Powell, and optimization reports. Its
purpose is to answer one question cleanly:

> Given the same lattice and the same physical tracked trajectories, which
> construction produces the better horizontal quasi-invariant `Ix`?

## Invariant constructors

`fq.available_methods()` returns:

### `a_box`

Original full 5-D weighted least-squares continuation of the horizontal
Courant-Snyder quadratic invariant:

```text
min_h ||(I - T_nn) h - T_nq Sx||_G
```

The polynomial basis is scaled using `A_BOX`.

### `a_box_y0`

Exact horizontal-slice version of `a_box`. Before the basis, Gram matrix, or
Lie algebra is built, FANQO restricts to

```text
y = py = 0
```

and keeps only monomials in

```text
(delta, x, px)
```

The least-squares solve is still G-weighted, but G is now the reduced horizontal
Gram matrix. This is not a tiny-y approximation.

### `hybrid`

Uses the same **unscaled physical Cartesian monomial representation** as the
eigen construction:

```text
C = 1
epsilon = 1
```

but does not diagonalize `T-I`. Instead it fixes the Courant-Snyder quadratic
block and solves

```text
(I - T_nn) h = T_nq Sx
```

with ordinary Euclidean least squares:

```python
np.linalg.lstsq(D, U)
```

No Gram metric `G` and no Cholesky weighting enter this construction.

### `eigen`

Full unscaled physical monomial basis. FANQO diagonalizes `T-I`, selects a
near-fixed real eigenvector, and normalizes its horizontal quadratic sector.

## Minimal workflow

From `user/`:

```python
import fanqo as fq

fq.load("general_config.py", force=True)

a = fq.construct("a_box")
b = fq.construct("a_box_y0")
h = fq.construct("hybrid")
e = fq.construct("eigen")
```

To inspect physical coefficients:

```python
c = fq.coefficients("hybrid")
```

To inspect the symbolic polynomial:

```python
Ix = fq.polynomial("a_box")
```

## Paired tracking comparison

The main research function is:

```python
result = fq.compare("a_box", "eigen")
```

The two methods are evaluated on the **same tracked particle trajectories**.

The plotted score is

```text
score = log10(D_name2) - log10(D_name1)
```

where `D` is the maximum relative Ix drift per completed ring turn.

Therefore:

- **red** = `name1` has smaller Ix drift;
- **blue** = `name2` has smaller Ix drift;
- white = comparable;
- gray = particle lost / invalid comparison.

For example:

```python
fq.compare("hybrid", "eigen")
```

means red = hybrid better and blue = eigen better.

If either method is `a_box_y0`, `compare()` automatically restricts the launch
set to the mathematically valid slice `y0=0`.

Each comparison writes:

```text
ix_construction_output/<name1>_vs_<name2>/
├── comparison.png
└── comparison.csv
```

## Configuration

`user/general_config.py` contains only:

- lattice selection;
- Hamiltonian and polynomial order;
- `A_BOX`;
- chromatic correction;
- physical tracking box, grid, turns, and integration steps;
- output controls.

There are no optimizer settings on this branch.

## Install

```bash
python -m pip install -e ".[tracking,dev]"
pytest -q
```

## Branch purpose

Do not merge optimization experiments into this branch. `Ixcononly` is meant
to remain a small, reproducible laboratory for mathematical comparisons between
invariant constructors.
