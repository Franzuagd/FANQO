# Structured invariant comparison

Two experimental constructors are available through `INVARIANT_CONSTRUCTION`:

- `graded_coupled`: solve small linear least-squares problems in ascending `(total degree, delta degree)`. Both homological residuals and the linearized bracket enter each block. Earlier blocks are frozen. The bracket is penalized, not constrained to vanish, and this is not the global nonlinear coupled minimizer.
- `canonical_graded`: fit a shared sequence of Lie generators to both CS actions. Apply each complete truncated exponential to both invariants, including all induced higher-order terms. Their projected Poisson bracket vanishes up to floating-point error by construction.

Both keep the on-momentum CS quadratic sectors fixed and set pure-delta coefficients to zero. They require the full two-plane basis and support delta order 0 or 1. Coupling alone does not eliminate action reparametrization freedom; the explicit gauge and ridge select a representative.

The canonical generators have transverse degree at least two and total degree at least three. Their derivations preserve the rectangular truncation ideal and increase total degree, so the finite exponential preserves the projected bracket. Dispersion-like generators `delta*x` are deliberately excluded: they can lower transverse degree and invalidate this argument. Consequently this constructor cannot fit delta-linear invariant terms. An exact zero bracket alone is not a guarantee of good tracking invariance.

For a graded block, the Poisson bracket of two unknown coefficients cannot contribute to that same block. At delta order one, products of two delta-dependent unknowns are discarded. This gives a linear block problem, even with previously fitted delta-linear terms. Tests cover this fact and canonical conjugation of an integrable map.

## Existing coupled baseline

`a_box_coupled_fixed` and `a_box_coupled_regularized` port the expensive nonlinear coupled solver from `Ixcononly` commit `73117df9f761cd3fddbb7e0104532bc7795fe2f3`. The objective and analytic gradient are preserved; its least-squares seed is adapted to this branch. A shared bracket tensor is cached outside repeated constructor timings. These methods respectively fix the quadratic sectors or permit penalized cross-action mixing. Their original dimensional objective weights are retained, not silently replaced by the new methods' normalized weights.

The legacy `a_box` and `eigen` defaults remain unchanged. New pair methods use the normalized monomial basis consistent with their box Gram matrix. Eigen retains its original basis and selector. The experiment compares physical coefficients in one common basis and reports quadratic seed mismatch, rather than assuming eigen solves the identical constrained problem.

## Local run

From a checkout of this branch:

```bash
python -m pip install -e '.[tracking]'
python user/run_structured_comparison.py --small --deltas 0 -.01 .01
python user/run_structured_comparison.py
```

Alternatively download only `user/run_structured_comparison.py` after updating the installed package. The runner uses a packaged copy of this branch's reference lattice, so no extra local files are needed. To use your own settings add `--config general_config.py`; its `LATTICE_FILE` is resolved relative to that file. The runner's order/grid/turn CLI settings control the experiment.

The default full run uses m=8, d=1, a 25x25 grid, 512 turns and delta=0,+/-0.01. The small run uses m=4,d=1,7x7,64 turns and delta=0. Add `--timing-only` to skip tracking. Constructor repeats default to three and BLAS threads to one. No lattice optimization is performed by this runner. The existing optimization API also accepts the new method names and constructs an auxiliary Iy even if only Ix is requested.

Set method-specific options with `INVARIANT_OPTIONS` in your configuration. New defaults are `STRUCTURED_BRACKET_WEIGHT=1.0` and `STRUCTURED_RIDGE=1e-10`. The coupled baseline defaults are bracket weight 1, seed-mixing weight 10, nonlinear regularization 1e-3, and maximum 200 iterations. See `user/general_config.py` for their exact option names.

Outputs include construction timings, diagnostics, coefficients, shared trajectories, tracking metrics, and comparison plots. Tracking uses the same corrected physical ring and trajectories for every method. The primary metric is maximum absolute invariant drift divided by a fixed initial CS-action scale (with a documented floor in the code); it does not reward a method for changing its own denominator. Lost particles are excluded from the survivor statistic and explicitly counted. Red in the difference plots means lower error than eigen; blue means higher error.

## Small measured test

Linux, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0; m=4,d=1,140 basis elements. Median of three constructor calls with shared setup and map excluded:

| Method | Time (ms) | Relative homological Ix residual | Relative projected bracket |
|---|---:|---:|---:|
| eigen (Ix only) | 7.65 | 9.56e-9 | not measured |
| coupled fixed (Ix,Iy) | 19.53 | 0.301 | 9.16e-4 |
| coupled regularized (Ix,Iy) | 19.90 | 0.301 | 9.16e-4 |
| graded coupled (Ix,Iy) | 3.76 | 0.0240 | 0.0249 |
| canonical graded (Ix,Iy) | 2.20 | 0.0985 | 2.07e-16 |

Common box setup was 0.106 s and its map 0.065 s; eigen setup was 0.145 s and its map 0.066 s. Both nonlinear coupled optimizations reported convergence after 91 iterations. These timings include constructor diagnostics, but exclude external comparison diagnostics and tracking. Eigen's quadratic sector differs from the fixed CS seed by 10.4% in the common box norm, so its much smaller homological residual is not evidence of superiority in the same feasible set.

Median maximum normalized tracking drift over surviving particles, on the same 7x7 grid up to 10 mm and 64 turns:

| Delta | Survivors / 49 | Eigen | Coupled fixed | Coupled regularized | Graded | Canonical |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 1.417 | 0.576 | 0.576 | 0.167 | 0.179 |
| -0.01 | 19 | 6.011 | 1.999 | 1.999 | 1.642 | 1.464 |
| +0.01 | 23 | 1.114 | 0.652 | 0.652 | 1.428 | 1.518 |

This is a construction and tracking smoke test, not a claim of uniformly good invariants. Many launches are lost, tail errors are large, and the new methods are worse at +0.01. Zero bracket and low polynomial homological residual do not imply small drift on the full physical tracking map. The m=8 experiment is supplied for local use but was not timed here. Full settings, individual timing samples, convergence information and tail statistics are saved in `benchmarks/structured_m4_d1.json`.
