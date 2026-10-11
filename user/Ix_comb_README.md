# Ix_comb local comparison

This copy starts from `IxOptCompare-0.3`. Use the same lattice/configuration
workflow. No magnet optimization is performed. Numerical results are local.

Install this copy from the repository root:

```bash
python -m pip install -e ".[tracking]"
python user/run_Ix_comb.py --config user/general_config.py
```

To use the supplied OPA lattice and make a smaller construction/tracking check:

```bash
python user/run_Ix_comb.py --config research/opa_config.py --steps 20 20 --turns 256
```

The default run forces `ORDER=8`, `DELTA_ORDER=1`, and uses the existing FMA
extent, grid and turn count. The original config file is not rewritten.
`--timing-only` constructs and audits without tracking. `--output PATH` chooses
the local output directory. `--deltas 0 -.01 .01 -.02 .02 -.03 .03` is the
default. These physical momentum offsets differ from `DELTA_ORDER=1`.

Available constructors:

| Option | Construction |
| --- | --- |
| `eigen` | Original physical-basis near-unit eigenvector selector. |
| `a_box` | Original weighted least squares with fixed CS quadratic coefficients. |
| `a_box_eigen_initial` | Eigen first; least-squares correction from its higher-sector coefficients, with A_box's usual CS quadratic constraint. |
| `a_box_coupled_fixed` | Existing coupled homological/bracket objective, fixed quadratic sector. |
| `a_box_coupled_regularized` | Existing coupled objective with a soft quadratic constraint. |
| `graded_coupled` | Coupled linear fits, layer by layer in delta, then by transverse degree. |
| `canonical_graded` | Existing shared Lie-generator construction, with the corrected layer ordering. |

The new option also works in `INVARIANT_CONSTRUCTION` in the normal config.
For a weighted design matrix `A`, right-hand side `b`, and eigen tail `h0`, it
uses `h = h0 + lstsq(A, b-A@h0)`. This retains the numerical nullspace component
of `h0`; calling ordinary `lstsq(A,b)` would discard that initial guess. Its
quadratic coefficients are still the prescribed CS seed, not eigen's seed.

Delta is an outer layer. All transverse degrees in delta layer `s` precede
layer `s+1`. Thus `delta**3*x*px` has transverse degree 2. Delta-dependent
forcing from an earlier layer may lower transverse degree without violating
this ordering. The genuine nontriangular fallback remains available.

Checks and limitations:

- `matrix_audit.json` independently differentiates every monomial column of
  the supplied Hamiltonian and checks retained entries of M, signs and basis
  scaling in both representations. Thin-octupole polynomial transport is
  checked against exact substitution.
- The eigen algorithm and its native map are preserved. The audit reports
  numerical differences between independently constructed maps after basis
  conversion. Near-unit eigenvector selection can be sensitive to this
  roundoff; agreement of the generators does not ensure numerically identical
  selected eigenvectors.
- Constructor diagnostics include coupled solver status. An iteration-limit
  result is a finite iterate, not evidence of convergence or a global minimum.
- `canonical_graded` retains its existing restriction on generators: it
  excludes dispersion-like transverse-linear generators to preserve projected
  involution. A zero bracket does not itself establish invariance.
- Rectangular truncation with dispersion remains a model limitation:
  discarded higher transverse degrees can feed retained terms in a later
  delta layer. The M audit verifies the projected generator, not equality of
  its exponential to an exact physical Taylor map. The correction here changes
  the degree-by-degree solver ordering, not the chosen truncation convention.
- The legacy A_box bracket-norm diagnostic now uses the actual coefficient
  scale C; this does not change its least-squares coefficients.

Each delta directory contains individual FMA, Ix-invariance and Ix-versus-turn
plots for every method, plus CSV data. Each non-eigen method has a separate
difference plot using `log10(E_eigen/E_method)`: **red = other method better;
blue = eigen better**. Blank cells represent lost trajectories. All methods
use identical physical trajectories and survivor masks. FMA is shared because
the lattice is fixed; selecting a different polynomial cannot change its tunes.

The invariance metric is the existing FANQO maximum relative drift per turn,
with its existing small-denominator floor. It is method-dependent and is not
a proof of a universally best invariant. The comparison is based on surviving
launches in the configured domain. Results at offsets beyond the delta box
and the first-order delta expansion are extrapolations.

Do not commit `Ix_comb_results/`: coefficients, CSVs and plots belong to the
local experiment. The run prints construction timings before starting FMA.
