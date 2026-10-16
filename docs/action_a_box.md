# Action-domain least squares and three-method comparison

Branch: `action-a-box-0.3`, based on `Ix_comb-layered-0.3` at
`bdca9771aa1fc720f9b45dc76303882b3932340a`.

## Install and run

```bat
python -m pip install --upgrade "fanqo[tracking] @ git+https://github.com/Franzuagd/FANQO.git@action-a-box-0.3"
python run_action_comparison.py --quick --deltas 0
python run_action_comparison.py
```

Put `user/run_action_comparison.py` beside your existing `general_config.py`
and its lattice file. The runner uses the installed FANQO package. It defaults
to m=8, d=1 and the three methods `action_a_box`, `a_box`, `eigen`. It computes
Ix only, with no magnet or box optimization. The full run uses the current
config's plot settings and tracking grid, at delta=0, +/-0.01, +/-0.02, +/-0.03.
Fitting uses the action delta range; diagnostics outside that range are
explicit extrapolation tests.

`--quick` retains m=8,d=1 but uses five intervals (six points) per tracking axis, 32 FMA turns per
frequency window, 64 Ix/Poincare turns and 80-point contour grids. It tests
the workflow, not publication-quality FMA. `--timing-only` skips plots.

Optional CLI settings:

```bat
python run_action_comparison.py --jx-max 1.75e-5 --jy-max 1.83e-5 --delta-max 0.01 --deltas 0 -0.01 0.01 --output comparison_action
```

The runner's editable ACTION_BOX/CLI values override general_config.py for
the comparison. In ordinary FANQO use, select:

```python
INVARIANT_CONSTRUCTION = 'action_a_box'
ACTION_BOX = {'Jx_max': 1.75e-5, 'Jy_max': 1.83e-5, 'delta_max': 0.01}
```

`INVARIANT_OPTIONS['action_a_box']['ACTION_BOX']`, when supplied, takes priority
over the top-level ACTION_BOX in ordinary API usage. All bounds must be
positive. A_BOX must also have positive widths to build the full basis.

## Exact mathematical definition

FANQO uses **no factor 1/2** in its seeds:

    Jx = Sx = gamma_x*x^2 + 2*alpha_x*x*px + beta_x*px^2,
    Jy = Sy = gamma_y*y^2 + 2*alpha_y*y*py + beta_y*py^2.

These J are twice the conventionally normalized canonical actions. The
constant Jacobian factor cancels in the normalized probability measure used
here. The fitting domain is Sx<=Jx_max, Sy<=Jy_max, |delta|<=delta_max.
It is uniform in both actions, both full angles, and delta. The Twiss
parameters are the on-momentum CS0 at the invariant observation point; the
domain is not shifted by off-momentum closed orbit or dispersion.

For the convention above, x_max=sqrt(beta_x*Jx_max), and analogously for y.
The supplied defaults correspond approximately to 10mm horizontal and 8mm
vertical extents for the reference branch lattice. They remain fixed action
bounds if a different lattice is used; the physical extents will change.

The polynomial space and projected transport are the existing FANQO ones.
This is the complete smooth polynomial version of the radial-Fourier ansatz:
rewriting every smooth harmonic in Cartesian monomials is an invertible
basis change. Construction does NOT use tracked samples, arbitrary nonsmooth
radial harmonics, or a separately truncated angular bandwidth.

For coefficient seed s and free-column insertion E, solve

    min_h || W (T-I) (s+E h) ||_2^2,  W.T W = G_action.

The entire delta=0 block of transverse degrees 0,1,2 is fixed to the CS seed.
All remaining FANQO coefficients, including chromatic layers, remain free,
matching regular a_box's admissible space. Delta degree is handled separately
from transverse degree. No Poisson-bracket constraint or ridge penalty is
added. The solver uses SVD least squares and the configured rcond.

`G_action` is computed from exact moment formulas in floating-point arithmetic.
For normalized disk coordinates X^2+P^2<=A,

    mean(X^(2i) P^(2j)) = A^(i+j) (2i)! (2j)! /
                          [4^(i+j) i! j! (i+j+1)!],

and odd moments vanish. The CS inverse transformation is
x=sqrt(beta) X, px=(P-alpha X)/sqrt(beta). Products of the two disk moments
and uniform-delta moments yield the full physical Gram matrix. It is then
converted to the stored e_i=C_i*monomial_i coefficient basis.

The original A_BOX supplies only internal monomial scaling for action_a_box;
the physical fitting domain comes from ACTION_BOX. Scaling and the SVD
threshold can affect numerical rank and selection among nonunique minimizers.
Metric matrices are refreshed when the Twiss parameters change.

## Eigen selection on this branch

Diagonalize T-I. Retain near-zero real eigenvalue candidates in the adaptive
window (including exact zero and either sign). Require a real eigenvector
with positive requested-plane quadratic determinant, normalize that
determinant to one with a positive momentum-squared coefficient, and select
the candidate minimizing

    ||quadratic(candidate)-quadratic(Sx)||_2 / ||quadratic(Sx)||_2.

The distance uses all ten delta=0 degree-two **physical monomial coefficients**.
The unwanted vertical and cross-plane targets are zero. Ix selection uses
only horizontal beta, alpha, gamma, with no Sy seed or joint Ix/Iy fit.
The explicit Iy API analogously selects against Sy alone.

This is selection among returned eigensolver vectors, not optimization over
their entire degenerate span. Distance depends on the specified physical
coefficient convention and is reported. Unlike LS, selection does not enforce
an exactly equal CS quadratic part. No claim of a fairer result is made solely
from changing the selector; inspect the reported seed mismatch and tracking.

## Outputs and comparisons

Per method and delta:

- Existing static Ix contour slices.
- Existing physical Poincare points over Ix contours.
- Existing FMA and tracked-Ix invariance maps and CSVs.
- Coefficients with exponent ordering and C scales; invariant report.

`comparison_plots/` includes common-color-scale invariance panels, pairwise
log-drift ratio maps/CSVs, excursion CDFs and drift versus launch radius.
Curve colors: action_a_box teal, a_box red, eigen blue. Ratio maps have their
own signed scale: negative/red means the numerator method is better;
positive/blue means the denominator is better. Thus a_box/eigen preserves
red=a_box better and blue=eigen better.

`comparison.json` records action bounds, original Cartesian bounds, CS0,
actual corrected lattice parameters, constructor diagnostics, quadratic
errors and tracking summaries. `construction_times.csv` separates setup
from map-plus-solve time. These are whole-call timings, not constructor-only
benchmarks.

FMA is computed once per delta and reused because the physical lattice does
not change. Invariant tracking is reevaluated on the same deterministic
launch grid for each polynomial. Existing plots retain the existing
method-specific Ix(0) normalization and its floor; these are relative-error
comparisons, not a common externally fixed Sx denominator.

An action-domain LS minimizer is not guaranteed to track better than a_box or
eigen. Changing the domain changes the objective. A small projected-map
residual does not establish small drift under the physical tracking map.

## Verification

Tests compare the analytic Gram matrix to independent action-angle quadrature
with nonzero Twiss alpha, check action moments, compare the constrained solve
with a sampled LS solve, exercise public dispatch and metric refresh, and
verify that changing Sy parameters cannot change the horizontal eigen choice.
Run `python -m pytest tests -q`.
