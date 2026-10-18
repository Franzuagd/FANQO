# Eigen reference with the advisor objective

On the existing `action-geometry-main-0.3` branch, select:

```python
INVARIANT_CONSTRUCTION = 'eigen'
OBJECTIVE = 'reference_fluctuation_index'
```

Or put `run_eigen.py` beside your general configuration and lattice file:

```bash
python run_eigen.py
```

This third runner follows the same main workflow as run_a_box.py and
run_action_geometry.py. It starts from the selected initial lattice, respects
CMA_TIME/POWELL_TIME_FRACTION, uses quick=False and saves to OUTPUT_ROOT/eigen.
Initial/final native plots, reports, final lattice saving, and optional
start/end FMA follow the existing runner logic. It does not use tracking to
construct or optimize the invariant.

## Eigen selection and normalization

The implementation recovers the selection previously used on the comparison
branch; it does not replace it with constrained LS or mix eigenvectors.
Main's existing transport and polynomial storage remain unchanged. If C is
the diagonal coefficient-scale matrix, the physical coefficient operator is

    T_physical = C T_stored C^(-1).

Diagonalize T_physical-I, selecting real eigenvectors in the earlier adaptive
window near zero (equivalently eigenvalues of T near one). Exact zero shifts
and shifts of either sign are accepted. The window reproduces the previous
reference implementation and is reported as eigen_cutoff.

For a horizontal quadratic part a*x^2+b*x*px+c*px^2, require

    D = a*c - b^2/4 > 0.

Multiply the **entire polynomial** by sign(c)/sqrt(D), giving positive
horizontal quadratic form and determinant one, as for Courant--Snyder.
Then choose the computed candidate minimizing the Euclidean distance of
**all ten physical delta=0 quadratic coefficients** to Sx. The target
coefficients for other-plane and cross-plane monomials are zero. Distance is
reported relative to the norm of the target. Iy is selected independently
against Sy using the corresponding vertical determinant. Full diagnostics
reuse one eigendecomposition for both planes; candidate evaluation needs
only Ix.

The quadratic part is selected, not reset to Sx. Setting it equal afterward
would generally destroy the eigenvector property. Coefficients are converted
back to main's stored representation for the existing plotting/reporting
functions. Invariant reports include the selected eigenvalue, distance,
normalization, cutoff and candidate count.

## Advisor objective

The recovered objective is `reference_fluctuation_index` from the earlier
implementation, with its previous settings. Remove the pure delta=0 terms
of transverse degree <=2 from Ix to form N. On a physical x-y grid with
px=py=0, evaluate

    g = (N_x)^2 + (N_y)^2 + (w*N_px)^2 + (w*N_py)^2.

At each chosen delta compute its mean mu, population standard deviation sigma
and population skewness gamma1. A constant field uses gamma1=0. Minimize

    F = max_delta sqrt(mu^2 + sigma^2 + gamma1^2).

This is the advisor's derivative/fluctuation objective, separate from both
the horizontal_shape and action_geometry objectives. No extra normalization
or gradient weighting is added to it. The original sampling convention is
retained, including its x-grid construction from two half-grids.

Defaults (all editable in general_config.py):

```python
REFERENCE_OBJECTIVE_X_RANGE = 2.5e-3
REFERENCE_OBJECTIVE_X_POINTS = 21
REFERENCE_OBJECTIVE_Y_RANGE = 0.7e-3
REFERENCE_OBJECTIVE_Y_POINTS = 11
REFERENCE_OBJECTIVE_DELTA_VALUES = (-3.4e-2,)
REFERENCE_OBJECTIVE_MOMENTUM_WEIGHT = 0.7
```

The momentum slice -0.034 is retained from the prior reference configuration;
it is not silently replaced by the action_geometry grid. Its polynomial
evaluation can be outside A_BOX, so choose the intended comparison slices
explicitly. GRADIENT_WEIGHT belongs only to horizontal_shape. Eigen and
reference_fluctuation_index can also be selected independently of each other
(the reference objective supports either LS constructor as well).

Validation covers analytic quadratic normalization, selection against the
full quadratic target, physical/stored transport similarity, analytic
derivative statistics, CS zero fluctuation, and the real dispatch through
both diagnostics and candidates. The runner was exercised with a short
optimization, plots and reports. An m=8,d=1 construction/objective check is
saved in benchmarks/eigen_reference_main_m8_d1.json.

For the supplied nonlinear strengths at m=8,d=1, the selected horizontal
eigenvalue was 1+9.55e-15, its quadratic determinant was 1, and the relative
stored transport defect ||T Ix-Ix||/||Ix|| was 3.12e-14. The full quadratic
relative distance to Sx was 0.2766: eigen normalization does not force the
quadratic part to equal Sx. The recovered advisor objective was finite
(6.0898 with the original delta=-0.034 slice). This is a construction check,
not a comparison of optimized physical stability.
