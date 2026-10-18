# Root geometry construction and delta-zero comparison

Branch: `root-geometry-0.3`, based on `action-a-box-0.3` at
`8ae1801b1afd2a9b33091569b58e538115f48a4d`.

## Install and run

```bat
python -m pip install --upgrade "fanqo[tracking] @ git+https://github.com/Franzuagd/FANQO.git@root-geometry-0.3"
python run_root_comparison.py --quick
python run_root_comparison.py
```

Download `user/run_root_comparison.py` and place it beside your existing
`general_config.py` and its lattice file. The runner uses the installed package,
not a hard-coded local FANQO checkout. From a repository checkout, alternatively
install with `python -m pip install -e ".[tracking]"` and run
`python user/run_root_comparison.py --quick`.

The default order is **m=8, d=1**, including in quick mode. All root fitting,
plots and tracking diagnostics are at **delta=0 only**. The initial vertical
coordinates are exactly zero. At Jy=0 the vertical angle is undefined; no
division by a vertical amplitude is performed.

```bat
python run_root_comparison.py --geometry-only
python run_root_comparison.py --jx 1e-6 3e-6 6e-6 1e-5 --turns 2048
python run_root_comparison.py --jx-max 1.2e-5 --root-iterations 100
```

`--geometry-only` does not build a tracking ring or import PyAT. It produces
coefficients, root tables, statistics and the algebraic action-angle plot.
`--quick` reduces angular grids and tracking turns to 128 and limits root
refinement to at most 15 iterations; it is a workflow check.

No magnet optimization, FMA, tune diffusion or legacy Poincare plots are run.
Physical phase-space overlays are included because the requested comparison
needs the real tracked (x,px) points against the algebraic curves.

## What the new constructor actually solves

The mathematical proposal specified root statistics, but did not uniquely
specify a coefficient construction. This branch implements the **root-based
refinement** selected for this experiment:

1. Construct `action_a_box` Ix from the projected transport and full action-disk
   metric. No tracked samples enter this step or the subsequent refinement.
2. Keep the full delta=0 quadratic block equal to the CS seed.
3. Search weak algebraic directions in the higher-order tail, allowing
   coupled/chromatic compensation. Rank those directions by their visible
   horizontal geometry change, and optimize the leading 12 by default.
4. For each fitting launch action J_l at phi_x=0, define the level by
   C_l=P(J_l,0), then solve P(g_l(phi),phi)=C_l on the horizontal slice.
5. Minimize the mean of `(g_l(phi)/J_l - 1)^2` over the fitting actions and
   angles, subject to the algebraic residual bound below.

The objective measures deviation from the **launch action**, not from the
numerical invariant label C_l. Thus different label normalizations are not
mistaken for changes of geometry. No endpoint-vanishing angular factor is
forced: it is an additional ansatz restriction, not a consequence of the
root proposal. Both endpoint roots are anchored by the launch level instead.

The admissible polynomial space is unchanged. Its action-angle form is
evaluated exactly as radial powers times trigonometric factors; there is no
independent nonsmooth angular ansatz. No Cartesian tracking data are used to
fit a Fourier series.

Let W.T W=G_action and R=T-I. With fixed quadratic seed s embedded in the full
basis, the relative algebraic residual is

```
e(c) = ||W R c||_2 / ||W s||_2.
e(c) <= max(ROOT_RESIDUAL_FACTOR * e(c_initial), ROOT_RESIDUAL_FLOOR).
```

Defaults are factor=1.1 and floor=1e-8. **The floor is an absolute allowance
in this normalized metric:** if the initial residual is near roundoff, the
refinement can increase it by many orders of magnitude while remaining below
1e-8. The report explicitly records initial, allowed and final residuals.
Use a smaller `--residual-floor` for a stricter experiment. This is a
constraint on a truncated algebraic map, not a bound on physical tracking drift.

SLSQP uses implicit derivatives of the selected roots. Its finite search
directions are preconditioned by an SVD of the full residual operator and an
SVD of its horizontal geometric effects, with bounded steps. The weakest 64
algebraic modes are considered by default (`ROOT_WEAK_MODES`); at most 12 visible
directions are optimized (`ROOT_DIRECTIONS`). Only an improving feasible incumbent is returned;
the candidate must also improve on a doubled angular grid. Otherwise the
initial action-domain polynomial is retained and reported as
`seed_retained`. This is a local finite-grid refinement, not a proof of a global
minimum. Increasing the iteration budget may help; it does not guarantee a
better tracking result.

Only complete seed branches are used for fitting. Excluded launch actions and
their reasons appear in `comparison.json`. Every requested plotting action is
still evaluated independently on the denser output grid, including those
outside the fitted action range.

## Roots, branches and conventions

FANQO defines **J=S, without 1/2**:

```
x  = sqrt(beta_x * J) * cos(phi_x)
px = -sqrt(J / beta_x) * (sin(phi_x) + alpha_x*cos(phi_x))
J  = gamma_x*x*x + 2*alpha_x*x*px + beta_x*px*px
phi_x = atan2(-(alpha_x*x + beta_x*px), x) mod 2*pi
```

The contour area is pi*mean(J) in this convention, rather than 2*pi*mean(J).
Canonical actions are J/2. At J=0 the angle is stored as NaN.

At each angle the code finds every distinct nonnegative real root in
u=sqrt(J/Jx_max). It continues the root from the known launch root. It stops
at absent roots, tiny radial derivatives, an orientation change, a large
unresolved angular step, or the configured inspection bound. It does not
bridge gaps or substitute zero. A numerical cutoff means outside the inspected
domain, not mathematical infinity. Roots are numerical, not certified.

Extra roots are saved separately and never mixed into a branch average.
Near-multiple roots are detected by the radial derivative, not counted as
separate branches solely because of numerical root splitting. Refine angular
resolution when a branch reports an unresolved step. A finer grid can reveal
defects hidden on the fitting grid.

For each complete output branch, the code saves mean, standard deviation,
mean absolute deviation from C, RMS deviation from C, bias, maximum action,
minimum absolute radial derivative and relative RMS deviation from launch J.
The duplicated angle 2*pi is excluded from the statistics.

## Outputs

Under `root_comparison/`:

- `algebraic_action_angle.png`: g_C(phi_x), **no tracking**.
- `tracking_action_angle.png`: real Jx versus phi_x samples only.
- `action_angle_overlay.png`: algebraic curves and real samples, per method.
- `phase_space_overlay.png`: algebraic curves and the same real (x,px) samples.
- `launch_XX_overlay.png`: separate action-angle and phase-space comparison
  for each individual initial action.
- `invariance_comparison.png`: invariant drift versus turn for every launch.
- `invariance_vs_action.png`: maximum drift versus launch action; losses marked.
- `tracking_XX.npz`: the shared raw trajectory, actions, angles, turns and loss
  status for each launch. Tracking is performed once, then reused for both Ix.
- Per method: `Ix_coefficients.npz`, branch CSVs, every-root CSVs, root
  statistics, invariant time series and tracking summaries.
- `comparison.json`: construction settings, fit diagnostics, lattice parameters,
  on-momentum optics, cell counts and validation summaries.

Conservation plots normalize `(Ix(n)-Ix(0))` by the **same launch Jx** for both
methods. Lost trajectories are retained and explicitly marked as partial data;
their apparent small excursions must not be treated as successful confinement.
Invariance lines use root_geometry=teal and a_box=red; branch colors identify
launch actions, consistently across panels. Algebraic curves are always lines;
real tracking is always points in the geometry overlays.

Full-ring tracking uses the same corrected physical strengths as construction.
The runner verifies those strengths, requires the on-momentum closed orbit at
the coordinate origin, and records the maximum tracked Jy. If a different
lattice generates Jy from a horizontal launch, the frozen Jy=0 curve is only a
slice, not the actual coupled trajectory. The invariance evaluation still uses
all tracked coordinates. Construction may represent one cell while validation
uses the physical ring; both counts are recorded.

## Ordinary API selection

```python
INVARIANT_CONSTRUCTION = 'root_geometry'
COMPUTE_IX = True
COMPUTE_IY = False
INVARIANT_OPTIONS['root_geometry'] = {
    'ROOT_ACTION_FRACTIONS': (.05, .15, .3, .5, .7),
    'ROOT_ANGLES': 64,
    'ROOT_MAX_ITER': 60,
    'ROOT_RESIDUAL_FACTOR': 1.1,
    'ROOT_RESIDUAL_FLOOR': 1e-8,
}
```

The constructor needs positive two-plane A_BOX widths and positive ACTION_BOX
bounds because its algebraic constraint is evaluated in the full action metric.
Jy=0 restricts root fitting and the launch conditions, not that metric.
Only Ix is implemented for this method; requesting Iy raises a clear error.

## Interpretation

A smaller root deviation means straighter action-angle contours. It does not
imply smaller drift under the physical map. The comparison is designed to show
that distinction, including cases where the new fit is geometrically better
but tracks worse. The procedure contains no theorem identifying the loss of a
radial graph with the actual stability boundary.

## Recorded validation on the bundled lattice

`benchmarks/root_geometry_m8_d1.json` records a full m=8,d=1 run with 64 fitting
angles, 360 output angles, and 1,024 full-ring turns. The geometric objective
decreased from 0.0143984 to 0.0125263 (about 13%). The relative algebraic residual
was 2.64e-9, below the configured 1e-8 limit. All six tracked launches survived.
However, a_box gave smaller invariant excursions at five of the six actions.
The largest root-refined output contour was incomplete and explicitly flagged;
that action lies outside the fitted range. This validates the workflow, not a
claim that root refinement improves dynamical stability.

Run `python -m pytest tests -q` for numerical regression checks.
