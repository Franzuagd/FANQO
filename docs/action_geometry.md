# Action-angle LS and geometric objective

Branch `action-geometry-main-0.3` starts directly from main at `8f3bcc2`.
It adds the radial--Fourier LS construction and the geometric objective without
changing main's CMA/Powell loop, transfer-map calculation or magnet updates.
Old configurations continue to use the original a_box construction and
horizontal shape objective. The old defaults in `user/general_config.py`
remain selected.

## Select the new calculation

Change only these selections in your general configuration:

```python
INVARIANT_CONSTRUCTION = 'action_angle_ls'
OBJECTIVE = 'action_geometry'
```

Then use the existing public calls:

```python
import fanqo as fq

fq.load('general_config.py')
fq.compute_invariants()
result = fq.optimize(run_start_end_fma=False)
print(result['final_details']['objective']) # Loss: smaller is better.
print(result['final_details']['quality'])   # Quality: 1 is the CS line.
print(result['final_details']['geometry']['status'])
```

No FMA/tracking is used to construct the invariant or evaluate the objective.
The existing optional start/end FMA behavior is unchanged; pass False as above
when you want to run without it. Existing plots and reports still work.
No new runner or optimizer is required.

The third [eigen reference](eigen_reference.md) uses the recovered advisor
objective and has its own `run_eigen.py` runner.

Two optional standalone runners preserve the original `user/run.py` sequence:

```bash
python run_a_box.py
python run_action_geometry.py
```

Put both beside the same `general_config.py` and its selected lattice file.
They require the package from this branch. Each reloads the initial experiment
and selects its own construction/objective in memory, without editing your
configuration. `run_a_box.py` selects a_box + horizontal_shape;
`run_action_geometry.py` selects action_angle_ls + action_geometry. The latter
provides the grid/domain defaults below if an older main config lacks them.
Explicit settings in your config take precedence over those grid defaults.

Both use `quick=False`: CMA_TIME and POWELL_TIME_FRACTION remain exactly as
configured. For example CMA_TIME=60 means 60 seconds of CMA budget, not 12
hours. Start/end FMA follows RUN_FMA_START_END when PyAT is available; as in
the original runner, missing PyAT skips tracking with a printed notice.
Optics, invariant checks, initial/final invariant plots, optimizer slice plots,
reports and final lattice saving follow main's original workflow. Outputs
are grouped under OUTPUT_ROOT/a_box and OUTPUT_ROOT/action_geometry, with
separate initial/final plot folders. Each script can be downloaded and run
independently; neither requires the other script or a shared helper.

## Grid and settings

The supplied config contains:

```python
ACTION_BOX = {'Jx_max': 4.3645e-6, 'Jy_max': 1.83e-5, 'delta_max': 0.01}
ACTION_GEOMETRY_POINTS = np.array([
    (dd, jy, fy, fraction * ACTION_BOX['Jx_max'])
    for dd in (-0.005, 0.0, 0.005)
    for jy, fy in ((0.0, 0.0), (0.1 * ACTION_BOX['Jy_max'], 0.0),
                   (0.1 * ACTION_BOX['Jy_max'], np.pi/2))
    for fraction in (0.1, 0.3, 0.5)
])
ACTION_GEOMETRY_OPTIONS = dict(
    angles=64, amplitude_scale=0.5, slope_scale=1.0, derivative_weight=0.1,
)
```

Rows are `(delta,Jy,phi_y,Jx0)`. Angles are radians. FANQO uses J=S, twice
canonical action. Choose strictly positive Jx0 and nonnegative Jy. Jy and
phi_y are fixed on each slice; they are not evolved along an orbit. Use
`ORDER` and `DELTA_ORDER` to change m and d, just as in main. `ACTION_BOX`
sets the LS integration domain; the points set the geometric evaluation grid.
They are different choices. Optional `weights` in ACTION_GEOMETRY_OPTIONS
are positive per-row weights. GRADIENT_WEIGHT still belongs only to the old
horizontal_shape objective; use derivative_weight for action_geometry.

At every candidate, coefficients are obtained by linear LS in the explicit
smooth radial--Fourier basis. Its degree/parity restrictions span the same
stored polynomial space. The transport is changed by exact similarity, and
the analytic metric is uniform in Jx,Jy, both phases and delta. Both seeds are
the Courant--Snyder quadratic invariants at delta=0. Diagnostics construct
both Ix and Iy as main did; candidate scoring constructs only Ix. The
horizontal score always uses Ix radial coefficients, including after an Iy
solve. Optics changes refresh the metric/basis, and rejected candidates
restore them along with the original context.

## Score

For each row, C=P(Jx0,0,Jy,phi_y;delta). At N equally spaced phi_x in [0,2pi),
solve for the regular radial graph through the launch. Set

```
u_k = Jx_k/Jx0
d_k = -P_phi/(Jx0*P_J)
A = max_k abs(u_k-1)
B = mean_k d_k**2/(slope_scale**2+d_k**2)
Q = exp(-(A/amplitude_scale)**2 - derivative_weight*B)
loss = 1 - weighted_mean(Q)
```

Courant--Snyder gives Q=1; increasingly large action excursions drive Q to
zero. The angular derivative is analytic, not a finite difference. The
repeated 2pi endpoint is excluded from the average.

The continuation follows a regular radial graph, not an arbitrary folded
component. Disconnected roots do not contribute. Missing, folded or
numerically unresolved graphs receive zero with an explicit status; zero
does not certify physical instability or unboundedness. Finite sampling can
miss narrow excursions or branch switches. Check angular convergence and
the returned status before interpreting a score. No action cutoff is used.
These geometric scales are adjustable choices, not established physical
stability thresholds.

## Validation

Tests compare analytic roots/derivatives, independently sampled LS residuals,
basis similarity, unchanged legacy outputs, candidate rollback, optics refresh,
Ix/Iy separation and a short run through the existing optimize API. Tracking
is prohibited in the integration tests. An m=8,d=1 end-to-end measurement is
stored in `benchmarks/action_geometry_main_m8_d1.json`.

For the supplied lattice parameters, the 27-point/64-angle objective took
0.58 seconds. With main's existing cached-map recomposition included, building
Ix and scoring it took 12.38 seconds versus 12.61 seconds to build Ix with
the original a_box path (three repetitions, median, one BLAS thread). Treat
these as comparable runtimes, not evidence of a speed improvement. The
transport matrices agreed between both paths. Full diagnostics, which also
construct Iy, reproduced the same horizontal loss, 0.7475613144. Twelve of
the 27 graphs resolved; the others received conservative zero scores.
