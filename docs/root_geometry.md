# Direct action-angle least squares and all-root comparison

Branch: `root-geometry-0.3`. The corrected constructor is `action_angle_ls`.
The earlier nonlinear root-refinement implementation was a misunderstanding
and has been removed. `root_geometry` is now a compatibility alias for the
same **linear LS** constructor. Its old benchmark and geometric-improvement
claims do not apply to this implementation.

## Install and run

```bat
python -m pip install --upgrade --force-reinstall --no-deps "fanqo @ git+https://github.com/Franzuagd/FANQO.git@root-geometry-0.3"
python run_root_comparison.py --quick
python run_root_comparison.py
```

For a new environment, install with the tracking extra instead:

```bat
python -m pip install "fanqo[tracking] @ git+https://github.com/Franzuagd/FANQO.git@root-geometry-0.3"
```

Download `user/run_root_comparison.py` and put it beside your existing
`general_config.py` and its lattice file. The runner uses the installed package.
From a checkout: `python -m pip install -e ".[tracking]"`, then
`python user/run_root_comparison.py --quick`.

Default: **m=8, d=1**, all diagnostics at **delta=0**. Quick mode leaves m,d
unchanged, reduces the root grid to 96 intervals, horizontal tracking to 128
turns and the invariance map to 9*9 launches and 64 turns.

```bat
python run_root_comparison.py --geometry-only
python run_root_comparison.py --levels 1e-6 3e-6 6e-6 1e-5
python run_root_comparison.py --jx 1e-6 3e-6 6e-6 --turns 2048
python run_root_comparison.py --map-grid 60 60 --map-turns 512
```

`--levels` sets literal invariant levels C for the combined all-root plot.
`--jx` sets horizontal launch actions for independent tracking validation.
When levels are omitted, the launch action values are also used as numerical
C levels. A launch action and its invariant value are not generally equal.
Individual overlays use C=P(Jx0,phi_x=0) separately for each method so the
predicted contour passes through the actual launch.

## Basis and constructor

FANQO uses J=S, **without the conventional factor 1/2**:

```
J = gamma*q^2 + 2*alpha*q*p + beta*p^2
q = sqrt(beta*J)*cos(phi)
p = -sqrt(J/beta)*(sin(phi)+alpha*cos(phi))
```

The explicit real radial-Fourier basis consists of

```
B = (delta/dmax)^d * (Jx/Jxmax)^(i/2) * (Jy/Jymax)^(j/2)
    * F_n(phi_x) * F_m(phi_y),
```

where F is sine or cosine. The zero harmonic has only cosine. Smooth
polynomial compatibility requires n<=i, i-n even, m<=j, j-m even.
Transverse degree is i+j<=ORDER; delta degree is handled independently.
All supported terms of the existing polynomial space are retained, including
its fixed low-degree seed block. No arbitrary nonsmooth radial functions,
nonlinear root optimization or tracking-fitted coefficients are introduced.

Let D map radial-Fourier coefficients a into stored polynomial coefficients c:

```
c = D a
T_J = inverse(D) T D
R_J = T_J - I.
```

The code constructs D explicitly from the Twiss parameters and homogeneous
radial harmonics. It obtains T_J by solving the degree blocks of this basis
change. The operator is equivalent to the existing polynomial transport,
expressed in the new basis. The saved files contain D, T_J and the actual
radial-Fourier coefficient vector a.

The fitting measure is uniform in Jx/Jxmax and Jy/Jymax on [0,1], both angles
on [0,2*pi), and delta/dmax on [-1,1]. Its Gram matrix G_J is analytic:
matching nonzero sine/cosine harmonics contribute 1/2, a matching zero harmonic
contributes 1, differing harmonics integrate to zero, radial moments are
1/(1+(i+i')/2), and delta moments are zero for odd degree or 1/(1+d+d') otherwise.

Writing a=(a_seed,h), with the quadratic part fixed to Jx, solve once:

```
min_h || W_J R_J (a_seed,h) ||_2,
W_J.T W_J = G_J.
```

This is linear SVD least squares with the existing LEAST_SQUARES_TOL. Root
values, root variance, branch existence and tracked samples do not enter the
solve. The ROOT_* refinement settings from the superseded implementation
are no longer used.

The admissible smooth polynomial space is equivalent to the Cartesian one.
Changing basis alone cannot improve the exact attainable residual. The action
metric, numerical conditioning, singular-value cutoff and minimum-norm choice
inside a nonunique solution set can change the computed polynomial. This
method is not mathematically guaranteed to track better than a_box.

## Unchanged a_box baseline

The runner loads the ordinary a_box state and calls the original weighted
Cartesian LS constructor without changing its A_BOX, metric, coefficient
normalization, quadratic constraints or rank threshold. That state is not
mutated when preparing the action-angle state. Both use the same physical
lattice and polynomial transport. Tests check the baseline independently.

## All roots: no branch selection

At fixed delta=0 and Jy=0, for every specified C and every sampled angle,
solve the univariate polynomial in u=sqrt(Jx/Jxmax):

```
P(Jxmax*u^2,phi_x) - C = 0, u>=0.
```

Every distinct finite nonnegative real u is retained and mapped to Jx. There
is no nearest-root filter, slope-based rejection, continuation rule or radial
cutoff. A missing root is not replaced by zero. Identically zero polynomials
are flagged as degenerate rather than represented by a finite root list.
Roots and repeated-root classification remain floating-point computations.

All requested C levels appear together, colored by level. The top panels are
explicit amplitude zooms; the bottom panels use a symmetric-log action scale
and show **all computed finite roots**, including remote components. Both
endpoints are included and axis ticks explicitly show 0, pi/2, pi, 3*pi/2, 2*pi.
The 2*pi endpoint uses exactly the same polynomial coefficients as 0.

All roots are also saved to CSV, along with per-angle counts. Roots belonging
to different components are not averaged into a single claimed contour.
Horizontal tracking starts with Jy=0; phi_y is undefined there, not evaluated
by dividing by a zero amplitude. These root cuts do not restrict the full
2-plane construction or the standard x-y launch grid of the invariance map.

## Legacy tracked-invariance maps: red versus blue

After both constructors and algebraic root diagrams are complete, track the
physical full ring at delta=0 on the configured FMA_COORDS_MM range. MAP_STEPS
in the runner controls the grid intervals; the default is 31*31 launches.
The same trajectories are evaluated for both invariants.

This restores the old metric:

```
normalizer = max(abs(Ix(0)), floor_fraction*max_grid(abs(Ix(0))))
drift_rate = max_n abs(Ix(n)-Ix(0)) / (normalizer*n)
```

The floor is computed separately for each method, matching the legacy API.
This is not the common launch-J normalization used for the optional horizontal
time series. Lost particles are marked gray and excluded from drift ratios.

The signed map uses

```
log10(drift_a_box / drift_action_angle_ls)
```

**Red/negative: a_box better. Blue/positive: action-angle LS better.**
Individual invariance maps share the configured logarithmic color range.
This uses physical tracking on the old FMA launch grid; no tune estimation,
frequency diffusion or separate Poincare diagrams are computed.

## Outputs

Under `action_angle_comparison/`:

- `all_roots_action_angle.png` / `algebraic_action_angle.png`: every C level
  and every nonnegative root over the complete 0..2*pi range, with zoom and
  full-range panels; no tracking used.
- `invariance_comparison.png`: the two tracked-invariance maps, common scale.
- `invariance_difference.png` and CSV: the red/blue legacy comparison.
- `invariance_time_series.png`: horizontal launch conservation, blue=new,
  red=a_box, using the same launch Jx denominator for both methods.
- `launch_XX_overlay.png`: all-root action-angle and physical phase-space
  zooms, plus real tracked points. Complete roots remain in its CSVs.
- Per method: stored polynomial coefficients, all-root CSVs and counts,
  tracked-invariance grid CSV, launch-specific root CSVs and invariant series.
- `action_angle_ls/radial_fourier_coefficients.npz`: actual radial-Fourier
  coefficients, basis labels, D, G_J and T_J. Label columns are
  (delta_degree,i,j,n,kind_x,m,kind_y).
- `invariance_map_tracking.npz`: shared grid coordinates and raw trajectories,
  padded with NaN after losses; numeric arrays, no pickle required.
- `tracking_XX.npz`: horizontal launch trajectories and action-angle samples.
- `comparison.json`: settings, algebraic solve details, corrected strengths,
  optics, cell counts and tracking summaries.

`--geometry-only` skips every tracking operation and does not require PyAT.
Use a fresh output folder when comparing against older runs; obsolete files
in an existing folder are not removed automatically.

## Public API

```
INVARIANT_CONSTRUCTION = 'action_angle_ls'
COMPUTE_IX = True
COMPUTE_IY = False
ACTION_BOX = {'Jx_max':1.75e-5,'Jy_max':1.83e-5,'delta_max':.01}
```

Positive full 2-plane A_BOX and ACTION_BOX bounds are required. The analogous
Iy construction is also supported. The root plotting runner constructs Ix only.

Run `python -m pytest tests -q`. Checks include direct trigonometric basis
values, analytic Gram versus independent disk moments, transport similarity,
LS versus independent action-angle quadrature, unchanged a_box, all-root
retention and the legacy drift metric.

`benchmarks/action_angle_ls_m8_d1.json` records the corrected m=8,d=1
validation: 256 horizontal turns, 441 FMA-grid launches for 128 turns, and a
360-interval all-root diagram. The earlier root-refinement benchmark was
removed because it tested the wrong construction.
