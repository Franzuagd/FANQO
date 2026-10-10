# FANQO physics, mathematics and code audit

Scope: the source modules, configuration and runners on `IxOptCompare-0.3`, starting at commit cee0a2985b790363aef7419289903d2d1f7956b7, plus the supplied OPA lattice. This is a targeted scientific/code review with executable checks, not a certification that every possible configuration or optimizer run is correct. Other Git branches were not exhaustively audited. The original OPA source was not supplied here, so its conversion factors cannot be independently certified.

## Confirmed defects fixed

1. **Thin octupole polynomial transport used only I+ML.** An integrated coordinate kick is cubic, but applying it to polynomial observables produces higher powers. On y=0, inverse pullback gives `(px+O*x^3)^2 = px^2+2*O*px*x^3+O^2*x^6`. The old matrix omitted the last term at m>=6. The corrected finite exponential retains it. Test with O=2: expected x^6 coefficient 4; old result 0; corrected coefficient error 5.6e-16. This affects this OPA lattice because O1/O2/O3 are nonzero. No artificial element length was introduced.

2. **Forced config reload could reuse stale bytecode.** Rewriting `a_box` to `eigen` can preserve both file size and timestamp resolution. Removing the module from sys.modules does not invalidate Python's timestamp/size bytecode cache. Explicit reload now compiles the current source bytes. Regression test deliberately preserves timestamp and size.

3. **Zero nonlinear fluctuation was rejected.** `scipy.stats.skew` returns NaN for a constant sampled field, so the reference fluctuation objective returned infinity even for zero nonlinear correction. Zero sample variance now contributes skewness zero. This is an objective correctness fix, not an assertion that this objective guarantees physical stability.

## Quantitative checks that passed

For the supplied OPA cell after the configured native chromatic correction:

- Cell length 24.189447305625198 m; net bend 18 degrees; 20 repeated cells form the tracking ring.
- Native 4D linear symplectic defect: 7.95e-15.
- Native versus PyAT linear-matrix relative difference: 1.00e-7 at 40 integration slices.
- Native beta values: (5.7280091281, 3.4984147306); PyAT: (5.7280085853, 3.4984145161).
- Native/PyAT thin-octupole kick convention agrees in a direct coordinate test.
- Existing algebraic tests cover physical bracket evaluation, shared-Lie involution, conjugated integrable maps, and the nontriangular fallback.

## Important unresolved mathematical/model limitations

**Rectangular truncation is not a closed jet algebra under dispersion.** A Hamiltonian term delta*x lowers transverse degree in a Poisson bracket while raising delta degree. A term discarded at transverse degree m+1 can feed a retained delta-dependent degree-m term later. An independent horizontal test increases workspace from m=4 to m=5 with d=1; the retained delta*x^2*px^2 coefficient in transported px^4 changes by 1.0471975512. This is not a floating-point effect. A consistent enlarged total-degree workspace followed by projection is a substantial next correction; simply removing the triangularity check does not repair it. Current new fallback fits the supplied matrix, including its modeling limitations.

**d=1 is first order in momentum offset.** The Hamiltonian supplied contains delta^2, but the basis drops it when d=1. Tests at +/-0.02 and +/-0.03 are also outside the default delta box +/-0.01. Such results must be labeled extrapolation, not guaranteed accuracy. The candidate evaluation here intentionally includes them because requested.

**Native zero chromaticity is not exact zero PyAT chromaticity.** Native residuals are about zero; PyAT gives (-0.00272975,-0.00087396) per cell. Native correction uses sampled Twiss/dispersion integrals and the simplified model, not the exact derivative of PyAT tunes. The corrected strengths differ slightly from the uploaded values: SF1 26.92306 -> 26.98998, SD1 -108.60513 -> -108.78588. Comparisons use the same corrected strengths for every method. This discrepancy is small but real.

**Near-unit eigenspaces have large nonuniqueness.** Current eigen selection retains strictly positive real eigenvalues of T-I within an adaptive window and ranks one residual component (xp_x). It can exclude exactly zero or slightly negative numerical eigenvalues. The minimum of one component is not a minimum of the full residual or of tracking error. An eigenvector of a repeated/near-repeated eigenvalue is sensitive to roundoff, basis and library implementation. Normalizing the horizontal quadratic determinant does not fix vertical-action mixing, nonlinear action reparametrization or pure-delta offsets.

**The expensive coupled objective mixes dimensional scales.** Its homological/bracket terms are raw squared box norms while nonlinear-tail and quadratic-mixing penalties are relative. Defaults therefore need not balance invariance and regularization; small physical actions can make regularization dominate. At m=8 the two baseline L-BFGS-B runs reached their iteration limit in this experiment. Their results are valid finite iterates, not demonstrated global optima. The runner logs convergence and the comparison must not conceal failure to converge.

**One cell versus full ring is a modeling choice.** An exact invariant of an identical cell is an invariant of its repetitions. A quasi-invariant has accumulating defects; a cell coefficient residual does not equal full-ring multi-turn drift. The production tracking builder re-solves native chromatic correction for the full ring. This is essentially consistent for repeated identical cells and zero chromaticity targets, but not automatically equivalent for arbitrary nonzero cell/ring targets.

## Metrics and objectives

- `h_reduction` is explicitly representation-dependent: eigen's unit-box Gram matrix and a_box's scaled Gram matrix are not the same physical norm. Its values and optimization objectives should not be interpreted as identical across methods without converting to a common physical metric.
- The original `max |I(k)-I(0)| / (k |I(0)|)` metric divides by a method-dependent quantity and emphasizes early drift. Adding a conserved pure-delta function changes this score without changing the numerator. A common CS-action denominator and maximum excursion are therefore reported alongside the original score.
- A zero bracket is involution, not invariance under the ring map. The canonical constructor also excludes dispersion-linear generators, so it is more constrained off momentum.
- Closed contours on a frozen 2D slice do not prove a 4D invariant barrier or dynamic aperture. Coupled trajectories need not remain on that slice; existing leakage diagnostics are relevant.
- Gradient norms combining x in metres and normalized momentum require an explicit coordinate metric/scaling. The geometric objectives use heuristic Euclidean combinations; they are not coordinate-invariant physical distances.
- The fluctuation objective combines dimensional mean/std terms with dimensionless skewness. It is a heuristic optimization score. The convex tracking regression directly targets an empirical invariance error, but can overfit or flatten an observable unless normalization and independent validation are imposed.

## Code and workflow limitations

- Linear `multipole` maps are identity, and the zero-length nonlinear path specializes to octupoles, while the PyAT converter accepts general K/S/O multipoles. The uploaded lattice only uses thin octupoles, where the test passes. Arbitrary thin quadrupoles/sextupoles or thick general multipoles need additional native support or explicit rejection.
- `load()` resets active state before construction finishes; a failed load can leave no active context. Optimizer candidate evaluation has snapshot/rollback handling, which is different.
- `current_lattice()` exposes mutable internal objects. Direct edits can bypass parameter dependency bookkeeping. Use `set_parameters()`.
- The optimizer time budget is soft: one evaluation may finish beyond it. The code documents this. Tests here do not certify long CMA/Powell optimization campaigns.
- Nonlinear solver status must be read; successful return of coefficients does not imply convergence. The full-residual fallback can be expensive at high order.
- The research launch convention is centered on each off-momentum closed orbit and includes random betatron phases. Production FMA starts from the on-momentum orbit with a delta offset. Both are legitimate but different test distributions. They must not be compared as identical plots.

Executable evidence: `audit_physics.py`, `audit_truncation.py`, `tests/test_audit_regressions.py`, and the supplied comparison runners. Read the numerical summary for method rankings and limitations; no universal winner is claimed.
