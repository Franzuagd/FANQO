"""Real radial roots and algebraic continuation; no particle tracking."""
import numpy as np


def positive_roots(coefficients, *, imaginary_tolerance=1e-7):
    """All distinct nonnegative u roots, scaled polynomial in ascending order.

    Exact zero leading coefficients are removed; small nonzero ones are kept
    so remote branches are not silently discarded. None means identically zero.
    """
    c = np.asarray(coefficients, float)
    if not np.isfinite(c).all():
        return np.empty(0)
    nz = np.flatnonzero(c != 0)
    if not len(nz):
        return None
    c = c[:nz[-1]+1] / np.max(np.abs(c))
    roots = np.polynomial.polynomial.polyroots(c)
    real = roots.real[np.abs(roots.imag) <= imaginary_tolerance*np.maximum(1, np.abs(roots.real))]
    real = np.sort(real[real >= -imaginary_tolerance])
    unique = []
    for r in np.maximum(real, 0):
        if not unique or abs(r-unique[-1]) > 1e-6*max(1, abs(r)):
            unique.append(float(r))
    return np.asarray(unique)


def trace_radial_branch(radial, launch_action, action_scale, *, slope_floor=1e-7,
                        max_action=None, max_step_fraction=.5):
    """Continue roots from the known launch root at the first angle.

    Angles must include the repeated endpoint (normally 0 and 2*pi). Missing,
    degenerate, ill-conditioned or discontinuous steps terminate the branch;
    they are never bridged. This is algebraic continuation, not tracking.
    All root lists remain available even when continuation terminates.
    """
    radial = np.asarray(radial, float) / action_scale
    u0 = np.sqrt(launch_action/action_scale)
    level = np.polynomial.polynomial.polyval(u0, radial[0])
    polynomials = radial.copy(); polynomials[:, 0] -= level
    all_roots = [positive_roots(c) for c in polynomials]
    values = np.full(len(radial), np.nan)
    slopes = values.copy(); previous = u0
    status = 'complete'
    for a, roots in enumerate(all_roots):
        if roots is None:
            status = 'degenerate_polynomial'; break
        if len(roots) == 0:
            status = 'missing_positive_root'; break
        r = float(roots[np.argmin(abs(roots-previous))])
        derivative = np.polynomial.polynomial.polyval(
            r, np.polynomial.polynomial.polyder(radial[a]))
        slope = derivative/(2*r) if r > 0 else np.nan
        J = action_scale*r*r
        if a == 0 and not np.isclose(r, u0, rtol=1e-5, atol=1e-8):
            status = 'launch_root_not_resolved'; break
        if not np.isfinite(slope) or abs(slope) <= slope_floor:
            status = 'multiple_or_ill_conditioned_root'; break
        if a and np.sign(slope) != np.sign(slopes[a-1]):
            status = 'branch_orientation_changed'; break
        if a and abs(J-values[a-1]) > max_step_fraction*max(launch_action, values[a-1]):
            status = 'unresolved_branch_step'; break
        if max_action is not None and J > max_action:
            status = 'outside_search_domain'; break
        values[a] = J; slopes[a] = slope; previous = r
    if status == 'complete' and not np.isclose(values[-1], launch_action, rtol=1e-5, atol=1e-12*action_scale):
        status = 'not_closed'
    complete = status == 'complete'
    # The endpoint duplicates the first angle and must not be counted twice.
    g = values[:-1]
    stats = {}
    if complete:
        mean = float(np.mean(g)); variance = float(np.mean((g-mean)**2))
        C = level*action_scale
        stats = dict(mean=mean, std=np.sqrt(variance), bias=mean-C,
                     mean_absolute_deviation=float(np.mean(abs(g-C))),
                     rms_deviation=float(np.sqrt(np.mean((g-C)**2))),
                     relative_launch_rms=float(np.sqrt(np.mean((g/launch_action-1)**2))),
                     max_action=float(np.max(g)), min_abs_slope=float(np.min(abs(slopes))))
    return dict(action=values, slopes=slopes, level=float(level*action_scale),
                all_roots=[None if r is None else action_scale*r*r for r in all_roots],
                complete=complete, status=status, statistics=stats)
