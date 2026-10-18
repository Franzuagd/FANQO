"""Delta-zero, horizontal root geometry versus a_box; tracking validates only.

No FMA, tune analysis, old Poincare plotting calls or magnet optimization.
Both invariants use one polynomial transport and exactly the same trajectories.
"""
import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from . import api, config_loader
from .comparison import _json
from .core import action_a_box, nonlinear as nl, optimization as opt
from .core import root_geometry as roots
from .plotting import get_pyplot

METHODS = ('root_geometry', 'a_box')
COLORS = {'root_geometry': '#008577', 'a_box': '#c53b36'}


def _write_json(path, value):
    # Invalid roots are NaN in numerical arrays, but JSON must stay standard.
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, np.ndarray): return clean(v.tolist())
        if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.bool_): return bool(v)
        return v
    path.write_text(json.dumps(clean(_json(value)), indent=2, allow_nan=False), encoding='utf-8')


def _save_geometry(out, method, curves, actions, angles):
    folder = out/method; folder.mkdir(exist_ok=True)
    for i, (J, curve) in enumerate(zip(actions, curves)):
        counts = [-1 if r is None else len(r) for r in curve['all_roots']]
        np.savetxt(folder/f'roots_{i:02d}.csv',
                   np.column_stack((angles, curve['action'], curve['slopes'], counts)),
                   delimiter=',', comments='', header='phi_x_rad,g_C_Jx,dP_dJx,distinct_nonnegative_root_count')
        with (folder/f'all_roots_{i:02d}.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(['phi_x_rad', 'root_number', 'Jx'])
            for phi, row in zip(angles, curve['all_roots']):
                if row is not None:
                    for k, value in enumerate(row): w.writerow([phi, k, value])
    _write_json(folder/'root_statistics.json', [
        dict(launch_Jx=float(J), level=c['level'], complete=c['complete'],
             status=c['status'], **c['statistics']) for J, c in zip(actions, curves)])


def _geometry_plots(out, states, curves, actions, angles, trajectories=None):
    plt = get_pyplot(False)
    palette = plt.get_cmap('viridis')(np.linspace(.1, .9, len(actions)))
    for tracked in (False, True) if trajectories is not None else (False,):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
        for ax, method in zip(axes, METHODS):
            for i, (J, curve) in enumerate(zip(actions, curves[method])):
                label = f'J0={J:.3g}' + ('' if curve['complete'] else f" [{curve['status']}]")
                ax.plot(angles, curve['action']*1e6, color=palette[i], lw=1.4, label=label)
                if tracked:
                    cs = states[method]['linear_cs0']
                    j, phi = roots.to_action_angle(trajectories[i][0], trajectories[i][1], cs[0], cs[1])
                    ax.scatter(phi, j*1e6, s=3, color=palette[i], alpha=.35, rasterized=True)
            ax.set(title=method, xlabel=r'$\phi_x$ [rad]', xlim=(0, 2*np.pi))
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        axes[0].set_ylabel(r'$J_x=S_x$ [$10^{-6}$ m]')
        fig.suptitle('Algebraic curves only; delta=0, Jy=0' if not tracked else
                     'Lines: algebraic roots; dots: real tracking; delta=0')
        fig.savefig(out/('action_angle_overlay.png' if tracked else 'algebraic_action_angle.png'), dpi=180)
        plt.close(fig)
    if trajectories is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True, constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        cs = states[method]['linear_cs0']
        for i, (J, curve) in enumerate(zip(actions, curves[method])):
            q, p = roots.from_action_angle(curve['action'], angles, cs[0], cs[1])
            ax.plot(q*1e3, p*1e3, color=palette[i], lw=1.4, label=f'J0={J:.3g}')
            ax.scatter(trajectories[i][0]*1e3, trajectories[i][1]*1e3,
                       s=3, color=palette[i], alpha=.35, rasterized=True)
        ax.set(title=method, xlabel='x [mm]'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    axes[0].set_ylabel('px [mrad]')
    fig.suptitle('Phase space: algebraic level curves (lines), real tracking (dots); delta=0')
    fig.savefig(out/'phase_space_overlay.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    cs = states['root_geometry']['linear_cs0']
    for i, (J, trajectory) in enumerate(zip(actions, trajectories)):
        j, phi = roots.to_action_angle(trajectory[0], trajectory[1], cs[0], cs[1])
        ax.scatter(phi, j*1e6, s=3, color=palette[i], label=f'J0={J:.3g}', rasterized=True)
    ax.set(xlabel=r'$\phi_x$ [rad]', ylabel=r'$J_x=S_x$ [$10^{-6}$ m]',
           title='Real tracking only; delta=0, initial Jy=0', xlim=(0, 2*np.pi))
    ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(out/'tracking_action_angle.png', dpi=180); plt.close(fig)

    # Individual launches remain readable even when many contours overlap.
    for i, J in enumerate(actions):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for column, method in enumerate(METHODS):
            cs = states[method]['linear_cs0']; curve = curves[method][i]
            z = trajectories[i]
            jt, phit = roots.to_action_angle(z[0], z[1], cs[0], cs[1])
            axes[0, column].plot(angles, curve['action']*1e6,
                                  color=COLORS[method], label='algebraic roots')
            axes[0, column].scatter(phit, jt*1e6, s=4, color='black', alpha=.3, label='real tracking')
            axes[0, column].set(title=f'{method}: {curve["status"]}', xlabel=r'$\phi_x$ [rad]',
                                ylabel=r'$J_x$ [$10^{-6}$ m]', xlim=(0, 2*np.pi))
            q, p = roots.from_action_angle(curve['action'], angles, cs[0], cs[1])
            axes[1, column].plot(q*1e3, p*1e3, color=COLORS[method], label='algebraic roots')
            axes[1, column].scatter(z[0]*1e3, z[1]*1e3, s=4, color='black', alpha=.3, label='real tracking')
            axes[1, column].set(xlabel='x [mm]', ylabel='px [mrad]')
        for ax in axes.ravel(): ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle(f'Launch Jx={J:.6g}, Jy=0, phi_x=0; delta=0')
        fig.savefig(out/f'launch_{i:02d}_overlay.png', dpi=180); plt.close(fig)


def _tracking_outputs(out, invariants, states, actions, trajectories, metadata):
    plt = get_pyplot(False)
    fig, axes = plt.subplots(len(actions), 1, figsize=(10, max(4, 2.2*len(actions))),
                             squeeze=False, constrained_layout=True)
    summary = []
    for method in METHODS:
        evaluate = api._make_ix_callable(invariants[method], states[method])
        method_rows = []
        for i, (J, z, meta) in enumerate(zip(actions, trajectories, metadata)):
            values = evaluate(z[4], z[0], z[2], z[1], z[3])
            drift = (values-values[0])/J  # common, externally fixed denominator
            turns = np.arange(z.shape[1])
            axes[i, 0].plot(turns, drift, color=COLORS[method], label=method, lw=.9)
            np.savetxt(out/method/f'invariance_{i:02d}.csv',
                np.column_stack((turns, values, drift)), delimiter=',', comments='',
                header='ring_turn,Ix,(Ix_minus_Ix_initial)_over_Jx_initial')
            row = dict(method=method, launch_Jx=float(J), level=float(values[0]),
                       max_relative_excursion=float(np.max(abs(drift))),
                       rms_relative_excursion=float(np.sqrt(np.mean(drift**2))), **meta)
            summary.append(row); method_rows.append(row)
        _write_json(out/method/'tracking_summary.json', method_rows)
    for i, (J, meta) in enumerate(zip(actions, metadata)):
        axes[i, 0].set(title=f'Jx0={J:.4g}; completed={meta["completed_turns"]}; survived={meta["survived"]}',
                       ylabel=r'$[I_x(n)-I_x(0)]/J_{x0}$')
        axes[i, 0].grid(alpha=.2); axes[i, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel('Full-ring turn')
    fig.suptitle('Invariant conservation on identical real trajectories; delta=0')
    fig.savefig(out/'invariance_comparison.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in METHODS:
        rows = [row for row in summary if row['method'] == method]
        ok = [row for row in rows if row['survived']]
        lost = [row for row in rows if not row['survived']]
        if ok:
            ax.semilogy([r['launch_Jx']*1e6 for r in ok],
                        [max(r['max_relative_excursion'], 1e-16) for r in ok],
                        'o-', color=COLORS[method], label=method)
        if lost:
            ax.semilogy([r['launch_Jx']*1e6 for r in lost],
                        [max(r['max_relative_excursion'], 1e-16) for r in lost],
                        'x', color=COLORS[method], label=method+' lost (partial data)')
    ax.set(xlabel=r'Launch $J_x=S_x$ [$10^{-6}$ m]', ylabel='Max relative invariant excursion',
           title='Delta=0; common Jx0 normalization; lost trajectories marked')
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/'invariance_vs_action.png', dpi=180); plt.close(fig)
    return summary


def run_root_comparison(config, *, output='root_comparison', order=8, delta_order=1,
                        action_box=None, launch_actions=None, action_fractions=(.05,.15,.3,.5,.7,.9),
                        angles=360, turns=1024, root_options=None, geometry_only=False):
    """Construct first without tracking, save roots, then optionally validate."""
    if order < 3 or delta_order not in (0, 1) or angles < 8 or turns < 1:
        raise ValueError('Require order>=3, delta_order in {0,1}, angles>=8, turns>=1.')
    config = Path(config).expanduser().resolve()
    cfg = config_loader.load_general_config(config, reload=True)
    if action_box is None:
        action_box = dict(getattr(cfg, 'ACTION_BOX', {}))
    if not action_box:
        raise ValueError('Set ACTION_BOX or supply action_box.')
    scale = float(action_box['Jx_max'])
    actions = np.asarray(launch_actions if launch_actions is not None
                         else scale*np.asarray(action_fractions), float)
    if actions.ndim != 1 or not len(actions) or not np.isfinite(actions).all() or np.any(actions <= 0):
        raise ValueError('Launch Jx values must be a nonempty positive finite list.')
    out = Path(output).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    options = dict(ACTION_BOX=action_box, **(root_options or {}))
    report = dict(delta=0., initial_Jy=0., initial_phi_y=None, launch_phi_x=0.,
                  order=order, delta_order=delta_order, launch_actions=actions,
                  action_box=action_box, root_options=options, config=str(config),
                  tracking_used_in_construction=False, methods={})
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', prefix='_root_compare_',
                                     dir=config.parent, delete=False) as handle:
        runtime = Path(handle.name)
    try:
        overrides = f'''\nINVARIANT_CONSTRUCTION = 'a_box'
ORDER = {int(order)}
DELTA_ORDER = {int(delta_order)}
COMPUTE_IX = True
COMPUTE_IY = False
RUN_FMA_START_END = False
SHOW_PLOTS = False
FMA_CORRECT_CHROMATICITY = CORRECT_CHROMATICITY
FMA_PHYSICAL_RING_CELLS = globals().get('FMA_PHYSICAL_RING_CELLS', None)
FMA_NUM_INT_STEPS = globals().get('FMA_NUM_INT_STEPS', 10)
FMA_CASE_LABEL = 'root_geometry_delta_zero'
'''
        runtime.write_text(config.read_text(encoding='utf-8')+overrides, encoding='utf-8')
        print(f'Building polynomial state and map: m={order}, d={delta_order}', flush=True)
        start = time.perf_counter(); context = api.load(runtime, force=True)
        base = context['state']
        transfer, tnn, tnq = nl.nonlinear_transfer(context['lattice'], base,
                                                   tol=cfg.LEAST_SQUARES_TOL)
        report['setup_and_transport_seconds'] = time.perf_counter()-start
        sx, _ = nl.quadratic_invariants(context['data'], base)
        baseline = opt._solve_invariant(sx, tnn, tnq, base, cfg.LEAST_SQUARES_TOL)
        # Both representations use epsilon internally for M. a_box's C is
        # epsilon times a single global normalization; T is therefore identical.
        new_state = dict(base, C=base['epsilon'].copy(), invariant_construction='root_geometry',
                         method_options=options)
        action_a_box.attach_metric(new_state, action_box)
        new_seed, _ = nl.quadratic_invariants(context['data'], new_state)
        print('Constructing root_geometry without tracking...', flush=True)
        start = time.perf_counter()
        refined, details = roots.solve(transfer, new_seed, new_state, cfg.LEAST_SQUARES_TOL)
        report['root_construction_seconds'] = time.perf_counter()-start
        report['methods']['root_geometry'] = details
        states = dict(root_geometry=new_state, a_box=base)
        invariants = dict(root_geometry=refined, a_box=baseline)
        baseline_in_root_basis = baseline*base['C']/new_state['C']
        W = new_state['action_weight']; R = transfer-np.eye(len(transfer))
        norm = np.linalg.norm(W @ np.r_[new_seed, np.zeros(len(refined)-len(new_seed))])
        report['methods']['a_box'] = dict(method='a_box',
            residual_in_common_action_metric=float(np.linalg.norm(W @ R @ baseline_in_root_basis)/norm))
        phi = np.linspace(0, 2*np.pi, int(angles)+1)
        curves = {}
        for method in METHODS:
            folder = out/method; folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/'Ix_coefficients.npz', Ix=invariants[method],
                C=states[method]['C'], powers=np.array(list(states[method]['idx_to_vec'].values())),
                CS0=states[method]['linear_cs0'])
            curves[method] = roots.root_curves(invariants[method], states[method], actions, phi, scale,
                                               max_action=4*max(scale, float(actions.max())))
            _save_geometry(out, method, curves[method], actions, phi)
        report['linear_cs0'] = base['linear_cs0']
        report['parameters'] = context['parameters']
        report['analysis_cells'] = int(cfg.ANALYSIS_CELLS)
        _geometry_plots(out, states, curves, actions, phi)
        _write_json(out/'comparison.json', report)
        print(f'Root constructor: {details["status"]}; residual={details["residual_final"]:.4g}', flush=True)
        if not geometry_only:
            # This import/build happens only AFTER coefficients and root CSVs
            # have been saved; no tracked samples reach roots.solve().
            print(f'Validating {len(actions)} shared launches for {turns} full-ring turns; delta=0', flush=True)
            ring, orbit, native = api._physical_tracking_ring(context)
            if np.max(abs(orbit[:5])) > 1e-10:
                raise ValueError('This Jy=0 comparison requires an on-momentum orbit at the origin.')
            if not np.allclose([native['parameters'][k] for k in context['parameters']],
                               list(context['parameters'].values()), rtol=1e-10, atol=1e-10):
                raise RuntimeError('Tracking lattice parameters differ from construction parameters.')
            bx, ax = base['linear_cs0'][:2]
            x0, px0 = roots.from_action_angle(actions, 0., bx, ax)
            initial = np.zeros((len(actions), 6)); initial[:, 0] = x0; initial[:, 1] = px0
            initial[:, 5] = orbit[5]
            trajectories, metadata = api._track_initial_conditions(
                ring, initial, turns, pool_size=1)
            report['physical_ring_cells'] = native['n_cells']
            cs = base['linear_cs0']; vertical_max = []
            for i, z in enumerate(trajectories):
                Jx, phix = roots.to_action_angle(z[0], z[1], cs[0], cs[1])
                Jy, phiy = roots.to_action_angle(z[2], z[3], cs[3], cs[4])
                vertical_max.append(float(np.max(Jy)))
                np.savez_compressed(out/f'tracking_{i:02d}.npz', coordinates=z,
                                     Jx=Jx, phi_x=phix, Jy=Jy, phi_y=phiy,
                                     turns=np.arange(z.shape[1]), **metadata[i])
            report['max_tracked_Jy'] = vertical_max
            report['tracking'] = _tracking_outputs(out, invariants, states, actions, trajectories, metadata)
            _geometry_plots(out, states, curves, actions, phi, trajectories)
            _write_json(out/'comparison.json', report)
    finally:
        runtime.unlink(missing_ok=True)
    print(f'Saved delta=0 comparison to {out}', flush=True)
    return report
