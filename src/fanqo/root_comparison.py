"""Radial--Fourier LS versus unchanged a_box: all roots and old tracking maps.

Construction is linear least squares. Root plots contain every nonnegative
real solution, not a selected branch. Tracking never enters construction.
"""
import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from . import api, config_loader
from .core import action_angle_ls, nonlinear as nl, optimization as opt
from .core import root_geometry as roots
from .plotting import get_pyplot

METHODS = ('action_angle_ls', 'a_box')
COLORS = {'action_angle_ls': '#2868b2', 'a_box': '#c53b36'}


def _write_json(path, value):
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, np.ndarray): return clean(v.tolist())
        if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.bool_): return bool(v)
        return v
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')


def _root_points(root_set, angles):
    phi = []; action = []
    for angle, row in zip(angles, root_set['all_roots']):
        if row is not None:
            phi.extend([angle]*len(row)); action.extend(row)
    return np.asarray(phi), np.asarray(action)


def _save_roots(folder, sets, angles):
    folder.mkdir(exist_ok=True)
    summaries = []
    for i, root_set in enumerate(sets):
        with (folder/f'all_roots_{i:02d}.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(['C', 'phi_x_rad', 'root_number', 'Jx'])
            for phi, row in zip(angles, root_set['all_roots']):
                if row is not None:
                    for k, value in enumerate(row): w.writerow([root_set['level'], phi, k, value])
        counts = [-1 if row is None else len(row) for row in root_set['all_roots']]
        np.savetxt(folder/f'root_counts_{i:02d}.csv', np.column_stack((angles, counts)),
                   delimiter=',', comments='', header='phi_x_rad,distinct_nonnegative_roots_minus1_if_degenerate')
        summaries.append(dict(level=root_set['level'], counts=counts,
                              degenerate_angles=[float(a) for a, row in zip(angles, root_set['all_roots']) if row is None]))
    _write_json(folder/'root_counts.json', summaries)


def _angle_axis(ax):
    ax.set_xlim(0, 2*np.pi)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                  ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_xlabel(r'$\phi_x$ [rad]'); ax.grid(alpha=.2)


def _all_root_plot(out, all_roots, angles, action_scale):
    plt = get_pyplot(False)
    count = len(all_roots[METHODS[0]])
    palette = plt.get_cmap('viridis')(np.linspace(.08, .92, count))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for column, method in enumerate(METHODS):
        for i, root_set in enumerate(all_roots[method]):
            p, j = _root_points(root_set, angles)
            for ax in axes[:, column]:
                ax.scatter(p, j*1e6, s=4, color=palette[i], label=f'C={root_set["level"]:.4g}', rasterized=True)
        axes[0, column].set(title=f'{method}: amplitude zoom', ylim=(0, 4*action_scale*1e6))
        axes[1, column].set(title=f'{method}: ALL finite roots')
        axes[1, column].set_yscale('symlog', linthresh=action_scale*1e6)
        for ax in axes[:, column]:
            _angle_axis(ax); ax.set_ylabel(r'$J_x$ [$10^{-6}$ m]'); ax.legend(fontsize=7)
    fig.suptitle('All nonnegative real roots of P(sqrt(Jx),phi_x)=C; delta=0, Jy=0')
    fig.savefig(out/'all_roots_action_angle.png', dpi=180)
    fig.savefig(out/'algebraic_action_angle.png', dpi=180); plt.close(fig)


def _horizontal_plots(out, invariants, states, actions, angles, scale, trajectories, metadata):
    plt = get_pyplot(False); summary = []
    fig, axes = plt.subplots(len(actions), 1, figsize=(10, max(4, 2*len(actions))),
                             squeeze=False, constrained_layout=True)
    for method in METHODS:
        fn = api._make_ix_callable(invariants[method], states[method])
        for i, (J, z, meta) in enumerate(zip(actions, trajectories, metadata)):
            values = fn(z[4], z[0], z[2], z[1], z[3])
            drift = (values-values[0])/J
            # Shared J0 denominator for horizontal time-series comparison.
            axes[i, 0].plot(np.arange(len(values)), drift, color=COLORS[method], label=method)
            np.savetxt(out/method/f'invariance_{i:02d}.csv',
                np.column_stack((np.arange(len(values)), values, drift)), delimiter=',',
                comments='', header='ring_turn,Ix,(Ix_minus_Ix0)_over_Jx0')
            summary.append(dict(method=method, launch_Jx=float(J),
                max_relative_excursion=float(np.max(abs(drift))), **meta))
    for i, ax in enumerate(axes[:, 0]):
        ax.set(title=f'Jx0={actions[i]:.5g}', ylabel='Relative Ix excursion'); ax.grid(alpha=.2); ax.legend()
    axes[-1, 0].set_xlabel('Full-ring turn')
    fig.savefig(out/'invariance_time_series.png', dpi=180); plt.close(fig)

    for i, J in enumerate(actions):
        z = trajectories[i]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for column, method in enumerate(METHODS):
            state = states[method]; cs = state['linear_cs0']
            fn = api._make_ix_callable(invariants[method], state)
            # The tracked launch fixes this overlay's C; roots themselves use
            # no trajectory samples, only the analytic launch coordinates.
            x, px = roots.from_action_angle(J, 0., cs[0], cs[1])
            C = float(fn(0., x, 0., px, 0.))
            root_set = roots.all_level_roots(invariants[method], state, [C], angles, scale,
                radial_coefficients=state.get('radial_last_coefficients'))[0]
            p, j = _root_points(root_set, angles)
            jt, pt = roots.to_action_angle(z[0], z[1], cs[0], cs[1])
            axes[0, column].scatter(p, j*1e6, s=5, color=COLORS[method], label='all algebraic roots')
            axes[0, column].scatter(pt, jt*1e6, s=3, color='black', alpha=.35, label='real tracking')
            axes[0, column].set(title=f'{method}; C={C:.5g}', ylim=(0, max(4*J, float(np.max(jt))*1.1)*1e6))
            _angle_axis(axes[0, column])
            axes[0, column].set_ylabel(r'$J_x$ [$10^{-6}$ m]')
            # Physical overlay is a zoom, not a deletion of remote roots:
            # the complete roots are saved in the CSV below.
            use = j <= max(4*J, float(np.max(jt))*1.1)
            q, momentum = roots.from_action_angle(j[use], p[use], cs[0], cs[1])
            axes[1, column].scatter(q*1e3, momentum*1e3, s=4, color=COLORS[method], label='algebraic roots (zoom)')
            axes[1, column].scatter(z[0]*1e3, z[1]*1e3, s=3, color='black', alpha=.35, label='real tracking')
            axes[1, column].set(xlabel='x [mm]', ylabel='px [mrad]')
            _save_roots(out/method/f'launch_{i:02d}', [root_set], angles)
        for ax in axes.ravel(): ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle(f'Launch Jx={J:.5g}, Jy=0; delta=0; root and phase-space zooms')
        fig.savefig(out/f'launch_{i:02d}_overlay.png', dpi=180); plt.close(fig)
    return summary


def _legacy_rows(fn, trajectories, metadata, coordinates, floor_fraction):
    """Exactly the legacy Ix0/floor normalization and max drift/turn metric."""
    initial = np.array([float(fn(z[4,0],z[0,0],z[2,0],z[1,0],z[3,0])) for z in trajectories])
    ref = float(np.max(abs(initial)))
    if not np.isfinite(ref) or ref == 0: ref = 1.
    floor = floor_fraction*ref; rows = []
    for i, (z, meta, xy) in enumerate(zip(trajectories, metadata, coordinates)):
        completed = meta['completed_turns']; ix0 = initial[i]
        if completed:
            v = fn(z[4,1:],z[0,1:],z[2,1:],z[1,1:],z[3,1:])
            d = v-ix0; rel = abs(d)/max(abs(ix0),floor)
            rate = rel/np.arange(1,completed+1)
            values = [float(v[-1]),float(np.max(abs(d))),float(np.max(rel)),
                      float(np.sqrt(np.mean(rel**2))),float(np.max(rate)),
                      float(np.log10(max(float(np.max(rate)),1e-300)))]
        else: values = [np.nan]*6
        rows.append([*xy,ix0,*values,completed,float(meta['survived'])])
    return np.asarray(rows)


def _invariance_maps(out, ring, orbit, invariants, states, coords, steps, turns, cfg):
    """One shared physical FMA launch grid; no frequency/tune calculation."""
    xs, ys = api._fma_grid(coords, steps)
    initial = np.vstack([api._initial_row(xs, y, orbit, 0.) for y in ys])
    coordinates = np.array([(x,y) for y in ys for x in xs])
    trajectories, metadata = api._track_initial_conditions(ring, initial, turns, pool_size=1)
    packed = np.full((len(trajectories), 6, turns+1), np.nan)
    for i, z in enumerate(trajectories):
        packed[i, :, :z.shape[1]] = z
    np.savez_compressed(out/'invariance_map_tracking.npz', initial_conditions=initial,
        coordinates_mm=coordinates, trajectories=packed,
        completed_turns=[m['completed_turns'] for m in metadata], survived=[m['survived'] for m in metadata])
    rows = {}
    for method in METHODS:
        fn = api._make_ix_callable(invariants[method], states[method])
        rows[method] = _legacy_rows(fn, trajectories, metadata, coordinates,
                                    float(getattr(cfg,'IX_INVARIANCE_NORM_FLOOR_FRACTION',1e-8)))
        np.savetxt(out/method/'Ix_invariance_map.csv', rows[method], delimiter=',', comments='',
            header='x_mm,y_mm,Ix_initial,Ix_last,max_abs_delta_Ix,max_relative_excursion,rms_relative_excursion,max_relative_drift_per_turn,log10_relative_drift_per_turn,completed_full_ring_turns,survived')
    plt = get_pyplot(False)
    lo = float(getattr(cfg,'IX_INVARIANCE_LOG_MIN',-10)); hi = float(getattr(cfg,'IX_INVARIANCE_LOG_MAX',-2))
    fig, axes = plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        data = rows[method]; valid = (data[:,10]>.5)&np.isfinite(data[:,8])
        im = ax.scatter(data[valid,0],data[valid,1],c=data[valid,8],cmap='viridis',vmin=lo,vmax=hi,s=20,marker='s')
        ax.scatter(data[~valid,0],data[~valid,1],c='gray',s=5,marker='x')
        ax.set(title=method,xlabel='x0 [mm]',ylabel='y0 [mm]',xlim=coords[:2],ylim=coords[2:],aspect='equal')
    fig.colorbar(im,ax=list(axes),label='log10 max relative Ix drift / ring turn')
    fig.suptitle('Legacy tracked-invariance metric; shared FMA launch grid; delta=0')
    fig.savefig(out/'invariance_comparison.png',dpi=180);plt.close(fig)
    a,b=rows['a_box'],rows['action_angle_ls']
    valid=(a[:,10]>.5)&(b[:,10]>.5)&np.isfinite(a[:,8])&np.isfinite(b[:,8])
    difference=a[valid,8]-b[valid,8]
    fig,ax=plt.subplots(figsize=(8,6),constrained_layout=True)
    im=ax.scatter(a[valid,0],a[valid,1],c=difference,cmap='RdBu',vmin=-2,vmax=2,s=25,marker='s')
    ax.scatter(a[~valid,0],a[~valid,1],c='gray',s=5,marker='x')
    ax.set(title='Red: a_box better | Blue: action-angle LS better; delta=0',
           xlabel='x0 [mm]',ylabel='y0 [mm]',xlim=coords[:2],ylim=coords[2:],aspect='equal')
    fig.colorbar(im,ax=ax,label='log10(drift_a_box / drift_action_angle_ls)')
    fig.savefig(out/'invariance_difference.png',dpi=180);plt.close(fig)
    np.savetxt(out/'invariance_difference.csv',np.column_stack((a[valid,:2],difference)),delimiter=',',
               comments='',header='x_mm,y_mm,log10_drift_a_box_over_action_angle_ls')
    return dict(launched=len(initial),survived=sum(m['survived'] for m in metadata),turns=turns,
                grid_steps=list(steps),coordinates_mm=list(coords))


def run_root_comparison(config, *, output='action_angle_comparison', order=8, delta_order=1,
                        action_box=None, launch_actions=None, action_fractions=(.05,.15,.3,.5,.7,.9),
                        level_values=None, angles=360, turns=1024, map_steps=(30,30),
                        map_turns=None, geometry_only=False, root_options=None):
    """Direct LS construction, all-root diagrams, then independent tracking."""
    # The original distributed runner supplied this keyword. Accept it so an
    # old local runner can launch the corrected LS code; never revive its
    # obsolete nonlinear-refinement settings.
    if root_options is not None:
        print('Compatibility: legacy root_options ignored; using direct linear action-angle LS.', flush=True)
    if order<3 or delta_order not in (0,1) or angles<8 or turns<1 or min(map_steps)<1:
        raise ValueError('Require order>=3, delta_order in {0,1}, angles>=8, turns>=1 and positive map steps.')
    config=Path(config).expanduser().resolve(); cfg=config_loader.load_general_config(config,reload=True)
    action_box=dict(action_box if action_box is not None else getattr(cfg,'ACTION_BOX',{}))
    if not action_box: raise ValueError('Set ACTION_BOX or supply action_box.')
    scale=float(action_box['Jx_max'])
    actions=np.asarray(launch_actions if launch_actions is not None else scale*np.asarray(action_fractions),float)
    levels=np.asarray(level_values if level_values is not None else actions,float)
    if actions.ndim!=1 or not len(actions) or not np.isfinite(actions).all() or np.any(actions<=0):
        raise ValueError('Launch actions must be a nonempty positive finite list.')
    out=Path(output).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    report=dict(delta=0.,root_slice_Jy=0.,root_slice_phi_y=None,order=order,delta_order=delta_order,
                launch_actions=actions,levels=levels,action_box=action_box,config=str(config),
                construction='direct linear least squares in radial-Fourier coefficients',
                tracking_used_in_construction=False,methods={},
                ignored_legacy_root_options=list(root_options) if root_options is not None else [])
    with tempfile.NamedTemporaryFile(mode='w',suffix='.py',prefix='_root_compare_',dir=config.parent,delete=False) as f:
        runtime=Path(f.name)
    try:
        override=f"""\nINVARIANT_CONSTRUCTION='a_box'
ORDER={int(order)}
DELTA_ORDER={int(delta_order)}
COMPUTE_IX=True
COMPUTE_IY=False
RUN_FMA_START_END=False
SHOW_PLOTS=False
FMA_CORRECT_CHROMATICITY=CORRECT_CHROMATICITY
FMA_PHYSICAL_RING_CELLS=globals().get('FMA_PHYSICAL_RING_CELLS',None)
FMA_NUM_INT_STEPS=globals().get('FMA_NUM_INT_STEPS',10)
FMA_CASE_LABEL='action_angle_ls_delta_zero'
"""
        runtime.write_text(config.read_text(encoding='utf-8')+override,encoding='utf-8')
        print(f'Building shared polynomial transport m={order}, d={delta_order}',flush=True)
        start=time.perf_counter();context=api.load(runtime,force=True);base=context['state']
        T,tnn,tnq=nl.nonlinear_transfer(context['lattice'],base,tol=cfg.LEAST_SQUARES_TOL)
        report['setup_and_transport_seconds']=time.perf_counter()-start
        sx,_=nl.quadratic_invariants(context['data'],base)
        baseline=opt._solve_invariant(sx,tnn,tnq,base,cfg.LEAST_SQUARES_TOL)
        state_j=dict(base,C=base['epsilon'].copy(),invariant_construction='action_angle_ls',method_options={})
        start=time.perf_counter();action_angle_ls.attach(state_j,action_box)
        seed_j,_=nl.quadratic_invariants(context['data'],state_j)
        invariant_j,details=action_angle_ls.solve(T,seed_j,state_j,cfg.LEAST_SQUARES_TOL)
        report['radial_basis_and_solve_seconds']=time.perf_counter()-start
        report['methods']['action_angle_ls']=details
        report['methods']['a_box']={'method':'a_box','constructor':'unchanged original weighted Cartesian LS'}
        states={'action_angle_ls':state_j,'a_box':base};invariants={'action_angle_ls':invariant_j,'a_box':baseline}
        phi=np.linspace(0,2*np.pi,int(angles)+1);all_roots={}
        for method in METHODS:
            folder=out/method;folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/'Ix_coefficients.npz',Ix=invariants[method],C=states[method]['C'],
                powers=np.array(list(states[method]['idx_to_vec'].values())),CS0=states[method]['linear_cs0'])
            all_roots[method]=roots.all_level_roots(invariants[method],states[method],levels,phi,scale,
                radial_coefficients=states[method].get('radial_last_coefficients'))
            _save_roots(folder,all_roots[method],phi)
        np.savez_compressed(out/'action_angle_ls'/'radial_fourier_coefficients.npz',
            coefficients=state_j['radial_last_coefficients'],labels=np.array(state_j['radial_labels']),
            transform=state_j['radial_transform'],gram=state_j['radial_gram'],
            transport=action_angle_ls.transform_transport(T,state_j))
        report['linear_cs0']=base['linear_cs0'];report['parameters']=context['parameters']
        report['analysis_cells']=int(cfg.ANALYSIS_CELLS)
        _all_root_plot(out,all_roots,phi,scale);_write_json(out/'comparison.json',report)
        if not geometry_only:
            print('Tracking for validation and legacy invariance maps only; delta=0',flush=True)
            ring,orbit,native=api._physical_tracking_ring(context)
            if np.max(abs(orbit[:5]))>1e-10: raise ValueError('Horizontal root comparison requires the on-momentum orbit at the origin.')
            if not np.allclose([native['parameters'][k] for k in context['parameters']],list(context['parameters'].values()),rtol=1e-10,atol=1e-10):
                raise RuntimeError('Tracking strengths differ from construction strengths.')
            bx,ax=base['linear_cs0'][:2];x0,p0=roots.from_action_angle(actions,0.,bx,ax)
            initial=np.zeros((len(actions),6));initial[:,0]=x0;initial[:,1]=p0;initial[:,5]=orbit[5]
            trajectories,meta=api._track_initial_conditions(ring,initial,turns,pool_size=1)
            for i,z in enumerate(trajectories):
                j,p=roots.to_action_angle(z[0],z[1],bx,ax)
                np.savez_compressed(out/f'tracking_{i:02d}.npz',coordinates=z,Jx=j,phi_x=p,**meta[i])
            report['horizontal_tracking']=_horizontal_plots(out,invariants,states,actions,phi,scale,trajectories,meta)
            report['physical_ring_cells']=native['n_cells']
            coords=getattr(cfg,'FMA_COORDS_MM',[-15.,15.,-15.,15.])
            report['invariance_map']=_invariance_maps(out,ring,orbit,invariants,states,coords,map_steps,
                                                     int(map_turns if map_turns is not None else turns),cfg)
            _write_json(out/'comparison.json',report)
    finally:
        runtime.unlink(missing_ok=True)
    print(f'Saved corrected delta=0 comparison to {out}',flush=True)
    return report
