"""Fixed-lattice action_a_box / a_box / Sx-selected eigen comparison.

Uses the public FANQO plotting functions and the existing tracking metrics.
FMA is shared because the physical lattice is identical for every constructor.
"""
import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from . import api
from . import config_loader
from .core import nonlinear as nl
from .comparison import _json

METHODS = ('action_a_box', 'a_box', 'eigen')
COLORS = {'action_a_box': '#008577', 'a_box': '#c53b36', 'eigen': '#2868b2'}


def _label(delta):
    return f'delta_{delta:+.5f}'


def _comparison_plots(data, root, cfg, delta):
    """Compare identical launch points; preserve the established drift metric."""
    from .plotting import get_pyplot
    plt = get_pyplot(False)
    methods = list(data)
    reference = data[methods[0]]
    for rows in data.values():
        if rows.shape != reference.shape or not np.allclose(rows[:,:2], reference[:,:2]):
            raise ValueError('Comparison grids differ between methods.')
    lo, hi = float(cfg.IX_INVARIANCE_LOG_MIN), float(cfg.IX_INVARIANCE_LOG_MAX)
    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 4.5),
                             squeeze=False, constrained_layout=True)
    for ax, method in zip(axes[0], methods):
        rows = data[method]
        valid = (rows[:,10] > .5) & np.isfinite(rows[:,8])
        im = ax.scatter(rows[valid,0], rows[valid,1], c=rows[valid,8],
                        vmin=lo, vmax=hi, cmap='viridis', marker='s', s=30)
        ax.set(title=method, xlabel='x0 [mm]', ylabel='y0 [mm]', aspect='equal')
        ax.set_xlim(cfg.FMA_COORDS_MM[:2]);ax.set_ylim(cfg.FMA_COORDS_MM[2:])
    fig.colorbar(im, ax=list(axes[0]), label='log10 max relative Ix drift / ring turn')
    fig.suptitle(f'Same lattice and launch grid; delta={delta:+g}')
    fig.savefig(root/'invariance_comparison.png', dpi=170); plt.close(fig)

    pairs = [('action_a_box','a_box'), ('action_a_box','eigen'), ('a_box','eigen')]
    fig, axes = plt.subplots(1, 3, figsize=(15,4.5), constrained_layout=True)
    for ax, (first,second) in zip(axes,pairs):
        a, b = data[first], data[second]
        valid = (a[:,10]>.5) & (b[:,10]>.5) & np.isfinite(a[:,8]) & np.isfinite(b[:,8])
        difference = a[valid,8]-b[valid,8]
        im = ax.scatter(a[valid,0],a[valid,1],c=difference,cmap='RdBu',
                        vmin=-2,vmax=2,marker='s',s=30)
        ax.set(title=f'{first} / {second}',xlabel='x0 [mm]',ylabel='y0 [mm]',aspect='equal')
        ax.set_xlim(cfg.FMA_COORDS_MM[:2]);ax.set_ylim(cfg.FMA_COORDS_MM[2:])
        np.savetxt(root/f'{first}_vs_{second}.csv',
                   np.column_stack((a[valid,:2],difference)), delimiter=',',
                   header='x_mm,y_mm,log10_drift_ratio_first_over_second',comments='')
    fig.colorbar(im,ax=list(axes),label='log10(first/second): red = first better; blue = second better')
    fig.savefig(root/'drift_ratios.png',dpi=170);plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for method,rows in data.items():
        valid=(rows[:,10]>.5)&np.isfinite(rows[:,5])&np.isfinite(rows[:,8])
        excursion=np.sort(np.maximum(rows[valid,5],1e-16))
        if len(excursion):
            axes[0].semilogx(excursion,np.arange(1,len(excursion)+1)/len(excursion),
                             label=method,color=COLORS[method])
        radius=np.hypot(rows[:,0],rows[:,1])
        edges=np.linspace(0,max(float(radius.max()),1e-12),13)
        centers=[];medians=[]
        for low,high in zip(edges[:-1],edges[1:]):
            use=valid&(radius>=low)&(radius<=high)
            if use.any():
                centers.append((low+high)/2);medians.append(np.median(rows[use,8]))
        axes[1].plot(centers,medians,'o-',label=method,color=COLORS[method])
    axes[0].set(xlabel='Max relative Ix excursion',ylabel='Fraction of surviving launches')
    axes[1].set(xlabel='Launch radius in x-y [mm]',ylabel='Median log10 relative drift / turn')
    for ax in axes: ax.grid(alpha=.25);ax.legend()
    fig.savefig(root/'excursion_and_radius.png',dpi=170);plt.close(fig)


def run_action_comparison(config, *, action_box, output='action_comparison',
                          deltas=(0.,-.01,.01), order=8, delta_order=1,
                          quick=False, timing_only=False):
    """Construct three Ix polynomials, then run the existing diagnostic plots."""
    config=Path(config).expanduser().resolve()
    cfg_original=config_loader.load_general_config(config,reload=True)
    base_text=config.read_text(encoding='utf-8')
    out=Path(output).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    report={'order':int(order),'delta_order':int(delta_order),'action_box':action_box,
            'action_convention':'J=S without factor 1/2',
            'construction_residual':'projected polynomial map, not tracking-fitted',
            'config':str(config),'a_box':np.asarray(cfg_original.A_BOX).tolist(),
            'quick':bool(quick),'deltas':list(deltas),'methods':{},
            'comparison_note':'Same lattice. FMA shared. Ix uses existing method-specific normalization.'}
    shared_fma={};comparison_data={delta:{} for delta in deltas};reference_lattice=None
    with tempfile.NamedTemporaryFile(mode='w',suffix='.py',prefix='_action_compare_',
                                      dir=config.parent,delete=False) as handle:
        runtime=Path(handle.name)
    try:
        for method in METHODS:
            root=out/method;root.mkdir(exist_ok=True)
            override=f'''\nfrom pathlib import Path
ORDER = {int(order)}
DELTA_ORDER = {int(delta_order)}
INVARIANT_CONSTRUCTION = {method!r}
ACTION_BOX = {action_box!r}
INVARIANT_OPTIONS = dict(globals().get('INVARIANT_OPTIONS', {{}}))
INVARIANT_OPTIONS['action_a_box'] = {{'ACTION_BOX': ACTION_BOX}}
COMPUTE_IX = True
COMPUTE_IY = False
RUN_FMA_START_END = False
SHOW_PLOTS = False
SAVE_PLOTS = True
FMA_SHOW_PLOT = False
FMA_SAVE_PLOT = True
OUTPUT_ROOT = Path({str(root)!r})
'''
            if quick:
                override+='\nFMA_STEPS = [5, 5]\nFMA_TURNS = 32\nFMA_POOL_SIZE = 1\nPOINCARE_TURNS = 64\nPOINCARE_GRID_POINTS = 80\nPLOT_GRID_POINTS = 80\n'
            runtime.write_text(base_text+override,encoding='utf-8')
            print(f'\nPreparing {method}: m={order}, d={delta_order}',flush=True)
            start=time.perf_counter();context=api.load(runtime,force=True)
            setup_seconds=time.perf_counter()-start
            signature=[(str(e[0]),repr(e)) for e in context['lattice']]
            if reference_lattice is None: reference_lattice=signature
            elif signature!=reference_lattice: raise RuntimeError('Physical lattice changed between constructors.')
            start=time.perf_counter();ix,_=api.compute_invariants()
            compute_seconds=time.perf_counter()-start
            state=context['state'];cfg=api._cfg()
            info={'setup_seconds':setup_seconds,'compute_seconds_including_map':compute_seconds,
                  'construction_details':api.STATE.invariant_details.get('Ix_construction_details',{}),
                  'parameters':context['parameters'],'linear_cs0':state['linear_cs0'],
                  'tracking':{}}
            report['methods'][method]=info
            np.savez_compressed(root/'Ix_coefficients.npz',Ix=ix,C=state['C'],
                                powers=np.array(list(state['idx_to_vec'].values())))
            sx,_=nl.quadratic_invariants(context['data'],state)
            delta_quad=(ix[:len(sx)]-sx)*state['C'][:len(sx)]
            info['fixed_block_physical_error']=float(np.linalg.norm(delta_quad))
            api.write_invariant_report(file_name=root/'invariant_report.txt')
            print(f'{method}: setup={setup_seconds:.3f}s; map+solve={compute_seconds:.3f}s',flush=True)
            if not timing_only:
                for delta in deltas:
                    name=_label(delta);dr=root/name;dr.mkdir(exist_ok=True)
                    cfg.FMA_DELTA=float(delta);cfg.POINCARE_DELTA=float(delta)
                    cfg.SLICE_DELTA_VALUES=(float(delta),)
                    cfg.FMA_OUTPUT_DIRECTORY=dr/'FMA';cfg.POINCARE_OUTPUT_DIRECTORY=dr/'poincare'
                    cfg.FMA_CASE_LABEL=f'{method}_{name}'
                    api.plot_invariant(folder=dr/'contours',save=True,show=False)
                    api.plot_invariant_tracking('x',delta=float(delta),turns=int(cfg.POINCARE_TURNS),
                        file_name=dr/'poincare.png',save=True,show=False)
                    if delta not in shared_fma:
                        diag=api.run_fma(stage=name,quick=False);shared_fma[delta]=diag
                    else:
                        # Reuse the identical physical frequency map. Evaluate the new Ix
                        # on the same deterministic grid and physical tracking convention.
                        import at
                        source=shared_fma[delta];ring=source['ring'];orbit,_=at.find_orbit(ring)
                        rows=api._track_ix(ring,ix,state,cfg.FMA_COORDS_MM,cfg.FMA_STEPS,
                            2*int(cfg.FMA_TURNS),orbit,float(delta),cfg.FMA_POOL_SIZE,
                            cfg.IX_INVARIANCE_NORM_FLOOR_FRACTION)
                        destination=dr/'FMA'/name;label=f'{method}_{name}'
                        api._save_csvs(source['fmap'],rows,destination,label)
                        api._tracking_plots(source['fmap'],rows,source['native'],destination,label,
                                            cfg.FMA_COORDS_MM,delta,True,False)
                        diag={'ix_data':rows}
                    rows=np.asarray(diag['ix_data']);comparison_data[delta][method]=rows
                    valid=(rows[:,10]>.5)&np.isfinite(rows[:,8])
                    info['tracking'][name]={'launched':len(rows),'survived':int(valid.sum()),
                        'median_log_drift_per_turn':float(np.median(rows[valid,8])) if valid.any() else None}
            (out/'comparison.json').write_text(json.dumps(_json(report),indent=2),encoding='utf-8')
        if not timing_only:
            for delta,data in comparison_data.items():
                root=out/'comparison_plots'/_label(delta);root.mkdir(parents=True,exist_ok=True)
                _comparison_plots(data,root,api._cfg(),delta)
        with (out/'construction_times.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['method','setup_seconds','compute_seconds_including_map'])
            for method,info in report['methods'].items():
                writer.writerow([method,info['setup_seconds'],info['compute_seconds_including_map']])
    finally:
        runtime.unlink(missing_ok=True)
    print(f'\nSaved comparison to {out}',flush=True)
    return _json(report)
