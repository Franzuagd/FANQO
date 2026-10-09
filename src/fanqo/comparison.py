"""Reproducible fixed-lattice comparison of invariant constructors.

Setup/map times and constructor-only times are separate. Every method is
assessed on the SAME PyAT trajectories. No lattice optimization is performed.
"""
import csv
import importlib.util
import json
import math
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
import sympy as sp

from .core import linear as lin, nonlinear as nl
from .core import structured_invariants as si

DEFAULT_METHODS = ("eigen", "a_box_coupled_fixed", "a_box_coupled_regularized",
                   "graded_coupled", "canonical_graded")


def _load_module(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("_fanqo_comparison_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configuration(config=None):
    if config:
        cfg = _load_module(config)
        lc = _load_module(Path(config).resolve().parent / cfg.LATTICE_FILE)
        return cfg, lc
    from . import comparison_lattice as lc
    delta, x, y, px, py = sp.symbols("delta x y px py")
    b1, b2, b3, b4, b5 = sp.symbols("b1 b2 b3 b4 b5")
    H = ((px**2+py**2)*(1-delta+delta**2)/2 - b1*x*delta
         + b1**2*x**2/2 + b2*(x**2-y**2)/2
         + b3*(x**3-3*x*y**2)/3 + b4*(x**4-6*x*x*y*y+y**4)/4)
    return SimpleNamespace(
        VARIABLES=(delta, x, y, px, py), FIELD_SYMBOLS=(b1,b2,b3,b4,b5),
        HAMILTONIAN=H, A_BOX=np.array([.01,.01,.008,.001,.0008]),
        ANALYSIS_CELLS=1, CORRECT_CHROMATICITY=True, LEAST_SQUARES_TOL=1e-14,
        INVARIANT_OPTIONS={}, FMA_NUM_INT_STEPS=10,
    ), lc


def _linear(cfg, lc):
    magnets, lattice, data, correction, parameters = lin.prepare_lattice(
        parameters=dict(lc.PARAMETERS), ring_names=list(lc.CELL_NAMES)*cfg.ANALYSIS_CELLS,
        magnet_builder=lc.define_magnets, energy_parameter=lc.ENERGY_PARAMETER,
        correction_parameter_map=lc.CORRECTION_PARAMETER_MAP,
        correct_chromatic=cfg.CORRECT_CHROMATICITY,
        family1=lc.CHROMATIC_FAMILY1, family2=lc.CHROMATIC_FAMILY2,
        target_chrom_x=lc.TARGET_CHROM_X, target_chrom_y=lc.TARGET_CHROM_Y,
        repetitions=lc.REPETITIONS, step=lc.STEP,
    )
    return magnets, lattice, data, correction, parameters


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k:_json(v) for k,v in value.items() if k != "canonical_generators"}
    if isinstance(value, (tuple,list)):
        return [_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def _evaluate(c, state, trajectory):
    z = np.asarray(trajectory)[[4,0,2,1,3]]
    out = np.zeros(z.shape[1:])
    for i,a in state["idx_to_vec"].items():
        if c[i] == 0:
            continue
        term = c[i]*state["C"][i]
        for k,power in enumerate(a):
            if power:
                term = term*z[k]**power
        out += term
    return out


def _track(cfg, lc, magnets, parameters, delta, points, turns, extent_mm):
    import at
    from .api import _at_element
    by_name={lin.magnet_field(e,"NAME"):e for e in magnets}
    cell=[by_name[name] for name in lc.CELL_NAMES]
    bend=sum(float(lin.magnet_field(e,"ANGLE")) for e in cell)
    ncells=int(round(360/abs(bend)))
    if not np.isclose(ncells*abs(bend),360,atol=1e-6):
        raise ValueError("Reference cell does not form an integer 360-degree ring.")
    elems=[_at_element(e,at,int(cfg.FMA_NUM_INT_STEPS)) for e in cell*ncells]
    ring=at.Lattice(elems,energy=float(parameters[lc.ENERGY_PARAMETER])*1e9)
    orbit,_=at.find_orbit4(ring,dp=float(delta))
    xs=np.linspace(-extent_mm,extent_mm,points)
    ys=np.linspace(-extent_mm,extent_mm,points)
    xx,yy=np.meshgrid(xs,ys,indexing='xy')
    initial=np.repeat(np.asarray(orbit).reshape(6,1),xx.size,axis=1)
    initial[0]+=xx.ravel()*1e-3+1e-9
    initial[2]+=yy.ravel()*1e-3+1e-9
    # find_orbit4(dp=delta) already sets delta. Do not add it a second time.
    initial[4]=delta
    tracked=at.lattice_pass(ring,np.asfortranarray(initial.copy()),turns)
    tracks=np.asarray(tracked)
    if tracks.ndim==4:
        tracks=tracks[:,:,0,:]
    coordinates=np.concatenate((initial[:,:,None],tracks),axis=2)
    survived=np.isfinite(coordinates).all(axis=(0,2))
    return coordinates,survived,xx,yy,ncells


def _tracking_metrics(ix,state,coordinates,survived,sigma):
    n=coordinates.shape[1]
    max_relative=np.full(n,np.nan);rms=np.full(n,np.nan);legacy=np.full(n,np.nan)
    ids=np.flatnonzero(survived)
    if len(ids):
        values=_evaluate(ix,state,coordinates[:,ids,:])
        err=np.abs(values[:,1:]-values[:,:1])
        max_relative[ids]=np.max(err,axis=1)/sigma[ids]
        rms[ids]=np.sqrt(np.mean(err**2,axis=1))/sigma[ids]
        floor=1e-12*max(float(np.max(np.abs(values[:,0]))),1e-300)
        legacy[ids]=np.max(err/np.arange(1,err.shape[1]+1),axis=1)/np.maximum(np.abs(values[:,0]),floor)
    return max_relative,rms,legacy


def run_comparison(*, order=8, delta_order=1, repeats=3, points=25, turns=512,
                   deltas=(0.,-.01,.01), extent_mm=10., config=None,
                   output='structured_comparison', methods=DEFAULT_METHODS,
                   coupled_maxiter=200, timing_only=False, options=None):
    """Return a JSON-serializable timing/quality report and save plots + arrays."""
    if order < 4 or delta_order not in (0,1):
        raise ValueError("Comparison supports order>=4 and delta_order=0 or 1.")
    if repeats<1 or points<2 or turns<1 or extent_mm<=0:
        raise ValueError("Require repeats>=1, points>=2, turns>=1, extent_mm>0.")
    unknown=set(methods)-set(DEFAULT_METHODS)-{"a_box"}
    if unknown:
        raise ValueError(f"Unknown methods {unknown}")
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    cfg,lc=_configuration(config)
    print(f'Building m={order}, d={delta_order} common experiment...',flush=True)
    start=time.perf_counter()
    magnets,lattice,data,correction,parameters=_linear(cfg,lc)
    linear_seconds=time.perf_counter()-start
    basis_args=dict(m=order,d=delta_order,hamiltonian=cfg.HAMILTONIAN,
                    a_box=cfg.A_BOX,variables=cfg.VARIABLES,
                    field_symbols=cfg.FIELD_SYMBOLS,n=2)
    start=time.perf_counter()
    base=nl.initialize_nonlinear_for_method(data,**basis_args,invariant_construction='graded_coupled')
    si.bracket_tensor(base)
    setup_seconds=time.perf_counter()-start
    start=time.perf_counter()
    T,_,_=nl.nonlinear_transfer(lattice,base,tol=cfg.LEAST_SQUARES_TOL)
    map_seconds=time.perf_counter()-start
    # Eigen uses its original physical-monomial map, preserving its selector.
    start=time.perf_counter()
    eigstate=nl.initialize_nonlinear_for_method(data,**basis_args,invariant_construction='eigen')
    eigen_setup_seconds=time.perf_counter()-start
    start=time.perf_counter()
    Te,_,_=nl.nonlinear_transfer(lattice,eigstate,tol=cfg.LEAST_SQUARES_TOL)
    eigen_map_seconds=time.perf_counter()-start
    sx,sy=si.seeds(data,base)
    scale=base['C']
    map_similarity_error=float(np.linalg.norm(Te-(scale[:,None]*T/scale[None,:]))/max(np.linalg.norm(Te),1e-300))
    report={
        'order':order,'delta_order':delta_order,'basis_size':len(T),'repeats':repeats,
        'python':platform.python_version(),'platform':platform.platform(),
        'numpy':np.__version__,'scipy':scipy.__version__,
        'timing_scope':'warm constructor only; setup, map building, external comparison diagnostics and tracking excluded; includes constructor diagnostics',
        'eigen_scope':'Ix only; pair methods solve Ix and auxiliary Iy',
        'linear_seconds':linear_seconds,'box_setup_seconds':setup_seconds,
        'box_map_seconds':map_seconds,'eigen_setup_seconds':eigen_setup_seconds,
        'eigen_map_seconds':eigen_map_seconds,'map_similarity_relative_error':map_similarity_error,
        'parameters':parameters,'a_box':np.asarray(cfg.A_BOX).tolist(),
        'analysis_cells':cfg.ANALYSIS_CELLS,'deltas':list(deltas),'turns':turns,
        'grid_points_per_axis':points,'extent_mm':extent_mm,'methods':{},'tracking':[],
    }
    results={};timing_rows=[]
    for method in methods:
        state=dict(eigstate if method=='eigen' else base)
        state['invariant_construction']=method
        opt=dict(getattr(cfg,'INVARIANT_OPTIONS',{}).get(method,{}))
        opt.update((options or {}).get(method,{}))
        opt['COUPLED_MAX_ITER']=coupled_maxiter
        opt['COUPLED_FIX_QUADRATIC']=method=='a_box_coupled_fixed'
        state['method_options']=opt
        durations=[]
        try:
            for _ in range(repeats):
                t0=time.perf_counter()
                if method=='eigen':
                    ix,details=nl.eigen_invariant(Te,state,plane='x');iy=None
                elif method=='a_box':
                    q=state['quad_size']
                    ix,details=si.least_squares_seed(T[q:,q:],T[q:,:q],sx[:q],state,cfg.LEAST_SQUARES_TOL);iy=None
                else:
                    ix,iy,details=si.construct_pair(T,data,state,cfg.LEAST_SQUARES_TOL)
                durations.append(time.perf_counter()-t0)
            common=ix*state['C']/base['C']
            metrics={'relative_homological_x':si._norm(T @ common-common,base['G'])/si._norm(sx,base['G']),
                     'quadratic_relative_error_x':si._norm((common-sx)[:15],base['G'][:15,:15])/si._norm(sx,base['G'])}
            if iy is not None:
                common_y=iy*state['C']/base['C']
                metrics.update(si.pair_diagnostics(T,common,common_y,sx,sy,base))
            row=dict(method=method,median_seconds=float(np.median(durations)),
                     min_seconds=min(durations),max_seconds=max(durations),
                     iterations=details.get('nit'),solver_success=details.get('solver_success'),**metrics)
            report['methods'][method]={'timing':row,'seconds':durations,'details':details,'options':opt}
            results[method]=(ix,iy,state)
            timing_rows.append(row)
            np.savez_compressed(out/f'{method}_coefficients.npz',Ix=ix,
                Iy=np.array([]) if iy is None else iy,C=state['C'],
                powers=np.array(list(state['idx_to_vec'].values())))
            print(f'{method:28s} {np.median(durations):.4f} s; homological={metrics["relative_homological_x"]:.3g}',flush=True)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
            report['methods'][method]={'error':str(exc),'seconds':durations}
            print(f'{method}: FAILED: {exc}',flush=True)
    _write_csv(out/'construction_times.csv',timing_rows)
    # Write partial results before potentially long tracking.
    (out/'comparison.json').write_text(json.dumps(_json(report),indent=2),encoding='utf-8')
    if not timing_only and results:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for delta in deltas:
            print(f'Tracking shared grid at delta={delta:+g}...',flush=True)
            coordinates,survived,xx,yy,ncells=_track(cfg,lc,magnets,parameters,delta,points,turns,extent_mm)
            initial_sx=np.abs(_evaluate(sx,base,coordinates[:,:,0]))
            sigma=np.maximum(initial_sx,1e-6*max(float(initial_sx.max()),1e-300))
            values={};quality=[]
            label=f'delta_{delta:+.5f}'
            for method,(ix,iy,state) in results.items():
                maxr,rms,legacy=_tracking_metrics(ix,state,coordinates,survived,sigma)
                values[method]=maxr
                finite=maxr[np.isfinite(maxr)]
                row=dict(method=method,delta=float(delta),launched=len(survived),survived=int(survived.sum()),
                         median_max_drift=float(np.median(finite)) if len(finite) else None,
                         p90_max_drift=float(np.quantile(finite,.9)) if len(finite) else None,
                         worst_max_drift=float(finite.max()) if len(finite) else None)
                quality.append(row)
                np.savez_compressed(out/f'{label}_{method}_quality.npz',max_drift=maxr,rms_drift=rms,legacy_rate=legacy,
                                     denominator=sigma,survived=survived,x_mm=xx,y_mm=yy)
            _write_csv(out/f'{label}_quality.csv',quality)
            report['tracking'].append({'delta':delta,'physical_ring_cells':ncells,'quality':quality})
            np.savez_compressed(out/f'{label}_trajectories.npz',coordinates=coordinates,survived=survived,x_mm=xx,y_mm=yy)
            fig,axes=plt.subplots(1,len(values),figsize=(4*len(values),4),squeeze=False,constrained_layout=True)
            for ax,(method,v) in zip(axes[0],values.items()):
                im=ax.pcolormesh(xx,yy,np.log10(np.maximum(v.reshape(xx.shape),1e-14)),
                                 cmap='viridis',vmin=-8,vmax=1,shading='auto')
                ax.set(title=method,xlabel='x [mm]',ylabel='y [mm]')
            fig.colorbar(im,ax=axes[0].tolist(),label='log10 max |Ix(k)-Ix(0)| / fixed Sx scale')
            fig.suptitle(f'Same trajectories, delta={delta:+g}; {survived.sum()}/{len(survived)} survived')
            fig.savefig(out/f'{label}_invariance.png',dpi=160);plt.close(fig)
            if 'eigen' in values and len(values)>1:
                other=[m for m in values if m!='eigen']
                fig,axes=plt.subplots(1,len(other),figsize=(4*len(other),4),squeeze=False,constrained_layout=True)
                for ax,m in zip(axes[0],other):
                    difference=np.log10(np.maximum(values[m],1e-14))-np.log10(np.maximum(values['eigen'],1e-14))
                    im=ax.pcolormesh(xx,yy,difference.reshape(xx.shape),cmap='RdBu',vmin=-2,vmax=2,shading='auto')
                    ax.set(title=f'{m} vs eigen',xlabel='x [mm]',ylabel='y [mm]')
                fig.colorbar(im,ax=axes[0].tolist(),label='log10 error(method)/error(eigen): red=method better, blue=eigen better')
                fig.savefig(out/f'{label}_vs_eigen.png',dpi=160);plt.close(fig)
            print('  '+', '.join(f'{r["method"]}: median={r["median_max_drift"]}' for r in quality),flush=True)
    (out/'comparison.json').write_text(json.dumps(_json(report),indent=2),encoding='utf-8')
    print(f'Saved comparison to {out}',flush=True)
    return _json(report)
