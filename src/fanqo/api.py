"""Public stateful API.

Internal numerical routines live under :mod:`fanqo.core`.
Users should normally import this package and call the functions defined here.
"""

from __future__ import annotations

from pathlib import Path
import gc
import importlib.util
import math
import numpy as np

from .state import STATE
from .config_loader import load_general_config, load_selected_lattice, analysis_ring_names
from .core import linear as lin
from .core import nonlinear as nl
from .core import optimization as opt


def _cfg():
    if STATE.config is None:
        raise RuntimeError("No experiment is loaded. Run load() first.")
    return STATE.config


def _require_context(action):
    if STATE.context is None or STATE.lattice_config is None:
        raise RuntimeError(f"{action} requires a loaded lattice. Run load() first.")
    return STATE.context


def _require_invariants(action):
    context = _require_context(action)
    if STATE.Ix is None or STATE.Iy is None:
        raise RuntimeError(f"{action} requires Ix and Iy. Run compute_invariants() first.")
    return context


def _set_invariants(details):
    STATE.invariant_details = details
    STATE.Ix = np.asarray(details["Ix"], dtype=float).copy()
    STATE.Iy = np.asarray(details["Iy"], dtype=float).copy()


def _clear_derived():
    STATE.Ix = None
    STATE.Iy = None
    STATE.invariant_details = None
    STATE.optimization_result = None
    STATE.diagnostics.clear()


def _config_dir():
    return Path(_cfg().__file__).resolve().parent


def _resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (_config_dir() / path).resolve()


def _output_root():
    return _resolve(getattr(_cfg(), "OUTPUT_ROOT", Path("optimization_output")))


def _slice_settings():
    cfg = _cfg()
    return {
        "y_values": cfg.SLICE_Y_VALUES,
        "x_values": cfg.SLICE_X_VALUES,
        "delta_values": cfg.SLICE_DELTA_VALUES,
        "frozen_momentum": cfg.SLICE_FROZEN_MOMENTUM,
        "levels": cfg.PLOT_LEVELS,
        "grid_points": cfg.PLOT_GRID_POINTS,
        "rmin": cfg.PLOT_RMIN,
        "rmax": cfg.PLOT_RMAX,
        "x_max": cfg.PLOT_X_MAX,
        "px_max": cfg.PLOT_PX_MAX,
        "y_max": cfg.PLOT_Y_MAX,
        "py_max": cfg.PLOT_PY_MAX,
    }


def _build_context(parameters):
    cfg = _cfg()
    lattice_cfg = STATE.lattice_config
    return opt.create_context(
        dict(parameters),
        ring_names=analysis_ring_names(cfg, lattice_cfg),
        magnet_builder=lattice_cfg.define_magnets,
        energy_parameter=lattice_cfg.ENERGY_PARAMETER,
        correction_parameter_map=lattice_cfg.CORRECTION_PARAMETER_MAP,
        correct_chromatic=cfg.CORRECT_CHROMATICITY,
        family1=lattice_cfg.CHROMATIC_FAMILY1,
        family2=lattice_cfg.CHROMATIC_FAMILY2,
        target_chrom_x=lattice_cfg.TARGET_CHROM_X,
        target_chrom_y=lattice_cfg.TARGET_CHROM_Y,
        repetitions=lattice_cfg.REPETITIONS,
        step=lattice_cfg.STEP,
        linear_variables=lattice_cfg.LINEAR_VARIABLES,
        chromatic_variables=lattice_cfg.CHROMATIC_VARIABLES,
        parameter_map=lattice_cfg.PARAMETER_MAP,
        m=cfg.ORDER,
        d=cfg.DELTA_ORDER,
        hamiltonian=cfg.HAMILTONIAN,
        a_box=cfg.A_BOX,
        variables=cfg.VARIABLES,
        field_symbols=cfg.FIELD_SYMBOLS,
        n_planes=cfg.N_PLANES,
    )


# =============================================================================
# EXPERIMENT STATE
# =============================================================================

def load(config_file="general_config.py", *, force=False):
    """Load a general config and the lattice file selected by LATTICE_FILE."""
    cfg = load_general_config(config_file, reload=bool(force))
    STATE.config = cfg
    STATE.config_path = str(Path(cfg.__file__).resolve())
    STATE.lattice_config = load_selected_lattice(cfg, reload=bool(force))
    STATE.context = None
    _clear_derived()
    gc.collect()
    STATE.context = _build_context(STATE.lattice_config.PARAMETERS)
    STATE.source = "loaded"
    print(f"Loaded config : {STATE.config_path}")
    print(f"Loaded lattice: {Path(STATE.lattice_config.__file__).resolve()}")
    print(f"Analysis cells: {int(cfg.ANALYSIS_CELLS)}")
    return STATE.context


def status():
    """Print and return the active-state summary."""
    cfg = STATE.config
    info = {
        "config_file": STATE.config_path,
        "lattice_file": str(Path(STATE.lattice_config.__file__).resolve()) if STATE.lattice_config else None,
        "analysis_cells": int(cfg.ANALYSIS_CELLS) if cfg else None,
        "context_loaded": STATE.context is not None,
        "parameters_state": STATE.source,
        "Ix_available": STATE.Ix is not None,
        "Iy_available": STATE.Iy is not None,
        "optimization_completed": STATE.optimization_result is not None,
        "diagnostics": tuple(sorted(STATE.diagnostics)),
    }
    print("=" * 72)
    print("CURRENT EXPERIMENT")
    print("=" * 72)
    for key, value in info.items():
        print(f"{key:<24}: {value}")
    print("=" * 72)
    return info


def current_parameters():
    return dict(_require_context("current_parameters()")["parameters"])


def current_lattice():
    return _require_context("current_lattice()")["lattice"]


def set_parameters(**changes):
    """Rebuild the current experiment after editing selected parameters."""
    context = _require_context("set_parameters()")
    parameters = dict(context["parameters"])
    unknown = set(changes) - set(parameters)
    if unknown:
        raise KeyError("Unknown parameter(s): " + ", ".join(sorted(unknown)))

    names = []
    values = []
    for name, value in changes.items():
        value = float(value)
        if value != float(parameters[name]):
            names.append(name)
            values.append(value)

    if not names:
        return parameters

    opt.apply_candidate(context, values, names)
    _clear_derived()
    STATE.source = "edited"
    return dict(context["parameters"])


# =============================================================================
# LINEAR USER API
# =============================================================================

def linear_summary():
    context = _require_context("linear_summary()")
    lc = STATE.lattice_config
    data = context["data"]
    summary = {
        "energy_GeV": float(context["parameters"][lc.ENERGY_PARAMETER]),
        "tune_x": float(lin.linear_data(data, "TUNE_X")),
        "tune_y": float(lin.linear_data(data, "TUNE_Y")),
        "chrom_x": float(lin.linear_data(data, "CHROM_X")),
        "chrom_y": float(lin.linear_data(data, "CHROM_Y")),
        "emittance": float(lin.linear_data(data, "EMITTANCE")),
        "circumference_m": float(lin.linear_data(data, "CIRCUMFERENCE")),
        "chromatic_correction": context["correction"],
    }
    print("=" * 72); print("LINEAR OPTICS"); print("=" * 72)
    for k,v in summary.items(): print(f"{k:<24}: {v}")
    return summary


def linear_checks():
    context = _require_context("linear_checks()")
    values = lin.check_linear_lattice(context["lattice"], context["data"])
    names = ("symplectic_error","twiss_closure","dispersion_closure",
             "cs_identity_x_error","cs_identity_y_error","total_bend_deg")
    checks = dict(zip(names, values))
    print("=" * 72); print("LINEAR CHECKS"); print("=" * 72)
    for k,v in checks.items(): print(f"{k:<28}: {v}")
    return checks


def plot_linear(*, file_name=None, show=False):
    context = _require_context("plot_linear()")
    data = context["data"]
    file_name = _output_root()/"plots"/"linear_optics.png" if file_name is None else _resolve(file_name)
    file_name = Path(file_name); file_name.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    s = lin.linear_data(data, "S_VALUES")
    cs = lin.linear_data(data, "CS_VALUES")
    disp = lin.linear_data(data, "DISP_VALUES")
    fig, ax = plt.subplots(figsize=(9,5.5))
    ax.plot(s, cs[:,0], label=r"$\beta_x$")
    ax.plot(s, cs[:,3], label=r"$\beta_y$")
    ax.plot(s, 100*disp[:,0], label=r"$100D_x$")
    ax.set_xlabel("s [m]"); ax.set_ylabel("Linear functions [m]")
    ax.legend(); ax.grid(True); fig.tight_layout(); fig.savefig(file_name, dpi=200)
    if show: plt.show()
    else: plt.close(fig)
    return file_name


# =============================================================================
# NONLINEAR / INVARIANT USER API
# =============================================================================

def compute_invariants():
    cfg = _cfg(); context = _require_context("compute_invariants()")
    details = opt.full_diagnostics(context, cfg.GRADIENT_WEIGHT, cfg.LEAST_SQUARES_TOL)
    _set_invariants(details)
    return STATE.Ix.copy(), STATE.Iy.copy()


def get_Ix():
    _require_invariants("get_Ix()"); return STATE.Ix.copy()


def get_Iy():
    _require_invariants("get_Iy()"); return STATE.Iy.copy()


def invariant_polynomial(plane="x"):
    context = _require_invariants("invariant_polynomial()")
    plane = str(plane).lower()
    if plane not in ("x","y"): raise ValueError("plane must be 'x' or 'y'.")
    vec = STATE.Ix if plane=="x" else STATE.Iy
    st = context["state"]
    return nl.vector_to_poly(np.asarray(vec)*np.asarray(st["C"]), st["monomial_basis"])


def _make_ix_callable(Ix, state):
    Ix=np.asarray(Ix,float); C=np.asarray(state["C"],float)
    if len(Ix)!=len(C): raise ValueError("Ix and state['C'] must have the same length.")
    terms=[(float(c), tuple(map(int,state["idx_to_vec"][k])))
           for k,c in enumerate(Ix*C) if c!=0.0]
    def evaluate(delta,x,y,px,py):
        arrays=np.broadcast_arrays(*[np.asarray(v,float) for v in (delta,x,y,px,py)])
        result=np.zeros(arrays[0].shape,float)
        for coeff,powers in terms:
            term=coeff
            for values,power in zip(arrays,powers):
                if power: term=term*values**power
            result=result+term
        return result
    return evaluate


def ix_callable():
    context=_require_invariants("ix_callable()")
    return _make_ix_callable(STATE.Ix, context["state"])


def nonlinear_checks():
    context=_require_context("nonlinear_checks()")
    checks=nl.check_nonlinear_state(context["state"])
    print("="*72); print("NONLINEAR CHECKS"); print("="*72)
    for k,v in checks.items(): print(f"{k:<30}: {v}")
    return checks


def plot_invariant(*, folder=None):
    context=_require_invariants("plot_invariant()"); cfg=_cfg()
    folder=_resolve(cfg.PLOT_ROOT)/"current" if folder is None else _resolve(folder)
    return opt.plot_slices({"Ix":STATE.Ix,"Iy":STATE.Iy}, context["state"], Path(folder), _slice_settings())


# =============================================================================
# FMA + FULL-RING IX TRACKING
# =============================================================================

def _fma_grid(coords_mm, steps):
    if len(coords_mm)!=4 or len(steps)!=2:
        raise ValueError("FMA grid needs coords=[xmin,xmax,ymin,ymax] and steps=[nx,ny].")
    xmin,xmax=sorted(map(float,coords_mm[:2])); ymin,ymax=sorted(map(float,coords_mm[2:]))
    nx,ny=map(int,steps)
    if nx<=0 or ny<=0: raise ValueError("FMA steps must be positive.")
    return (np.linspace(xmin,xmax,nx+1), np.linspace(ymin,ymax,ny+1))


def _infer_physical_ring_cells(parameters):
    lc=STATE.lattice_config
    magnets=lc.define_magnets(parameters)
    by_name={lin.magnet_field(e,"NAME"):e for e in magnets}
    bend=0.0
    for name in lc.CELL_NAMES:
        e=by_name[name]
        if lin.magnet_field(e,"TYPE")=="bending":
            bend += float(lin.magnet_field(e,"ANGLE"))
    if abs(bend)<1e-12:
        raise ValueError("Configured cell has zero net bend; set FMA_PHYSICAL_RING_CELLS.")
    cells_float=360.0/abs(bend); cells=int(round(cells_float))
    if cells<1 or not np.isclose(cells_float,cells,rtol=0,atol=1e-7):
        raise ValueError(f"Cell bend {bend:.12g} deg does not make an integer 360-degree ring.")
    return cells,bend


def _prepare_physical_ring(parameters):
    cfg=_cfg(); lc=STATE.lattice_config
    if cfg.FMA_PHYSICAL_RING_CELLS is None:
        n_cells,cell_bend=_infer_physical_ring_cells(parameters)
    else:
        n_cells=int(cfg.FMA_PHYSICAL_RING_CELLS)
        if n_cells<1: raise ValueError("FMA_PHYSICAL_RING_CELLS must be positive.")
        _,cell_bend=_infer_physical_ring_cells(parameters)
    ring_names=list(lc.CELL_NAMES)*n_cells
    magnets,lattice,data,correction,p=lin.prepare_lattice(
        parameters=dict(parameters), ring_names=ring_names,
        magnet_builder=lc.define_magnets, energy_parameter=lc.ENERGY_PARAMETER,
        correction_parameter_map=lc.CORRECTION_PARAMETER_MAP,
        correct_chromatic=cfg.FMA_CORRECT_CHROMATICITY,
        family1=lc.CHROMATIC_FAMILY1, family2=lc.CHROMATIC_FAMILY2,
        target_chrom_x=lc.TARGET_CHROM_X, target_chrom_y=lc.TARGET_CHROM_Y,
        repetitions=1, step=lc.STEP,
    )
    total_bend=sum(float(lin.magnet_field(e,"ANGLE")) for e in lattice
                   if lin.magnet_field(e,"TYPE")=="bending")
    return {"magnets":magnets,"lattice":lattice,"data":data,"correction":correction,
            "parameters":p,"n_cells":n_cells,"cell_bend_deg":cell_bend,
            "total_bend_deg":total_bend}


def _at_element(elem, at, nsteps):
    f=lin.magnet_field
    name=str(f(elem,"NAME")); typ=str(f(elem,"TYPE")).lower()
    L=float(f(elem,"LENGTH")); angle=float(f(elem,"ANGLE"))
    K=float(f(elem,"K")); S=float(f(elem,"S")); O=float(f(elem,"O"))
    if typ=="drift": return at.Drift(name,L)
    if typ=="quadrupole":
        if S or O:
            b=np.array([0.,K,S,O]); return at.Multipole(name,L,np.zeros_like(b),b,NumIntSteps=nsteps)
        return at.Quadrupole(name,L,k=K,NumIntSteps=nsteps)
    if typ=="sextupole":
        if K or O:
            b=np.array([0.,K,S,O]); return at.Multipole(name,L,np.zeros_like(b),b,NumIntSteps=nsteps)
        return at.Sextupole(name,L,h=S,NumIntSteps=nsteps)
    if typ=="multipole":
        b=np.array([0.,K,S,O]); a=np.zeros_like(b)
        return at.ThinMultipole(name,a,b) if L==0 else at.Multipole(name,L,a,b,NumIntSteps=nsteps)
    if typ=="bending":
        if S or O: raise NotImplementedError(f"Bend {name} contains S or O.")
        return at.Dipole(name,L,bending_angle=math.radians(angle),k=K,NumIntSteps=nsteps)
    raise ValueError(f"Unknown magnet type {typ!r}: {name}")


def _build_at_ring(native, at):
    cfg=_cfg(); lc=STATE.lattice_config
    elements=[_at_element(e,at,int(cfg.FMA_NUM_INT_STEPS)) for e in native["lattice"]]
    energy=float(native["parameters"][lc.ENERGY_PARAMETER])*1e9
    return at.Lattice(elements,name=f"FMA_{cfg.FMA_CASE_LABEL}",energy=energy)


def _initial_row(x_mm,y_mm,orbit,delta):
    orbit=np.asarray(orbit,float).reshape(6)
    z=np.zeros((len(x_mm),6),float); z+=orbit; z[:,4]+=float(delta)
    z[:,0]+=1e-3*np.asarray(x_mm)+1e-9; z[:,2]+=1e-3*float(y_mm)+1e-9
    return z


def _eval_ix_coordinates(fn,c):
    c=np.asarray(c,float)
    return fn(c[4],c[0],c[2],c[1],c[3])


def _track_ix(ring,Ix,state,coords,steps,nturns,orbit,delta,pool_size,floor_fraction):
    from at.tracking import patpass
    fn=_make_ix_callable(Ix,state)
    xs,ys=_fma_grid(coords,steps)
    init_rows=[_initial_row(xs,y,orbit,delta) for y in ys]
    init_ix=[np.asarray(_eval_ix_coordinates(fn,row.T),float).reshape(-1) for row in init_rows]
    all0=np.concatenate(init_ix); ref=float(np.max(np.abs(all0))) if all0.size else 1.0
    if ref==0 or not np.isfinite(ref): ref=1.0
    floor=float(floor_fraction)*ref
    rows=[]
    for y,z0,ix0row in zip(ys,init_rows,init_ix):
        kwargs={"losses":True}
        if pool_size is not None: kwargs["pool_size"]=int(pool_size)
        tracked,loss=patpass(ring,np.asfortranarray(z0.T),int(nturns),**kwargs)
        tracked=np.asarray(tracked,float); tracks=tracked[:,:,0,:]
        lost=np.asarray(loss.get("islost",np.zeros(len(xs),bool)),bool)
        for i,x in enumerate(xs):
            part=tracks[:,i,:]; finite=np.all(np.isfinite(part),axis=0)
            completed=int(np.flatnonzero(~finite)[0]) if not np.all(finite) else int(nturns)
            completed=max(0,min(completed,int(nturns)))
            survived=completed==int(nturns) and not bool(lost[i])
            ix0=float(ix0row[i]); denom=max(abs(ix0),floor)
            if completed:
                vals=np.asarray(_eval_ix_coordinates(fn,part[:,:completed]),float).reshape(-1)
                d=vals-ix0; rel=np.abs(d)/denom
                rate=rel/np.arange(1,completed+1,dtype=float)
                ixlast=float(vals[-1]); maxabs=float(np.max(np.abs(d)))
                maxrel=float(np.max(rel)); rms=float(np.sqrt(np.mean(rel**2)))
                maxrate=float(np.max(rate)); lograte=float(np.log10(max(maxrate,1e-300)))
            else:
                ixlast=maxabs=maxrel=rms=maxrate=lograte=np.nan
            rows.append([x,y,ix0,ixlast,maxabs,maxrel,rms,maxrate,lograte,completed,float(survived)])
    return np.asarray(rows,float)


def _save_csvs(fmap,ix_data,out,label):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    fma_path=out/f"{label}_fma.csv"; ix_path=out/f"{label}_Ix_invariance.csv"
    np.savetxt(fma_path,fmap,delimiter=",",
               header="x_mm,y_mm,nux,nuy,dnux,dnuy,log10_tune_diffusion",comments="")
    np.savetxt(ix_path,ix_data,delimiter=",",
               header=("x_mm,y_mm,Ix_initial,Ix_last,max_abs_delta_Ix,"
                       "max_relative_excursion,rms_relative_excursion,"
                       "max_relative_drift_per_turn,log10_relative_drift_per_turn,"
                       "completed_full_ring_turns,survived"),comments="")
    return fma_path,ix_path


def _tracking_plots(fmap,ix_data,native,out,label,coords,delta,show):
    import matplotlib.pyplot as plt
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    fma_path=out/f"{label}_frequency_map.png"
    if len(fmap):
        fig,ax=plt.subplots(figsize=(8.2,6.6))
        s=ax.scatter(fmap[:,0],fmap[:,1],c=fmap[:,6],s=34,marker="s",vmin=-10,vmax=-2)
        fig.colorbar(s,ax=ax).set_label(r"$\log_{10}$ tune diffusion")
        ax.set(xlabel=r"$x_0$ [mm]",ylabel=r"$y_0$ [mm]",
               title=f"Frequency Map Analysis\n{label} | {native['n_cells']} cells | delta={float(delta):g}")
        ax.set_xlim(coords[0],coords[1]); ax.set_ylim(coords[2],coords[3]); ax.set_aspect("equal"); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(fma_path,dpi=220)
        if show: plt.show()
        else: plt.close(fig)
    else:
        raise RuntimeError("No valid FMA points survived.")

    ix_path=out/f"{label}_Ix_invariance_map.png"
    valid=(ix_data[:,10]>.5)&np.isfinite(ix_data[:,8])
    if not np.any(valid): raise RuntimeError("No particle survived the Ix tracking interval.")
    shown=ix_data[valid]; cfg=_cfg()
    fig,ax=plt.subplots(figsize=(8.2,6.6))
    s=ax.scatter(shown[:,0],shown[:,1],c=shown[:,8],s=34,marker="s",
                 vmin=float(cfg.IX_INVARIANCE_LOG_MIN),vmax=float(cfg.IX_INVARIANCE_LOG_MAX))
    fig.colorbar(s,ax=ax).set_label(r"$\log_{10}$ max relative $I_x$ drift / ring turn")
    ax.set(xlabel=r"$x_0$ [mm]",ylabel=r"$y_0$ [mm]",
           title=f"Horizontal Invariant Tracking\n{label} | physical ring={native['n_cells']} cells")
    ax.set_xlim(coords[0],coords[1]); ax.set_ylim(coords[2],coords[3]); ax.set_aspect("equal"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ix_path,dpi=220)
    if show: plt.show()
    else: plt.close(fig)
    return fma_path,ix_path


def run_fma(*, stage="current", quick=False):
    """Run FMA and full-ring Ix tracking using the active lattice and Ix."""
    context=_require_invariants("run_fma()"); cfg=_cfg()
    if importlib.util.find_spec("at") is None:
        raise ImportError('Install tracking support with: python -m pip install -e ".[tracking]"')
    import at
    from at.physics import find_orbit
    from at.physics.frequency_maps import fmap_parallel_track

    coords=list(cfg.FMA_COORDS_MM); steps=list(cfg.FMA_STEPS); turns=int(cfg.FMA_TURNS)
    if quick:
        steps=[min(5,int(steps[0])),min(5,int(steps[1]))]; turns=min(8,turns)

    native=_prepare_physical_ring(context["parameters"])
    if not np.isclose(abs(native["total_bend_deg"]),360.0,atol=1e-6):
        raise ValueError(f"Physical tracking ring bends {native['total_bend_deg']} deg, not 360.")
    ring=_build_at_ring(native,at); orbit,_=find_orbit(ring)
    offset=np.zeros(6); offset[4]=float(cfg.FMA_DELTA)
    kwargs={}
    if cfg.FMA_POOL_SIZE is not None: kwargs["pool_size"]=int(cfg.FMA_POOL_SIZE)
    fmap,losses=fmap_parallel_track(
        ring,coords=coords,steps=steps,turns=turns,orbit=orbit,
        add_offset6D=offset,verbose=True,lossmap=True,**kwargs
    )
    ix_data=_track_ix(
        ring,STATE.Ix,context["state"],coords,steps,2*turns,orbit,cfg.FMA_DELTA,
        cfg.FMA_POOL_SIZE,cfg.IX_INVARIANCE_NORM_FLOOR_FRACTION
    )
    stage=str(stage); label=f"{cfg.FMA_CASE_LABEL}_{stage}"
    out=_resolve(cfg.FMA_OUTPUT_DIRECTORY)/stage
    fma_csv,ix_csv=_save_csvs(fmap,ix_data,out,label)
    fma_plot,ix_plot=_tracking_plots(fmap,ix_data,native,out,label,coords,cfg.FMA_DELTA,bool(cfg.FMA_SHOW_PLOT))
    result={"fmap":fmap,"losses":losses,"ix_data":ix_data,"ring":ring,"native":native,
            "fma_data_path":fma_csv,"ix_data_path":ix_csv,
            "frequency_plot_path":fma_plot,"ix_plot_path":ix_plot,
            "output_directory":out,"case_label":label,"quick":bool(quick)}
    STATE.diagnostics[stage]=result
    return result


# =============================================================================
# REPORT WRITERS
# =============================================================================

def _write_text(path,text):
    path=_resolve(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text.rstrip()+"\n",encoding="utf-8"); return path


def write_linear_report(file_name=None):
    summary=linear_summary(); checks=linear_checks()
    file_name=_output_root()/"reports"/"linear_report.txt" if file_name is None else file_name
    lines=["="*90,"LINEAR OPTICS REPORT","="*90,"","SUMMARY","-"*90]
    lines += [f"{k:<30} {v}" for k,v in summary.items()]
    lines += ["","CONSISTENCY CHECKS","-"*90]
    lines += [f"{k:<30} {v}" for k,v in checks.items()]
    return _write_text(file_name,"\n".join(lines))


def write_invariant_report(file_name=None):
    context=_require_invariants("write_invariant_report()")
    checks=nl.check_nonlinear_state(context["state"])
    details=STATE.invariant_details or {}
    file_name=_output_root()/"reports"/"invariant_report.txt" if file_name is None else file_name
    lines=["="*90,"NONLINEAR INVARIANT REPORT","="*90,"",
           f"Ix coefficients       : {len(STATE.Ix)}",
           f"Iy coefficients       : {len(STATE.Iy)}",
           f"Ix finite             : {bool(np.all(np.isfinite(STATE.Ix)))}",
           f"Iy finite             : {bool(np.all(np.isfinite(STATE.Iy)))}"]
    for key in ("objective","value_norm","gradient_norm"):
        if key in details: lines.append(f"{key:<22}: {details[key]}")
    lines += ["","NONLINEAR STATE CHECKS","-"*90]
    lines += [f"{k:<30} {v}" for k,v in checks.items()]
    return _write_text(file_name,"\n".join(lines))


def _optimization_report_text(result):
    cfg=_cfg(); lc=STATE.lattice_config
    sp=result["start_parameters"]; fp=result["final_parameters"]
    sm=result["start_snapshot"]; fm=result["final_snapshot"]
    f=lambda x:f"{float(x):.16e}"
    lines=["="*90,"NONLINEAR OPTIMIZATION REPORT","="*90,"",
           "OBJECTIVE","-"*90,
           f"gradient weight = {cfg.GRADIENT_WEIGHT}",
           f"initial J       = {f(result['start_details']['objective'])}",
           f"final J         = {f(result['final_details']['objective'])}","",
           "OPTIMIZED PARAMETERS","-"*90,
           f"{'parameter':<14}{'initial':>24}{'final':>24}{'change':>24}"]
    for name in cfg.VARY:
        a,b=float(sp[name]),float(fp[name]); lines.append(f"{name:<14}{f(a):>24}{f(b):>24}{f(b-a):>24}")
    lines += ["","MAGNET CHANGES","-"*90,
              f"{'parameter':<12}{'magnet':<12}{'field':<10}{'initial':>22}{'final':>22}{'change':>22}"]
    keys={"LENGTH":"length","ANGLE":"angle","K":"K","S":"S","O":"O"}
    for parameter in cfg.VARY:
        for magnet,field in lc.PARAMETER_MAP.get(parameter,[]):
            key=keys[field.upper()]; a=sm[magnet][key]; b=fm[magnet][key]
            lines.append(f"{parameter:<12}{magnet:<12}{field:<10}{f(a):>22}{f(b):>22}{f(b-a):>22}")
    lines += ["","CHROMATIC CORRECTION","-"*90]
    for label,corr in (("initial",result["start_correction"]),("final",result["final_correction"])):
        if corr is None: lines.append(f"{label}: none")
        else: lines.append(f"{label}: {corr[0]}={f(corr[1])}, {corr[2]}={f(corr[3])}, chrom=({f(corr[4])}, {f(corr[5])})")
    lines += ["","PLOTS","-"*90]
    for stage in ("start","end"):
        diag=result.get(f"fma_{stage}")
        if diag: lines += [f"{stage} FMA: {diag['frequency_plot_path']}",f"{stage} Ix : {diag['ix_plot_path']}"]
    return "\n".join(lines)


def write_optimization_report(file_name=None,result=None):
    result=STATE.optimization_result if result is None else result
    if result is None: raise RuntimeError("No optimization result. Run optimize() first.")
    if file_name is None: file_name=getattr(_cfg(),"REPORT_FILE",_output_root()/"reports"/"optimization_report.txt")
    return _write_text(file_name,_optimization_report_text(result))


def write_tracking_report(stage="current",file_name=None):
    stage=str(stage); diag=STATE.diagnostics.get(stage)
    if diag is None: raise RuntimeError(f"No tracking result named {stage!r}. Run run_fma(stage={stage!r}) first.")
    file_name=_output_root()/"reports"/f"tracking_{stage}_report.txt" if file_name is None else file_name
    fmap=np.asarray(diag["fmap"]); data=np.asarray(diag["ix_data"])
    valid=(data[:,10]>.5)&np.isfinite(data[:,8])
    lines=["="*90,f"FMA + IX TRACKING REPORT: {stage}","="*90,"",
           f"Physical ring cells      : {diag['native']['n_cells']}",
           f"Total bend [deg]         : {diag['native']['total_bend_deg']}",
           f"Valid FMA points         : {len(fmap)} / {len(data)}"]
    if len(fmap):
        lines += [f"Median log diffusion     : {np.median(fmap[:,6]):.16e}",
                  f"Best log diffusion       : {np.min(fmap[:,6]):.16e}",
                  f"Worst log diffusion      : {np.max(fmap[:,6]):.16e}"]
    lines.append(f"Valid Ix points          : {np.count_nonzero(valid)} / {len(data)}")
    if np.any(valid):
        v=data[valid,8]
        lines += [f"Median log Ix drift      : {np.median(v):.16e}",
                  f"Best log Ix drift        : {np.min(v):.16e}",
                  f"Worst log Ix drift       : {np.max(v):.16e}"]
    lines += ["",f"FMA data: {diag['fma_data_path']}",f"FMA plot: {diag['frequency_plot_path']}",
              f"Ix data : {diag['ix_data_path']}",f"Ix plot : {diag['ix_plot_path']}"]
    return _write_text(file_name,"\n".join(lines))


def write_full_report(file_name=None):
    _require_context("write_full_report()")
    file_name=_output_root()/"reports"/"full_report.txt" if file_name is None else file_name
    summary=linear_summary(); checks=linear_checks()
    lines=["="*90,"FULL ACCELERATOR ANALYSIS REPORT","="*90,"","LINEAR OPTICS","-"*90]
    lines += [f"{k:<30} {v}" for k,v in summary.items()]
    lines += ["","LINEAR CHECKS","-"*90] + [f"{k:<30} {v}" for k,v in checks.items()]
    if STATE.Ix is not None:
        lines += ["","NONLINEAR INVARIANTS","-"*90,
                  f"Ix coefficients: {len(STATE.Ix)}",f"Iy coefficients: {len(STATE.Iy)}"]
        for key in ("objective","value_norm","gradient_norm"):
            if STATE.invariant_details and key in STATE.invariant_details:
                lines.append(f"{key}: {STATE.invariant_details[key]}")
    if STATE.optimization_result is not None:
        lines += ["",_optimization_report_text(STATE.optimization_result)]
    for stage,diag in sorted(STATE.diagnostics.items()):
        fmap=np.asarray(diag["fmap"]); data=np.asarray(diag["ix_data"])
        valid=(data[:,10]>.5)&np.isfinite(data[:,8])
        lines += ["",f"TRACKING: {stage}","-"*90,
                  f"valid FMA points: {len(fmap)} / {len(data)}",
                  f"valid Ix points : {np.count_nonzero(valid)} / {len(data)}",
                  f"FMA plot        : {diag['frequency_plot_path']}",
                  f"Ix plot         : {diag['ix_plot_path']}"]
    return _write_text(file_name,"\n".join(lines))


# =============================================================================
# OPTIMIZATION
# =============================================================================

def optimize(*,run_start_end_fma=None,quick=False):
    """Run the configured hybrid optimizer; optimized state stays active."""
    if STATE.context is None: load()
    cfg=_cfg()
    if importlib.util.find_spec("cma") is None:
        raise ImportError("optimize() requires cma. Install the project with: python -m pip install -e .")
    do_fma=bool(cfg.RUN_FMA_START_END) if run_start_end_fma is None else bool(run_start_end_fma)
    if do_fma and importlib.util.find_spec("at") is None:
        raise ImportError('Tracking is enabled. Install with: python -m pip install -e ".[tracking]" or use optimize(run_start_end_fma=False).')

    context=STATE.context; v0=opt.initial_vector(cfg.VARY,context["parameters"])
    start=opt.full_diagnostics(context,cfg.GRADIENT_WEIGHT,cfg.LEAST_SQUARES_TOL)
    _set_invariants(start)
    fma_start=run_fma(stage="start",quick=quick) if do_fma else None

    cma_time=float(cfg.CMA_TIME); pop=cfg.CMA_POPSIZE; pfrac=float(cfg.POWELL_TIME_FRACTION)
    if quick:
        cma_time=min(cma_time,3.0)
        if pop is not None: pop=min(int(pop),4)
        pfrac=min(pfrac,.10)

    print("="*80); print("STARTING NONLINEAR OPTIMIZATION"); print("="*80)
    if quick: print("Mode: QUICK TUTORIAL / SMOKE TEST")
    result=opt.hybrid_optimize(
        context,v0,cfg.VARY,gradient_weight=cfg.GRADIENT_WEIGHT,tol=cfg.LEAST_SQUARES_TOL,
        invalid_penalty=cfg.INVALID_PENALTY,sigma=cfg.CMA_SIGMA,scales=cfg.SCALES,
        cma_time=cma_time,popsize=pop,print_every=cfg.PRINT_EVERY,
        powell_time_fraction=pfrac,plot_start_end_slices=cfg.PLOT_START_END_SLICES,
        plot_root=_resolve(cfg.PLOT_ROOT),slice_settings=_slice_settings()
    )
    STATE.context=result["context"]; _set_invariants(result["final_details"])
    STATE.source="optimized"
    fma_end=run_fma(stage="end",quick=quick) if do_fma else None
    result["fma_start"]=fma_start; result["fma_end"]=fma_end
    STATE.optimization_result=result

    opt.save_final_lattice(_resolve(cfg.FINAL_LATTICE_FILE),STATE.context)
    write_optimization_report(result=result)
    if fma_start is not None: write_tracking_report("start")
    if fma_end is not None: write_tracking_report("end")
    return result


def save_current_lattice(file_name=None):
    context=_require_context("save_current_lattice()"); cfg=_cfg()
    file_name=cfg.FINAL_LATTICE_FILE if file_name is None else file_name
    return opt.save_final_lattice(_resolve(file_name),context)
