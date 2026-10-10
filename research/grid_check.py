"""Zero-momentum FMA-style grid, shared by all saved constructions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import at
from fanqo import comparison as cp
from fanqo.api import _at_element,_initial_row
from fanqo.core import nonlinear as nl,linear as lin,structured_invariants as si
from track_fit import features
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--selected',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
p.add_argument('--points',type=int,default=9);p.add_argument('--turns',type=int,default=256);p.add_argument('--extent',type=float,default=4.);a=p.parse_args()
root=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);r=json.load(open(root/'report.json'))
arr=np.load(root/'coefficients.npz');new=np.load(Path(a.selected)/'selected_coefficients.npz')
names=['eigen','LS_CS_seed','LS_eigen_seed','a_box_coupled_fixed','a_box_coupled_regularized','graded_coupled','canonical_graded','kernel_fit']
C=np.column_stack([arr[n] for n in names[:-1]]+[new['selected']]);cfg,lc=cp._configuration(a.config)
magnets,lattice,data,_,params=cp._linear(cfg,lc);s=nl.initialize_nonlinear_for_method(data,r['settings']['order'],1,cfg.HAMILTONIAN,cfg.A_BOX,cfg.VARIABLES,cfg.FIELD_SYMBOLS,invariant_construction='graded_coupled');sx,_=si.seeds(data,s)
byname={e[0]:e for e in magnets};cell=[byname[n] for n in lc.CELL_NAMES];nc=int(round(360/abs(sum(e[3] for e in cell))))
ring=at.Lattice([_at_element(e,at,cfg.FMA_NUM_INT_STEPS) for e in cell*nc],energy=params[lc.ENERGY_PARAMETER]*1e9)
orbit,_=at.find_orbit4(ring,dp=0.)
xs=np.linspace(-a.extent,a.extent,a.points);xx,yy=np.meshgrid(xs,xs);records=[]
for delta in r['settings']['deltas']:
 print('Grid delta',delta,flush=True)
 z0=np.concatenate([_initial_row(xs,y,orbit,delta) for y in xs]).T
 t=at.lattice_pass(ring,np.asfortranarray(z0.copy()),a.turns)[:,:,0,:];tr=np.concatenate([z0[:,:,None],t],axis=2)
 P0=features(z0,s);initial_values=P0@C;initial_sx=P0@sx
 common_scale=np.maximum(initial_sx,1e-6*max(float(initial_sx.max()),1e-14))
 relative_scale=np.maximum(abs(initial_values),1e-12*np.maximum(np.max(abs(initial_values),axis=0),1e-300))
 alive=np.isfinite(tr).all(axis=(0,2));ids=np.flatnonzero(alive);E=np.full((xx.size,len(names)),np.nan);L=E.copy()
 for start in range(0,len(ids),16):
  j=ids[start:start+16];tt=tr[:,j];P=features(tt,s).reshape(len(j),a.turns+1,-1);values=P@C
  scale=common_scale[j]
  drift=abs(values[:,1:]-values[:,:1]);E[j]=drift.max(axis=1)/scale[:,None]
  L[j]=(drift/np.arange(1,a.turns+1)[None,:,None]).max(axis=1)/relative_scale[j]
 rec={'delta':delta,'survived':len(ids),'launched':xx.size,'methods':{}}
 for i,n in enumerate(names):
  rec['methods'][n]={'median':float(np.nanmedian(E[:,i])),'p90':float(np.nanquantile(E[:,i],.9)),
                     'legacy_median':float(np.nanmedian(L[:,i]))}
 records.append(rec);np.savez_compressed(out/f'delta_{delta:+.3f}.npz',max_drift=E,legacy_rate=L,survived=alive,x_mm=xx,y_mm=yy,names=names)
 fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
 for ax,i in zip(axes.flat,range(len(names))):
  im=ax.pcolormesh(xx,yy,np.log10(np.maximum(E[:,i].reshape(xx.shape),1e-12)),vmin=-5,vmax=3,cmap='viridis',shading='auto');ax.set(title=names[i],xlabel='x [mm]',ylabel='y [mm]')
 fig.colorbar(im,ax=list(axes.flat),label='log10 maximum drift / common CS action')
 fig.suptitle(f'OPA, delta={delta:+.2f}; {len(ids)}/{xx.size} survived; {a.turns} turns')
 fig.savefig(out/f'delta_{delta:+.3f}.png',dpi=150);plt.close(fig)
(out/'report.json').write_text(json.dumps({'settings':vars(a),'launch':'on-momentum orbit plus delta; px=py=0 offsets','results':records},indent=2))
