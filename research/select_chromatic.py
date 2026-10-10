"""Select a near-unit eigenvector by held-out physical invariance.
Uses validation trajectories from track_fit.py; creates a NEW confirmation set.
The polynomial order and algebraic eigenvector candidates remain unchanged.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
import at
from fanqo import comparison as cp
from fanqo.api import _at_element
from fanqo.core import nonlinear as nl,linear as lin,structured_invariants as si
from track_fit import features,tracks,score,matrices

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--config',required=True)
 ap.add_argument('--output',required=True);ap.add_argument('--seed',type=int,default=99101)
 ap.add_argument('--test',type=int,default=64);ap.add_argument('--turns',type=int,default=512)
 args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 source=Path(args.source);r=json.load(open(source/'report.json'));cfg,lc=cp._configuration(args.config)
 magnets,lattice,data,_,params=cp._linear(cfg,lc);m=r['settings']['order'];d=1
 es=nl.initialize_nonlinear_for_method(data,m,d,cfg.HAMILTONIAN,cfg.A_BOX,cfg.VARIABLES,cfg.FIELD_SYMBOLS,invariant_construction='eigen')
 base=nl.initialize_nonlinear_for_method(data,m,d,cfg.HAMILTONIAN,cfg.A_BOX,cfg.VARIABLES,cfg.FIELD_SYMBOLS,invariant_construction='graded_coupled')
 T,_,_=nl.nonlinear_transfer(lattice,es);e,details=nl.eigen_invariant(T,es);eig=e/base['C']
 sx,sy=si.seeds(data,base);N=len(T);vals,vecs=np.linalg.eig(T-np.eye(N))
 # Retain exactly the same spectral window as the reference constructor.
 cutoff=details['eigen_cutoff'];lookup=es['vec_to_idx']
 x2=lookup[(0,2,0,0,0)];xp=lookup[(0,1,0,1,0)];p2=lookup[(0,0,0,2,0)]
 cs=[eig];metadata=[{'kind':'reference','eigenvalue':None}]
 for k,v in enumerate(vals):
  if abs(v.imag)>1e-12 or not 0<v.real<cutoff:continue
  c=vecs[:,k].real.copy();det=c[x2]*c[p2]-(c[xp]/2)**2
  if not np.isfinite(det) or det<=0:continue
  c*=np.sign(c[p2])/np.sqrt(det);c/=base['C']
  cs.append(c);metadata.append({'kind':'spectral_candidate','eigenvalue':float(v.real)})
 # Fit combinations inside the near-unit spectral space, with the entire
 # reference quadratic block fixed. Selection still uses validation only.
 spectral_count=len(cs)
 train=np.load(source/'trajectories.npz')['train']
 D,V=matrices(train,base,sx)
 # Include pure higher-order near-unit modes even when they have no
 # normalizable quadratic part: these are precisely the action-gauge freedom.
 modes=[eig]
 for k,v in enumerate(vals):
  if abs(v)<=1e-8:
   for component in (vecs[:,k].real,vecs[:,k].imag):
    if np.linalg.norm(component)>1e-12:modes.append(component/base['C'])
 C=np.column_stack(modes)
 # Column scaling limits numerical rank decisions dominated by enormous modes.
 scale=np.maximum(np.linalg.norm(V@C,axis=0),1e-30)
 C=C/scale
 fixed=[i for i,a in base['idx_to_vec'].items() if a[0]==0 or not sum(a[1:])]
 Z=null_space(C[fixed],rcond=1e-10)
 B=C@Z
 B[fixed]=0. # Enforce the on-momentum polynomial exactly, including roundoff.
 if B.shape[1]:
  A=D@B;E=V@B
  # Use SVD least squares on an augmented system rather than normal equations.
  for lam in (1e-6,1e-4,.01,1.,100.):
   design=np.vstack([A/np.sqrt(len(A)),np.sqrt(lam)*E/np.sqrt(len(E))])
   rhs=np.r_[-D@eig/np.sqrt(len(A)),np.zeros(len(E))]
   u=np.linalg.lstsq(design,rhs,rcond=1e-10)[0]
   cs.append(eig+B@u);metadata.append({'kind':'spectral_subspace_fit','lambda':lam,
        'quadratic_max_error':float(np.max(abs((B@u)[:base['quad_size']])))})
 print('Near-unit basis dimension',C.shape[1], 'free subspace',B.shape[1],flush=True)
 print('Spectral candidates',spectral_count,'plus constrained fits',len(cs)-spectral_count,flush=True)
 val=np.load(source/'trajectories.npz')['validation'];labels=val[4,:,0]
 P=features(val,base).reshape(val.shape[1],val.shape[2],N)
 sigma=np.maximum(P[:,0]@sx,1e-14)
 errors=[];legacy=[];radials=[];stats=[]
 # Require a horizontal quadratic action: limit vertical/mixed quadratic norm.
 q=base['quad_size'];Gq=base['G'][:q,:q]
 for i,c in enumerate(cs):
  v=P@c;err=np.max(abs(v-v[:,:1]),axis=1)/sigma
  errors.append(err)
  legacy.append(np.max(abs(v[:,1:]-v[:,:1])/np.arange(1,v.shape[1]),axis=1)/np.maximum(abs(v[:,0]),1e-12*max(abs(v[:,0]))))
  cross=c[:q].copy();cross[[x2,xp,p2]]=0
  ratio=si._norm(cross,Gq)/si._norm(c[:q],Gq)
  stats.append({'index':i,'median':float(np.median(err)),'p90':float(np.quantile(err,.9)),
                'legacy_median':float(np.median(legacy[-1])),'nonhorizontal_quadratic_fraction':ratio,**metadata[i]})
 ref=stats[0]
 eligible=[s for s in stats if (s['index']==0 or
  (s['kind']=='spectral_subspace_fit' and s['nonhorizontal_quadratic_fraction']<=max(.25,ref['nonhorizontal_quadratic_fraction'])
   and s['median']<=ref['median'] and s['legacy_median']<=ref['legacy_median']))]
 best=min(eligible,key=lambda s:s['p90']);selected=cs[best['index']]
 (out/'selection.json').write_text(json.dumps({'reference':ref,'selected':best,'all':stats},indent=2))
 np.savez_compressed(out/'selected_coefficients.npz',eigen=eig,selected=selected,C=base['C'],powers=np.array(list(base['idx_to_vec'].values())))
 del P
 print('Frozen selected candidate',best,flush=True)
 byname={lin.magnet_field(e,'NAME'):e for e in magnets};cell=[byname[n] for n in lc.CELL_NAMES]
 ncells=int(round(360/abs(sum(e[3] for e in cell))))
 ring=at.Lattice([_at_element(e,at,cfg.FMA_NUM_INT_STEPS) for e in cell*ncells],energy=params[lc.ENERGY_PARAMETER]*1e9)
 test,labels,survival=tracks(ring,lin.linear_data(data,'CS0'),args.test,args.turns,args.seed,r['settings']['extent'],r['settings']['deltas'])
 report={'settings':vars(args),'selection':best,'survival':survival,'results':{}}
 errors={}
 for name,c in [('eigen',eig),('selected_eigen',selected)]:
  sc,err=score(c,test,base,sx,eig,labels);report['results'][name]=sc;errors[name]=err
  print(name,sc,flush=True)
 ratio=errors['selected_eigen']/np.maximum(errors['eigen'],1e-30);rng=np.random.default_rng(811)
 bs=np.median(ratio[rng.integers(0,len(ratio),(2000,len(ratio)))],axis=1)
 report['paired_ratio']={'median':float(np.median(ratio)),'bootstrap95':np.quantile(bs,[.025,.975]).tolist()}
 (out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'test_errors.npz',**errors,delta=labels)
 np.savez_compressed(out/'test_trajectories.npz',test=test)
 print('Saved',out,flush=True)
if __name__=='__main__':main()
