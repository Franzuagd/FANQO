"""Compare saved coefficients on identical saved trajectories, by delta."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,argparse
from pathlib import Path
import numpy as np
from track_fit import features
from fanqo import comparison as cp
from fanqo.core import nonlinear as nl,structured_invariants as si
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--config',required=True);a=p.parse_args()
root=Path(a.source);r=json.load(open(root/'report.json'));arr=np.load(root/'coefficients.npz');tr=np.load(root/'trajectories.npz')['test']
cfg,lc=cp._configuration(a.config);_,_,data,_,_=cp._linear(cfg,lc)
s=nl.initialize_nonlinear_for_method(data,r['settings']['order'],1,cfg.HAMILTONIAN,cfg.A_BOX,cfg.VARIABLES,cfg.FIELD_SYMBOLS,invariant_construction='graded_coupled');sx,_=si.seeds(data,s)
names=list(r['test_results']);C=np.column_stack([arr[k] for k in names]);errs=[];legacy=[]
for start in range(0,tr.shape[1],16):
 t=tr[:,start:start+16];P=features(t,s).reshape(t.shape[1],t.shape[2],-1);v=P@C
 scale=np.maximum(P[:,0]@sx,1e-14);d=abs(v[:,1:]-v[:,:1]);errs.append(d.max(axis=1)/scale[:,None])
 legacy.append((d/np.arange(1,v.shape[1])[None,:,None]).max(axis=1)/np.maximum(abs(v[:,0]),1e-20))
E=np.concatenate(errs);L=np.concatenate(legacy);labels=tr[4,:,0];out={}
for i,n in enumerate(names):
 out[n]={str(d):{'median':float(np.median(E[labels==d,i])),'p90':float(np.quantile(E[labels==d,i],.9)),
                  'legacy_median':float(np.median(L[labels==d,i])),'survivors':int(sum(labels==d))} for d in np.unique(labels)}
(root/'per_delta_metrics.json').write_text(json.dumps(out,indent=2))
print('Saved per-delta metrics',root)
