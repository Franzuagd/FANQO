"""Post-selection diagnostic: reject a purely flat-observable interpretation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np
from track_fit import features
from fanqo import comparison as cp
from fanqo.core import nonlinear as nl,structured_invariants as si,linear as lin
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--config',required=True);a=p.parse_args()
root=Path(a.source);cfg,lc=cp._configuration(a.config);_,_,data,_,_=cp._linear(cfg,lc)
arrays=np.load(root/'selected_coefficients.npz');powers=arrays['powers'];m=int(powers[:,1:].sum(axis=1).max())
s=nl.initialize_nonlinear_for_method(data,m,1,cfg.HAMILTONIAN,cfg.A_BOX,cfg.VARIABLES,cfg.FIELD_SYMBOLS,invariant_construction='graded_coupled')
sx,sy=si.seeds(data,s);tr=np.load(root/'test_trajectories.npz')['test'];P0=features(tr[:,:,0],s)
cs=lin.linear_data(data,'CS0');metric_scale=np.sqrt(np.maximum(P0@(sx+sy),1e-14))
result={};norms={};geom={}
for name in ['eigen','selected']:
 c=arrays[name];g=[P0@(nl.build_derivative_matrix(s,k)@c) for k in [1,3,2,4]]
 bx,ax,_,by,ay,_=cs
 norm=np.sqrt((np.sqrt(bx)*g[0]-ax/np.sqrt(bx)*g[1])**2+(g[1]/np.sqrt(bx))**2+
              (np.sqrt(by)*g[2]-ay/np.sqrt(by)*g[3])**2+(g[3]/np.sqrt(by))**2)
 from fanqo.comparison import _evaluate
 values=_evaluate(c,s,tr)
 drift=np.max(abs(values-values[:,:1]),axis=1)
 denom=np.maximum(norm*metric_scale,1e-12*np.max(norm*metric_scale))
 e=drift/denom;geom[name]=e;norms[name]=norm
 result[name]={'median_geometric_defect':float(np.median(e)),'p90_geometric_defect':float(np.quantile(e,.9)),
               'gradient_norm_quantiles':np.quantile(norm,[.1,.5,.9]).tolist()}
result['selected_to_eigen_gradient_ratio_quantiles']=np.quantile(norms['selected']/np.maximum(norms['eigen'],1e-30),[.1,.5,.9]).tolist()
result['geometric_win_fraction']=float(np.mean(geom['selected']<geom['eigen']))
result['scope']='First-order normal-displacement proxy using initial full transverse gradient in CS coordinates; not an exact finite-displacement distance.'
(root/'geometry.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
