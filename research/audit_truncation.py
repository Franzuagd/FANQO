"""Show rectangular-boundary sensitivity under a dispersive Hamiltonian."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,numpy as np,sympy as sp
from pathlib import Path
from fanqo.core import nonlinear as nl
v=sp.symbols('delta x y px py');d,x,y,p,py=v;f=sp.symbols('b1 b2 b3 b4 b5')
H=p*p/2+f[2]*x**3/3-f[0]*d*x
states=[];maps=[]
for m in (4,5):
 s=nl.initialize_nonlinear([np.array([1,0,1,1,0,1.])],m,1,H,[1,1,0,1,0],v,f)
 T=nl.element_transfer(['test','bending',.5,10.,0.,2.,0.,None,None],s)[0]
 states.append(s);maps.append(s['C'][:,None]*T/s['C'][None,:])
a,b=states;idx=[b['vec_to_idx'][tuple(p)] for p in a['idx_to_vec'].values()]
diff=maps[1][np.ix_(idx,idx)]-maps[0]
loc=np.unravel_index(np.argmax(abs(diff)),diff.shape)
r={'max_retained_coefficient_change':float(abs(diff[loc])),
   'output_exponent':a['idx_to_vec'][loc[0]],'input_exponent':a['idx_to_vec'][loc[1]],
   'interpretation':'Increasing transverse workspace by one changes retained delta terms. This is not roundoff; delta-linear Hamiltonian terms can lower transverse degree.'}
Path(__file__).with_name('audit_truncation.json').write_text(json.dumps(r,indent=2));print(r)
