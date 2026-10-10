"""Independent physics checks on the supplied OPA lattice."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
import sympy as sp
import at
from scipy.linalg import expm
from fanqo import comparison as cp
from fanqo.api import _at_element
from fanqo.core import nonlinear as nl, linear as lin
cfg,lc=cp._configuration(str(Path(__file__).with_name('opa_config.py')))
magnets,lattice,data,corr,params=cp._linear(cfg,lc)
cell=at.Lattice([_at_element(e,at,40) for e in lattice],energy=params['energy']*1e9)
lindata,ringdata,_=at.get_optics(cell,method=at.linopt4,get_chrom=True)
M=lin.linear_data(data,'LATTICE_M4');J=np.kron(np.eye(2),[[0,1],[-1,0]])
M_at,_=at.find_m44(cell,orbit=np.zeros(6))
report={'native_linear_symplectic_error':float(np.linalg.norm(M.T@J@M-J)),
        'AT_linear_matrix_relative_error':float(np.linalg.norm(M-M_at)/np.linalg.norm(M)),
        'native_beta':lin.linear_data(data,'CS0')[[0,3]].tolist(),
        'AT_beta':lindata.beta.tolist(),'native_cell_chromaticity':[lin.linear_data(data,'CHROM_X'),lin.linear_data(data,'CHROM_Y')],
        'AT_cell_chromaticity':ringdata.chromaticity.tolist(),'AT_cell_tune':ringdata.tune.tolist(),
        'chromatic_strengths_before':{k:lc.PARAMETERS[k] for k in ('ksf1','ksd1')},
        'chromatic_strengths_after':{k:params[k] for k in ('ksf1','ksd1')},
        'cell_bend_degrees':sum(e[3] for e in lattice),'cell_length':sum(e[2] for e in lattice)}
# Small horizontal basis: exact polynomial substitution for a thin kick.
v=cfg.VARIABLES;delta,x,y,px,py=v
state=nl.initialize_nonlinear([np.array([1,0,1,1,0,1.])],8,1,cfg.HAMILTONIAN,[.03,.01,0,.001,0],v,cfg.FIELD_SYMBOLS)
e=['O','multipole',0.,0.,0.,0.,2.,None,None]
T=nl.element_transfer(e,state)[0];j=state['vec_to_idx'][(0,0,0,2,0)]
c=np.zeros(len(T));c[j]=1/state['C'][j]
actual=T@c
expected=np.asarray(nl.poly_to_vector((px+2*x**3)**2,v,state['vec_to_idx']),float)/state['C']
old=np.eye(len(T))-2*state['M_octupole_unit']
report['thin_octupole_px2_corrected_max_coefficient_error']=float(np.max(abs((actual-expected)*state['C'])))
report['thin_octupole_px2_legacy_x6_coefficient']=float((old@c)[state['vec_to_idx'][(0,6,0,0,0)]]*state['C'][state['vec_to_idx'][(0,6,0,0,0)]])
report['thin_octupole_px2_expected_x6_coefficient']=4.
# Native/AT kick convention, all transverse coordinates nonzero.
z=np.array([.002,.0003,.001,-.0002,.01,0.]);z0=z.copy()
at.element_pass(_at_element(e,at,10),z.reshape(6,1))
expected_z=z0.copy();expected_z[1]-=2*(z0[0]**3-3*z0[0]*z0[2]**2);expected_z[3]+=2*(3*z0[0]**2*z0[2]-z0[2]**3)
report['AT_octupole_kick_error']=float(np.max(abs(z-expected_z)))
# Changing constant central terms cannot change absolute tracking drift.
report['remarks']=['AT uses 40 slices for this linear cross-check. Native correction uses STEP=0.01.',
 'A unit-square polynomial norm is not the same norm as the physical A_BOX norm.',
 'Off-momentum closed-orbit launches in research differ from production zero-orbit offsets.']
Path(__file__).with_name('audit_physics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
