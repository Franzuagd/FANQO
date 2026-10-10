import os
import numpy as np
import sympy as sp
from fanqo.config_loader import load_python_file
from fanqo.core import nonlinear as nl


def test_config_force_reload_bypasses_same_size_timestamp_cache(tmp_path):
    p=tmp_path/'config.py';p.write_text("METHOD='a_box'\n")
    stamp=p.stat().st_mtime
    assert load_python_file(p).METHOD=='a_box'
    p.write_text("METHOD='eigen'\n");os.utime(p,(stamp,stamp))
    assert load_python_file(p,reload=True).METHOD=='eigen'


def test_thin_octupole_keeps_squared_kick_in_polynomial_observable():
    v=sp.symbols('delta x y px py');delta,x,y,px,py=v
    fields=sp.symbols('b1 b2 b3 b4 b5')
    H=(px**2+py**2)/2+fields[3]*(x**4-6*x*x*y*y+y**4)/4
    state=nl.initialize_nonlinear([np.array([1,0,1,1,0,1.])],8,0,H,[1,1,0,1,0],v,fields)
    c=np.zeros(len(state['C']));c[state['vec_to_idx'][(0,0,0,2,0)]]=1/state['C'][state['vec_to_idx'][(0,0,0,2,0)]]
    T=nl.element_transfer(['O','multipole',0.,0.,0.,0.,2.,None,None],state)[0]
    expected=np.asarray(nl.poly_to_vector((px+2*x**3)**2,v,state['vec_to_idx']),float)
    assert np.allclose((T@c)*state['C'],expected,atol=1e-12)


def test_zero_nonlinearity_has_zero_fluctuation_objective():
    from fanqo.core.objective_functions import reference_fluctuation_index
    z=np.zeros((1,1))
    state={'idx_to_vec':{0:[0,0,0,0,0]},'C':np.ones(1),
           'D_x':z,'D_y':z,'D_px':z,'D_py':z}
    value,_=reference_fluctuation_index({'Ix':np.zeros(1),'state':state})
    assert value==0.
