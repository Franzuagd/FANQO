"""Algebraic guarantees and public dispatch of the experimental constructors."""
import copy
import numpy as np
import pytest
import sympy as sp
from scipy.linalg import expm
from fanqo.core import nonlinear as nl, structured_invariants as si
from fanqo.core import optimization as opt


@pytest.fixture(scope='module')
def problem():
    variables=sp.symbols('delta x y px py')
    delta,x,y,px,py=variables
    fields=sp.symbols('b1 b2 b3 b4 b5')
    H=sp.Rational(31,200)*(x*x+px*px)+sp.Rational(47,200)*(y*y+py*py)
    data=[np.array([1.,0.,1.,1.,0.,1.])]
    state=nl.initialize_nonlinear_for_method(data,4,1,H,np.ones(5),variables,fields,
        invariant_construction='canonical_graded',method_options={'STRUCTURED_RIDGE':0.})
    sx,sy=si.seeds(data,state)
    chi=np.zeros(len(sx))
    for a,c in [((0,2,1,0,0),.025),((1,1,1,0,0),.01)]:
        i=state['vec_to_idx'][a];chi[i]=c/state['C'][i]
    C=expm(si.bracket_matrix(chi,state,fixed_left=False))
    M=nl.assemble_M(np.asarray(state['H_vec_func'](0,0,0,0,0),float).ravel(),state['M_basis'])
    T=C @ expm(-M) @ np.linalg.inv(C)
    return data,state,sx,sy,chi,T


def test_shared_lie_jet_matches_matrix_exponential_and_commutes(problem):
    _,state,sx,sy,chi,_=problem
    C=expm(si.bracket_matrix(chi,state,fixed_left=False))
    ix,iy=si.lie_transform(sx,chi,state),si.lie_transform(sy,chi,state)
    assert np.allclose(ix,C @ sx,atol=1e-13)
    assert np.allclose(iy,C @ sy,atol=1e-13)
    assert np.max(np.abs(si.bracket(ix,iy,state)))<1e-12
    assert np.array_equal(ix[:15],sx[:15])


def test_unsafe_dispersion_generator_rejected(problem):
    _,state,sx,_,_,_=problem
    chi=np.zeros(len(sx));chi[state['vec_to_idx'][(1,1,0,0,0)]]=1.
    with pytest.raises(ValueError,match='transverse order'):
        si.lie_transform(sx,chi,state)


@pytest.mark.parametrize('method',['graded_coupled','canonical_graded'])
def test_integrable_conjugate_rotation(problem,method):
    data,state,sx,sy,_,T=problem
    state=dict(state,invariant_construction=method)
    ix,iy,details=si.construct_pair(T,data,state,1e-12)
    assert np.array_equal(ix[:15],sx[:15])
    assert np.array_equal(iy[:15],sy[:15])
    assert details['relative_homological_x']<1e-8
    assert details['relative_homological_y']<1e-8
    assert details['relative_poisson_bracket']<1e-8
    assert ix[state['vec_to_idx'][(1,0,0,0,0)]]==0.


def test_graded_block_bracket_linearization_with_delta_linear_terms(problem):
    _,state,sx,sy,_,_=problem
    ix,iy=sx.copy(),sy.copy()
    ix[state['vec_to_idx'][(1,1,0,0,0)]]=.02
    iy[state['vec_to_idx'][(1,0,0,0,1)]]=-.03
    for grade,rows in si.grade_groups(state):
        if grade[1]==0 or (grade[0]==0 and grade[1]<=2):continue
        u=np.zeros(len(ix));v=u.copy()
        u[rows]=np.linspace(-.01,.01,len(rows));v[rows]=np.linspace(.02,-.02,len(rows))
        exact=si.bracket(ix+u,iy+v,state)[rows]
        linear=(si.bracket(ix,iy,state)+si.bracket(u,iy,state)+si.bracket(ix,v,state))[rows]
        assert np.allclose(exact,linear,atol=1e-13)


@pytest.mark.parametrize('method',si.PAIR_METHODS)
def test_objective_dispatch_computes_pair_once(problem,monkeypatch,method):
    data,state,sx,sy,_,T=problem
    state=dict(state,invariant_construction=method)
    context={'state':state,'data':data}
    calls=[]
    monkeypatch.setattr(opt,'nonlinear_transfer',lambda c,t:(T,T[15:,15:],T[15:,:15]))
    def construct(*args):
        calls.append(1);return sx.copy(),sy.copy(),{'method':method}
    monkeypatch.setattr(si,'construct_pair',construct)
    result=opt.compute_requested_invariants(context,1e-12,compute_ix=True,compute_iy=False)
    assert len(calls)==1 and 'Ix' in result and 'Iy' not in result
    assert result['Ix_construction_details']['method']==method


def test_bracket_tensor_scale_matches_symbolic_polynomials(problem):
    _,state,sx,sy,_,_=problem
    a=np.zeros(len(sx));b=a.copy()
    a[state['vec_to_idx'][(0,2,1,0,0)]]=.13
    b[state['vec_to_idx'][(0,0,0,2,0)]]=-.27
    bracket=si.bracket(a,b,state)
    physical=nl.vector_to_poly(bracket*state['C'],state['monomial_basis'])
    f=nl.vector_to_poly(a*state['C'],state['monomial_basis'])
    g=nl.vector_to_poly(b*state['C'],state['monomial_basis'])
    _,x,y,px,py=state['variables']
    expected=sp.diff(f,x)*sp.diff(g,px)-sp.diff(f,px)*sp.diff(g,x)+sp.diff(f,y)*sp.diff(g,py)-sp.diff(f,py)*sp.diff(g,y)
    assert all(abs(float(c))<1e-11 for c in sp.Poly(sp.expand(physical-expected),x,y,px,py).coeffs())

@pytest.mark.parametrize('method',['graded_coupled','canonical_graded'])
def test_nontriangular_full_residual_path(problem, method):
    data,state,sx,sy,_,T=problem
    state=dict(state,invariant_construction=method,
               method_options={'STRUCTURED_RIDGE':1e-8,'STRUCTURED_MAX_NFEV':4})
    T=T.copy()
    T[state['vec_to_idx'][(1,1,0,0,0)],state['vec_to_idx'][(0,3,0,0,0)]] += .02
    ix,iy,details=si.construct_pair(T,data,state)
    assert details['fitting_strategy']=='full_residual_nontriangular'
    assert np.isfinite(ix).all() and np.isfinite(iy).all()
    assert np.array_equal(ix[:15],sx[:15])
    if method=='canonical_graded':
        assert details['relative_poisson_bracket']<1e-10
