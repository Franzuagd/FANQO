"""Independent quadrature, constrained LS and eigen-selection regressions."""
import numpy as np
import pytest
import sympy as sp

from fanqo.core import nonlinear as nl, action_a_box as action
from fanqo.core import optimization as opt


@pytest.fixture(scope='module')
def problem():
    v=sp.symbols('delta x y px py');f=sp.symbols('b1 b2 b3 b4 b5')
    _,x,y,px,py=v
    cs=np.array([2.,.4,(1+.4**2)/2,3.,-.3,(1+.3**2)/3])
    data=[cs]
    box={'Jx_max':1.2,'Jy_max':.7,'delta_max':.2}
    state=nl.initialize_nonlinear_for_method(data,3,1,(x*x+y*y+px*px+py*py)/2,
        np.ones(5),v,f,invariant_construction='action_a_box',method_options={'ACTION_BOX':box})
    return data,state,box


def _quadrature(state,box):
    # Gauss in action, uniform Fourier quadrature in angle, Gauss in delta.
    g,w=np.polynomial.legendre.leggauss(5)
    theta=np.arange(16)*2*np.pi/16
    def disk(beta,alpha,A):
        J=A*(g+1)/2
        X=(np.sqrt(J)[:,None]*np.cos(theta)).ravel()
        P=-(np.sqrt(J)[:,None]*np.sin(theta)).ravel()
        return np.sqrt(beta)*X,(P-alpha*X)/np.sqrt(beta),np.repeat(w/2/16,16)
    bx,ax,_,by,ay,_=state['linear_cs0']
    x,px,wx=disk(bx,ax,box['Jx_max']);y,py,wy=disk(by,ay,box['Jy_max'])
    gd,wd=np.polynomial.legendre.leggauss(2)
    i,j,k=np.meshgrid(np.arange(len(x)),np.arange(len(y)),np.arange(2),indexing='ij')
    i,j,k=i.ravel(),j.ravel(),k.ravel()
    z=np.column_stack((gd[k]*box['delta_max'],x[i],y[j],px[i],py[j]))
    weights=wx[i]*wy[j]*wd[k]/2
    powers=np.array(list(state['idx_to_vec'].values()))
    Phi=np.prod(z[:,None,:]**powers[None,:,:],axis=2)*state['C']
    return Phi,weights


def test_action_gram_matches_independent_quadrature(problem):
    _,state,box=problem
    Phi,w=_quadrature(state,box)
    measured=Phi.T @ (w[:,None]*Phi)
    np.testing.assert_allclose(state['G'],measured,rtol=2e-12,atol=2e-11)


def test_action_moments_and_fixed_seed(problem):
    data,state,box=problem
    sx,sy=nl.quadratic_invariants(data,state);q=len(sx);G=state['G'][:q,:q]
    assert np.isclose(sx@G@sx,box['Jx_max']**2/3)
    assert np.isclose(sx@G@sy,box['Jx_max']*box['Jy_max']/4)
    ix,details=action.solve(np.eye(len(state['C'])),sx,state)
    assert np.array_equal(ix[:q],sx)
    assert np.count_nonzero(ix[q:])==0 and details['residual_norm']==0


def test_action_ls_matches_sampled_residual_minimum(problem):
    data,state,box=problem
    sx,_=nl.quadratic_invariants(data,state);q=len(sx);N=len(state['C']);n=N-q
    rng=np.random.default_rng(42)
    R=np.zeros((N,N));R[q:,:q]=rng.normal(size=(n,q))
    R[q:,q:]=rng.normal(size=(n,n-4))@rng.normal(size=(n-4,n))
    ix,_=action.solve(np.eye(N)+R,sx,state,1e-12)
    Phi,w=_quadrature(state,box)
    weighted=np.sqrt(w)[:,None]*Phi
    A=weighted@R[:,q:];b=-weighted@R[:,:q]@sx
    expected=np.linalg.lstsq(A,b,rcond=1e-12)[0]
    np.testing.assert_allclose(ix[q:],expected,rtol=1e-8,atol=1e-8)
    assert np.array_equal(ix[:q],sx)


def test_action_public_dispatch_and_metric_refresh(problem,monkeypatch):
    data,state,box=problem
    state=dict(state);N=len(state['C']);q=state['quad_size'];T=np.eye(N)
    monkeypatch.setattr(opt,'nonlinear_transfer',lambda *args:(T,T[q:,q:],T[q:,:q]))
    result=opt.compute_requested_invariants({'state':state,'data':data},1e-12,
                                           compute_ix=True,compute_iy=False)
    assert 'Ix' in result and 'Iy' not in result
    before=state['G'].copy();cs=data[0].copy();cs[0]*=2;cs[2]/=2
    opt._refresh_nonlinear_normalization(state,[cs])
    assert not np.allclose(state['G'],before)
    assert state['action_box']==box


def test_eigen_uses_only_requested_seed_and_full_quadratic_distance(problem,monkeypatch):
    _,st,_=problem
    state=dict(st,C=np.ones(len(st['C'])))
    state['linear_cs0']=np.array([1.,0.,1.,99.,45.,999.])
    N=len(st['C']);index=st['vec_to_idx']
    horizontal=[index[(0,2,0,0,0)],index[(0,0,0,2,0)]]
    vertical=index[(0,0,2,0,0)]
    vectors=np.zeros((N,3));vectors[horizontal,:]=1
    vectors[vertical,:]=[.6,.05,.3]
    # The closest quadratic belongs to lambda=1 EXACTLY. The old positive
    # filter excluded it; this also verifies inclusion of negative deviations.
    monkeypatch.setattr(np.linalg,'eig',lambda _: (np.array([1e-12,0.,-1e-12]),vectors))
    ix,details=nl.eigen_invariant(np.eye(N),state)
    assert ix[vertical]==pytest.approx(.05)
    assert details['target_seed']=='Sx'
    state['linear_cs0'][3:]=[-100.,float('nan'),-4.]
    other,_=nl.eigen_invariant(np.eye(N),state)
    np.testing.assert_array_equal(ix,other)


def test_invalid_action_bounds_rejected(problem):
    _,state,box=problem
    with pytest.raises(ValueError,match='positive'):
        action.action_gram(state,dict(box,Jx_max=0.))
