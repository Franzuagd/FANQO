"""Verify the explicit radial basis, operator similarity and LS independently."""
import numpy as np
import pytest
import sympy as sp

from fanqo import api
from fanqo.core import action_angle_ls as aa, nonlinear as nl
def from_action_angle(action, phi, beta, alpha):
    radius = np.sqrt(action)
    return np.sqrt(beta)*radius*np.cos(phi), -radius*(np.sin(phi)+alpha*np.cos(phi))/np.sqrt(beta)


@pytest.fixture(scope='module', params=[0,1])
def problem(request):
    v=sp.symbols('delta x y px py');f=sp.symbols('b1 b2 b3 b4 b5')
    _,x,y,px,py=v
    cs=np.array([2.,.4,(1+.4**2)/2,3.,-.3,(1+.3**2)/3])
    st=nl.initialize_nonlinear([cs],3,request.param,(px*px+py*py+x*x+y*y)/2,
        np.ones(5),v,f)
    st['C']=st['epsilon'].copy()
    aa.attach(st,dict(Jx_max=1.2,Jy_max=.7,delta_max=.2))
    return [cs],st


def test_basis_values_are_radial_fourier_functions(problem):
    _,st=problem;D=st['radial_transform'];box=st['action_box'];cs=st['linear_cs0']
    rng=np.random.default_rng(42);N=20
    jx=rng.uniform(.01,box['Jx_max'],N);jy=rng.uniform(.01,box['Jy_max'],N)
    phix=rng.uniform(0,2*np.pi,N);phiy=rng.uniform(0,2*np.pi,N)
    delta=rng.uniform(-box['delta_max'],box['delta_max'],N)
    x,px=from_action_angle(jx,phix,*cs[:2]);y,py=from_action_angle(jy,phiy,*cs[3:5])
    for column,label in enumerate(st['radial_labels']):
        d,i,j,n,kx,m,ky=label
        expected=(delta/box['delta_max'])**d*(jx/box['Jx_max'])**(i/2)*(jy/box['Jy_max'])**(j/2)
        expected*=np.cos(n*phix) if kx=='c' else np.sin(n*phix)
        expected*=np.cos(m*phiy) if ky=='c' else np.sin(m*phiy)
        actual=api._make_ix_callable(D[:,column],st)(delta,x,y,px,py)
        np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=2e-12)


def test_analytic_gram_matches_existing_independent_disk_moments(problem):
    _,st=problem;D=st['radial_transform']
    from fanqo.core.action_metric import action_gram
    physical, _ = action_gram(st, st['action_box'])
    np.testing.assert_allclose(D.T@physical@D,st['radial_gram'],rtol=1e-10,atol=1e-12)


def test_transport_is_exact_similarity_in_radial_basis(problem):
    _,st=problem;D=st['radial_transform'];rng=np.random.default_rng(5)
    T=rng.normal(size=D.shape);Tj=aa.transform_transport(T,st)
    np.testing.assert_allclose(D@Tj,T@D,rtol=1e-11,atol=1e-11)


def test_ls_matches_independent_sampled_radial_residual(problem):
    data,st=problem;cs=st['linear_cs0'];box=st['action_box'];D=st['radial_transform'];q=st['quad_size']
    g,w=np.polynomial.legendre.leggauss(4);phi=np.arange(8)*2*np.pi/8
    jx,jy,fx,fy,dd=np.meshgrid(box['Jx_max']*(g+1)/2,box['Jy_max']*(g+1)/2,
                              phi,phi,g*box['delta_max'],indexing='ij')
    x,px=from_action_angle(jx.ravel(),fx.ravel(),*cs[:2])
    y,py=from_action_angle(jy.ravel(),fy.ravel(),*cs[3:5])
    z=np.column_stack((dd.ravel(),x,y,px,py))
    powers=np.array(list(st['idx_to_vec'].values()))
    Phi=np.prod(z[:,None,:]**powers[None,:,:],axis=2)*st['C']
    weights=np.einsum('i,j,k,l,m->ijklm',w/2,w/2,np.ones(8)/8,np.ones(8)/8,w/2).ravel()
    rng=np.random.default_rng(2026);R=rng.normal(size=D.shape)
    seed,_=nl.quadratic_invariants(data,st)
    result,info=aa.solve(np.eye(len(D))+R,seed,st,1e-12)
    seedj=np.linalg.solve(D[:q,:q],seed)
    sampled=np.sqrt(weights)[:,None]*(Phi@R@D)
    h=np.linalg.lstsq(sampled[:,q:],-sampled[:,:q]@seedj,rcond=1e-12)[0]
    expected=D@np.r_[seedj,h]
    np.testing.assert_allclose(result,expected,rtol=1e-8,atol=1e-8)
    np.testing.assert_array_equal(result[:q],seed)
    assert info['method']=='action_angle_ls'
