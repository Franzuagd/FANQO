"""Held-out physical tracking experiment. Does not alter production constructors.
Run from checkout: OPENBLAS_NUM_THREADS=1 python research/track_fit.py --order 8
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import at
from fanqo import comparison as cmp
from fanqo.api import _at_element
from fanqo.core import nonlinear as nl,linear as lin,structured_invariants as si

def features(coords,state):
    z=coords[[4,0,2,1,3]].reshape(5,-1).T
    powers=np.asarray(list(state['idx_to_vec'].values()))
    P=np.ones((len(z),len(powers)))
    for k in range(5):
        table=z[:,k,None]**np.arange(int(powers[:,k].max())+1)[None,:]
        P*=table[:,powers[:,k]]
    return P*state['C']

def tracks(ring,cs,n,turns,seed,extent,deltas):
    rng=np.random.default_rng(seed); out=[]; labels=[]; starts=[]
    bx,ax,gx,by,ay,gy=cs
    for delta in deltas:
        orbit,_=at.find_orbit4(ring,dp=delta)
        a=rng.uniform(.15,extent,size=(2,n))*1e-3
        phi=rng.uniform(0,2*np.pi,size=(2,n))
        z=np.repeat(orbit[:,None],n,axis=1)
        z[0]+=a[0]*np.cos(phi[0]);z[1]+=-a[0]*(np.sin(phi[0])+ax*np.cos(phi[0]))/bx
        z[2]+=a[1]*np.cos(phi[1]);z[3]+=-a[1]*(np.sin(phi[1])+ay*np.cos(phi[1]))/by
        z[4]=delta
        t=np.asarray(at.lattice_pass(ring,np.asfortranarray(z.copy()),turns))[:,:,0,:]
        full=np.concatenate([z[:,:,None],t],axis=2)
        alive=np.isfinite(full).all(axis=(0,2))
        out.append(full[:,alive]);labels.extend([delta]*int(alive.sum()))
        starts.append({'delta':delta,'launched':n,'survived':int(alive.sum())})
    return np.concatenate(out,axis=1),np.array(labels),starts

def matrices(coords,state,sx):
    # Long-lag residuals at logarithmic and evenly spaced horizons.
    ids=np.unique(np.r_[0,1,2,4,8,16,np.linspace(1,coords.shape[2]-1,17,dtype=int)])
    ids=ids[ids<coords.shape[2]]
    P=features(coords[:,:,ids],state).reshape(coords.shape[1],len(ids),-1)
    scale=np.maximum(P[:,0]@sx,1e-14)
    return ((P[:,1:]-P[:,:1])/scale[:,None,None]).reshape(-1,P.shape[2]),P[:,0]/scale[:,None]

def score(c,coords,state,sx,eigen,labels):
    P=features(coords,state).reshape(coords.shape[1],coords.shape[2],-1)
    if coords.shape[1]==0: raise ValueError('No survivors in scoring split')
    scale=np.maximum(P[:,0]@sx,1e-14)
    v=P@c;e=P@eigen
    err=np.max(np.abs(v-v[:,:1]),axis=1)/scale
    ref=np.max(np.abs(e-e[:,:1]),axis=1)/scale
    distortion=np.sqrt(np.mean(((v[:,0]-e[:,0])/scale)**2))
    legacy=np.max(np.abs(v[:,1:]-v[:,:1])/np.arange(1,v.shape[1]),axis=1)/np.maximum(abs(v[:,0]),1e-12*max(abs(v[:,0])))
    result={'median':float(np.median(err)),'p90':float(np.quantile(err,.9)),
            'worst':float(max(err)),'win_fraction':float(np.mean(err<ref)),
            'initial_value_distortion':float(distortion),'legacy_median':float(np.median(legacy)),
            'by_delta':{str(d):{'median':float(np.median(err[labels==d])),
                               'p90':float(np.quantile(err[labels==d],.9))} for d in np.unique(labels)}}
    return result,err

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--order',type=int,default=4)
    ap.add_argument('--legacy-thin',action='store_true',help='Reproduce the old I+ML octupole map for audit only')
    ap.add_argument('--deltas',type=float,nargs='+',default=[0.,-.01,.01])
    ap.add_argument('--config');ap.add_argument('--output',default='research_results')
    ap.add_argument('--extent',type=float,default=4.);ap.add_argument('--seed',type=int,default=1701)
    ap.add_argument('--train',type=int,default=48);ap.add_argument('--test',type=int,default=96)
    ap.add_argument('--turns',type=int,default=256);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.legacy_thin:
        original_element=nl.element_transfer
        def legacy_element(elem,state,*a,**kw):
            if lin.magnet_field(elem,'TYPE')=='multipole' and lin.magnet_field(elem,'LENGTH')==0:
                t=np.eye(len(state['C']))+state['transport_sign']*lin.magnet_field(elem,'O')*state['M_octupole_unit']
                q=state['quad_size'];return t,t[:q,:q],t[q:,q:],t[:q,q:],t[q:,:q]
            return original_element(elem,state,*a,**kw)
        nl.element_transfer=legacy_element
    cfg,lc=cmp._configuration(args.config)
    magnets,lattice,data,correction,parameters=cmp._linear(cfg,lc)
    kw=dict(m=args.order,d=1,hamiltonian=cfg.HAMILTONIAN,a_box=cfg.A_BOX,
            variables=cfg.VARIABLES,field_symbols=cfg.FIELD_SYMBOLS,n=2)
    base=nl.initialize_nonlinear_for_method(data,**kw,invariant_construction='graded_coupled')
    es=nl.initialize_nonlinear_for_method(data,**kw,invariant_construction='eigen')
    t0=time.perf_counter();T,_,_=nl.nonlinear_transfer(lattice,base,tol=cfg.LEAST_SQUARES_TOL)
    Te,_,_=nl.nonlinear_transfer(lattice,es,tol=cfg.LEAST_SQUARES_TOL)
    map_seconds=time.perf_counter()-t0
    t0=time.perf_counter();eigen,ed=nl.eigen_invariant(Te,es)
    eigen=eigen*es['C']/base['C'];eig_seconds=time.perf_counter()-t0
    sx,sy=si.seeds(data,base);q=base['quad_size']
    byname={lin.magnet_field(e,'NAME'):e for e in magnets};cell=[byname[n] for n in lc.CELL_NAMES]
    bend=sum(float(lin.magnet_field(e,'ANGLE')) for e in cell);ncells=int(round(360/abs(bend)))
    if not np.isclose(ncells*abs(bend),360):raise ValueError('Noninteger ring')
    ring=at.Lattice([_at_element(e,at,cfg.FMA_NUM_INT_STEPS) for e in cell*ncells],energy=parameters[lc.ENERGY_PARAMETER]*1e9)
    cs=lin.linear_data(data,'CS0')
    print('Tracking training and validation',flush=True)
    t0=time.perf_counter()
    train,lt,st=tracks(ring,cs,args.train,64,args.seed,args.extent,args.deltas)
    valid,lv,sv=tracks(ring,cs,args.train,128,args.seed+1,args.extent,args.deltas)
    track_seconds=time.perf_counter()-t0
    fitting_start=time.perf_counter()
    D,V=matrices(train,base,sx)
    # Preserve ALL eigen quadratic, constant, linear and pure-delta coefficients.
    free=np.array([i for i,a in base['idx_to_vec'].items() if i>=q and sum(a[1:])>0])
    U=D[:,free]; b=-(D@eigen)
    scaling=np.maximum(np.sqrt(np.mean(V[:,free]**2,axis=0)),1e-10)
    A=U/scaling
    # Ridge is dimensionless after empirical initial-value column scaling.
    H=A.T@A/len(A);g=A.T@b/len(A)
    candidates={'eigen':eigen};timings={'eigen':eig_seconds};selection=[]
    for lam in (1e-8,1e-6,1e-4,.01,1.,100.):
        t0=time.perf_counter();u=cho_solve(cho_factor(H+lam*np.eye(len(free))),g)
        c=eigen.copy();c[free]+=u/scaling
        sc,_=score(c,valid,base,sx,eigen,lv)
        blend=min(1.,.249/max(sc['initial_value_distortion'],1e-30))
        c=eigen+blend*(c-eigen)
        name=f'track_ridge_{lam:g}';candidates[name]=c;timings[name]=time.perf_counter()-t0
        sc,_=score(c,valid,base,sx,eigen,lv)
        sc['blend']=blend
        selection.append({'method':name,'lambda':lam,**sc})
    fitting_seconds=time.perf_counter()-fitting_start
    se,_=score(eigen,valid,base,sx,eigen,lv)
    eligible=[r for r in selection if r['initial_value_distortion']<=.25]
    winner=min([{'method':'eigen',**se}]+eligible,key=lambda r:r['p90'])['method']
    # Identical-seed LS isolates the constraint mismatch explanation.
    t0=time.perf_counter()
    R=T-np.eye(len(T));W=np.linalg.cholesky(base['G']).T
    candidates['LS_eigen_seed']=np.r_[eigen[:q],np.linalg.lstsq(W@R[:,q:],-W@R[:,:q]@eigen[:q],rcond=cfg.LEAST_SQUARES_TOL)[0]]
    timings['LS_eigen_seed']=time.perf_counter()-t0
    candidates['LS_CS_seed'],_=si.least_squares_seed(T[q:,q:],T[q:,:q],sx[:q],base,cfg.LEAST_SQUARES_TOL)
    for method in si.PAIR_METHODS:
        t0=time.perf_counter();state=dict(base,invariant_construction=method,method_options={'COUPLED_FIX_QUADRATIC':method=='a_box_coupled_fixed'})
        candidates[method],_,construction_details=si.construct_pair(T,data,state,cfg.LEAST_SQUARES_TOL)
        print('Constructed',method, 'success',construction_details.get('solver_success'),flush=True)
        timings[method]=time.perf_counter()-t0
    print('Frozen selection:',winner,'; tracking untouched test',flush=True)
    (out/'selection.json').write_text(json.dumps({'winner':winner,'validation':selection},indent=2))
    test,labels,ss=tracks(ring,cs,args.test,args.turns,args.seed+2,args.extent,args.deltas)
    # Test is evaluated only after hyperparameters and coefficients are frozen.
    report={'settings':vars(args),'thin_map':'legacy_I_plus_ML' if args.legacy_thin else 'finite_exponential','parameters':parameters,'map_seconds':map_seconds,
            'fitting_and_validation_seconds':fitting_seconds,'training_validation_tracking_seconds':track_seconds,'training':st,'validation':sv,'test':ss,
            'selected':winner,'validation_candidates':selection,'test_results':{},'timings':timings}
    errors={}
    for name in ['eigen',winner,'LS_eigen_seed','LS_CS_seed',*si.PAIR_METHODS]:
        if name in errors:continue
        sc,err=score(candidates[name],test,base,sx,eigen,labels);errors[name]=err
        R=T@candidates[name]-candidates[name]
        sc['homological_norm']=si._norm(R,base['G'])
        report['test_results'][name]=sc
        print(name,json.dumps(sc),flush=True)
    rng=np.random.default_rng(909);ratio=errors[winner]/np.maximum(errors['eigen'],1e-30)
    boot=np.median(ratio[rng.integers(0,len(ratio),(2000,len(ratio)))] ,axis=1)
    report['selected_paired_ratio']={'median':float(np.median(ratio)),'bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'coefficients.npz',**candidates,C=base['C'],powers=np.array(list(base['idx_to_vec'].values())))
    np.savez_compressed(out/'test_errors.npz',**errors,delta=labels)
    np.savez_compressed(out/'trajectories.npz',train=train,validation=valid,test=test)
    print('Saved',out,flush=True)
if __name__=='__main__':main()
