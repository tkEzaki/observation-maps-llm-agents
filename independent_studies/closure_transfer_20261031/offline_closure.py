"""Retrospective grouped transfer audit. No API calls, no edits to earlier studies."""
from pathlib import Path
import sys,json,hashlib,time
sys.dont_write_bytecode=True
LOCAL=Path(__file__).resolve().parent;ROOT=LOCAL.parents[1]
sys.path.insert(0,str(ROOT/'independent_studies/collective_dynamics_20261004'))
from common import histogram_batch,payload,REPS,CENTERS,protected_snapshot,digest
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax,logsumexp
from scipy.spatial import cKDTree
OLD=ROOT/'independent_studies/reflection_confirmation_20261005'
NEW=ROOT/'independent_studies/mechanism_controls_20261031'
FIXED=ROOT/'independent_studies/same_state_validation_20261031'
OUT=LOCAL/'offline';OUT.mkdir(parents=True,exist_ok=True)
KINDS=['constant','linear6','quadratic6','knn64']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding='utf-8')
def hist_many(theta):
    q=np.asarray(theta);batch,n=q.shape
    delta=(q[:,None,:]-q[:,:,None]+np.pi)%(2*np.pi)-np.pi
    ids=np.floor(np.round((delta+np.pi)*24/(2*np.pi),12)).astype(int)%24
    row=np.broadcast_to(np.arange(batch*n).reshape(batch,n,1),ids.shape)
    mask=np.broadcast_to(~np.eye(n,dtype=bool),ids.shape)
    return np.bincount((24*row+ids)[mask],minlength=batch*n*24).reshape(batch*n,24)/(n-1)
def features(x,kind):
    parts=[np.ones((len(x),1)),x]
    if kind=='quadratic6':parts.append(np.column_stack([x[:,i]*x[:,j] for i in range(6) for j in range(i,6)]))
    return np.column_stack(parts)
class Model:
    def __init__(self,kind,x,y):
        self.kind=kind;self.base=(np.bincount(y,minlength=3)+.5)/(len(y)+1.5)
        if kind=='constant':return
        if kind=='knn64':self.x=x;self.y=y;self.tree=cKDTree(x);return
        f=features(x,kind);one=np.eye(3)[y];reg=1e-4
        def loss(w):
            w=w.reshape(f.shape[1],3);z=f@w;p=softmax(z,axis=1)
            return np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y])+reg/2*np.sum(w*w),((f.T@(p-one))/len(y)+reg*w).ravel()
        fit=minimize(loss,np.zeros(f.shape[1]*3),jac=True,method='L-BFGS-B',options=dict(maxiter=1000,ftol=1e-10,gtol=1e-7))
        assert fit.success,fit.message
        self.coef=fit.x.reshape(f.shape[1],3)
    def predict(self,x):
        if self.kind=='constant':return np.tile(self.base,(len(x),1))
        if self.kind=='knn64':
            _,ix=self.tree.query(x,k=64,workers=1);counts=np.eye(3)[self.y[ix]].sum(axis=1)
            return (counts+.5)/65.5
        return softmax(features(x,self.kind)@self.coef,axis=1)
    def export(self,p):
        if self.kind=='knn64':np.savez_compressed(p,x=self.x,y=self.y)
        else:np.savez_compressed(p,base=self.base,coef=getattr(self,'coef',np.empty((0,3))))
def dataset(folder):
    plan=read(folder/'protocol.json');cp=read(folder/'runs/real/checkpoint.json');xs=[];ys=[];meta=[]
    for run in plan['runs']:
        t=cp['trajectories'][run['run_id']]
        for step in range(60):
            h=histogram_batch(np.array(t['phases'][step]));mir=np.array(t['mirror_flags'][step],bool);h[mir]=h[mir,::-1]
            xs.append(payload(h,REPS[0]));ys.extend(np.array(t['raw_values'][step],int)+1)
            meta.extend([dict(seed=run['physical_seed'],controller=run['controller'],step=step,agent=i) for i in range(17)])
    return np.concatenate(xs),np.array(ys),pd.DataFrame(meta)
def score(y,p):return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean())
def main():
    original=protected_snapshot()
    sources=[OLD/'protocol.json',OLD/'runs/real/checkpoint.json',NEW/'protocol.json',NEW/'runs/real/checkpoint.json',NEW/'design.json',FIXED/'protocol.json',FIXED/'design.json',FIXED/'analysis_real/by_state.csv']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sources}
    plan=dict(retrospective=True,api_calls=0,training='Old eight physical seeds only; native and independently mirrored trajectories.',
        selection='Four folds, holding out consecutive seed pairs. Minimum equal-seed mean raw-action negative log likelihood. No selection on later cohorts.',
        candidates=KINDS,regularization=1e-4,knn_neighbors=64,knn_smoothing_per_class=.5,
        test='Later four-seed four-policy cohort, and fixed-state held-out responses; outcomes already known at study design, therefore not prospective confirmation.',
        rollout_draws=32,rollout_steps=60,late_steps='post-update41..60',
        inputs=hashes,script_sha256=digest(Path(__file__)))
    save(OUT/'analysis_plan.json',plan)
    rng=np.random.default_rng(20261031701);test=rng.uniform(-10,10,(11,17))
    assert np.array_equal(hist_many(test),np.concatenate([histogram_batch(q) for q in test]))
    x,y,meta=dataset(OLD);seeds=sorted(meta.seed.unique());cv=[]
    for k in KINDS:
        for fold in range(4):
            mask=meta.seed.isin(seeds[fold*2:fold*2+2]).to_numpy();m=Model(k,x[~mask],y[~mask]);p=m.predict(x[mask]);sub=meta[mask].reset_index(drop=True)
            for seed in sub.seed.unique():
                ix=(sub.seed==seed).to_numpy();cv.append(dict(model=k,fold=fold,seed=int(seed),nll=score(y[mask][ix],p[ix])))
        print('CV completed:',k,flush=True)
    cv=pd.DataFrame(cv);cv.to_csv(OUT/'old_seed_cv.csv',index=False)
    mean=cv.groupby('model').nll.mean();chosen=str(mean.idxmin());save(OUT/'selection.json',dict(selected=chosen,old_seed_cv_mean=mean.to_dict(),test_used_for_selection=False))
    models={k:Model(k,x,y) for k in KINDS}
    for k,m in models.items():m.export(OUT/(k+'.npz'))
    xt,yt,mt=dataset(NEW);scores=[];tree=cKDTree(x);distance=tree.query(xt,k=1)[0];dist=[]
    for (seed,controller),ix in mt.groupby(['seed','controller']).groups.items():
        ix=np.array(list(ix));dist.append(dict(seed=int(seed),controller=controller,median_nearest_training_distance=float(np.median(distance[ix])),p95_nearest_training_distance=float(np.quantile(distance[ix],.95))))
    for k,m in models.items():
        p=m.predict(xt)
        for (seed,controller),ix in mt.groupby(['seed','controller']).groups.items():
            ix=np.array(list(ix));scores.append(dict(model=k,seed=int(seed),controller=controller,nll=score(yt[ix],p[ix]),accuracy=float((p[ix].argmax(1)==yt[ix]).mean())))
    pd.DataFrame(scores).to_csv(OUT/'later_cohort_action_scores.csv',index=False);pd.DataFrame(dist).to_csv(OUT/'input_distance.csv',index=False)
    # At the eight fixed states, no repeated-response data enter model fitting.
    fp=read(FIXED/'protocol.json');fd=read(FIXED/'design.json');flags=np.zeros((32,8,2,17),bool)
    for t in fd['tasks']:
        if t['stage']=='validate':flags[t['round'],t['state'],t['arm'],t['agent']]=t['mirrored']
    phi=np.array([s['theta'] for s in fp['states']])+np.array(fp['omega']);z=np.exp(1j*phi)
    hx=np.concatenate([histogram_batch(np.array(s['theta'])) for s in fp['states']]);observed=pd.read_csv(FIXED/'analysis_real/by_state.csv').observed_gap.to_numpy();fixed=[]
    for k,m in models.items():
        c0=(m.predict(payload(hx,REPS[0]))@np.exp(.1j*np.arange(-1,2))).reshape(8,17)
        c1=(m.predict(payload(hx[:,::-1],REPS[0]))@np.exp(-.1j*np.arange(-1,2))).reshape(8,17)
        c=np.where(flags,c1[None,:,None,:],c0[None,:,None,:]);e=abs((z[None,:,None,:]*c).mean(-1))**2+(1-abs(c)**2).sum(-1)/17**2
        gap=(e[:,:,1]-e[:,:,0]).mean(0)
        for s in range(8):fixed.append(dict(model=k,state=s,prediction=float(gap[s]),observed=float(observed[s]),residual=float(observed[s]-gap[s])))
    pd.DataFrame(fixed).to_csv(OUT/'fixed_state_transfer.csv',index=False)
    # Closed-loop surrogate simulations under each original intervention schedule.
    np_=read(NEW/'protocol.json');nd=read(NEW/'design.json');cp=read(NEW/'runs/real/checkpoint.json');runs=np_['runs'];draws=32
    initial=np.repeat(np.array([np_['initial_phases'][str(r['physical_seed'])] for r in runs]),draws,axis=0)
    true=[]
    for r in runs:
        theta=np.array(cp['trajectories'][r['run_id']]['phases']);v=abs(np.exp(1j*theta[41:61]).mean(1)).mean();true.append(float(v))
    rollout=[];trajectory=[]
    for k,m in models.items():
        theta=initial.copy();rgen=np.random.default_rng(20261031711);late=[]
        for step in range(60):
            h=hist_many(theta);trs=[nd['transformations'][f"{r['run_id']}__t{step:03d}"] for r in runs]
            mir=np.repeat(np.array([t['mirror'] for t in trs],bool),draws,axis=0)
            h[mir.ravel()]=h[mir.ravel(),::-1]
            p=m.predict(payload(h,REPS[0]));raw=(rgen.random(len(h))[:,None]>p.cumsum(1)).sum(1)-1
            a=np.where(mir,-raw.reshape(-1,17),raw.reshape(-1,17))
            permutation=np.repeat(np.array([t['permutation'] for t in trs]),draws,axis=0)
            theta+=np.array(np_['omega'])+.1*np.take_along_axis(a,permutation,axis=1)
            coherence=abs(np.exp(1j*theta).mean(1)).reshape(len(runs),draws)
            for i,r in enumerate(runs):trajectory.append(dict(model=k,run_id=r['run_id'],step=step+1,mean=float(coherence[i].mean())))
            if step>=40:late.append(coherence)
        values=np.mean(late,axis=0)
        for i,r in enumerate(runs):
            rollout.append(dict(model=k,seed=r['physical_seed'],controller=r['controller'],prediction=float(values[i].mean()),p05=float(np.quantile(values[i],.05)),p95=float(np.quantile(values[i],.95)),observed=true[i]))
        print('Rollout completed:',k,flush=True)
    df=pd.DataFrame(rollout);df.to_csv(OUT/'closed_loop.csv',index=False);pd.DataFrame(trajectory).to_csv(OUT/'predicted_trajectories.csv',index=False)
    effects=[]
    for k in KINDS:
        pred=df[df.model==k].pivot(index='seed',columns='controller',values='prediction');obs=df[df.model==k].pivot(index='seed',columns='controller',values='observed')
        for a,b in [('shared_mirror','independent_mirror'),('independent_mirror','shuffled_mirror'),('independent_mirror','native')]:
            for seed in pred.index:effects.append(dict(model=k,seed=int(seed),contrast=a+' - '+b,predicted=float(pred.loc[seed,a]-pred.loc[seed,b]),observed=float(obs.loc[seed,a]-obs.loc[seed,b])))
    ef=pd.DataFrame(effects);ef.to_csv(OUT/'intervention_effects.csv',index=False)
    summary=[]
    for k in KINDS:
        f=pd.DataFrame(fixed).query('model == @k');b=df.query('model == @k');a=pd.DataFrame(scores).query('model == @k')
        summary.append(dict(model=k,selected_on_old_cv=k==chosen,old_cv_nll=float(mean[k]),later_nll=float(a.nll.mean()),fixed_state_gap_rmse=float(np.sqrt(np.mean(f.residual**2))),closed_loop_r1_rmse=float(np.sqrt(np.mean((b.prediction-b.observed)**2)))))
    save(OUT/'summary.json',dict(retrospective=True,api_calls=0,selected=chosen,models=summary,
        contrasts=ef.groupby(['model','contrast'])[['predicted','observed']].mean().reset_index().to_dict(orient='records')))
    assert protected_snapshot()==original
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    save(OUT/'verification.json',dict(source_files_unchanged=True,vectorized_encoder_matches=True,train_rows=len(y),later_rows=len(yt),closed_loop_simulations=len(KINDS)*len(runs)*draws))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
