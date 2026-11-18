"""Post hoc four-parameter response mechanism; fitted only to the old cohort."""
from offline_closure import *

def probabilities(x,w):
    eta=w[0]+x[:,0]*w[1]+x[:,1]*w[2]
    return softmax(np.column_stack([-eta,np.full(len(x),w[3]),eta]),axis=1)
def fit_model(x,y):
    f=np.column_stack([np.ones(len(x)),x[:,:2]]);one=np.eye(3)[y];reg=1e-4
    def objective(w):
        eta=f@w[:3];z=np.column_stack([-eta,np.full(len(y),w[3]),eta]);p=softmax(z,axis=1);err=p-one
        return np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y])+reg/2*(w@w),np.r_[f.T@(err[:,2]-err[:,0])/len(y),err[:,1].mean()]+reg*w
    r=minimize(objective,np.zeros(4),jac=True,method='L-BFGS-B',options=dict(maxiter=1000,ftol=1e-11,gtol=1e-8));assert r.success
    return r.x
def main():
    out=LOCAL/'phase_lag';out.mkdir(exist_ok=True)
    save(out/'plan.json',dict(post_hoc=True,api_calls=0,reason='Feature ablation implicated the first cosine component; now test a four-parameter phenomenological response.',
        form='p(retard,stay,advance)=softmax(-eta,lambda,eta); eta=b0+bc*c1+bs*s1.',
        selection='No tuning on later outcomes. Same old eight-seed grouped validation; later four seeds are retrospective transfer tests.',
        
        script_sha256=digest(Path(__file__))))
    original=protected_snapshot();x,y,meta=dataset(OLD);xt,yt,mt=dataset(NEW);seeds=sorted(meta.seed.unique());cv=[]
    for fold in range(4):
        v=meta.seed.isin(seeds[2*fold:2*fold+2]).to_numpy();w=fit_model(x[~v],y[~v]);p=probabilities(x[v],w);sub=meta[v].reset_index(drop=True)
        for seed in sub.seed.unique():
            ix=(sub.seed==seed).to_numpy();cv.append(dict(seed=int(seed),nll=score(y[v][ix],p[ix])))
    w=fit_model(x,y);p=probabilities(xt,w);nll=mt.assign(loss=-np.log(np.maximum(p[np.arange(len(yt)),yt],1e-15))).groupby('seed').loss.mean().mean()
    # Algebraic identity and reflection decomposition on all original inputs.
    radius=np.hypot(x[:,0],x[:,1]);angle=np.arctan2(x[:,1],x[:,0]);alpha=np.arctan2(w[1],w[2]);amplitude=np.hypot(w[1],w[2])
    assert np.allclose(x[:,0]*w[1]+x[:,1]*w[2],amplitude*radius*np.sin(angle+alpha),atol=1e-12)
    save(out/'coefficients.json',dict(intercept=w[0],cosine=w[1],sine=w[2],stay_logit=w[3],amplitude=amplitude,phase_shift_rad=alpha,phase_shift_deg=float(np.degrees(alpha))))
    plan=read(NEW/'protocol.json');design=read(NEW/'design.json');cp=read(NEW/'runs/real/checkpoint.json');runs=plan['runs'];draws=32
    theta=np.repeat(np.array([plan['initial_phases'][str(r['physical_seed'])] for r in runs]),draws,axis=0);rng=np.random.default_rng(20261031711);late=[]
    for step in range(60):
        h=hist_many(theta);trs=[design['transformations'][f"{r['run_id']}__t{step:03d}"] for r in runs]
        mir=np.repeat(np.array([t['mirror'] for t in trs],bool),draws,axis=0);h[mir.ravel()]=h[mir.ravel(),::-1]
        p=probabilities(payload(h,REPS[0]),w);raw=(rng.random(len(h))[:,None]>p.cumsum(1)).sum(1)-1
        a=np.where(mir,-raw.reshape(-1,17),raw.reshape(-1,17));perm=np.repeat(np.array([t['permutation'] for t in trs]),draws,axis=0)
        theta+=np.array(plan['omega'])+.1*np.take_along_axis(a,perm,axis=1)
        if step>=40:late.append(abs(np.exp(1j*theta).mean(1)).reshape(len(runs),draws))
    val=np.mean(late,axis=0);rows=[]
    for i,r in enumerate(runs):
        th=np.array(cp['trajectories'][r['run_id']]['phases'])[41:61];obs=float(abs(np.exp(1j*th).mean(1)).mean())
        rows.append(dict(seed=r['physical_seed'],controller=r['controller'],prediction=float(val[i].mean()),observed=obs))
    df=pd.DataFrame(rows);df.to_csv(out/'closed_loop.csv',index=False);pred=df.pivot(index='seed',columns='controller',values='prediction');obs=df.pivot(index='seed',columns='controller',values='observed')
    effects=[]
    for a,b in [('shared_mirror','independent_mirror'),('independent_mirror','shuffled_mirror'),('independent_mirror','native')]:effects.append(dict(contrast=a+' - '+b,predicted=float((pred[a]-pred[b]).mean()),observed=float((obs[a]-obs[b]).mean())))
    result=dict(post_hoc=True,old_cv_nll=float(np.mean([d['nll'] for d in cv])),later_nll=float(nll),closed_loop_rmse=float(np.sqrt(np.mean((df.prediction-df.observed)**2))),effects=effects,
         )
    save(out/'results.json',result);assert original==protected_snapshot();print(json.dumps(result,indent=2))

if __name__=='__main__':main()
