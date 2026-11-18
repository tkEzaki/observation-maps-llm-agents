"""Freeze new-state closed-loop predictions; do NOT collect any new phase responses."""
from offline_closure import *
from phase_lag_response import probabilities

def main():
    out=LOCAL/'prospective_phase';out.mkdir(exist_ok=True)
    assert not (out/'freeze.json').exists(),'Retain existing prospective predictions.'
    seeds=list(range(20261031801,20261031809));policies=['native','independent_mirror','shared_mirror'];omega=np.linspace(-.05,.05,17);runs=[];initial={};transforms={}
    for seed in seeds:
        initial[str(seed)]=np.random.default_rng(seed).uniform(-np.pi,np.pi,17)
        for policy in policies:
            rid=f'Claude__{policy}__seed{seed}';runs.append(dict(run_id=rid,physical_seed=seed,controller=policy))
            rng=np.random.default_rng(seed+policies.index(policy)*10000)
            for step in range(60):
                mir=np.zeros(17,bool) if policy=='native' else rng.integers(0,2,17).astype(bool) if policy=='independent_mirror' else np.repeat(bool(rng.integers(0,2)),17)
                transforms[f'{rid}__t{step:03d}']=mir
    tasks=8*3*17*60
    protocol=dict(created_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        physical_seeds=seeds,initial_phases=initial,runs=runs,omega=omega,coupling=.1,steps=60,n_agents=17,planned_responses=tasks,
        provider='Anthropic',model='claude-haiku-4-5-20251001',temperature=.7,workers=8,
        primary_model='quadratic6 selected using only old eight-seed cross-validation',secondary_model='four-parameter first-moment response; fixed after retrospective ablation',
        primary_metric='RMSE of predicted versus observed late-window r1 over24 trajectories, post-update41..60.',
        engineering_targets='Primary-model RMSE<=0.05 and absolute prediction error<=0.05 for each of three mean paired contrasts; report all per-seed errors and bootstrap uncertainty without changing targets.',
        contrasts=['shared_mirror - independent_mirror','independent_mirror - native','shared_mirror - native'],
        
        )
    save(out/'protocol.json',protocol);save(out/'transformations.json',transforms)
    quad=np.load(LOCAL/'offline/quadratic6.npz')['coef'];coef=read(LOCAL/'phase_lag/coefficients.json');w=np.array([coef[k] for k in ['intercept','cosine','sine','stay_logit']])
    draws=64;rows=[]
    for kind in ['quadratic6','four_parameter']:
        theta=np.repeat(np.array([initial[str(r['physical_seed'])] for r in runs]),draws,axis=0);rng=np.random.default_rng(20261031899);late=[]
        for step in range(60):
            h=hist_many(theta);mir=np.repeat(np.array([transforms[f"{r['run_id']}__t{step:03d}"] for r in runs]),draws,axis=0);h[mir.ravel()]=h[mir.ravel(),::-1]
            x=payload(h,REPS[0]);p=softmax(features(x,'quadratic6')@quad,axis=1) if kind=='quadratic6' else probabilities(x,w)
            raw=(rng.random(len(h))[:,None]>p.cumsum(1)).sum(1)-1;a=np.where(mir,-raw.reshape(-1,17),raw.reshape(-1,17));theta+=omega+.1*a
            if step>=40:late.append(abs(np.exp(1j*theta).mean(1)).reshape(len(runs),draws))
        val=np.mean(late,axis=0)
        for i,r in enumerate(runs):rows.append(dict(model=kind,**r,prediction=float(val[i].mean()),conditional_p05=float(np.quantile(val[i],.05)),conditional_p95=float(np.quantile(val[i],.95))))
    df=pd.DataFrame(rows);df.to_csv(out/'predictions.csv',index=False);effects=[]
    for kind in ['quadratic6','four_parameter']:
        p=df[df.model==kind].pivot(index='physical_seed',columns='controller',values='prediction')
        for a,b in [('shared_mirror','independent_mirror'),('independent_mirror','native'),('shared_mirror','native')]:effects.append(dict(model=kind,contrast=a+' - '+b,mean_prediction=float((p[a]-p[b]).mean())))
    save(out/'predicted_effects.json',effects)
    files=[Path(__file__),out/'protocol.json',out/'transformations.json',out/'predictions.csv',out/'predicted_effects.json',LOCAL/'offline/quadratic6.npz',LOCAL/'phase_lag/coefficients.json',LOCAL/'offline_closure.py',LOCAL/'phase_lag_response.py']
    save(out/'freeze.json',dict(created_at_utc=protocol['created_at_utc'],phase_api_calls=0,files={p.relative_to(LOCAL).as_posix():digest(p) for p in files},virtual_trajectories=len(runs)*draws*2,
        ))
    print(json.dumps(dict(prepared_responses=tasks,effects=effects),indent=2))

if __name__=='__main__':main()
