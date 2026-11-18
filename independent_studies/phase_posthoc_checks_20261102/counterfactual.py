"""Post-hoc fixed-model simulations: coin sharing x categorical quantile sharing.
No API. New coupling of action random numbers is artificial, not identified by
single-agent probabilities. Raw-label and physical-action conventions are both
reported. Fresh independent coin histories distinguish policy from one schedule.
"""
from pathlib import Path
import sys,json,hashlib,itertools
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DESIGN=HERE.parent/'phase_persistence_design_20261101'
sys.path.insert(0,str(DESIGN))
from forecast import coefficients,hist_many,payload,REPS,features,probabilities,rollout
from scipy.special import softmax
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(HERE/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding='utf-8')
CONDITIONS=[('independent','independent','raw'),('shared','independent','raw'),('independent','shared','raw'),('shared','shared','raw'),('independent','shared','physical'),('shared','shared','physical')]

def draw_action(p,mir,u,convention):
    if convention=='physical':
        pp=np.where(mir[...,None],p[...,::-1],p)
        return (u[...,None]>pp.cumsum(-1)).sum(-1)-1
    raw=(u[...,None]>p.cumsum(-1)).sum(-1)-1
    return np.where(mir,-raw,raw)

def validate_sampler():
    # For fixed states, common U must preserve every individual marginal under
    # either convention. Integrate exactly over all CDF breakpoints, no MC.
    rng=np.random.default_rng(20261102012);p=rng.dirichlet(np.ones(3),size=5);mir=np.array([0,1,1,0,1],bool)
    target=np.where(mir[:,None],p[:,::-1],p);err=[]
    for convention in ['raw','physical']:
        pp=p if convention=='raw' else target
        edges=np.unique(np.r_[0,pp.cumsum(-1).ravel(),1]);counts=np.zeros_like(p)
        for lo,hi in zip(edges[:-1],edges[1:]):
            a=draw_action(p,mir,np.full(5,(lo+hi)/2),convention)
            counts[np.arange(5),a+1]+=hi-lo
        err.append(float(np.max(abs(counts-target))))
    assert max(err)<1e-12
    return err

def main():
    p=read(DESIGN/'protocol.json');d=read(DESIGN/'design.json');q,w=coefficients()
    for name,h in read(DESIGN/'prediction_freeze.json')['files'].items():assert sha(DESIGN/name)==h,name
    for name,h in read(DESIGN/'input_freeze.json')['sources'].items():assert sha(ROOT/name)==h,name
    inputs={str((DESIGN/n).relative_to(ROOT)):sha(DESIGN/n) for n in ['protocol.json','design.json','prediction_freeze.json','input_freeze.json','trajectories_quadratic6.npz','trajectories_four_parameter.npz']}
    save('COUNTERFACTUAL_PLAN.json',dict(post_hoc=True,api_calls=0,refitting=False,models=['quadratic6','four_parameter'],
        conditions=[dict(coin=c,action_uniform=u,convention=a) for c,u,a in CONDITIONS],
        steps=200,physical_seeds=p['physical_seeds'],draws_per_seed_condition=256,chunk_size=32,
        simulation_seed=20261102020,windows=p['windows'],
        fresh_reflection_schedule_per_draw=True,paired_common_random_numbers_between_conditions=True,
        convention='Raw: same U applied to retard/stay/advance before sign mapping. Physical: same U applied after probabilities map to negative/zero/positive physical increments.',
        
        stop='Fixed two models, six conditions, 256 draws; no outcome-selected expansion or new API experiments.',sources=inputs,script_sha256=sha(Path(__file__))))
    verification=dict(exact_marginal_max_errors=validate_sampler())
    # Verify the original sampler on the first frozen block before new conditions.
    initial_runs=[d['initial_phases'][str(r['physical_seed'])] for r in d['runs']]
    for kind in ['quadratic6','four_parameter']:
        reg=rollout(kind,initial_runs,np.array(d['frames'],bool)[:,:3],p['omega'],p['coupling'],64,p['simulation_seed'])
        expected=np.load(DESIGN/f'trajectories_{kind}.npz')['r1'][:4,:,:64]
        error=float(abs(reg-expected).max());assert error<1e-12;verification[kind+'_original_sampler_error']=error
    theta0=np.array([d['initial_phases'][str(seed)] for seed in p['physical_seeds']])
    for kind in ['quadratic6','four_parameter']:
        pieces=[]
        for chunk in range(8):
            n=32;theta=np.broadcast_to(theta0[None,:,None,:],(6,8,n,17)).copy()
            rng=np.random.default_rng(20261102020+chunk);out=[abs(np.exp(1j*theta).mean(-1))]
            for step in range(200):
                # Shared random variates across conditions improve contrast precision;
                # independent across seeds/draws/time. Coin and action streams differ.
                ci=rng.integers(0,2,(8,n,17)).astype(bool);cs=rng.integers(0,2,(8,n,1)).astype(bool)
                ui=rng.random((8,n,17));us=rng.random((8,n,1))
                mirrors=np.stack([ci if c=='independent' else np.broadcast_to(cs,(8,n,17)) for c,_,_ in CONDITIONS])
                h=hist_many(theta.reshape(-1,17));mask=mirrors.ravel();h[mask]=h[mask,::-1]
                x=payload(h,REPS[0]);prob=softmax(features(x,'quadratic6')@q,axis=1) if kind=='quadratic6' else probabilities(x,w)
                prob=prob.reshape(6,8,n,17,3)
                action=np.stack([draw_action(prob[j],mirrors[j],ui if u=='independent' else us,convention) for j,(_,u,convention) in enumerate(CONDITIONS)])
                theta+=np.array(p['omega'])+p['coupling']*action
                out.append(abs(np.exp(1j*theta).mean(-1)))
            pieces.append(np.array(out));print(f'{kind}: {chunk+1}/8 chunks complete',flush=True)
        np.savez_compressed(HERE/f'counterfactual_{kind}.npz',r1=np.concatenate(pieces,axis=3))
    summarize(p)
    assert all(sha(ROOT/n)==h for n,h in inputs.items())
    verification.update(passed=True,sources_unchanged=True,api_calls=0,virtual_trajectories=2*6*8*256)
    save('counterfactual_verification.json',verification)

def summarize(p):
    levels=[];contrasts=[];curves=[]
    comparisons=[('coin_shared_minus_independent__independent_actions',1,0),('coin_shared_minus_independent__shared_raw_U',3,2),
        ('coin_shared_minus_independent__shared_physical_U',5,4),('action_U_shared_minus_independent__independent_coins_raw',2,0),
        ('action_U_shared_minus_independent__shared_coins_raw',3,1),('action_U_shared_minus_independent__independent_coins_physical',4,0),
        ('action_U_shared_minus_independent__shared_coins_physical',5,1),('convention_physical_minus_raw__independent_coins',4,2),('convention_physical_minus_raw__shared_coins',5,3)]
    for kind in ['quadratic6','four_parameter']:
        y=np.load(HERE/f'counterfactual_{kind}.npz')['r1'];assert y.shape==(201,6,8,256)
        for c,(coin,u,convention) in enumerate(CONDITIONS):
            for t in range(201):curves.append(dict(model=kind,condition=c,coin=coin,action_uniform=u,convention=convention,step=t,mean=float(y[t,c].mean())))
        for window,(lo,hi) in p['windows'].items():
            m=y[lo:hi+1].mean(0)
            for c,(coin,u,convention) in enumerate(CONDITIONS):
                levels.append(dict(model=kind,window=window,condition=c,coin=coin,action_uniform=u,convention=convention,mean=float(m[c].mean()),
                    conditional_MC_se=float(np.sqrt(np.sum(m[c].var(1,ddof=1)/256))/8),per_seed=m[c].mean(1)))
            for label,a,b in comparisons:
                delta=m[a]-m[b];se=float(np.sqrt(np.sum(delta.var(1,ddof=1)/256))/8);mean=float(delta.mean())
                contrasts.append(dict(model=kind,window=window,contrast=label,mean=mean,conditional_MC_se=se,MC95=[mean-1.96*se,mean+1.96*se],
                    per_seed=delta.mean(1),negative_seeds=int((delta.mean(1)<0).sum())))
    save('counterfactual_levels.json',levels);save('counterfactual_contrasts.json',contrasts)
    pd.DataFrame(curves).to_csv(HERE/'counterfactual_curves.csv',index=False)
    print(json.dumps([r for r in contrasts if r['window']=='primary_61_200'],indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),flush=True)
if __name__=='__main__':main()
