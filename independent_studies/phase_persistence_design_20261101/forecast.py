"""Offline 200-step persistence forecasts using previously frozen response laws.
No fitting, API clients, credentials or external requests.
"""
from pathlib import Path
import sys,json,hashlib,datetime,argparse
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=HERE.parent/'closure_transfer_20261031';PREVIOUS=HERE.parent/'prospective_closure_validation_20261031'
sys.dont_write_bytecode=True;sys.path.insert(0,str(SOURCE))
from offline_closure import hist_many,features,payload,REPS,histogram_batch,protected_snapshot
from phase_lag_response import probabilities
from scipy.special import softmax

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()),encoding='utf-8')

def source_files():
    return [SOURCE/'offline/quadratic6.npz',SOURCE/'phase_lag/coefficients.json',SOURCE/'offline_closure.py',SOURCE/'phase_lag_response.py',SOURCE/'prospective_phase/protocol.json',SOURCE/'prospective_phase/transformations.json',SOURCE/'prospective_phase/predictions.csv',SOURCE/'prospective_phase/freeze.json',PREVIOUS/'analysis_real/results.json',PREVIOUS/'protocol.json',ROOT/'independent_studies/collective_dynamics_20261004/common.py']

def check():
    f=read(HERE/'input_freeze.json')
    for name,h in f['sources'].items():assert sha(ROOT/name)==h,name
    for name,h in f['local'].items():assert sha(HERE/name)==h,name

def prepare():
    assert not (HERE/'protocol.json').exists(),'Preserve frozen design'
    prior=read(SOURCE/'prospective_phase/freeze.json')
    for name,h in prior['files'].items():assert sha(SOURCE/name)==h,name
    old=read(PREVIOUS/'protocol.json');seeds=list(range(202611011501,202611011509));policies=['native','independent_mirror','shared_mirror']
    existing=[]
    for p in (ROOT/'independent_studies').rglob('protocol.json'):
        if p.parent==HERE:continue
        try:obj=read(p);existing.extend(obj.get('physical_seeds',[]))
        except (ValueError,TypeError):continue
    assert not set(seeds)&set(existing)
    runs=[];initial={};frames=[]
    for seed in seeds:
        initial[str(seed)]=np.random.default_rng(seed).uniform(-np.pi,np.pi,17)
        for policy in policies:
            runs.append(dict(run_id=f'Claude__{policy}__seed{seed}',physical_seed=seed,controller=policy))
            rng=np.random.default_rng(seed+policies.index(policy)*10000);seq=[]
            for step in range(200):seq.append(np.zeros(17,bool) if policy=='native' else rng.integers(0,2,17).astype(bool) if policy=='independent_mirror' else np.repeat(bool(rng.integers(0,2)),17))
            frames.append(seq)
    save(HERE/'design.json',dict(initial_phases=initial,runs=runs,frames=np.array(frames),generation='Eight consecutive seeds fixed before this rollout; no screening or rejection of seeds. Same uniform initial-phase rule and policy-dependent mirror RNG scheme as the earlier prospective study.'))
    save(HERE/'protocol.json',dict(created_at_utc=now(),steps=200,n_agents=17,physical_seeds=seeds,policies=policies,omega=old['omega'],coupling=old['coupling'],provider=old['provider'],model=old['model'],temperature=old['temperature'],primary_predictor='quadratic6',secondary_predictor='four_parameter',refitting=False,process_draws_per_run=256,chunk_size=64,simulation_seed=202611011599,
        windows={'early_41_60':[41,60],'primary_61_200':[61,200],'terminal_161_200':[161,200]},primary_effect='independent_mirror - native in mean post-update r1 over61..200',secondary_effect='shared_mirror - independent_mirror and shared_mirror - native; terminal-window effects and change from41..60',
        
        
        planned_responses_if_executed=81600,api_calls=0,stopping='No new seed selection, new models or extra rescue experiments based on outcomes. Long-time simulation does not establish stationarity.'))
    save(HERE/'input_freeze.json',dict(created_at_utc=now(),sources={p.relative_to(ROOT).as_posix():sha(p) for p in source_files()},local={name:sha(HERE/name) for name in ['protocol.json','design.json']},script_at_design=sha(Path(__file__)),api_calls=0))

def coefficients():
    q=np.load(SOURCE/'offline/quadratic6.npz')['coef'];v=read(SOURCE/'phase_lag/coefficients.json');w=np.array([v[k] for k in ['intercept','cosine','sine','stay_logit']]);return q,w

def rollout(kind,initial,frames,omega,coupling,draws,seed):
    q,w=coefficients();nrun=len(initial);theta=np.repeat(np.array(initial),draws,axis=0);rng=np.random.default_rng(seed)
    out=[abs(np.exp(1j*theta).mean(1)).reshape(nrun,draws)]
    for step in range(frames.shape[1]):
        h=hist_many(theta);mir=np.repeat(frames[:,step],draws,axis=0);h[mir.ravel()]=h[mir.ravel(),::-1]
        x=payload(h,REPS[0]);p=softmax(features(x,'quadratic6')@q,axis=1) if kind=='quadratic6' else probabilities(x,w)
        assert np.isfinite(p).all() and np.max(abs(p.sum(1)-1))<1e-12
        raw=(rng.random(len(h))[:,None]>p.cumsum(1)).sum(1)-1
        a=np.where(mir,-raw.reshape(-1,17),raw.reshape(-1,17));theta+=np.array(omega)+coupling*a
        out.append(abs(np.exp(1j*theta).mean(1)).reshape(nrun,draws))
    out=np.array(out);assert out.min()>=0 and out.max()<=1+1e-12;return out

def verify_old():
    check();p=read(SOURCE/'prospective_phase/protocol.json');tr=read(SOURCE/'prospective_phase/transformations.json');initial=[p['initial_phases'][str(r['physical_seed'])] for r in p['runs']];frames=np.array([[tr[f"{r['run_id']}__t{step:03d}"] for step in range(60)] for r in p['runs']])
    expected=pd.read_csv(SOURCE/'prospective_phase/predictions.csv');errors={}
    for kind in ['quadratic6','four_parameter']:
        y=rollout(kind,initial,frames,p['omega'],p['coupling'],64,20261031899)
        target=expected[expected.model==kind].set_index('run_id').loc[[r['run_id'] for r in p['runs']],'prediction'].to_numpy()
        errors[kind]=float(np.max(abs(y[41:61].mean((0,2))-target)));assert errors[kind]<1e-12
    samples=np.random.default_rng(202611011598).uniform(-20,20,(12,17));assert np.array_equal(hist_many(samples),np.concatenate([histogram_batch(v) for v in samples]))
    save(HERE/'verification_old.json',dict(passed=True,api_calls=0,source_forecasts_reproduced_max_error=errors,vectorized_encoder_matches_scalar=True,))
    print('Existing forecast regression:',errors,flush=True)

def predict():
    check();assert read(HERE/'verification_old.json')['passed'];p=read(HERE/'protocol.json');d=read(HERE/'design.json');initial=[d['initial_phases'][str(r['physical_seed'])] for r in d['runs']];frames=np.array(d['frames'],bool)
    for kind in ['quadratic6','four_parameter']:
        if (HERE/f'trajectories_{kind}.npz').exists():continue
        pieces=[]
        for chunk in range(p['process_draws_per_run']//p['chunk_size']):
            y=rollout(kind,initial,frames,p['omega'],p['coupling'],p['chunk_size'],p['simulation_seed']+chunk)
            pieces.append(y);print(f'{kind}: {chunk+1}/4 chunks;200 steps',flush=True)
        y=np.concatenate(pieces,axis=2);np.savez_compressed(HERE/f'trajectories_{kind}.npz',r1=y)
    summarize()
    check()

def summarize():
    check();p=read(HERE/'protocol.json');d=read(HERE/'design.json');rows=[];effects=[];changes=[];curves=[]
    pairs=[('independent_mirror','native'),('shared_mirror','independent_mirror'),('shared_mirror','native')];policies=p['policies'];rng=np.random.default_rng(202611011600)
    seedix=rng.integers(0,8,size=(10000,8));processix=rng.integers(0,p['process_draws_per_run'],size=10000)
    for kind in ['quadratic6','four_parameter']:
        y=np.load(HERE/f'trajectories_{kind}.npz')['r1'];win={}
        for step in range(201):
            for i,r in enumerate(d['runs']):curves.append(dict(model=kind,**r,step=step,r1_mean=float(y[step,i].mean()),process_p025=float(np.quantile(y[step,i],.025)),process_p975=float(np.quantile(y[step,i],.975))))
        for name,(lo,hi) in p['windows'].items():
            val=y[lo:hi+1].mean(0);win[name]=val.reshape(8,3,-1)
            for i,r in enumerate(d['runs']):rows.append(dict(model=kind,window=name,**r,prediction=float(val[i].mean()),mean_mc_se=float(val[i].std(ddof=1)/np.sqrt(val.shape[1])),conditional_process_p025=float(np.quantile(val[i],.025)),conditional_process_p975=float(np.quantile(val[i],.975))))
            for aa,bb in pairs:
                diff=win[name][:,policies.index(aa)]-win[name][:,policies.index(bb)];cohort=diff.mean(0);mixed=diff[seedix,processix[:,None]].mean(1)
                effects.append(dict(model=kind,window=name,contrast=aa+' - '+bb,prediction=float(cohort.mean()),mean_mc_se=float(cohort.std(ddof=1)/np.sqrt(len(cohort))),fixed_eight_process95=np.quantile(cohort,[.025,.975]),resampled_eight_plus_process95=np.quantile(mixed,[.025,.975]),positive_seed_mean_count=int((diff.mean(1)>0).sum()),per_seed_predictions=diff.mean(1),conditional_fraction_cohort_positive=float((cohort>0).mean())))
        for aa,bb in pairs:
            early=win['early_41_60'][:,policies.index(aa)]-win['early_41_60'][:,policies.index(bb)]
            late=win['terminal_161_200'][:,policies.index(aa)]-win['terminal_161_200'][:,policies.index(bb)]
            difference=late-early;changes.append(dict(model=kind,contrast=aa+' - '+bb,terminal_minus_early_effect=float(difference.mean()),fixed_eight_process95=np.quantile(difference.mean(0),[.025,.975])))
    pd.DataFrame(rows).to_csv(HERE/'predictions.csv',index=False);pd.DataFrame(curves).to_csv(HERE/'predicted_trajectories.csv',index=False)
    save(HERE/'effects.json',effects);save(HERE/'effect_changes.json',changes)
    primary=[e for e in effects if e['model']=='quadratic6' and e['window']=='primary_61_200'];save(HERE/'summary.json',dict(created_at_utc=now(),api_calls=0,virtual_trajectories=2*24*256,steps=200,primary_window_effects=primary))
    files=['protocol.json','design.json','input_freeze.json','predictions.csv','predicted_trajectories.csv','effects.json','effect_changes.json','summary.json','trajectories_quadratic6.npz','trajectories_four_parameter.npz','verification_old.json','forecast.py']
    save(HERE/'prediction_freeze.json',dict(created_at_utc=now(),api_calls_before_freeze=0,files={name:sha(HERE/name) for name in files},not_a_public_preregistration=True))
    print(json.dumps(read(HERE/'summary.json'),indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','verify_old','predict']);args=a.parse_args();globals()[args.stage]()
