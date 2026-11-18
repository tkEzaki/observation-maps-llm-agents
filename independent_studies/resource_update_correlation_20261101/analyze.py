"""Offline exploration, declared after calibration but before this kernel analysis."""
from pathlib import Path
import json,hashlib,datetime,argparse
import numpy as np
import pandas as pd
import engine as e
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];SOURCE=HERE.parent/'resource_collective_design_20261031'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())+'\n',encoding='utf-8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def prepare():
    assert not (HERE/'freeze.json').exists()
    plan=dict(created_at_utc=now(),api_calls=0,n=17,resources=3,simultaneous_update_sizes=[1,4,17],policies=['native','independent','shared'],baseline_policies=['least_load_random_ties','least_load_fixed_ties','uniform'],input_prompt='Unchanged; unaware of simultaneous decisions. All selected individuals observe the same pre-update state.',clock='Uniform subset without replacement each batch; total1020 individual updates=60 expected updates per agent. m=1:1020 batches,m=4:255,m=17:60.',primary_diagnostic='Mean post-batch normalized congestion over all1020 updates; cost=sum(n_j^2)/97. m is constant within each run, so equal batch weights equal update-count weights.',secondary_diagnostic='Post-batch cost after680 through1020 individual updates (last20 sweeps). Lag1/lag2 squared occupancy-fraction changes are descriptive and not a proof of a period2 cycle.',initial='Each agent initially chooses each resource with probability1/3 independently. Equal-count-state weights and all171 initial states are descriptive sensitivity.',estimators=['MLE conditional on valid replies','Dirichlet0.5 posterior mean conditional on valid replies'],missing='Original18 missing choices retained. Main estimates condition on parseability. For sensitivity assign all missing mass to A,B,C or own resource, then separately repeat symmetrization; these are scenarios, not worst-case bounds.',contrast='shared minus independent cost (positive means common frame is harmful). This has the same raw sign as positive shared minus independent phase order, but opposite utility. Opposite utility alone is not a new physical law.',engineering_gate='Suggest live validation only if a prespecified m has absolute full-window shared-minus-independent contrast above0.01 (1% of optimal cost), same sign for MLE/posterior mean/all4 missing scenarios, and conditional central95% draw interval entirely outside[-0.01,+0.01].',stop_rule='One comparison over m=1,4,17 and three policies; no prompt/temperature/extra-m search to rescue small or unstable effects.',review_corrections=['m17 is motivated by matching phase update timing but remains post-calibration exploration.','Phase primary predicted shared-independent order contrast is0.0231667; observed0.0314868.','Transport missingness cannot be proven independent of content from an outage alone; delayed completion never erases original missingness or restores original timing.','New initial states and contemporaneous controls preferred for a future persistence test.'])
    save(HERE/'protocol.json',plan)
    sources=[SOURCE/'analysis_real/counts.npy',SOURCE/'analysis_real/joint_counts.npy',SOURCE/'analysis_real/answers.npy',SOURCE/'analysis_real/results.json',SOURCE/'audit_real.json',SOURCE/'prompts.json',SOURCE/'freeze.json',SOURCE/'kernel.py',HERE.parent/'prospective_closure_validation_20261031/analysis_real/contrasts.csv']
    save(HERE/'freeze.json',dict(created_at_utc=now(),sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources},local={p:sha(HERE/p) for p in ['engine.py','analyze.py','protocol.json','verify_engine.py']}))

def check():
    f=read(HERE/'freeze.json')
    for p,h in f['sources'].items():assert sha(ROOT/p)==h,p
    for p,h in f['local'].items():assert sha(HERE/p)==h,p

def summaries(p,label,full=True):
    rows=[];contrasts=[];state_rows=[];out=HERE/'analysis';out.mkdir(exist_ok=True)
    for m in [1,4,17]:
        ts=e.kernels(p,m);vals={}
        for policy,T in ts.items():
            forecast=e.forecast(T,m,full_states=full);vals[policy]=forecast
            row=dict(estimator=label,m=m,policy=policy,mean_congestion=forecast['mean_congestion'],late_congestion=forecast['late_congestion'])
            if full:
                row.update(e.jump_diagnostics(T,m));row['mean_congestion_equal_counts']=float(forecast['by_state_mean'].mean());row['late_congestion_equal_counts']=float(forecast['by_state_late'].mean())
                np.save(out/f'T_{label}_m{m}_{policy}.npy',T);np.save(out/f'curve_{label}_m{m}_{policy}.npy',forecast['curves'])
                for i,n in enumerate(e.states(17)):state_rows.append(dict(estimator=label,m=m,policy=policy,A=int(n[0]),B=int(n[1]),C=int(n[2]),mean_congestion=float(forecast['by_state_mean'][i]),late_congestion=float(forecast['by_state_late'][i])))
            rows.append(row)
        for target,reference in [('shared','independent'),('native','independent'),('shared','native')]:
            contrasts.append(dict(estimator=label,m=m,contrast=f'{target}-{reference}',mean_difference=vals[target]['mean_congestion']-vals[reference]['mean_congestion'],late_difference=vals[target]['late_congestion']-vals[reference]['late_congestion']))
    return rows,contrasts,state_rows

def main():
    check();assert read(HERE/'engine_verification.json')['passed']
    counts=np.load(SOURCE/'analysis_real/counts.npy');joint=np.load(SOURCE/'analysis_real/joint_counts.npy');valid=counts.sum(1,keepdims=True)
    out=HERE/'analysis';out.mkdir(exist_ok=True);rows=[];cons=[];sr=[]
    tables={'MLE':counts/valid,'Dirichlet_half':(counts+.5)/(valid+1.5)}
    tables.update({name:e.baseline(name) for name in ['least_load_random_ties','least_load_fixed_ties','uniform']})
    for label,p in tables.items():
        a,b,c=summaries(p,label);rows.extend(a);cons.extend(b);sr.extend(c);print(f'Point tables complete: {label}',flush=True)
    for action in ['A','B','C','own']:
        table=counts.astype(float).copy()
        for i,(_,own) in enumerate(e.inputs(17)):table[i,own if action=='own' else 'ABC'.index(action)]+=joint[i,3]
        a,b,c=summaries(table/32,'missing_'+action,False);rows.extend(a);cons.extend(b)
    pd.DataFrame(rows).to_csv(out/'point_summaries.csv',index=False);pd.DataFrame(cons).to_csv(out/'point_contrasts.csv',index=False);pd.DataFrame(sr).to_csv(out/'by_initial_state.csv',index=False)
    rng=np.random.default_rng(202611019502);posterior=[]
    for draw in range(256):
        p=rng.gamma(counts+.5);p/=p.sum(1,keepdims=True)
        _,c,_=summaries(p,f'draw{draw}',False)
        for row in c:row['draw']=draw;posterior.append(row)
        if (draw+1)%32==0:pd.DataFrame(posterior).to_csv(out/'posterior_contrasts.csv',index=False);print(f'Posterior draws: {draw+1}/256',flush=True)
    df=pd.DataFrame(posterior);intervals=[];con=pd.DataFrame(cons);decisions=[]
    for (m,contrast),sub in df.groupby(['m','contrast']):
        r=dict(m=int(m),contrast=contrast,draws=256)
        for metric in ['mean_difference','late_difference']:
            low,med,high=np.quantile(sub[metric],[.025,.5,.975]);r.update({metric+'_low':float(low),metric+'_median':float(med),metric+'_high':float(high)})
        intervals.append(r)
        if contrast=='shared-independent':
            point=con[(con.m==m)&(con.contrast==contrast)&con.estimator.isin(['MLE','Dirichlet_half','missing_A','missing_B','missing_C','missing_own'])].mean_difference.to_numpy()
            sign_consistent=bool(np.all(point>0) or np.all(point<0));large=bool(np.all(abs(point)>.01));clear=r['mean_difference_low']>.01 or r['mean_difference_high']<-.01
            decisions.append(dict(m=int(m),same_sign_across_estimators_and_missing_scenarios=sign_consistent,all_point_effects_exceed_001=large,posterior95_outside_practical_null=bool(clear),numerical_gate_passed=bool(sign_consistent and large and clear),))
    pd.DataFrame(intervals).to_csv(out/'posterior_intervals.csv',index=False)
    check();save(out/'results.json',dict(created_at_utc=now(),api_calls=0,source_valid_choices=int(counts.sum()),source_missing_choices=int(joint[:,3].sum()),posterior_draws=256,exploratory=True,new_collective_validation=False,decisions=decisions,intervals=intervals))
    print(json.dumps(decisions,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args();prepare() if args.prepare else main()
