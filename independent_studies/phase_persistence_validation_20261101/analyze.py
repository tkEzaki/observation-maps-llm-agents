"""Prespecified finite-horizon effects and frozen-prediction scores, no fitting."""
from study import *
from math import comb

def analyze(mode):
    check();v=read(HERE/f'verification_{mode}.json');assert v['passed']
    folder=HERE/'runs'/mode;assert digest(folder/'attempts.jsonl')==v['journal_sha256'];assert digest(folder/'checkpoint.json')==v['checkpoint_sha256']
    p=read(HERE/'protocol.json');cp=read(folder/'checkpoint.json');pred=pd.read_csv(PRED/'predictions.csv');out=HERE/f'analysis_{mode}';out.mkdir(exist_ok=True)
    rows=[];curves=[]
    for r in p['runs']:
        theta=np.array(cp['trajectories'][r['run_id']]['phases']);y=abs(np.exp(1j*theta).mean(1))
        curves.extend(dict(**r,step=i,r1=float(x)) for i,x in enumerate(y))
        for name,(lo,hi) in p['windows'].items():rows.append(dict(**r,window=name,observed=float(y[lo:hi+1].mean())))
    obs=pd.DataFrame(rows);obs.to_csv(out/'observed_by_run.csv',index=False);pd.DataFrame(curves).to_csv(out/'observed_trajectories.csv',index=False)
    merged=pred.merge(obs,on=['run_id','physical_seed','controller','window'],validate='many_to_one');merged['residual']=merged.observed-merged.prediction;merged.to_csv(out/'prediction_scores.csv',index=False)
    ix=np.random.default_rng(202611011702).integers(0,8,(10000,8));pol=p['policies'];seeds=p['physical_seeds'];effects=[];scores=[];seedrows=[]
    for window in p['windows']:
        for model in ['quadratic6','four_parameter']:
            sub=merged[(merged.window==window)&(merged.model==model)]
            o=sub.pivot(index='physical_seed',columns='controller',values='observed').loc[seeds,pol].to_numpy();pr=sub.pivot(index='physical_seed',columns='controller',values='prediction').loc[seeds,pol].to_numpy();err=o-pr;own=[]
            for ai,bi in [(1,0),(2,1),(2,0)]:
                delta=o[:,ai]-o[:,bi];prediction=pr[:,ai]-pr[:,bi];res=delta-prediction;k=int((delta>0).sum())
                e=dict(model=model,window=window,contrast=pol[ai]+' - '+pol[bi],observed=float(delta.mean()),predicted=float(prediction.mean()),residual=float(res.mean()),observed_bootstrap95=np.quantile(delta[ix].mean(1),[.025,.975]),residual_bootstrap95=np.quantile(res[ix].mean(1),[.025,.975]),positive_seeds=k,negative_seeds=int((delta<0).sum()),zero_seeds=int((delta==0).sum()),per_seed=delta,within_005=bool(abs(res.mean())<=.05),sign_agreement=bool(np.sign(delta.mean())==np.sign(prediction.mean())))
                if ai==1 and bi==0 and window=='primary_61_200':e['primary_one_sided_sign_p']=sum(comb(8,j) for j in range(k,9))/256
                effects.append(e);own.append(e)
                seedrows.extend(dict(model=model,window=window,contrast=e['contrast'],physical_seed=seed,observed=float(delta[i]),predicted=float(prediction[i]),residual=float(res[i])) for i,seed in enumerate(seeds))
            rmse=float(np.sqrt((err**2).mean()));scores.append(dict(model=model,window=window,rmse=rmse,rmse_bootstrap95=np.quantile(np.sqrt((err[ix]**2).mean((1,2))),[.025,.975]),mean_residual=float(err.mean()),maximum_absolute_run_error=float(abs(err).max()),engineering_pass=bool(rmse<=.05 and all(e['within_005'] for e in own))))
    changes=[]
    for model in ['quadratic6','four_parameter']:
        for contrast in {e['contrast'] for e in effects}:
            a=next(e for e in effects if e['model']==model and e['contrast']==contrast and e['window']=='early_41_60');b=next(e for e in effects if e['model']==model and e['contrast']==contrast and e['window']=='terminal_161_200');delta=b['per_seed']-a['per_seed']
            changes.append(dict(model=model,contrast=contrast,terminal_minus_early=float(delta.mean()),bootstrap95=np.quantile(delta[ix].mean(1),[.025,.975])))
    result=dict(created_at_utc=now(),mode=mode,synthetic_only=mode=='mock',responses=81600,steps=200,initial_conditions=8,refitted=False,models=scores,effects=effects,changes=changes,primary_effect='independent_mirror - native; steps61–200',)
    save(out/'results.json',result);pd.DataFrame(seedrows).to_csv(out/'contrasts_by_seed.csv',index=False)
    print(f'{mode}: analyzed all3 frozen windows; no fitting.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['mock','real']);analyze(ap.parse_args().mode)
