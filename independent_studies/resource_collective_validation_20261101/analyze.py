"""Replay every raw result, audit frozen forecasts, then score the completed cohort."""
from support import *
import pandas as pd
from math import comb

def main(mode):
    check();pf=prediction_check();design=read(HERE/'design.json');catalog=read(HERE/'prompts.json');folder=HERE/f'runs/{mode}';rows=journal_rows(folder/'attempts.jsonl')
    d={v['dispatch_id']:v for v in rows if v['event']=='dispatch'};r={v['dispatch_id']:v for v in rows if v['event']=='result'};assert set(d)==set(r)
    terminal={v['task_id']:v for v in rows if v['event']=='result' and v['terminal']};assert len(terminal)==16320 and sum(v['terminal'] for v in r.values())==16320
    configs=np.array(design['initials']);states=[configs.copy()];fall=0;seen=[]
    for step in range(60):
        tasks=tasks_for_step(configs,step,design,catalog);new=configs.copy()
        for task in tasks:
            v=terminal[task['task_id']];assert v['mode']==mode and v['compatible_model'] and v['raw_response'] is not None
            for name in task:assert task[name]==v[name]==d[v['dispatch_id']][name],name
            try:parsed=decode(v['raw_response'],'resource')
            except (ValueError,TypeError):parsed=None
            assert parsed==v['parsed'] and v['valid']==(parsed is not None)
            action=v['permutation'][parsed['value']] if parsed is not None else v['own'];new[v['run'],v['agent']]=action;fall+=parsed is None;seen.append(task['task_id'])
        configs=new;states.append(configs.copy())
    assert len(set(seen))==16320;states=np.array(states);assert np.array_equal(states,np.load(folder/'states.npy'))
    if mode=='real':assert datetime.datetime.fromisoformat(pf['created_at_utc'])<min(datetime.datetime.fromisoformat(v['timestamp_utc']) for v in d.values())
    counts=np.stack([(states==a).sum(2) for a in range(3)],axis=2);congestion=(counts**2).sum(2)/97
    means=congestion[1:].mean(0);late=congestion[41:].mean(0);delta=means[1::2]-means[::2]
    point=np.array(read(HERE/'predictions/point_predictions.json')['primary']['mean']);pred_delta=float((point[1::2]-point[::2]).mean());summary=read(HERE/'predictions/summary.json');lo,hi=summary['cohort_contrast_predictive95']
    rng=np.random.default_rng(202611011107);boot=delta[rng.integers(0,8,size=(20000,8))].mean(1);nonzero=delta[abs(delta)>1e-12];positive=int((nonzero>0).sum());pvalue=sum(comb(len(nonzero),i) for i in range(positive,len(nonzero)+1))/2**len(nonzero) if len(nonzero) else 1.
    out=HERE/f'analysis_{mode}';out.mkdir(exist_ok=True);np.save(out/'states.npy',states);np.save(out/'congestion.npy',congestion)
    table=[]
    for run in range(16):table.append(dict(**design['runs'][run],mean_congestion=float(means[run]),late_congestion=float(late[run]),predicted_mean=float(point[run]),residual=float(means[run]-point[run])))
    pd.DataFrame(table).to_csv(out/'run_metrics.csv',index=False)
    result=dict(created_at_utc=now(),mode=mode,synthetic=mode=='mock',responses=16320,format_fallbacks=fall,dispatches=len(d),transport_extra_attempts=len(d)-16320,observed_contrast=float(delta.mean()),paired_seed_bootstrap95=np.quantile(boot,[.025,.975]),positive_pairs=int((delta>1e-12).sum()),one_sided_sign_test=pvalue,primary_predicted_contrast=pred_delta,contrast_residual=float(delta.mean()-pred_delta),run_mean_rmse=float(np.sqrt(np.mean((means-point)**2))),frozen_predictive95=[lo,hi],contrast_within_predictive95=bool(lo<=delta.mean()<=hi),absolute_contrast_error_le_010=bool(abs(delta.mean()-pred_delta)<=.10),run_rmse_le_015=bool(np.sqrt(np.mean((means-point)**2))<=.15),)
    save(out/'results.json',result);save(HERE/f'verification_{mode}.json',dict(passed=True,created_at_utc=now(),responses=16320,all_raw_responses_reparsed=True,all_step_states_replayed=True,same_step_snapshot_verified=True,prediction_hashes_unchanged=True,prior_sources_unchanged=True,format_fallbacks=fall,journal_sha256=sha(folder/'attempts.jsonl')));print(json.dumps(result,indent=2,default=lambda a:a.tolist()))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['mock','real'],required=True);args=parser.parse_args();main(args.mode)
