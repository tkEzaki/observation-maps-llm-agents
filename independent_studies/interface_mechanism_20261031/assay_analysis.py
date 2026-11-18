"""Prespecified fixed-input analysis; no acquisition or model choice after collective outcomes."""
from core import *
KINDS=['constant','sign','four_parameter','linear6','quadratic6']

def effect(v):
    # rounds, providers, orders, base fields, reflected branches
    a=v[:8].mean(0);b=v[8:].mean(0)
    delta_a=a[:,1]-a[:,0];delta_b=b[:,1]-b[:,0]
    even_a=a.mean(-1);even_b=b.mean(-1)
    return np.mean(delta_a*delta_b,axis=(1,2)),np.mean(even_a[:,1]*even_b[:,1]-even_a[:,0]*even_b[:,0],axis=1)

def analyze(mode):
    from assay import OUT,verify,PROVIDERS,ORDERS
    terminal=verify(mode);tasks=read(OUT/'tasks.json');prompts=read(OUT/'prompts.json');fields=read(OUT/'fields.json');out=OUT/f'analysis_{mode}';out.mkdir(exist_ok=True)
    a=np.zeros((16,2,2,64,2));data=[]
    for t in tasks:
        v=terminal[t['task_id']]['parsed']['value'];a[t['round'],PROVIDERS.index(t['provider']),ORDERS.index(t['order']),t['field'],t['branch']]=v
        data.append(dict(provider=t['provider'],order=t['order'],field=t['field'],branch=t['branch'],source_seed=fields[t['field']]['source_seed'],x=from_text(prompts[t['prompt_id']]['prompt']),y=v+1))
    u,e=effect(a);rng=np.random.default_rng(20261031921);boot=[]
    for _ in range(10000):
        ix=np.r_[rng.integers(0,8,8),rng.integers(8,16,8)];bu,be=effect(a[ix]);boot.append(np.r_[bu,be])
    boot=np.array(boot);effects=[]
    for k,p in enumerate(PROVIDERS):effects.append(dict(provider=p,order_effect_squared=float(u[k]),order_effect_bonferroni_interval=np.quantile(boot[:,k],[.0125,.9875]),even_energy_change_sin_minus_cos=float(e[k]),even_change_bootstrap95=np.quantile(boot[:,k+2],[.025,.975])))
    cv=[];cell_predictions=[];selected=[];full=[]
    for provider in PROVIDERS:
        for order in ORDERS:
            d=[r for r in data if r['provider']==provider and r['order']==order];x=np.array([r['x'] for r in d]);y=np.array([r['y'] for r in d]);seeds=np.array([r['source_seed'] for r in d]);fids=np.array([r['field']*2+r['branch'] for r in d])
            for kind in KINDS:
                pred=np.zeros((len(d),3))
                for seed in sorted(set(seeds)):
                    v=seeds==seed;m=Model(kind,x[~v],y[~v]);pred[v]=m.predict(x[v])
                for fid in sorted(set(fids)):
                    v=fids==fid;obs=float((y[v]-1).mean());guess=float((pred[v,2]-pred[v,0]).mean())
                    cv.append(dict(provider=provider,order=order,model=kind,field_branch=int(fid),source_seed=int(seeds[v][0]),nll=nll(y[v],pred[v]),observed_mean=obs,predicted_mean=guess,error=guess-obs))
                m=Model(kind,x,y);m.export(out/f'{provider}_{order}_{kind}.json')
                if kind=='four_parameter':full.append(dict(provider=provider,order=order,coef=m.coef))
            frame=pd.DataFrame(cv);sub=frame[(frame.provider==provider)&(frame.order==order)];ranking=sub.groupby('model').nll.mean();winner=str(ranking.idxmin());s=sub[sub.model==winner];rmse=float(np.sqrt(np.mean(s.error**2)))
            gate=rmse<=.25 and ranking[winner]<.9*ranking['constant']
            selected.append(dict(provider=provider,order=order,selected=winner,cv_nll=float(ranking[winner]),constant_cv_nll=float(ranking['constant']),cv_mean_action_rmse=rmse,quality_gate=bool(gate)))
            print(f'{mode}: {provider}/{order} selected {winner}, CV mean-action RMSE={rmse:.4f}',flush=True)
    pd.DataFrame(cv).to_csv(out/'cv_by_field.csv',index=False)
    counts=[]
    for pi,provider in enumerate(PROVIDERS):
        for oi,order in enumerate(ORDERS):
            for field in range(64):
                for branch in range(2):
                    v=a[:,pi,oi,field,branch];counts.append(dict(provider=provider,order=order,field=field,branch=branch,n=16,retard=int((v==-1).sum()),stay=int((v==0).sum()),advance=int((v==1).sum()),mean=float(v.mean())))
    pd.DataFrame(counts).to_csv(out/'responses_by_input.csv',index=False)
    gate=bool(np.any(u>.01) and all(s['quality_gate'] for s in selected))
    result=dict(mode=mode,synthetic=mode=='mock',effects=effects,selected=selected,four_parameter=full,collective_gate_passed=gate,
        gate_components=dict(order_effect_gt_001=bool(np.any(u>.01)),all_four_cv_models_pass=all(s['quality_gate'] for s in selected)),
        
        training_only_for_future_collective=True)
    save(out/'results.json',result);save(out/'model_freeze.json',dict(created_at_utc=now(),mode=mode,files={p.name:digest(p) for p in sorted(out.glob('*.json'))},input_journal_sha256=digest(OUT/f'runs/{mode}/attempts.jsonl')))
    print(json.dumps(dict(mode=mode,effects=effects,collective_gate_passed=gate),default=lambda x:x.tolist(),indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['mock','real'],required=True);analyze(p.parse_args().mode)
