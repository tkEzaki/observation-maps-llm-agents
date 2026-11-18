"""Frozen trajectory predictions conditional on actual initial states and frames."""
from support import *

def simulate(p,failure,design,reps,rng):
    x=np.broadcast_to(np.array(design['initials']), (reps,16,17)).copy();frames=np.array(design['frames']);costs=[]
    for step in range(60):
        counts=np.stack([(x==a).sum(2) for a in range(3)],axis=-1)
        peers=counts[:,:,None,:]-np.eye(3,dtype=int)[x]
        perm=np.broadcast_to(frames[None,:,step,:,:],peers.shape)
        display=np.take_along_axis(peers,perm,axis=-1);own=(perm==x[:,:,:,None]).argmax(3)
        a=display[:,:,:,0];b=display[:,:,:,1];ix=3*(17*a-a*(a-1)//2+b)+own
        probabilities=p[ix];u=rng.random(x.shape);action=(u>probabilities[:,:,:,0]).astype(int)+(u>probabilities[:,:,:,:2].sum(3)).astype(int)
        y=np.take_along_axis(perm,action[:,:,:,None],axis=3)[:,:,:,0]
        failprob=failure[ix] if np.ndim(failure) else failure
        x=np.where(rng.random(x.shape)<failprob,x,y)
        counts=np.stack([(x==a).sum(2) for a in range(3)],axis=-1);costs.append((counts**2).sum(2)/97)
    costs=np.array(costs)
    return dict(mean=costs.mean(0),late=costs[40:].mean(0),curve=costs.mean(1))

def point(p,f,design,reps,seed):
    rng=np.random.default_rng(seed);out=[];late=[];curves=[]
    for start in range(0,reps,128):
        r=min(128,reps-start);s=simulate(p,f,design,r,rng);out.append(s['mean']);late.append(s['late']);curves.append(s['curve']*r)
    out=np.concatenate(out);late=np.concatenate(late)
    return dict(mean=out.mean(0),late=late.mean(0),mean_mc_se=out.std(0,ddof=1)/np.sqrt(reps),curve=np.sum(curves,axis=0)/reps,process_replicates=reps)

def main():
    check();assert not (HERE/'prediction_freeze.json').exists() and not (HERE/'runs/real').exists()
    out=HERE/'predictions';out.mkdir(exist_ok=True);design=read(HERE/'design.json');counts=np.load(CAL/'analysis_real/counts.npy');valid=counts.sum(1,keepdims=True);meanp=(counts+.5)/(valid+1.5)
    received=np.zeros(459,int);bad=received.copy()
    for v in journal_rows(CAL/'runs/real/attempts.jsonl'):
        if v['event']=='result' and v['terminal'] and v['raw_response'] is not None:
            i=int(v['prompt_id'][1:]);received[i]+=1;bad[i]+=not v['valid']
    assert bad.sum()==2 and received.sum()==14672
    estimates={
      'primary':point(meanp,2.5/14673,design,4096,202611011102),
      'empirical_secondary':point(counts/valid,2/14672,design,4096,202611011103),
      'input_specific_failure_sensitivity':point(meanp,bad/received,design,2048,202611011104),
      'one_percent_failure_stress':point(meanp,.01,design,2048,202611011105)}
    save(out/'point_predictions.json',estimates)
    rng=np.random.default_rng(202611011106);means=[];lates=[];rates=[]
    for draw in range(256):
        p=rng.gamma(counts+.5);p/=p.sum(1,keepdims=True);f=float(rng.beta(2.5,14670.5));s=simulate(p,f,design,32,rng)
        means.append(s['mean']);lates.append(s['late']);rates.append(f)
        if (draw+1)%32==0:print(f'Prediction uncertainty: {draw+1}/256',flush=True)
    means=np.concatenate(means);lates=np.concatenate(lates);contrast=(means[:,1::2]-means[:,::2]).mean(1)
    latecontrast=(lates[:,1::2]-lates[:,::2]).mean(1)
    np.savez_compressed(out/'predictive_samples.npz',mean=means,late=lates,contrast=contrast,late_contrast=latecontrast,format_failure_rates=rates)
    summaries={}
    for name,v in estimates.items():
        y=np.array(v['mean']);summaries[name]=dict(mean_shared=float(y[1::2].mean()),mean_independent=float(y[::2].mean()),contrast=float((y[1::2]-y[::2]).mean()),largest_point_mc_se=float(np.max(v['mean_mc_se'])))
    save(out/'summary.json',dict(created_at_utc=now(),point=summaries,cohort_contrast_predictive95=np.quantile(contrast,[.025,.975]),late_contrast_predictive95=np.quantile(latecontrast,[.025,.975]),per_run_mean_predictive95=np.quantile(means,[.025,.975],axis=0),posterior_parameter_draws=256,process_replicates_per_draw=32,source_format_failures=2,source_received=14672))
    save(HERE/'prediction_freeze.json',dict(created_at_utc=now(),real_dispatches_before_freeze=0,files={p.relative_to(HERE).as_posix():sha(p) for p in sorted(out.iterdir()) if p.is_file()},protocol_sha256=sha(HERE/'protocol.json')))
    print(json.dumps(read(out/'summary.json'),indent=2))

if __name__=='__main__':main()
