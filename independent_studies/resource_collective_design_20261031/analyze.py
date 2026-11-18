"""Frozen calibration diagnostics and exact finite-kernel forecasts."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('resource_complete_study',HERE/'study.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
k=s.k;read=s.read;save=s.save
def analyze(mode):
    terminal=s.verify(mode);tasks=read(HERE/'tasks.json');prompts=read(HERE/'prompts.json');answers=np.zeros((32,459),int)
    for t in tasks:
        v=terminal[t['task_id']];answers[t['round'],int(t['prompt_id'][1:])]=v['parsed']['value'] if v['valid'] else 3
    joint=np.array([np.bincount(answers[:,i],minlength=4) for i in range(459)]);held_joint=np.array([np.bincount(answers[16:,i],minlength=4) for i in range(459)])
    counts=joint[:,:3];held=held_joint[:,:3];valid=counts.sum(1,keepdims=True);held_valid=held.sum(1,keepdims=True);assert valid.min()>0 and held_valid.min()>0
    frozen=read(HERE/f'half_predictions_{mode}.json');first=np.array(frozen['probabilities']);first_joint=np.array(frozen['joint_probabilities']);emp=held/held_valid;emp_joint=held_joint/16;cost=np.array([p['counts'] for p in prompts.values()]);excess=cost-cost.min(1,keepdims=True)
    metrics=dict(held_half_nll=float(-np.mean((emp_joint*np.log(first_joint)).sum(1))),held_half_brier=float(np.mean(1-2*(emp_joint*first_joint).sum(1)+(first_joint**2).sum(1))),score_classes='A,B,C,invalid_or_incomplete',held_half_conditional_nll=float(-np.sum(held*np.log(first))/held.sum()),held_half_probability_mse=float(np.mean(((emp-first)**2).sum(1))),held_half_mean_excess_load_rmse=float(np.sqrt(np.mean(((emp*excess).sum(1)-(first*excess).sum(1))**2))),missing_choices=int(joint[:,3].sum()),valid_choices=int(valid.sum()),unobserved_action_cells=int((counts==0).sum()),total_action_cells=1377,inputs_with_all32_same_choice=int((counts.max(1)==32).sum()))
    out=HERE/f'analysis_{mode}';out.mkdir(exist_ok=True);rows=[]
    for i,p in enumerate(prompts.values()):rows.append(dict(prompt_id=f'r{i:03d}',A=p['counts'][0],B=p['counts'][1],C=p['counts'][2],own=p['own'],n=32,valid=int(valid[i,0]),missing=int(joint[i,3]),choose_A=int(counts[i,0]),choose_B=int(counts[i,1]),choose_C=int(counts[i,2]),accuracy=float(counts[i,excess[i]==0].sum()/valid[i,0]),accuracy_all_dispatches=float(counts[i,excess[i]==0].sum()/32),mean_excess_load=float(counts[i]@excess[i]/valid[i,0])))
    pd.DataFrame(rows).to_csv(out/'by_input.csv',index=False);np.save(out/'answers.npy',answers);np.save(out/'counts.npy',counts);np.save(out/'joint_counts.npy',joint)
    initial=k.uniform_stationary();equal=np.full(171,1/171);results=[];state_rows=[]
    for estimator,p in [('MLE',counts/valid),('Dirichlet_half',(counts+.5)/(valid+1.5))]:
        for policy,table in [('native',p),('permutation_average',k.symmetric(p))]:
            T=k.transition(table);curves=k.trajectories(T);early,ev=k.window_moments(T,start=1,end=170);late,lv=k.window_moments(T,start=681,end=1020)
            np.save(out/f'T_{estimator}_{policy}.npy',T);np.save(out/f'curves_{estimator}_{policy}.npy',curves)
            results.append(dict(estimator=estimator,policy=policy,early_mean_iid_initial=float(initial@early),late_mean_iid_initial=float(initial@late),early_mean_equal_counts=float(equal@early),late_mean_equal_counts=float(equal@late)))
            for i,n in enumerate(k.compositions(17)):state_rows.append(dict(estimator=estimator,policy=policy,A=int(n[0]),B=int(n[1]),C=int(n[2]),early_mean=float(early[i]),early_process_variance=float(ev[i]),late_mean=float(late[i]),late_process_variance=float(lv[i])))
    pd.DataFrame(state_rows).to_csv(out/'forecasts_by_initial_count.csv',index=False)
    rng=np.random.default_rng(202610319403);draws=8 if mode=='mock' else 256;unc=[]
    for draw in range(draws):
        p=rng.gamma(counts+.5);p/=p.sum(1,keepdims=True);values=[]
        for table in [p,k.symmetric(p)]:
            curves=k.trajectories(k.transition(table));values.append([float(initial@curves[:,1:171].mean(1)),float(initial@curves[:,681:1021].mean(1))])
        unc.append(dict(draw=draw,native_early=values[0][0],symmetric_early=values[1][0],native_late=values[0][1],symmetric_late=values[1][1],benefit_early=values[0][0]-values[1][0],benefit_late=values[0][1]-values[1][1]))
        if (draw+1)%32==0:print(f'{mode}: kernel uncertainty {draw+1}/{draws}',flush=True)
    u=pd.DataFrame(unc);u.to_csv(out/'posterior_kernel_draws.csv',index=False)
    result=dict(mode=mode,synthetic=mode=='mock',new_collective_api_calls=0,calibration=metrics,forecasts=results,conditional_dirichlet95={col:np.quantile(u[col],[.025,.975]) for col in ['benefit_early','benefit_late']},posterior_table_draws=draws,
       )
    save(out/'results.json',result);print(f'{mode}: calibration analysis complete',flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['mock','real'],required=True);analyze(p.parse_args().mode)
